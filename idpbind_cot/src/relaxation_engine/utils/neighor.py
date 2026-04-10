# code for getting neighbor lists.
import torch
from torch_cluster import radius_graph
from typing import FrozenSet, Optional, Tuple



class NeighborListManager:
    """
    Manages dynamic spatial partitioning using a Verlet buffer strategy.
    """
    def __init__(self, cutoff: float = 10.0, skin: float = 3.0, max_neighbors: int = 512) -> None:
        self.cutoff = cutoff
        self.skin = skin
        self.r_list = cutoff + skin
        self.max_neighbors = max_neighbors
        
        # State variables to track when a rebuild is necessary
        self.cached_indices = None
        self.reference_coords = None
        self._excl_codes = None  # Cached encoded exclusion pairs for fast vectorized filtering

    def build_list(self, coords: torch.Tensor, exclusion_set: Optional[FrozenSet[Tuple[int, int]]] = None) -> torch.Tensor:
        """
        Constructs the neighbor list if atoms have moved beyond the skin tolerance.
        
        Args:
            coords: Tensor of shape (N, 3).
            exclusion_set: A frozenset of (i, j) tuples for excluded pairs.
        Returns:
            Tensor of shape (2, P) containing valid interacting pairs.
        """
        # 1. Check if a rebuild is necessary (Verlet skin criterion)
        if self.cached_indices is not None and self.reference_coords is not None:
            # Calculate maximum displacement of any single atom since last build
            displacement = coords.detach() - self.reference_coords
            max_dist_sq = (displacement * displacement).sum(dim=-1).max()
            
            # If no atom moved more than skin/2, the cached list is strictly valid
            if max_dist_sq < (self.skin / 2.0) ** 2:
                return self.cached_indices

        # 2. Rebuild the graph using torch_cluster on the detached coordinates
        # We use .detach() because building the topological graph is not a differentiable 
        # operation; the gradients flow through the distances calculated *using* these indices.
        clean_coords = coords.detach()
        
        # radius_graph returns shape (2, num_edges)
        edge_indices = radius_graph(
            clean_coords, 
            r=self.r_list, 
            loop=False, 
            max_num_neighbors=self.max_neighbors
        )
        
        # 3. Apply topological exclusions (1-2, 1-3, and 1-4 pairs)
        # Vectorized: encode each (i, j) pair as i*N + j for O(1) set membership via torch.isin.
        # The encoded exclusion tensor is cached so the Python-side conversion only happens once.
        if exclusion_set is not None:
            N = clean_coords.shape[0]
            if self._excl_codes is None:
                excl_list = list(exclusion_set)
                if excl_list:
                    excl_tensor = torch.tensor(excl_list, dtype=torch.long, device=edge_indices.device)
                    self._excl_codes = excl_tensor[:, 0] * N + excl_tensor[:, 1]
                else:
                    self._excl_codes = torch.empty(0, dtype=torch.long, device=edge_indices.device)
            edge_codes = edge_indices[0] * N + edge_indices[1]
            keep = ~torch.isin(edge_codes, self._excl_codes)
            edge_indices = edge_indices[:, keep]

        # 4. Update the state
        self.cached_indices = edge_indices
        self.reference_coords = clean_coords.clone()

        return self.cached_indices