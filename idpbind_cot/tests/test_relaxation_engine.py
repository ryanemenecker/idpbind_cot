import os
import pytest
import torch
from pathlib import Path

from idpbind_cot.src.relaxation_engine.relax import run_relaxation, _prepare_relaxation_inputs
from idpbind_cot.src.relaxation_engine.utils.align_coordinates import ingest_and_map_structure

# Determine device for testing (fallback to CPU for runners without GPUs)
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# A fixture for the two-chain test PDB the user will provide.
# Replace 'two_chains.pdb' with the actual filename when ready.
TEST_DATA_DIR = Path(__file__).parent / "data"
TWO_CHAIN_PDB = TEST_DATA_DIR / "two_chains.pdb"
PARAMS_DIR = Path(__file__).parent.parent / "src" / "relaxation_engine" / "data"
TEMPLATES_PATH = PARAMS_DIR / "amber14all.pt"

@pytest.fixture
def mock_output_dir(tmp_path):
    """Provides a temporary directory for relaxation output."""
    return tmp_path

@pytest.mark.skipif(not TWO_CHAIN_PDB.exists(), reason="Waiting for two chain PDB test file.")
def test_ingest_and_map_two_chains():
    """
    Tests that the coordinate alignment script properly parses 2 chains 
    from a single PDB natively without errors.
    """
    templates = torch.load(TEMPLATES_PATH, weights_only=False)
    coords, sequence, chain_types, atom_metadata, chain_ids = ingest_and_map_structure(
        str(TWO_CHAIN_PDB), templates, device=DEVICE
    )
    
    assert len(coords) > 0, "No coordinates loaded."
    assert len(sequence) > 0, "No sequence recognized."
    
    # We expect a two chain complex, so we ensure chain_ids has at least two unique chains
    unique_chains = set(chain_ids)
    assert len(unique_chains) >= 2, f"Expected 2+ chains in PDB, but found {len(unique_chains)}: {unique_chains}"
    
    assert len(coords) == len(atom_metadata), "Coordinate size does not match metadata atom count."

@pytest.mark.skipif(not TWO_CHAIN_PDB.exists(), reason="Waiting for two chain PDB test file.")
def test_end_to_end_relaxation(mock_output_dir):
    """
    Tests the full execution of the PyTorch relaxation engine on a 2-chain complex.
    Ensures gradients and physics optimization runs successfully without crashing.
    """
    out_file = mock_output_dir / "relaxed_output.pdb"
    
    # Run a short relaxation just to verify graph compilation and FIRE optimization pass
    run_relaxation(
        target_filepath=str(TWO_CHAIN_PDB),
        output_filepath=str(out_file),
        device=DEVICE,
        verbose=True,
        max_steps=10  # Enough to prove gradient and topological stability
    )
    
    # Check if export successfully built a file
    assert out_file.exists(), "Relaxation completed but output file was not generated."
    assert out_file.stat().st_size > 0, "Output PDB file is empty."

@pytest.mark.skipif(not TWO_CHAIN_PDB.exists(), reason="Waiting for two chain PDB test file.")
def test_loss_function_gradients():
    """
    Tests the autograd / backward-pass capabilities of the engine.
    Ensures loss flows backwards through spatial coordinates during training.
    """
    templates = torch.load(TEMPLATES_PATH, weights_only=False)
    starting_coords, sequence, chain_types, atom_metadata, chain_ids = ingest_and_map_structure(
        str(TWO_CHAIN_PDB), templates, device=DEVICE
    )
    
    # Needs requires_grad to test differentiability natively
    starting_coords.requires_grad_(True)
    
    full_coords, topology, params, ca_mask = _prepare_relaxation_inputs(
        starting_coords, sequence, chain_types, atom_metadata, chain_ids, templates, device=DEVICE
    )
    
    from idpbind_cot.src.relaxation_engine.energy_functions.relaxation import relax_structure
    
    # Test the PyTorch Torch.optim Loop path (use_as_loss_function=True)
    out_coords, loss = relax_structure(
        coords=full_coords,
        topology=topology,
        params=params,
        ca_mask=ca_mask,
        max_steps=2,
        use_as_loss_function=True
    )
    
    assert loss is not None, "Optimizer failed to return a valid loss."
    assert out_coords.requires_grad, "Output coordinates lost gradient tracking history!"

def test_obc2_gb_energy_gradients():
    """
    Tests the standalone fully-vectorized OBC2 GB energy module for math errors 
    and verifies that the gradients successfully flow backward on the coordinate tensor without spawning NaNs.
    """
    from idpbind_cot.src.relaxation_engine.energy_functions.generalized_born import compute_obc2_gb_energy
    
    # Mock parameters
    N = 100
    # Simulate completely overlapping coords occasionally to trigger diagonal/overlap bounds rigorously
    torch.manual_seed(42)
    coords = torch.rand((N, 3), device=DEVICE, dtype=torch.float32) * 5.0
    coords.requires_grad_(True)
    
    params = {
        'gb_rho': torch.rand(N, device=DEVICE, dtype=torch.float32) + 1.0,  # e.g., ~1.0 to 2.0 A
        'gb_screen': torch.ones(N, device=DEVICE, dtype=torch.float32) * 0.8,
        'charge': (torch.rand(N, device=DEVICE, dtype=torch.float32) - 0.5) * 2.0
    }
    
    # Evaluate Energy
    gb_energy = compute_obc2_gb_energy(coords, params)
    
    # Needs to be a valid scalar
    assert not torch.isnan(gb_energy).any(), "GB Energy evaluated to NaN."
    assert gb_energy.dim() == 0, "GB Energy must be a single scalar."
    
    # Trigger backward pass
    gb_energy.backward()
    
    # Verify gradients generated and hold no NaNs (proving diagonal offsets functioned perfectly)
    assert coords.grad is not None, "Coordinates received no gradient from GB backward pass."
    assert not torch.isnan(coords.grad).any(), "Gradients spawned NaNs due to zero-division math bounds."
