import json
import os

def construct_steric_shield(full_sequence, anchor_window, pad_n=5, pad_c=5):
    """
    Constructs the Steric Shield enclosing the Anchor Motif to force 
    surface cleft generation instead of deep tunnels (The Pore Artifact).
    
    Args:
        full_sequence (list or str): The complete IDR sequence array.
        anchor_window (tuple): (start_idx, end_idx) identifying the motif.
        pad_n (int): N-terminal flank length
        pad_c (int): C-terminal flank length
        
    Returns:
        shield_seq (str): The expanded IDR motif with flanks.
        shield_bounds (tuple): Exact index offsets for the new extracted block.
    """
    start_idx, end_idx = anchor_window
    
    min_bound = max(0, start_idx - pad_n)
    max_bound = min(len(full_sequence) - 1, end_idx + pad_c)
    
    shield_seq = ''.join(full_sequence[min_bound:max_bound+1])
    shield_bounds = (min_bound, max_bound)
    
    print(f"Steric Shield Constructed: Bounds {shield_bounds} [{shield_seq}]")
    return shield_seq, shield_bounds


def format_chroma_generator(seed_seq, shield_seq, target_residues=None, output_path="chroma_config.json"):
    """
    Builds the agnostic scaffolding constraints mapping the Chroma configuration API.
    Agnostic JSON outputs ensure safe execution when transferred between the MacBook 
    and the Nvidia Blackwell validation machines.
    """
    config = {
        "generator_type": "chroma",
        "fixed_contexts": [
            {
                "description": "IDR_Steric_Shield",
                "sequence": shield_seq,
                "chain": "B",
                "constraints": "FIXED_ALL"
            },
            {
                "description": "Seed_Anchor",
                "sequence": seed_seq,
                "chain": "C", # Dummy chain to be scaffolded
                "constraints": "FIXED_ALL"
            }
        ],
        "scaffolding_task": {
            "monomeric_span": [60, 100],
            "connect_anchors": ["Seed_Anchor"],
            "steric_penalties": {
                "active": True,
                "volume_exclude": ["IDR_Steric_Shield"]
            }
        }
    }
    
    if target_residues:
        config["scaffolding_task"]["target_override_sites"] = target_residues
        
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=4)
        
    print(f"Chroma constraint mapping successfully written to {output_path}.")
    return config


def format_esm3_multichain(seed_seq, shield_seq, target_residues=None, output_path="esm3_prompt.fasta"):
    """
    Builds the native FASTA formatting structure utilizing ESM3 sequence masking 
    hallucinating the loop connections.
    """
    # ESM3 prompt structure hallucinating spans using '_' masks.
    scaffold_hallucination = "_" * 80 # default approx 80 len scaffold
    
    fasta_content = f">IDR_Steric_Shield\n{shield_seq}\n"
    fasta_content += f">Binder_Scaffold_Prompt\n{seed_seq}{scaffold_hallucination}\n"
    
    if target_residues:
        fasta_content += f"\n# Target Residues Active Map: {target_residues}\n"
        
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(fasta_content)
        
    print(f"ESM3 generative prompt scaffold written to {output_path}.")
    return fasta_content

