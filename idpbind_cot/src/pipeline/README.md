# Neuro-Symbolic CoT Pipeline: GPU Deployment Manual

This library executes a strict 3-tier sequence identifying Intrinsically Disordered Region (IDR) Anchor Motifs, optimizing a sequence-blind $C_\beta$ spatial Seed against the pocket using PyTorch's native `Gumbel-Softmax` differentiability, and seamlessly formatting the spatial geometry blocks into generative REST models preventing tunnel mapping artifacts!

## 1. Local Testing & Verification
You can evaluate your constraints or manually verify target index overrides using the standalone execution block included at the base of `module1_premise.py` directly from your local terminal.

```bash
# Verify the contact map locator and KD hydropathy metric logic local execution:
python src/pipeline/module1_premise.py --pdb_dir data/starling_pdbs/

# Override the sliding window math forcing a manual structural locus:
python src/pipeline/module1_premise.py --pdb_dir data/starling_pdbs/ --target_indices 14,15,16,17,18
```

## 2. Master Implementation Script

Once transferred over SSH to your Nvidia Blackwell architecture, run the following Python script natively within a standard file (e.g. `run_pipeline.py`). Since the pipeline utilizes entirely PyTorch native masking arrays, it completely circumvents the CPU bounds.

```python
import torch

# 1. Import modules bypassing Rosetta/OpenMM entirely
from idpbind_cot.src.pipeline.module1_premise import parse_starling_ensemble, compute_fractional_contact_map, extract_anchor_motif
from idpbind_cot.src.pipeline.module2_reasoning import run_reasoning_loop
from idpbind_cot.src.pipeline.module3_scaffold import construct_steric_shield, format_chroma_generator, format_esm3_multichain

def execute_pipeline():
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Deploying PyTorch Engine across: {DEVICE}")

    # ==========================================
    # MODULE 1: THE PREMISE
    # ==========================================
    ca_coords, cb_coords, seq = parse_starling_ensemble("data/starling_pdbs/", device=DEVICE)
    prob_map = compute_fractional_contact_map(cb_coords, threshold=8.0)
    
    # Extracts optimal Anchor bounding sequence hydropathy bounds 
    best_window, anchor_seq = extract_anchor_motif(prob_map, seq, min_len=6, max_len=10)
    start, end = best_window
    
    # Compress Ensemble down to target spatial anchors
    anchor_ca_mean = torch.mean(ca_coords[:, start:end+1, :], dim=0)
    anchor_cb_mean = torch.mean(cb_coords[:, start:end+1, :], dim=0)

    # ==========================================
    # MODULE 2: PYTORCH COT REASONING
    # ==========================================
    # Backpropagate through spatial geometries and discrete AA logits concurrently
    # Seed length defaults to `3` per Version 1 spec
    reasoner_results = run_reasoning_loop(
        anchor_ca=anchor_ca_mean,
        anchor_cb=anchor_cb_mean,
        seed_len=3, 
        steps=1000, 
        device=DEVICE
    )
    
    seed_sequence = reasoner_results['seed_seq']

    # ==========================================
    # MODULE 3: SCAFFOLDING & EXPORT
    # ==========================================
    # Computes padding bounds preventing internal scaffolding pore collisions 
    shield_seq, shield_bounds = construct_steric_shield(seq, best_window, pad_n=5, pad_c=5)

    # Dumps format configurations directly translating geometry metrics into generator limits
    format_chroma_generator(
        seed_seq=seed_sequence, 
        shield_seq=shield_seq, 
        output_path="outputs/chroma_run_config.json"
    )
    
    format_esm3_multichain(
        seed_seq=seed_sequence,
        shield_seq=shield_seq,
        output_path="outputs/esm3_scaffold_prompt.fasta"
    )

if __name__ == "__main__":
    execute_pipeline()
```

## Hyperparameter Guide
Before executing the massive GPU block, check the following scaling boundaries inside the modules.
- **`tau_init` / `tau_final` (Module 2)**: The Gumbel-Softmax discrete temperature starts natively at `5.0` annealing dynamically to `0.1`. If your seed pocket collides excessively early, raise `tau_init` to force a softer distribution initially!
- **`tether_k` (Module 2)**: Constrains your Anchor CA trace preventing global collapse. Sits at `10.0` but can be loosened if your IDR inherently flexes severely.
- **`seed_len` (Module 2)**: Fixed parameter inside `run_reasoning_loop()` initialized to `3`. Modify this explicit variable to generate heavier bindings interacting deep over extended clefts!
