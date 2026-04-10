import torch
from bowerbird2.backend.relaxation_engine.utils.build_ff_parameter_dict import build_template_dict_from_system
import os

# get path to module
module_dir = os.path.dirname(os.path.abspath(__file__))
datapath = module_dir.replace('scripts', 'data')

def compile_master_tensor_database(forcefields=['amber14-all.xml']):
    """
    Compiles protein, DNA, and RNA rosetta stone PDBs into a single 
    unified PyTorch tensor dictionary, safely isolating global parameters.
    """
    master_templates = {}
    master_cmap_grids = None

    def safe_merge(new_templates):
        """Merges residue templates while protecting global keys from being overwritten."""
        nonlocal master_cmap_grids
        
        # Intercept and protect the global CMAP grids
        if 'global_cmap_grids' in new_templates:
            extracted_cmap = new_templates.pop('global_cmap_grids')
            # If we haven't stored a global CMAP yet, keep the first one we find (usually from proteins)
            if master_cmap_grids is None:
                master_cmap_grids = extracted_cmap

        # Now it is safe to bulk-update the standard residue templates (ALA, DA, G, etc.)
        master_templates.update(new_templates)

    # 1. Extract Protein Templates
    print("--- Processing Protein ---")
    protein_templates = build_template_dict_from_system(
        f'{datapath}/fold_rosetta_aa_model_0_added_cyx_hid.cif', 
        forcefields=forcefields
    )
    safe_merge(protein_templates)

    # 2. Extract DNA Templates
    print("--- Processing DNA ---")
    dna_templates = build_template_dict_from_system(
        f'{datapath}/fold_rosetta_dna_model_0.cif', 
        forcefields=forcefields
    )
    safe_merge(dna_templates)

    # 3. Extract RNA Templates
    print("--- Processing RNA ---")
    rna_templates = build_template_dict_from_system(
        f'{datapath}/fold_rosetta_rna_model_0.cif', 
        forcefields=forcefields
    )
    safe_merge(rna_templates)

    # 4. Reattach any global parameters safely
    if master_cmap_grids is not None:
        master_templates['global_cmap_grids'] = master_cmap_grids

    # 5. Save the unified tensor library to disk
    output_filename = f'{datapath}/master_residue_templates.pt'
    torch.save(master_templates, output_filename)
    
    print(f"\nSuccess! Compiled {len(master_templates) - (1 if master_cmap_grids is not None else 0)} total residue templates.")
    print(f"Saved to: {output_filename}")
    print("You can now discard OpenMM from your main engine environment.")

if __name__ == "__main__":
    compile_master_tensor_database()