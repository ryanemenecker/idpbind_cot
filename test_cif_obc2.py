import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import sys
sys.path.append('.')

from idpbind_cot.src.relaxation_engine.utils.align_coordinates import ingest_and_map_structure
from idpbind_cot.src.relaxation_engine.relax import _prepare_relaxation_inputs

file_path = "idpbind_cot/src/relaxation_engine/data/fold_q01831_847_to_863_p41208_94_to_172_model_0.cif"
templates = torch.load("idpbind_cot/src/relaxation_engine/data/amber14all.pt", weights_only=False)
print("Loaded templates.")

try:
    coords, sequence, chain_types, atom_metadata, chain_ids = ingest_and_map_structure(
        file_path, templates, device='cpu'
    )
    print("Ingested structure successfully.")
    
    full_coords, topology, params, ca_mask = _prepare_relaxation_inputs(
        coords, sequence, chain_types, atom_metadata, chain_ids, templates, device='cpu'
    )
    print("Parameter keys:", list(params.keys()))
except Exception as e:
    import traceback
    traceback.print_exc()

