#!/usr/bin/env python3
import os, sys, torch, math

# repo-relative imports
from bowerbird2.backend.ca_to_all_atom.kinematics.nerf import extract_dihedral
from bowerbird2.backend.relaxation_engine.utils.align_coordinates import ingest_and_map_structure
from bowerbird2.backend.relaxation_engine.utils.compile_polymer import PolymerCompiler
from bowerbird2.backend.relaxation_engine.hydrogen_bond_mods.place_hydrogens import VectorizedHydrogenBuilder
from bowerbird2.backend.relaxation_engine.hydrogen_bond_mods.hydrogen_constants import H_RULES

def human_atom(meta):
    return f"{meta.get('label_atom_id',meta.get('atom_name'))}:{meta.get('label_comp_id',meta.get('res_name','UNK'))}"

def main(pdb_path):
    # Resolve the relaxation_engine/data directory robustly
    params_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    print("Using params_dir:", params_dir)
    templates = torch.load(os.path.join(params_dir, 'amber14all.pt'), map_location='cpu', weights_only=False)

    coords, sequence, chain_types, atom_metadata, chain_ids = ingest_and_map_structure(pdb_path, templates, device='cpu')
    h_builder = VectorizedHydrogenBuilder(sequence, atom_metadata, H_RULES, device='cpu')
    full_coords = h_builder.build_hydrogens(coords)

    compiler = PolymerCompiler(templates, device='cpu')
    topology, params = compiler.compile_sequence(sequence, chain_types, chain_ids)

    angles = topology['angles'].cpu()            # (3, A)
    k_theta = params['k_theta'].cpu()
    theta_0 = params['theta_0'].cpu()

    posA = full_coords[angles[0]].cpu()
    posB = full_coords[angles[1]].cpu()
    posC = full_coords[angles[2]].cpu()

    v1 = posA - posB
    v2 = posC - posB
    eps = 1e-6
    v1n = torch.sqrt((v1*v1).sum(-1) + eps)
    v2n = torch.sqrt((v2*v2).sum(-1) + eps)
    dot = torch.sum(v1*v2, -1)
    cos_theta = dot / (v1n * v2n)
    cos_theta = torch.clamp(cos_theta, -1.0+eps, 1.0-eps)
    theta = torch.acos(cos_theta)
    delta = theta - theta_0
    per_angle = 0.5 * k_theta * (delta*delta)

    # Stats
    print("k_theta stats: min, median, max =", float(k_theta.min()), float(k_theta.median()), float(k_theta.max()))
    print("Angles total:", per_angle.numel())
    small_v1 = (v1n < 0.3).nonzero(as_tuple=False).squeeze(-1)
    small_v2 = (v2n < 0.3).nonzero(as_tuple=False).squeeze(-1)
    print("Angles with tiny arm lengths: v1 <", 0.3, ":", small_v1.numel(), " v2 <", 0.3, ":", small_v2.numel())

    # Top offenders
    topk = min(50, per_angle.numel())
    vals, idx = torch.topk(per_angle, topk)
    print(f"\nTop {topk} angle energies (energy, index, theta, theta0, k_theta, atoms):")
    for v,i in zip(vals.tolist(), idx.tolist()):
        a,b,c = angles[:,i].tolist()
        metaA = atom_metadata[a]; metaB = atom_metadata[b]; metaC = atom_metadata[c]
        print(f"{v:10.4f} | {i:4d} | theta={float(theta[i]):6.3f} | theta0={float(theta_0[i]):6.3f} | k={float(k_theta[i]):8.3f} | {human_atom(metaA)} - {human_atom(metaB)} - {human_atom(metaC)} | v1n={float(v1n[i]):5.3f} v2n={float(v2n[i]):5.3f}")

    # pick a top offending angle index (e.g. 507)
    angle_idx = 507
    a, b, c = topology['angles'][:, angle_idx].tolist()
    # find the H index for the same residue: search atom_metadata for atom_name 'HA' in same residue as atom b (CA)
    res_id = (atom_metadata[b]['chain_id'], atom_metadata[b]['res_seq'], atom_metadata[b].get('ins_code'))
    h_idx = None
    for i, meta in enumerate(atom_metadata):
        if (meta['chain_id'], meta['res_seq'], meta.get('ins_code')) == res_id and meta.get('atom_name') == 'HA':
            h_idx = i
            break

    print("Indices A,B,C,D:", a, b, c, h_idx)
    if h_idx is None:
        print("HA not found for residue", res_id, "; skipping dihedral check.")
    else:
        # Use the hydrogenized coordinates (full_coords) when measuring dihedral
        di = extract_dihedral(
            full_coords[a].unsqueeze(0),
            full_coords[b].unsqueeze(0),
            full_coords[c].unsqueeze(0),
            full_coords[h_idx].unsqueeze(0),
        ).item()
        print("Measured dihedral (rad,deg):", di, di * 180.0 / math.pi)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/angle_diagnostics.py path/to/structure.pdb")
        sys.exit(1)
    main(sys.argv[1])