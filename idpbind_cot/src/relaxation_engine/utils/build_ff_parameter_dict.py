import torch
import openmm as mm
import openmm.app as app
import openmm.unit as unit

from idpbind_cot.src.relaxation_engine.utils.constants import KJ_TO_KCAL, NM_TO_ANGSTROM

def build_template_dict_from_system(structure_file, forcefields=['amber14-all.xml']):
    """
    Extracts isolated residue templates from a fully parameterized polymeric system,
    handling CMAP grids and decoupling physical covalent bonds from harmonic springs.
    """
    print("Loading Rosetta Stone PDB and parameterizing system...")
    if structure_file.endswith('.pdb'):
        structure = app.PDBFile(structure_file)
    elif structure_file.endswith('.cif') or structure_file.endswith('.mmcif'):
        structure = app.PDBxFile(structure_file)
    else:
        raise ValueError("Unsupported structure file format. Please provide a PDB or CIF file.")

    ff = app.ForceField(*forcefields)
    
    print("Checking for missing hydrogens and standardizing topology...")
    modeller = app.Modeller(structure.topology, structure.positions)

    # --- SANITIZER BLOCK ---
    atoms_to_delete = []
    for chain in modeller.topology.chains():
        residues = list(chain.residues())
        if len(residues) > 0:
            first_res = residues[0]
            if first_res.name in ['DA', 'DC', 'DG', 'DT', 'A', 'C', 'G', 'U']:
                for atom in first_res.atoms():
                    if atom.name in ['P', 'OP1', 'OP2', 'OP3', 'O1P', 'O2P']:
                        atoms_to_delete.append(atom)
    
    if atoms_to_delete:
        modeller.delete(atoms_to_delete)
        print(f" -> Sanitizer: Deleted {len(atoms_to_delete)} incompatible terminal phosphate atoms.")
    # --------------------------
    
    modeller.addHydrogens(ff)
    
    print("Parameterizing system...")
    system = ff.createSystem(modeller.topology, nonbondedMethod=app.NoCutoff, constraints=None)

    forces = {type(f).__name__: f for f in system.getForces()}
    bond_force = forces.get('HarmonicBondForce')
    angle_force = forces.get('HarmonicAngleForce')
    torsion_force = forces.get('PeriodicTorsionForce')
    nb_force = forces.get('NonbondedForce')
    cmap_force = forces.get('CMAPTorsionForce')
    gbsa_force = forces.get('GBSAOBCForce')

    if nb_force is None:
        raise ValueError("No NonbondedForce found in the parameterized system. "
                         "Check that your force field file includes non-bonded parameters.")
    
    templates_dict = {}
    
    # --- Task 1: Dynamic 1-4 Scaling Factor Extraction ---
    # Detect force field type and set global 1-4 scaling factors.
    # NOTE: This is a simplification for CHARMM. Proper handling of CHARMM 1-4
    # interactions requires specific "exception" parameters for sigma and epsilon,
    # not just a global scaling factor. This implementation approximates by scaling VDW by 1.0.
    global_metadata = {}
    is_charmm = any('charmm' in f.lower() for f in forcefields)
    
    if is_charmm:
        print(" -> Detected CHARMM force field. Using CHARMM 1-4 scaling.")
        global_metadata['elec_scale_14'] = 1.0
        global_metadata['vdw_scale_14'] = 1.0
    else:
        print(" -> Detected AMBER force field (or defaulting). Using AMBER 1-4 scaling.")
        global_metadata['elec_scale_14'] = 1.0 / 1.2  # Amber default for electrostatics
        global_metadata['vdw_scale_14'] = 1.0 / 2.0  # Amber default for Van der Waals
        
    templates_dict['global_metadata'] = global_metadata
    # ---------------------------------------------------------
    
    # Identify terminal residues
    terminal_residues = set()
    for chain in modeller.topology.chains():
        res_list = list(chain.residues())
        if len(res_list) > 0:
            terminal_residues.add(res_list[0])
            terminal_residues.add(res_list[-1])

    # Build a fast lookup set for TRUE covalent bonds (bypassing force field lists)
    true_covalent_edges = set()
    for bond in modeller.topology.bonds():
        true_covalent_edges.add((bond[0].index, bond[1].index))
        true_covalent_edges.add((bond[1].index, bond[0].index))

    # Process global CMAP grids if present
    if cmap_force:
        print(" -> Extracting global CMAP grids...")
        global_cmap_grids = []
        for i in range(cmap_force.getNumMaps()):
            size, energy_list = cmap_force.getMapParameters(i)
            grid = torch.tensor(energy_list, dtype=torch.float32).view(size, size) * KJ_TO_KCAL
            global_cmap_grids.append(grid)
        templates_dict['global_cmap_grids'] = torch.stack(global_cmap_grids).unsqueeze(0)

    print("Extracting polymeric residue parameters...")
    for residue in modeller.topology.residues():
        res_name = residue.name
        
        if res_name in templates_dict or residue in terminal_residues:
            continue

        print(f" -> Compiling template for: {res_name}")
        
        # 1. Map indices FIRST
        atoms = list(residue.atoms())
        num_atoms = len(atoms)
        abs_to_local = {atom.index: local_idx for local_idx, atom in enumerate(atoms)}
        atom_names = [atom.name for atom in atoms]
        atom_name_to_idx = {atom.name: local_idx for local_idx, atom in enumerate(atoms)}
        
        abs_indices_set = set(abs_to_local.keys())
        
        # 2. Extract CMAP map assignments safely now that abs_indices_set exists
        cmap_map_id = -1
        if cmap_force:
            for i in range(cmap_force.getNumTorsions()):
                map_id, a1, a2, a3, a4, a5, a6, a7, a8 = cmap_force.getTorsionParameters(i)
                if {a2, a3, a4}.issubset(abs_indices_set):
                    cmap_map_id = map_id
                    break
        
        # 3. Extract Non-bonded & GBSA OBC
        charges, sigmas, epsilons, gb_rhos, gb_screens = [], [], [], [], []
        for atom in atoms:
            q, sig, eps = nb_force.getParticleParameters(atom.index)
            charges.append(q.value_in_unit(unit.elementary_charge))
            sigmas.append(sig.value_in_unit(unit.nanometer) * NM_TO_ANGSTROM)
            epsilons.append(eps.value_in_unit(unit.kilojoule_per_mole) * KJ_TO_KCAL)
            
            if gbsa_force:
                _, radius, scalingFactor = gbsa_force.getParticleParameters(atom.index)
                gb_rhos.append(radius.value_in_unit(unit.nanometer) * NM_TO_ANGSTROM)
                gb_screens.append(scalingFactor)
            else:
                gb_rhos.append(0.0)
                gb_screens.append(0.0)
            
        # 4. Extract Bonds (Decoupling Springs from Covalent Graph)
        springs_idx, covalent_bonds_idx, k_b, b_0 = [], [], [], []
        for i in range(bond_force.getNumBonds()):
            p1, p2, length, k = bond_force.getBondParameters(i)
            if p1 in abs_indices_set and p2 in abs_indices_set:
                
                # All definitions go to the harmonic spring list (including Urey-Bradley)
                springs_idx.append([abs_to_local[p1], abs_to_local[p2]])
                b_0.append(length.value_in_unit(unit.nanometer) * NM_TO_ANGSTROM)
                k_val = k.value_in_unit(unit.kilojoule_per_mole / unit.nanometer**2)
                k_b.append(k_val * KJ_TO_KCAL / (NM_TO_ANGSTROM**2))
                
                # Only topologically verified bonds go to the chemical graph list
                if (p1, p2) in true_covalent_edges:
                    covalent_bonds_idx.append([abs_to_local[p1], abs_to_local[p2]])
                
        # 5. Extract Angles
        angles_idx, k_theta, theta_0 = [], [], []
        if angle_force:
            for i in range(angle_force.getNumAngles()):
                p1, p2, p3, angle, k = angle_force.getAngleParameters(i)
                if {p1, p2, p3}.issubset(abs_indices_set):
                    angles_idx.append([abs_to_local[p1], abs_to_local[p2], abs_to_local[p3]])
                    theta_0.append(angle.value_in_unit(unit.radian))
                    k_val = k.value_in_unit(unit.kilojoule_per_mole / unit.radian**2)
                    k_theta.append(k_val * KJ_TO_KCAL)
                    
        # 6. Extract Dihedrals
        dihedrals_idx, k_phi, n_list, gamma_list = [], [], [], []
        if torsion_force:
            for i in range(torsion_force.getNumTorsions()):
                p1, p2, p3, p4, periodicity, phase, k = torsion_force.getTorsionParameters(i)
                if {p1, p2, p3, p4}.issubset(abs_indices_set):
                    dihedrals_idx.append([abs_to_local[p1], abs_to_local[p2], abs_to_local[p3], abs_to_local[p4]])
                    n_list.append(periodicity)
                    gamma_list.append(phase.value_in_unit(unit.radian))
                    k_val = k.value_in_unit(unit.kilojoule_per_mole)
                    k_phi.append(k_val * KJ_TO_KCAL)

        # 7. Package
        templates_dict[res_name] = {
            'num_atoms': num_atoms,
            'atom_names': atom_names,
            'atom_name_to_idx': atom_name_to_idx,
            
            'charges': torch.tensor(charges, dtype=torch.float32),
            'sigma': torch.tensor(sigmas, dtype=torch.float32),
            'epsilon': torch.tensor(epsilons, dtype=torch.float32),
            'gb_rho': torch.tensor(gb_rhos, dtype=torch.float32),
            'gb_screen': torch.tensor(gb_screens, dtype=torch.float32),
            
            # The separated bond arrays
            'intra_springs': torch.tensor(springs_idx, dtype=torch.long).t() if springs_idx else torch.empty((2, 0), dtype=torch.long),
            'intra_covalent_bonds': torch.tensor(covalent_bonds_idx, dtype=torch.long).t() if covalent_bonds_idx else torch.empty((2, 0), dtype=torch.long),
            'k_b': torch.tensor(k_b, dtype=torch.float32),
            'b_0': torch.tensor(b_0, dtype=torch.float32),
            
            'intra_angles': torch.tensor(angles_idx, dtype=torch.long).t() if angles_idx else torch.empty((3, 0), dtype=torch.long),
            'k_theta': torch.tensor(k_theta, dtype=torch.float32),
            'theta_0': torch.tensor(theta_0, dtype=torch.float32),
            
            'intra_dihedrals': torch.tensor(dihedrals_idx, dtype=torch.long).t() if dihedrals_idx else torch.empty((4, 0), dtype=torch.long),
            'k_phi': torch.tensor(k_phi, dtype=torch.float32),
            'n': torch.tensor(n_list, dtype=torch.float32),
            'gamma': torch.tensor(gamma_list, dtype=torch.float32),
            
            'cmap_map_id': cmap_map_id
        }

    return templates_dict