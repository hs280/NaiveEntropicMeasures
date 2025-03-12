from Bio import PDB
from Bio.PDB.Polypeptide import is_aa
import numpy as np
import copy


def get_focus_min_distances(pdb_path, focus_seq):
    """
    Compute the minimum distance from each residue in the focus segment of chain A
    to any neighboring residue in the structure using Bio.PDB.

    Parameters:
        pdb_path (str): Path to the PDB file.
        focus_seq (str): The focus sequence in one-letter code.

    Returns:
        list: A list of minimum distances for each residue in the focus segment.
    """
    from Bio import PDB
    from Bio.PDB.Polypeptide import is_aa
    from Bio.PDB.NeighborSearch import NeighborSearch
    import numpy as np

    # Mapping from three-letter codes to one-letter codes.
    three_to_one = {
        "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
        "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
        "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
        "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y"
    }

    # Parse the structure.
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("struct", pdb_path)
    model = structure[0]  # Using the first model.

    # Extract chain A and its standard amino acid residues.
    chainA = model['A']
    chainA_residues = [res for res in chainA if is_aa(res, standard=True)]

    # Construct the one-letter sequence for chain A.
    chainA_sequence = "".join([three_to_one.get(res.get_resname(), "X") for res in chainA_residues])
    print("Chain A sequence:", chainA_sequence)

    # Locate the focus sequence within chain A.
    start_idx = chainA_sequence.find(focus_seq)
    if start_idx == -1:
        raise ValueError("Focus sequence not found in chain A.")
    end_idx = start_idx + len(focus_seq)
    focus_residues = chainA_residues[start_idx:end_idx]
    print(f"Focus segment from residue {focus_residues[0].get_id()[1]} to {focus_residues[-1].get_id()[1]}")

    # Define neighbor residues: all standard amino acids in the model that are NOT in the focus segment.
    neighbors_residues = []
    for chain in model:
        for res in chain:
            if is_aa(res, standard=True) and res not in focus_residues:
                neighbors_residues.append(res)

    # Build a list of all atoms from the neighbor residues.
    neighbor_atoms = [atom for res in neighbors_residues for atom in res.get_atoms()]
    ns = NeighborSearch(neighbor_atoms)

    # For each residue in the focus segment, compute the minimum distance to any neighbor atom.
    distances_list = []
    for res in focus_residues:
        min_distance = float('inf')
        for atom in res.get_atoms():
            # Use a large radius to capture all potential neighbor atoms.
            nearby_atoms = ns.search(atom.get_coord(), 1000)
            for n_atom in nearby_atoms:
                d = np.linalg.norm(atom.get_coord() - n_atom.get_coord())
                if d < min_distance:
                    min_distance = d
        distances_list.append(min_distance)

    return distances_list

def split_pdb_by_focus(pdb_path, focus_seq, focus_output, non_focus_output):
    """
    Split the input PDB into two files:
      - One containing only the focus sequence (from chain A).
      - One containing everything else (the entire structure with the focus residues removed from chain A).
    
    Parameters:
        pdb_path (str): Path to the input PDB file.
        focus_seq (str): Focus sequence in one-letter code.
        focus_output (str): Output filename for the focus-only PDB.
        non_focus_output (str): Output filename for the non-focus PDB.
    """
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("struct", pdb_path)
    model = structure[0]  # working with the first model

    # Get chain A and its residues.
    chainA = model['A']
    chainA_residues = [res for res in chainA if is_aa(res, standard=True)]
    
    # Mapping from three-letter codes to one-letter codes.
    three_to_one = {
        "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
        "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
        "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
        "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y"
    }
    
    # Create the one-letter sequence for chain A.
    chainA_sequence = "".join([three_to_one.get(res.get_resname(), "X") for res in chainA_residues])
    start_idx = chainA_sequence.find(focus_seq)
    if start_idx == -1:
        raise ValueError("Focus sequence not found in chain A.")
    end_idx = start_idx + len(focus_seq)

    # --- Create the focus-only structure ---
    # Deepcopy to avoid altering the original structure.
    focus_structure = copy.deepcopy(structure)
    focus_model = focus_structure[0]
    
    # Remove all chains except chain A.
    chains_to_remove = [chain.id for chain in focus_model if chain.id != 'A']
    for cid in chains_to_remove:
        focus_model.detach_child(cid)
    
    # In chain A, keep only the focus residues.
    focus_chain = focus_model['A']
    # List out residues with index info (using list index based on order in chain).
    residues = list(focus_chain)
    for i, res in enumerate(residues):
        if not (start_idx <= i < end_idx):
            focus_chain.detach_child(res.get_id())
    
    # Save the focus-only PDB.
    io = PDB.PDBIO()
    io.set_structure(focus_structure)
    io.save(focus_output)
    print(f"Focus PDB saved to {focus_output}")
    
    # --- Create the non-focus structure ---
    # Deepcopy the original structure.
    non_focus_structure = copy.deepcopy(structure)
    non_focus_model = non_focus_structure[0]
    chainA_non_focus = non_focus_model['A']
    residues = list(chainA_non_focus)
    # Remove the focus residues from chain A.
    for i, res in enumerate(residues):
        if start_idx <= i < end_idx:
            chainA_non_focus.detach_child(res.get_id())
    
    # Save the non-focus PDB.
    io.set_structure(non_focus_structure)
    io.save(non_focus_output)
    print(f"Non-focus PDB saved to {non_focus_output}")

# Example usage:
pdb_file = "Data/Antibodies/docked.pdb"
focus_seq = "EVQLVETGGGLVQPGGSLRLSCAASGFDLNSYGISWVRQAPGKGPEWVSVIYSDGRRTFYGDSFKGRFTISRDTSTNTVYLQMNSLRVEDTAVYYCAKGRAAGRFDSWGQGTLVTVSSGGGGSGGGGSGGGGSDVVMTQSPESLAVSLGERATISCKSSQSVLYESRNKNSVAWYQQKAGQPPKLLIYWASTRESGVPDRFSGSGSGTDFTLTISSLQAEDAAVYYCQQYHRLPLSFGGGTKVEIK"

# Calculate distances for the focus segment (as in your original function).
distances = get_focus_min_distances(pdb_file, focus_seq)
print("Focus residue minimum distances:", distances)

# Split the PDB into focus and non-focus files.
split_pdb_by_focus(pdb_file, focus_seq, "Data/Antibodies/docked_antibody.pdb", "Data/Antibodies/docked_antigen.pdb")
print("PDB splitting complete.")