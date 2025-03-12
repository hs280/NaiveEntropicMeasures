from Bio import PDB

def renumber_pdb_dynamic(input_pdb, output_pdb):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("complex", input_pdb)

    chain_ids = sorted([chain.id for model in structure for chain in model])  # Get unique chain IDs
    chain_residue_counts = {chain_id: 0 for chain_id in chain_ids}

    # Count residues for each chain
    for model in structure:
        for chain in model:
            chain_residue_counts[chain.id] = len([res for res in chain if res.id[0] == " "])  # Count only standard residues

    # Track the cumulative offset for residue numbering
    offset = 0
    for model in structure:
        for chain in model:
            for res in chain:
                if res.id[0] == " ":  # Only consider standard residues, not HETATM
                    # Update residue number with cumulative offset
                    res.id = (" ", offset + res.id[1], " ")
                    # Iterate over all atoms and update their residue number as well
                    for atom in res:
                        atom.set_id(atom.get_id()[0], atom.get_id()[1], atom.get_id()[2])

            # Update the offset for the next chain
            offset += chain_residue_counts[chain.id]  # Add the number of residues in this chain

    # Now write the modified structure back to a PDB file
    io = PDB.PDBIO()
    io.set_structure(structure)
    io.save(output_pdb)
    print(f"Renumbering complete! Saved as {output_pdb}")

# Usage Example
renumber_pdb_dynamic("Data/Antibodies/docked_antigen.pdb", "Data/Antibodies/docked_antigen_unique.pdb")
