import numpy as np
from Bio import PDB
import Bin as Bin
from typing import Tuple
import pandas as pd
import os

def process_fasta(file_path: str) -> Tuple[str, str]:
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    first_seq = ""
    recording = False
    
    for line in lines:
        if line.startswith('>'):
            if recording:
                break
            recording = True
            continue
        if recording:
            first_seq += line.strip()
            
    first_seq_no_gaps = first_seq.replace('-', '')
    
    return first_seq_no_gaps, first_seq

def convert_arrays_to_lists(values_list):
    """Convert a list of numpy arrays to a list of lists with float entries."""
    return [list(map(float, arr)) for arr in values_list]

def create_annotated_lists(csv_file, length):
    """Create three lists of specified length based on CSV data."""
    df = pd.read_csv(csv_file)
    delta_elbo_list = [0] * length
    neg_delta_elbo_list = [0] * length
    abs_delta_elbo_list = [0] * length
    
    for _, row in df.iterrows():
        pos = int(row["position"]) - 1  # Convert 1-based to 0-based index
        if 0 <= pos < length:
            delta_elbo_list[pos] = row["delta_elbo"]
            neg_delta_elbo_list[pos] = -1 * row["delta_elbo"]
            abs_delta_elbo_list[pos] = row["abs_delta_elbo"]
    
    return [delta_elbo_list, neg_delta_elbo_list, abs_delta_elbo_list]

def normalize_values(values):
    """Rank the list from 0 to len(values) - 1 based on the values, handling ties."""
    # Create a sorted list of tuples (original index, value)
    values = list(values)
    sorted_values = sorted(enumerate(values), key=lambda x: x[1])
    
    # Create a ranking list with the same length as values
    ranks = [0] * len(values)
    
    # Initialize the rank counter
    rank = 0
    
    # Iterate through the sorted list and assign ranks, handling ties
    for i in range(len(values)):
        # If it's not the first value or if the current value is different from the previous one
        if i > 0 and sorted_values[i][1] != sorted_values[i-1][1]:
            rank = i  # Set rank to current index
        ranks[sorted_values[i][0]] = rank
    
    return [r/(len(ranks)-1) for r in ranks]

def annotate_pdb(input_pdb, values, output_pdb):
    """Modify the B-factor column of a PDB file to reflect normalized values."""
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("protein", input_pdb)
    
    normalized_values = normalize_values(values)
    
    atom_index = 0
    for model in structure:
        for chain in model:
            for residue in chain:
                if PDB.is_aa(residue):  # Ensure it's an amino acid
                    for atom in residue:
                        atom.set_bfactor(normalized_values[atom_index] * 100)  # Scale to [0,100] for visibility
                    atom_index += 1
    
    # Save the modified PDB
    io = PDB.PDBIO()
    io.set_structure(structure)
    io.save(output_pdb)

def annotate_multiple_pdbs(input_pdb, values_list, names_list,save_path = './TestResults/Antibodies/'):
    """Generate multiple annotated PDB files based on lists of values and names."""
    os.makedirs(save_path, exist_ok=True)
    for values, name in zip(values_list, names_list):
        output_pdb = save_path+f"{name}.pdb"
        annotate_pdb(input_pdb, values, output_pdb)
        print(f"Annotated PDB saved as {output_pdb}")

import matplotlib.pyplot as plt


def get_focus_avg_n_distances(pdb_path, focus_seq, n=1):
    """
    Compute the average distance of the n closest neighboring atoms for each residue
    in the focus segment of chain A using Bio.PDB.

    Parameters:
        pdb_path (str): Path to the PDB file.
        focus_seq (str): The focus sequence in one-letter code.
        n (int, list of int, or [None]): Number of closest points to consider for averaging.
                                         If [None], uses all values from 1 to the max number of neighbors.

    Returns:
        list or list of lists: A list of average distances if n is an int, otherwise a list of lists.
    """
    from Bio import PDB
    from Bio.PDB.Polypeptide import is_aa
    from Bio.PDB.NeighborSearch import NeighborSearch
    import numpy as np

    # Ensure n is a list for consistency
    is_single_n = isinstance(n, int)
    if is_single_n:
        n = [n]

    # Mapping from three-letter to one-letter amino acid codes.
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

    # Locate the focus sequence within chain A.
    start_idx = chainA_sequence.find(focus_seq)
    if start_idx == -1:
        raise ValueError("Focus sequence not found in chain A.")
    end_idx = start_idx + len(focus_seq)
    focus_residues = chainA_residues[start_idx:end_idx]

    # Define neighbor residues: all standard amino acids in the model that are NOT in the focus segment.
    neighbors_residues = [res for chain in model for res in chain if is_aa(res, standard=True) and res not in focus_residues]

    # Build a list of all atoms from the neighbor residues.
    neighbor_atoms = [atom for res in neighbors_residues for atom in res.get_atoms()]
    ns = NeighborSearch(neighbor_atoms)

    # For each residue in the focus segment, compute the sorted distances to all neighbor atoms.
    all_avg_distances = []  # Stores results for different n values

    for res in focus_residues:
        all_distances = []
        for atom in res.get_atoms():
            # Use a large radius to capture all potential neighbor atoms.
            nearby_atoms = ns.search(atom.get_coord(), 1000)
            # Compute distances
            distances = [np.linalg.norm(atom.get_coord() - n_atom.get_coord()) for n_atom in nearby_atoms]
            all_distances.extend(distances)

        # Sort distances once
        all_distances.sort()

        # If n=[None], use n values from 1 to max available distances
        if n == [None]:
            n_values = list(range(1, len(all_distances) + 1))
        else:
            n_values = n

        avg_distances_for_residue = []
        for n_val in n_values:
            if all_distances:
                avg_distance = np.mean(all_distances[:n_val])  # Take the n_val closest distances
            else:
                avg_distance = float('inf')  # If no neighbors found, return infinity

            avg_distances_for_residue.append(avg_distance)

        all_avg_distances.append(avg_distances_for_residue)

    # Transpose the result: Instead of per-residue lists, return per-n lists
    all_avg_distances = list(map(list, zip(*all_avg_distances)))

    # If n was a single integer, return a flat list instead of a list of lists
    return all_avg_distances[0] if is_single_n else all_avg_distances


def plot_scatter_vs_distance_nature(distance_list, list_of_lists, y_names):
    """
    Create scatter plots of each inner list (y-values) versus the distance list (x-values)
    using a Nature-style aesthetic. Each inner list is normalized to range from 0 to 1
    by subtracting its minimum and dividing by the shifted maximum.

    The plots are arranged in a 3x3 grid.

    Parameters:
        distance_list (list of float): List of distance values (x-axis).
        list_of_lists (list of list of float): Each inner list contains y-values corresponding to each focus residue.
        y_names (list of str): Names to label the y-axis of each subplot.
    """
    # Set a clean, Nature-style aesthetic.
    
    n_series = len(list_of_lists)
    n_rows, n_cols = 3, 3  # 3x3 grid
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 10), sharex=True, sharey=True)

    # Flatten the axes array for easier iteration
    axes = axes.flatten()

    for i, (ax, y_data, name) in enumerate(zip(axes, list_of_lists, y_names)):
        # Normalize the y_data to [0, 1].
        y_min = min(y_data)
        y_shifted = [y - y_min for y in y_data]  # Shift to start from 0.
        y_max = max(y_shifted)
        y_norm = [y / y_max if y_max != 0 else 0 for y in y_shifted]  # Avoid division by zero
        
        # Scatter plot: 'x' marker, larger size, alpha 0.8
        ax.scatter(distance_list, y_norm, marker='x', s=100, alpha=0.8, color='black')

        ax.set_ylim(0, 1)
        ax.set_ylabel(name, fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=10)
    
    # Hide unused subplots if fewer than 9 datasets
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.supxlabel("Distance", fontsize=14)
    fig.supylabel("Normalized Values", fontsize=14)
    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt
import matplotlib.animation as animation

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def animate_scatter_vs_distance(distance_lists, list_of_lists, y_names, n_values, interval=1000, file_path=None):
    """
    Create an animated scatter plot where each frame corresponds to a different n-value.
    
    Parameters:
        distance_lists (list of list of float): List of distance lists, one for each n.
        list_of_lists (list of list of float): Each inner list contains y-values corresponding to each focus residue.
        y_names (list of str): Names to label the y-axis of each subplot.
        n_values (list of int): List of n-values for different frames in the animation.
        interval (int): Interval between frames in milliseconds (default: 1000ms = 1s).
        file_path (str): Path to save the animation as a file. If None, the plot will not be saved.
    
    Returns:
        FuncAnimation: The animation object.
        dict: Dictionary of R² values for each dataset across n-values.
    """
    n_series = len(list_of_lists)
    n_rows, n_cols = 3, 3  # 3x3 grid
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 10), sharex=True, sharey=True)

    # Flatten the axes array for easier iteration
    axes = axes.flatten()

    # Add an annotation for the changing "n" value above the plots
    title_text = fig.text(0.5, 1.02, "", ha="center", fontsize=16, fontweight="bold")

    # Dictionary to store R² values for each dataset across different n-values
    r2_scores = {name: [] for name in y_names}

    def update(frame):
        """Update the scatter plots for the current n value."""
        n = n_values[frame]
        distance_list = distance_lists[frame]  # Select the corresponding distance list
        title_text.set_text(f"n = {n}")  # Update the title above the plots

        for i, (ax, y_data, name) in enumerate(zip(axes, list_of_lists, y_names)):
            ax.clear()
            # Normalize the y_data to [0, 1].
            y_min = min(y_data)
            y_shifted = [y - y_min for y in y_data]  # Shift to start from 0.
            y_max = max(y_shifted)
            y_norm = [y / y_max if y_max != 0 else 0 for y in y_shifted]  # Avoid division by zero
            
            # Scatter plot: 'x' marker, larger size, alpha 0.8
            ax.scatter(distance_list, y_norm, marker='x', s=100, alpha=0.8, color='black')
            
            ax.set_ylim(0, 1)
            ax.set_ylabel(name, fontsize=12)
            ax.tick_params(axis='both', which='major', labelsize=10)

            # Compute R² for this dataset at the current n
            model = LinearRegression().fit(np.array(distance_list).reshape(-1, 1), np.array(y_norm))
            r2 = r2_score(y_norm, model.predict(np.array(distance_list).reshape(-1, 1)))
            r2_scores[name].append(r2)

        # Hide unused subplots if fewer than 9 datasets
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=len(n_values), interval=interval, repeat=True)

    fig.supxlabel("Distance", fontsize=14)
    fig.supylabel("Normalized Values", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to prevent overlap

    # Save the animation if file_path is provided
    if file_path:
        ani.save(file_path, writer="imagemagick", fps=10)

    plt.show()
    plt.close()

    return ani, r2_scores


def plot_r2_vs_n(r2_scores, n_values, file_path=None):
    """
    Generate a multi-line plot where the x-axis is n-values and the y-axis is R² values.
    
    Parameters:
        r2_scores (dict): Dictionary where keys are dataset names and values are lists of R² values across n-values.
        n_values (list of int): List of n-values.
        file_path (str): Path to save the plot as a file. If None, the plot will not be saved.
    """
    plt.figure(figsize=(8, 6))

    for name, r2_list in r2_scores.items():
        plt.plot(n_values, r2_list[0:len(n_values)], marker='o', linestyle='-', label=name)

    plt.xlabel("n", fontsize=14)
    plt.ylabel("R²", fontsize=14)
    plt.ylim(0, 1)  # R² is between 0 and 1
    plt.title("R² vs. n", fontsize=16)
    plt.legend(title="Datasets")
    plt.grid(True, linestyle="--", alpha=0.6)

    # Save the plot if file_path is provided
    if file_path:
        plt.savefig(file_path, dpi=300, bbox_inches='tight')

    plt.show()
    plt.close()

# Example usage
input_pdb = "./Data/Antibodies/docked_antibody.pdb"  # Replace with actual PDB file path
sequence_file_path ="./Data/Antibodies/sequences.fasta"
focus, focus_aligned = process_fasta(sequence_file_path)

## naive method
identified_residues_folder = './TestResults/Antibodies'
_,data = Bin.naive_loader(focus_aligned,focus,identified_residues_folder)
legends = ['Ent','Theil Ent','Cramer Ent','MI','Theil MI','Cramer MI']
data = convert_arrays_to_lists(data)

## existing method 
identified_residues_folder = './Data/Antibodies/key_tuning_residues_no_rep.csv'
exist_data = create_annotated_lists(identified_residues_folder, len(focus))
exist_legends = ['delta_elbo', 'neg_delta_elbo', 'abs_delta_elbo']


values_list = data+exist_data # Replace with actual list of lists of values
names_list = legends+exist_legends  # Replace with actual list of names

annotate_multiple_pdbs(input_pdb, values_list, names_list,save_path = './TestResults/Antibodies/pdbs/')

pdb_file = "Data/Antibodies/docked.pdb"
focus_seq = "EVQLVETGGGLVQPGGSLRLSCAASGFDLNSYGISWVRQAPGKGPEWVSVIYSDGRRTFYGDSFKGRFTISRDTSTNTVYLQMNSLRVEDTAVYYCAKGRAAGRFDSWGQGTLVTVSSGGGGSGGGGSGGGGSDVVMTQSPESLAVSLGERATISCKSSQSVLYESRNKNSVAWYQQKAGQPPKLLIYWASTRESGVPDRFSGSGSGTDFTLTISSLQAEDAAVYYCQQYHRLPLSFGGGTKVEIK"  # e.g., "ACDEFGH"
distances = get_focus_avg_n_distances(pdb_file, focus_seq,n=[1,2,4,8,16,32,64,128,256])

plot_scatter_vs_distance_nature(distances[0], values_list, names_list)

ani, r2_scores = animate_scatter_vs_distance(distances, values_list, names_list, n_values=[1,2,4,8,16,32,64,128,256], interval=1000,file_path='./TestResults/Antibodies/pdbs/distance_correl_anim.mp4')

plot_r2_vs_n(r2_scores, n_values=[1,2,4,8,16,32,64,128,256],file_path='./TestResults/Antibodies/pdbs/distance_correl.png')