import numpy as np
import matplotlib.pyplot as plt
from Bio.PDB import PDBParser
from scipy.spatial import ConvexHull
import Bin as Bin
from typing import Tuple
import pandas as pd
import seaborn as sns

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

import numpy as np
import matplotlib.pyplot as plt
from Bio.PDB import PDBParser, ShrakeRupley

def process_pdb_for_projections(pdb_path, importance_dict, surface_threshold=30):
    """
    Process a PDB file to extract surface residue projections and importance values.
    
    Parameters:
        pdb_path (str): Path to the PDB file.
        importance_dict (dict): Dictionary with algorithm names as keys and lists of importance values.
        surface_threshold (float): ASA threshold (in Å²) for defining surface residues.
    
    Returns:
        dict: A dictionary containing processed data for plotting:
            - 'phi_deg': Azimuthal angles in degrees
            - 'theta_deg': Polar angles in degrees
            - 'surface_indices': Indices of surface residues
            - 'importance_values': Dictionary of importance values for surface residues by algorithm
            - 'vmin', 'vmax': Common color scale bounds
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)
    
    # Extract CA coordinates and residues
    ca_coords = []
    residues = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if "CA" in residue:
                    ca_coords.append(residue["CA"].get_coord())
                    residues.append(residue)
    ca_coords = np.array(ca_coords)
    
    # Compute the protein centroid (using all CA atoms)
    centroid = np.mean(ca_coords, axis=0)
    
    # Compute SASA for each residue using the Shrake–Rupley algorithm
    sr = ShrakeRupley()
    sr.compute(structure, level="R")
    
    # Identify surface residues (those with ASA >= surface_threshold)
    surface_indices = []
    for i, residue in enumerate(residues):
        asa = residue.sasa
        if asa >= surface_threshold:
            surface_indices.append(i)
    surface_indices = np.array(surface_indices, dtype=int)
    
    # Filter CA coordinates for surface residues
    surface_coords = ca_coords[surface_indices]
    
    # Convert surface residue coordinates to spherical coordinates relative to the centroid
    vectors = surface_coords - centroid
    r = np.linalg.norm(vectors, axis=1)
    theta = np.arccos(vectors[:, 2] / r)  # polar angle from z-axis
    phi = np.arctan2(vectors[:, 1], vectors[:, 0])  # azimuthal angle in x-y plane
    
    # Convert radians to degrees
    theta_deg = np.degrees(theta)
    phi_deg = np.degrees(phi)
    
    # Extract importance values for surface residues for each algorithm
    importance_values = {}
    for alg_name, imp_list in importance_dict.items():
        importance_values[alg_name] = np.array(imp_list)[surface_indices]
    
    # Determine common color scaling from all importance lists
    all_imp = np.concatenate(list(importance_values.values()))
    vmin, vmax = all_imp.min(), all_imp.max()
    
    # Return the processed data
    return {
        'phi_deg': phi_deg,
        'theta_deg': theta_deg,
        'surface_indices': surface_indices,
        'importance_values': importance_values,
        'vmin': vmin,
        'vmax': vmax
    }


def plot_residue_projections(projection_data, layout='grid', figsize=None, cmap='coolwarm', 
                            point_size=40, dpi=1200, out_file=None):
    """
    Plot residue importance projections based on processed data.
    
    Parameters:
        projection_data (dict): Processed data from process_pdb_for_projections function
        layout (str): 'grid' for grid layout or 'horizontal' for horizontal layout
        figsize (tuple): Figure size (width, height) in inches, or None for auto
        cmap (str or matplotlib.colors.Colormap): Colormap for the plots
        point_size (int): Size of the scatter points
        dpi (int): DPI for saving the figure
        out_file (str): Optional path to save the resulting plot
        
    Returns:
        tuple: (fig, axes) matplotlib figure and axes objects
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import math
    
    # Extract data
    phi_deg = projection_data['phi_deg']
    theta_deg = projection_data['theta_deg']
    importance_values = projection_data['importance_values']
    vmin = projection_data['vmin']
    vmax = projection_data['vmax']
    
    xticks = [-180, 180]
    yticks = [0, 180]
    
    num_alg = len(importance_values)
    
    if layout == 'grid':
        ncols = min(3, num_alg)
        nrows = (num_alg + ncols - 1) // ncols
        if figsize is None:
            figsize = (4.5 * ncols, 4.5 * nrows)  # Adjusted for square aspect ratio
    else:  
        nrows, ncols = 1, num_alg
        if figsize is None:
            figsize = (5 * num_alg, 5)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    plt.style.use('seaborn-v0_8-whitegrid')
    base_fontsize = plt.rcParams.get('font.size', 14)
    title_fontsize = base_fontsize + 4
    axis_fontsize = base_fontsize + 2
    
    last_sc = None
    flat_axes = axes.flatten()

    for idx, (alg_name, imp_values) in enumerate(importance_values.items()):
        if idx < len(flat_axes):
            ax = flat_axes[idx]
            last_sc = ax.scatter(phi_deg, theta_deg, c=imp_values, cmap=cmap,
                                s=point_size, vmin=vmin, vmax=vmax, alpha=0.8, 
                                edgecolors='black', linewidths=0.5)
            
            ax.set_xlabel("Phi (°)", fontsize=axis_fontsize)
            ax.set_ylabel("Theta (°)", fontsize=axis_fontsize)
            ax.set_title(alg_name, fontsize=title_fontsize)
            ax.set_xticks(xticks)
            ax.set_yticks(yticks)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Ensure square aspect ratio
            #ax.set_aspect('equal', adjustable='box')

    for i in range(idx+1, len(flat_axes)):
        flat_axes[i].set_visible(False)



    plt.suptitle("2D Projection of Surface Residue Values", fontsize=title_fontsize+2, y=0.98)
    
    # Increase spacing between subplots
    fig.subplots_adjust(wspace=0.4, hspace=0.4)


    cbar = fig.colorbar(last_sc, ax=axes.ravel().tolist())
    cbar.set_label("Residue Importance", fontsize=axis_fontsize+4)
    # Set colorbar tick positions and labels to [0, 1]
    cbar.set_ticks([0, 1])

    # Increase font size of colorbar ticks
    cbar.ax.tick_params(labelsize=axis_fontsize + 2)  # Adjust as needed

    if out_file:
        plt.savefig(out_file, dpi=dpi, bbox_inches='tight')

    plt.show()

    return fig, axes

def plot_residue_projections_multiple(pdb_path, importance_dict, surface_threshold=30, out_file=None):
    """
    Compare different importance lists by projecting their values onto the protein surface.
    
    This is a wrapper function that combines the processing and plotting steps.
    
    Parameters:
        pdb_path (str): Path to the PDB file.
        importance_dict (dict): Dictionary with algorithm names as keys and lists of importance values.
        surface_threshold (float): ASA threshold (in Å²) for defining surface residues. Default: 30.
        out_file (str): Optional path to save the resulting plot (e.g., "multi_projection.png").
    
    Returns:
        tuple: (fig, axes) matplotlib figure and axes objects
    """
    # Process the PDB data
    projection_data = process_pdb_for_projections(pdb_path, importance_dict, surface_threshold)
    
    # Plot the projections
    fig, axes = plot_residue_projections(projection_data, out_file=out_file)
    
    return fig, axes

def plot_nature_style_bar(csv_path1, csv_path2, output_path=None):
    """
    Reads two CSV files and creates a Nature-style bar chart for the first dataset.
    - First three bars are light blue, next three bars are red.
    - A thick dashed black line marks the best value from the second dataset.

    Parameters:
        csv_path1 (str): Path to the first CSV file (to be plotted).
        csv_path2 (str): Path to the second CSV file (for reference line only).
        output_path (str, optional): Path to save the figure. If None, shows the plot.
    """
    
    # Load the first CSV (to be plotted)
    df1 = pd.read_csv(csv_path1)

    # Load the second CSV (for reference line)
    df2 = pd.read_csv(csv_path2)
    best_value = df2['Value'].max()  # Best value for reference line

    # Define colors: First 3 bars → Light blue, Next 3 bars → Red
    colors = ['#A8DADC'] * 3 + ['#ff000d'] * (len(df1) - 3)  

    # Nature-style figure settings
    sns.set_style("whitegrid")
    plt.figure(figsize=(6, 8))

    # Create bar plot with specified colors
    ax = sns.barplot(data=df1, x="Key", y="Value", palette=colors)

    # Add dashed thick black line for best value from second CSV
    plt.axhline(y=best_value, color='black', linestyle='dashed', linewidth=2)

    # Labels and formatting
    plt.xlabel("", fontsize=14)
    plt.ylabel("Data Based Performance", fontsize=14)
    plt.xticks(rotation=45, fontsize=12, ha='right')
    plt.yticks(fontsize=12)

    # Remove unnecessary spines
    sns.despine()

    # Save or show the figure
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()

if __name__ == "__main__":

    input_pdb = "./Data/Antibodies/bionemo.pdb"  # Replace with actual PDB file path
    sequence_file_path ="./Data/Antibodies/sequences.fasta"
    focus, focus_aligned = process_fasta(sequence_file_path)

    # Example usage    ## naive method
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
    values_list = [[(l-min(sublist))/(max(sublist)-min(sublist)) for l in sublist] for sublist in values_list]
    algorithm_names = names_list
    importance_dict = dict(zip(names_list, values_list))

    plot_residue_projections_multiple(input_pdb, importance_dict=importance_dict, out_file="TestResults/Antibodies/pdbs/projection.png")

    plot_nature_style_bar('TestResults/Antibodies/naive_analysis/merit_values.csv', 
                          'TestResults/Antibodies/existing_analysis/merit_values.csv', 
                          output_path='TestResults/Antibodies/data_bar_plot.png')