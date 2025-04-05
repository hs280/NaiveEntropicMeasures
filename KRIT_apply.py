import sys
import os
import numpy as np 
from pathlib import Path  # Module for working with filesystem paths
script_path = os.path.abspath(__file__)
curent_path = os.path.dirname(script_path)
directory_folder = str(Path(curent_path).parents[0])
sys.path.insert(0, directory_folder)
import Bin as Bin
import pickle as pkl
import matplotlib.pyplot as plt
import os


def run_analysis(msa_path, target_path, store_path, protein_family, results_folder="./Results_AC"):
    """
    Run the information theory and cross-correlation analysis on a single MSA and target data file.
    
    Parameters:
        msa_path (str): Path to the aligned MSA data file.
        target_path (str): Path to the target data file.
        store_path (str): Folder to store output files (e.g., plots and pickle files).
        protein_family (str): Name of the protein family for labeling plots.
        results_folder (str, optional): Folder to store aggregated results. Default is "./Results_AC".
    """
    
    # Ensure the output directories exist
    os.makedirs(store_path, exist_ok=True)
    os.makedirs(results_folder, exist_ok=True)
    
    # Read data
    aligned_residues_df = Bin.fasta_to_dataframe(msa_path)
    target_df = Bin.read_dat_file(target_path)
    
    # Define output file paths
    cross_correl_data = os.path.join(store_path, "cross_correlations.pkl")
    entropies_data    = os.path.join(store_path, "entropies_data.pkl")
    information_data  = os.path.join(store_path, "information_data.pkl")
    
    # Run the information theory analysis
    Bin.handle_info_theory('y', 
                           aligned_residues_df, store_path, 
                           cross_correl_data, entropies_data,
                           'y', 
                           store_path, target_df, information_data)
    
    # Generate individual plots for the MSA and analysis results
    Bin.plot_residue_probability_heatmap(aligned_residues_df, column_name='Sequence', save_folder=store_path)
    Bin.remake_plots(entropies_data, information_data, cross_correl_data, save_folder=store_path, 
                     Protein_Family=protein_family)
    
    # For the single dataset, wrap the output paths in lists
    entropies_dict = Bin.process_list_autocorrelation([entropies_data], 
                                                      ['Entropy', 'Theil Entropy', 'Cramer Entropy'], 
                                                      [store_path])
    information_dict = Bin.process_list_autocorrelation([information_data], 
                                                        ['MI', 'Theil MI', 'Cramer MI'], 
                                                        [store_path])
    
    # Plot bar charts with error bars based on the processed data
    Bin.plot_bar_chart_with_error_bars(entropies_dict, information_dict, rotation=90, outfolder=results_folder)
    
    # Compute and plot the average cross-correlation matrix
    cross_corr_labels = ['Entropy', 'Theil Entropy', 'Cramer Entropy', 'MI', 'Theil MI', 'Cramer MI']
    cross_corr_mat = Bin.average_cross_correlation_matrix([entropies_data], [information_data],
                                                          plot_flag=True, labels=cross_corr_labels,
                                                          Protein_Families=[protein_family],
                                                          Store_paths=[store_path])
    Bin.plot_colormap_cross_corr(cross_corr_mat, cross_corr_labels, cbar_label='CrossCorrelation',
                                 save_path=os.path.join(results_folder, "cross_correlation_colormap.png"))

# Example usage:
if __name__ == "__main__":
    msa_file      = "./Data/Antibodies/sequences.fasta"
    target_file   = "./Data/Antibodies/pred_affinity.dat"
    output_folder = "./TestResults/Antibodies"
    protein_name  = "COV Antibodies"
    results_folder = output_folder

    run_analysis(msa_file, target_file, output_folder, protein_name,results_folder)


    