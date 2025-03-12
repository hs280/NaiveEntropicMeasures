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




MSA_list = ['./Data/AlkMonoxygenase/alkanal_monoxygenase_aligned.dat',
            './Data/BacRhod/residues_aligned_reordered.dat',
            './Data/GFP/aligned_sequences.dat',
            './Data/GFP/aligned_sequences.dat',
            './Data/GFP/aligned_sequences.dat'
            ]

Target_list = ['./Data/AlkMonoxygenase/target_phopt.dat',
               './Data/BacRhod/wavelengths.dat',
               './Data/GFP/fp_fluorescence_wavelengths.dat',
               './Data/GFP/fp_emission_wavelengths.dat',
               './Data/GFP/fp_quantum_yield.dat'
               ]

Store_paths = ['./Results/AlkMonoxygenase',
               './Results/BacRhod',
               './Results/GFP_fluor',
               './Results/GFP_emission',
               './Results/GFP_QY'
               ]

Protein_Families = ['Alkane Monoxygenase',
               'Rhodopsin',
               'Fluorescent Proteins',
               'Fluorescent Proteins',
               'Fluorescent Proteins'
               ]

results_folder = "./Results_AC"

os.makedirs(results_folder,exist_ok=True)

entropies_paths = []
inormation_data_paths = []
for i in range(len(MSA_list)):
    MSA = MSA_list[i]
    target_path = Target_list[i]
    outfile = Store_paths[i]
    aligned_residues_df = Bin.fasta_to_dataframe(MSA)
    target_df = Bin.read_dat_file(target_path)
    cross_correl_data = f"{outfile}/cross_correlations.pkl"
    entropies_data = f"{outfile}/entropies_data.pkl"
    information_data = f"{outfile}/information_data.pkl"

    Bin.handle_info_theory('n',aligned_residues_df,outfile,cross_correl_data,entropies_data,
                        'y',outfile,target_df,information_data,
                        )

    entropies_paths.append(entropies_data)
    inormation_data_paths.append(information_data)

    Bin.plot_residue_probability_heatmap(aligned_residues_df,column_name='Sequence',save_folder=outfile)
    Bin.remake_plots(entropies_data,information_data,cross_correl_data,save_folder=outfile,Protein_Family = Protein_Families[i])

entropies_dict = Bin.process_list_autocorrelation(entropies_paths, ['Entropy','Theil Entropy','Cramer Entropy'],Store_paths)
information_dict = Bin.process_list_autocorrelation(inormation_data_paths, ['MI','Theil MI','Cramer MI'],Store_paths)

Bin.plot_bar_chart_with_error_bars(entropies_dict,information_dict,rotation=90,outfolder=results_folder)

cross_corr_labels = ['Entropy','Theil Entropy','Cramer Entropy','MI','Theil MI','Cramer MI']

cross_corr_mat = Bin.average_cross_correlation_matrix(entropies_paths,inormation_data_paths,plot_flag=True,labels = cross_corr_labels,Protein_Families=Protein_Families,Store_paths=Store_paths)

Bin.plot_colormap_cross_corr(cross_corr_mat, cross_corr_labels,cbar_label='CrossCorrelation',save_path=f'{results_folder}/cross_correlation_colormap.png')
