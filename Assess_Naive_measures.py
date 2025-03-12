import sys
import os
import numpy as np 
from pathlib import Path  # Module for working with filesystem paths
import pickle as pkl
script_path = os.path.abspath(__file__)
curent_path = os.path.dirname(script_path)
directory_folder = str(Path(curent_path).parents[0])
sys.path.insert(0, directory_folder)
import Bin as Bin
import csv

def save_dict_to_csv(data_dict, output_path):
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write the header
        writer.writerow(['Key', 'Value'])
        
        # Write the data
        for key, value in data_dict.items():
            writer.writerow([key, value])

def process_data_based(seq_data_path,target_data_path,rank_data,save_path,file_names):
    aligned_residues_df = Bin.fasta_to_dataframe(seq_data_path)
    target_df = Bin.read_dat_file(target_data_path)
    os.makedirs(save_path,exist_ok=True)
    rank_data_path = f'{save_path}/rank_data.pkl'
    with open(rank_data_path,'wb') as f:
        pkl.dump(rank_data,f)
    num_samples=np.inf

    max_seq_length = len(aligned_residues_df.values[0])

    Bin.search_sequence_lengths(save_path, 
                            aligned_residues_df, 
                            target_df, 
                            rank_data_path, 
                            num_samples, 
                            split_fraction=0.2, 
                            max_seq_length=max_seq_length, 
                            num_runs=5,
                            file_names=file_names,
                            alpha=0)
    
    Bin.sanitize_directory(save_path)

    resultr = Bin.calculate_sum(save_path)

    save_dict_to_csv(resultr, f'{save_path}/merit_values.csv')

    return resultr


from typing import Tuple

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

def assess_with_data_measures(sequence_file_path: str, target_file_path: str,
                           identified_residues_folder: str, save_path: str)->list:
    ## load_data 
    focus, focus_aligned = process_fasta(sequence_file_path)
    data,_ = Bin.naive_loader(focus_aligned,focus,identified_residues_folder)
    legends = ['Ent','Theil Ent','Cramer Ent','MI','Theil MI','Cramer MI']


    import ray
    os.environ["RAY_memory_usage_threshold"] = "0.9"
    # Initialize Ray
    ray.init(num_cpus=8)

    result = process_data_based(sequence_file_path,target_file_path,data,save_path,legends)
    ray.shutdown()
    return result

if __name__ =="__main__":
    sequence_file_path ="./Data/Antibodies/sequences.fasta"
    target_file_path = "./Data/Antibodies/pred_affinity.dat"
    identified_residues_folder = './TestResults/Antibodies'
    save_path = './TestResults/Antibodies/naive_analysis'
    assess_with_data_measures(sequence_file_path, target_file_path, identified_residues_folder, save_path)
