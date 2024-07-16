import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
 
def map_scores_to_aligned(aligned_seq, unaligned_scores):
    # Initialize the output array with NaN
    aligned_scores = np.full(len(aligned_seq), 0)
    
    unaligned_index = 0
    for i, residue in enumerate(aligned_seq):
        if residue != '-':
            if unaligned_index < len(unaligned_scores):
                aligned_scores[i] = unaligned_scores[unaligned_index]
                unaligned_index += 1
    
    return aligned_scores

def construct_elbo_lists(input_string, file_path):
    # Load the CSV file
    data = pd.read_csv(file_path)
    
    # Initialize the lists
    pos_elbo = [0] * len(input_string)
    neg_elbo = [0] * len(input_string)
    abs_elbo = [0] * len(input_string)
    
    # Create a dictionary for quick lookup
    delta_elbo_dict = data.set_index('position')['delta_elbo'].to_dict()
    
    # Populate the lists
    for idx, char in enumerate(input_string):
        if (idx + 1) in delta_elbo_dict:  # Positions are 1-based in the CSV
            delta_elbo = delta_elbo_dict[idx + 1]
            pos_elbo[idx] = delta_elbo
            neg_elbo[idx] = -delta_elbo
            abs_elbo[idx] = abs(delta_elbo)
    
    return pos_elbo, neg_elbo, abs_elbo

def load_and_convert_plmc(file_path):
    # Load the CSV file
    data = pd.read_csv(file_path, header=None)
    
    # Extract the column into a list
    values = data[0].tolist()
    
    # Create the negative values list
    neg_values = [-1 * value for value in values]
    
    return values, neg_values

def construct_value_list_hotspot(input_string, file_path):
    # Load the CSV file
    data = pd.read_csv(file_path, header=None)
    
    # Initialize the list with NaNs
    value_list = [0] * len(input_string)
    
    # Create a dictionary for quick lookup
    value_dict = data.set_index(0)[1].to_dict()
    
    # Populate the list
    for idx, char in enumerate(input_string):
        if (idx + 1) in value_dict:  # Positions are 1-based in the CSV
            value_list[idx] = value_dict[idx + 1]
    
    return value_list

def reduce_aligned_to_unaligned(aligned_seq, aligned_scores):
    unaligned_scores = []
    for i, residue in enumerate(aligned_seq):
        if residue != '-':
            unaligned_scores.append(aligned_scores[i])
    return unaligned_scores

def main_loader(aligned_seq,unaligned_seq,elbo_path,plmc_path,hotspot_path):

    hotspot_data = construct_value_list_hotspot(unaligned_seq, hotspot_path)
    hotspot_data = map_scores_to_aligned(aligned_seq, hotspot_data)

    plmc_val,plmc_neg = load_and_convert_plmc(plmc_path)

    pos_elbo, neg_elbo, abs_elbo = construct_elbo_lists(aligned_seq,elbo_path)

    data = [hotspot_data,plmc_val,plmc_neg, pos_elbo, neg_elbo, abs_elbo]

    new_data = [reduce_aligned_to_unaligned(aligned_seq,data_i) for data_i in data]

    return (data,new_data)

def naive_loader(aligned_seq,unaligned_seq,path_to_data): #'MI','Theil Adjusted MI','Cramer Adjusted MI'
    entropies_data = f"{path_to_data}/entropies_data.pkl"
    information_data = f"{path_to_data}/information_data.pkl"
    with open(entropies_data, 'rb') as file:
        ent1, ent2, ent3 = pickle.load(file)
    with open(information_data, 'rb') as file:
        inf1, inf2, inf3 = pickle.load(file)

    data = [ent1,ent2,ent3,inf1,inf2,inf3]

    new_data = [reduce_aligned_to_unaligned(aligned_seq,data_i) for data_i in data]

    return (data,new_data)


# Example usage
# aligned_seq = "ACDEFGHIKLMNPQRSTVWY"
# unaligned_sequence = ""
# pos_elbo, neg_elbo, abs_elbo = construct_elbo_lists(input_string, file_path)

# pos_elbo, neg_elbo, abs_elbo
