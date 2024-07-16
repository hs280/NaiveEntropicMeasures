import numpy as np
import pandas as pd

def map_scores_to_aligned(aligned_seq, unaligned_scores):
    # Initialize the output array with NaN
    aligned_scores = np.full(len(aligned_seq), np.nan)
    
    unaligned_index = 0
    for i, residue in enumerate(aligned_seq):
        if residue != '-':
            if unaligned_index < len(unaligned_scores):
                aligned_scores[i] = unaligned_scores[unaligned_index]
                unaligned_index += 1
    
    return aligned_scores

def reduce_aligned_to_unaligned(aligned_seq, aligned_scores):
    unaligned_scores = []
    for i, residue in enumerate(aligned_seq):
        if residue != '-':
            unaligned_scores.append(aligned_scores[i])
    return unaligned_scores

# Load unaligned scores from the .dat file
def load_scores(file_path):
    data = pd.read_csv(file_path, header=None)
    unaligned_indices = data[0].values
    scores = data[1].values
    return unaligned_indices, scores

# # Example usage
# aligned_seq = "M--K-T-WQ-"
# unaligned_seq = "MKTWQ"
# unaligned_indices, scores = load_scores('unaligned_scores.dat')

# # Assuming the unaligned indices match the unaligned sequence
# mapped_scores = map_scores_to_aligned(aligned_seq, scores)
# print(mapped_scores)
