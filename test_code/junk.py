import numpy as np

def normalized_crosscorrelation_with_pulses(lists_of_lists, indices):
    results = []
    
    for sublist in lists_of_lists:
        # Get the values at specified indices
        values_at_indices = np.array(sublist)[indices]
        
        # Sum of values at specified indices
        sum_at_indices = np.sum(values_at_indices)
        
        # Sum of all values in the sublist
        total_sum = np.sum(sublist)
        
        # Calculate the normalized value
        normalized_value = sum_at_indices / total_sum
        
        results.append(normalized_value)
    
    return results


# Example usage:
lists_of_lists = [[0, 3, 0, 4, 0], [0, 1, 0, 1, 0], [3, 4, 3, 4, 3]]
indices = [1, 3]

results = normalized_crosscorrelation_with_pulses(lists_of_lists, indices)
print(results)
