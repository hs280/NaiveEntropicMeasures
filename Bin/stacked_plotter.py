import matplotlib.pyplot as plt
import numpy as np

def normalize(data):
    """
    Normalizes each list in the data to be between 0 and 1.
    """
    data = np.array(data)
    min_val = np.min(data)
    max_val = np.max(data)
    return (data - min_val) / (max_val - min_val + 1E-6)

def calculate_optimal_values(data, key_residues):
    """
    Calculate the optimal values and n_optimal for each list in the dataset.
    """
    optimal_values = []
    n_optimals = []
    for line_data in data:
        line_data = np.array(line_data)
        sorted_indices = np.argsort(-line_data)  # indices of sorted elements in descending order
        max_expression = 0
        n_optimal = 0
        for n in range(1, len(sorted_indices) + 1):
            top_n_indices = sorted_indices[:n]
            expression_value = len(set(top_n_indices) & set(key_residues))**2 / (n * len(key_residues))
            if expression_value > max_expression:
                max_expression = expression_value
                n_optimal = n/len(line_data)
        optimal_values.append(max_expression)
        n_optimals.append(n_optimal)
    return optimal_values, n_optimals

def plot_stacked_lines(data, labels, legends, vertical_lines, min_distance=1, output_filename='stacked_lines_plot.png', Protein_Families=[]):
    """
    Plots three stacked line plots with a legend in nature style and saves the plot as an image.

    Parameters:
    - data: List of 3 lists, each containing 6 lists of equal length.
    - labels: List of 3 labels for the three sets of lines.
    - legends: List of labels for the legend.
    - vertical_lines: List of 3 lists, each containing positions for vertical dashed lines.
    - min_distance: Minimum distance to maintain between lines (default is 1).
    - output_filename: Filename to save the plot image (default is 'stacked_lines_plot.png').
    - Protein_Families: List of labels for each subplot (optional).
    """
    fig, axs = plt.subplots(3, 1, figsize=(1.5*11.69/2, 11.69), sharex=False)  # Changed sharex to False

    for k, (data_set, label) in enumerate(zip(data, labels)):
        # Normalize each set of lines
        normalized_data_set = [normalize(line_data) for line_data in data_set]

        # Calculate the shift to maintain the minimum distance between lines
        shifts = [0] * len(normalized_data_set)
        for j in range(1, len(normalized_data_set)):
            differences = normalized_data_set[j] - normalized_data_set[j-1]
            min_diff = np.min(differences)
            if min_diff < min_distance:
                shifts[j] = shifts[j-1] + (min_distance - min_diff)
            else:
                shifts[j] = shifts[j-1]

        # Plot the shifted lines
        for j, line_data in enumerate(normalized_data_set):
            axs[k].plot(line_data + shifts[j], label=legends[j] if k == 2 else None,linewidth=2)  # Only label the last subplot

        # Add vertical dashed lines
        for vline in vertical_lines[k]:
            axs[k].axvline(x=vline, color='grey', linestyle='--')

        # Styling the plot to have a 'nature' feel
        axs[k].set_facecolor('white')
        axs[k].grid(False)
        if len(Protein_Families) > 0:
            axs[k].set_ylabel(Protein_Families[k], fontsize=24)
        axs[k].tick_params(axis='y', left=False, labelleft=False)

        # Set x-axis ticks and labels for all subplots
        axs[k].set_xticks([0,len(normalized_data_set[0])-1])
        axs[k].set_xticklabels([1, len(normalized_data_set[0])], fontsize=20)

        # Set x-axis label for bottom subplot only
        if k == 2:
            axs[k].set_xlabel('Location', fontsize=20)

    # Adjust spacing and alignment
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2, hspace=0.4)  # Adjust bottom padding as needed


    # Add legend for the last subplot only
    handles, _ = axs[2].get_legend_handles_labels()

    legend = fig.legend(handles, legends, loc='lower center', bbox_to_anchor=(0.5, 0), frameon=False, ncol=len(legends)//2, fontsize=20, handlelength=1)

    for line in legend.get_lines():
        line.set_linewidth(5)  # Set thicker line width for legend lines


    plt.savefig(output_filename, dpi=300)
    plt.show()




def plot_optimal_bars(data, key_residues, legends, output_filename='optimal_bars_plot.png'):
    """
    Plots two bar plots showing the optimal values and n_optimal for each line in the dataset.

    Parameters:
    - data: List of 3 lists, each containing 6 lists of equal length.
    - key_residues: List of 3 lists, each containing key residue positions.
    - legends: List of labels for the legend.
    - output_filename: Filename to save the plot image (default is 'optimal_bars_plot.png').
    """
    fig, axs = plt.subplots(1, 2, figsize=(11.69, 8.27/3))

    optimal_values_all = []
    n_optimals_all = []
    for data_set, key_set in zip(data, key_residues):
        optimal_values, n_optimals = calculate_optimal_values(data_set, key_set)
        optimal_values_all.append(optimal_values)
        n_optimals_all.append(n_optimals)

    # Define the positions for the bars
    bar_width = 0.2
    num_lines = len(data[0])
    num_sets = len(data)
    indices = np.arange(num_lines)
    group_positions = np.arange(num_sets) * (num_lines + 1) * bar_width

    # Plot the optimal values
    for line in range(num_lines):
        positions = group_positions + line * bar_width
        values = [opt_vals[line] for opt_vals in optimal_values_all]
        axs[0].bar(positions, values, bar_width, label=legends[line])
    axs[0].set_yticks([0, 1])
    axs[0].set_yticklabels([0, 1], fontsize=14)
    axs[0].set_ylabel('Optimal Value', fontsize=15)
    axs[0].set_xticks(group_positions + bar_width * (num_lines / 2 - 0.5))
    axs[0].set_xticklabels(['Set 1', 'Set 2', 'Set 3'], fontsize=14)

    # Plot the n_optimal values
    for line in range(num_lines):
        positions = group_positions + line * bar_width
        values = [n_opts[line] for n_opts in n_optimals_all]
        axs[1].bar(positions, values, bar_width)
    axs[1].set_yticks([0, 1])
    axs[1].set_yticklabels([0, 1], fontsize=14)
    axs[1].set_ylabel('Proportion Required', fontsize=15)
    axs[1].set_xticks(group_positions + bar_width * (num_lines / 2 - 0.5))
    axs[1].set_xticklabels(['Set 1', 'Set 2', 'Set 3'], fontsize=14)

    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.2), frameon=False, ncol=num_lines, fontsize=14)

    plt.tight_layout(rect=[0, 0.15, 1, 1])
    plt.savefig(output_filename, dpi=300)
    plt.show()

def calculate_minimal_values(data, key_residues):
    """
    Calculate the minimum value of N such that all key residues are found in each list in the dataset.
    """
    minimal_values = []
    for line_data in data:
        line_data = np.array(line_data)
        sorted_indices = np.argsort(-line_data)  # indices of sorted elements in descending order
        for n in range(1, len(sorted_indices) + 1):
            top_n_indices = sorted_indices[:n]
            if all(residue in top_n_indices for residue in key_residues):
                minimal_values.append(n)
                break
        else:
            minimal_values.append(len(sorted_indices))  # if not all key residues are found, use the full length
    return minimal_values


def normalized_crosscorrelation_with_pulses(lists_of_lists, indices):
    results = []
    
    for sublist in lists_of_lists:
        
        min_val = np.min(sublist)
        max_val = np.max(sublist)

        sublist = (sublist - min_val) / (max_val - min_val)

        test = sublist*0
        test[indices] = 1
        
        # Calculate the normalized value
        normalized_value = pulse_distance(sublist,indices)
        offset_factor = min_adjacent_swaps(sublist,indices)
        
        
        results.append(normalized_value*(1-offset_factor))
    
    return results

def min_adjacent_swaps(nums, gt_indices):
    # Step 1: Get the n highest values from nums and their corresponding binary form
    n = len(gt_indices)  # Number of highest values
    length = len(nums)
    max_swaps = n*(length-n)

    # if max_swaps==0:
    #     print('')
    
    # Step 2: Sort nums and get the indices of the n highest values
    sorted_indices = sorted(range(length), key=lambda i: nums[i], reverse=True)[:n]

    # Step 3: Sort both `sorted_indices` and `gt_indices` for a one-to-one matching
    sorted_indices.sort()
    gt_indices.sort()

    # Step 4: Count the number of swaps needed to move each element in sorted_indices
    # to match the corresponding position in gt_indices
    def count_inversions(arr1, arr2):
        swaps = 0
        for i in range(n):
            # Check how far each element in sorted_indices (arr1) is from gt_indices (arr2)
            swaps += abs(arr1[i] - arr2[i])
        return swaps

    return count_inversions(sorted_indices, gt_indices)/max_swaps

def pulse_distance(x,indices):
    x_2 = x/np.sum(x)
    return np.sum(x_2[indices])

def cosine_distance(x, y):
    numerator = np.dot(x, y)
    denominator = np.sqrt(np.dot(x, x)) * np.sqrt(np.dot(y, y))
    cosine_similarity = numerator / denominator if denominator != 0 else 0
    # Cosine Distance between 0 and 1
    return 1 - cosine_similarity


def normalized_cross_correlation(x, y):
    # Subtract mean from signals
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    # Numerator of NCC
    numerator = np.sum((x - x_mean) * (y - y_mean))
    
    # Denominator of NCC
    denominator = np.sqrt(np.sum((x - x_mean)**2) * np.sum((y - y_mean)**2))
    
    # Return normalized cross-correlation
    return numerator / denominator if denominator != 0 else 0

def ncc_distance(x, y):
    ncc = normalized_cross_correlation(x, y)
    # Convert NCC to a distance measure between 0 and 1
    distance = 1 - (ncc + 1) / 2
    return distance

def plot_minimal_bars(data, key_residues, legends, output_filename='minimal_bars_plot.png'):
    """
    Plots a bar plot showing the minimum value of N such that all key residues are found for each line in the dataset.

    Parameters:
    - data: List of 3 lists, each containing 6 lists of equal length.
    - key_residues: List of 3 lists, each containing key residue positions.
    - legends: List of labels for the legend.
    - output_filename: Filename to save the plot image (default is 'minimal_bars_plot.png').
    """
    #fig, ax = plt.subplots(figsize=(11.69, 8.27 / 3))

    minimal_values_all = []
    min_ratio_dict = {legend: [] for legend in legends}
    for data_set, key_set in zip(data, key_residues):
        minimal_values = normalized_crosscorrelation_with_pulses(data_set, key_set)
        minimal_values_all.append(minimal_values)
        for legend, minimal_value in zip(legends, minimal_values):
            min_ratio_dict[legend].append(minimal_value)

    # Define the positions for the bars
    # bar_width = 0.2
    # num_lines = len(data[0])
    # num_sets = len(data)
    # indices = np.arange(num_lines)
    # group_positions = np.arange(num_sets) * (num_lines + 1) * bar_width

    # # Plot the minimal values
    # for line in range(num_lines):
    #     positions = group_positions + line * bar_width
    #     values = [min_ratio_dict[legends[line]][set_idx] for set_idx in range(num_sets)]
    #     ax.bar(positions, values, bar_width, label=legends[line])

    # ax.set_yticks([0, 1])
    # ax.set_yticklabels([0, 1], fontsize=14)
    # ax.set_ylabel('Proportion of Key Residues', fontsize=15)
    # ax.set_xticks(group_positions + bar_width * (num_lines / 2 - 0.5))
    # ax.set_xticklabels(['Set 1', 'Set 2', 'Set 3'], fontsize=14)

    # fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.2), frameon=False, ncol=num_lines, fontsize=14)

    # plt.tight_layout(rect=[0, 0.15, 1, 1])
    # plt.savefig(output_filename, dpi=300)
    # plt.show()

    return min_ratio_dict

# Test script
if __name__ == "__main__":
    data = [
        [[1, 2, 3, 4], [2, 3, 4, 5], [1, 1, 1, 1], [4, 5, 6, 7], [5, 6, 7, 8], [6, 7, 8, 9]],
        [[2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7], [5, 6, 7, 8], [6, 7, 8, 9], [7, 8, 9, 10]],
        [[3, 4, 5, 6], [4, 5, 6, 7], [5, 6, 7, 8], [6, 7, 8, 9], [7, 8, 9, 10], [8, 9, 10, 11]]
    ]
    labels = ['Set 1', 'Set 2', 'Set 3']
    legends = ['Line 1', 'Line 2', 'Line 3', 'Line 4', 'Line 5', 'Line 6']
    vertical_lines = [[1, 3], [2, 4], [1, 2, 3]]
    key_residues = [[1, 3], [2, 4], [1, 2, 3]]

    plot_stacked_lines(data, labels, legends, vertical_lines)
    plot_optimal_bars(data, key_residues, legends)
