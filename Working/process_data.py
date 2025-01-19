import sys
import os
import numpy as np 
from pathlib import Path  # Module for working with filesystem paths
import pickle as pkl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_ind
import textwrap
from matplotlib.ticker import AutoMinorLocator

script_path = os.path.abspath(__file__)
curent_path = os.path.dirname(script_path)
directory_folder = str(Path(curent_path).parents[0])
sys.path.insert(0, directory_folder)
import Bin as Bin

def merge_dicts(dict_list, names=None):
    merged_dict = {}

    # Merge dictionaries
    for d in dict_list:
        for key, value in d.items():
            if key not in merged_dict:
                merged_dict[key] = []
            if isinstance(value, (list, np.ndarray)):
                merged_dict[key].extend(value)
            else:
                merged_dict[key].append(value)

    # Reorder dictionary based on names if provided
    if names:
        ordered_dict = {name: merged_dict.get(name, []) for name in names}
        return ordered_dict

    return merged_dict



def process_results(paths,names=None):
    results = []
    for path in paths:
        result_paths = Bin.calculate_sum(path)
        results.append(result_paths)

    return merge_dicts(results,names)

import matplotlib.pyplot as plt
import numpy as np

def plot_bar_chart_with_error_bars(dict1, dict2, labels=['dict1', 'dict2'], ylabel='Data Based \n Performance',
                                   yticks=[0, 1], ylim=(0, 1.1), yticklabels=[0,1],x_labels=[],
                                   rotation=90,figsize=(11.69, 11.69),figsize_cross_corr=(11.69/1.5, 11.69/1.5)):
    keys1 = list(dict1.keys())
    keys2 = list(dict2.keys())
    all_keys = keys1 + keys2
    
    values1 = [np.mean(dict1[key]) for key in keys1]
    errors1 = [np.std(dict1[key]) for key in keys1]
    values2 = [np.mean(dict2[key]) for key in keys2]
    errors2 = [np.std(dict2[key]) for key in keys2]

    # Number of groups
    num_groups1 = len(keys1)
    num_groups2 = len(keys2)

    #plt.figure(figsize=(11.69/2, 11.69/2))

    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    # Define bar width and positions
    bar_width = 0.45
    index1 = np.arange(num_groups1)
    index2 = np.arange(num_groups1, num_groups1 + num_groups2)

    # Plot bars with error bars
    # Plot bars with error bars
    bar1 = ax.bar(index1, values1, bar_width, yerr=errors1, label='Label 1', capsize=10, color='lightblue',
                error_kw={'linewidth': 2})  # Set error bar line width and cap size
    bar2 = ax.bar(index2, values2, bar_width, yerr=errors2, label='Label 2', capsize=10, color='r',
                error_kw={'linewidth': 2})  # Set error bar line width and cap size

    all_keys_alt = [textwrap.fill(y_label, width=8) for y_label in x_labels]



    # Add horizontal dashed line at 1
    ax.axhline(y=1, color='gray', linestyle='--')

    # Set labels and title
    ax.set_ylabel(ylabel,fontsize=24)
    ax.set_xticks(np.concatenate((index1, index2)))
    ax.set_xticklabels(all_keys_alt, rotation=rotation,fontsize=20)

    ax.minorticks_on()
    ax.yaxis.set_minor_locator(AutoMinorLocator(21))
    ax.grid(True, which='both', linestyle='--', alpha=0.5,linewidth=1)
    
    #
    # Place legend outside the plot area
    #ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=14)

    # Adjust the layout to make room for the legend
    plt.subplots_adjust(right=0.75)

    # Set y-axis limits and ticks
    ax.set_ylim(ylim)
    ax.set_yscale('symlog')
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontsize=20, rotation=0)


    # Show plot
    plt.tight_layout()
    #plt.show()
    plt.savefig('bar_chart_databased_all.png', dpi=1200)


def calculate_p_value(dict1, dict2):
    # Find the key with the highest average in dict1
    key1 = max(dict1, key=lambda k: np.mean(dict1[k]))
    values1 = dict1[key1]

    # Find the key with the highest average in dict2
    key2 = max(dict2, key=lambda k: np.mean(dict2[k]))
    values2 = dict2[key2]

    # Perform a t-test between the two sets of values
    t_stat, p_value = ttest_ind(values1, values2)

    return p_value, key1, key2

def calculate_hedges_g(values1, values2):
    n1, n2 = len(values1), len(values2)
    mean1, mean2 = np.mean(values1), np.mean(values2)
    std1, std2 = np.std(values1, ddof=1), np.std(values2, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * std1 ** 2 + (n2 - 1) * std2 ** 2) / (n1 + n2 - 2))

    # Calculate Cohen's d
    cohens_d = (mean1 - mean2) / pooled_std

    # Apply correction factor for Hedge's g
    correction_factor = 1 - (3 / (4 * (n1 + n2) - 9))
    hedges_g = cohens_d * correction_factor

    return hedges_g

def generate_comparison_table(dict1, dict2):
    all_keys = list(dict1.keys()) + list(dict2.keys())
    data = []

    # Calculate mean and standard deviation for each key
    for key in all_keys:
        if key in dict1:
            values = dict1[key]
        else:
            values = dict2[key]
        mean_val = np.mean(values)
        std_val = np.std(values, ddof=1)
        data.append([key, mean_val, std_val])
    
    # Create initial DataFrame
    df = pd.DataFrame(data, columns=['Key', 'Mean Performance', 'Stdev Performance'])

    # Calculate Hedge's g for each pair of keys
    hedge_matrix = np.zeros((len(all_keys), len(all_keys)))

    for i, key1 in enumerate(all_keys):
        for j, key2 in enumerate(all_keys):
            if i != j:
                if key1 in dict1 and key2 in dict1:
                    values1, values2 = dict1[key1], dict1[key2]
                elif key1 in dict2 and key2 in dict2:
                    values1, values2 = dict2[key1], dict2[key2]
                elif key1 in dict1 and key2 in dict2:
                    values1, values2 = dict1[key1], dict2[key2]
                else:
                    values1, values2 = dict2[key1], dict1[key2]

                hedge_matrix[i, j] = calculate_hedges_g(values1, values2)

    # Create a DataFrame for the Hedge's g matrix
    hedge_df = pd.DataFrame(hedge_matrix, columns=all_keys, index=all_keys)

    # Concatenate the two DataFrames
    result_df = pd.concat([df.set_index('Key'), hedge_df], axis=1)

    return result_df, hedge_df

from scipy.stats import t

def ttest_unpaired_from_stats(mean1, std1, n1, mean2, std2, n2):
    """Calculate p-value for an unpaired t-test given summary statistics."""
    se1 = std1 / np.sqrt(n1)
    se2 = std2 / np.sqrt(n2)
    se_diff = np.sqrt(se1**2 + se2**2)
    
    t_stat = (mean1 - mean2) / se_diff
    df = 8 #((se1**2 + se2**2)**2) / (((se1**2)**2 / (n1 - 1)) + ((se2**2)**2 / (n2 - 1)))
    
    p_val = 2 * t.sf(np.abs(t_stat), df)
    return p_val

def generate_comparison_table_with_pvalues(dict1, dict2, n_trials):
    all_keys = list(set(list(dict1.keys()) + list(dict2.keys())))
    data = []

    # Calculate mean and standard deviation for each key
    stats = {}
    for key in all_keys:
        if key in dict1:
            values = dict1[key]
        else:
            values = dict2[key]
        mean_val = np.mean(values)
        std_val = np.std(values, ddof=1)
        stats[key] = (mean_val, std_val, n_trials)
        data.append([key, mean_val, std_val])

    # Create initial DataFrame
    df = pd.DataFrame(data, columns=['Key', 'Mean Performance', 'Stdev Performance'])

    # Calculate p-values for each pair of keys
    pvalue_matrix = np.ones((len(all_keys), len(all_keys)))  # Initialize with 1s (default for diagonal)

    for i, key1 in enumerate(all_keys):
        for j, key2 in enumerate(all_keys):
            if i != j:
                mean1, std1, n1 = stats[key1]
                mean2, std2, n2 = stats[key2]

                # Perform t-test to compute p-value
                p_val = ttest_unpaired_from_stats(mean1, std1, n1, mean2, std2, n2)

                # Adjust p-value based on n_trials (Bonferroni correction)
                corrected_p_val = min(p_val * n_trials, 1.0)  # Ensure p-value is at most 1.0
                pvalue_matrix[i, j] = p_val

    # Create a DataFrame for the p-value matrix
    pvalue_df = pd.DataFrame(pvalue_matrix, columns=all_keys, index=all_keys)

    # Concatenate the two DataFrames
    result_df = pd.concat([df.set_index('Key'), pvalue_df], axis=1)

    return result_df, pvalue_df


import matplotlib.colors as mcolors

def plot_colormap(hedge_df,labels):
    plt.close()
    plt.figure(figsize=(11.69, 11.69))
    
    # Set the limits for the colormap
    vmin, vmax = -1, 1

    # Create the heatmap
    ax = sns.heatmap(hedge_df, annot=False, cmap='vlag', center=0, linewidths=1, vmin=vmin, vmax=vmax)
    
    # Determine the midpoint to draw the lines
    mid_point = len(hedge_df) / 2
    
    # Draw vertical and horizontal black lines
    plt.axvline(x=mid_point, color='black', linewidth=4)
    plt.axhline(y=mid_point, color='black', linewidth=4)
    
    # Customize the colorbar
    cbar = ax.collections[0].colorbar
    cbar.set_ticks([vmin, vmin/2, 0, vmax/2, vmax])
    cbar.set_ticklabels([f'<{vmin}', f'{vmin/2}', '0', f'{vmax/2}', f'>{vmax}'])
    cbar.ax.tick_params(labelsize=20)  
    cbar.set_label('Hedge\'s-G Data measures', fontsize=24)

    all_keys_alt = [textwrap.fill(y_label, width=8) for y_label in labels]

    ax.set_xticklabels(all_keys_alt, fontsize=20,rotation=90)
    ax.set_yticklabels(all_keys_alt, fontsize=20,rotation=0)

        # Manually draw lines between cells using annotate
    for i in range(hedge_df.shape[0]):
        ax.annotate('', xy=(-1, i), xytext=(hedge_df.shape[1], i),
                    xycoords='data', textcoords='data',
                    arrowprops=dict(color='grey', linestyle='--', linewidth=4, arrowstyle='-',alpha=0.15),
                    annotation_clip=False)

    for j in range(hedge_df.shape[1]):
        ax.annotate('', xy=(j + 1, 0), xytext=(j + 1, hedge_df.shape[0] + 1),
                    xycoords='data', textcoords='data',
                    arrowprops=dict(color='grey', linestyle='--', linewidth=4, arrowstyle='-',alpha=0.15),
                    annotation_clip=False)


    plt.tight_layout()
    plt.savefig('Hedge_g_databased_al.png', dpi=1200)

paths_existing = ['./Results/AlkMonoxygenase/databased','./Results/BacRhod/databased','./Results/GFP_fluor/databased','./Results/GFP_emission/databased','./Results/GFP_QY/databased']
paths_naive = ['./Results/AlkMonoxygenase/databased_naive','./Results/BacRhod/databased_naive','./Results/GFP_fluor/databased_naive','./Results/GFP_emission/databased_naive','./Results/GFP_QY/databased_naive']

# 
existing_names = ['HotspotWizard', 'EV+', 'EV-', 'Deep+', 'Deep-', 'Abs(Deep)']

naive_names = ['Ent','Theil Ent','Cramer Ent','MI','Theil MI','Cramer MI']

labels = ['Hotspot Wizard', 'EV+', 'EV-', 'Deep+', 'Deep-', 'AbsDeep','Ent','Theil Ent','Cramer Ent','MI','Theil MI','Cramer MI']

results_existing = process_results(paths_existing,existing_names)

results_naive = process_results(paths_naive,naive_names)

plot_bar_chart_with_error_bars(results_existing,results_naive,labels=['Existing','Naive'],x_labels=labels)

result_df, hedge_df = generate_comparison_table(results_existing,results_naive)

p_res_df, p_df = generate_comparison_table_with_pvalues(results_existing,results_naive,5)

plot_colormap(hedge_df,labels=labels)

print(result_df)

print(p_res_df)