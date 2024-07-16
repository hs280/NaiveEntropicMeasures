import numpy as np
import pandas as pd
import scipy.stats as ss
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import mutual_info_regression
import matplotlib.pyplot as plt
from dython.nominal import associations
import numpy as np
from scipy.signal import welch
from scipy.stats import entropy as calc_entropy
from scipy.signal import correlate
import matplotlib.pyplot as plt
import pickle as pkl
import textwrap
from matplotlib.ticker import AutoMinorLocator




import pickle 
import os



def remake_plots(entropies_data,information_data,cross_correl_data,save_folder=None,Protein_Family = None):
    with open(cross_correl_data, 'rb') as f:
        theil_full,chi_full = pickle.load(f)

    with open(entropies_data, 'rb') as f:
        entropy_full,entropy_adjusted_th,entropy_adjusted_chi = pickle.load(f)

    with open(information_data, 'rb') as f:
        mi_full,mi_adjusted_th,mi_adjuested_chi = pickle.load(f)
    
    plot_heatmap(chi_full[::-1,:],x_label='Residue Location',y_label='Residue Location',name='Cramers.png',Label = 'Cramer\'s V',save_folder=save_folder)

    # Plot Theil's U heatmap
    plot_heatmap(theil_full[::-1,:],save_folder=save_folder,Label = 'Theil\'s U')

    plot_entropy_multipanel([entropy_full,entropy_adjusted_th,entropy_adjusted_chi],
                                    ['Entropy','Theil Entropy','Cramer Entropy'],
                                    save_folder=save_folder,
                                    fig_name='entropy_multi_plot.png')
    
    plot_entropy_multipanel([mi_full,mi_adjusted_th,mi_adjuested_chi],
                                ['MI','Theil MI','Cramer MI'],
                                save_folder=save_folder,
                                fig_name='MI_multi_plot.png')
    
    plot_autocorr_multipanel([entropy_full,entropy_adjusted_th,entropy_adjusted_chi,mi_full,mi_adjusted_th,mi_adjuested_chi],
                                ['Entropy','Theil Entropy','Cramer Entropy','MI','Theil MI','Cramer MI'],
                                save_folder=save_folder,
                                fig_name='autocorr_multi_plot.png',
                                protein_family= Protein_Family)

def plot_autocorr_multipanel(listin,names,save_folder,fig_name,protein_family):
    list_out = []
    for entry in listin:
        _,list_out_new= periodic_autocorrelation(entry,None,None)
        list_out.append(list_out_new)
    if protein_family:
        cbar_label=f'{protein_family} \n Autocorrelation'
    else:
        cbar_label = protein_family
    plot_colormap_rnyttopy_mp(list_out, y_labels=names, save_folder=save_folder, fig_name=fig_name,fig_size=(11.69/1.5,11.69/1.5),xlabel='other',cbar_label=cbar_label)
    
def handle_info_theory(calculate_entropy_flag,aligned_residues_df,entropy_directory,cross_correl_data,entropies_data,
                       calculate_information_flag,information_directory,target_df,information_data,
                       ):
    if calculate_entropy_flag =="y":
        print('calculate entropies')
        entropy_full, theil_full, chi_full = plot_all(aligned_residues_df, save_folder=entropy_directory, column_name='Sequence',theil=True,chi=True)
        entropy_adjusted_th,_ = process_vectors(entropy_full,theil_full['corr'].to_numpy())
        entropy_adjusted_chi,_ = process_vectors(entropy_full,chi_full)

        with open(cross_correl_data, 'wb') as f:
            pickle.dump([theil_full['corr'].to_numpy(),chi_full], f)

        with open(entropies_data, 'wb') as f:
            pickle.dump([entropy_full,entropy_adjusted_th,entropy_adjusted_chi], f)

        plot_entropy_multipanel([entropy_full,entropy_adjusted_th,entropy_adjusted_chi],
                                           ['Entropy','Theil Adjusted Entropy','Cramer Adjusted Entropy'],
                                           save_folder=entropy_directory,
                                           fig_name='entropy_multi_plot.png')


    if calculate_information_flag =="y":
        print('calculate information')
        os.makedirs(information_directory, exist_ok=True)
        mi_full = mi_shell(aligned_residues_df,target_df,save_folder=information_directory)

        with open(cross_correl_data, 'rb') as f:
            cross_correl = pickle.load(f)

        theil_full = cross_correl[0]
        chi_full = cross_correl[1]

        mi_adjusted_th,_ = process_vectors(mi_full,theil_full)
        mi_adjuested_chi, _ = process_vectors(mi_full,chi_full)

        with open(information_data, 'wb') as f:
            pickle.dump([mi_full,mi_adjusted_th,mi_adjuested_chi], f)

        plot_entropy_multipanel([mi_full,mi_adjusted_th,mi_adjuested_chi],
                                       ['MI','Theil Adjusted MI','Cramer Adjusted MI'],
                                       save_folder=information_directory,
                                       fig_name='MI_multi_plot.png')

def periodic_autocorrelation(signal,outfile,key):


    n = len(signal)
    mean_signal = np.mean(signal)
    signal = signal/mean_signal
    
    # Compute full autocorrelation using FFT
    f_signal = np.fft.fft(signal)
    f_autocorr = f_signal * np.conj(f_signal)
    autocorr = np.fft.ifft(f_autocorr).real
    
    # Normalize the autocorrelation
    autocorr /= np.var(signal) * n

    autocorr/=autocorr[0]
    
    lags = np.arange(n)
    
    # Plot the autocorrelation function
    plt.close()
    plt.plot(lags, autocorr)
    plt.title('Periodic Autocorrelation')
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.grid(True)
    if outfile !=None:
        plt.savefig(f'{outfile}/autocorrelation_{key}',dpi=600)

    plt.close()

    return np.mean(autocorr),autocorr

def plot_bar_chart_with_error_bars(dict1, dict2, labels=['dict1', 'dict2'], ylabel='Noise Proportion',
                                   yticks=[0.01, 0.1, 1], ylim=(0.01, 1.1), yticklabels=['1E-2', '1E-1', '1'],
                                   rotation=0,figsize=(11.69/1.5, 11.69/1.5),figsize_cross_corr=(11.69/1.5, 11.69/1.5),outfolder=None):
    keys1 = list(dict1.keys())
    keys2 = list(dict2.keys())
    all_keys = keys1 + keys2
    y_label_stripped = ylabel.replace(" ","")
    
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

    all_keys_alt = [textwrap.fill(y_label, width=8) for y_label in all_keys]



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
    ax.set_yscale('log')
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontsize=20, rotation=0)


    # Show plot
    plt.tight_layout()
    plt.savefig(f'{outfolder}/bar_chart{y_label_stripped}.png',dpi=600)

    _,hedge_df = generate_comparison_table(dict1, dict2)

    plot_colormap_cross_corr(hedge_df.values,list(hedge_df.keys()),cbar_label=f'Hedge\'s G \n {ylabel}',vrange=(-1,1),
        vticks=(-1,0,1),figsize=figsize_cross_corr,save_path = f'{outfolder}/hedge_g_{y_label_stripped}.png')

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

import matplotlib.colors as mcolors

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

def plot_colormap_hedges_G(hedge_df):
    plt.close()
    plt.figure(figsize=(11.69/1.5, 11.69/1.5))
    
    # Set the limits for the colormap
    vmin, vmax = -1, 1

    # Create the heatmap
    ax = sns.heatmap(hedge_df, annot=False, cmap='vlag', center=0, linewidths=1, vmin=vmin, vmax=vmax)
    
    # Determine the midpoint to draw the lines
    mid_point = len(hedge_df) / 2
    
    # Draw vertical and horizontal black lines
    plt.axvline(x=mid_point, color='black', linewidth=4)
    plt.axhline(y=mid_point, color='black', linewidth=4)
    
    all_keys_alt = [textwrap.fill(y_label, width=10) for y_label in hedge_df.keys()]

    # Customize the colorbar
    cbar = ax.collections[0].colorbar
    cbar.set_ticks([vmin, vmin/2, 0, vmax/2, vmax])
    cbar.set_ticklabels([f'<{vmin}', f'{vmin/2}', '0', f'{vmax/2}', f'>{vmax}'])
    cbar.ax.tick_params(labelsize=14)  
    ax.set_xticklabels(all_keys_alt, fontsize=14,rotation=90)
    ax.set_yticklabels(all_keys_alt, fontsize=14,rotation=0)


    plt.tight_layout()
    plt.show()

def periodic_autocorrelation_list(data_path_in,outfile,keys):
    with open(data_path_in,'rb') as f:
        signals = pkl.load(f)

    results = []
    for signal,key in zip(signals,keys):
        result,_ = periodic_autocorrelation(signal,outfile,key)
        results.append(result)

    return results#/result[0]

from collections import OrderedDict
def remove_duplicates_ordered(data):
    for key in data:
        data[key] = list(OrderedDict.fromkeys(data[key]))
    return data

def process_list_autocorrelation(data_path_in, keys,outfiles):
    if outfiles ==None:
        outfiles = [None]*len(data_path_in)
        
    list_out = []
    for entry,outfile in zip(data_path_in,outfiles):
        list_out_new = periodic_autocorrelation_list(entry,outfile,keys)
        list_out.append(list_out_new)
    
    # Initialize a dictionary to hold the results
    result_dict = {key: [] for key in keys}
    
    # Populate the dictionary
    for output in list_out:
        for key, value in zip(keys, output):
            result_dict[key].append(value)
    
    result_dict = remove_duplicates_ordered(result_dict)

    return result_dict

def load_pkl_file(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)

def compute_cross_correlation(lists):
    n = len(lists)
    cross_corr_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            cross_corr = np.correlate(lists[i], lists[j], mode='full')
            mid = len(cross_corr) // 2
            norm_factor = np.sqrt(np.correlate(lists[i], lists[i], mode='full')[mid] *
                                  np.correlate(lists[j], lists[j], mode='full')[mid])
            if norm_factor != 0:
                cross_corr_matrix[i, j] = cross_corr[mid] / norm_factor
            else:
                cross_corr_matrix[i, j] = 0
    
    return cross_corr_matrix

def average_cross_correlation_matrix(paths1, paths2,plot_flag=False,labels=[],Protein_Families=[],Store_paths =[]):
    assert len(paths1) == len(paths2), "Lists of paths must be of the same length"
    
    num_files = len(paths1)
    sum_matrix = np.zeros((6, 6))
    
    i=0
    for path1, path2 in zip(paths1, paths2):
        lists1 = load_pkl_file(path1)
        lists2 = load_pkl_file(path2)
        
        assert len(lists1) == 3 and len(lists2) == 3, "Each .pkl file must contain a list of length 3"
        new_list = lists1.copy()
        new_list.extend(list(lists2))
        cross_corr_matrix = compute_cross_correlation(new_list)
        sum_matrix += cross_corr_matrix
        if plot_flag:
            plot_colormap_cross_corr(cross_corr_matrix,labels,cbar_label=f'{Protein_Families[i]} \n CrossCorrelation',save_path=f'{Store_paths[i]}/cross_corr_metrics.png')
  

        i+=1
    average_matrix = sum_matrix / num_files
    return average_matrix

def plot_colormap_cross_corr(matrix, labels,cbar_label=[],vrange=(0,1),vticks=[0,1],save_path=False,figsize=(11.69/1.5, 11.69/1.5)):
    plt.close()
    plt.figure(figsize=figsize)
    
    # Set the limits for the colormap
    vmin, vmax = vrange

    # Create the heatmap
    ax = sns.heatmap(matrix, annot=False, cmap='vlag',linewidths=1, vmin=vmin, vmax=vmax, xticklabels=labels, yticklabels=labels)

    
    # Customize the colorbar
    cbar = ax.collections[0].colorbar
    cbar.set_ticks(vticks)
    #cbar.set_ticklabels(['0', f'{vmax}'])
    cbar.ax.tick_params(labelsize=20)
    cbar.set_label(cbar_label, fontsize=22)
    
    # Customize axis tick labels
    labels = [textwrap.fill(y_label, width=8) for y_label in labels]
    ax.set_xticklabels(labels, fontsize=21, rotation=90)
    ax.set_yticklabels(labels, fontsize=21, rotation=0)

    # Manually draw lines between cells using annotate
    for i in range(matrix.shape[0]):
        ax.annotate('', xy=(-1, i), xytext=(matrix.shape[1], i),
                    xycoords='data', textcoords='data',
                    arrowprops=dict(color='grey', linestyle='--', linewidth=4, arrowstyle='-',alpha=0.15),
                    annotation_clip=False)

    for j in range(matrix.shape[1]):
        ax.annotate('', xy=(j + 1, 0), xytext=(j + 1, matrix.shape[0] + 1),
                    xycoords='data', textcoords='data',
                    arrowprops=dict(color='grey', linestyle='--', linewidth=4, arrowstyle='-',alpha=0.15),
                    annotation_clip=False)

    # Determine the midpoint to draw the lines
    mid_point = len(matrix) / 2
    
    # Draw vertical and horizontal black lines
    plt.axvline(x=mid_point, color='black', linewidth=4)
    plt.axhline(y=mid_point, color='black', linewidth=4)




    plt.tight_layout()

    if save_path:
        plt.savefig(save_path,dpi=600)
        plt.close()
    else:
        plt.show()

def generate_data(num_sequences=1000, sequence_length=50):
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    sequences = []
    wavelengths = []
    
    for _ in range(num_sequences):
        sequence = np.random.choice(list(amino_acids), sequence_length)
        
        # Introduce a clear correlation at positions 10 and 20
        if sequence[18] in ['F', 'Y', 'W']:  # Assume these contribute to higher wavelengths
            wavelength = np.random.uniform(550, 650)  # Higher range for wavelength
        elif sequence[19] in ['R', 'H', 'K']:  # Assume these too contribute differently
            wavelength = np.random.uniform(450, 550)  # Different range to simulate variation
        else:
            wavelength = np.random.uniform(300, 450)  # Base range for others
        
        sequences.append(''.join(sequence))
        wavelengths.append(wavelength)
    
    return pd.DataFrame({'Sequence': sequences}), pd.DataFrame({'wavelength': wavelengths})

def calculate_mi_per_position(X_data, y_data):
    """
    Calculate the mutual information between each amino acid position and the wavelength.

    Parameters:
    - X_data: DataFrame containing amino acid sequences.
    - y_data: DataFrame containing wavelengths.

    Returns:
    - mi_scores: List of mutual information scores for each amino acid position.
    """
    sequence_length = len(X_data['Sequence'][0])
    mi_scores = []
    
    # Process each position individually
    for pos in range(sequence_length):
        # # Extract the amino acid at the current position from each sequence
        pos_amino_acids = X_data['Sequence'].apply(lambda seq: seq[pos])

        entropy_continuous = calculate_entropy_continuous(y_data)
        
        # Example for discrete entropy
        entropy_discrete = calculate_entropy_discrete(pos_amino_acids)
        
        # Example for joint entropy with mixed types
        joint_entropy_mixed = calculate_joint_entropy(y_data, pos_amino_acids, types=('continuous', 'categorical'))

        mi = (entropy_discrete+entropy_continuous-joint_entropy_mixed)/entropy_continuous

        mi_scores.append(np.mean(mi))  # Averaging MI scores over all amino acids at this position
    
    return mi_scores

from scipy.stats import entropy as entropy_ss

def calculate_entropy_continuous(data, bins='auto'):
    """
    Calculate entropy of a continuous variable by discretizing it into bins.
    
    Parameters:
    data : numpy array or list
        The continuous variable.
    bins : int or sequence of scalars or str, optional
        If bins is an int, it defines the number of equal-width bins in the range of data.
        If bins is a sequence, it defines the bin edges.
        If bins is 'auto', it uses the Freedman-Diaconis rule to determine the optimal bin width.
        Default is 'auto'.
        
    Returns:
    entropy_value : float
        Entropy of the input continuous variable.
    """
    # Compute histogram with specified bins
    hist, bin_edges = np.histogram(data, bins=bins, density=True)
    
    # Compute probabilities from histogram counts
    probabilities = hist * np.diff(bin_edges)
    
    # Compute entropy
    entropy_value = entropy_ss(probabilities)
    
    return entropy_value

def calculate_entropy_discrete(vector):
    """
    Calculate entropy of a discrete vector.
    
    Parameters:
    vector : numpy array or list
        The discrete vector (categorical data).
        
    Returns:
    entropy_value : float
        Entropy of the input vector.
    """
    # Compute frequency counts of each unique value in the vector
    _, counts = np.unique(vector, return_counts=True)
    
    # Compute probabilities from counts
    probabilities = counts / len(vector)
    
    # Compute entropy
    entropy_value = entropy_ss(probabilities)
    
    return entropy_value

def calculate_joint_entropy(X, Y, types=('continuous', 'continuous'), num_bins=None):
    """
    Calculate the joint entropy H(X, Y) of variables X and Y.
    
    Parameters:
    X : numpy array or list
        Variable X (continuous or categorical).
    Y : numpy array or list
        Variable Y (continuous or categorical).
    types : tuple of str, optional
        Types of X and Y ('continuous' or 'categorical'). Default is ('continuous', 'continuous').
    num_bins : int or None, optional
        Number of bins to use for continuous variables. If None, determined using Freedman-Diaconis rule.
        
    Returns:
    joint_entropy : float
        Joint entropy H(X, Y).
    """
    assert len(X) == len(Y), "X and Y must have the same length"
    assert len(types) == 2, "types should be a tuple of length 2"
    
    if types[0] == 'continuous':
        # Determine number of bins for X
        if num_bins is None:
            _, bins_X = np.histogram(X, bins='fd')
            num_bins_X = len(bins_X)
        else:
            num_bins_X = num_bins
            
        # Compute histogram with specified bins
        hist_X, _ = np.histogram(X, bins=num_bins_X, density=True)
        prob_X = hist_X * np.diff(_)
        
    elif types[0] == 'categorical':
        # Compute probabilities directly for X
        _, counts_X = np.unique(X, return_counts=True)
        prob_X = counts_X / len(X)
        
    else:
        raise ValueError("Unsupported type for X. Should be 'continuous' or 'categorical'")
    
    if types[1] == 'continuous':
        # Determine number of bins for Y
        if num_bins is None:
            _, bins_Y = np.histogram(Y, bins='fd')
            num_bins_Y = len(bins_Y)
        else:
            num_bins_Y = num_bins
            
        # Compute histogram with specified bins
        hist_Y, _ = np.histogram(Y, bins=num_bins_Y, density=True)
        prob_Y = hist_Y * np.diff(_)
        
    elif types[1] == 'categorical':
        # Compute probabilities directly for Y
        _, counts_Y = np.unique(Y, return_counts=True)
        prob_Y = counts_Y / len(Y)
        
    else:
        raise ValueError("Unsupported type for Y. Should be 'continuous' or 'categorical'")
    
    # Compute joint probabilities
    if types == ('continuous', 'continuous'):
        joint_hist, _, _ = np.histogram2d(X, Y, bins=(num_bins_X, num_bins_Y), density=True)
        joint_prob = joint_hist.flatten() / np.sum(joint_hist)
        
    elif types == ('categorical', 'categorical'):
        joint_prob = np.zeros((len(prob_X), len(prob_Y)))
        for i in range(len(X)):
            joint_prob[np.where(X[i] == np.unique(X))[0][0], np.where(Y[i] == np.unique(Y))[0][0]] += 1
        joint_prob /= len(X)
        joint_prob = joint_prob.flatten()
        
    elif types == ('continuous', 'categorical'):
        # Convert continuous X into categorical bins
        bin_index_X = np.digitize(X, bins_X[:-1])
        
        joint_prob = np.zeros(num_bins_X * len(prob_Y))
        for i in range(len(X)):
            index = bin_index_X[i] * len(prob_Y) + np.where(Y[i] == np.unique(Y))[0][0]
            if index < len(joint_prob):
                joint_prob[index] += 1
        joint_prob /= len(X)
        
    elif types == ('categorical', 'continuous'):
        # Convert continuous Y into categorical bins
        bin_index_Y = np.digitize(Y, bins_Y[:-1])
        
        joint_prob = np.zeros(len(prob_X) * num_bins_Y)
        for i in range(len(X)):
            index = np.where(X[i] == np.unique(X))[0][0] * num_bins_Y + bin_index_Y[i]
            if index < len(joint_prob):
                joint_prob[index] += 1
        joint_prob /= len(X)
        
    else:
        raise ValueError("Unsupported combination of types. Both should be 'continuous' or 'categorical'")
    
    # Compute joint entropy
    joint_entropy = entropy_ss(joint_prob)
    
    return joint_entropy

def calculate_mi_per_position_old(X_data, y_data):
    """
    Calculate the mutual information between each amino acid position and the wavelength.

    Parameters:
    - X_data: DataFrame containing amino acid sequences.
    - y_data: DataFrame containing wavelengths.

    Returns:
    - mi_scores: List of mutual information scores for each amino acid position.
    """
    sequence_length = len(X_data['Sequence'][0])
    mi_scores = []
    
    # Process each position individually
    for pos in range(sequence_length):
        # Extract the amino acid at the current position from each sequence
        pos_amino_acids = X_data['Sequence'].apply(lambda seq: seq[pos])
        
        # One-hot encode the amino acids for this position
        encoder = OneHotEncoder(sparse_output=False)
        pos_encoded = encoder.fit_transform(pos_amino_acids.to_numpy().reshape(-1, 1))
        
        # Calculate mutual information of this position's encoding with the wavelengths
        mi = mutual_info_regression(pos_encoded, y_data.squeeze())
        mi_scores.append(np.mean(mi))  # Averaging MI scores over all amino acids at this position
    
    return mi_scores


def plot_mi_per_position(mi_scores, save_folder=None):
    """
    Plot the mutual information of each amino acid position with the wavelength in a specified style.

    Parameters:
    - mi_scores: List of mutual information scores for each amino acid position.
    - save_folder: Optional; directory to save the plot. If None, the plot is shown directly.
    """
    # Normalize mutual information scores between 0 and 1 based on the mean mutual information
    mean_mi = np.mean(mi_scores)
    normalized_mi = [(mi ) / (mean_mi) for mi in mi_scores]

    plt.figure(figsize=(6, 4))  # Adjust the figure size
    plt.plot(normalized_mi, color='black', linewidth=1.5)  # Specify line color and thickness
    plt.fill_between(range(len(normalized_mi)), normalized_mi, color='lightgrey', alpha=0.5)  # Shade area underneath the plot
    plt.title('Normalized Mutual Information of Amino Acid Positions', fontsize=12, fontweight='bold')  # Title with increased font size and bold weight
    plt.xlabel('Amino Acid Position', fontsize=10)  # X-axis label with increased font size
    plt.ylabel('Normalized Mutual Information', fontsize=10)  # Y-axis label with increased font size
    plt.xticks(fontsize=8)  # Increase font size of x-axis tick labels
    plt.yticks(fontsize=8)  # Increase font size of y-axis tick labels
    plt.grid(True, linestyle='--', alpha=0.5)  # Add grid with dashed lines and reduced opacity
    plt.tight_layout()  # Adjust layout to prevent clipping of labels
    
    if save_folder is not None:
        plt.savefig(os.path.join(save_folder, 'mutual_information_plot.png'))
    else:
        plt.show()

def mi_shell(X_data,y_data,save_folder=None):
    mi_scores = calculate_mi_per_position(X_data, y_data)
    plot_mi_per_position(mi_scores, save_folder)
    return mi_scores
    
def calculate_entropy(df,column_name='amino_acid_sequence'):
    # Initialize an empty list to store entropy values for each position
    entropy = []

    # Iterate over the length of sequences
    sequence_length = len(df[column_name][0])  # Assuming all sequences have the same length
    for i in range(sequence_length):
        # Extract amino acids at position i for all sequences
        column = df[column_name].str[i]

        # Calculate the frequency of each amino acid at position i
        counts = column.value_counts(normalize=True)

        # Calculate entropy for position i
        position_entropy = -(counts * np.log2(counts)).sum()

        # Append entropy value for position i to the list
        entropy.append(position_entropy)

    return entropy

def plot_entropy(entropy,save_folder=None):
    # Normalize entropy between 0 and 1
    mean_entropy = np.mean(entropy)
    normalized_entropy = [(e ) / (mean_entropy) for e in entropy]

    plt.figure(figsize=(6, 4))  # Adjust the figure size
    plt.plot(normalized_entropy, color='black', linewidth=1.5)  # Specify line color and thickness
    plt.fill_between(range(len(normalized_entropy)), normalized_entropy, color='lightgrey', alpha=0.5)  # Shade area underneath the plot
    plt.title('Normalized Entropy of Amino Acid Positions', fontsize=12, fontweight='bold')  # Title with increased font size and bold weight
    plt.xlabel('Amino Acid Position', fontsize=10)  # X-axis label with increased font size
    plt.ylabel('Normalized Entropy', fontsize=10)  # Y-axis label with increased font size
    plt.xticks(fontsize=8)  # Increase font size of x-axis tick labels
    plt.yticks(fontsize=8)  # Increase font size of y-axis tick labels
    plt.grid(True, linestyle='--', alpha=0.5)  # Add grid with dashed lines and reduced opacity
    plt.tight_layout()  # Adjust layout to prevent clipping of labels
    
    if save_folder != None:
        plt.savefig(os.path.join(save_folder, 'entropy_plot.png'),dpi=600)
    else:
        plt.show()

def create_dataframe_dict(df):
    # Calculate entropies
    entropy = calculate_entropy(df)
    
    # Determine the number of residues
    num_residues = len(df['Sequence'][0])

    # Create a dictionary to store DataFrames
    df_dict = {}

    # Iterate over the number of residues
    for i in range(num_residues + 1):
        # Select top i residues with the highest entropy
        top_residues_indices = np.argsort(entropy)[-i:]
        
        # Create a reduced DataFrame with only the selected residues
        reduced_df = df.copy()
        reduced_df['Sequence'] = reduced_df['Sequence'].apply(lambda seq: ''.join([seq[j] for j in top_residues_indices]))

        # Store the reduced DataFrame in the dictionary
        df_dict[i] = reduced_df

    return df_dict

# Function to calculate conditional entropy
def calculate_conditional_entropy(x, y):
    # Create DataFrame to count occurrences of each pair of values
    df = pd.DataFrame({'x': x, 'y': y})
    xy_counts = df.groupby(['x', 'y']).size().reset_index(name='count')
    
    total_occurrences = xy_counts['count'].sum()
    
    # Calculate conditional entropy
    entropy = 0
    for index, row in xy_counts.iterrows():
        p_xy = row['count'] / total_occurrences
        p_x = df['x'].value_counts(normalize=True)[row['x']]
        entropy -= p_xy * np.log2(p_xy / p_x)
    
    return entropy

# Function to calculate Theil's U
def calculate_theils_u(x, y):
    s_xy = calculate_conditional_entropy(x, y)
    x_counter = Counter(x)
    total_occurrences = sum(x_counter.values())
    p_x = list(map(lambda n: n/total_occurrences, x_counter.values()))
    s_x = ss.entropy(p_x)
    if s_x == 0:
        return 1
    else:
        return (s_x - s_xy) / s_x

# Function to calculate Theil's U matrix
def calculate_theils_u_matrix(df,column_name='amino_acid_sequence'):
    df = pd.DataFrame(df[column_name].apply(list).tolist(), columns=[f'Position{i+1}' for i in range(len(df[column_name][0]))])
    num_columns = len(df.columns)                                                                            
    theils_u_matrix = np.zeros((num_columns, num_columns))
    for i, column1 in enumerate(df.columns):
        for j, column2 in enumerate(df.columns):
            theils_u_matrix[i, j] = calculate_theils_u(df[column1], df[column2])
    return theils_u_matrix

# Function to normalize Theil's U matrix
def normalize_theils_u(matrix):
    # Normalize matrix values to be between 0 and 1
    min_val = np.min((np.min(matrix),0))
    max_val = np.max((np.max(matrix),1))
    normalized_matrix = (matrix - min_val) / (max_val - min_val)
    return normalized_matrix

def cramers_v(data):
    num_cols = len(data.columns)
    cramer_matrix = np.zeros((num_cols, num_cols))  # Using a numpy array for efficiency

    for i in range(num_cols):
        for j in range(i + 1, num_cols):
            # Select columns and check if categorical
            col1, col2 = data.columns[i], data.columns[j]            
            confusion_matrix = pd.crosstab(data[col1], data[col2])
            chi2 = ss.chi2_contingency(confusion_matrix)[0]
            n = confusion_matrix.sum().sum()
            phi2 = chi2 / n
            r, k = confusion_matrix.shape
            phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
            rcorr = r - ((r - 1) ** 2) / (n - 1)
            kcorr = k - ((k - 1) ** 2) / (n - 1)
            cramers_v_val = np.sqrt(phi2corr / max(1E-12,min((kcorr - 1), (rcorr - 1))))
            if min((kcorr,rcorr))==1:
                cramers_v_val=0
            cramer_matrix[i, j] = cramers_v_val
            cramer_matrix[j, i] = cramers_v_val

    # Set diagonal values to 1
    np.fill_diagonal(cramer_matrix, 1)

    # Convert to DataFrame for better readability
    #cramer_df = pd.DataFrame(cramer_matrix, index=data.columns, columns=data.columns)
    return cramer_matrix

# Function to plot heatmap for Theil's U matrix
def plot_heatmap(matrix, x_label='Predictor Location', y_label='Target Location', name='theils_u_heatmap.png',Label = '', save_folder=None):
    # Plot heatmap
    plt.figure(figsize=(11.69/2, 11.69/2))
    ax = sns.heatmap(matrix, annot=False, cmap="rocket_r")
    plt.xlabel(x_label, fontsize=24)
    plt.ylabel(y_label, fontsize=24)
    ax.collections[0].colorbar.ax.tick_params(labelsize=20)
    ax.collections[0].colorbar.set_ticks([0,1])
    ax.collections[0].colorbar.set_label(Label, fontsize=24)

    # Remove x and y tick labels apart from the first and last
    plt.xticks(ticks=[])
    plt.yticks(ticks=[])
    plt.xticks(ticks=[0, matrix.shape[1] - 1], labels=[1, matrix.shape[1]], fontsize=20)
    plt.yticks(ticks=[0, matrix.shape[0] - 1], labels=[matrix.shape[0], 1], fontsize=20)
    plt.tight_layout()

    if save_folder is not None:
        plt.savefig(os.path.join(save_folder, name), dpi=600)
    else:
        plt.show()

    # Plot sum of matrix at a given x location
    plt.figure(figsize=(11.69/2, 11.69/2))
    x_sums = matrix.sum(axis=0)
    plt.plot(range(matrix.shape[1]), x_sums, color='black', linewidth=0.5)
    
    # Adding grid for better readability
    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    
    plt.xlim(0, matrix.shape[1] - 1)
    plt.ylim(0, max(x_sums))  # Adjust the y-axis to fit the data range

    # Enhancing the plot with labels and a title
    plt.xlabel(x_label, fontsize=24)
    plt.ylabel('Coupling', fontsize=24)
    plt.xticks(ticks=[0, matrix.shape[1] - 1], labels=[0, matrix.shape[1] - 1], fontsize=20)
    plt.yticks(ticks=[])
    
    # Tight layout for a clean look
    plt.tight_layout()

    if save_folder is not None:
        sum_plot_name = 'sum_' + name
        plt.savefig(os.path.join(save_folder, sum_plot_name), dpi=600)
    else:
        plt.show()

# Function to plot residue probability heatmap
def plot_residue_probability_heatmap(df, column_name='amino_acid_sequence', save_folder=None):
    """
    Plots a vertically stacked four-panel plot:
    1. Residue probability heatmap
    2. Empty panel for alignment
    3. Conservation heatmap
    4. Empty panel for alignment and colorbars
    
    Parameters:
    df : pandas DataFrame
        DataFrame containing the amino acid sequences.
    column_name : str
        The column name in the DataFrame that contains the amino acid sequences.
    save_folder : str, optional
        Folder to save the plot. If None, the plot is displayed.
    """
    # Convert each amino acid sequence into a list and create a DataFrame where each position is a separate column
    df = pd.DataFrame(df[column_name].apply(list).tolist(), columns=[f'Position{i+1}' for i in range(len(df[column_name].iloc[0]))])

    # Determine all unique amino acids present in the input sequences
    all_amino_acids = set()
    for seq in df.values.flatten():
        all_amino_acids.update(seq)

    # Sort the amino acids alphabetically
    all_amino_acids = sorted(all_amino_acids)
    num_amino_acids = len(all_amino_acids)

    # Calculate the proportion of each residue type at each location
    residue_counts = {}
    for column in df.columns:
        counts = df[column].value_counts(normalize=True)
        counts = counts.reindex(all_amino_acids, fill_value=0)
        residue_counts[column] = counts

    residue_counts = pd.DataFrame(residue_counts)

    # Calculate the proportion of gaps and the conservation score
    max_entropy = np.log(num_amino_acids)
    conservation_scores = residue_counts.apply(lambda x: entropy_ss(x) / max_entropy, axis=0)

    # Create figure and subplots with appropriate gridspec, specifying colorbar axes
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(11.69/2, 11.69*1.15), 
                             gridspec_kw={'width_ratios': [1, 0.075], 'height_ratios': [1, 1/4]})

    # Plot residue probability heatmap in the first subplot
    sns.heatmap(residue_counts, cmap="rocket_r", ax=axes[0, 0], cbar_ax=axes[0, 1], cbar_kws={'label': 'Observed Probability', 'ticks': [0, 1]},vmin=0,vmax=1)
    axes[0, 0].set_ylabel('Amino Acid Type', fontsize=24)
    axes[0, 0].set_xticks(ticks=[0, len(residue_counts.columns) - 1], labels=[])
    axes[0, 0].tick_params(axis='x', labelsize=20)
    axes[0, 0].tick_params(axis='y', labelsize=20)

    # Plot conservation scores as a heatmap in the second subplot
    sns.heatmap(1 - np.array([conservation_scores]), cmap='rocket_r', ax=axes[1, 0], cbar_ax=axes[1, 1],vmin=0,vmax=1)
    axes[1, 0].set_ylabel('Conservation', fontsize=24)
    axes[1, 0].set_yticks([])
    axes[1, 0].set_xticks([0, 1])
    axes[1, 0].set_xticks(ticks=[1, len(conservation_scores)], labels=[1, len(residue_counts.columns)], rotation=0)
    axes[1, 0].set_xlabel('Position', fontsize=24)
    axes[1, 0].tick_params(labelsize=20)

       # Customize the colorbar for residue probability heatmap
    cbar1 = axes[0 ,0].collections[0].colorbar
    cbar1.set_ticks([0, 1])
    cbar1.set_ticklabels(['0', '1'])
    cbar1.ax.tick_params(labelsize=20)
    cbar1.set_label('Observed Probability', fontsize=24)

    # Customize the colorbar for conservation heatmap
    cbar2 = axes[1, 0].collections[0].colorbar
    cbar2.set_ticks([0, 1])
    cbar2.set_ticklabels(['0', '1'])
    cbar2.ax.tick_params(labelsize=20)
    cbar2.set_label('  ', fontsize=24)

    # Adjust layout to make room for wider colorbars
    plt.tight_layout()

    # Display or save the plot
    if save_folder is None:
        plt.show()
    else:
        plt.savefig(os.path.join(save_folder, 'residue_probability_heatmap.png'), dpi=600)

    plt.close()


# def plot_residue_probability_heatmap(df, column_name='amino_acid_sequence', save_folder=None):
#     # Convert each amino acid sequence into a list and create a DataFrame where each position is a separate column
#     df = pd.DataFrame(df[column_name].apply(list).tolist(), columns=[f'Position{i+1}' for i in range(len(df[column_name].iloc[0]))])

#     # Determine all unique amino acids present in the input sequences
#     all_amino_acids = set()
#     for seq in df.values.flatten():
#         all_amino_acids.update(seq)

#     # Sort the amino acids alphabetically
#     all_amino_acids = sorted(all_amino_acids)


#     # Calculate the proportion of each residue type at each location
#     # Initialize a dictionary to hold the value counts for each position
#     residue_counts = {}
#     for column in df.columns:
#         # Count each amino acid's occurrence in the column and normalize to get frequency
#         counts = df[column].value_counts(normalize=True)
#         # Reindex with all amino acids to include those not present, filling with 0
#         counts = counts.reindex(all_amino_acids, fill_value=0)
#         residue_counts[column] = counts

#     # Convert the dictionary back to DataFrame
#     residue_counts = pd.DataFrame(residue_counts)

#     # Plot heatmap
#     plt.close()
#     plt.figure(figsize=(11.69/2, 11.69/2))
#     ax = sns.heatmap(residue_counts, cmap="rocket_r")

#     # Set title and labels with appropriate font sizes
#     plt.xlabel('Residue Location', fontsize=24)
#     plt.ylabel('Amino Acid Type', fontsize=24)

#     # Adjust the x-ticks to show the first and last positions
#     plt.xticks(ticks=[0, len(residue_counts.columns) - 1], labels=[1, len(residue_counts.columns)], fontsize=14)
#     plt.yticks(fontsize=14)

#     # Customize the colorbar
#     cbar = ax.collections[0].colorbar
#     cbar.set_ticks([0, 1])
#     cbar.set_ticklabels(['0', '1'])
#     cbar.ax.tick_params(labelsize=14)

#     plt.tight_layout()

#     # Display or save the plot
#     if save_folder is None:
#         plt.show()
#     else:
#         plt.savefig(os.path.join(save_folder, 'residue_probability_heatmap.png'),dpi=600)

#     plt.close()

# def plot_conservation_and_gap_proportion(df, column_name='amino_acid_sequence', save_folder=None):
#     # Convert each amino acid sequence into a list and create a DataFrame where each position is a separate column
#     from scipy.stats import entropy
    
#     df = pd.DataFrame(df[column_name].apply(list).tolist(), columns=[f'Position{i+1}' for i in range(len(df[column_name].iloc[0]))])

#     # Determine all unique amino acids present in the input sequences
#     all_amino_acids = set()
#     for seq in df.values.flatten():
#         all_amino_acids.update(seq)

#     # Sort the amino acids alphabetically
#     all_amino_acids = sorted(all_amino_acids)

#     # Calculate the proportion of each residue type at each location
#     residue_counts = {}
#     for column in df.columns:
#         counts = df[column].value_counts(normalize=True)
#         counts = counts.reindex(all_amino_acids, fill_value=0)
#         residue_counts[column] = counts

#     residue_counts = pd.DataFrame(residue_counts)

#     # Calculate the proportion of gaps and the conservation score
#     gap_proportions = df.apply(lambda x: x.value_counts(normalize=True).get('-', 0), axis=0)
#     conservation_scores = residue_counts.apply(lambda x: entropy(x), axis=0)

#     fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(11.69/2, 11.69/2), gridspec_kw={'height_ratios': [1, 1/3]})

#     # Plot gap proportions as a bar chart
#     axes[0].bar(range(1, len(gap_proportions) + 1), gap_proportions, color='gray')
#     axes[0].set_xlim([0, len(gap_proportions) + 1])
#     axes[0].set_ylabel('Gap Proportion', fontsize=20)
#     axes[0].set_xticks([])
#     axes[0].set_yticks([])
#     axes[0].tick_params(axis='y',labelsize=14)

#     # Plot conservation scores as a heatmap
#     sns.heatmap(np.array([conservation_scores]), ax=axes[1], cmap='rocket_r', cbar=False)
#     axes[1].set_ylabel('Conservation', fontsize=20)
#     axes[1].set_yticks([])
#     axes[1].set_xticks([0,1])
#     axes[1].set_xticks(ticks=[1,len(conservation_scores)],labels=['Start','End'])
#     axes[1].tick_params(labelsize=14)
#     axes[1].set_xlabel('Location', fontsize=20)

#     plt.tight_layout()

#     # Display or save the plot
#     if save_folder is None:
#         plt.show()
#     else:
#         plt.savefig(os.path.join(save_folder, 'conservation_and_gap_proportion.png'), dpi=600)

#     plt.close()

def plot_normalized_cumsum(data,save_folder = None):
    """
    Plots the normalized cumulative sum of a given list in a style mimicking figures from Nature papers.
    Includes a dashed horizontal line at y=0.9 and a dashed vertical line from the curve intercept with y=0.9 to y=0.
    
    Parameters:
    - data: List or array-like, the data to be plotted.
    """
    # Sort the list
    sorted_data = data.copy()
    sorted_data.sort(reverse=True)

    # Calculate cumulative sum and normalize
    cumulative_sum = np.cumsum(sorted_data)
    total_sum = cumulative_sum[-1]
    normalized_cumulative_sum = cumulative_sum / total_sum

    # Find the index where the cumulative sum crosses 0.9
    index_90_percent = np.argmax(normalized_cumulative_sum >= 0.9)

    # Plot setup for a professional look
    plt.figure(figsize=(8, 6))
    
    # Plotting the cumulative sum
    plt.plot(np.arange(len(normalized_cumulative_sum)), normalized_cumulative_sum, 
             color='black', linewidth=4)
    
    # Horizontal line
    plt.plot([0, index_90_percent],[0.9,0.9], color='black', linestyle='--', linewidth=2)
    
    # Vertical line
    plt.plot([index_90_percent, index_90_percent], [0, 0.9], 
             color='black', linestyle='--', linewidth=2)
    
    # Adding grid for better readability
    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    
    plt.xlim(0, len(sorted_data)-1)  # Assuming you want the x-axis to show the index of the sorted data
    plt.ylim(0, 1)  # Since it's a normalized sum, the y-axis should go from 0 to 1

    # Enhancing the plot with labels and a title
    plt.xlabel('Number Of Residues', fontsize=14)
    plt.ylabel('Normalized Cumulative Sum', fontsize=14)

    
    # Tight layout for a clean look
    plt.tight_layout()

    if save_folder==None:
        plt.show()
    else:
        plt.savefig(os.path.join(save_folder, 'residue_entropy_cumsum.png'),dpi=600)

def process_vectors(e, X):
    # Create copies of e and initialize k
    e_hat = np.array(e, copy=True)
    k = np.ones_like(e_hat)
    
    # Initialize the list of unprocessed indices
    unprocessed_indices = list(range(len(e_hat)))
    
    # Process until all indices have been processed
    while unprocessed_indices:
        # Find the index corresponding to the maximum value in e_hat from unprocessed indices
        current_index = unprocessed_indices[np.argmax(e_hat[unprocessed_indices])]
        current_value = e_hat[current_index]
        
        # Get the corresponding column in X
        x_i = X[:, current_index]
        
        # Temporarily store the indices to remove from unprocessed list
        to_remove = []
        
        # Update e_hat and k for all unprocessed values that are less than the current maximum
        for j in unprocessed_indices:
            if e_hat[j] < current_value:
                e_hat[j] *= (1 - x_i[j] * k[j])
                k[j] *= (1 - x_i[j] * k[j])
            if j == current_index:
                to_remove.append(j)
        
        # Update the list of unprocessed indices by removing processed index
        for idx in to_remove:
            unprocessed_indices.remove(idx)

    return e_hat, k


def plot_entropy_multipanel(entropy_lists, y_labels=[], save_folder=None,fig_name='entropy_multi_plot.png'):
    num_plots = len(entropy_lists)
    
    # Determine number of rows and columns for subplot arrangement
    # square = np.sqrt(num_plots)
    # if square%1==0:
    #     num_cols = int(square)
    #     num_rows = int(square)
    # elif square<5:
    #     num_cols = int(np.ceil(square))
    #     num_rows = int(np.ceil(num_plots/square))
    # else:
    #     num_rows = num_plots // 5 + (num_plots % 5 != 0)
    #     num_cols = min(num_plots, 5)
    num_rows = num_plots
    num_cols = 1

    # Setup subplots
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(11.69, 1.075*11.69/2))
    axs = np.array(axs).reshape(-1)  # Flatten the array of axes to simplify indexing

    tick_indices = [0, len(entropy_lists[0]) - 1]

    # Normalize entropy between 0 and 1 and plot each entropy set
    for i in range(num_plots):
        working_ent = entropy_lists[i]
        mean_entropy = np.mean(working_ent)
        min_entropy = min(working_ent)
        max_ent = np.max(working_ent)
        normalized_entropy = [(e - min_entropy) / (max_ent - min_entropy) for e in working_ent]

        # Plot entropy
        axs[i].plot(normalized_entropy, color='r', linewidth=1)
        axs[i].fill_between(range(len(normalized_entropy)), normalized_entropy, color='r', alpha=0.5)
        
        # Set y-label with forced line break
        if y_labels:
            label = textwrap.fill(y_labels[i], width=10)  # Adjust width as needed
            axs[i].set_ylabel(label, fontsize=24)
        else:
            axs[i].set_ylabel('Normalized Entropy', fontsize=24)
        
 

        # Enable grid lines for both major and minor ticks

        axs[i].minorticks_on()
        axs[i].yaxis.set_minor_locator(AutoMinorLocator(21))
        axs[i].xaxis.set_minor_locator(AutoMinorLocator(21))
        axs[i].grid(True, which='both', linestyle='--', alpha=0.5)

        # Hide x-ticks and x-labels for all but the bottom plot
        if i < num_plots - 1:
            axs[i].set_xticks(tick_indices)
            axs[i].set_xticklabels([])
            axs[i].set_yticks([0,1])
            axs[i].set_yticklabels(labels=[], fontsize=20)
        else:
            axs[i].set_xticks(tick_indices)
            axs[i].set_yticks([0,1])
            axs[i].set_xticklabels(labels=['1',len(entropy_lists[i])], fontsize=20)
            axs[i].set_yticklabels(labels=[], fontsize=20)
            axs[i].set_xlabel('Residue Location', fontsize=24)


    # Turn off unused subplots
    for j in range(i + 1, len(axs)):
        axs[j].axis('off')

    # Adjust layout
    plt.tight_layout()

    # Save or show the figure
    if save_folder is not None:
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        plt.savefig(os.path.join(save_folder,fig_name),dpi=1200)
    else:
        plt.show()

    plot_colormap_rnyttopy_mp(entropy_lists, y_labels, save_folder, 'colormap_'+fig_name)

def plot_colormap_rnyttopy_mp(entropy_lists, y_labels, save_folder=None, fig_name='entropy_colormap.png',fig_size=(11.69/2, 11.69/2),xlabel=None,cbar_label=False):
    # Convert entropy lists to a 2D array for the heatmap
    entropy_array = np.array(entropy_lists)
    
    # Plot heatmap
    plt.figure(figsize=fig_size)
    ax = sns.heatmap(entropy_array, cmap="rocket_r", cbar=True,vmin=0,vmax=1)
    
    y_labels_alt = [textwrap.fill(y_label, width=8) for y_label in y_labels]

    # Set title and labels with appropriate font sizes
    plt.xlabel('Position', fontsize=20)
    plt.xticks(ticks=[0, entropy_array.shape[1] - 1], labels=['Start', 'End'], fontsize=20,rotation=0)
    plt.yticks(ticks=np.arange(len(y_labels)) + 0.5, labels=y_labels_alt, fontsize=24,rotation=0)
    ax.set_ylabel('')  # Remove the ylabel
    if xlabel!=None:
        ax.set_xlabel('Phase Shift/'+u'\u00b0',fontsize=24)
        plt.xticks(ticks=[0, entropy_array.shape[1] - 1], labels=['0', '360'], fontsize=24,rotation=0)

    if cbar_label:
        cbar1 = ax.collections[0].colorbar
        cbar1.set_ticks([0, 1])
        cbar1.set_ticklabels(['0', '1'])
        cbar1.ax.tick_params(labelsize=20)
        cbar1.set_label(cbar_label, fontsize=24)


    plt.tight_layout()

    # Save or show the plot
    if save_folder is not None:
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        plt.savefig(os.path.join(save_folder, fig_name), dpi=1200)
    else:
        plt.show()

def plot_all(df,save_folder,column_name='amino_acid_sequence',theil=False,chi=False):

    # Calculate entropy
    entropy = calculate_entropy(df,column_name=column_name)

    if save_folder!=None:
            # Create save folder if it does not exist
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        
    plot_residue_probability_heatmap(df,column_name=column_name,save_folder=save_folder)
    
                # Plot entropy
    plot_entropy(entropy,save_folder)

    plot_normalized_cumsum(entropy,save_folder = save_folder)

    df_ex = pd.DataFrame(df[column_name].apply(list).tolist(), columns=[f'Position{i+1}' for i in range(len(df[column_name][0]))])
    df_ex = df_ex.apply(lambda x: x.astype("object") if x.dtype == "category" else x)

    if chi:
        cramers = cramers_v(df_ex)
        plot_heatmap(cramers[::-1,:],x_label='Residue',y_label='Residue',name='Cramers.png',save_folder=save_folder)

    if theil:
        theils_u = associations(df_ex,
                nom_nom_assoc = 'theil',
                compute_only =True,
                nan_strategy='drop_samples')

        # Calculate Theil's U matrix
        #theils_u_matrix = calculate_theils_u_matrix(df,column_name=column_name)
        #normalized_theils_u_matrix = normalize_theils_u(theils_u_matrix)

        # Plot Theil's U heatmap
        plot_heatmap(theils_u['corr'].to_numpy()[::-1,:],save_folder=save_folder)

    if chi and theil:
        return entropy, theils_u, cramers
    elif chi:
        return entropy, cramers
    elif theil:
        return entropy, theils_u
    else:
        return entropy


def reduce_data(Y_data, Column_name, entropy_data, num_residues):
    """
    Reduce the size of Y_data based on highest entropy residues.

    Args:
        Y_data (pd.DataFrame): Input DataFrame.
        Column_name (str): Name of the column in Y_data containing strings.
        entropy (list): List of entropies.
        num_residues (int): Number of residues to keep.

    Returns:
        pd.DataFrame: Transformed DataFrame with reduced string lengths.
    """
    # Create a copy of the DataFrame to store the transformed data
    transformed_data = Y_data[[Column_name]].copy()
    top_indices = [i for i, _ in sorted(enumerate(entropy_data), key=lambda x: x[1], reverse=True)[:num_residues]]

    # Iterate over each row and reduce the string length based on entropy
    for i, row in Y_data.iterrows():
        sequence = row[Column_name]
        reduced_sequence = ''.join(sequence[i] for i in top_indices)
        transformed_data.at[i, Column_name] = reduced_sequence

    return transformed_data


# Example usage
# Assuming df is your Pandas DataFrame with aligned amino acid sequences
# 'Sequence' column contains aligned amino acid sequences as strings

# Generate protein sequences with cross-correlation
def generate_sequences_with_correlation(num_sequences, sequence_length, conservation_rate):
    sequences = []
    for _ in range(num_sequences):
        sequence = ''
        for _ in range(sequence_length):
            # Determine if the next amino acid should be the same as the previous one
            if np.random.rand() < conservation_rate and sequence:
                sequence += sequence[-1]
            else:
                sequence += np.random.choice(['A', 'C', 'D', 'E','F'])
        sequences.append(sequence)
    return sequences

