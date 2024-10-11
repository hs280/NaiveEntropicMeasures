import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import os
import pickle
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import sys
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D
import shutil
import ray

# Import custom modules
script_path = os.path.abspath(__file__)
current_path = os.path.dirname(script_path)
directory_folder = str(Path(current_path).parents[1])
sys.path.insert(0, os.path.join(directory_folder, 'Bin'))

import Bin.variability as assess_var
import Bin.encode_sequences as encoders
import Bin.dimension_reduction as DimReduction
import Bin.AllRegressors as AR
import heapq
import concurrent.futures
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import os
from scipy.interpolate import PchipInterpolator
import matplotlib.pyplot as plt
import heapq
import concurrent.futures
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.spatial import ConvexHull
from alphashape import alphashape

def eval_metric(r2_data):
    return np.sum(1-r2_data)#np.diff(r2_data)/np.flip(np.arange(1,len(r2_data),1)))

def create_monotonic_function(values,lengths):
    # Ensure inputs are numpy arrays
    lengths = np.array(lengths)
    values = np.array(values)

    # Adjust values to ensure they are monotonically increasing
    for i in range(1, len(values)):
        if values[i] < values[i - 1]:
            values[i] = values[i - 1]
    
    # Create the PCHIP interpolator
    pchip = PchipInterpolator(lengths, values)
    
    # Create a range of values with spacing of 1
    min_length = np.min(lengths)
    max_length = np.max(lengths)
    interp_lengths = np.arange(min_length, max_length + 1)
    
    # Evaluate the interpolator at these points
    interp_values = pchip(interp_lengths)
    
    return interp_lengths, interp_values

def plot_surfaces(directory_path, num_cycles, names, save_path):
    sns.set(style="whitegrid")
    num_plots = len(names)
    fig_width = 5 * num_plots
    fig, axs = plt.subplots(1, num_plots, figsize=(fig_width, 5), sharey=True)

    all_r2_global = []

    for name in names:
        for cycle in range(num_cycles):
            file_path = os.path.join(directory_path, f'{cycle}', name, 'analysis_results.csv')
            if os.path.exists(file_path):
                data = np.genfromtxt(file_path, delimiter=',', skip_header=1)
                all_r2_global.extend(data[:, 2])

    vmin = 0
    vmax = 1

    sum_1_minus_r2 = {name: [] for name in names}

    for i, name in enumerate(names):
        all_lengths = []
        all_cycles = []
        all_r2 = []

        for cycle in range(num_cycles):
            file_path = os.path.join(directory_path, f'{cycle}', name, 'analysis_results.csv')
            if os.path.exists(file_path):
                data = np.genfromtxt(file_path, delimiter=',', skip_header=1)
                lengths = data[:, 0]
                r2 = data[:, 2]
                cycles = np.full_like(lengths, cycle)

                all_lengths.extend(lengths)
                all_cycles.extend(cycles)
                all_r2.extend(r2)

                sum_1_minus_r2[name].append(np.sum(np.diff(r2)/np.flip(np.arange(1,len(r2),1))))#np.sum(np.max(r2) - r2))

        all_lengths = np.array(all_lengths)
        all_cycles = np.array(all_cycles)
        all_r2 = np.array(all_r2)

        unique_lengths = np.unique(all_lengths)
        unique_cycles = np.unique(all_cycles)

        heatmap_data = np.zeros((len(unique_cycles), len(unique_lengths)))

        for length, cycle, r2 in zip(all_lengths, all_cycles, all_r2):
            x_idx = np.where(unique_lengths == length)[0][0]
            y_idx = np.where(unique_cycles == cycle)[0][0]
            heatmap_data[y_idx, x_idx] = r2

        sns.heatmap(
            heatmap_data,
            ax=axs[i],
            cmap='viridis',
            cbar=False,
            vmin=vmin,
            vmax=vmax,
            xticklabels=[f'{unique_lengths[0]:.1f}', f'{unique_lengths[-1]:.1f}'],
            yticklabels=unique_cycles
        )
        axs[i].set_title(name, fontsize=14, weight='bold')
        axs[i].set_xlabel('Length (log scale)', fontsize=12)
        axs[i].set_ylabel('Cycle Number', fontsize=12)
        axs[i].tick_params(axis='both', which='major', labelsize=10)
        axs[i].grid(visible=True, color='gray', linestyle='--', linewidth=0.5)
        axs[i].set_xticks([0, len(unique_lengths) - 1])
        axs[i].set_xticklabels([f'{unique_lengths[0]:.1f}', f'{unique_lengths[-1]:.1f}'])
        axs[i].set_xscale('log')
        axs[i].set_xlim(1, unique_lengths[-1])

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(axs[0].collections[0], cax=cbar_ax)
    cbar.set_label('R2', fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    plt.tight_layout(rect=[0, 0, 0.9, 1])

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(os.path.join(save_path, 'heatmap_plot.png'), dpi=600, bbox_inches='tight')

    # Plot R2 vs Length for each cycle
    for name in names:
        plt.figure(figsize=(10, 6))
        for cycle in range(num_cycles):
            file_path = os.path.join(directory_path, f'{cycle}', name, 'analysis_results.csv')
            if os.path.exists(file_path):
                data = np.genfromtxt(file_path, delimiter=',', skip_header=1)
                lengths = data[:, 0]
                r2 = data[:, 2]
                plt.plot(lengths, r2, label=f'Cycle {cycle}')
        plt.title(f'R2 vs. Length for {name}', fontsize=14, weight='bold')
        plt.xlabel('Length', fontsize=12)
        plt.ylabel('R2', fontsize=12)
        plt.legend()
        plt.grid(visible=True, color='gray', linestyle='--', linewidth=0.5)
        plt.xscale('log')
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f'R2_vs_Length_{name}.png'), dpi=600)


    # Plot sum(1-R2) vs Cycle number for each name
    for name in names:
        plt.figure(figsize=(10, 6))
        cycles = np.arange(num_cycles)
        sum_1_minus_r2_values = sum_1_minus_r2[name]
        plt.plot(cycles, sum_1_minus_r2_values, marker='o')
        plt.title(f'Sum of (1-R2) vs. Cycle Number for {name}', fontsize=14, weight='bold')
        plt.xlabel('Cycle Number', fontsize=12)
        plt.ylabel('Sum of (1-R2)', fontsize=12)
        plt.grid(visible=True, color='gray', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f'Sum_1_minus_R2_vs_Cycle_{name}.png'), dpi=600)

def r2_rmse_against_mean(data):
    # Calculate the mean of the data
    mean_value = np.nanmean(data)
    
    # Calculate the total sum of squares (SS_tot)
    y_filled = mean_value*np.ones(data.shape)
    
    # Calculate the residual sum of squares (SS_res)
    r2 = r2_score(data.values,y_filled)
    rmse = np.sqrt(mean_squared_error(data,y_filled))
    
    return r2, rmse

def process_data(X_data, encoder_name, reducer_name, reducer, dimension, entropy, seq_length):
    reduced_x_data = assess_var.reduce_data(X_data, 'Sequence', entropy, seq_length)
    encoded_residues = encoders.apply_encoder(reduced_x_data, 'Sequence', encoder_name)
    if encoded_residues.shape[1] <= dimension:
        return encoded_residues, None
    if reducer is None:
        reduced_encoded_residues, reducer, _ = DimReduction.reduce_dimensions_with_method(encoded_residues.values, reducer_name, dimension)
    else:
        reduced_encoded_residues = DimReduction.apply_reducer_to_data(reducer, encoded_residues.values, reducer_name)
    return pd.DataFrame(reduced_encoded_residues), reducer


@ray.remote
def test_single_length_remote(X_data,y_data,entropy,seq_length,split_fraction,num_runs):
    encoder_name = 'one_hot'
    reducer_name = 'pca'
    model_name = 'BayesianRidge'
    dimension = 300
    
    if seq_length == 0:
        r2_test, rmse_test = r2_rmse_against_mean(y_data)

    else:
        try:
            X_data_processed, _ = process_data(X_data, encoder_name, reducer_name, None, dimension, entropy, seq_length)
        except:
            return 0, np.inf
        
        r2_test = []
        rmse_test = []
        for _ in range(num_runs):
            try:
                result = AR.ML_SHELL(X_data_processed, y_data, model_name, split_fraction=split_fraction)
                r2_test.append(result['metrics_df']['R2 Test'].values[0])
                rmse_test.append(result['metrics_df']['RMSE Test'].values[0])
            except:
                pass

        r2_test = np.average(r2_test)
        rmse_test = np.average(rmse_test)
    
    return r2_test,rmse_test

def test_single_length(X_data,y_data,entropy,seq_length,split_fraction,num_runs):
    encoder_name = 'one_hot'
    reducer_name = 'pca'
    model_name = 'BayesianRidge'
    dimension = 300
    
    if seq_length == 0:
        r2_test, rmse_test = r2_rmse_against_mean(y_data)

    else:
        try:
            X_data_processed, _ = process_data(X_data, encoder_name, reducer_name, None, dimension, entropy, seq_length)
        except:
            return 0, np.inf
        
        r2_test = []
        rmse_test = []
        for _ in range(num_runs):
            try:
                result = AR.ML_SHELL(X_data_processed, y_data, model_name, split_fraction=split_fraction)
                r2_test.append(result['metrics_df']['R2 Test'].values[0])
                rmse_test.append(result['metrics_df']['RMSE Test'].values[0])
            except:
                pass

        r2_test = np.average(r2_test)
        rmse_test = np.average(rmse_test)
    
    return r2_test,rmse_test

def brute_force_search(sequence_data, target_data, entropy, max_seq_length, split_fraction, num_runs):
    # Create lists to hold the remote tasks
    futures = []
    lengths = []

    # Launch tasks for each sequence length
    for seq_length in range(0, max_seq_length + 1):
        lengths.append(seq_length)
        # Call the remote function
        future = test_single_length_remote.remote(sequence_data, target_data, entropy, seq_length, split_fraction, num_runs)
        futures.append(future)

    # Gather results from all futures
    results = ray.get(futures)

    # Unpack results into separate arrays
    r2_data = np.array([result[0] for result in results])
    rmse_data = np.array([result[1] for result in results])

    return r2_data, rmse_data, np.array(lengths)

def evaluate(seq_length, sequence_data, target_data, entropy, split_fraction, num_runs):
    global evaluation_count
    if seq_length in results_dict:
        return results_dict[seq_length]
    r2, rmse = test_single_length(sequence_data, target_data, entropy, seq_length, split_fraction, num_runs)
    results_dict[seq_length] = (r2, rmse)
    evaluation_count += 1
    return r2, rmse

def add_segment(start_length, end_length, sequence_data, target_data, entropy, split_fraction, num_runs):
    if start_length < end_length:
        mid_length = (start_length + end_length) // 2
        
        # Evaluate the start, mid, and end segments if not already evaluated
        if start_length not in evaluated_segments:
            r2_start, rmse_start = evaluate(start_length, sequence_data, target_data, entropy, split_fraction, num_runs)
            evaluated_segments.add(start_length)
        else:
            r2_start, rmse_start = results_dict[start_length]

        if mid_length not in evaluated_segments:
            r2_mid, rmse_mid = evaluate(mid_length, sequence_data, target_data, entropy, split_fraction, num_runs)
            evaluated_segments.add(mid_length)
        else:
            r2_mid, rmse_mid = results_dict[mid_length]

        if end_length not in evaluated_segments:
            r2_end, rmse_end = evaluate(end_length, sequence_data, target_data, entropy, split_fraction, num_runs)
            evaluated_segments.add(end_length)
        else:
            r2_end, rmse_end = results_dict[end_length]

        change_start_mid = abs(r2_start - r2_mid) + abs(rmse_start - rmse_mid)
        change_mid_end = abs(r2_mid - r2_end) + abs(rmse_mid - rmse_end)

        if change_start_mid > change_threshold:
            heapq.heappush(priority_queue, (-change_start_mid, start_length, mid_length))
        if change_mid_end > change_threshold:
            heapq.heappush(priority_queue, (-change_mid_end, mid_length + 1, end_length))

        r2_data[start_length] = r2_start
        rmse_data[start_length] = rmse_start
        r2_data[mid_length] = r2_mid
        rmse_data[mid_length] = rmse_mid
        r2_data[end_length] = r2_end
        rmse_data[end_length] = rmse_end

def adaptive_search(sequence_data, target_data, max_seq_length, entropy, split_fraction, n_trials, num_runs):
    global max_iterations, change_threshold, r2_data, rmse_data, priority_queue, evaluated_segments, results_dict, evaluation_count

    # Parameters for adaptive search
    max_iterations = n_trials // 3
    change_threshold = 0.01
    evaluation_count = 0

    r2_data = {}
    rmse_data = {}
    lengths = []

    priority_queue = []
    evaluated_segments = set()
    results_dict = {}

    # Evaluate length 0
    r2_zero, rmse_zero = evaluate(0, sequence_data, target_data, entropy, split_fraction, num_runs)
    r2_data[0] = r2_zero
    rmse_data[0] = rmse_zero
    evaluated_segments.add(0)

    initial_start_length = 1
    initial_end_length = max_seq_length
    initial_mid_length = max_seq_length // 2

    # Evaluate the initial segments: start, mid, and end
    add_segment(initial_start_length, initial_end_length, sequence_data, target_data, entropy, split_fraction, num_runs)

    while evaluation_count < n_trials:
        if not priority_queue:
            # If the priority queue is empty, refill it with remaining segments
            for length in range(1, max_seq_length + 1):
                if length not in evaluated_segments:
                    heapq.heappush(priority_queue, (0, length, length))

        _, start_length, end_length = heapq.heappop(priority_queue)
        mid_length = (start_length + end_length) // 2
        add_segment(start_length, end_length, sequence_data, target_data, entropy, split_fraction, num_runs)

    sorted_lengths = sorted(r2_data.keys())
    sorted_r2_data = [r2_data[length] for length in sorted_lengths]
    sorted_rmse_data = [rmse_data[length] for length in sorted_lengths]

    return np.array(sorted_r2_data), np.array(sorted_rmse_data), np.array(sorted_lengths)


def enforce_monotonicity(data, increasing=True):
    """
    Enforces monotonicity on the data list.
    If increasing is True, ensures the data is monotonically increasing.
    If increasing is False, ensures the data is monotonically decreasing.
    """
    if increasing:
        for i in range(1, len(data)):
            if data[i] < data[i - 1]:
                data[i] = data[i - 1]
    else:
        for i in range(1, len(data)):
            if data[i] > data[i - 1]:
                data[i] = data[i - 1]
    return data

def analyse_data(r2_data, rmse_data, lengths, max_seq_length, entropy_save_path):
    # Ensure the entropy_save_path exists
    os.makedirs(entropy_save_path, exist_ok=True)
    
    # Combine the data into a single list of tuples and sort by length
    combined_data = sorted(zip(lengths, r2_data, rmse_data))
    sorted_lengths, sorted_r2_data, sorted_rmse_data = zip(*combined_data)
    
    # Enforce monotonicity
    sorted_r2_data = enforce_monotonicity(list(sorted_r2_data), increasing=True)
    sorted_rmse_data = enforce_monotonicity(list(sorted_rmse_data), increasing=False)
    
    # Create a dataframe to store the results
    df = pd.DataFrame({'Length': np.arange(0, max_seq_length + 1, 1)})
    
    # Initialize columns for rmse and r2 with NaN values
    df['RMSE'] = np.inf
    df['R2'] = 0
    
    # Fill the dataframe with the provided data
    for i, length in enumerate(sorted_lengths):
        df.loc[df['Length'] == length, 'RMSE'] = sorted_rmse_data[i]
        df.loc[df['Length'] == length, 'R2'] = sorted_r2_data[i]
    
    # Infill missing values based on the previous values
    df.fillna(method='ffill', inplace=True)
    
    # Save the dataframe to a CSV file
    csv_path = os.path.join(entropy_save_path, 'analysis_results.csv')
    df.to_csv(csv_path, index=False)

    return csv_path

def get_concave_hull_section(x, y, alpha=1.0):
    points = np.vstack((x, y)).T
    if alpha is not None:
        hull = alphashape(points, alpha)
    else:
        return points[ConvexHull(points).vertices]
    
    if hull.geom_type == 'Polygon' and alpha is not None:
        x_hull, y_hull = hull.exterior.xy
        envelope = np.vstack((x_hull, y_hull)).T
    else:
        envelope = points[ConvexHull(points).vertices]

    return envelope

def get_lower(polygon):
    minx = np.argmin(polygon[:, 0])
    maxx = np.argmax(polygon[:, 0]) + 1
    if minx >= maxx:
        lower_curve = np.concatenate([polygon[minx:], polygon[:maxx]])
    else:
        lower_curve = polygon[minx:maxx]
    return lower_curve

def get_upper(polygon):
    lower_curve = get_lower(polygon)
    upper_curve = []
    minx_index = np.argmin(polygon[:, 0])
    maxx_index = np.argmax(polygon[:, 0])
    upper_curve.append(polygon[maxx_index,:])
    for i, point in enumerate(polygon):
        if not any(np.array_equal(point, lc_point) for lc_point in lower_curve):
            upper_curve.append(point)
    upper_curve.append(polygon[minx_index,:])
    upper_curve = np.array(upper_curve)
    return upper_curve[upper_curve[:, 0].argsort()]

def plot_convex_hull_section(x, y, direction):
    #points = np.vstack((x, y)).T
    mask = ~np.isnan(x) & ~np.isnan(y)
    filtered_x = x[mask]
    filtered_y = y[mask]

    envelope = get_concave_hull_section(filtered_x, filtered_y, alpha=None)

    if direction == 'maximize':
        envelope = get_upper(envelope)
        plt.plot(envelope[:, 0], envelope[:, 1], color='red', linewidth=4)
    else:
        envelope = get_lower(envelope)
        plt.plot(envelope[:, 0], envelope[:, 1], color='green', linewidth=4)
    
    return envelope

def plot_results(r2_data, rmse_data, lengths, max_seq_length, entropy_save_path):
    os.makedirs(entropy_save_path, exist_ok=True)
    
    # Plot full R2
    plt.figure(figsize=(10, 6))
    plt.scatter(lengths, r2_data, label='R2', color='black')
    plt.xlabel('Sequence Length', fontsize=24)
    plt.ylabel('R2', fontsize=24)
    plt.xticks([np.floor(min(lengths)), np.ceil(max(lengths))], fontsize=20)
    plt.ylim([0, 1])
    plt.yticks([0, 1], fontsize=20)
    
    # Draw upper convex hull for R2
    plot_convex_hull_section(lengths, r2_data, direction='maximize')
    
    # Save the full R2 plot
    r2_plot_path = os.path.join(entropy_save_path, 'R2_vs_Sequence_Length.png')
    plt.tight_layout()
    plt.savefig(r2_plot_path)
    plt.close()

    # Plot full RMSE
    plt.figure(figsize=(10, 6))
    plt.scatter(lengths, rmse_data, label='RMSE', color='black')
    plt.xlabel('Sequence Length', fontsize=24)
    plt.ylabel('RMSE', fontsize=24)
    plt.xticks([np.floor(min(lengths)), np.ceil(max(lengths))], fontsize=20)
    try:
        plt.ylim([0, np.ceil(max(rmse_data))])
        plt.yticks([0, np.ceil(max(rmse_data))], fontsize=20)
    except:
        plt.ylim([0, np.ceil(rmse_data[0])])
        plt.yticks([0, np.ceil(rmse_data[0])], fontsize=20)

    # Draw lower convex hull for RMSE
    plot_convex_hull_section(lengths, rmse_data, direction='minimize')
    
    # Save the full RMSE plot
    rmse_plot_path = os.path.join(entropy_save_path, 'RMSE_vs_Sequence_Length.png')
    plt.tight_layout()
    plt.savefig(rmse_plot_path)
    plt.close()

    ### First 10% plots ###

    # Calculate the range corresponding to the first 10% of the sequence lengths
    ten_percent_index = int(0.1 * len(lengths))
    lengths_10 = lengths[:ten_percent_index]
    r2_data_10 = r2_data[:ten_percent_index]
    rmse_data_10 = rmse_data[:ten_percent_index]

    # Plot R2 for the first 10% of sequence lengths
    plt.figure(figsize=(10, 6))
    plt.scatter(lengths_10, r2_data_10, label='R2', color='black')
    plt.xlabel('Sequence Length (First 10%)', fontsize=24)
    plt.ylabel('R2', fontsize=24)
    plt.xticks([np.floor(min(lengths_10)), np.ceil(max(lengths_10))], fontsize=20)
    plt.ylim([0, 1])
    plt.yticks([0, 1], fontsize=20)
    
    # Draw upper convex hull for R2 (first 10%)
    plot_convex_hull_section(lengths_10, r2_data_10, direction='maximize')
    
    # Save the R2 plot (first 10%)
    r2_plot_10_path = os.path.join(entropy_save_path, 'R2_vs_Sequence_Length_10_percent.png')
    plt.tight_layout()
    plt.savefig(r2_plot_10_path)
    plt.close()

    # Plot RMSE for the first 10% of sequence lengths
    plt.figure(figsize=(10, 6))
    plt.scatter(lengths_10, rmse_data_10, label='RMSE', color='black')
    plt.xlabel('Sequence Length (First 10%)', fontsize=24)
    plt.ylabel('RMSE', fontsize=24)
    plt.xticks([np.floor(min(lengths_10)), np.ceil(max(lengths_10))], fontsize=20)
    try:
        plt.ylim([0, np.ceil(max(rmse_data_10))])
        plt.yticks([0, np.ceil(max(rmse_data_10))], fontsize=20)
    except:
        plt.ylim([0, np.ceil(rmse_data_10[0])])
        plt.yticks([0, np.ceil(rmse_data_10[0])], fontsize=20)

    # Draw lower convex hull for RMSE (first 10%)
    plot_convex_hull_section(lengths_10, rmse_data_10, direction='minimize')
    
    # Save the RMSE plot (first 10%)
    rmse_plot_10_path = os.path.join(entropy_save_path, 'RMSE_vs_Sequence_Length_10_percent.png')
    plt.tight_layout()
    plt.savefig(rmse_plot_10_path)
    plt.close()
    
def linear_interpolate_non_negative(x, y):
    new_x = np.arange(x[0], x[-1] + 1, 1)
    new_y = np.interp(new_x, x, y)
    
    for i in range(1, len(new_y)):
        if new_y[i] > new_y[i - 1]:
            new_y[i] = new_y[i - 1]
    
    return new_x, new_y

def calculate_derivative_prop(lengths, rmse_data):
    # Ensure lengths are sorted
    sorted_indices = np.argsort(lengths)
    lengths = lengths[sorted_indices]
    rmse_data = enforce_monotonicity(rmse_data[sorted_indices],False)
    
    # Interpolate to ensure non-negative gradient
    new_lengths, new_rmse_data = linear_interpolate_non_negative(lengths, rmse_data)

    # Calculate the backward difference dy/dx
    dy = np.diff(new_rmse_data)  # y[n] - y[n-1]
    dx = np.diff(new_lengths)  # x[n] - x[n-1]

    dy_dx = dy / dx

    mod_deriv = -1 * dy_dx# / new_rmse_data[:-1]

    # Prepare the output DataFrame
    derivative_df = pd.DataFrame({
        'X': new_lengths[1:],  # use non-padded x-values for alignment with derivative results
        'dy/dx': mod_deriv     # derivative results
    })

    return derivative_df

def assign_ranking_based_on_ent(result_df, Ent):
    Ent = np.asarray(Ent)

    # Sort the DataFrame by dy/dx
    sorted_df = result_df.sort_values('dy/dx', ascending=False).reset_index(drop=True)

    # Initialize ranking array of zeros with the same length as Ent
    ranking = np.zeros(len(Ent))

    # Iterate over sorted DataFrame
    for _, row in sorted_df.iterrows():
        # Get X value as index, note that X is assumed 1-based for ranking purposes
        x_value = int(row['X']) - 1  # Adjust for 0-based index in Python

        # Find the X[i]th highest value in Ent, if X is out of bounds, skip it
        if x_value < len(Ent):
            sorted_indices = np.argsort(-Ent)  # Get indices that would sort Ent in descending order
            target_index = sorted_indices[x_value]  # X[i]th highest, adjusted for zero-index
            ranking[target_index] = row['dy/dx']

    return ranking

def rerank(rmse_data, lengths, entropy):
    # Ensure lengths and rmse_data are numpy arrays for processing
    lengths = np.array(lengths)
    rmse_data = np.array(rmse_data)
    
    # Calculate the derivative proportions
    derivative_df = calculate_derivative_prop(lengths, rmse_data)

    # Assign ranking based on entropy
    ranking = assign_ranking_based_on_ent(derivative_df, entropy)

    return ranking

def search_sequence_lengths(save_folder, 
                            sequence_data, 
                            target_data, 
                            Data_path, 
                            n_trials=5, 
                            split_fraction=0.2, 
                            max_seq_length=554, 
                            num_runs=3,
                            file_names=['min','max','mean','boltzmann'],
                            alpha=0):
    with open(Data_path, 'rb') as f:
        entropy_data = pickle.load(f)
    
    new_entropies = []
    metrics = []

    for i in range(len(entropy_data)):
        entropy_save_path = f"{save_folder}/{file_names[i]}"
        os.makedirs(entropy_save_path,exist_ok=True)
        entropy = entropy_data[i]
        if n_trials>max_seq_length:
            r2_data,rmse_data,lengths = brute_force_search(sequence_data,target_data,entropy,max_seq_length,split_fraction,num_runs)
        else:
            r2_data,rmse_data,lengths = adaptive_search(sequence_data,target_data,max_seq_length,entropy,split_fraction,n_trials,num_runs)

        analyse_data(r2_data,rmse_data,lengths,max_seq_length,entropy_save_path)

        plot_results(r2_data,rmse_data,lengths,max_seq_length,entropy_save_path)


        _,metric= create_monotonic_function(r2_data,lengths)
        metric = eval_metric(metric)

        metrics.append(metric)

    with open(f'{save_folder}/new_entropies.pkl', 'wb') as f:
        pickle.dump(new_entropies,f)

    

    return f'{save_folder}/new_entropies.pkl', metrics

def sanitize_csv(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Ensure the data contains the required columns
    if 'Length' in df.columns and 'RMSE' in df.columns and 'R2' in df.columns:
        lengths = df['Length'].values
        r2_data = df['R2'].values
        rmse_data = df['RMSE'].values
        
        # Ensure R2 is monotonically increasing
        r2_data = enforce_monotonicity(r2_data, increasing=True)
        
        # Ensure RMSE is monotonically decreasing
        rmse_data = enforce_monotonicity(rmse_data, increasing=False)
        
        # Update the dataframe with the modified data
        df['R2'] = r2_data
        df['RMSE'] = rmse_data
        
        # Overwrite the original CSV file with the modified data
        df.to_csv(file_path, index=False)
        #print(f"Processed and updated: {file_path}")
    #else:
        #print(f"Skipping file (missing required columns): {file_path}")

def sanitize_directory(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    first_line = f.readline().strip()
                    if first_line == 'Length,RMSE,R2':
                        sanitize_csv(file_path)
                    #else:
                        #print(f"Skipping file (first line does not match required format): {file_path}")

def calculate_sum(directory):
    results = {}
    
    # Loop through each folder in the given directory
    for folder_name in os.listdir(directory):
        folder_path = os.path.join(directory, folder_name)
        
        # Check if it is a directory
        if os.path.isdir(folder_path):
            file_path = os.path.join(folder_path, 'analysis_results.csv')
            
            # Check if the analysis_results.csv file exists in the directory
            if os.path.exists(file_path):
                # Load the CSV file
                df = pd.read_csv(file_path)
                
                # Ensure the CSV has at least three columns
                if df.shape[1] >= 3:
                    # Get the first and third columns
                    col1 = df.iloc[:, 0]
                    col3 = df.iloc[:, 2]
                    
                    # Calculate the weighted sum of the third column
                    length = len(col3)
                    exponent = 2
                    weights = np.arange(length, 0, -1)
                    norm_col3 = ((col3-col3[0])/(np.max(col3)-col3[0]))**exponent
                    weighted_sum_col3 = np.sum(norm_col3 * (weights**exponent))/np.sum(weights**exponent)
                    
                    
                    # Store the result in the dictionary
                    results[folder_name] = weighted_sum_col3
    
    return results