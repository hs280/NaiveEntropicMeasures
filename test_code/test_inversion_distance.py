import random
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


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

def generate_random_case(n, L):
    # Generate a random unsorted list of integers
    nums = random.sample(range(1, L+1), L)  # Sample L unique integers from 1 to L
    # Generate random ground truth indices for n highest values
    gt_indices = random.sample(range(L), n)  # Select n unique indices from 0 to L-1
    return nums, gt_indices

# Example usage
n = 5  # Number of highest values
L = 10  # Length of nums
random_cases = []
swap_counts = []  # List to store swap counts for histogram

for _ in range(500):
    nums, gt_indices = generate_random_case(n, L)
    result = min_adjacent_swaps(nums, gt_indices)
    random_cases.append((nums, gt_indices, result))
    swap_counts.append(result)  # Store the swap count

# Plotting the histogram
plt.figure(figsize=(5, 3))
plt.hist(swap_counts, bins=15, color='blue', alpha=0.7)
plt.title('Distribution of Minimum Adjacent Swaps for Random Cases')
plt.xlabel('Number of Adjacent Swaps')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)
plt.show()

# Function to generate the best case: Ground truth already matched
def get_best_case(L, n):
    # Ground truth matches the best case where the top n values are at the beginning
    set_values = [1] * n + [0] * (L - n)  # n ones followed by L-n zeros
    gt = list(range(1, n + 1))  # Ground truth is already ordered
    return gt, set_values

# Function to generate the worst case: Evenly spaced 1's and 0's
def get_worst_case(L, n):
    # Ground truth is ordered as [1, 2, 3, ..., n]
    gt = list(range(1, n + 1))
    
    # Worst case where the ones are spaced as far back as possible
    set_values = [0] * (L - n) + [1] * n
    return  min_adjacent_swaps(set_values,gt)

# Monte Carlo simulation for a range of L and n
def monte_carlo_swaps(L_values, n_values, samples=1):
    max_swaps = np.zeros((len(L_values), len(n_values)))
    min_swaps = np.ones((len(L_values), len(n_values)))

    for i, L in enumerate(L_values):
        for j, n in enumerate(n_values):
            if n < L:
                swap_counts = []
                for _ in range(samples*(L-n)):
                    nums, gt_indices = generate_random_case(n, L)
                    result = min_adjacent_swaps(nums, gt_indices)
                    swap_counts.append(result)  # Store the swap count
                min_swaps[i, j] = get_best_case(L,n)
                max_swaps[i, j] = get_worst_case(L,n)

    return max_swaps, min_swaps

# Parameters for the simulation
L_values = range(1, 21)  # Length of the list from 1 to 20
n_values = range(1, 21)  # Number of highest values to consider

# Run Monte Carlo simulations
min_swaps, max_swaps = monte_carlo_swaps(L_values, n_values)

# Plotting the heatmaps
plt.figure(figsize=(12, 6))
# Enable interactive mode
# plt.ion()  # Turn on interactive mode


# Heatmap for maximum swaps
plt.subplot(1, 2, 1)
sns.heatmap(max_swaps, annot=False, fmt=".3f", cmap='YlGnBu',
            xticklabels=n_values, yticklabels=L_values)
plt.title('Max Swaps Heatmap')
plt.xlabel('n (Number of Highest Values)')
plt.ylabel('L (Length of List)')

# Heatmap for minimum swaps
plt.subplot(1, 2, 2)
sns.heatmap(min_swaps, annot=False, fmt=".3f", cmap='YlGnBu',
            xticklabels=n_values, yticklabels=L_values)
plt.title('Min Swaps Heatmap')
plt.xlabel('n (Number of Highest Values)')
plt.ylabel('L (Length of List)')

plt.tight_layout()
plt.show()



# Plotting the multi-line line plots
plt.figure(figsize=(14, 12))

# Plot 1: Max Swaps vs L for different n values
plt.subplot(2, 2, 1)
for n in [5, 10, 20, 50, 100]:  # Sample n values
    if n <= len(n_values):
        plt.plot(L_values, max_swaps[:, n-1], label=f'n={n}')  # n-1 for index
plt.title('Max Swaps vs L for Different n')
plt.xlabel('L (Length of List)')
plt.ylabel('Max Swaps')
plt.legend()
plt.grid()

# Plot 2: Min Swaps vs L for different n values
plt.subplot(2, 2, 2)
for n in [5, 10, 20, 50, 100]:  # Sample n values
    if n <= len(n_values):
        plt.plot(L_values, min_swaps[:, n-1], label=f'n={n}')  # n-1 for index
plt.title('Min Swaps vs L for Different n')
plt.xlabel('L (Length of List)')
plt.ylabel('Min Swaps')
plt.legend()
plt.grid()

# Plot 3: Max Swaps vs n for different L values
plt.subplot(2, 2, 3)
for L in [5, 10, 20, 50, 100]:  # Sample L values
    if L <= len(L_values):
        plt.plot(n_values, max_swaps[L-1, :], label=f'L={L}')  # L-1 for index
plt.title('Max Swaps vs n for Different L')
plt.xlabel('n (Number of Highest Values)')
plt.ylabel('Max Swaps')
plt.legend()
plt.grid()

# Plot 4: Min Swaps vs n for different L values
plt.subplot(2, 2, 4)
for L in [5, 10, 20, 50, 100]:  # Sample L values
    if L <= len(L_values):
        plt.plot(n_values, min_swaps[L-1, :], label=f'L={L}')  # L-1 for index
plt.title('Min Swaps vs n for Different L')
plt.xlabel('n (Number of Highest Values)')
plt.ylabel('Min Swaps')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

# New plot: Max Swaps at n = L/2 vs. L
plt.figure(figsize=(8, 6))

# Calculate max swaps at n = L/2 for each L
half_n_swaps = [max_swaps[L-1, (L // 2) - 1] if L > 1 else 0 for L in L_values]

# Calculate theoretical L^2/4 values for comparison
theoretical_swaps = [(L**2) / 4 for L in L_values]

# Plot the max swaps at n = L/2
plt.plot(L_values, half_n_swaps, marker='o', linestyle='-', color='b', label='Max Swaps at n=L/2')

# Plot the theoretical upper bound ZL^2/4
plt.plot(L_values, theoretical_swaps, marker='x', linestyle='--', color='r', label='Theoretical L^2/4')

plt.title('Max Swaps at n = L/2 vs. L')
plt.xlabel('L (Length of List)')
plt.ylabel('Max Swaps at n = L/2')
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()
