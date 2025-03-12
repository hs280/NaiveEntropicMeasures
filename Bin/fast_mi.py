import numpy as np
from scipy.stats import entropy
import matplotlib.pyplot as plt

import numpy as np
from scipy.stats import entropy
from numba import jit
from scipy.sparse import csr_matrix
from typing import Union, Tuple, List
import pandas as pd

def extract_array_from_input(data: Union[np.ndarray, pd.Series, pd.DataFrame, List, Tuple, str]) -> np.ndarray:
    """
    Extract numpy array from various input types.
    """
    if isinstance(data, pd.DataFrame):
        if data.shape[1] != 1:
            raise ValueError("DataFrame input must have exactly one column")
        return data.iloc[:, 0].values
    elif isinstance(data, pd.Series):
        return data.values
    elif isinstance(data, (list, tuple, str)):
        return np.array(data)
    elif isinstance(data, np.ndarray):
        return data
    else:
        raise ValueError(f"Unsupported input type: {type(data)}")

def preprocess_input(data: Union[np.ndarray, pd.Series, pd.DataFrame, List, Tuple], 
                    is_continuous: bool) -> Tuple[np.ndarray, bool, np.ndarray]:
    """
    Preprocess input data to handle different types efficiently.
    Now supports pandas DataFrame and Series inputs.
    """
    # Extract array from input
    data = extract_array_from_input(data)
    
    if data.dtype.kind in {'U', 'S', 'O'}:  # string or object types
        is_continuous = False
        unique_vals, data = np.unique(data, return_inverse=True)
        return data, False, unique_vals
    
    if data.dtype.kind in {'i', 'u'}:
        if is_continuous:
            return data.astype(np.float32), True, None
        unique_vals, data = np.unique(data, return_inverse=True)
        return data, False, unique_vals
        
    if data.dtype.kind == 'f':
        return data.astype(np.float32), True, None
    
    if data.dtype.kind == 'b':
        unique_vals, data = np.unique(data, return_inverse=True)
        return data, False, unique_vals
    
    raise ValueError(f"Unsupported data type: {data.dtype}")

@jit(nopython=True)
def _fast_digitize(data, bins):
    """Optimized digitization for continuous data"""
    indices = np.zeros(len(data), dtype=np.int64)
    for i in range(len(data)):
        left, right = 0, len(bins) - 1
        x = data[i]
        while left < right:
            mid = (left + right) // 2
            if x > bins[mid]:
                left = mid + 1
            else:
                right = mid
        indices[i] = left
    return indices

def calculate_joint_entropy(X: Union[np.ndarray, pd.Series, pd.DataFrame, List, Tuple], 
                          Y: Union[np.ndarray, pd.Series, pd.DataFrame, List, Tuple],
                          types: Tuple[str, str] = ('continuous', 'continuous'),
                          num_bins: Union[int, None] = None,
                          sparse_threshold: float = 0.1) -> float:
    """
    Calculate joint entropy H(X,Y) with support for pandas DataFrame and Series inputs.
    
    Parameters:
    -----------
    X : array-like, pandas Series, or DataFrame
        First variable. If DataFrame, must contain exactly one column.
    Y : array-like, pandas Series, or DataFrame
        Second variable. If DataFrame, must contain exactly one column.
    types : tuple of str
        Types of X and Y ('continuous' or 'categorical')
    num_bins : int or None
        Number of bins for continuous variables
    sparse_threshold : float
        Threshold for using sparse matrix representation
        
    Returns:
    --------
    float
        Joint entropy H(X,Y)
        
    Examples:
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'X': [1,2,3], 'Y': ['A','B','A']})
    >>> calculate_joint_entropy(df['X'], df['Y'], types=('continuous', 'categorical'))
    >>> calculate_joint_entropy(df[['X']], df.Y, types=('continuous', 'categorical'))
    """
    is_x_continuous = types[0].lower() == 'continuous'
    is_y_continuous = types[1].lower() == 'continuous'
    
    # Preprocess inputs
    X_processed, is_x_continuous, x_unique = preprocess_input(X, is_x_continuous)
    Y_processed, is_y_continuous, y_unique = preprocess_input(Y, is_y_continuous)
    
    n_samples = len(X_processed)
    assert len(Y_processed) == n_samples, "X and Y must have the same length"
    
    # Quick exit for degenerate cases
    if len(np.unique(X_processed)) == 1 or len(np.unique(Y_processed)) == 1:
        return 0.0

    def get_bins_and_indices(data, is_continuous, unique_vals=None):
        if not is_continuous:
            return len(unique_vals), data
        
        if num_bins is None:
            if len(data) > 1000:
                q25, q75 = np.percentile(data, [25, 75])
                bin_width = 2 * (q75 - q25) * len(data)**(-1/3)
                num_bins_auto = max(10, int((data.max() - data.min()) / bin_width))
                edges = np.linspace(data.min(), data.max(), num_bins_auto + 1)
            else:
                edges = np.histogram_bin_edges(data, bins='fd')
        else:
            edges = np.linspace(data.min(), data.max(), num_bins + 1)
            
        indices = _fast_digitize(data, edges)
        return len(edges) - 1, indices

    # Get dimensions and indices
    nx, x_indices = get_bins_and_indices(X_processed, is_x_continuous, x_unique)
    ny, y_indices = get_bins_and_indices(Y_processed, is_y_continuous, y_unique)
    
    # Ensure indices are within bounds
    x_indices = np.clip(x_indices, 0, nx - 1)
    y_indices = np.clip(y_indices, 0, ny - 1)
    
    # Choose processing method based on sparsity
    expected_sparsity = min(1.0, (nx * ny) / (n_samples * 10))
    
    if expected_sparsity > sparse_threshold:
        # Sparse matrix for high-dimensional cases
        joint_counts = csr_matrix(
            (np.ones(n_samples, dtype=np.float32),
             (x_indices, y_indices)),
            shape=(nx, ny)
        )
        joint_prob = (joint_counts.data / n_samples).astype(np.float32)
    else:
        # Dense matrix
        joint_counts = np.zeros((nx, ny), dtype=np.float32)
        np.add.at(joint_counts, (x_indices, y_indices), 1)
        joint_prob = (joint_counts / n_samples).ravel()
    
    # Filter out zeros and compute entropy
    joint_prob = joint_prob[joint_prob > 0]
    return float(entropy(joint_prob, base=np.e))

def test_joint_entropy():
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate test data
    n_samples = 10000
    
    # Create continuous X data with 3 clear clusters
    cluster_means = [0, 5, 10]
    cluster_stds = [1, 1, 1]
    cluster_sizes = [3000, 4000, 3000]
    
    X = np.concatenate([
        np.random.normal(mu, std, size) 
        for mu, std, size in zip(cluster_means, cluster_stds, cluster_sizes)
    ])
    
    # Create categorical Y data correlated with X clusters
    # We'll use 'A', 'B', 'C' with probabilities dependent on X value
    def assign_category(x):
        probs = [
            np.exp(-(x - mu)**2 / (2*std**2)) 
            for mu, std in zip(cluster_means, cluster_stds)
        ]
        probs = np.array(probs) / sum(probs)
        return np.random.choice(['A', 'B', 'C'], p=probs)
    
    Y = np.array([assign_category(x) for x in X])
    
    # Calculate joint entropy
    result = calculate_joint_entropy(X, Y, types=('continuous', 'categorical'))
    
    # Calculate marginal entropies for comparison
    _, x_bins = np.histogram(X, bins='fd')
    x_hist = np.histogram(X, bins=x_bins, density=True)[0]
    x_entropy = entropy(x_hist[x_hist > 0], base=np.e)
    
    y_values, y_counts = np.unique(Y, return_counts=True)
    y_probs = y_counts / len(Y)
    y_entropy = entropy(y_probs, base=np.e)
    
    # Print results
    print("\nTest Results:")
    print(f"Number of samples: {n_samples}")
    print(f"X range: [{X.min():.2f}, {X.max():.2f}]")
    print(f"Y categories: {y_values}")
    print("\nEntropy Values:")
    print(f"H(X): {x_entropy:.3f}")
    print(f"H(Y): {y_entropy:.3f}")
    print(f"H(X,Y): {result:.3f}")
    
    # Mutual information: I(X;Y) = H(X) + H(Y) - H(X,Y)
    mi = x_entropy + y_entropy - result
    print(f"I(X;Y): {mi:.3f}")
    
    # Visualization
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Distribution of X colored by Y
    plt.subplot(131)
    for category in y_values:
        mask = Y == category
        plt.hist(X[mask], bins=50, alpha=0.5, label=category, density=True)
    plt.title('Distribution of X by Category')
    plt.xlabel('X value')
    plt.ylabel('Density')
    plt.legend()
    
    # Plot 2: Category counts
    plt.subplot(132)
    plt.bar(y_values, y_counts)
    plt.title('Category Counts')
    plt.xlabel('Category')
    plt.ylabel('Count')
    
    # Plot 3: 2D histogram
    plt.subplot(133)
    x_bins = np.linspace(X.min(), X.max(), 50)
    y_bins = np.arange(len(y_values) + 1)
    plt.hist2d(X, np.searchsorted(y_values, Y), bins=[x_bins, y_bins])
    plt.title('2D Distribution')
    plt.xlabel('X value')
    plt.yticks(np.arange(len(y_values)) + 0.5, y_values)
    plt.ylabel('Category')
    plt.colorbar(label='Count')
    
    plt.tight_layout()
    plt.show()
    
    # Validation checks
    assert result >= 0, "Joint entropy should be non-negative"
    assert result <= x_entropy + y_entropy, "Joint entropy should be <= sum of marginal entropies"
    assert mi >= 0, "Mutual information should be non-negative"
    
    print("\nValidation checks passed!")
    
    return {
        'joint_entropy': result,
        'x_entropy': x_entropy,
        'y_entropy': y_entropy,
        'mutual_information': mi,
        'X': X,
        'Y': Y
    }

import pandas as pd
import numpy as np

def test_dataframe_inputs():
    """
    Test various DataFrame input formats for the joint entropy calculator.
    """
    # Create sample data
    n_samples = 1000
    
    # Create a DataFrame with mixed types
    df = pd.DataFrame({
        'continuous': np.random.normal(0, 1, n_samples),
        'categorical': np.random.choice(['A', 'B', 'C'], n_samples),
        'integer': np.random.randint(0, 5, n_samples),
        'boolean': np.random.choice([True, False], n_samples)
    })
    
    print("Testing different input combinations:\n")
    
    # Test 1: Series inputs
    print("1. Series inputs (continuous-categorical):")
    result = calculate_joint_entropy(
        df['continuous'], 
        df['categorical'],
        types=('continuous', 'categorical')
    )
    print(f"Joint entropy: {result:.3f}\n")
    
    # Test 2: DataFrame column inputs
    print("2. DataFrame column inputs (continuous-categorical):")
    result = calculate_joint_entropy(
        df[['continuous']], 
        df[['categorical']],
        types=('continuous', 'categorical')
    )
    print(f"Joint entropy: {result:.3f}\n")
    
    # Test 3: Mixed DataFrame and Series inputs
    print("3. Mixed DataFrame and Series inputs (continuous-categorical):")
    result = calculate_joint_entropy(
        df[['continuous']], 
        df.categorical,
        types=('continuous', 'categorical')
    )
    print(f"Joint entropy: {result:.3f}\n")
    
    # Test 4: Categorical-Categorical
    print("4. Categorical-Categorical inputs:")
    result = calculate_joint_entropy(
        df['categorical'], 
        df['boolean'],
        types=('categorical', 'categorical')
    )
    print(f"Joint entropy: {result:.3f}\n")
    
    # Test 5: Continuous-Continuous
    print("5. Continuous-Continuous inputs:")
    result = calculate_joint_entropy(
        df['continuous'], 
        df['continuous'] * 2 + np.random.normal(0, 0.1, n_samples),
        types=('continuous', 'continuous')
    )
    print(f"Joint entropy: {result:.3f}\n")
    
    # Test 6: Integer handling
    print("6. Integer handling:")
    # As continuous
    result1 = calculate_joint_entropy(
        df['integer'], 
        df['continuous'],
        types=('continuous', 'continuous')
    )
    # As categorical
    result2 = calculate_joint_entropy(
        df['integer'], 
        df['continuous'],
        types=('categorical', 'continuous')
    )
    print(f"Integer as continuous: {result1:.3f}")
    print(f"Integer as categorical: {result2:.3f}\n")
    
    # Test error cases
    print("7. Testing error handling:")
    try:
        # Try to input multiple columns
        result = calculate_joint_entropy(
            df[['continuous', 'categorical']], 
            df['boolean'],
            types=('continuous', 'categorical')
        )
    except ValueError as e:
        print(f"Caught expected error: {e}")

# Run the test
if __name__ == '__main__':
    results = test_joint_entropy()
    test_dataframe_inputs()