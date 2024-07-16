import numpy as np
from scipy.stats import entropy

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
    entropy_value = entropy(probabilities)
    
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
    entropy_value = entropy(probabilities)
    
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
    joint_entropy = entropy(joint_prob)
    
    return joint_entropy

# Example usage:
if __name__ == "__main__":
    # Example for continuous entropy
    data_continuous = np.random.normal(loc=0, scale=1, size=500)
    entropy_continuous = calculate_entropy_continuous(data_continuous)
    print("Entropy of continuous data:", entropy_continuous)
    
    # Example for discrete entropy
    data_discrete = np.random.choice(['A', 'B', 'C'], size=500)
    entropy_discrete = calculate_entropy_discrete(data_discrete)
    print("Entropy of discrete data:", entropy_discrete)
    
    # Example for joint entropy with mixed types
    joint_entropy_mixed = calculate_joint_entropy(data_discrete, data_continuous, types=('categorical', 'continuous'))
    print("Joint entropy of X (continuous) and Y (categorical):", joint_entropy_mixed)


    print('Mutual Info of X (continuous) and Y (categorical):',(entropy_discrete+entropy_continuous-joint_entropy_mixed)/(entropy_continuous))