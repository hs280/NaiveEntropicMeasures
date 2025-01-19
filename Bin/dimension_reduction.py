import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, KernelPCA, SparsePCA, IncrementalPCA, TruncatedSVD, FastICA
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
#from tensorflow.keras.layers import Input, Dense
#from tensorflow.keras.models import Model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.model_selection import KFold
from sklearn.utils import resample
import os

def parse_model_name(method_name):
    """Convert the method name to lowercase for consistency."""
    return method_name.lower()

def build_autoencoder(input_dim, target_dimension):
    """
    Build a simple autoencoder model.

    Args:
        input_dim (int): Dimensionality of the input data.
        target_dimension (int): Desired output dimensionality of the encoded representation.

    Returns:
        tf.keras.Model: Autoencoder model.
    """
    # input_layer = Input(shape=(input_dim,))
    # encoded = Dense(64, activation='relu')(input_layer)  # You can adjust the size of the encoded layer
    # encoded = Dense(target_dimension, activation='relu')(encoded)  # Ensure target_dimension here
    # decoded = Dense(input_dim, activation='sigmoid')(encoded)

    # autoencoder = Model(input_layer, decoded)
    # autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    # # Create a separate encoder model for extracting the encoded representation
    # encoder = Model(input_layer, encoded)

    # return autoencoder, encoder

def reduce_dimensionality_with_autoencoder(X_data, target_dimension):
    """
    # Reduce dimensionality of binary input data using an autoencoder.

    # Args:
    #     X_data (pd.DataFrame): Input DataFrame with binary data.
    #     target_dimension (int): Desired output dimensionality of the encoded representation.

    # Returns:
    #     pd.DataFrame: Encoded representations of the input data.
    #     object: Autoencoder model.
    #     object: Inverse autoencoder model for reconstruction.
    # """
    # # Convert DataFrame to numpy array
    # features = X_data.values

    # # Normalize data to [0, 1] range
    # scaler = MinMaxScaler()
    # features_normalized = scaler.fit_transform(features)

    # # Build autoencoder model
    # input_dim = features.shape[1]
    # autoencoder, encoder = build_autoencoder(input_dim, target_dimension)

    # # Train the autoencoder using all of X_data
    # autoencoder.fit(features_normalized, features_normalized, epochs=50, batch_size=32, verbose=0)

    # # Use the trained autoencoder to encode and decode the input data
    # encoded_data = encoder.predict(features_normalized)
    # decoded_data = autoencoder.predict(features_normalized)

    # # Convert encoded data back to DataFrame
    # encoded_df = pd.DataFrame(encoded_data, columns=[f'Encoded_{i}' for i in range(1, target_dimension + 1)])

    # # Create an inverse autoencoder for reconstruction
    # input_layer = Input(shape=(target_dimension,))
    # decoded_layer = autoencoder.layers[-1](input_layer)
    # inverse_autoencoder = Model(input_layer, decoded_layer)

    # return encoded_df, encoder, inverse_autoencoder

def apply_auto_encoder_to_data(X_data_reducer, X_data):
    """
    Apply a previously obtained reducer to new data.

    Args:
        X_data_reducer: Transformer obtained from dimensionality reduction.
        X_data (pd.DataFrame): New data to be reduced.

    Returns:
        pd.DataFrame: Reduced data.
    """
    return pd.DataFrame(X_data_reducer.predict(X_data), columns=[f'Encoded_{i}' for i in range(1, X_data_reducer.layers[-1].output_shape[1] + 1)])

def reduce_dimensions_with_method(X_data, method_name, num_dimensions):
    """
    Reduce dimensions using the specified method.

    Args:
        X_data (pd.DataFrame): Input DataFrame.
        method_name (str): Name of the dimensionality reduction method.
        num_dimensions (int): Number of dimensions for the reduced data.

    Returns:
        pd.DataFrame: Reduced data.
        object: Transformer used for dimensionality reduction.
        object: Inverse transformer for reconstruction.
    """
    model_name = parse_model_name(method_name)
    X_data = np.float32(X_data)
    if model_name == "pca":
        transformer = PCA(n_components=num_dimensions)
        X_data_transformed = transformer.fit_transform(X_data)
        inverse_transformer = transformer.inverse_transform#lambda x: np.dot(x, transformer.components_) + transformer.mean_
    elif model_name == "kernelpca":
        transformer = KernelPCA(n_components=num_dimensions,fit_inverse_transform=True)
        X_data_transformed = transformer.fit_transform(X_data)
        inverse_transformer = transformer.inverse_transform
    elif model_name == "sparsepca":
        transformer = SparsePCA(n_components=num_dimensions)
        X_data_transformed = transformer.fit_transform(X_data)
        inverse_transformer = transformer.inverse_transform
    elif model_name == "incrementalpca":
        transformer = IncrementalPCA(n_components=num_dimensions)
        X_data_transformed = transformer.fit_transform(X_data)
        inverse_transformer = transformer.inverse_transform
    elif model_name == "truncatedsvd":
        transformer = TruncatedSVD(n_components=num_dimensions)
        X_data_transformed = transformer.fit_transform(X_data)
        inverse_transformer = transformer.inverse_transform
    elif model_name == "gaussianrandomprojection":
        transformer = GaussianRandomProjection(n_components=num_dimensions)
        X_data_transformed = transformer.fit_transform(X_data)
        inverse_transformer = transformer.inverse_transform
    elif model_name == "sparserandomprojection":
        transformer = SparseRandomProjection(n_components=num_dimensions)
        X_data_transformed = transformer.fit_transform(X_data)
        inverse_transformer = transformer.inverse_transform
    elif model_name == "ica":
        transformer = FastICA(n_components=num_dimensions)
        X_data_transformed = transformer.fit_transform(X_data)
        inverse_transformer = transformer.inverse_transform
    elif model_name == "autoencoder":
        X_data_transformed, transformer, inverse_transformer = reduce_dimensionality_with_autoencoder(X_data, num_dimensions)
    else:
        raise ValueError(f"Invalid method name: {method_name}")

    return pd.DataFrame(X_data_transformed), transformer, inverse_transformer

def apply_reducer_to_data(X_data_reducer, X_data, method_name):
    """
    Apply a previously obtained reducer to new data.

    Args:
        X_data_reducer: Transformer obtained from dimensionality reduction.
        X_data (pd.DataFrame): New data to be reduced.

    Returns:
        pd.DataFrame: Reduced data.
    """
    if method_name != "autoencoder":
        return pd.DataFrame(X_data_reducer.transform(X_data))
    else:
        return apply_auto_encoder_to_data(X_data_reducer, X_data)
    
def apply_inverse_reducer_to_data(inverse_reducer, reduced_X_data, method_name):
    """
    Apply a previously obtained reducer to new data.

    Args:
        X_data_reducer: Transformer obtained from dimensionality reduction.
        X_data (pd.DataFrame): New data to be reduced.

    Returns:
        pd.DataFrame: Reduced data.
    """
    if method_name != "autoencoder":
        return pd.DataFrame(inverse_reducer(reduced_X_data))
    else:
        return apply_auto_encoder_to_data(inverse_reducer, reduced_X_data)
    
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import explained_variance_score, r2_score

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import explained_variance_score, r2_score

from sklearn.metrics import mean_squared_error

def rmspe(y_true, y_pred):
    """
    Calculate Root Mean Square Percentage Error (RMSPE).

    Parameters:
        y_true (array-like): True values.
        y_pred (array-like): Predicted values.

    Returns:
        float: RMSPE value.
    """
    return mean_squared_error(y_true, y_pred)#np.sqrt(np.mean(np.abs((2*(y_true - y_pred)) ** 2)))

def test_dimensionality_reduction_methods(X_data, methods, num_dims_range, save_folder, plot_flag=True, num_bootstrap_samples=100, num_cv_folds=10):
    """
    Test various dimensionality reduction methods and return the generated data or plot the results, 
    using R2, explained variance, and RMSPE as the evaluation metrics.

    Parameters:
        X_data (DataFrame or array-like): Input data for dimensionality reduction.
        methods (list): List of dimensionality reduction methods to test.
        num_dims_range (list or range): Range of target dimensionalities to test each method against.
        save_folder (str): Path to folder where the plots will be saved.
        plot_flag (bool): Whether to plot the results (default is True).
        num_bootstrap_samples (int): Number of bootstrap samples to estimate the metrics (default is 100).
        num_cv_folds (int): Number of folds to use in cross-validation (default is 10).

    Returns:
        dict: Dictionary containing the generated data.
    """

    # Check if the folder exists, if not, create it
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Initialize dictionaries to store R2, explained variance, and RMSPE results for each method
    r2_results = {method: [] for method in methods}
    explained_variance_results = {method: [] for method in methods}
    rmspe_results = {method: [] for method in methods}

    # Set up K-Fold cross-validator with a fixed random state for reproducibility
    kf = KFold(n_splits=num_cv_folds, shuffle=True, random_state=42)
    original_dim = X_data.shape[1]

    for method in methods:
        for num_dims in num_dims_range:
            # Check if the number of dimensions selected is greater than the original data dimensions
            if num_dims >= original_dim:
                # If yes, directly set R², EVS, and RMSPE to 1 for this method and dimensionality
                r2_results[method].append(1)
                explained_variance_results[method].append(1)
                rmspe_results[method].append(1)
                continue  # Move to the next dimensionality

            # Initialize lists to store R², EVS, and RMSPE samples for bootstrap samples
            r2_samples = []
            explained_variance_samples = []
            rmspe_samples = []

            for _ in range(num_bootstrap_samples):
                r2_cv_fold_sum = 0
                explained_variance_cv_fold_sum = 0
                rmspe_cv_fold_sum = 0

                for train_index, test_index in kf.split(X_data):
                    X_train, X_test = X_data.iloc[train_index], X_data.iloc[test_index]

                    # Dimensionality reduction logic (pseudo-code)
                    reduced_X_data, reducer, inverse_reducer = reduce_dimensions_with_method(X_train, method, num_dims)
                    reduced_X_test = apply_reducer_to_data(reducer, X_test, method)
                    reconstructed_X_test = apply_inverse_reducer_to_data(inverse_reducer, reduced_X_test, method)

                    # Calculate R², EVS, and RMSPE for the current CV fold
                    r2_cv_fold_sum += r2_score(X_test, reconstructed_X_test)
                    explained_variance_cv_fold_sum += explained_variance_score(X_test, reconstructed_X_test)
                    rmspe_cv_fold_sum += rmspe(X_test, reconstructed_X_test)

                # Average R², EVS, and RMSPE over all CV folds
                r2_samples.append(r2_cv_fold_sum / num_cv_folds)
                explained_variance_samples.append(explained_variance_cv_fold_sum / num_cv_folds)
                rmspe_samples.append(rmspe_cv_fold_sum / num_cv_folds)

            # Store the average of bootstrap samples for the current method and dimensionality
            r2_results[method].append(np.mean(r2_samples))
            explained_variance_results[method].append(np.mean(explained_variance_samples))
            rmspe_results[method].append(np.mean(rmspe_samples))

    # If plot_flag is True, plot the results
    if plot_flag:
        plt.figure(figsize=(18, 6))
        plt.subplot(1, 3, 1)
        for method in methods:
            plt.plot(num_dims_range, r2_results[method], label=method)
        plt.xlabel('Number of Dimensions')
        plt.ylabel('R2 Score')
        plt.title('R2 Score vs. Number of Dimensions')
        plt.legend()
        plt.ylim((0, 1.5))

        plt.subplot(1, 3, 2)
        for method in methods:
            plt.plot(num_dims_range, explained_variance_results[method], label=method)
        plt.xlabel('Number of Dimensions')
        plt.ylabel('Explained Variance')
        plt.title('Explained Variance vs. Number of Dimensions')
        plt.legend()
        plt.ylim((0, 1.5))

        plt.subplot(1, 3, 3)
        for method in methods:
            plt.plot(num_dims_range, rmspe_results[method], label=method)
        plt.xlabel('Number of Dimensions')
        plt.ylabel('RMSPE')
        plt.title('RMSPE vs. Number of Dimensions')
        plt.legend()
        plt.tight_layout()

        plt.savefig(os.path.join(save_folder, dpi=1200))


    # Return the generated data
    return {'r2_results': r2_results, 'explained_variance_results': explained_variance_results, 'rmspe_results': rmspe_results}



# Example usage:
# Assuming you have a DataFrame named df with binary features
# X_data = pd.DataFrame(np.random.choice([0, 1], size=(500,6000)), columns=[f'Feature_{i}' for i in range(1, 6001)])
# methods_to_test = ["pca","autoencoder", "kernelpca", "sparsepca", "incrementalpca", "truncatedsvd", "gaussianrandomprojection", "sparserandomprojection", "ica"]
# max_dimensions = 6000

# test_dimensionality_reduction_methods(X_data, methods_to_test, max_dimensions,num_bootstrap_samples=100)


# Example usage:
# Assuming you have a DataFrame named df with binary features
# X_data = pd.DataFrame(np.random.choice([0, 1], size=(100, 20)), columns=[f'Feature_{i}' for i in range(1, 21)])
# method_name = "autoencoder"  # Replace with desired method name
# desired_dimensions = 20

# # Reduce and reconstruct data
# reduced_X_data, reducer, inverse_reducer = reduce_dimensions_with_method(X_data, method_name, desired_dimensions)
# reconstructed_X_data = apply_inverse_reducer_to_data(inverse_reducer, reduced_X_data, method_name)

# ## new Data
# X_data = pd.DataFrame(np.random.choice([0, 1], size=(100, 20)), columns=[f'Feature_{i}' for i in range(1, 21)])
# reduced_new = apply_reducer_to_data(reducer, reduced_X_data, method_name)

# # Display the results
# print("Original Data:")
# print(X_data.head())
# print("\nReduced Data:")
# print(reduced_X_data.head())
# print("\nReconstructed Data:")
# print(reconstructed_X_data.head())
# print("\nNew Data:")
# print(reduced_new.head())