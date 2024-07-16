import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
import lightgbm as lgb
import sklearn.linear_model as sklearn_LML_Model
from sklearn.ensemble import StackingRegressor
from scipy.stats import norm 
import time 
import os
import optuna
from optuna.pruners import MedianPruner
import warnings
from sklearn.base import clone

warnings.filterwarnings('ignore')

def model_select(model_name):
    """
    Select and return the appropriate model based on user input.

    Args:
        model_name (str): Name of the model.

    Returns:
        Model: Selected model.
    """
    model_options = {
        'GradientBoost': GradientBoostingRegressor(),
        'RandomForest': RandomForestRegressor(n_estimators=600,max_depth=10),
        'DecisionTree': DecisionTreeRegressor(),
        'SVR': SVR(),
        'LightGBM': lgb.LGBMRegressor(),
        'AdaBoostR2': AdaBoostRegressor(), 
        'ARD': sklearn_LML_Model.ARDRegression(compute_score=True),
        'LASSO': sklearn_LML_Model.Lasso(),
        'BayesianRidge': sklearn_LML_Model.BayesianRidge(compute_score=True),
        'ElasticNet': sklearn_LML_Model.ElasticNet(),
    }

    if model_name in model_options:
        return model_options[model_name]
    else:
        raise ValueError(f"Invalid model_name: {model_name}. Choose from {list(model_options.keys()) + ['Stacking']}.")

import optuna

import optuna

def get_model_params(model_name):
    """
    Return hyperparameter distributions for the specified model.

    Args:
        model_name (str): Name of the model.

    Returns:
        dict: Dictionary containing hyperparameter distributions for the model.
    """
    if model_name == 'RandomForest':
        return {
            'n_estimators': optuna.distributions.IntUniformDistribution(10, 1000),
            'max_depth': optuna.distributions.IntUniformDistribution(1, 20) if model_name != 'XGBoost' else optuna.distributions.IntUniformDistribution(1, 20),
            'min_samples_split': optuna.distributions.IntUniformDistribution(2, 10),
            'min_samples_leaf': optuna.distributions.IntUniformDistribution(1, 50)
        }
    elif model_name == 'GradientBoost':
        return {
            'n_estimators': optuna.distributions.IntUniformDistribution(100, 500),
            'learning_rate': optuna.distributions.LogUniformDistribution(0.01, 0.2),
            'max_depth': optuna.distributions.IntUniformDistribution(1, 10),
            'min_samples_split': optuna.distributions.IntUniformDistribution(2, 20)
        }
    elif model_name == 'DecisionTree':
        return {
            'max_depth': optuna.distributions.IntUniformDistribution(1, 40),
            'min_samples_split': optuna.distributions.IntUniformDistribution(2, 10),
            'min_samples_leaf': optuna.distributions.IntUniformDistribution(1, 5)
        }
    elif model_name == 'SVR':
        return {
            'C': optuna.distributions.LogUniformDistribution(0.1, 10),
            'kernel': optuna.distributions.CategoricalDistribution(['rbf', 'linear'])
        }
    elif model_name == 'LightGBM':
        return {
            'n_estimators': optuna.distributions.IntUniformDistribution(100, 300),
            'learning_rate': optuna.distributions.LogUniformDistribution(0.01, 0.2),
            'max_depth': optuna.distributions.IntUniformDistribution(3, 10),
            'num_leaves': optuna.distributions.IntUniformDistribution(15, 63)
        }
    elif model_name == 'XGBoost':
        return {
            'n_estimators': optuna.distributions.IntUniformDistribution(2, 500),
            'learning_rate': optuna.distributions.LogUniformDistribution(0.0001, 0.2),
            'max_depth': optuna.distributions.IntUniformDistribution(1, 20),
            'gamma': optuna.distributions.LogUniformDistribution(0.0001, 0.2),
        }
    elif model_name == 'AdaBoostR2':
        return {
            'n_estimators': optuna.distributions.IntUniformDistribution(50, 200),
            'learning_rate': optuna.distributions.LogUniformDistribution(0.01, 0.2)
        }
    elif model_name in ['ARD', 'BayesianRidge']:
        return {
            'n_iter': optuna.distributions.IntUniformDistribution(300, 700),
            'alpha_1': optuna.distributions.LogUniformDistribution(1e-6, 1e-4),
            'alpha_2': optuna.distributions.LogUniformDistribution(1e-6, 1e-4),
        }
    elif model_name in ['LASSO', 'ElasticNet']:
        return {
            'alpha': optuna.distributions.LogUniformDistribution(0.1, 10),
            'l1_ratio': optuna.distributions.CategoricalDistribution([0.1, 0.5, 0.9])
        }
    else:
        raise ValueError(f"Invalid model_name: {model_name}. Choose from ['RandomForest', 'GradientBoost', 'DecisionTree', 'SVR', 'LightGBM', 'XGBoost', 'AdaBoostR2', 'ARD', 'LASSO', 'BayesianRidge', 'ElasticNet'].")


def objective(trial, X, y, model_name, num_folds):

    # Determine if this is the first trial
    is_first_trial = trial.number == 0

    model_base = model_select(model_name)

    # Define hyperparameter search space or use defaults
    if not is_first_trial:
        params = get_model_params(model_name)  # Retrieve hyperparameter grid for the model
        # Sample hyperparameters from the distributions or use defaults
        sampled_params = {key: trial.suggest_categorical(key, [value]) if is_first_trial
                        else trial._suggest(key, value)
                        for key, value in params.items()}
        model_base.set_params(**sampled_params)
        


    # Perform k-fold cross-validation
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    val_scores = []
    for train_index, val_index in kf.split(X):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # Train model with sampled hyperparameters
        model = clone(model_base)

        eval_set = [(X_val,y_val)]
        try:
            model.fit(X_train, y_train, early_stopping_rounds=10, eval_metric="rmse", eval_set=eval_set)
        except:
            model.fit(X_train, y_train)
        
        # Predict and calculate validation score
        predictions = model.predict(X_val)
        val_score = 1-r2_score(y_val, predictions)
        val_scores.append(val_score)

    # Average validation score across all folds
    avg_val_score = np.mean(val_scores)
    return avg_val_score

def train_model(X, y, model_name,X_full_val = None,Y_full_val=None, num_folds=10, num_trials=0):
    start_time = time.time()

    if isinstance(y, pd.DataFrame):
        y = y.values.flatten()
    elif isinstance(y, np.ndarray) and y.ndim > 1:
        y = y.flatten()

    if isinstance(X, pd.DataFrame):
        X = np.asarray(X)

    if  model_name in ['LASSO', 'ElasticNet','ARD', 'BayesianRidge']:
        num_trials = 0

    if num_trials>0:
        study = optuna.create_study(direction='minimize',pruner=optuna.pruners.HyperbandPruner())
        objective_partial = lambda trial: objective(trial, X, y, model_name, num_folds)  # Partially apply X, y, model_name, and num_folds
        study.optimize(objective_partial, n_trials=num_trials,n_jobs=-1)

        # Get best hyperparameters
        best_params = study.best_params

        # Train model with best hyperparameters
        model_base = model_select(model_name)
        model_base.set_params(**best_params)
    else:
        model_base = model_select(model_name)

    all_models = []

    # try:
    #     model_base.fit(X, y, early_stopping_rounds=10, eval_metric="rmse", eval_set=[{X_full_val,Y_full_val}])
    # except:
    try:
        model_base.fit(X, y)
    except:
        model_base.fit(X, y)

    all_models.append(model_base)

    end_time = time.time()
    training_time = end_time - start_time

    #print(f"Best hyperparameters: {best_params}")
    #print(f"Training time: {training_time:.2f} seconds")
    #print(f'fitted_{model_name}')

    return all_models, training_time



def assess_model(models, X_train, y_train, X_test, y_test):
    """
    Predict and assess the model on both training and testing datasets.

    Args:
        models (list): List of trained ML models.
        X_train (pd.DataFrame): Training features.
        y_train (pd.DataFrame): Training target variable.
        X_test (pd.DataFrame): Testing features.
        y_test (pd.DataFrame): Testing target variable.

    Returns:
        tuple: Predicted values for training, predicted values for testing, metrics DataFrame.
    """
    if isinstance(X_train, pd.DataFrame):
        X_train = np.asarray(X_train)

    if isinstance(X_test, pd.DataFrame):
        X_test = np.asarray(X_test)
    
    # Predictions for the training data
    predictions_train = np.array([model.predict(X_train) for model in models])
    y_train_predicted = np.mean(predictions_train, axis=0)

    # Predictions for the testing data
    predictions_test = np.array([model.predict(X_test) for model in models])
    y_test_predicted = np.mean(predictions_test, axis=0)

    # Calculate metrics for both training and testing sets
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_predicted))
    r2_train = r2_score(y_train, y_train_predicted)
    var_train = explained_variance_score(y_train, y_train_predicted)

    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_predicted))
    r2_test = r2_score(y_test, y_test_predicted)
    var_test = explained_variance_score(y_test, y_test_predicted)

    metrics_df = pd.DataFrame({'RMSE Train': [rmse_train], 'R2 Train': [r2_train],'EVS Train': [var_train], 'RMSE Test': [rmse_test], 'R2 Test': [r2_test],'EVS Test': [var_test]})
    
    return y_train_predicted, y_test_predicted, metrics_df

def plot_results(X_train, y_train, y_train_predicted, X_test, y_test, y_test_predicted, model_name, metrics_df, save_path):
    """
    Plot predictions vs actual values for both training and testing datasets, and histogram of residuals.

    Args:
        X_train, y_train: Training data and labels.
        y_train_predicted: Predictions on training data.
        X_test, y_test: Testing data and labels.
        y_test_predicted: Predictions on testing data.
        model_name: Name of the model for titling plots.
        metrics_df: DataFrame containing performance metrics.
        save_path: Path to save plots.

    Returns:
        None.
    """
    plt.close()
    plt.figure(figsize=(6, 6))
    plt.scatter(y_train, y_train_predicted, label='Train Predictions', color='red', zorder=3, s=5)
    plt.scatter(y_test, y_test_predicted, label='Test Predictions', color='green', zorder=3, s=5)  # Different color for test
    plt.plot(y_train, y_train, color='black', label='Ideal', zorder=2)
    plt.title(f'{model_name} Predictions vs Actual')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.tight_layout()
    plt.legend()

    # Annotate the plot with formatted RMSE and R2 values
    rmse_train = metrics_df['RMSE Train'].values[0]
    r2_train = metrics_df['R2 Train'].values[0]
    EVS_train = metrics_df['EVS Train'].values[0]
    rmse_test = metrics_df['RMSE Test'].values[0]
    r2_test = metrics_df['R2 Test'].values[0]
    EVS_test = metrics_df['EVS Test'].values[0]

    additional_info_train = f'Train: RMSE = {rmse_train:.2f}, R2 = {r2_train:.3f}, EVS = {EVS_train:.2f}'
    additional_info_test = f'Test: RMSE = {rmse_test:.2f}, R2 = {r2_test:.3f}, EVS = {EVS_test:.2f}'
    plt.annotate(additional_info_train, xy=(0.35, 0.075), xycoords='axes fraction', fontsize=10, color='red')
    plt.annotate(additional_info_test, xy=(0.35, 0.025), xycoords='axes fraction', fontsize=10, color='green')

    save_dir = os.path.join(save_path, f'{model_name}_predictions.png')
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(save_dir)
    plt.close()


    # Plot Residuals Histogram
    plt.close()
    plt.figure(figsize=(6, 6))
    residuals = y_test_predicted - y_test.values.flatten()
    plt.hist(residuals, bins=20, density=True, alpha=0.8, color='blue', label='Residuals')
    plt.title('Histogram of Residuals')
    plt.xlabel('Residuals')
    plt.ylabel('Density')
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, np.mean(residuals), np.std(residuals))
    plt.plot(x, p, 'k', linewidth=2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, model_name+'Residual_Hist.png'))
    plt.close()

def Ensemble_Modeller(X_data, y_data, model_name,X_full_val = None,Y_full_val=None, num_folds=10, num_trials=0):
    if type(model_name) == list:
        training_time = 0
        models = []
        for model in model_name:
            new_models, new_training_time = train_model(X_data, y_data, model_name,X_full_val = None,Y_full_val=None, num_folds=10, num_trials=num_trials)
            training_time+=new_training_time
            models = models + new_models
    elif model_name =="Ensemble_NonLinear":
         model_name = ['GradientBoost', 'RandomForest', 'DecisionTree','SVR','LightGBM','XGBoost','AdaBoostR2','Stacking']
         models, training_time = Ensemble_Modeller(X_data, y_data, model_name,X_full_val = None,Y_full_val=None, num_folds=10, num_trials=num_trials)
    elif model_name =="Ensemble_Linear":
        model_name = ['ARD','LASSO','BayesianRidge','ElasticNet']
        models, training_time = Ensemble_Modeller(X_data, y_data, model_name,X_full_val = None,Y_full_val=None, num_folds=10, num_trials=num_trials)
    elif model_name == "Ensemble":
        model_name = ['ARD','LASSO','BayesianRidge','ElasticNet','GradientBoost', 'RandomForest', 'DecisionTree','SVR','LightGBM','XGBoost','AdaBoostR2','Stacking']
        models, training_time = Ensemble_Modeller(X_data, y_data, model_name,X_full_val = None,Y_full_val=None, num_folds=10, num_trials=num_trials)
    else:
         models, training_time = train_model(X_data, y_data, model_name,X_full_val = None,Y_full_val=None, num_folds=10, num_trials=num_trials)
    
    return models, training_time
    
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

def custom_train_test_split(X_data, y_data, test_ratio=0.25, random_state=None):
    if np.isscalar(test_ratio):
        # Perform normal train-test split
        return train_test_split(X_data, y_data, test_size=test_ratio, random_state=random_state)
    elif isinstance(test_ratio, list):
        # Ensure data frames are aligned and clean
        data = pd.concat([X_data, y_data], axis=1)
        data = data.replace([np.inf, -np.inf], np.nan).dropna()
        X_data_clean = data.iloc[:, :-1]
        y_data_clean = data.iloc[:, -1]

        # Sort the data based on the target variable
        data_sorted = data.sort_values(by=data.columns[-1]).to_numpy()

        # Total size of the test set
        total_test_samples = int(len(data_sorted) * sum(c for _, _, c in test_ratio))

        # Initialize lists to store slices
        test_slices = []
        train_indices = set(range(len(data_sorted)))

        # Calculate slices for each range in test_ratio
        for (start_percentile, end_percentile, fraction) in test_ratio:
            lower_idx = int(len(data_sorted) * start_percentile)
            upper_idx = int(len(data_sorted) * end_percentile)
            span = upper_idx - lower_idx
            samples_in_slice = int(total_test_samples * fraction / sum(c for _, _, c in test_ratio))

            selected_indices = np.random.choice(range(lower_idx, upper_idx), size=samples_in_slice, replace=False)
            test_slices.append(data_sorted[selected_indices])
            train_indices -= set(selected_indices)
        
        # Construct test and train data
        test_data = np.vstack(test_slices)
        train_data = data_sorted[list(train_indices)]

        # Shuffle to prevent any order biases
        if random_state is not None:
            np.random.seed(random_state)
            np.random.shuffle(train_data)
            np.random.shuffle(test_data)

        # Extract features and targets
        X_train = pd.DataFrame(train_data[:, :-1], columns=X_data_clean.columns)
        X_test = pd.DataFrame(test_data[:, :-1], columns=X_data_clean.columns)
        y_train = pd.Series(train_data[:, -1])
        y_test = pd.Series(test_data[:, -1])

        return X_train, X_test, y_train, y_test
    else:
        raise ValueError("test_ratio must be a scalar or a list of tuples.")

# Example usage:
# X_train_df, X_test_df, y_train_series, y_test_series = custom_train_test_split(
#     X_data, y_data, test_ratio=[[0, 0.05, 0.05], [0.05, 0.95, 0.1], [0.95, 1.0, 0.05]], random_state=42)



def ML_SHELL(X_data, y_data, model_name, split_fraction=0.2, num_folds=5,num_trials=0, save_path=None,random_state=None):
    """
    Execute the machine learning pipeline using ensemble approach.

    Args:
        X_data (pd.DataFrame): Features.
        y_data (pd.DataFrame): Target variable.
        model_name (string): Name of the model.
        split_fraction (float): Fraction of the data to be used for testing.
        num_folds (int): Number of folds for cross-validation.
        save_path (str): Path to save the plots.

    Returns:
        dict: Dictionary containing various outputs from the pipeline.
    """
    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = custom_train_test_split(X_data, y_data, test_ratio=split_fraction,random_state=random_state)

    # Train the model on the training set
    models, training_time = Ensemble_Modeller(X_train, y_train, model_name,X_full_val = X_test,Y_full_val=y_test, num_folds=num_folds, num_trials=num_trials)

    if model_name !='Bayesian_Reg':
        # Assess the model on both training and testing sets
        y_train_predicted, y_test_predicted, metrics_df = assess_model(models, X_train, y_train, X_test, y_test)

        # Plot results
        if save_path==None:
            k=0
        else:
            plot_results(X_train, y_train, y_train_predicted, X_test, y_test, y_test_predicted, model_name, metrics_df, save_path)

        return {
            "models": models,
            "training_time": training_time,
            "y_train_predicted": y_train_predicted,
            "y_test_predicted": y_test_predicted,
            "metrics_df": metrics_df
        }
    else:
        models[0].plot_results_bayesian(X_train, y_train, X_test, y_test, model_name, save_path=save_path)
        return {
            "models": models
            }
# Example usage:
# ML_SHELL(X_data, y_data, 'BayesianRidge', num_folds=5, save_path='')
