import os
import random
import numpy as np
from pathlib import Path
from typing import List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import xgboost as xgb
import matplotlib.pyplot as plt

AMINO_ACIDS: List[str] = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_INDEX: dict = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
NUM_AA: int = len(AMINO_ACIDS)

def one_hot_encode(sequences: List[str]) -> np.ndarray:
    encoded: List[np.ndarray] = []
    for seq in sequences:
        arr = np.zeros((len(seq), NUM_AA))
        for i, aa in enumerate(seq):
            if aa in AA_TO_INDEX:
                arr[i, AA_TO_INDEX[aa]] = 1
        encoded.append(arr.flatten())
    return np.array(encoded)

def generate_test_data(
    output_folder: str,
    num_proteins: int,
    protein_length: int,
    num_key_residues: int,
    correlation_strength: float = 0.9,
    degree_of_coupling: float = 0.5,
    seed: int = 42
) -> Tuple[List[str], np.ndarray, List[int]]:
    """
    Generate synthetic protein sequences with evolutionary coupling between key residues.
    correlation_strength controls signal/noise in target.
    degree_of_coupling controls how correlated key residues are (0 = independent, 1 = fully coupled).
    """
    random.seed(seed)
    np.random.seed(seed)
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    key_positions: List[int] = sorted(random.sample(range(protein_length), num_key_residues))
    print(f"Key residue positions: {key_positions}")
    print(f"Degree of coupling: {degree_of_coupling}")

    sequences: List[str] = []
    data_values: List[float] = []

    for _ in range(num_proteins):
        seq = [random.choice(AMINO_ACIDS) for _ in range(protein_length)]

        # Generate "master" residue for first key position
        master_residue = random.choice(AMINO_ACIDS)

        # Set key residues with coupling
        for idx, pos in enumerate(key_positions):
            if idx == 0:
                seq[pos] = master_residue
            else:
                if random.random() < degree_of_coupling:
                    seq[pos] = master_residue
                else:
                    seq[pos] = random.choice(AMINO_ACIDS)

        # Target signal is sum of ASCII codes at key residues
        signal = sum(ord(seq[pos]) for pos in key_positions)
        noise = np.random.normal(0, (1 - correlation_strength) * signal)
        value = correlation_strength * signal + noise

        sequences.append("".join(seq))
        data_values.append(value)

    # Write FASTA file
    fasta_path = os.path.join(output_folder, "test_sequences.fasta")
    with open(fasta_path, "w") as f:
        for i, seq in enumerate(sequences):
            f.write(f">protein_{i}\n{seq}\n")

    # Write .dat file with just target values
    dat_path = os.path.join(output_folder, "test_data.dat")
    with open(dat_path, "w") as f:
        for val in data_values:
            f.write(f"{val:.3f}\n")


    return sequences, np.array(data_values), key_positions

def train_and_plot_model(
    sequences: List[str],
    targets: np.ndarray,
    protein_length: int,
    output_folder: str,
    key_positions: List[int]
) -> None:
    X = one_hot_encode(sequences)
    X_train, X_test, y_train, y_test = train_test_split(X, targets, test_size=0.2, random_state=42)

    model = xgb.XGBRegressor(n_estimators=100, max_depth=4, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f"Test R² score: {r2:.3f}")

    importances = model.feature_importances_
    pos_importance = np.zeros(protein_length)
    for i in range(protein_length):
        pos_importance[i] = np.sum(importances[i * NUM_AA:(i + 1) * NUM_AA])

    real_importance = np.zeros(protein_length)
    real_importance[key_positions] = 1.0

    plt.figure(figsize=(12, 4))
    plt.bar(range(protein_length), pos_importance, color='skyblue', label='Model Importance')
    plt.scatter(key_positions, real_importance[key_positions]*pos_importance.max(), color='red', s=50, label='True Key Residues')
    plt.xlabel("Residue Position")
    plt.ylabel("Importance")
    plt.title("Feature Importance by Residue Position (XGBoost) with True Key Residues")
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(output_folder, "feature_importance.png")
    plt.savefig(plot_path)
    plt.show()
    print(f"Feature importance plot saved to: {plot_path}")

    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, c='dodgerblue', alpha=0.6, edgecolors='k')
    min_val = min(min(y_test), min(y_pred))
    max_val = max(max(y_test), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label="Perfect Prediction")
    plt.xlabel("Actual Target Value")
    plt.ylabel("Predicted Target Value")
    plt.title("Predicted vs. Actual Target Values")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    pred_plot_path = os.path.join(output_folder, "predicted_vs_actual.png")
    plt.savefig(pred_plot_path)
    plt.show()
    print(f"Predicted vs Actual plot saved to: {pred_plot_path}")

if __name__ == "__main__":
    output_dir = "KRIT_Out/TestData"
    sequences, targets, key_positions = generate_test_data(
        output_folder=output_dir,
        num_proteins=1000,
        protein_length=20,
        num_key_residues=5,
        correlation_strength=0.95,  # Example: 80% correlation
        degree_of_coupling=0.7  # Example: 70% coupling
    )

    train_and_plot_model(
        sequences=sequences,
        targets=targets,
        protein_length=20,
        output_folder=output_dir,
        key_positions=key_positions
    )
