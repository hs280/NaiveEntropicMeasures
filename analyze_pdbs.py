from Bio.PDB import PDBParser, NeighborSearch
import numpy as np
import numpy as np
from Bio import PDB
import Bin as Bin
from typing import Tuple
import pandas as pd
import os
from typing import Dict, List, Union
import math

def process_fasta(file_path: str) -> Tuple[str, str]:
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    first_seq = ""
    recording = False
    
    for line in lines:
        if line.startswith('>'):
            if recording:
                break
            recording = True
            continue
        if recording:
            first_seq += line.strip()
            
    first_seq_no_gaps = first_seq.replace('-', '')
    
    return first_seq_no_gaps, first_seq

def convert_arrays_to_lists(values_list):
    """Convert a list of numpy arrays to a list of lists with float entries."""
    return [list(map(float, arr)) for arr in values_list]

def create_annotated_lists(csv_file, length):
    """Create three lists of specified length based on CSV data."""
    df = pd.read_csv(csv_file)
    delta_elbo_list = [0] * length
    neg_delta_elbo_list = [0] * length
    abs_delta_elbo_list = [0] * length
    
    for _, row in df.iterrows():
        pos = int(row["position"]) - 1  # Convert 1-based to 0-based index
        if 0 <= pos < length:
            delta_elbo_list[pos] = row["delta_elbo"]
            neg_delta_elbo_list[pos] = -1 * row["delta_elbo"]
            abs_delta_elbo_list[pos] = row["abs_delta_elbo"]
    
    return [delta_elbo_list, neg_delta_elbo_list, abs_delta_elbo_list]

def normalize_values(values):
    """Rank the list from 0 to len(values) - 1 based on the values, handling ties."""
    # Create a sorted list of tuples (original index, value)
    values = list(values)
    sorted_values = sorted(enumerate(values), key=lambda x: x[1])
    
    # Create a ranking list with the same length as values
    ranks = [0] * len(values)
    
    # Initialize the rank counter
    rank = 0
    
    # Iterate through the sorted list and assign ranks, handling ties
    for i in range(len(values)):
        # If it's not the first value or if the current value is different from the previous one
        if i > 0 and sorted_values[i][1] != sorted_values[i-1][1]:
            rank = i  # Set rank to current index
        ranks[sorted_values[i][0]] = rank
    
    return [r/(len(ranks)-1) for r in ranks]

from Bio.PDB import PDBParser, NeighborSearch, ShrakeRupley

def get_interface_residues_with_distances(antibody_pdb, antigen_pdb, distance=5.0, filter_sasa=False, sasa_threshold=10.0):
    parser = PDBParser(QUIET=True)
    
    # Process antibody (single chain)
    antibody = parser.get_structure("antibody", antibody_pdb)
    antibody_chain = next(antibody[0].get_chains())  # Get first/only chain
    antibody_residues = list(antibody_chain.get_residues())
    
    # Process antigen (multi-chain)
    antigen = parser.get_structure("antigen", antigen_pdb)
    antigen_residues = [(res, chain.get_id()) for chain in antigen[0] for res in chain.get_residues()]
    
    # Create lookup dictionaries mapping residue objects to 0-based indices, with chain info
    ab_res_to_idx = {res: i for i, res in enumerate(antibody_residues)}
    ag_res_to_idx = {(res, chain_id): i for i, (res, chain_id) in enumerate(antigen_residues)}
    
    # Compute SASA if filtering is enabled
    if filter_sasa:
        sr = ShrakeRupley()
        sr.compute(antibody[0], level="R")  # Compute per-residue SASA
        antibody_residues = [res for res in antibody_residues if res.sasa >= sasa_threshold]
    
    # Build spatial index for antigen atoms
    antigen_atoms = [atom for res, chain_id in antigen_residues for atom in res.get_atoms()]
    ns = NeighborSearch(antigen_atoms)
    
    interface_map = {}       # antibody residue index -> sorted list of antigen residue indices
    distance_map = {}        # antibody residue index -> list of minimum distances (parallel to above)
    coord_map = {}           # antibody residue index -> list of coordinate difference info (parallel to above)
    
    for ab_res in antibody_residues:
        ab_idx = ab_res_to_idx[ab_res]  # Get 0-based index
        neighbors = set()
        contact_data = {}  # key: antigen residue index, value: list of tuples (dist, (d_x,d_y,d_z))
        
        if ab_idx == 56:
            print('wub')

        for atom in ab_res.get_atoms():
            atom_coord = atom.get_coord()
            # Search antigen atoms within the given threshold distance
            for contact_atom in ns.search(atom_coord, distance, level="A"):
                contact_res = contact_atom.get_parent()  # Get residue of the contact atom
                if contact_res in [r[0] for r in antigen_residues]:
                    # Find chain id for the contact residue
                    ag_res_chain_id = next(chain_id for res, chain_id in antigen_residues if res == contact_res)
                    ag_idx = ag_res_to_idx[(contact_res, ag_res_chain_id)]
                    neighbors.add(ag_idx)
                    diff = contact_atom.get_coord() - atom_coord
                    dist = np.linalg.norm(diff)
                    # Append the distance and vector difference tuple.
                    contact_data.setdefault(ag_idx, []).append((dist, tuple(diff)))
        
        if neighbors:
            sorted_neighbors = sorted(neighbors)
            interface_map[ab_idx] = sorted_neighbors
            
            # For each neighbor, take the contact (if multiple) with the minimum distance.
            distance_map[ab_idx] = []
            coord_map[ab_idx] = []
            for ag_idx in sorted_neighbors:
                best_contact = min(contact_data[ag_idx], key=lambda x: x[0])
                distance_map[ab_idx].append(best_contact[0])
                coord_map[ab_idx].append(best_contact)  # (distance, (d_x, d_y, d_z))
    
    sorted_indices = sorted(interface_map.keys())
    
    antibody_indices = sorted_indices
    antigen_neighbors = [interface_map[idx] for idx in sorted_indices]
    distances = [distance_map[idx] for idx in sorted_indices]
    coord_neighbors = [coord_map[idx] for idx in sorted_indices]
    
    return antibody_indices, antigen_neighbors, distances, coord_neighbors

import numpy as np
from Bio.PDB import PDBParser, Polypeptide

from scipy.optimize import minimize

def calculate_max_binding_energy_with_flexibility(
    antibody_pdb: str,
    antigen_pdb: str,
    antibody_interface: List[int],
    antigen_neighbors: List[List[int]],
    distances: List[List[float]],
    coord_neighbors: List[List[Tuple[float, Tuple[float, float, float]]]],
    flex_threshold: float = 4.0
) -> Dict[int, Tuple[float, str, Tuple[float, float, float]]]:
    """
    For each antibody interface residue, determine the optimal candidate amino acid and a small 
    displacement vector (δₓ, δᵧ, δ_z) (with norm ≤ flex_threshold) that minimizes the total binding energy.
    The energy for each contact is computed using an adjusted distance:
        new_distance = norm(original_vector + δ)
    and then summed over all antigen contacts for that residue.
    
    Instead of grid search, we use an optimization algorithm (SLSQP) to search over the displacement vector.
    
    Args:
        antibody_pdb (str): Path to antibody PDB file.
        antigen_pdb (str): Path to antigen PDB file.
        antibody_interface (list): List of antibody residue indices at the interface.
        antigen_neighbors (list of lists): For each antibody residue, a list of antigen residue indices in contact.
        distances (list of lists): For each antibody residue, a list of minimum distances for each antigen contact.
        coord_neighbors (list of lists): For each antibody residue, a list of tuples 
                                          (min_distance, (d_x, d_y, d_z)) for each antigen neighbor contact.
        flex_threshold (float): Maximum allowed displacement (Å). Default is 3.0.
    
    Returns:
        dict: Mapping from antibody residue index (int) to a tuple:
              (min_energy (float), best_candidate (str), best_delta (tuple of 3 floats))
    """
    # Energy contribution estimates (kcal/mol)
    energy_estimates = {
        'hydrogen_bond': -4.0,
        'salt_bridge':   -3.0,
        'hydrophobic':   -1.0,
        'van_der_waals': -0.5
    }
    
    # Amino acid properties
    aa_properties = {
        'polar':       ['SER', 'THR', 'ASN', 'GLN'],
        'charged':     ['ARG', 'LYS', 'ASP', 'GLU', 'HIS'],
        'hydrophobic': ['ALA', 'VAL', 'ILE', 'LEU', 'MET', 'PHE', 'TRP', 'PRO'],
        'special':     ['CYS', 'GLY', 'TYR']
    }
    
    parser = PDBParser(QUIET=True)
    antibody_structure = parser.get_structure("antibody", antibody_pdb)
    antigen_structure  = parser.get_structure("antigen", antigen_pdb)
    
    # Assume antibody has one chain and antigen may have multiple chains.
    antibody_residues = list(next(antibody_structure.get_chains()).get_residues())
    antigen_residues = [res for chain in antigen_structure.get_chains() for res in chain.get_residues()]
    
    max_energy_contributions = {}
    
    # For each antibody interface residue:
    for i, ab_idx in enumerate(antibody_interface):
        original_residue = antibody_residues[ab_idx]
        best_energy = 0.0  # energies are negative, so lower is better
        best_candidate = original_residue.get_resname()
        best_delta = (0.0, 0.0, 0.0)
        
        # Prepare a list of contacts for this antibody residue.
        # Each contact is a tuple: (orig_vector, antigen_residue_name)
        contacts = []
        for j, ag_idx in enumerate(antigen_neighbors[i]):
            antigen_residue = antigen_residues[ag_idx]
            ag_res_name = antigen_residue.get_resname()
            # Use the coordinate difference vector from coord_neighbors
            orig_vec = np.array(coord_neighbors[i][j][1])
            contacts.append((orig_vec, ag_res_name))
        
        # For each candidate amino acid (3-letter code)
        for candidate in Polypeptide.aa3:
            # Define an objective function: given a displacement delta, return total energy.
            def objective(delta: np.ndarray) -> float:
                total_energy = 0.0
                for orig_vec, ag_res_name in contacts:
                    new_vec = orig_vec + delta  # adjusted contact vector
                    new_distance = np.linalg.norm(new_vec)
                    # Calculate energy for this contact using the candidate amino acid.
                    total_energy += calculate_residue_pair_energy(candidate, ag_res_name, new_distance, energy_estimates, aa_properties,new_vec)
                return total_energy
            
            # Constraint: norm(delta) <= flex_threshold
            cons = ({'type': 'ineq', 'fun': lambda delta: flex_threshold - np.linalg.norm(delta)})
            # Initial guess: no displacement.
            delta0 = np.array([0.0, 0.0, 0.0])
            # Run the optimization using SLSQP.
            res = minimize(objective, delta0, method='SLSQP', constraints=cons)
            if res.success:
                opt_delta = res.x
                total_energy = res.fun
            else:
                # If optimization fails, fallback to no displacement.
                opt_delta = delta0
                total_energy = objective(delta0)
            
            # Update best candidate if this candidate yields lower energy.
            if total_energy < best_energy:
                best_energy = total_energy
                best_candidate = candidate
                best_delta = tuple(opt_delta.tolist())
        
        max_energy_contributions[ab_idx] = (best_energy, best_candidate)
    
    return max_energy_contributions

# --- Provided Helper Functions ---

def exponential_distance_decay(distance: float, min_dist: float, max_dist: float, alpha: float = 3.0) -> float:
    """
    Compute a decay factor using an exponential function based on the distance relative 
    to an optimal interaction range.
    
    Args:
        distance (float): Actual distance between residues in Å.
        min_dist (float): Minimum optimal distance.
        max_dist (float): Maximum optimal distance.
        alpha (float): Decay constant controlling steepness (default: 3.0).
    
    Returns:
        float: Decay factor.
    """
    if distance < min_dist:
        return 1.0
    elif distance > max_dist:
        return 0.0
    else:
        normalized = (distance - min_dist) / (max_dist - min_dist)
        return math.exp(-alpha * normalized)

import math
from typing import Dict, List, Tuple

def calculate_residue_pair_energy(
    ab_residue: str, 
    ag_residue: str, 
    distance: float, 
    energy_estimates: Dict[str, float], 
    aa_properties: Dict[str, List[str]],
    bond_vector: Tuple[float, float, float]
) -> float:
    """
    Calculate the energy contribution between a pair of residues using an improved decay model
    that includes both attractive and repulsive effects. In this example, the van der Waals 
    interactions are modeled using a Lennard-Jones potential, and the hydrogen bonds incorporate
    an angular factor based on the bond_vector.
    
    Args:
        ab_residue (str): 3-letter code of the candidate antibody residue.
        ag_residue (str): 3-letter code of the antigen residue.
        distance (float): Distance between the residues in Å.
        energy_estimates (dict): Energy contributions for different interaction types.
        aa_properties (dict): Amino acid properties.
        bond_vector (tuple): The contact bond vector (d_x, d_y, d_z).
        
    Returns:
        float: Energy contribution (kcal/mol).
    """
    energy = 0.0
    # Define typical distance ranges for different interactions.
    bond_distances = {
        'hydrogen_bond': (1.5, 3.5),
        'salt_bridge':   (2.0, 5.5),
        'hydrophobic':   (2.5, 5.0),
        'van_der_waals': (3.0, 6.0)
    }
    
    polar_charged = set(aa_properties.get('polar', []) + aa_properties.get('charged', []))
    hydrophobic = set(aa_properties.get('hydrophobic', []))
    
    # --- Helper Functions ---
    def exponential_distance_decay(d: float, d_min: float, d_max: float, alpha: float = 3.0) -> float:
        """Return a decay factor based on the distance using an exponential function."""
        if d < d_min:
            return 1.0
        elif d > d_max:
            return 0.0
        else:
            normalized = (d - d_min) / (d_max - d_min)
            return math.exp(-alpha * normalized)
    
    def angle_between(v1: Tuple[float, float, float], v2: Tuple[float, float, float]) -> float:
        """Return the angle (in degrees) between two vectors."""
        dot = sum(a * b for a, b in zip(v1, v2))
        norm1 = math.sqrt(sum(a * a for a in v1))
        norm2 = math.sqrt(sum(b * b for b in v2))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        cos_angle = dot / (norm1 * norm2)
        cos_angle = max(min(cos_angle, 1.0), -1.0)
        return math.degrees(math.acos(cos_angle))
    
    def lennard_jones_potential(r: float, epsilon: float, sigma: float) -> float:
        """
        Compute the Lennard-Jones potential which includes both repulsive and attractive terms.
        
        Args:
            r (float): Distance between atoms.
            epsilon (float): Depth of the potential well.
            sigma (float): Finite distance at which the inter-particle potential is zero.
            
        Returns:
            float: Lennard-Jones energy.
        """
        term = sigma / r
        return 4 * epsilon * (term**12 - term**6)
    
    # --- Angular Dependence for Hydrogen Bonds ---
    # Define an ideal hydrogen bond direction (e.g., along the z-axis).
    ideal_hbond_vector = (0.0, 0.0, 1.0)
    angle_deviation = angle_between(bond_vector, ideal_hbond_vector)
    # Gaussian decay for angular deviation (tolerance ~30°).
    angle_decay = math.exp(- (angle_deviation ** 2) / (2 * (30.0 ** 2)))
    
    # --- Energy Calculations ---
    # For polar interactions: consider salt bridges or hydrogen bonds.
    if ab_residue in polar_charged and ag_residue in polar_charged:
        # Salt bridge for charged residues.
        if ab_residue in aa_properties.get('charged', []) and ag_residue in aa_properties.get('charged', []):
            min_d, max_d = bond_distances['salt_bridge']
            if min_d <= distance <= max_d:
                energy += energy_estimates.get('salt_bridge', 0.0) * exponential_distance_decay(distance, min_d, max_d)
        else:
            # Hydrogen bond with angular dependence.
            min_d, max_d = bond_distances['hydrogen_bond']
            if min_d <= distance <= max_d:
                energy += (energy_estimates.get('hydrogen_bond', 0.0) *
                           exponential_distance_decay(distance, min_d, max_d) *
                           angle_decay)
    # Hydrophobic interactions.
    elif ab_residue in hydrophobic and ag_residue in hydrophobic:
        min_d, max_d = bond_distances['hydrophobic']
        if min_d <= distance <= max_d:
            energy += energy_estimates.get('hydrophobic', 0.0) * exponential_distance_decay(distance, min_d, max_d)
    
    # Van der Waals interactions: use Lennard-Jones potential to capture both attraction and repulsion.
    min_d, max_d = bond_distances['van_der_waals']
    if min_d <= distance <= max_d:
        # Example parameters (adjust epsilon and sigma as needed):
        epsilon = 0.2  # Depth of the potential well (kcal/mol)
        sigma = 3.5    # Distance at which the potential is zero (Å)
        energy += energy_estimates.get('van_der_waals', 0.0) * lennard_jones_potential(distance, epsilon, sigma)
    
    return energy

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import precision_score, recall_score, f1_score

def overlap_analysis(algorithm_importances, interface_residues):
    """
    Calculate the ratio of importance at interface residues to total importance.
    
    Args:
    algorithm_importances (list): List of importance scores for each residue
    interface_residues (list): List of indices of interface residues
    
    Returns:
    float: Ratio of importance at interface to total importance
    """
    interface_importance = sum(algorithm_importances[i] for i in interface_residues)
    total_importance = sum(algorithm_importances)
    return interface_importance / total_importance if total_importance > 0 else 0

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import numpy as np

import xgboost as xgb
from sklearn.metrics import r2_score

def correlation_test(algorithm_importances, binding_energy_indices, binding_energies, n_permutations=100):
    """
    Perform a nonlinear correlation test using XGBoost with Box-Cox transformation
    for proper AIC calculation on non-Gaussian data.
    
    Args:
        algorithm_importances (list): List of importance scores for each residue.
        binding_energy_indices (list): List of indices for binding energies.
        binding_energies (list): List of binding energy values.
        n_permutations (int): Number of permutations for significance testing.
    
    Returns:
        tuple: (R² score, p-value, AIC, lambda)
            - R² score: Proportion of variance explained.
            - p-value: Fraction of times permuted R² scores exceed the observed R².
            - AIC: Akaike Information Criterion after Box-Cox transformation.
            - lambda: The Box-Cox transformation parameter.
    """
    import numpy as np
    import xgboost as xgb
    from sklearn.metrics import r2_score
    from scipy import stats
    
    # Extract the relevant importance scores and reshape for XGBR
    importances = np.array([[algorithm_importances[i]] for i in binding_energy_indices])
    binding_energies_array = np.array(binding_energies)
    
    # Ensure data is positive for Box-Cox transformation
    if np.min(binding_energies_array) <= 0:
        # Shift data to positive values if needed
        shift = abs(np.min(binding_energies_array)) + 1e-6
        binding_energies_positive = binding_energies_array + shift
    else:
        binding_energies_positive = binding_energies_array
        shift = 0
    
    # Apply Box-Cox transformation to normalize the response variable
    transformed_binding, lambda_value = stats.boxcox(binding_energies_positive)
    
    # Train XGBoost Regressor on the transformed data
    xgbr = xgb.XGBRegressor(n_estimators=20, max_depth=3, random_state=42)
    xgbr.fit(importances, transformed_binding)
    
    # Predict on the training set (in transformed space)
    predictions_transformed = xgbr.predict(importances)
    
    # Transform predictions back to original scale for R² calculation
    if lambda_value == 0:
        predictions_original = np.exp(predictions_transformed) - shift
    else:
        predictions_original = (lambda_value * predictions_transformed + 1)**(1/lambda_value) - shift
    
    # Compute the original R² score on untransformed data
    r2_original = r2_score(binding_energies_array, predictions_original)
    
    # Calculate AIC in the transformed space where normality assumption is valid
    residuals_transformed = transformed_binding - predictions_transformed
    RSS_transformed = np.sum(residuals_transformed ** 2)
    n = len(binding_energies_array)
    
    # Calculate effective parameters - simplified approach
    # Each tree contributes approximately log2(n) effective parameters
    n_trees = xgbr.get_params()['n_estimators']
    max_depth = xgbr.get_params()['max_depth']
    k = n_trees * max_depth  # Simplified parameter count
    
    # Add +1 for the Box-Cox transformation parameter lambda
    k += 1
    
    # Calculate AIC in transformed space
    sigma_squared = RSS_transformed / n
    log_likelihood = -n/2 * np.log(2 * np.pi * sigma_squared) - RSS_transformed/(2 * sigma_squared)
    
    # Add Jacobian correction for the transformation
    # This accounts for the change of variables in the probability density
    if lambda_value == 0:
        log_jacobian = np.sum(np.log(binding_energies_positive))
    else:
        log_jacobian = (lambda_value - 1) * np.sum(np.log(binding_energies_positive))
    
    # Adjusted log-likelihood with Jacobian correction
    adjusted_log_likelihood = log_likelihood + log_jacobian
    
    # Calculate AIC
    AIC = -2 * adjusted_log_likelihood + 2 * k
    
    # Permutation testing (in original space)
    permuted_r2_scores = []
    
    for _ in range(n_permutations):
        # Create a permuted copy of the binding energies
        shuffled_binding = np.random.permutation(binding_energies_array)
        
        # Ensure positivity for Box-Cox
        if np.min(shuffled_binding) <= 0:
            shuffled_binding_positive = shuffled_binding + abs(np.min(shuffled_binding)) + 1e-6
        else:
            shuffled_binding_positive = shuffled_binding
        
        # Apply same Box-Cox transformation (using same lambda)
        if lambda_value == 0:
            shuffled_transformed = np.log(shuffled_binding_positive)
        else:
            shuffled_transformed = (shuffled_binding_positive**lambda_value - 1) / lambda_value
        
        # Create a new model for each permutation
        xgbr_perm = xgb.XGBRegressor(n_estimators=10, max_depth=3, random_state=42)
        xgbr_perm.fit(importances, shuffled_transformed)
        
        # Predict and transform back to original space
        perm_pred_transformed = xgbr_perm.predict(importances)
        
        if lambda_value == 0:
            perm_pred_original = np.exp(perm_pred_transformed)
            if shift > 0:
                perm_pred_original -= shift
        else:
            perm_pred_original = (lambda_value * perm_pred_transformed + 1)**(1/lambda_value)
            if shift > 0:
                perm_pred_original -= shift
        
        # Calculate R² on original scale
        permuted_r2_scores.append(r2_score(shuffled_binding, perm_pred_original))
    
    # Compute p-value
    p_value = np.mean(np.array(permuted_r2_scores) >= r2_original)
    
    return r2_original, p_value, AIC

def plot_results(algorithm_names, algorithm_importances_list, interface_residues, binding_energy_indices, binding_energies):
    # Overlap analysis
    overlap_results = [overlap_analysis(imp, interface_residues) for imp in algorithm_importances_list]
    
    # Bar plot for overlap analysis
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(algorithm_names))
    
    ax.bar(x, overlap_results, color='#377eb8')
    
    ax.set_ylabel('Interface Importance Ratio', fontsize=14)
    ax.set_title('Overlap Analysis Results', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(algorithm_names, rotation=45, ha='right', fontsize=12)
    ax.set_ylim([0, 1])
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    #plt.show()

    # Scatter plots
    n_rows, n_cols = 3, 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 15), sharex=True, sharey=True)
    axes = axes.flatten()

    for i, (ax, alg_name, importances) in enumerate(zip(axes, algorithm_names, algorithm_importances_list)):
        importances = [imp-min(importances) for imp in importances]
        importances = [imp/max(importances) for imp in importances]
        x_values = [energy for energy in binding_energies]
        y_values = [importances[idx] for idx in binding_energy_indices]
        
        ax.scatter(x_values, y_values, marker='x', s=100, alpha=0.8, color='black')
        ax.set_ylim(0, 1)
        ax.set_ylabel(alg_name, fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=10)
    
    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.supxlabel("Bond Energy", fontsize=14)
    fig.supylabel("Importance", fontsize=14)
    plt.tight_layout()
    #plt.show()

    plot_correlation_results(algorithm_names, algorithm_importances_list, binding_energy_indices, binding_energies, output_path='./correlation_results_r2_AIC.png')


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_correlation_results(algorithm_names, algorithm_importances_list, binding_energy_indices, binding_energies, output_path=None):
    """
    Performs correlation tests for each algorithm using the provided correlation_test function,
    prints the results, and then plots a dual-axis bar chart:
      - Left axis: R² values (square of Pearson correlation coefficient).
      - Right axis: AIC values.
      
    The plot adopts a Nature-style similar to plot_nature_style_bar.
    
    Parameters:
        algorithm_names (list): List of algorithm names.
        algorithm_importances_list (list): List of importance arrays corresponding to each algorithm.
        binding_energy_indices (list/array): Indices for the binding energies.
        binding_energies (list/array): Binding energy values.
        output_path (str, optional): If provided, the plot is saved to this path; otherwise, it is shown.
    """
    # Define hatch patterns for distinct shading for each triplet
    hatch_patterns = ['x', '+', '#']  # Diagonal, Backslash, Crosshatch

    # Lists to store computed R² and AIC values
    r2_values = []
    aic_values = []
    
    # Run the correlation test for each algorithm and print results.
    # (Assumes correlation_test returns: (corr_coeff, p_value, AIC))
    for alg_name, importances in zip(algorithm_names, algorithm_importances_list):
        corr_coeff, p_value, AIC = correlation_test(importances, binding_energy_indices, binding_energies)
        print(f"\nCorrelation Test Results for {alg_name}:")
        print(f"Pearson Correlation Coefficient: {corr_coeff:.2f}")
        print(f"P-value: {p_value:.4f}")
        print(f"AIC: {AIC:.4f}")
        
        # Compute R² from the correlation coefficient
        r2 = corr_coeff
        r2_values.append(r2)
        aic_values.append(AIC)
    
    # Create a DataFrame for easier plotting
    df = pd.DataFrame({
        'Algorithm': algorithm_names,
        'R2': r2_values,
        'AIC': aic_values
    })
    
    # Set up a Nature-style plot
    sns.set_style("whitegrid")
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax2 = ax1.twinx()  # Create a twin axis for AIC
    
    # Define positions and bar width for side-by-side bars
    x = np.arange(len(algorithm_names))
    width = 0.4
    
    # Define colors: here, light blue for R² and red for AIC
    r2_color = '#0000FF'
    aic_color = '#ff000d'
    
    # Plot R² values on the left y-axis with different hatch patterns
    bars1 = []
    for i in range(len(algorithm_names)):
        bar = ax1.bar(x[i] - width/2, df['R2'][i], width, label='R²', color=r2_color, hatch=hatch_patterns[i // 3])
        bars1.append(bar)

    # Plot AIC values on the right y-axis with different hatch patterns
    bars2 = []
    for i in range(len(algorithm_names)):
        bar = ax2.bar(x[i] + width/2, df['AIC'][i], width, label='AIC', color=aic_color, hatch=hatch_patterns[i // 3])
        bars2.append(bar)
    
    # Set x-axis labels and tick positions
    ax1.set_xlabel("Algorithm", fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(algorithm_names, fontsize=12, rotation=45, ha='right')
    
    # Set y-axis labels
    ax1.set_ylabel("R²", fontsize=14, color=r2_color)
    ax2.set_ylabel("AIC", fontsize=14, color=aic_color)
    
    # Optionally, annotate each bar with its height
    def autolabel(bars, ax, fmt="{:.2f}"):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(fmt.format(height),
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # offset in points
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)
                        
    #autolabel(bars1, ax1)
    #autolabel(bars2, ax2)
    
    # Create a combined legend; here, we manually specify labels
    lines = [bars1, bars2]
    labels = ['R²', 'AIC']
    #ax1.legend(lines, labels, loc='upper left', fontsize=12)
    
    # Remove extra spines to mimic Nature-style aesthetics
    sns.despine(ax=ax1)
    sns.despine(ax=ax2, left=True)
    ax1.yaxis.grid(False)
    ax2.yaxis.grid(False)
    ax2.set_ylim(250, 350)  # Set y-axis limits for AIC
    ax2.set_yticks([250, 300, 350])  # Set exactly 3 yticks

    
    fig.tight_layout()
    
    # Save or display the plot
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()


def rescale_list(list: List[float]):
    """
    Rescale a list of values to the range [0, 1].
    
    Args:
    list (list): List of numerical values.
    
    Returns:
    list: Rescaled list of values.
    """
    min_val = min(list)
    max_val = max(list)
    return [(val - min_val) / (max_val - min_val) for val in list]

def count_resids(data: List[List[float]],
                 threshold: float = 0.1) -> List[int]:
    """
    Count the number of residues with importance above a given threshold for each algorithm.    
    """

    return [sum(1 for imp in rescale_list(imp_list) if imp > threshold) for imp_list in data]

def get_indices_greater_than_threshold(lst, threshold=0.1):
    """
    Apply Min-Max scaling to the list and return indices based on the specified threshold.
    
    If threshold is a number greater than 1, it is interpreted as the number of top residues to return.
    Otherwise, indices where the scaled values exceed the threshold are returned.
    
    Parameters:
        lst (list): The input list of numeric values.
        threshold (float): If <= 1, the scaled value threshold; if > 1, the number of top residues to select.
    
    Returns:
        list: A list of indices satisfying the condition.
    """
    if len(lst) == 0:
        return []
    
    min_val = min(lst)
    max_val = max(lst)
    
    # Handle the case where all values are the same
    if max_val - min_val == 0:
        if threshold > 1:
            n = int(threshold)
            return list(range(len(lst)))[:n]
        else:
            return [i for i, value in enumerate(lst) if value > threshold]
    
    # Min-Max scaling
    scaled_lst = [(value - min_val) / (max_val - min_val) for value in lst]
    
    if threshold > 1:
        # Return indices of the top n scaled values
        n = int(threshold)
        sorted_indices = sorted(range(len(scaled_lst)), key=lambda i: scaled_lst[i], reverse=True)
        return sorted_indices[:n]
    else:
        # Return indices where scaled value exceeds the threshold
        return [i for i, value in enumerate(scaled_lst) if value > threshold]

def count_members(list1, list2):
    """
    Counts how many elements from list1 are present in list2.

    Parameters:
        list1 (list): The first list whose elements are to be counted in list2.
        list2 (list): The second list where membership is checked.

    Returns:
        int: The number of elements in list1 that are in list2.
    """
    return len([item for item in list1 if item in list2])

def generate_residue_tables(data: List[List[float]], 
                            threshold: float = 0.1, 
                            algorithm_names: List[str] = None,
                            alg_flag: List[bool] = None,
                            interface_residues: List[int] = None,
                            binding_energy_indices: List[int] = None,
                            binding_energies: List[float] = None,
                            save_path: str = None) -> pd.DataFrame:
    """
    Generate a table of residue importance values for each algorithm,
    with additional columns for interface residues, binding energies,
    and the number of residues above the threshold. Additionally, compute
    the full lists of overlapping indices from each algorithm and perform
    a sensitivity analysis of the threshold.
    
    If 'threshold' is < 1, the sensitivity analysis is conducted for thresholds
    between 0 and the given value. If 'threshold' is > 1, it is treated as a top-n
    selection and the analysis is performed for integer thresholds from 1 to the given value.
    
    For each algorithm flagged True in 'alg_flag' (a list of booleans of the same length 
    as algorithm_names), the sensitivity analysis produces two plots:
      - Binding Overlap Count vs Threshold
      - Interface Overlap Count vs Threshold
      
    Parameters:
        data: List of lists of importance values for each algorithm.
        threshold: Threshold to select key residues. If <= 1, it is a fraction; if > 1, it indicates top-n residues.
        algorithm_names: Optional names for the algorithms.
        alg_flag: List of booleans indicating whether to include the corresponding algorithm in the sensitivity plots.
        interface_residues: List of indices that are considered interface residues.
        binding_energy_indices: List of residue indices corresponding to binding energies.
        binding_energies: List of binding energy values.
        save_path: Optional file path to save the resulting DataFrame as CSV.
        
    Returns:
        A pandas DataFrame with the following columns:
          - Algorithm: Name of the algorithm.
          - Total_Chosen_Count: Total number of residues with importance above the threshold.
          - Binding_Overlap_Count: Count of residues overlapping with reduced binding indices.
          - Interface_Overlap_Count: Count of residues overlapping with interface residues.
          - Binding_Overlap_Indices: List of overlapping indices with reduced binding.
          - Interface_Overlap_Indices: List of overlapping indices with interface residues.
          - Chosen_Indices: List of indices with importance above the threshold.
    """
    # Filter binding energy indices for those with binding energies < 0 (reduced binding)
    reduced_binding = [idx for idx, value in zip(binding_energy_indices, binding_energies) if value < 0]
    
    binding_overlap_count = []
    interface_overlap_count = []
    binding_overlap_indices_list = []
    interface_overlap_indices_list = []
    chosen_indices_list = []
    total_chosen_count = []
    
    # Iterate over each algorithm's importance values for the given threshold
    for importance_values in data:
        # Get indices with importance above the threshold.
        chosen_indices = get_indices_greater_than_threshold(importance_values, threshold=threshold)
        chosen_indices_list.append(chosen_indices)
        total_chosen_count.append(len(chosen_indices))
        
        # Compute overlapping indices with reduced binding and interface residues.
        binding_overlap_indices = [idx for idx in chosen_indices if idx in reduced_binding]
        interface_overlap_indices = [idx for idx in chosen_indices if idx in interface_residues]
        
        binding_overlap_indices_list.append(binding_overlap_indices)
        interface_overlap_indices_list.append(interface_overlap_indices)
        
        binding_overlap_count.append(len(binding_overlap_indices))
        interface_overlap_count.append(len(interface_overlap_indices))
    
    # Construct the DataFrame.
    df = pd.DataFrame({
        "Algorithm": algorithm_names if algorithm_names is not None else [f"Algorithm_{i}" for i in range(len(data))],
        "Total_Chosen_Count": total_chosen_count,
        "Binding_Overlap_Count": binding_overlap_count,
        "Interface_Overlap_Count": interface_overlap_count,
        "Binding_Overlap_Indices": binding_overlap_indices_list,
        "Interface_Overlap_Indices": interface_overlap_indices_list,
        "Chosen_Indices": chosen_indices_list
    })
    
    # Save the DataFrame to CSV if a save path is provided.
    if save_path:
        df.to_csv(save_path, index=False)
    
    # --- Sensitivity Analysis Plots ---
    # Only perform plots if alg_flag is provided and at least one algorithm is flagged True.
    if alg_flag is not None and any(alg_flag):
        # Define the threshold range for sensitivity analysis.
        if threshold > 1:
            # For top-n selection, use integer thresholds from 1 to the given threshold.
            threshold_range = list(range(1, int(threshold) + 1))
        else:
            # For fractional threshold, generate 10 evenly spaced values between 0 and the given threshold.
            threshold_range = np.linspace(0, threshold, 10)
        
        # Prepare dictionaries to hold sensitivity results for each algorithm.
        sensitivity_binding = {alg: [] for alg, flag in zip(algorithm_names, alg_flag) if flag}
        sensitivity_interface = {alg: [] for alg, flag in zip(algorithm_names, alg_flag) if flag}
        
        # For each algorithm that is flagged for plotting:
        for i, (importance_values, flag) in enumerate(zip(data, alg_flag)):
            if not flag:
                continue
            alg_name = algorithm_names[i] if algorithm_names is not None else f"Algorithm_{i}"
            for th in threshold_range:
                chosen = get_indices_greater_than_threshold(importance_values, threshold=th)
                bind_overlap = len([idx for idx in chosen if idx in reduced_binding])
                interf_overlap = len([idx for idx in chosen if idx in interface_residues])
                sensitivity_binding[alg_name].append(bind_overlap)
                sensitivity_interface[alg_name].append(interf_overlap)
        

        # Increase default font size by 2pt
        base_fontsize = plt.rcParams.get('font.size', 12)
        new_fontsize = base_fontsize + 4
        plt.rcParams.update({'font.size': new_fontsize})
        
        # Plot: Binding Overlap Count vs Threshold (square aspect ratio, no markers)
        fig1, ax1 = plt.subplots(figsize=(5, 5))
        for alg, counts in sensitivity_binding.items():
            ax1.plot(threshold_range, counts, linewidth=2, label=alg)  # no markers
        ax1.set_xlabel("Threshold", fontsize=new_fontsize)
        ax1.set_ylabel("Binding Overlap Count", fontsize=new_fontsize)
        box = ax1.get_position()
        ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        legend1 = ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1, fontsize=new_fontsize)    
        # Remove top and right spines for a clean look.
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        # Set square aspect ratio.
        # Save the plot
        fig1.savefig("binding_overlap_sensitivity.png", dpi=600, bbox_inches='tight')
        #plt.show()
        
        # Plot: Interface Overlap Count vs Threshold (square aspect ratio, no markers)
        fig2, ax2 = plt.subplots(figsize=(5, 5))
        for alg, counts in sensitivity_interface.items():
            ax2.plot(threshold_range, counts, linewidth=2, label=alg)  # no markers
        ax2.set_xlabel("Threshold", fontsize=new_fontsize)
        ax2.set_ylabel("Interface Overlap Count", fontsize=new_fontsize)
        box = ax2.get_position()
        ax2.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        legend1 = ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1, fontsize=new_fontsize)    
        # Remove top and right spines.
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        # Save the plot
        fig2.savefig("interface_overlap_sensitivity.png", dpi=600, bbox_inches='tight')
        #plt.show()
    
    return df

from Bio.PDB import PDBParser, PDBIO, Select

def color_residues_by_indices(input_pdb, output_pdb, first_list_indices, second_list_indices, sequence=None):
    """
    Colors atoms in a PDB based on two lists of residue indices.
    
    Args:
        input_pdb (str): Path to input PDB file
        output_pdb (str): Path to save the modified PDB file
        first_list_indices (list): List of residue indices to color with B-factor 0.5 (purple)
        second_list_indices (list): List of residue indices to color with B-factor 1.0
        sequence (str, optional): Amino acid sequence for validation, if provided
    """
    # Parse the PDB file
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", input_pdb)
    
    # Convert indices to sets for faster lookup
    first_set = set(first_list_indices)
    second_set = set(second_list_indices)
    
    # Find overlapping indices
    overlap = first_set.intersection(second_set)
    if overlap:
        print(f"{len(overlap)} residues appear in both lists: {sorted(overlap)}")
    
    # Process each model, chain, residue, atom
    for model in structure:
        for chain in model:
            # Track residues actually colored
            colored_first = set()
            colored_second = set()
            
            for residue in chain:
                # Skip non-amino acid residues (like waters, ligands)
                if residue.id[0] != " ":
                    continue
                
                # Get residue number
                res_num = residue.id[1] - 1
                
                # Check if this residue index is in our lists
                if res_num in first_set:
                    # Apply b-factor 0.5 to all atoms in this residue
                    for atom in residue:
                        atom.set_bfactor(0.5)
                    colored_first.add(res_num)
                
                if res_num in second_set:
                    # Apply b-factor 1.0 to all atoms in this residue
                    for atom in residue:
                        atom.set_bfactor(1.0)
                    colored_second.add(res_num)
    
    # Save the modified PDB
    io = PDBIO()
    io.set_structure(structure)
    io.save(output_pdb)
    
    # Report on coloring results
    print(f"Modified PDB saved to {output_pdb}")
    print(f"First list (purple, b-factor=0.5): {len(colored_first)} of {len(first_set)} residues colored")
    if len(colored_first) < len(first_set):
        missing = first_set - colored_first
        print(f"  Missing residues from first list: {sorted(missing)}")
    
    print(f"Second list (b-factor=1.0): {len(colored_second)} of {len(second_set)} residues colored")
    if len(colored_second) < len(second_set):
        missing = second_set - colored_second
        print(f"  Missing residues from second list: {sorted(missing)}")


# Example usage
if __name__ == "__main__":
    antibody_pdb = "Data/Antibodies/docked_antibody.pdb"
    antigen_pdb = "Data/Antibodies/docked_antigen.pdb"

    # Example usage:
    antibody_interface, antigen_neighbors, distances,coord_distances  = get_interface_residues_with_distances(antibody_pdb,
                                                                            antigen_pdb, distance=10,
                                                                            filter_sasa=True)
    print(f"Antibody interface residues (0-based): {antibody_interface}")
    print(f"Corresponding antigen neighbors: {antigen_neighbors}")
    print(f"Distances (Å): {distances}")

    # Calculate max binding energies
    energy_contributions = calculate_max_binding_energy_with_flexibility(antibody_pdb, antigen_pdb, 
                                                                         antibody_interface, antigen_neighbors, distances,coord_distances)

    for res, (energy, aa) in energy_contributions.items():
        print(f"Residue {res}: Max binding energy contribution = {energy:.2f} kcal/mol (Amino acid: {aa})")

    input_pdb = antibody_pdb
    sequence_file_path ="./Data/Antibodies/sequences.fasta"
    focus, focus_aligned = process_fasta(sequence_file_path)

    ## naive method
    identified_residues_folder = './TestResults/Antibodies'
    _,data = Bin.naive_loader(focus_aligned,focus,identified_residues_folder)
    legends = ['Ent','Theil Ent','Cramer Ent','MI','Theil MI','Cramer MI']
    data = convert_arrays_to_lists(data)

    ## existing method 
    identified_residues_folder = './Data/Antibodies/key_tuning_residues_no_rep.csv'
    exist_data = create_annotated_lists(identified_residues_folder, len(focus))
    exist_legends = ['DeepSequence+', 'DeepSequence-', 'Abs(DeepSequence)']


    values_list = data+exist_data # Replace with actual list of lists of values
    names_list = legends+exist_legends  # Replace with actual list of names

    algorithm_names = names_list

    algorithm_importances_list = values_list
    interface_residues = antibody_interface
    binding_energy_indices = [ res for res, (energy, aa) in energy_contributions.items()]
    binding_energies = [ energy for res, (energy, aa) in energy_contributions.items()]

    reduced_binding = [idx for idx, value in zip(binding_energy_indices, binding_energies) if value < 0]

    color_residues_by_indices(antibody_pdb, "./TestResults/Antibodies/energy_colored_antibody.pdb", 
                           antibody_interface, reduced_binding, focus_aligned)

    generate_residue_tables(values_list,threshold=75,
                            algorithm_names=names_list,
                            interface_residues=antibody_interface,
                            binding_energy_indices=binding_energy_indices,
                            binding_energies=binding_energies,
                            save_path="./TestResults/Antibodies/residue_importance_results.csv",
                            alg_flag = [1,1,1,1,1,1,0,0,1])

    plot_results(algorithm_names, algorithm_importances_list, interface_residues, binding_energy_indices, binding_energies)