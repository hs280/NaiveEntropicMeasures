import pandas as pd
#from transformers import BertTokenizer, BertModel
from Bio.SeqUtils import seq1
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from Bio.SeqUtils import ProtParam
from Bio.SeqUtils.ProtParamData import kd
import re



def encode_one_hot_sklearn(df, column_name,amino_acids=None):#'ACDEFGHIKLMNPQRSTVWY-'):
    """
    Encode protein amino acid sequences using one-hot encoding with sklearn.

    Parameters:
    - df: Pandas DataFrame containing protein amino acid sequences.
    - column_name: Name of the column with amino acid sequences.

    Returns:
    - one_hot_encoded_sequences: DataFrame of one-hot encoded protein sequences.
    """
    # Convert amino acid sequences to uppercase
    df[column_name] = df[column_name].str.upper()

    if amino_acids!=None:
    # Define the categories list for encoding
        amino_acids = list(amino_acids)
        categories = [list(amino_acids)] * df[column_name].str.len().max()
        # Initialize the OneHotEncoder with the predefined categories
        encoder = OneHotEncoder(categories=categories, sparse_output=False)
    else:
        # Initialize the OneHotEncoder
        encoder = OneHotEncoder(sparse_output=False)

    # Fit and transform the data
    one_hot_encoded_data = encoder.fit_transform(df[column_name].apply(list).apply(lambda x: pd.Series(x)))

    # Create a DataFrame from the encoded data
    one_hot_encoded_sequences = pd.DataFrame(one_hot_encoded_data)

    return one_hot_encoded_sequences


def encode_ordinal(data, column='amino_acid_sequence'):
    """
    Encode amino acid sequences using ordinal encoding.

    Args:
        data (pd.DataFrame): DataFrame containing amino acid sequences.
        column (str): Column name containing the amino acid sequences.

    Returns:
        pd.DataFrame: Encoded DataFrame with ordinal columns.
    """
    # Generate a list of all possible amino acids from the DataFrame
    amino_acids = list(set(''.join(data[column])))
    
    # Create a mapping from amino acid to ordinal
    amino_acid_mapping = {acid: i + 1 for i, acid in enumerate(amino_acids)}

    # Create a copy of the original DataFrame
    new_data = data.copy()

    # Split the sequences into separate columns
    n = len(data[column].iloc[0])
    split_data = data[column].apply(lambda x: list(x))
    split_dataframe = pd.DataFrame(split_data.tolist(), columns=[f"{column}_{i+1}" for i in range(n)])

    # Apply the mapping to each character in the sequence
    encoded_data = split_dataframe.map(lambda x: amino_acid_mapping.get(x, 0))  # Use get to handle unknown amino acids

    return encoded_data


def prot_bert_encoding(df, column_name):
    """
    Encode protein amino acid sequences using proteinBert embeddings.

    Args:
        df (pd.DataFrame): Input DataFrame.
        column_name (str): Name of the column containing protein amino acid sequences.

    Returns:
        pd.DataFrame: DataFrame with a new column for each feature in the encoded sequences.
    """
    model_name = "Rostlab/prot_bert_bfd"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)

    encoded_sequences = []

    for sequence in df[column_name]:
        sequence = sequence.upper()
        sequence = re.sub(r"[UZOB]", "X", sequence)
        sequence = ' '.join(sequence)
        encoded_input = tokenizer(sequence, return_tensors='pt')
        output = model(**encoded_input)
        embeddings = output.last_hidden_state.mean(dim=1).squeeze().detach().numpy()
        encoded_sequences.append(embeddings)

    num_features = embeddings.shape[0]
    # Create a DataFrame with columns for each feature in the encoded sequences
    encoded_df = pd.DataFrame(np.vstack(encoded_sequences), columns=[f"feature_{i}" for i in range(num_features)])

    # Concatenate the original DataFrame with the new encoded DataFrame
    result_df = pd.concat([df, encoded_df], axis=1)

    return encoded_df

def encode_protein_sequences_with_physical_properties(df_column):
    """
    Encode protein amino acid sequences using physical properties.

    Parameters:
    - df_column: Pandas DataFrame column containing protein amino acid sequences.

    Returns:
    - encoded_sequences: DataFrame with physical property features for each sequence.
    """
    encoded_sequences = []

    for sequence in df_column:
        protein_properties = ProtParam.ProteinAnalysis(sequence)
        # Additional physical properties for each amino acid
        properties_dict = {
            "molecular_weight": protein_properties.molecular_weight(),
            "isoelectric_point": protein_properties.isoelectric_point(),
            "aromaticity": protein_properties.aromaticity(),
            "instability_index": protein_properties.instability_index(),
            "flexibility": protein_properties.flexibility(),
            "hydrophobicity": protein_properties.gravy(),
            "net_charge": protein_properties.charge_at_pH(7.0),
            "secondary_structure_fraction": protein_properties.secondary_structure_fraction(),
            "helix_fraction": protein_properties.secondary_structure_fraction()[0],
            "turn_fraction": protein_properties.secondary_structure_fraction()[1],
            "sheet_fraction": protein_properties.secondary_structure_fraction()[2],
            "mean_flexibility": protein_properties.flexibility()[0],
            "mean_hydrophobicity": protein_properties.gravy(),
            "mean_net_charge": protein_properties.charge_at_pH(7.0),
            "mol_percent_C": protein_properties.molecular_weight() / protein_properties.molecular_weight() * 100,
            "aromatic_amino_acids": protein_properties.count_amino_acids()['Y'] + protein_properties.count_amino_acids()['F'],
            "polar_amino_acids": protein_properties.count_amino_acids()['S'] + protein_properties.count_amino_acids()['T'] + protein_properties.count_amino_acids()['C'] + protein_properties.count_amino_acids()['N'] + protein_properties.count_amino_acids()['Q'],
            "nonpolar_amino_acids": protein_properties.count_amino_acids()['A'] + protein_properties.count_amino_acids()['G'] + protein_properties.count_amino_acids()['I'] + protein_properties.count_amino_acids()['L'] + protein_properties.count_amino_acids()['P'] + protein_properties.count_amino_acids()['V'],
            "charged_amino_acids": protein_properties.count_amino_acids()['D'] + protein_properties.count_amino_acids()['E'] + protein_properties.count_amino_acids()['H'] + protein_properties.count_amino_acids()['K'] + protein_properties.count_amino_acids()['R'],
            "total_amino_acids": sum(protein_properties.count_amino_acids().values()),
            # Add even even even more properties as needed
        }

        encoded_sequences.append(properties_dict)


    # Create a DataFrame from the list of lists of dictionaries
    encoded_sequences_df = pd.DataFrame(encoded_sequences)

    return encoded_sequences_df

def load_map_data(file_path):
    """
    Load amino acid map data from a CSV file.

    Parameters:
        file_path (str): Path to the CSV file containing the map data.

    Returns:
        DataFrame: DataFrame containing the amino acid map data.
    """
    map_df = pd.read_csv(file_path)

    map_df2 = map_df.copy()
    map_df2.index = map_df['Unnamed: 0'].values
    map_df2 = map_df2.drop('Unnamed: 0',axis=1)

    return map_df2

def convert_and_expand_sequences(df, map_df,column_name="amino_acid_sequence"):
    """
    Convert amino acid sequences to numeric values based on the provided map DataFrame
    and expand each sequence into a single row with 144 columns.

    Parameters:
        df (DataFrame): DataFrame containing the amino acid sequences.
        map_df (DataFrame): DataFrame containing the mapping of amino acids to numeric values.

    Returns:
        DataFrame: Expanded DataFrame with each sequence represented as a single row.
    """
    # Initialize an empty list to store the expanded sequences
    expanded_sequences = []

    # Iterate over each sequence in the input DataFrame
    for sequence in df[column_name]:
        # Convert each character in the sequence to its corresponding numeric values
        numeric_sequence = [map_df.loc[aa.lower()].tolist() for aa in sequence]

        # Flatten the list of lists to create a single list
        flat_numeric_sequence = [item for sublist in numeric_sequence for item in sublist]

        # Append the flat numeric sequence to the list of expanded sequences
        expanded_sequences.append(flat_numeric_sequence)

    # Create a DataFrame from the list of expanded sequences
    expanded_df = pd.DataFrame(expanded_sequences)

    return expanded_df

def encode_sequence_with_map(df,column_name):
    map_df = load_map_data("/home/hs280/ML_DE_RhodDes/Bin/Encode/amino_acid_feature.csv")


    # Convert and expand sequences
    expanded_df = convert_and_expand_sequences(df, map_df,column_name)

    return expanded_df
def apply_encoder(df, column_name, encoder_name, target_column=None):
    """
    Apply a specified encoding method on a DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.
        column_name (str): Name of the column to be encoded.
        encoder_name (str): Name of the encoding method ('one_hot', 'ordinal', 'target', 'prot_bert').
        target_column (str, optional): Name of the target column for target encoding.

    Returns:
        pd.DataFrame or list: Encoded DataFrame or list of encoded sequences.
    """
    if encoder_name == 'one_hot':
        return encode_one_hot_sklearn(df,column_name)
    elif encoder_name == 'ordinal':
        return encode_ordinal(df, column_name)
    elif encoder_name == 'prot_bert':
        return prot_bert_encoding(df, column_name)
    elif encoder_name == 'properties':
        return encode_protein_sequences_with_physical_properties(df[column_name])
    elif encoder_name =='feature_map':
        return encode_sequence_with_map(df,column_name)
    else:
        raise ValueError(f"Invalid encoder name: {encoder_name}")


def generate_fake_data(num_samples=100, seq_length=50):
    """
    Generate fake sample data with protein amino acid sequences.

    Args:
        num_samples (int): Number of samples to generate.
        seq_length (int): Length of each amino acid sequence.

    Returns:
        pd.DataFrame: DataFrame with fake sample data.
    """
    np.random.seed(42)  # Set seed for reproducibility

    # Generate random amino acid sequences
    amino_acid_sequences = [''.join(np.random.choice(list('ACDEFGHIKLMNPQRSTVWY'), seq_length)) for _ in range(num_samples)]

    # Create DataFrame
    fake_data = pd.DataFrame({
        'protein_id': [f'Protein_{i+1}' for i in range(num_samples)],
        'amino_acid_sequence': amino_acid_sequences,
        'target_value': np.random.rand(num_samples)
    })

    return fake_data

# # Example usage:
# fake_data = generate_fake_data(num_samples=100, seq_length=250)
# print(fake_data.head())

# # Example usage:
# # Assuming 'data' is your DataFrame and 'amino_acid_sequence' is the column to be encoded


# encoded_data = apply_encoder(fake_data, column_name='amino_acid_sequence', encoder_name='feature_map')
# print(encoded_data)