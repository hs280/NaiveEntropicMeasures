import os
import subprocess
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import MultipleSeqAlignment

def add_string_length_column(df, column_name, new_column_name='StringLength'):
    """
    Add a new column with the length of strings in a specified column.

    Args:
        df (pd.DataFrame): Input DataFrame.
        column_name (str): Name of the column containing strings.
        new_column_name (str): Name for the new column with string lengths.

    Returns:
        pd.DataFrame: DataFrame with the new column added.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in the DataFrame.")
    df[new_column_name] = df[column_name].str.len()
    return df

def ungap_sequences(df, sequence_column):
    """
    Ungap sequences in a Pandas DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with a column containing aligned sequences.
        sequence_column (str): Name of the column containing aligned sequences.

    Returns:
        pd.DataFrame: DataFrame with unaligned sequences.
    """
    aligned_sequences = df[sequence_column]
    alignment = MultipleSeqAlignment([SeqRecord(Seq(seq), id=str(idx)) for idx, seq in enumerate(aligned_sequences)])
    unaligned_sequences = [str(record.seq).replace('-', '') for record in alignment]
    return pd.DataFrame({sequence_column: unaligned_sequences})

def fasta_to_dataframe(file_path):
    """
    Convert a FASTA file to a Pandas DataFrame with a column for each sequence location.

    Args:
        file_path (str): Path to the FASTA file.

    Returns:
        pd.DataFrame: DataFrame with sequence information and a column for each location.
    """
    records = list(SeqIO.parse(file_path, "fasta"))
    seq_length = len(records[0].seq)
    data = {
        "ID": [record.id for record in records],
        "Description": [record.description for record in records],
        "Sequence": [str(record.seq) for record in records]
    }
    for i in range(seq_length):
        data[f"Pos_{i+1}"] = [str(record.seq[i]) for record in records]
    return pd.DataFrame(data)

def read_dat_file(file_path, col_name='Wavelengths'):
    """
    Read a .dat file with a single column of numerics into a Pandas DataFrame.

    Args:
        file_path (str): Path to the .dat file.
        col_name (str): Name for the single column.

    Returns:
        pd.DataFrame: DataFrame containing the numeric values.
    """
    return pd.read_csv(file_path, header=None, names=[col_name])

def merge_and_average(df1, df2):
    """
    Merge two single-column DataFrames and compute the average, handling NaN values.

    Args:
        df1 (pd.DataFrame): First DataFrame.
        df2 (pd.DataFrame): Second DataFrame.

    Returns:
        pd.DataFrame: Merged DataFrame with the specified handling of NaN values.
    """
    return df1.combine_first(df2)

def remove_outliers(df, column_name, scale=2):
    """
    Remove rows with values more than 2 standard deviations away from the mean in the specified column.

    Args:
        df (pd.DataFrame): Input DataFrame.
        column_name (str): Name of the column to check for outliers.
        scale (int, optional): Scaling factor for the threshold. Defaults to 2.

    Returns:
        pd.DataFrame: DataFrame with outliers removed.
    """
    mean_value = df[column_name].mean()
    std_dev = df[column_name].std()
    threshold = scale * std_dev
    return df[
        (df[column_name] >= mean_value - threshold) &
        (df[column_name] <= mean_value + threshold)
    ]

def align_sequences(file_path, output_path):
    """
    Align sequences from a file using Muscle and save the alignment to an output file.

    Args:
        file_path (str): Path to the file containing sequences.
        output_path (str): Path to save the aligned sequences.
    """
    with open(file_path, 'r') as file:
        amino_acid_sequences = [line.strip() for line in file]

    fasta_file_path = 'temp.fasta'
    with open(fasta_file_path, 'w') as fasta_file:
        for i, seq in enumerate(amino_acid_sequences):
            fasta_file.write(f'>Seq{i}\n{seq}\n')

    muscle_cmd = ['muscle','-super5',fasta_file_path,'-output',output_path]
    subprocess.run(muscle_cmd)
    os.remove(fasta_file_path)

def pad_sequences_fasta(file_path, output_path, pad_char='-'):
    """
    Pad sequences in a FASTA file to ensure they are all the same length. Padding is applied evenly
    on both ends of the sequence. If an odd number of padding characters is needed, the extra 
    character is added to the end.

    Args:
        file_path (str): Path to the input FASTA file.
        output_path (str): Path to the output FASTA file with padded sequences.
        pad_char (str): Character used for padding. Defaults to '-'.
    """
    records = list(SeqIO.parse(file_path, "fasta"))
    max_length = max(len(record.seq) for record in records)

    padded_records = []
    for record in records:
        seq_len = len(record.seq)
        total_padding = max_length - seq_len
        left_padding = total_padding // 2
        right_padding = total_padding - left_padding
        new_seq = pad_char * left_padding + str(record.seq) + pad_char * right_padding
        padded_records.append(SeqRecord(Seq(new_seq), id=record.id, description=record.description))

    SeqIO.write(padded_records, output_path, "fasta")

def read_sequences_from_file(file_path):
    """
    Read protein sequences from a given file.

    Parameters:
    - file_path: Path to the file containing protein sequences, one per line.

    Returns:
    - A set of protein sequences.
    """
    sequences = set()
    duplicates = set()  # Set to store duplicates
    invalid_count = 0  # Counter for invalid sequences
    with open(file_path, 'r') as file:
        for line in file:
            sequence = line.strip()  # Remove leading/trailing whitespace
            # Check if sequence contains only Latin alphabet characters
            if all(char.isalpha() and ord(char) < 128 for char in sequence):
                if sequence in sequences:
                    # Sequence is a duplicate
                    duplicates.add(sequence)
                else:
                    sequences.add(sequence)
            else:
                invalid_count += 1  # Increment counter for invalid sequences
    # Print number of invalid sequences
    if invalid_count > 0:
        print(f"{invalid_count} invalid sequences found in file {file_path}.")
    # Print duplicates
    if duplicates:
        print(f"Duplicates found in {file_path}:")
        print(f"{len(duplicates)} duplicates in File")
    print(f"{len(sequences)} unique sequences in File")
    return sequences



def normalize_sequence(sequence):
    # Normalize sequence by removing leading/trailing whitespace and converting to lowercase
    return sequence.strip().lower()

def merge_datasets(file_paths,output_path):
    """
    Merge datasets from multiple .dat files into a single set without duplicates.

    Parameters:
    - file_paths: A list of paths to the .dat files.

    Returns:
    - A set of unique protein sequences from all the files.
    """
    merged_dataset = set()
    length = 0
    for file_path in file_paths:
        sequences = read_sequences_from_file(file_path)
        for sequence in sequences:
            # Normalize sequence
            normalized_sequence = normalize_sequence(sequence)
            # Add normalized sequence to the set
            if normalized_sequence not in merged_dataset:
                # Print debug message to identify sequences being added
            #    print(f"Adding sequence: {normalized_sequence}")
                merged_dataset.add(normalized_sequence)
            #else:
                # Print debug message for sequences identified as duplicates
                #print(f"Duplicate sequence found: {normalized_sequence}")
        print(f"adding {len(merged_dataset)-length} sequences from file")
        length = len(merged_dataset)

    
    with open(output_path, 'w') as file:
        for i, sequence in enumerate(merged_dataset, 1):
            file.write(f">seq{i}\n{sequence}\n")

    print(f"Merged dataset saved to '{output_path}'.")

    return 

def dat_to_fasta(dat_file_path, fasta_file_path):
    """
    Convert a .dat file to FASTA format.

    Args:
        dat_file_path (str): Path to the input .dat file.
        fasta_file_path (str): Path to save the output FASTA file.
    """
    with open(dat_file_path, 'r') as dat_file, open(fasta_file_path, 'w') as fasta_file:
        # Read sequences from the .dat file
        sequences = dat_file.readlines()

        # Write sequences to the FASTA file
        for i, sequence in enumerate(sequences, start=1):
            fasta_file.write(f">Seq{i}\n{sequence}\n")