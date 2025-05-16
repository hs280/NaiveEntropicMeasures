# Codebase for "Decoupled Information Theoretic Feature Selection for Rapid Protein Key Tuning Residue Identification"

Identifying key residues that govern protein function, stability, and interactions remains a significant challenge in protein engineering. We present an information-theoretic approach, inspired by signal processing methods to detect such ‘key tuning’ residues from protein sequences. Our approach integrates both unsupervised and supervised methods using Shannon entropy and mutual information as initial priors, respectively. A key aspect of our method is the use of Cramer's V and Thiel's U to decouple coevolved residues. The supervised method combining mutual information and evolutionary decoupling,  significantly outperforms an existing state of the art, DeepSequence, (pvalue=0.0149). The unsupervised approach that requires only a BLAST derived multiple sequence alignment (MSA) offers a simple yet competitive alternative (pvalue=0.14). We validated our methods by applying them to three distinct protein datasets: Green Fluorescent Proteins (GFPs), rhodopsins, and alkanal monooxygenases, demonstrating a high predictive efficacy compared to established tools. When applied to well-characterised nanobody-antigen interactions, our methods not only effectively identified the binding residues but also accurately reconstructed a contiguous binding surface, surpassing current state-of-the-art computational tools. This work provides a flexible, interpretable, computationally efficient and transparent strategy for identifying key residues, paving the way for new strategies in rational protein design, enzyme optimization and drug discovery.

Paper link: [Decoupled Information Theoretic Feature Selection for Rapid Protein Key Tuning Residue Identification](https://doi.org/10.1101/2023.10.02.560679)

## Authors
- [Haris Saeed](https://hs280portfolio.netlify.app/)
- Aidong Yang
- Wei Huang

## Table of Contents
- [Table of Contents](#table-of-contents)
- [Installation](#installation)
- [KRiT Analysis Tool](#krit-analysis-tool)
    - [Usage](#usage)
    - [Outputs](#outputs)
- [License](#license)
- [Acknowledgements](#acknowledgements)
- [Contact](#contact)

## Installation
The code is designed to be run in a Python environment. It has been tested on Python 3.8 and above. The code is compatible with both Windows and Linux operating systems. The code requires several Python packages, including NumPy, SciPy, Matplotlib, Seaborn, and scikit-learn. These packages can be installed using pip or conda. Requirements for python installation are listed in the requirements.txt file. To install the environment with anaconda use the following command:

```bash
conda create -n <env_name> python=3.8
```
To install the required packages, use the following command:
```bash
pip install -r requirements.txt
```

# KRiT Analysis Tool
`Krit_apply.py` is a command-line tool for applying the **KRiT (Key Residue Identification Tool)** analysis pipeline to protein sequence data. It performs information-theoretic feature selection using MSA (multiple sequence alignment) and target property data to identify key tuning residues. As outlined in the paper, the tool uses a combination of unsupervised and supervised methods to achieve this.

## Usage

Run the script from the command line:

```bash
python Krit_apply.py -m <msa_file> -t <target_property_file> -o <output_folder> -n <protein_name>
```

### Command Line Arguments
The following parameters can be specified when running the script:
| Argument	| Type	| Description	| Default |
|----------------|----------------|------------------------------------------------------------------|----------------|
| -m, --msa_file	| str	| Path to the input MSA file in FASTA format	| ./Data/Antibodies/sequences.fasta |
| -t, --target_file	| str	| Path to the file containing target property values (e.g., predicted affinity)	| ./Data/Antibodies/pred_affinity.dat |
| -o, --output_folder	| str	| Folder to store output files. Created automatically if missing	| ./TestResults/Antibodies |
| -n, --protein_name	| str	| Name of the protein. Used for labeling and output filenames	| "COV Antibodies" |
| --no_entropy	| flag	| Disable entropy calculation and plotting	| (entropy on by default) |
| --no_information	| flag	| Disable mutual information (MI) calculation and plotting	| (MI on by default) |
| -h, --help	| flag	| Show help message and exit	|  |


When run without any arguments the script will launch a GUI for selecting the input files and parameters. The GUI will also allow you to select the output folder and protein name. 

```bash
python Krit_apply.py
```

## Outputs
The script generates several output files and plots in the specified output folder. The output includes:
- **Cross-correlation matrix**: A matrix showing the cross-correlations between residues in the MSA.
- **Entropy and mutual information plots**: Visualizations of the entropy and mutual information values for the residues.
- **Cramer's V and Thiel's U plots**: Visualizations of the decoupled coevolution measures.
- **Bar charts**: Bar charts showing the entropy, mutual information, and cross-correlation values for the residues.
- **Multi-panel plots**: Multi-panel plots showing the entropy, mutual information, and autocorrelation values for the residues.
- **Heatmaps**: Heatmaps showing the Cramer's V and Thiel's U values for the residues.
- **Autocorrelation plots**: Plots showing the autocorrelation values for the residues.
- **Cross-correlation colormap**: A colormap showing the cross-correlations between residues.
- **Pickle files**: Pickle files containing the cross-correlation, entropy, and mutual information data for the residues.

## License

This work is licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/).

![CC BY-NC-SA 4.0](https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png)