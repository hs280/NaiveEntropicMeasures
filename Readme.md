<table>
  <tr>
    <td width="150">
      <img src="KRIT.png" alt="Project Logo" width="120"/>
    </td>
    <td>
      <h1>
        Codebase for <em>"Decoupled Information Theoretic Feature Selection for Rapid Protein Key Tuning Residue Identification"</em>
      </h1>
    </td>
  </tr>
</table>



Identifying key residues that govern protein function, stability, and interactions remains a significant challenge in protein engineering. We present an information-theoretic approach, inspired by signal processing methods to detect such ‘key tuning’ residues from protein sequences. Our approach integrates both unsupervised and supervised methods using Shannon entropy and mutual information as initial priors, respectively. A key aspect of our method is the use of Cramer's V and Thiel's U to decouple coevolved residues. The supervised method combining mutual information and evolutionary decoupling,  significantly outperforms an existing state of the art, DeepSequence, (pvalue=0.0149). The unsupervised approach that requires only a BLAST derived multiple sequence alignment (MSA) offers a simple yet competitive alternative (pvalue=0.14). We validated our methods by applying them to three distinct protein datasets: Green Fluorescent Proteins (GFPs), rhodopsins, and alkanal monooxygenases, demonstrating a high predictive efficacy compared to established tools. When applied to well-characterised nanobody-antigen interactions, our methods not only effectively identified the binding residues but also accurately reconstructed a contiguous binding surface, surpassing current state-of-the-art computational tools. This work provides a flexible, interpretable, computationally efficient and transparent strategy for identifying key residues, paving the way for new strategies in rational protein design, enzyme optimization and drug discovery.

Paper link: [Decoupled Information Theoretic Feature Selection for Rapid Protein Key Tuning Residue Identification](https://doi.org/10.1101/2025.05.28.653817)

## Authors
- [Haris Saeed](https://hs280portfolio.netlify.app/)
- Aidong Yang
- Wei Huang

## Table of Contents
- [Installation](#installation)
- [KRiT Analysis Tool](#krit-analysis-tool)
    - [Usage](#usage)
    - [Outputs](#outputs)
- [License](#license)
- [Acknowledgements](#acknowledgements)
- [Contact](#contact)

## Installation
The base code is designed to be run in a Python environment. It has been tested on Python 3.8 and above. The code is compatible with both Windows and Linux operating systems. The code requires several Python packages, including NumPy, SciPy, Matplotlib, Seaborn, and scikit-learn. These packages can be installed using pip or conda. Requirements for python installation are listed in the requirements.txt file. To install the environment with anaconda use the following commands:

```bash
conda create -n <env_name> python=3.8
pip install -r requirements.txt
```
or 
```bash
conda env create -f environment.yml
```

When using executable only, this is not required and the code can be run from ./dist/KRIT 

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
or 

```bash 
./dist/KRIT
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

## Generation of publication Figures
In order to generate figures for publication there are two key scripts to run:
- **./Working/generate_figures.py**: generates all figures for the non-antibody dataset
- **./generate_figures_antibodies.py**: generates all non-structural figures for teh antibody dataset 

## License

This work is licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/).

![CC BY-NC-SA 4.0](https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png)

## Acknowledgements
This work was supported by the [ESPRCC](https://www.ukri.org/councils/epsrc/) (EP/M002403/1 and EP/Y014073/1) and the [University of Oxford](https://www.ox.ac.uk/). We thank the University of Oxford for providing the computational resources and support for this research.
We also acknowledge the use of [DeepSequence](https://github.com/debbiemarkslab/DeepSequence), [HotspotWizard](https://doi.org/10.1093/nar/gky417), and [EVCouplings](https://github.com/debbiemarkslab/EVcouplings) for benchmarking our methods. We thank the authors of these tools for their contributions to the field of protein engineering and bioinformatics. We also acknowledge the use of [BLAST](https://blast.ncbi.nlm.nih.gov/Blast.cgi) for generating the MSA data used in this study. The BLAST tool is a widely used and powerful tool for sequence alignment and similarity searching, and we thank the developers for their contributions to the field of bioinformatics. We also acknowledge the use of [ClusPro](https://cluspro.org/help.php) for antibody-antigen docking simulations. ClusPro is a powerful tool for predicting antibody-antigen interactions, and we thank the developers for their contributions to the field of structural biology.

## Contact
For any questions or issues related to the code, please contact:
- Haris Saeed: [haris.saeed@eng.ox.ac.uk](mailto:haris.saeed@eng.ox.ac.uk)



