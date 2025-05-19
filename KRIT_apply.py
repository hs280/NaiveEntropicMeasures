import sys
import os
import numpy as np 
from pathlib import Path  # Module for working with filesystem paths
script_path = os.path.abspath(__file__)
curent_path = os.path.dirname(script_path)
directory_folder = str(Path(curent_path).parents[0])
sys.path.insert(0, directory_folder)
import Bin as Bin
import pickle as pkl
import matplotlib.pyplot as plt
import os
import argparse
from typing import Optional, List, Dict
import sys
import os
import argparse
from PyQt5 import QtWidgets, QtGui, QtCore
import qdarkstyle
from functools import partial

def run_analysis(
    msa_path: str,
    target_path: str,
    store_path: str,
    protein_family: str,
    results_folder: str = "./Results_AC",
    calculate_entropy_flag: str = 'y',
    calculate_information_flag: str = 'y',
) -> None:
    """
    Run the information theory and cross-correlation analysis on a single MSA and target data file.

    Parameters
    ----------
    msa_path : str
        Path to the aligned MSA data file.
    target_path : str
        Path to the target data file.
    store_path : str
        Folder to store output files (e.g., plots and pickle files).
    protein_family : str
        Name of the protein family for labeling plots.
    results_folder : str, optional
        Folder to store aggregated results. Default is "./Results_AC".
    calculate_entropy_flag : {'y', 'n'}, optional
        Whether to calculate entropy. Defaults to 'y'.
    calculate_information_flag : {'y', 'n'}, optional
        Whether to calculate mutual information. Defaults to 'y'.

    Returns
    -------
    None
    """
    # Ensure the output directories exist
    os.makedirs(store_path, exist_ok=True)
    os.makedirs(results_folder, exist_ok=True)

    # Read input data
    aligned_residues_df = Bin.fasta_to_dataframe(msa_path)
    target_df = Bin.read_dat_file(target_path)

    # Define output file paths or skip based on flags
    cross_correl_data = os.path.join(store_path, "cross_correlations.pkl")
    entropies_data = os.path.join(store_path, "entropies_data.pkl") if calculate_entropy_flag == 'y' else None
    information_data = os.path.join(store_path, "information_data.pkl") if calculate_information_flag == 'y' else None

    # Run the information theory analysis (entropy and/or mutual information)
    Bin.handle_info_theory(
        calculate_entropy_flag=calculate_entropy_flag,
        aligned_residues_df=aligned_residues_df,
        entropy_directory=store_path,
        cross_correl_data=cross_correl_data,
        entropies_data=entropies_data,
        calculate_information_flag=calculate_information_flag,
        information_directory=store_path,
        target_df=target_df,
        information_data=information_data
    )

    # Generate individual plots for the MSA and analysis results
    Bin.plot_residue_probability_heatmap(
        aligned_residues_df,
        column_name='Sequence',
        save_folder=store_path
    )

    # Remake plots based on available data
    Bin.remake_plots(
        entropies_data=entropies_data,
        information_data=information_data,
        cross_correl_data=cross_correl_data,
        save_folder=store_path,
        protein_family=protein_family
    )

    # Process and aggregate autocorrelation results
    if calculate_entropy_flag == 'y':
        entropies_dict = Bin.process_list_autocorrelation(
            [entropies_data],
            ['Entropy', 'Theil Entropy', 'Cramer Entropy'],
            [store_path]
        )
    else:
        entropies_dict = {}

    if calculate_information_flag == 'y':
        information_dict = Bin.process_list_autocorrelation(
            [information_data],
            ['MI', 'Theil MI', 'Cramer MI'],
            [store_path]
        )
    else:
        information_dict = {}

    # Plot bar charts with error bars
    Bin.plot_bar_chart_with_error_bars(
        entropies_dict,
        information_dict,
        rotation=90,
        outfolder=results_folder
    )

    # Compute and plot the average cross-correlation matrix
    labels: List[str] = []
    if calculate_entropy_flag == 'y':
        labels.extend(['Entropy', 'Theil Entropy', 'Cramer Entropy'])
    if calculate_information_flag == 'y':
        labels.extend(['MI', 'Theil MI', 'Cramer MI'])

    cross_corr_mat = Bin.average_cross_correlation_matrix(
        [entropies_data] if entropies_data else [],
        [information_data] if information_data else [],
        plot_flag=True,
        labels=labels,
        Protein_Families=[protein_family],
        Store_paths=[store_path]
    )
    Bin.plot_colormap_cross_corr(
        cross_corr_mat,
        labels,
        cbar_label='CrossCorrelation',
        save_path=os.path.join(results_folder, "cross_correlation_colormap.png")
    )


def main_cli():
    parser = argparse.ArgumentParser(
        description="Apply Krit analysis to a protein sequence dataset."
    )

    parser.add_argument(
        "-m", "--msa_file",
        type=str,
        default="./Data/Antibodies/sequences.fasta",
        help="Path to the MSA file (default: ./Data/Antibodies/sequences.fasta)"
    )
    parser.add_argument(
        "-t", "--target_file",
        type=str,
        default="./Data/Antibodies/pred_affinity.dat",
        help="Path to the target property file (default: ./Data/Antibodies/pred_affinity.dat)"
    )
    parser.add_argument(
        "-o", "--output_folder",
        type=str,
        default="./TestResults/Antibodies",
        help="Folder to store the output (default: ./TestResults/Antibodies)"
    )
    parser.add_argument(
        "-n", "--protein_name",
        type=str,
        default="COV Antibodies",
        help="Name of the protein (default: COV Antibodies)"
    )
    parser.add_argument(
        "--no_entropy",
        action="store_true",
        help="Disable entropy calculation and plotting"
    )
    parser.add_argument(
        "--no_information",
        action="store_true",
        help="Disable mutual information calculation and plotting"
    )

    args = parser.parse_args()

    entropy_flag = 'n' if args.no_entropy else 'y'
    information_flag = 'n' if args.no_information else 'y'

    os.makedirs(args.output_folder, exist_ok=True)

    run_analysis(
        msa_path=args.msa_file,
        target_path=args.target_file,
        store_path=args.output_folder,
        protein_family=args.protein_name,
        results_folder=args.output_folder,
        calculate_entropy_flag=entropy_flag,
        calculate_information_flag=information_flag
    )

class KritGUI(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        icon_path = os.path.join(curent_path, "KRIT.png")
        self.setWindowIcon(QtGui.QIcon(icon_path))
        self.setWindowTitle('KRiT Analysis Tool')
        # Main layout and styling
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(10)

        # Group box for file inputs
        file_group = QtWidgets.QGroupBox('Input Files and Output', self)
        file_layout = QtWidgets.QFormLayout()
        file_group.setLayout(file_layout)

        self.msa_line = QtWidgets.QLineEdit(placeholderText='Select MSA file (.fasta, .fa)')
        btn_msa = QtWidgets.QToolButton()
        btn_msa.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_DirOpenIcon))
        btn_msa.clicked.connect(partial(self.browse_file, self.msa_line, 'MSA File', "*.fasta *.fa"))
        file_layout.addRow('MSA File:', self._with_button(self.msa_line, btn_msa))

        self.target_line = QtWidgets.QLineEdit(placeholderText='Select target file (.dat, .txt)')
        btn_target = QtWidgets.QToolButton()
        btn_target.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_DirOpenIcon))
        btn_target.clicked.connect(partial(self.browse_file, self.target_line, 'Target File', "*.dat *.txt"))
        file_layout.addRow('Target File:', self._with_button(self.target_line, btn_target))

        self.output_line = QtWidgets.QLineEdit(placeholderText='Select or create output folder')
        btn_output = QtWidgets.QToolButton()
        btn_output.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_DialogOpenButton))
        btn_output.clicked.connect(partial(self.browse_folder, self.output_line, 'Output Folder'))
        file_layout.addRow('Output Folder:', self._with_button(self.output_line, btn_output))

        main_layout.addWidget(file_group)

        # Protein name and flags
        settings_group = QtWidgets.QGroupBox('Analysis Settings', self)
        settings_layout = QtWidgets.QGridLayout()
        settings_group.setLayout(settings_layout)

        settings_layout.addWidget(QtWidgets.QLabel('Protein Name:'), 0, 0)
        self.protein_line = QtWidgets.QLineEdit(placeholderText='Enter protein family name')
        settings_layout.addWidget(self.protein_line, 0, 1, 1, 2)

        self.entropy_check = QtWidgets.QCheckBox('Calculate Entropy')
        self.entropy_check.setChecked(True)
        settings_layout.addWidget(self.entropy_check, 1, 0)
        self.info_check = QtWidgets.QCheckBox('Calculate Information (MI)')
        self.info_check.setChecked(True)
        settings_layout.addWidget(self.info_check, 1, 1)

        main_layout.addWidget(settings_group)

        # Run button centered
        btn_layout = QtWidgets.QHBoxLayout()
        btn_layout.addStretch()
        self.run_btn = QtWidgets.QPushButton('Run Analysis')
        self.run_btn.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_MediaPlay))
        self.run_btn.setMinimumWidth(150)
        self.run_btn.clicked.connect(self.run_analysis)
        btn_layout.addWidget(self.run_btn)
        btn_layout.addStretch()
        main_layout.addLayout(btn_layout)

        # Log output
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setPlaceholderText('Log output will appear here...')
        main_layout.addWidget(self.log)

        # Dark theme
        self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())

    def _with_button(self, widget, button):
        container = QtWidgets.QHBoxLayout()
        container.setContentsMargins(0, 0, 0, 0)
        container.addWidget(widget)
        container.addWidget(button)
        wrapper = QtWidgets.QWidget()
        wrapper.setLayout(container)
        return wrapper

    def browse_file(self, line_edit, title, filter):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, title, '', filter)
        if path:
            line_edit.setText(path)

    def browse_folder(self, line_edit, title):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, title, '')
        if path:
            line_edit.setText(path)

    def run_analysis(self):
        msa = self.msa_line.text().strip()
        target = self.target_line.text().strip()
        output = self.output_line.text().strip()
        protein = self.protein_line.text().strip()
        entropy_flag = 'y' if self.entropy_check.isChecked() else 'n'
        info_flag = 'y' if self.info_check.isChecked() else 'n'

        self.log.clear()
        if not all([msa, target, output, protein]):
            self.log.appendPlainText('Error: Please fill in all fields.')
            return

        try:
            os.makedirs(output, exist_ok=True)
            run_analysis(
                msa_path=msa,
                target_path=target,
                store_path=output,
                protein_family=protein,
                results_folder=output,
                calculate_entropy_flag=entropy_flag,
                calculate_information_flag=info_flag
            )
            self.log.appendPlainText('✅ Analysis completed successfully.')
        except Exception as e:
            self.log.appendPlainText(f'❌ Error during analysis: {e}')

# Example usage:
if __name__ == "__main__":
    # msa_file      = "./Data/Antibodies/sequences.fasta"
    # target_file   = "./Data/Antibodies/pred_affinity.dat"
    # output_folder = "./TestResults/Antibodies"
    # protein_name  = "COV Antibodies"
    # results_folder = output_folder

    # run_analysis(msa_file, target_file, output_folder, protein_name,results_folder)

    # If no args passed (only script name), launch GUI
    if len(sys.argv) == 1:
        app = QtWidgets.QApplication(sys.argv)
        icon_path = os.path.join(curent_path, "KRIT.png")
        app.setWindowIcon(QtGui.QIcon(icon_path))  # Set app-level icon
        app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        gui = KritGUI()
        gui.show()
        sys.exit(app.exec_())
    else:
        main_cli()
