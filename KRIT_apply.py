import sys
import os
import shutil
import numpy as np 
from pathlib import Path
from functools import partial
from typing import List
from PyQt5 import QtWidgets, QtGui, QtCore
import qdarkstyle
import Bin as Bin
import argparse
import csv
import pickle as pkl

def save_dict_to_csv(data_dict, output_path):
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write the header
        writer.writerow(['Key', 'Value'])
        
        # Write the data
        for key, value in data_dict.items():
            writer.writerow([key, value])

def process_data_based(seq_data_path,target_data_path,rank_data,save_path,file_names):
    aligned_residues_df = Bin.fasta_to_dataframe(seq_data_path)
    target_df = Bin.read_dat_file(target_data_path)
    os.makedirs(save_path,exist_ok=True)
    rank_data_path = f'{save_path}/rank_data.pkl'
    with open(rank_data_path,'wb') as f:
        pkl.dump(rank_data,f)
    num_samples=np.inf

    max_seq_length = len(aligned_residues_df.values[0])

    Bin.search_sequence_lengths(save_path, 
                            aligned_residues_df, 
                            target_df, 
                            rank_data_path, 
                            num_samples, 
                            split_fraction=0.2, 
                            max_seq_length=max_seq_length, 
                            num_runs=5,
                            file_names=file_names)
    
    Bin.sanitize_directory(save_path)

    resultr = Bin.calculate_sum(save_path)

    save_dict_to_csv(resultr, f'{save_path}/merit_values.csv')

    return resultr


from typing import Tuple

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

def assess_with_data_measures(sequence_file_path: str, target_file_path: str,
                           identified_residues_folder: str, save_path: str)->list:
    ## load_data 
    focus, focus_aligned = process_fasta(sequence_file_path)
    data,_ = Bin.naive_loader(focus_aligned,focus,identified_residues_folder)
    legends = ['Ent','Theil Ent','Cramer Ent','MI','Theil MI','Cramer MI']


    import ray
    os.environ["RAY_memory_usage_threshold"] = "0.9"
    # Initialize Ray
    ray.init(num_cpus=8)

    result = process_data_based(sequence_file_path,target_file_path,data,save_path,legends)
    ray.shutdown()
    return result

# Setup paths
script_path = os.path.abspath(__file__)
curent_path = os.path.dirname(script_path)
directory_folder = str(Path(curent_path).parents[0])
sys.path.insert(0, directory_folder)

def run_analysis(
    msa_path: str,
    target_path: str,
    store_path: str,
    protein_family: str,
    results_folder: str = "./Results_AC",
    calculate_entropy_flag: str = 'y',
    calculate_information_flag: str = 'y',
    assess_with_data_measures_flag: str = 'y',
) -> None:
    os.makedirs(store_path, exist_ok=True)
    os.makedirs(results_folder, exist_ok=True)

    aligned_residues_df = Bin.fasta_to_dataframe(msa_path)
    target_df = Bin.read_dat_file(target_path)

    cross_correl_data = os.path.join(store_path, "cross_correlations.pkl")
    entropies_data = os.path.join(store_path, "entropies_data.pkl") if calculate_entropy_flag == 'y' else None
    information_data = os.path.join(store_path, "information_data.pkl") if calculate_information_flag == 'y' else None

    Bin.handle_info_theory(
        calculate_entropy_flag,
        aligned_residues_df,
        store_path,
        cross_correl_data,
        entropies_data,
        calculate_information_flag,
        store_path,
        target_df,
        information_data
    )

    Bin.plot_residue_probability_heatmap(aligned_residues_df, 'Sequence', store_path)
    Bin.remake_plots(entropies_data, information_data, cross_correl_data, store_path, protein_family)

    labels = []
    if calculate_entropy_flag == 'y':
        Bin.process_list_autocorrelation([entropies_data], ['Entropy', 'Theil Entropy', 'Cramer Entropy'], [store_path])
        labels.extend(['Entropy', 'Theil Entropy', 'Cramer Entropy'])
    if calculate_information_flag == 'y':
        Bin.process_list_autocorrelation([information_data], ['MI', 'Theil MI', 'Cramer MI'], [store_path])
        labels.extend(['MI', 'Theil MI', 'Cramer MI'])

    try:
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
            'CrossCorrelation',
            os.path.join(results_folder, "cross_correlation_colormap.png")
        )
    except Exception as e:
        print(f"Error generating cross-correlation matrix: {e}")

    if assess_with_data_measures_flag == 'y':
        identified_residues_folder = results_folder
        result = assess_with_data_measures(msa_path, target_path, identified_residues_folder, results_folder)
        print(f"Data measures assessment result: {result}")

class KritGUI(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('KRiT Analysis Tool')
        self.setWindowIcon(QtGui.QIcon(os.path.join(curent_path, "KRIT.png")))
        self.init_ui()

    def init_ui(self):
        main_layout = QtWidgets.QVBoxLayout(self)
        horizontal_split = QtWidgets.QHBoxLayout()
        left_panel = QtWidgets.QVBoxLayout()
        right_panel = QtWidgets.QVBoxLayout()
        horizontal_split.addLayout(left_panel, 2)
        horizontal_split.addLayout(right_panel, 1)
        main_layout.addLayout(horizontal_split)

        # File inputs
        file_group = QtWidgets.QGroupBox('Input Files and Output')
        file_layout = QtWidgets.QFormLayout()
        self.msa_line = QtWidgets.QLineEdit(placeholderText='Select MSA file (.fasta)')
        btn_msa = QtWidgets.QToolButton()
        btn_msa.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_DirOpenIcon))
        btn_msa.clicked.connect(partial(self.browse_file, self.msa_line, 'MSA File', '*.fasta *.fa'))
        file_layout.addRow('MSA File:', self._with_button(self.msa_line, btn_msa))

        self.target_line = QtWidgets.QLineEdit(placeholderText='Select target file (.dat)')
        btn_target = QtWidgets.QToolButton()
        btn_target.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_DirOpenIcon))
        btn_target.clicked.connect(partial(self.browse_file, self.target_line, 'Target File', '*.dat *.txt'))
        file_layout.addRow('Target File:', self._with_button(self.target_line, btn_target))

        self.output_line = QtWidgets.QLineEdit(placeholderText='Select output folder')
        btn_output = QtWidgets.QToolButton()
        btn_output.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_DialogOpenButton))
        btn_output.clicked.connect(partial(self.browse_folder, self.output_line, 'Output Folder'))
        file_layout.addRow('Output Folder:', self._with_button(self.output_line, btn_output))

        file_group.setLayout(file_layout)
        left_panel.addWidget(file_group)

        # Settings
        settings_group = QtWidgets.QGroupBox('Analysis Settings')
        settings_layout = QtWidgets.QGridLayout()

        self.protein_line = QtWidgets.QLineEdit(placeholderText='Protein Family Name')
        self.entropy_check = QtWidgets.QCheckBox('Calculate Entropy')
        self.entropy_check.setChecked(True)

        self.info_check = QtWidgets.QCheckBox('Calculate MI')
        self.info_check.setChecked(True)

        self.assess_data_check = QtWidgets.QCheckBox('Assess Measures with Data')
        self.assess_data_check.setChecked(False)  # Default unchecked, adjust as needed

        settings_layout.addWidget(QtWidgets.QLabel('Protein Name:'), 0, 0)
        settings_layout.addWidget(self.protein_line, 0, 1, 1, 2)
        settings_layout.addWidget(self.entropy_check, 1, 0)
        settings_layout.addWidget(self.info_check, 1, 1)
        settings_layout.addWidget(self.assess_data_check, 1, 2)  # Add new checkbox to same row

        settings_group.setLayout(settings_layout)
        left_panel.addWidget(settings_group)

        self.run_btn = QtWidgets.QPushButton('Run Analysis')
        self.run_btn.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_MediaPlay))
        self.run_btn.clicked.connect(self.run_analysis)
        left_panel.addWidget(self.run_btn)

        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        left_panel.addWidget(self.log)

        # Image Browser
        self.image_browser = QtWidgets.QScrollArea()
        self.image_browser.setWidgetResizable(True)
        self.image_container = QtWidgets.QWidget()
        self.image_grid = QtWidgets.QGridLayout(self.image_container)
        self.image_browser.setWidget(self.image_container)

        refresh_btn = QtWidgets.QPushButton("🔄 Refresh")
        refresh_btn.clicked.connect(self.load_images)
        right_panel.addWidget(refresh_btn, alignment=QtCore.Qt.AlignRight)
        right_panel.addWidget(self.image_browser)

        self.refresh_timer = QtCore.QTimer(self)
        self.refresh_timer.timeout.connect(self.load_images)
        self.refresh_timer.start(20000)

        self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())

    def _with_button(self, widget, button):
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(widget)
        layout.addWidget(button)
        wrapper = QtWidgets.QWidget()
        wrapper.setLayout(layout)
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
        assess_data_flag = 'y' if self.assess_data_check.isChecked() else 'n'

        self.log.clear()
        if not all([msa, target, output, protein]):
            self.log.appendPlainText("⚠️ Fill in all fields.")
            return

        try:
            os.makedirs(output, exist_ok=True)
            run_analysis(msa, target, output, protein, output, entropy_flag, info_flag, assess_data_flag)
            self.log.appendPlainText("✅ Analysis completed.")
        except Exception as e:
            self.log.appendPlainText(f"❌ Error: {e}")

    def load_images(self):
        folder = self.output_line.text().strip()
        if not folder or not os.path.isdir(folder):
            return
        for i in reversed(range(self.image_grid.count())):
            widget = self.image_grid.itemAt(i).widget()
            if widget:
                widget.setParent(None)

        images = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        images.sort()
        row, col = 0, 0
        for img in images:
            path = os.path.join(folder, img)
            pixmap = QtGui.QPixmap(path).scaled(200, 200, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            label = QtWidgets.QLabel()
            label.setPixmap(pixmap)
            label.setToolTip(img)
            label.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
            label.mousePressEvent = self.get_image_click_handler(path)
            self.image_grid.addWidget(label, row, col)
            col += 1
            if col >= 3:
                col = 0
                row += 1

    def get_image_click_handler(self, path):
        def handler(event):
            self.show_image_dialog(path)
        return handler

    def show_image_dialog(self, path):
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle(os.path.basename(path))
        dlg.resize(900, 700)
        layout = QtWidgets.QVBoxLayout(dlg)
        label = QtWidgets.QLabel()
        pixmap = QtGui.QPixmap(path)
        label.setPixmap(pixmap.scaled(850, 650, QtCore.Qt.KeepAspectRatio))
        layout.addWidget(label)

        btns = QtWidgets.QHBoxLayout()
        zoom_in = QtWidgets.QPushButton("🔍 Zoom In")
        zoom_out = QtWidgets.QPushButton("🔎 Zoom Out")
        delete_btn = QtWidgets.QPushButton("🗑 Delete")
        copy_btn = QtWidgets.QPushButton("📋 Copy")
        btns.addWidget(zoom_in)
        btns.addWidget(zoom_out)
        btns.addWidget(copy_btn)
        btns.addWidget(delete_btn)
        layout.addLayout(btns)

        zoom_in.clicked.connect(lambda: label.setPixmap(pixmap.scaled(label.pixmap().size() * 1.25, QtCore.Qt.KeepAspectRatio)))
        zoom_out.clicked.connect(lambda: label.setPixmap(pixmap.scaled(label.pixmap().size() * 0.8, QtCore.Qt.KeepAspectRatio)))
        copy_btn.clicked.connect(lambda: self.copy_image_to(dlg, path))
        delete_btn.clicked.connect(lambda: self.confirm_delete_image(dlg, path))
        dlg.exec_()

    def confirm_delete_image(self, parent, path):
        if QtWidgets.QMessageBox.question(parent, "Confirm Delete", f"Delete {os.path.basename(path)}?", QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No) == QtWidgets.QMessageBox.Yes:
            os.remove(path)
            QtWidgets.QMessageBox.information(parent, "Deleted", "Image deleted.")
            parent.accept()
            self.load_images()

    def copy_image_to(self, parent, path):
        folder = QtWidgets.QFileDialog.getExistingDirectory(parent, "Select Destination")
        if folder:
            shutil.copy(path, folder)
            QtWidgets.QMessageBox.information(parent, "Copied", "Image copied.")


def main_cli():
    parser = argparse.ArgumentParser(description="Apply Krit analysis to a protein sequence dataset.")
    parser.add_argument("-m", "--msa_file", type=str, default="./Data/Antibodies/sequences.fasta", help="Path to the MSA file")
    parser.add_argument("-t", "--target_file", type=str, default="./Data/Antibodies/pred_affinity.dat", help="Path to the target file")
    parser.add_argument("-o", "--output_folder", type=str, default="./TestResults/Antibodies", help="Output folder")
    parser.add_argument("-n", "--protein_name", type=str, default="COV Antibodies", help="Protein name")
    parser.add_argument("--no_entropy", action="store_true", help="Disable entropy calculation")
    parser.add_argument("--no_information", action="store_true", help="Disable MI calculation")
    parser.add_argument("--assess_data_measures", action="store_true", help="Assess measures with data")
    args = parser.parse_args()

    entropy_flag = 'n' if args.no_entropy else 'y'
    information_flag = 'n' if args.no_information else 'y'
    assess_data_flag = 'y' if args.assess_data_measures else 'n'

    os.makedirs(args.output_folder, exist_ok=True)

    run_analysis(
        msa_path=args.msa_file,
        target_path=args.target_file,
        store_path=args.output_folder,
        protein_family=args.protein_name,
        results_folder=args.output_folder,
        calculate_entropy_flag=entropy_flag,
        calculate_information_flag=information_flag,
        assess_data_flag=assess_data_flag
    )


if __name__ == "__main__":
    if len(sys.argv) == 1:
        app = QtWidgets.QApplication(sys.argv)
        icon_path = os.path.join(curent_path, "KRIT.png")
        app.setWindowIcon(QtGui.QIcon(icon_path))
        app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        gui = KritGUI()
        gui.resize(1200, 800)
        gui.show()
        sys.exit(app.exec_())
    else:
        main_cli()

    # msa_file      = 'KRIT_Out/TestData/test_sequences.fasta'  #"./Data/Antibodies/sequences.fasta"
    # target_file   = "KRIT_Out/TestData/test_data.dat"
    # output_folder = "KRIT_Out/TestResults"
    # protein_name  = "TestProt"
    # results_folder = output_folder
    # calculate_entropy_flag='y'
    # calculate_information_flag='y'
    # assess_with_data_measures_flag = 'y'


    # run_analysis(msa_file, target_file, 
    #              output_folder, protein_name,results_folder,
    #              calculate_entropy_flag, calculate_information_flag,
    #              assess_with_data_measures_flag)