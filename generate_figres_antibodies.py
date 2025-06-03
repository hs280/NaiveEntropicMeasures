import os
import subprocess

def run_selected_python_files(directory):
    selected_files = [
        'analyze_pdbs.py',
        'annotate_pdbs.py',
        'Assess_Existing_Deep.py',
        'assess_measures.py',
        'Assess_Naive_measures.py'
    ]

    for file in selected_files:
        filepath = os.path.join(directory, file)
        if os.path.isfile(filepath):
            print(f"\n=== Running {filepath} ===")
            result = subprocess.run(['python', filepath], capture_output=True, text=True)
            print(f"--- Output ---\n{result.stdout}")
            if result.stderr:
                print(f"--- Errors ---\n{result.stderr}")
        else:
            print(f"File not found: {filepath}")

# Usage
folder_path = './'  # generate measure assement figures for antibodies
run_selected_python_files(folder_path)
