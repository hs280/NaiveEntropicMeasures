import os

def list_python_files(directory):
    python_files = [f for f in os.listdir(directory) if f.endswith('.py') and os.path.isfile(os.path.join(directory, f))]
    return python_files


import os
import subprocess

def run_selected_python_files(directory):
    selected_files = [
        'generate_measures.py',
        'data_naive.py',
        'process_data.py',
        'data_existing.py',
        'literature_all.py'
    ]

    for file in selected_files:
        filepath = os.path.join(directory, file)
        if os.path.isfile(filepath):
            print(f"Running {filepath}...")
            result = subprocess.run(['python', filepath], capture_output=True, text=True)
            print(f"Output of {file}:\n{result.stdout}")
            if result.stderr:
                print(f"Errors in {file}:\n{result.stderr}")
        else:
            print(f"File not found: {filepath}")


# Usage
folder_path = 'Working'  # Replace with your folder path
files = list_python_files(folder_path)
print("Python files in folder:")
for file in files:
    print(file)

run_selected_python_files(folder_path)