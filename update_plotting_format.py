import os
import re

def replace_savefig(folder_path):
    """
    Simplify plt.savefig calls in all Python files within a folder
    by keeping only the filepath and adding dpi=600.
    """
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Simplify plt.savefig calls
                # Match plt.savefig(filepath, dpi=600) and replace with plt.savefig(filepath, dpi=600)
                content = re.sub(
                    r"(plt\.savefig\s*\(\s*[^,]+).*?\)",
                    r"\1, dpi=600)",
                    content,
                    flags=re.DOTALL
                )

                # Save changes back to the file
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)

    print(f"Updated plt.savefig calls to use dpi=600 in all files within {folder_path}.")

# Example usage
replace_savefig('./')
