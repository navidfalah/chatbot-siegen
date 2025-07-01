import os

def ensure_dir(directory_path):
    """Ensures that a directory exists, creating it if necessary."""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Created directory: {directory_path}")
    else:
        print(f"Directory already exists: {directory_path}")

if __name__ == '__main__':
    # Example usage:
    ensure_dir("../../data/pdfs")
    ensure_dir("../../data/extracted_text")
