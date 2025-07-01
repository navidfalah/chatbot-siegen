# ==============================================================================
# SCRIPT 1: 01_extract_raw_markdown.py
#
# PURPOSE:
#   - Uses the marker-pdf library directly and correctly in Python.
#   - Explicitly configures Marker using a dictionary and the ConfigParser,
#     translating your desired flags (use_llm, ollama_service) into code.
#   - Processes PDFs in parallel using ProcessPoolExecutor for efficiency.
#
# HOW TO USE:
#   1. Ensure 'marker-pdf' is installed: pip install marker-pdf
#   2. Ensure your Ollama server is running with the specified model.
#   3. Place your PDFs in the 'PDF_INPUT_DIR'.
#   4. Run this script: python 01_extract_raw_markdown.py
# ==============================================================================
import os
import json
from concurrent.futures import ProcessPoolExecutor, as_completed

# --- Configuration ---
PDF_INPUT_DIR = "data/pdfs"
OUTPUT_RAW_MARKDOWN_DIR = "data/raw_markdown"
METADATA_FILE_PATH = "awmf_pdfs_metadata_20250606_115400.json"
WORKERS = 3

# This dictionary replicates your desired command-line flags in Python
MARKER_CONFIG = {
    # "use_llm": True,
    "output_format": "markdown",
    # "llm_service": "marker.services.ollama.OllamaService",
    # "ollama_base_url": "https://ollama.wineme.wiwi.uni-siegen.de",
    # # Replace with your exact model name if it differs
    # "ollama_model": "mistral:latest"
}


def process_pdf_wrapper(pdf_path, config_dict):
    """
    A wrapper function for a single process to handle PDF conversion.
    This function will be executed in its own process by the ProcessPoolExecutor.
    """
    # Imports are placed inside the function to ensure they are available in the spawned process
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict
    from marker.config.parser import ConfigParser

    base_filename = os.path.splitext(os.path.basename(pdf_path))[0]
    output_path = os.path.join(OUTPUT_RAW_MARKDOWN_DIR, f"{base_filename}.md")

    if os.path.exists(output_path):
        return f"SKIPPED: {os.path.basename(pdf_path)} already exists."

    try:
        # Load the models required by marker for conversion (once per worker)
        models = create_model_dict()

        # Set up the converter with the specific configuration
        config_parser = ConfigParser(config_dict)
        converter = PdfConverter(
            config=config_parser.generate_config_dict(),
            artifact_dict=models,
            processor_list=config_parser.get_processors(),
            renderer=config_parser.get_renderer(),
            llm_service=config_parser.get_llm_service()
        )

        # The converter object is called directly with the file path
        rendered = converter(pdf_path)

        # Manually save the converted content as a markdown file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(rendered.markdown)

        return f"SUCCESS: {os.path.basename(pdf_path)}"
    except Exception as e:
        # Added traceback for better error diagnosis
        import traceback
        tb_str = traceback.format_exc()
        return f"FAILED:  {os.path.basename(pdf_path)} with error: {str(e)}\n{tb_str}"


def pythonic_parallel_extract():
    """
    Main function to orchestrate the parallel PDF processing.
    """
    print("--- Stage 1: Starting Pythonic PDF to Raw Markdown Extraction ---")

    os.makedirs(OUTPUT_RAW_MARKDOWN_DIR, exist_ok=True)

    try:
        with open(METADATA_FILE_PATH, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: Metadata file not found at {METADATA_FILE_PATH}")
        return

    # Create a list of full paths to all PDFs that need processing
    tasks = []
    for guideline in metadata.get("guidelines", []):
        for document in guideline.get("documents", []):
            pdf_filename = document.get("filename")
            if pdf_filename:
                relative_pdf_path = os.path.basename(
                    document.get("file_path", pdf_filename))
                full_pdf_path = os.path.join(PDF_INPUT_DIR, relative_pdf_path)
                if os.path.exists(full_pdf_path):
                    tasks.append(full_pdf_path)
                else:
                    print(
                        f"WARNING: PDF file not found, skipping: {full_pdf_path}")

    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        print(f"Starting extraction with {WORKERS} worker(s)...")
        # Submit all tasks to the executor, passing the config dictionary to each
        futures = [executor.submit(
            process_pdf_wrapper, pdf_path, MARKER_CONFIG) for pdf_path in tasks]

        # Process results as they are completed
        for future in as_completed(futures):
            try:
                result = future.result()
                print(result)
            except Exception as e:
                print(f"An error occurred in a worker process: {e}")

    print("\n--- Stage 1: Pythonic Raw Markdown Extraction Complete ---")


if __name__ == "__main__":
    pythonic_parallel_extract()
