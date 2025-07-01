# ==============================================================================
# SCRIPT 2: 02_clean_and_enrich.py
#
# PURPOSE:
#   - Loads the raw Markdown files from the flat directory created in Stage 1.
#   - Applies cleaning rules to remove non-content sections.
#   - Enriches the cleaned content with its corresponding metadata.
#   - Saves the clean, enriched data into a new 'clean_json' directory.
#
# HOW TO USE:
#   1. Ensure Stage 1 has been run successfully.
#   2. Run this script: python 02_clean_and_enrich.py
# ==============================================================================
import os
import json
import re

# --- Configuration ---
RAW_MARKDOWN_DIR = "data/raw_markdown"
METADATA_FILE_PATH = "awmf_pdfs_metadata_20250606_115400.json"
OUTPUT_CLEAN_JSON_DIR = "data/clean_json"


def clean_markdown_text(text: str) -> str:
    """
    Applies a series of regex rules to clean Markdown text.
    """
    text = re.sub(
        r'(?im)^#\s*inhalt(sverzeichnis)?\s*\n(.+\n)+?(?=\n#\s)', '', text)
    end_sections = [
        "literatur", "referenzen", "literaturverzeichnis", "anhang", "appendix",
        "interessenkonflikte", "autoren", "impressum", "danksagung",
        "federführende fachgesellschaft", "ansprechpartner"
    ]
    text = re.sub(r'(?im)^#+\s*(' + '|'.join(end_sections) +
                  r')\s*\n(.|\n)*', '', text)
    text = re.sub(r'(?m)^\s*-\s*\d+\s*-\s*$', '', text)
    text = re.sub(r'(?m)(?i)seite\s+\d+\s*$', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text).strip()
    return text


def clean_and_enrich():
    """
    Main function to clean raw markdown and enrich it with metadata.
    """
    print("\n--- Stage 2: Starting Markdown Cleaning and Enrichment ---")

    os.makedirs(OUTPUT_CLEAN_JSON_DIR, exist_ok=True)

    with open(METADATA_FILE_PATH, 'r', encoding='utf-8') as f:
        metadata_lookup = {
            doc['filename']: doc
            for guideline in json.load(f).get("guidelines", [])
            for doc in guideline.get("documents", [])
        }

    # Files are now in a flat structure, so os.listdir is appropriate.
    raw_files = [f for f in os.listdir(RAW_MARKDOWN_DIR) if f.endswith('.md')]

    for raw_filename in raw_files:
        pdf_filename_key = raw_filename.replace('.md', '.pdf')
        print(f"\nProcessing: {raw_filename}")

        output_json_path = os.path.join(
            OUTPUT_CLEAN_JSON_DIR, f"{os.path.splitext(pdf_filename_key)[0]}.json")
        if os.path.exists(output_json_path):
            print("  [*] Clean JSON file already exists. Skipping.")
            continue

        raw_md_path = os.path.join(RAW_MARKDOWN_DIR, raw_filename)
        with open(raw_md_path, 'r', encoding='utf-8') as f:
            raw_content = f.read()

        print("  -> Cleaning raw markdown...")
        cleaned_content = clean_markdown_text(raw_content)

        document_metadata = metadata_lookup.get(pdf_filename_key)
        if not document_metadata:
            print(
                f"  [!] Metadata not found for {pdf_filename_key}. Skipping.")
            continue

        enriched_data = {
            "content_markdown": cleaned_content,
            "metadata": document_metadata
        }

        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(enriched_data, f, ensure_ascii=False, indent=4)
        print(f"  -> Saved clean, enriched data to {output_json_path}")

    print("\n--- Stage 2: Cleaning and Enrichment Complete ---")


if __name__ == "__main__":
    clean_and_enrich()
