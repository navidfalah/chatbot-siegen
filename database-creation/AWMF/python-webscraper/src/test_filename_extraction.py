"""
Test script for the extract_filename_from_url function to ensure it works correctly
with AWMF guideline URLs.
"""

import re


def sanitize_filename(name):
    """Sanitizes a string to be used as a filename."""
    # Remove invalid characters for filenames
    name = "".join(c if c.isalnum() or c in (
        '.', '_', '-') else '_' for c in name)
    # Truncate if too long (Windows max path component is often 255)
    return name[:200]


def extract_filename_from_url(url):
    """
    Extracts a descriptive filename from a PDF URL.

    Args:
        url: The URL of the PDF file

    Returns:
        A sanitized, descriptive filename based on the URL
    """
    if not url:
        return "unknown"

    # Handle URL encoding
    url = url.replace('%20', '_').replace('%2D', '-').replace('%5F', '_')

    # Extract the filename part from the URL
    url_parts = url.split('/')
    if len(url_parts) > 0:
        raw_filename = url_parts[-1]

        # Extract query parameters if present (e.g., filename.pdf?version=1.2)
        if '?' in raw_filename:
            raw_filename = raw_filename.split('?')[0]

        # Remove the file extension if present
        if raw_filename.lower().endswith('.pdf'):
            raw_filename = raw_filename[:-4]

        # AWMF guidelines often have clear, descriptive filenames with patterns like:
        # 017-071l_S2k_Cochlea-Implantat-Versorgung-zentral-auditorische-Implantate_2020-12
        if len(raw_filename) > 5:
            # Check if it contains guideline info (like ###-###)
            if re.search(r'\d{3}-\d{3}', raw_filename):
                return sanitize_filename(raw_filename)

            # Look for specific AWMF patterns (S1, S2k, S2e, S3, etc.)
            if re.search(r'S[123][ke]?', raw_filename):
                return sanitize_filename(raw_filename)

            # If it has descriptive underscores or hyphens (common in AWMF URLs)
            if ('_' in raw_filename or '-' in raw_filename) and len(raw_filename) > 10:
                return sanitize_filename(raw_filename)

    # Look for 'assets/guidelines' pattern which is common in AWMF URLs
    for i, part in enumerate(url_parts):
        if part == 'assets' and i+1 < len(url_parts) and url_parts[i+1] == 'guidelines':
            # The last part after 'assets/guidelines/' is usually the filename
            if len(url_parts) > i+2:
                return sanitize_filename(url_parts[-1].replace('.pdf', ''))

        # Special case for nested guidelines in specialty subdirectories
        if 'guidelines' in part and i+1 < len(url_parts):
            # Check for specialty directory pattern with a PDF file after
            if len(url_parts) > i+2 and url_parts[-1].lower().endswith('.pdf'):
                # Get specialty name and filename
                specialty = url_parts[i+1].replace('_', '-')
                filename = url_parts[-1].replace('.pdf', '')

                # If specialty info is not in the filename, combine them
                if specialty not in filename and not re.search(r'\d{3}-\d{3}', filename):
                    return sanitize_filename(f"{specialty}_{filename}")
                return sanitize_filename(filename)

    # Check for AWMF file ID patterns in the URL path
    for part in url_parts:
        # Look for patterns like ###-### which are AWMF guideline IDs
        if re.search(r'\d{3}-\d{3}', part):
            # If there's an ID but it's not in the filename, use it
            if raw_filename and len(raw_filename) > 5 and part not in raw_filename:
                return sanitize_filename(f"{part}_{raw_filename}")

    # As a last resort, return the last part of the URL
    if len(url_parts[-1]) > 0:
        return sanitize_filename(url_parts[-1].replace('.pdf', ''))
    else:
        return "unknown_document"


# Test URLs from AWMF guidelines with various patterns
test_urls = [
    # Standard pattern with guideline number and description
    "https://register.awmf.org/assets/guidelines/017-071l_S2k_Cochlea-Implantat-Versorgung-zentral-auditorische-Implantate_2020-12.pdf",

    # Nested in specialty directory
    "https://register.awmf.org/assets/guidelines/017_D_G_f_Hals-Nasen-Ohrenheilkunde__Kopf-_und_Halschirurgie/017-071i_S2k_Cochlea-Implantat-Versorgung-zentral-auditorische-Implantate_2020-12.pdf",

    # Generic name with no descriptive info
    "https://www.awmf.org/uploads/tx_szleitlinien/Download.pdf",

    # URL with query parameters
    "https://register.awmf.org/assets/guidelines/001-012_S3_Leitlinie_Analgesie_Sedierung_Delirmanagement_2015-08.pdf?version=1.2",

    # URL with encoded characters
    "https://register.awmf.org/assets/guidelines/015%2D010_S3_Diagnostik%5Ftherapie%5Fbipolarer%5Fst%C3%B6rungen_2019-04.pdf",

    # URL with guideline number but generic filename
    "https://register.awmf.org/de/leitlinien/detail/017-071/attachment/01-original.pdf",

    # Different domain with descriptive name
    "https://www.dgn.org/leitlinien/3630-030-132-akuttherapie-des-ischaemischen-schlaganfalls-ergaenzung-2015",

    # URL ending with filename that doesn't clearly identify content
    "https://register.awmf.org/assets/guidelines/document.pdf"
]


def main():
    # Run tests
    print("Testing filename extraction from URLs:\n")
    print("-" * 80)

    for url in test_urls:
        extracted_name = extract_filename_from_url(url)
        print(f"\nURL: {url}")
        print(f"Extracted filename: {extracted_name}")

        # Also show what the final filename would look like with a register number
        register_number = "017-071"
        if register_number not in extracted_name:
            final_name = f"{register_number}_{extracted_name}.pdf"
        else:
            final_name = f"{extracted_name}.pdf"

        print(f"Final filename: {final_name}")
        print("-" * 80)


if __name__ == "__main__":
    main()
