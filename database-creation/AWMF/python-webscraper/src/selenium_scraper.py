"""
AWMF Web Scraper using Selenium for JavaScript rendering.
This implementation uses a web browser automation tool (Selenium) to properly render the
JavaScript-based website before scraping content.
"""

import os
import time
import argparse
import re
import json
from datetime import datetime
from urllib.parse import urljoin

# Import required Selenium components
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options as ChromeOptions
    from selenium.webdriver.firefox.options import Options as FirefoxOptions
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import TimeoutException, NoSuchElementException

    # Import webdriver-manager for automatic driver installation
    from webdriver_manager.chrome import ChromeDriverManager
    from webdriver_manager.firefox import GeckoDriverManager
    from selenium.webdriver.chrome.service import Service as ChromeService
    from selenium.webdriver.firefox.service import Service as FirefoxService

    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    print("Selenium or webdriver-manager is not installed. Please install them using:")
    print("pip install selenium webdriver-manager")

import requests
from bs4 import BeautifulSoup

from utils import ensure_dir

# Base URLs for the AWMF website
MAIN_URL = "https://register.awmf.org"
FACHGESELLSCHAFT_BASE_URL = "https://register.awmf.org/de/leitlinien/aktuelle-leitlinien/fachgesellschaft"

# Define output directories
# Relative to the src directory
PDF_OUTPUT_DIR = os.path.join("data", "pdfs")
METADATA_OUTPUT_DIR = os.path.join(
    "data", "metadata")  # Directory for metadata files


def setup_webdriver(headless=True, browser='chrome'):
    """Set up the Selenium WebDriver for browser automation with automatic driver management."""
    if not SELENIUM_AVAILABLE:
        raise ImportError(
            "Selenium is not available. Please install it first.")

    if browser.lower() == 'chrome':
        options = ChromeOptions()
        if headless:
            options.add_argument('--headless=new')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')
        options.add_argument('--window-size=1920,1080')

        # Add user agent to avoid detection
        options.add_argument(
            '--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36')

        try:
            # Use webdriver-manager to automatically download the latest driver
            service = ChromeService(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=options)
        except Exception as e:
            print(f"Failed to use webdriver-manager: {e}")
            # Fallback to standard initialization
            driver = webdriver.Chrome(options=options)

    else:  # Use Firefox as default fallback
        options = FirefoxOptions()
        if headless:
            options.add_argument('--headless')
        options.add_argument('--width=1920')
        options.add_argument('--height=1080')

        try:
            # Use webdriver-manager to automatically download the latest driver
            service = FirefoxService(GeckoDriverManager().install())
            driver = webdriver.Firefox(service=service, options=options)
        except Exception as e:
            print(f"Failed to use webdriver-manager: {e}")
            # Fallback to standard initialization
            driver = webdriver.Firefox(options=options)

    # Set page load timeout to avoid indefinite waits
    driver.set_page_load_timeout(30)

    return driver


def download_pdf(pdf_url, base_filename, metadata=None):
    """
    Downloads a PDF from a given URL and saves it to the specified filename.
    Handles duplicate filenames by appending a counter.

    Args:
        pdf_url: URL of the PDF to download
        base_filename: Base filename to save the PDF as
        metadata: Dictionary containing metadata about the PDF

    Returns:
        Tuple of (success_bool, actual_filename) where
        success_bool is True if download was successful
        actual_filename is the final filename used (may differ from base_filename if duplicate)
    """
    ensure_dir(PDF_OUTPUT_DIR)
    ensure_dir(METADATA_OUTPUT_DIR)

    # Extract filename components
    name_parts = os.path.splitext(base_filename)
    base_name = name_parts[0]
    extension = name_parts[1] if len(name_parts) > 1 else '.pdf'

    # Check if file exists, if so, append counter
    filepath = os.path.join(PDF_OUTPUT_DIR, base_filename)
    counter = 1

    while os.path.exists(filepath):
        # Try to extract URL-specific info to make unique filename
        url_parts = pdf_url.split('/')
        url_filename = url_parts[-1]

        if counter == 1 and url_filename.endswith('.pdf'):
            # Get the last part of the URL without extension as a unique identifier
            identifier = url_filename[:-4].split('_')[-1]
            if identifier.isalnum() and len(identifier) < 10:  # Reasonable identifier
                new_filename = f"{base_name}_{identifier}{extension}"
                filepath = os.path.join(PDF_OUTPUT_DIR, new_filename)
                if not os.path.exists(filepath):
                    break

        # If that didn't work, use a counter
        new_filename = f"{base_name}_{counter}{extension}"
        filepath = os.path.join(PDF_OUTPUT_DIR, new_filename)
        counter += 1

    final_filename = os.path.basename(filepath)

    try:
        # Add headers to mimic a browser request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Referer': MAIN_URL,  # Set the referer to the main website
        }

        response = requests.get(pdf_url, stream=True,
                                timeout=30, headers=headers)
        response.raise_for_status()  # Raise an exception for bad status codes

        # Check if the response is actually a PDF
        content_type = response.headers.get('Content-Type', '')
        if 'application/pdf' not in content_type and not pdf_url.lower().endswith('.pdf'):
            print(
                f"Warning: URL {pdf_url} does not return a PDF (Content-Type: {content_type})")

        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Successfully downloaded: {final_filename}")

        # Save metadata if provided
        if metadata:
            # Add enhanced metadata for RAG applications
            metadata['download_timestamp'] = datetime.now().isoformat()
            metadata['downloaded_filename'] = final_filename
            metadata['file_path'] = os.path.join(
                PDF_OUTPUT_DIR, final_filename)
            metadata['file_size_bytes'] = os.path.getsize(filepath)
            metadata['content_type'] = content_type

            # Extract source guideline information more cleanly
            source_url = metadata.get('source_page', '')
            metadata['guideline_id'] = extract_register_number_from_url(
                source_url)

            # Extract useful headers from the response
            if response.headers.get('Last-Modified'):
                metadata['last_modified'] = response.headers.get(
                    'Last-Modified')

            # Create a JSON file with the same base name for individual metadata
            metadata_filename = os.path.splitext(final_filename)[0] + '.json'
            metadata_path = os.path.join(
                METADATA_OUTPUT_DIR, metadata_filename)

            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            print(f"Saved metadata to: {metadata_filename}")

        return True, final_filename
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {pdf_url}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while downloading {pdf_url}: {e}")
    return False, None


def sanitize_filename(name):
    """Sanitizes a string to be used as a filename."""
    # Remove invalid characters for filenames
    name = "".join(c if c.isalnum() or c in (
        '.', '_', '-') else '_' for c in name)
    # Truncate if too long (Windows max path component is often 255)
    return name[:200]


def get_fachgesellschaft_ids():
    """Returns a list of fachgesellschaft IDs used in URLs."""
    # This is a list of some common fachgesellschaft IDs
    return [
        "001", "002", "003", "004", "005", "006", "007", "008", "009", "010",
        "011", "012", "013", "014", "015", "016", "017", "018", "019", "020",
        "021", "022", "023", "024", "025", "026", "027", "028", "029", "030"
    ]


def extract_register_number_from_url(url):
    """Extract the register number from a URL."""
    if not url:
        return ""

    # URLs are like https://register.awmf.org/de/leitlinien/detail/001-005
    # We want to extract the "001-005" part
    parts = url.split('/')
    if len(parts) > 0:
        last_part = parts[-1]
        # Check if it matches the typical pattern of ###-### or similar
        if re.match(r'\d{3}-\d{3}', last_part) or re.match(r'\d{3}-\d{2,}', last_part):
            return last_part

    return ""


def extract_guideline_links_with_selenium(driver, fach_url):
    """Extract links to guideline detail pages using Selenium."""
    guideline_links = []
    max_retries = 3
    retry_count = 0

    while retry_count < max_retries:
        try:
            # Navigate to the page
            print(f"Navigating to {fach_url}")
            driver.get(fach_url)

            # Wait for the page to load (adjust timeout as needed)
            WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "ion-row"))
            )

            # Allow some extra time for all content to render
            time.sleep(3)

            # Find all guideline links
            rows = driver.find_elements(
                By.CSS_SELECTOR, "ion-row.guideline-listing-row")
            print(f"Found {len(rows)} guideline rows")

            if not rows:
                # Try a secondary method if no rows found
                print(
                    "No rows found with primary selector, trying alternate approach...")
                links = driver.find_elements(
                    By.CSS_SELECTOR, "a[href*='/leitlinien/detail/']")
                for link in links:
                    href = link.get_attribute("href")
                    title = link.text.strip()
                    if href and 'leitlinien/detail' in href:
                        register_number = extract_register_number_from_url(
                            href)
                        guideline_links.append({
                            'url': href,
                            'title': title or "Unknown Title",
                            'register_number': register_number
                        })
                        print(
                            f"Found guideline: {title or 'Unknown Title'} - {href}")
            else:
                # Process rows normally
                for row in rows:
                    try:
                        # Find the guideline link
                        link_elem = row.find_element(By.TAG_NAME, "a")
                        if link_elem:
                            href = link_elem.get_attribute("href")
                            title = link_elem.text.strip()

                            # Find register number (usually in the first column)
                            register_number = ""
                            cols = row.find_elements(By.TAG_NAME, "ion-col")
                            if cols and len(cols) > 0:
                                register_number = cols[0].text.strip()

                            # If register number not found in column, try to extract from URL
                            if not register_number:
                                register_number = extract_register_number_from_url(
                                    href)

                            guideline_links.append({
                                'url': href,
                                'title': title or "Unknown Title",
                                'register_number': register_number
                            })
                            print(
                                f"Found guideline: {title or 'Unknown Title'} - {href}")
                    except NoSuchElementException:
                        continue  # Skip rows without links

            break  # Successfully retrieved links, break out of retry loop

        except TimeoutException:
            retry_count += 1
            print(
                f"Timeout waiting for page {fach_url} to load (attempt {retry_count}/{max_retries})")
            if retry_count >= max_retries:
                print(
                    f"Failed to load {fach_url} after {max_retries} attempts")
                break
            time.sleep(2)  # Wait before retrying

        except Exception as e:
            print(f"Error extracting guideline links from {fach_url}: {e}")
            break

    return guideline_links


def extract_pdf_links_with_selenium(driver, guideline_url, guideline_title, register_number=""):
    """Extract PDF links from a guideline page using Selenium."""
    pdf_links = []

    try:
        # Navigate to the page
        print(f"Navigating to {guideline_url}")
        driver.get(guideline_url)

        # Wait for the page to load (adjust timeout as needed)
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.TAG_NAME, "a"))
        )

        # Allow some extra time for all content to render
        time.sleep(3)

        # Find all links on the page that might be PDFs
        links = driver.find_elements(By.TAG_NAME, "a")
        for link in links:
            try:
                href = link.get_attribute("href")
                if href and href.lower().endswith('.pdf'):
                    link_text = link.text.strip()

                    # If the link text is empty, try to find nearby context
                    if not link_text:
                        # Try to find parent elements with better context
                        parent_script = """
                        var element = arguments[0];
                        var parent = element.parentElement;
                        var prev = parent.previousElementSibling;
                        if (prev && (prev.tagName === 'H3' || prev.tagName === 'H2' || 
                                     prev.tagName === 'STRONG' || prev.tagName === 'B')) {
                            return prev.textContent;
                        }
                        return '';
                        """
                        context = driver.execute_script(parent_script, link)
                        if context:
                            link_text = context
                        else:
                            link_text = guideline_title

                    # Determine PDF type
                    pdf_type = ""
                    href_lower = href.lower()
                    if any(term in href_lower for term in ["langfassung", "ll_lf", "lang", "_lf_", "lf."]):
                        pdf_type = "Langfassung"
                    elif any(term in href_lower for term in ["kurzfassung", "ll_kf", "kurz", "_kf_", "kf."]):
                        pdf_type = "Kurzfassung"
                    elif "patienten" in href_lower:
                        pdf_type = "Patientenleitlinie"

                    # Create a descriptive filename
                    if register_number:
                        base_name = f"{register_number}_{sanitize_filename(link_text)}"
                    else:
                        base_name = sanitize_filename(link_text)

                    if pdf_type and pdf_type.lower() not in base_name.lower():
                        filename = f"{base_name}_{pdf_type}.pdf"
                    else:
                        filename = f"{base_name}.pdf"

                    pdf_links.append({
                        'url': href,
                        'name': filename,
                        'type': pdf_type
                    })
                    print(f"Found PDF: {filename} - {href}")
            except Exception as e:
                print(f"Error processing link: {e}")
                continue

    except TimeoutException:
        print(f"Timeout waiting for page {guideline_url} to load")
    except Exception as e:
        print(f"Error extracting PDF links from {guideline_url}: {e}")

    return pdf_links


def save_consolidated_metadata(all_pdf_metadata):
    """
    Saves consolidated metadata about all downloaded PDFs to a single JSON file.
    This makes it easier to work with the data in RAG applications by organizing
    the PDFs by guideline and creating clear relationships between documents.

    Args:
        all_pdf_metadata: List of metadata dictionaries for all PDFs
    """
    ensure_dir(METADATA_OUTPUT_DIR)

    # Create a timestamped filename for the consolidated metadata
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metadata_file = os.path.join(
        METADATA_OUTPUT_DIR, f"awmf_pdfs_metadata_{timestamp}.json")

    # Organize PDFs by guideline for better structure in RAG
    guidelines = {}

    # Group PDFs by their guideline
    for pdf in all_pdf_metadata:
        source_url = pdf.get('source_page', '')
        guideline_id = pdf.get(
            'guideline_id') or extract_register_number_from_url(source_url)
        guideline_title = pdf.get('guideline_title', '')

        if not guideline_id:
            # Use source URL as fallback
            guideline_key = source_url
        else:
            guideline_key = guideline_id

        if guideline_key not in guidelines:
            guidelines[guideline_key] = {
                'id': guideline_id,
                'title': guideline_title,
                'url': source_url,
                'documents': []
            }

        # Add this PDF to the guideline's documents
        guidelines[guideline_key]['documents'].append({
            'filename': pdf.get('downloaded_filename', ''),
            'file_path': pdf.get('file_path', ''),
            'type': pdf.get('type', ''),
            'url': pdf.get('pdf_url', pdf.get('url', '')),
            'file_size_bytes': pdf.get('file_size_bytes', 0),
            'download_timestamp': pdf.get('download_timestamp', '')
        })

    # Structure the consolidated data for RAG
    consolidated_data = {
        "collection_info": {
            "timestamp": datetime.now().isoformat(),
            "total_pdfs": len(all_pdf_metadata),
            "total_guidelines": len(guidelines)
        },
        "guidelines": list(guidelines.values())
    }

    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(consolidated_data, f, ensure_ascii=False, indent=2)

    print(
        f"Saved consolidated metadata for {len(all_pdf_metadata)} PDFs across {len(guidelines)} guidelines to: {metadata_file}")

    # Also create a simple CSV index for easy viewing
    csv_file = os.path.join(METADATA_OUTPUT_DIR,
                            f"awmf_pdfs_index_{timestamp}.csv")
    try:
        with open(csv_file, 'w', encoding='utf-8') as f:
            f.write("Guideline ID,Guideline Title,Document Type,Filename,File Path\n")
            for guideline_id, guideline_info in guidelines.items():
                for doc in guideline_info['documents']:
                    f.write(
                        f"\"{guideline_info['id']}\",\"{guideline_info['title']}\",\"{doc['type']}\",\"{doc['filename']}\",\"{doc['file_path']}\"\n")
        print(f"Created CSV index at: {csv_file}")
    except Exception as e:
        print(f"Error creating CSV index: {e}")


def scrape_awmf_with_selenium(browser='chrome', headless=True, test_mode=False,
                              max_fachgesellschaft=None, max_guidelines=None, max_pdfs=None):
    """
    Main function to scrape the AWMF website using Selenium.

    Args:
        browser (str): Browser to use ('chrome' or 'firefox')
        headless (bool): Whether to run the browser in headless mode
        test_mode (bool): If True, enable test mode with limited downloads
        max_fachgesellschaft (int): Max number of fachgesellschaft pages to process
        max_guidelines (int): Max number of guideline pages per fachgesellschaft
        max_pdfs (int): Max number of PDFs to download
    """
    if not SELENIUM_AVAILABLE:
        print(
            "Selenium is not installed. Please install it first with: pip install selenium")
        return 0

    print(
        f"Starting AWMF scraper with Selenium. Output directory: {os.path.abspath(PDF_OUTPUT_DIR)}")
    if test_mode:
        print(f"RUNNING IN TEST MODE - Limited to {max_fachgesellschaft} fachgesellschaft pages, " +
              f"{max_guidelines} guidelines per fachgesellschaft, and {max_pdfs} PDFs total")

    ensure_dir(PDF_OUTPUT_DIR)

    # Set up the WebDriver
    driver = setup_webdriver(headless=headless, browser=browser)

    try:
        # Step 1: Get fachgesellschaft IDs
        fachgesellschaft_ids = get_fachgesellschaft_ids()

        if test_mode and max_fachgesellschaft and max_fachgesellschaft < len(fachgesellschaft_ids):
            print(
                f"Test mode: limiting to {max_fachgesellschaft} fachgesellschaft pages")
            fachgesellschaft_ids = fachgesellschaft_ids[:max_fachgesellschaft]

        all_guideline_links = []
        all_pdf_links = []
        all_pdf_metadata = []

        # Step 2: For each fachgesellschaft page, extract guideline links
        for idx, fach_id in enumerate(fachgesellschaft_ids):
            fach_url = f"{FACHGESELLSCHAFT_BASE_URL}/{fach_id}"
            print(
                f"\nProcessing Fachgesellschaft {idx+1}/{len(fachgesellschaft_ids)}: {fach_id}")

            guideline_links = extract_guideline_links_with_selenium(
                driver, fach_url)

            if test_mode and max_guidelines and max_guidelines < len(guideline_links):
                print(f"Test mode: limiting to {max_guidelines} guidelines")
                guideline_links = guideline_links[:max_guidelines]

            all_guideline_links.extend(guideline_links)
            time.sleep(1)  # Be respectful to the server
            # Break early if we already have enough guidelines and we're in test mode
            if test_mode and max_guidelines and max_fachgesellschaft and len(all_guideline_links) >= max_guidelines * max_fachgesellschaft:
                break

        print(f"\nFound a total of {len(all_guideline_links)} guideline links")

        # Step 3: For each guideline page, extract PDF links
        for idx, guideline in enumerate(all_guideline_links):
            url = guideline['url']
            title = guideline['title']
            register_number = guideline.get('register_number', '')

            print(
                f"\nProcessing guideline {idx+1}/{len(all_guideline_links)}: {title}")

            pdf_links = extract_pdf_links_with_selenium(
                driver, url, title, register_number)

            if pdf_links:
                for pdf in pdf_links:
                    pdf['source_page'] = url
                    all_pdf_links.append(pdf)

                    # Check if we've reached the PDF limit in test mode
                    if test_mode and max_pdfs and len(all_pdf_links) >= max_pdfs:
                        print(f"Test mode: reached limit of {max_pdfs} PDFs")
                        break

            # Break if we've reached the PDF limit in test mode
            if test_mode and max_pdfs and len(all_pdf_links) >= max_pdfs:
                break

            time.sleep(1)  # Be respectful to the server

        # Step 4: Download all PDFs
        print(f"\nFound a total of {len(all_pdf_links)} PDF links to download")

        successfully_downloaded = 0
        for idx, pdf_info in enumerate(all_pdf_links):
            if not pdf_info.get('url'):  # Skip entries with missing URLs
                continue

            name = pdf_info.get('name', 'unknown.pdf')
            print(
                f"Downloading PDF {idx+1}/{len(all_pdf_links)}: {name} from {pdf_info['url']}")

            # Enhance PDF metadata before download - find corresponding guideline info
            # This is crucial for RAG to understand document context
            guideline_info = next(
                (g for g in all_guideline_links if g['url'] == pdf_info.get('source_page')), None)
            if guideline_info:
                pdf_info['guideline_title'] = guideline_info['title']
                pdf_info['guideline_id'] = guideline_info.get(
                    'register_number', '')

            success, filename = download_pdf(pdf_info['url'], name, pdf_info)
            if success:
                successfully_downloaded += 1
                all_pdf_metadata.append(pdf_info)
            time.sleep(1)  # Be respectful to the server

        # Save consolidated metadata
        save_consolidated_metadata(all_pdf_metadata)

        print(
            f"\nScraping complete. Successfully downloaded {successfully_downloaded} PDFs to {os.path.abspath(PDF_OUTPUT_DIR)}")
        return successfully_downloaded

    finally:
        # Always make sure to close the browser
        driver.quit()


def main():
    parser = argparse.ArgumentParser(
        description='Scrape AWMF website for guideline PDFs using Selenium')
    parser.add_argument('--browser', choices=['chrome', 'firefox'], default='chrome',
                        help='Browser to use for Selenium (default: chrome)')
    parser.add_argument('--no-headless', action='store_true',
                        help='Run the browser in visible mode (not headless)')
    parser.add_argument('--test', action='store_true',
                        help='Run in test mode with limited downloads')
    parser.add_argument('--max-fachgesellschaft', type=int, default=2,
                        help='Max number of fachgesellschaft pages in test mode (default: 2)')
    parser.add_argument('--max-guidelines', type=int, default=3,
                        help='Max guidelines per fachgesellschaft in test mode (default: 3)')
    parser.add_argument('--max-pdfs', type=int, default=5,
                        help='Max total PDFs to download in test mode (default: 5)')

    args = parser.parse_args()

    scrape_awmf_with_selenium(
        browser=args.browser,
        headless=not args.no_headless,
        test_mode=args.test,
        max_fachgesellschaft=args.max_fachgesellschaft,
        max_guidelines=args.max_guidelines,
        max_pdfs=args.max_pdfs
    )


if __name__ == "__main__":
    main()
