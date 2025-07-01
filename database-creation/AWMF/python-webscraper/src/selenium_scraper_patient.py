"""
AWMF Web Scraper using Selenium for JavaScript rendering.
This implementation uses a web browser automation tool (Selenium) to properly render the
JavaScript-based website before scraping content.

MODIFIED: This version starts from a search URL to find and download PDFs specifically 
labeled as "Patientenleitlinie". It includes robust lazy-loading handling by scrolling the correct
<ion-content> element, runs as a single-threaded process, and reports any failed downloads.
"""

import os
import time
import argparse
import re
import json
from datetime import datetime
from urllib.parse import urljoin
import atexit
from typing import List, Dict, Any, Tuple, Optional
# Import tqdm for progress bars if available
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("tqdm not installed. Progress bars will not be available.")
    print("Install tqdm for progress bars: pip install tqdm")

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

# Base URLs for the AWMF website
MAIN_URL = "https://register.awmf.org"
SEARCH_URL = "https://register.awmf.org/de/suche#versionlabel=Guideline&doctype=patientGuideline&sorting=relevance"

# Define output directories
# Relative to the src directory
PDF_OUTPUT_DIR = os.path.join("data", "pdfs")
METADATA_OUTPUT_DIR = os.path.join(
    "data", "metadata")  # Directory for metadata files


def ensure_dir(directory):
    """Ensures that a directory exists, creating it if necessary."""
    if not os.path.exists(directory):
        os.makedirs(directory)


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
    driver.set_page_load_timeout(60)

    return driver


def download_pdf(pdf_url, base_filename, metadata=None):
    """
    Downloads a PDF from a given URL and saves it to the specified filename.
    Handles duplicate filenames by appending a counter.
    """
    ensure_dir(PDF_OUTPUT_DIR)
    ensure_dir(METADATA_OUTPUT_DIR)

    # Extract filename components
    name_parts = os.path.splitext(base_filename)
    base_name = name_parts[0]
    extension = name_parts[1] if len(name_parts) > 1 else '.pdf'

    # Always add a hash to ensure uniqueness for every file
    import hashlib
    # Create a short hash (8 chars) of the URL and a timestamp for guaranteed uniqueness
    timestamp = str(time.time())
    url_hash = hashlib.md5((pdf_url + timestamp).encode()).hexdigest()[:8]

    # Create a filename with hash for guaranteed uniqueness
    new_filename = f"{base_name}_{url_hash}{extension}"
    filepath = os.path.join(PDF_OUTPUT_DIR, new_filename)
    counter = 1

    while os.path.exists(filepath):
        new_filename = f"{base_name}_{counter}_{url_hash}{extension}"
        filepath = os.path.join(PDF_OUTPUT_DIR, new_filename)
        counter += 1

    final_filename = os.path.basename(filepath)

    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Referer': MAIN_URL,
        }
        response = requests.get(pdf_url, stream=True,
                                timeout=30, headers=headers)
        response.raise_for_status()

        content_type = response.headers.get('Content-Type', '')
        if 'application/pdf' not in content_type and not pdf_url.lower().endswith('.pdf'):
            print(
                f"Warning: URL {pdf_url} does not return a PDF (Content-Type: {content_type})")

        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Successfully downloaded: {final_filename}")
        if metadata:
            current_time = datetime.now()
            metadata['download_timestamp'] = current_time.isoformat()
            metadata['download_date'] = current_time.strftime("%Y-%m-%d")
            metadata['download_url'] = pdf_url
            metadata['source_page_url'] = metadata.get('source_page', '')
            metadata['downloaded_filename'] = final_filename
            metadata['file_path'] = os.path.join(
                PDF_OUTPUT_DIR, final_filename)
            metadata['file_size_bytes'] = os.path.getsize(filepath)
            metadata['content_type'] = content_type
            metadata['url_hash'] = url_hash
            source_url = metadata.get('source_page', '')
            if not metadata.get('guideline_id'):
                metadata['guideline_id'] = extract_register_number_from_url(
                    source_url)
            if 'guideline_title' not in metadata or not metadata['guideline_title']:
                if source_url:
                    url_parts = source_url.split('/')
                    if len(url_parts) > 2:
                        metadata['source_page_name'] = url_parts[-1]
            if response.headers.get('Last-Modified'):
                metadata['last_modified'] = response.headers.get(
                    'Last-Modified')

            metadata_filename = os.path.splitext(final_filename)[0] + '.json'
            metadata_path = os.path.join(
                METADATA_OUTPUT_DIR, metadata_filename)
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            print(f"Saved metadata to: {metadata_filename}")

        return True, final_filename, metadata
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {pdf_url}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while downloading {pdf_url}: {e}")
    return False, None, metadata


def sanitize_filename(name):
    """Sanitizes a string to be used as a filename."""
    name = "".join(c if c.isalnum() or c in (
        '.', '_', '-') else '_' for c in name)
    return name[:200]


def extract_filename_from_url(url):
    """Extracts a descriptive filename from a PDF URL."""
    if not url:
        return "unknown"
    url = url.replace('%20', '_').replace('%2D', '-').replace('%5F', '_')
    url_parts = url.split('/')
    if len(url_parts) > 0:
        raw_filename = url_parts[-1]
        if '?' in raw_filename:
            raw_filename = raw_filename.split('?')[0]
        if raw_filename.lower().endswith('.pdf'):
            raw_filename = raw_filename[:-4]
        if len(raw_filename) > 5:
            if re.search(r'\d{3}-\d{3}', raw_filename) or re.search(r'S[123][ke]?', raw_filename) or ('_' in raw_filename or '-' in raw_filename):
                return sanitize_filename(raw_filename)
    if len(url_parts[-1]) > 0:
        return sanitize_filename(url_parts[-1].replace('.pdf', ''))
    return "unknown_document"


def extract_register_number_from_url(url):
    """Extract the register number from a URL."""
    if not url:
        return ""
    parts = url.split('/')
    if len(parts) > 0:
        last_part = parts[-1]
        if re.match(r'\d{3}-\d{3}', last_part) or re.match(r'\d{3}-\d{2,}', last_part):
            return last_part
    return ""


def extract_guideline_links_from_search(driver, search_url, test_mode=False, max_guidelines=None):
    """Extract links to guideline detail pages from the main search page."""
    guideline_links = []

    print(f"Navigating to search page: {search_url}")
    driver.get(search_url)

    try:
        # --- ROBUST LAZY LOADING LOGIC ---
        # 1. Find the specific scrollable element
        scroll_container = WebDriverWait(driver, 20).until(
            EC.presence_of_element_located(
                (By.CSS_SELECTOR, "app-suche ion-content"))
        )
        print("Found the scrollable content container.")

        # 2. Loop scrolling until the "no more results" message is found
        while True:
            # This JS command targets the inner scrollable part of the ion-content component
            # and scrolls it to the bottom. This is the correct way to trigger lazy loading.
            driver.execute_script(
                "arguments[0].scrollToBottom(500);", scroll_container)
            time.sleep(3)  # Wait for network and rendering

            try:
                # Check for the definitive "end of results" message
                end_of_results_msg = driver.find_element(
                    By.XPATH, "//h2[contains(text(), 'Ihre Suche lieferte keine weiteren Treffer.')]")
                if end_of_results_msg.is_displayed():
                    print("Found the 'end of results' message. All content loaded.")
                    break
            except NoSuchElementException:
                current_count = len(driver.find_elements(
                    By.CSS_SELECTOR, "div.search_result"))
                print(f"Scrolling... Found {current_count} results so far.")

        # 3. After all results are loaded, extract the links
        all_loaded_items = driver.find_elements(
            By.CSS_SELECTOR, "div.search_result")
        print(
            f"\nFinished scrolling. Extracting links from {len(all_loaded_items)} loaded results.")

        for item in all_loaded_items:
            try:
                if "Ihre Suche lieferte keine weiteren Treffer" in item.text:
                    continue
                link_elem = item.find_element(By.CSS_SELECTOR, "h2 a.link")
                href = link_elem.get_attribute("href")
                if href and '/leitlinien/detail/' in href:
                    title = link_elem.get_attribute(
                        "title").strip() or link_elem.text.strip()
                    register_number = extract_register_number_from_url(href)
                    if not any(g['url'] == href for g in guideline_links):
                        guideline_links.append(
                            {'url': href, 'title': title, 'register_number': register_number})
            except NoSuchElementException:
                continue

        print(
            f"Successfully extracted {len(guideline_links)} unique guideline links.")

        if test_mode and max_guidelines and len(guideline_links) > max_guidelines:
            print(
                f"Test mode: limiting to {max_guidelines} guidelines from search results.")
            return guideline_links[:max_guidelines]

    except TimeoutException:
        print("Timeout waiting for initial search results to load.")
    except Exception as e:
        print(f"An error occurred while scraping the search page: {e}")

    return guideline_links


def extract_pdf_links_with_selenium(driver, guideline_url, guideline_title, register_number=""):
    """Extracts PDF links and metadata from a guideline detail page."""
    pdf_links = []
    actual_guideline_title = guideline_title

    try:
        print(f"Navigating to {guideline_url}")
        driver.get(guideline_url)
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.TAG_NAME, "ion-row")))
        time.sleep(3)

        # --- Title Extraction Logic ---
        title_found = False
        try:
            guideline_details_title = driver.find_element(
                By.CSS_SELECTOR, "div.guideline-details h1")
            if guideline_details_title and guideline_details_title.text.strip():
                actual_guideline_title = guideline_details_title.text.strip()
                title_found = True
        except NoSuchElementException:
            pass

        if not title_found:
            try:
                meta_title = driver.find_element(
                    By.CSS_SELECTOR, "meta[property='og:title']")
                if meta_title:
                    title_content = meta_title.get_attribute("content")
                    if title_content and len(title_content) > 10:
                        actual_guideline_title = title_content.strip()
                        title_found = True
            except:
                pass

        if title_found:
            print(f"Identified guideline title: {actual_guideline_title}")

        all_rows = driver.find_elements(By.TAG_NAME, "ion-row")
        tags_list = []
        try:
            for row in all_rows:
                if "Schlüsselwörter:" in row.text:
                    columns = row.find_elements(By.TAG_NAME, "ion-col")
                    if len(columns) > 1:
                        raw_tags = columns[1].text.strip()
                        tags_list.extend([tag.strip() for tag in raw_tags.replace(
                            '\n', ',').split(',') if tag.strip()])
                        print(f"Found {len(tags_list)} keywords.")
                        break
        except Exception as e:
            print(f"Could not extract keywords: {e}")

        found_patientenleitlinie = False
        for row in all_rows:
            try:
                if "Patientenleitlinie" in row.text:
                    link_element = row.find_element(By.TAG_NAME, "a")
                    href = link_element.get_attribute("href")
                    if href and href.lower().endswith('.pdf'):
                        found_patientenleitlinie = True
                        url_filename = extract_filename_from_url(href)
                        base_name = f"{register_number}_{url_filename}" if register_number else url_filename
                        filename = f"{base_name}_Patientenleitlinie.pdf"
                        if filename.lower().endswith(".pdf.pdf"):
                            filename = filename[:-4]

                        pdf_links.append({
                            'url': href, 'name': filename, 'type': "Patientenleitlinie",
                            'guideline_title': actual_guideline_title,
                            'guideline_id': register_number or extract_register_number_from_url(guideline_url),
                            'tags': tags_list
                        })
                        print(
                            f"Found Patientenleitlinie PDF: {filename} - {href}")
                        break
            except NoSuchElementException:
                continue

        if not found_patientenleitlinie:
            print(f"No 'Patientenleitlinie' PDF found on this page.")
    except Exception as e:
        print(f"Error extracting PDF links from {guideline_url}: {e}")
    return pdf_links


def save_consolidated_metadata(all_pdf_metadata):
    """Saves consolidated metadata about all downloaded PDFs."""
    ensure_dir(METADATA_OUTPUT_DIR)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metadata_file = os.path.join(
        METADATA_OUTPUT_DIR, f"awmf_pdfs_metadata_{timestamp}.json")

    guidelines = {}
    for pdf in all_pdf_metadata:
        guideline_id = pdf.get('guideline_id') or extract_register_number_from_url(
            pdf.get('source_page', ''))
        if not guideline_id:
            guideline_id = pdf.get('source_page', 'unknown')

        if guideline_id not in guidelines:
            guidelines[guideline_id] = {
                'id': guideline_id, 'title': pdf.get('guideline_title', ''),
                'url': pdf.get('source_page', ''), 'documents': []
            }
        guidelines[guideline_id]['documents'].append(pdf)

    consolidated_data = {
        "collection_info": {"timestamp": datetime.now().isoformat(), "total_pdfs": len(all_pdf_metadata), "total_guidelines": len(guidelines)},
        "guidelines": list(guidelines.values())
    }
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(consolidated_data, f, ensure_ascii=False, indent=2)
    print(f"Saved consolidated JSON metadata to: {metadata_file}")

    csv_file = os.path.join(METADATA_OUTPUT_DIR,
                            f"awmf_pdfs_index_{timestamp}.csv")
    try:
        with open(csv_file, 'w', encoding='utf-8') as f:
            f.write("Guideline ID,Guideline Title,Document Type,Filename,File Path,Download URL,Source Page URL,Download Date,Download Timestamp,File Size (bytes),Tags\n")
            for _, guideline_info in guidelines.items():
                for doc in guideline_info['documents']:
                    tags_str = "; ".join(doc.get('tags', []))
                    f.write(
                        f"\"{guideline_info['id']}\",\"{doc.get('guideline_title', '')}\",\"{doc.get('type', '')}\",\"{doc.get('downloaded_filename', '')}\",\"{doc.get('file_path', '')}\",\"{doc.get('download_url', '')}\",\"{doc.get('source_page_url', '')}\",\"{doc.get('download_date', '')}\",\"{doc.get('download_timestamp', '')}\",\"{doc.get('file_size_bytes', '')}\",\"{tags_str}\"\n"
                    )
        print(f"Created CSV index at: {csv_file}")
    except Exception as e:
        print(f"Error creating CSV index: {e}")


def scrape_awmf_sequentially(browser='chrome', headless=True, test_mode=False,
                             max_guidelines=None, max_pdfs=None):
    """Main function to scrape the AWMF website sequentially."""
    if not SELENIUM_AVAILABLE:
        print(
            "Selenium not installed. Please install: pip install selenium webdriver-manager")
        return 0

    print(
        f"Starting AWMF scraper. Output directory: {os.path.abspath(PDF_OUTPUT_DIR)}")
    if test_mode:
        print(
            f"RUNNING IN TEST MODE - Limited to {max_guidelines} guidelines and {max_pdfs} PDFs total")

    ensure_dir(PDF_OUTPUT_DIR)

    driver = None
    all_pdf_metadata = []
    failed_downloads = []
    try:
        driver = setup_webdriver(headless=headless, browser=browser)

        print("\nStep 1: Extracting all guideline links from the search page...")
        all_guideline_links = extract_guideline_links_from_search(
            driver, SEARCH_URL, test_mode=test_mode, max_guidelines=max_guidelines
        )
        print(f"\nFound {len(all_guideline_links)} guideline links.")

        print(
            f"\nStep 2: Processing {len(all_guideline_links)} guideline pages...")

        guideline_iterator = all_guideline_links
        if TQDM_AVAILABLE:
            guideline_iterator = tqdm(
                all_guideline_links, desc="Processing Guidelines")

        for guideline_info in guideline_iterator:
            if test_mode and max_pdfs and len(all_pdf_metadata) >= max_pdfs:
                print(f"Test mode: reached PDF limit of {max_pdfs}.")
                break

            pdf_links_found = extract_pdf_links_with_selenium(
                driver,
                guideline_url=guideline_info['url'],
                guideline_title=guideline_info['title'],
                register_number=guideline_info.get('register_number', '')
            )

            for pdf_info in pdf_links_found:
                pdf_info['source_page'] = guideline_info['url']
                success, _, updated_metadata = download_pdf(
                    pdf_info['url'], pdf_info['name'], pdf_info
                )
                if success:
                    all_pdf_metadata.append(updated_metadata)
                else:
                    failed_downloads.append(pdf_info)

        if all_pdf_metadata:
            save_consolidated_metadata(all_pdf_metadata)
            print(
                f"\nScraping complete. Successfully downloaded {len(all_pdf_metadata)} PDFs.")
        else:
            print(
                "\nScraping complete. No 'Patientenleitlinie' PDFs were found or downloaded.")

        if failed_downloads:
            print("\n--- Summary of Failed Downloads ---")
            for failed_item in failed_downloads:
                print(
                    f"Could not download PDF for guideline: {failed_item.get('guideline_title', 'N/A')}")
                print(f"  - URL: {failed_item.get('url', 'N/A')}")
            print("------------------------------------")

    finally:
        if driver:
            driver.quit()
            print("\nBrowser closed.")


def main():
    parser = argparse.ArgumentParser(
        description='Scrape AWMF website for "Patientenleitlinie" PDFs.')
    parser.add_argument('--browser', choices=['chrome', 'firefox'], default='chrome',
                        help='Browser to use for Selenium (default: chrome)')
    parser.add_argument('--headless', action='store_true',
                        help='Run the browser in headless mode')
    parser.add_argument('--test', action='store_true',
                        help='Run in test mode with limited downloads')
    parser.add_argument('--max-guidelines', type=int,
                        help='Max guidelines to process in test mode')
    parser.add_argument('--max-pdfs', type=int,
                        help='Max total PDFs to download in test mode')

    args = parser.parse_args()

    scrape_awmf_sequentially(
        browser=args.browser,
        headless=args.headless,
        test_mode=args.test,
        max_guidelines=args.max_guidelines,
        max_pdfs=args.max_pdfs,
    )


if __name__ == "__main__":
    main()
