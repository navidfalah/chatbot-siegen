"""
AWMF Web Scraper using Selenium for JavaScript rendering with multithreading support.
This implementation uses a web browser automation tool (Selenium) to properly render the
JavaScript-based website before scraping content, and uses multithreading to improve performance.
"""

import os
import time
import argparse
import re
import json
from datetime import datetime
from urllib.parse import urljoin
import concurrent.futures
import threading
import queue
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

from utils import ensure_dir

# Base URLs for the AWMF website
MAIN_URL = "https://register.awmf.org"
FACHGESELLSCHAFT_BASE_URL = "https://register.awmf.org/de/leitlinien/aktuelle-leitlinien/fachgesellschaft"

# Define output directories
# Relative to the src directory
PDF_OUTPUT_DIR = os.path.join("data", "pdfs")
METADATA_OUTPUT_DIR = os.path.join(
    "data", "metadata")  # Directory for metadata files

# Define thread-safe queues and locks
guideline_links_queue = queue.Queue()
pdf_links_queue = queue.Queue()
pdf_metadata_list_lock = threading.Lock()
guideline_links_lock = threading.Lock()
pdf_links_lock = threading.Lock()
webdriver_lock = threading.Lock()  # For thread-safe webdriver operations
filename_lock = threading.Lock()   # For thread-safe filename operations
filename_registry = {}             # To track filenames being processed across threads


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


def download_pdf(pdf_url, base_filename, metadata=None, thread_id=None):
    """
    Downloads a PDF from a given URL and saves it to the specified filename.
    Handles duplicate filenames by appending a counter.

    Args:
        pdf_url: URL of the PDF to download
        base_filename: Base filename to save the PDF as
        metadata: Dictionary containing metadata about the PDF
        thread_id: Optional thread identifier for logging

    Returns:
        Tuple of (success_bool, actual_filename, updated_metadata) where
        success_bool is True if download was successful
        actual_filename is the final filename used (may differ from base_filename if duplicate)
        updated_metadata is the metadata with added download information
    """
    thread_prefix = f"[Thread-{thread_id}] " if thread_id is not None else ""

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

    # In the extremely unlikely case that even with the hash there's a collision
    # (could happen if the same URL is downloaded simultaneously), add a counter
    with filename_lock:  # Use a lock to prevent race conditions when checking/creating files
        while os.path.exists(filepath):
            new_filename = f"{base_name}_{counter}_{url_hash}{extension}"
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
                f"{thread_prefix}Warning: URL {pdf_url} does not return a PDF (Content-Type: {content_type})")

        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)        # Save metadata if provided
        print(f"{thread_prefix}Successfully downloaded: {final_filename}")
        if metadata:
            # Add enhanced metadata for RAG applications
            current_time = datetime.now()
            metadata['download_timestamp'] = current_time.isoformat()
            metadata['download_date'] = current_time.strftime(
                "%Y-%m-%d")  # Human-readable date
            metadata['download_url'] = pdf_url  # The actual PDF download link
            metadata['source_page_url'] = metadata.get(
                'source_page', '')  # Page where download link was found
            metadata['downloaded_filename'] = final_filename
            metadata['file_path'] = os.path.join(
                PDF_OUTPUT_DIR, final_filename)
            metadata['file_size_bytes'] = os.path.getsize(filepath)
            metadata['content_type'] = content_type
            # Add the hash used for filename creation
            metadata['url_hash'] = url_hash

            # Extract source guideline information more cleanly
            source_url = metadata.get('source_page', '')
            if not metadata.get('guideline_id'):
                metadata['guideline_id'] = extract_register_number_from_url(
                    source_url)

            # Make sure the title is preserved
            if 'guideline_title' in metadata and metadata['guideline_title']:
                # Use the title already in metadata
                pass
            elif source_url:
                # Try to extract a better title from the URL if possible
                url_parts = source_url.split('/')
                if len(url_parts) > 2:
                    metadata['source_page_name'] = url_parts[-1]

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
            print(f"{thread_prefix}Saved metadata to: {metadata_filename}")

        return True, final_filename, metadata
    except requests.exceptions.RequestException as e:
        print(f"{thread_prefix}Error downloading {pdf_url}: {e}")
    except Exception as e:
        print(
            f"{thread_prefix}An unexpected error occurred while downloading {pdf_url}: {e}")
    return False, None, metadata


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


def extract_guideline_links_with_selenium(driver, fach_url, thread_id=None):
    """Extract links to guideline detail pages using Selenium."""
    thread_prefix = f"[Thread-{thread_id}] " if thread_id is not None else ""
    guideline_links = []
    max_retries = 3
    retry_count = 0

    while retry_count < max_retries:
        try:
            # Use webdriver_lock to make browser interactions thread-safe
            with webdriver_lock:
                # Navigate to the page
                print(f"{thread_prefix}Navigating to {fach_url}")
                driver.get(fach_url)

                # Wait for the page to load (adjust timeout as needed)
                WebDriverWait(driver, 15).until(
                    EC.presence_of_element_located(
                        (By.CSS_SELECTOR, "ion-row"))
                )

                # Allow some extra time for all content to render
                time.sleep(3)

                # Find all guideline links
                rows = driver.find_elements(
                    By.CSS_SELECTOR, "ion-row.guideline-listing-row")
                print(f"{thread_prefix}Found {len(rows)} guideline rows")

                if not rows:
                    # Try a secondary method if no rows found
                    print(
                        f"{thread_prefix}No rows found with primary selector, trying alternate approach...")
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
                                f"{thread_prefix}Found guideline: {title or 'Unknown Title'} - {href}")
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
                                cols = row.find_elements(
                                    By.TAG_NAME, "ion-col")
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
                                    f"{thread_prefix}Found guideline: {title or 'Unknown Title'} - {href}")
                        except NoSuchElementException:
                            continue  # Skip rows without links

            break  # Successfully retrieved links, break out of retry loop

        except TimeoutException:
            retry_count += 1
            print(
                f"{thread_prefix}Timeout waiting for page {fach_url} to load (attempt {retry_count}/{max_retries})")
            if retry_count >= max_retries:
                print(
                    f"{thread_prefix}Failed to load {fach_url} after {max_retries} attempts")
                break
            time.sleep(2)  # Wait before retrying

        except Exception as e:
            print(
                f"{thread_prefix}Error extracting guideline links from {fach_url}: {e}")
            break

    return guideline_links


def extract_pdf_links_with_selenium(driver, guideline_url, guideline_title, register_number="", thread_id=None):
    """Extract PDF links from a guideline page using Selenium."""
    thread_prefix = f"[Thread-{thread_id}] " if thread_id is not None else ""
    pdf_links = []
    actual_guideline_title = guideline_title  # Default to the provided title

    try:
        # Use webdriver_lock to make browser interactions thread-safe
        with webdriver_lock:
            # Navigate to the page
            print(f"{thread_prefix}Navigating to {guideline_url}")
            driver.get(guideline_url)

            # Wait for the page to load (adjust timeout as needed)
            WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.TAG_NAME, "a"))
            )

            # Allow some extra time for all content to render
            time.sleep(3)

            # Enhanced title extraction with multiple fallback strategies
            title_found = False

            # Strategy 1: Look for guideline-details div with h1
            try:
                guideline_details_title = driver.find_element(
                    By.CSS_SELECTOR, "div.guideline-details h1")
                if guideline_details_title and guideline_details_title.text.strip():
                    actual_guideline_title = guideline_details_title.text.strip()
                    print(
                        f"{thread_prefix}Found guideline title from guideline-details: {actual_guideline_title}")
                    title_found = True
            except NoSuchElementException:
                pass

            # Strategy 2: Look for title div or section with specific classes
            if not title_found:
                try:
                    title_elements = driver.find_elements(
                        By.CSS_SELECTOR, "div.guideline-title, div.main-title, h1.main-title, .title-section h1")
                    for title_elem in title_elements:
                        if title_elem and title_elem.text.strip():
                            actual_guideline_title = title_elem.text.strip()
                            print(
                                f"{thread_prefix}Found guideline title from title element: {actual_guideline_title}")
                            title_found = True
                            break
                except:
                    pass

            # Strategy 3: Try all H1 elements
            if not title_found:
                try:
                    h1_elements = driver.find_elements(By.TAG_NAME, "h1")
                    for h1 in h1_elements:
                        title_text = h1.text.strip()
                        # Filter out generic titles and navigation headers
                        if (title_text and
                            len(title_text) > 10 and  # Avoid very short titles
                            "AWMF" not in title_text and  # Avoid site headers
                            "Suche" not in title_text and  # Avoid search headers
                                "Navigation" not in title_text):  # Avoid navigation headers
                            actual_guideline_title = title_text
                            print(
                                f"{thread_prefix}Found guideline title from H1: {actual_guideline_title}")
                            title_found = True
                            break
                except:
                    pass

            # Strategy 4: Extract from meta tags
            if not title_found:
                try:
                    meta_title = driver.find_element(
                        By.CSS_SELECTOR, "meta[property='og:title']")
                    if meta_title:
                        title_content = meta_title.get_attribute("content")
                        if title_content and len(title_content) > 10:
                            actual_guideline_title = title_content.strip()
                            print(
                                f"{thread_prefix}Found guideline title from meta tag: {actual_guideline_title}")
                            title_found = True
                except:
                    pass
                  # If we still don't have a title, try to extract from the URL or register number
            if not title_found and register_number:
                actual_guideline_title = f"Leitlinie {register_number}"
                print(
                    f"{thread_prefix}Using register number as title: {actual_guideline_title}")
            elif not title_found:
                print(
                    f"{thread_prefix}Could not find guideline title, using provided title: {guideline_title}")

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
                            context = driver.execute_script(
                                parent_script, link)
                            if context:
                                link_text = context
                            else:
                                # Use the actual guideline title if we found one
                                link_text = actual_guideline_title                        # Determine PDF type
                        pdf_type = ""
                        href_lower = href.lower()
                        link_text_lower = link_text.lower() if link_text else ""

                        # Enhanced detection of PDF types with more patterns
                        if any(term in href_lower or term in link_text_lower for term in
                               ["langfassung", "ll_lf", "lang", "_lf_", "lf.", "langversion", "long version"]):
                            pdf_type = "Langfassung"
                        elif any(term in href_lower or term in link_text_lower for term in
                                 ["kurzfassung", "ll_kf", "kurz", "_kf_", "kf.", "kurzversion", "short version"]):
                            pdf_type = "Kurzfassung"
                        elif any(term in href_lower or term in link_text_lower for term in
                                 ["patienten", "patient", "patientenleitlinie", "patientenversion"]):
                            pdf_type = "Patientenleitlinie"
                        elif any(term in href_lower or term in link_text_lower for term in
                                 ["english", "englisch", "en_", "_en", "engl"]):
                            pdf_type = "English Version"
                        elif any(term in href_lower or term in link_text_lower for term in
                                 ["praxishilfe", "praxis", "hilfe", "tool", "formular"]):
                            # Extract a better filename from the URL instead of using link text
                            pdf_type = "Praxishilfe"
                        url_filename = extract_filename_from_url(href)

                        # Create a descriptive filename - prioritize the URL filename over link text
                        if url_filename and len(url_filename) > 5:
                            # Check if register number is already in the URL filename
                            if register_number and register_number not in url_filename:
                                base_name = f"{register_number}_{url_filename}"
                            else:
                                base_name = url_filename
                        else:
                            # Fallback to link text if URL extraction failed
                            # Avoid generic link texts like "Download", "PDF", "weiterlesen", etc.
                            generic_texts = ["download", "pdf", "weiterlesen",
                                             "herunterladen", "lesen", "click", "here", "link"]
                            if link_text and not any(text in link_text.lower() for text in generic_texts) and len(link_text) > 5:
                                if register_number:
                                    base_name = f"{register_number}_{sanitize_filename(link_text)}"
                                else:
                                    base_name = sanitize_filename(link_text)
                            else:
                                # Use guideline title with register number as last resort
                                if register_number:
                                    if actual_guideline_title:
                                        base_name = f"{register_number}_{sanitize_filename(actual_guideline_title)}"
                                    else:
                                        base_name = f"{register_number}_document"
                                else:
                                    # Ultimate fallback
                                    # Add PDF type if not already in the name and clean up any duplicate information
                                    base_name = sanitize_filename(
                                        actual_guideline_title) if actual_guideline_title else "unknown_document"
                        if pdf_type and pdf_type.lower() not in base_name.lower():
                            filename = f"{base_name}_{pdf_type}.pdf"
                        else:
                            filename = f"{base_name}.pdf"

                        # Ensure we don't have ".pdf.pdf" in the filename
                        if filename.lower().endswith(".pdf.pdf"):
                            filename = filename[:-4]
                            # Create final metadata entry
                        pdf_links.append({
                            'url': href,
                            'name': filename,
                            'type': pdf_type,
                            'original_url_filename': url_filename,  # Store the original filename from URL
                            'link_text': link_text,  # Store the original link text
                            'guideline_title': actual_guideline_title,  # Store the actual guideline title
                            'guideline_id': register_number if register_number else extract_register_number_from_url(guideline_url)
                        })
                        print(f"{thread_prefix}Found PDF: {filename} - {href}")
                except Exception as e:
                    print(f"{thread_prefix}Error processing link: {e}")
                    continue

    except TimeoutException:
        print(f"{thread_prefix}Timeout waiting for page {guideline_url} to load")
    except Exception as e:
        print(f"{thread_prefix}Error extracting PDF links from {guideline_url}: {e}")

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
        guideline_id = pdf.get('guideline_id') or pdf.get(
            'register_number') or extract_register_number_from_url(source_url)
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
            # PDF download link
            'download_url': pdf.get('download_url', pdf.get('pdf_url', pdf.get('url', ''))),
            # Page where link was found
            'source_page_url': pdf.get('source_page_url', pdf.get('source_page', '')),
            # Human-readable date
            'download_date': pdf.get('download_date', ''),
            # ISO timestamp
            'download_timestamp': pdf.get('download_timestamp', ''),
            'file_size_bytes': pdf.get('file_size_bytes', 0),
            'guideline_title': pdf.get('guideline_title', ''),
            'guideline_id': pdf.get('guideline_id', guideline_id)
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
            # Enhanced CSV header with all relevant metadata fields
            f.write(
                "Guideline ID,Guideline Title,Document Type,Filename,File Path,Download URL,Source Page URL,Download Date,Download Timestamp,File Size (bytes)\n")

            for guideline_id, guideline_info in guidelines.items():
                for doc in guideline_info['documents']:
                    # Write a row with all important metadata fields
                    f.write(
                        f"\"{guideline_info['id']}\",\"{guideline_info['title']}\",\"{doc['type']}\",\"{doc['filename']}\",\"{doc['file_path']}\",\"{doc['download_url']}\",\"{doc['source_page_url']}\",\"{doc['download_date']}\",\"{doc['download_timestamp']}\",\"{doc['file_size_bytes']}\"\n")

        print(f"Created enhanced CSV index at: {csv_file}")
    except Exception as e:
        print(f"Error creating CSV index: {e}")


# Thread worker functions

def guideline_link_worker(driver, fach_url, all_guideline_links, max_guidelines=None, test_mode=False, thread_id=None):
    """Worker function to extract guideline links from a fachgesellschaft page."""
    try:
        guideline_links = extract_guideline_links_with_selenium(
            driver, fach_url, thread_id)

        # Apply test mode limit if requested
        if test_mode and max_guidelines and max_guidelines < len(guideline_links):
            print(
                f"[Thread-{thread_id}] Test mode: limiting to {max_guidelines} guidelines")
            guideline_links = guideline_links[:max_guidelines]

        # Add to shared list in a thread-safe manner
        with guideline_links_lock:
            all_guideline_links.extend(guideline_links)

    except Exception as e:
        print(f"[Thread-{thread_id}] Error in guideline link worker: {e}")


def pdf_link_worker(driver, guideline_info, all_pdf_links, thread_id=None):
    """Worker function to extract PDF links from a guideline page."""
    try:
        url = guideline_info['url']
        title = guideline_info['title']
        register_number = guideline_info.get('register_number', '')

        pdf_links = extract_pdf_links_with_selenium(
            driver, url, title, register_number, thread_id)

        if pdf_links:
            for pdf in pdf_links:
                # Add source page information
                pdf['source_page'] = url

                # Ensure proper metadata propagation
                # If we found a better title on the detail page, propagate it back to the guideline info
                if 'guideline_title' in pdf and pdf['guideline_title'] and pdf['guideline_title'] != "Unknown Title":
                    # Update the guideline_info title with the more accurate one from the detail page
                    guideline_info['title'] = pdf['guideline_title']
                    print(
                        f"[Thread-{thread_id}] Updated guideline title to: {pdf['guideline_title']}")
                else:
                    # If the PDF doesn't have a good title, use the guideline info title
                    pdf['guideline_title'] = guideline_info['title']

                # Ensure register number is available
                if register_number:
                    pdf['register_number'] = register_number
                    pdf['guideline_id'] = register_number

                # Add to shared list in a thread-safe manner
                with pdf_links_lock:
                    all_pdf_links.append(pdf)

    except Exception as e:
        print(f"[Thread-{thread_id}] Error in PDF link worker: {e}")


def pdf_download_worker(pdf_info, all_pdf_metadata, thread_id=None):
    """Worker function to download a PDF and save its metadata."""
    try:
        if not pdf_info.get('url'):  # Skip entries with missing URLs
            return False

        name = pdf_info.get('name', 'unknown.pdf')
        print(
            f"[Thread-{thread_id}] Downloading PDF: {name} from {pdf_info['url']}")

        success, filename, updated_metadata = download_pdf(
            pdf_info['url'], name, pdf_info, thread_id)

        if success:
            # Add to shared list in a thread-safe manner
            with pdf_metadata_list_lock:
                all_pdf_metadata.append(updated_metadata)
            return True
        return False

    except Exception as e:
        print(f"[Thread-{thread_id}] Error in PDF download worker: {e}")
        return False


def create_browser_pool(num_browsers, headless=True, browser='chrome'):
    """Creates a pool of browser instances for parallel processing."""
    browser_pool = []

    # Set up cleanup function to ensure all browsers are closed
    def cleanup_browsers():
        for driver in browser_pool:
            try:
                driver.quit()
                print("Cleaned up browser instance")
            except:
                pass

    # Register the cleanup function to run at exit
    atexit.register(cleanup_browsers)

    # Initialize browsers with progress bar if available
    if TQDM_AVAILABLE:
        browser_range = tqdm(range(num_browsers),
                             desc="Initializing browsers", unit="browser")
    else:
        browser_range = range(num_browsers)
        print(f"Initializing {num_browsers} browser instances...")

    for i in browser_range:
        try:
            driver = setup_webdriver(headless=headless, browser=browser)
            browser_pool.append(driver)
            if not TQDM_AVAILABLE:
                print(f"Initialized browser {i+1}/{num_browsers}")
        except Exception as e:
            print(f"Failed to initialize browser {i+1}: {e}")

    print(
        f"Successfully initialized {len(browser_pool)} of {num_browsers} requested browsers")
    return browser_pool


def scrape_awmf_with_selenium_threaded(browser='chrome', headless=True, test_mode=False,
                                       max_fachgesellschaft=None, max_guidelines=None, max_pdfs=None,
                                       max_workers=4):
    """
    Main function to scrape the AWMF website using Selenium with multithreading.

    Args:
        browser (str): Browser to use ('chrome' or 'firefox')
        headless (bool): Whether to run the browser in headless mode
        test_mode (bool): If True, enable test mode with limited downloads
        max_fachgesellschaft (int): Max number of fachgesellschaft pages to process
        max_guidelines (int): Max guidelines per fachgesellschaft
        max_pdfs (int): Max number of PDFs to download
        max_workers (int): Max number of worker threads to use
    """
    if not SELENIUM_AVAILABLE:
        print(
            "Selenium is not installed. Please install it first with: pip install selenium")
        return 0

    print(
        f"Starting AWMF scraper with Selenium (Threaded). Output directory: {os.path.abspath(PDF_OUTPUT_DIR)}")
    print(f"Using up to {max_workers} worker threads")

    if test_mode:
        print(f"RUNNING IN TEST MODE - Limited to {max_fachgesellschaft} fachgesellschaft pages, " +
              f"{max_guidelines} guidelines per fachgesellschaft, and {max_pdfs} PDFs total")

    ensure_dir(PDF_OUTPUT_DIR)

    # Adjust number of browsers based on workers but keep it reasonable
    # Don't create too many browser instances
    num_browsers = min(max_workers, 4)
    print(f"Creating {num_browsers} browser instances for parallel processing")

    # Create a pool of browsers
    browser_pool = create_browser_pool(num_browsers, headless, browser)

    if not browser_pool:
        print("Failed to create any browser instances. Exiting.")
        return 0

    try:
        # Step 1: Get fachgesellschaft IDs
        fachgesellschaft_ids = get_fachgesellschaft_ids()

        if test_mode and max_fachgesellschaft and max_fachgesellschaft < len(fachgesellschaft_ids):
            print(
                f"Test mode: limiting to {max_fachgesellschaft} fachgesellschaft pages")
            fachgesellschaft_ids = fachgesellschaft_ids[:max_fachgesellschaft]

        all_guideline_links = []
        all_pdf_links = []
        # Step 2: For each fachgesellschaft page, extract guideline links using thread pool
        all_pdf_metadata = []
        print(
            f"\nProcessing {len(fachgesellschaft_ids)} fachgesellschaft pages with parallel workers")

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(browser_pool), len(fachgesellschaft_ids))) as executor:
            # Submit tasks to the executor
            futures = []
            for idx, fach_id in enumerate(fachgesellschaft_ids):
                fach_url = f"{FACHGESELLSCHAFT_BASE_URL}/{fach_id}"
                print(
                    f"Submitting Fachgesellschaft {idx+1}/{len(fachgesellschaft_ids)}: {fach_id}")

                # Use modulo to distribute work among available browsers
                browser_idx = idx % len(browser_pool)
                future = executor.submit(
                    guideline_link_worker,
                    browser_pool[browser_idx],
                    fach_url,
                    all_guideline_links,
                    max_guidelines,
                    test_mode,
                    idx
                )
                futures.append(future)

            # Wait for all futures to complete with progress bar if available
            completed = 0
            total = len(futures)

            if TQDM_AVAILABLE:
                pbar = tqdm(
                    total=total, desc="Processing fachgesellschaft pages", unit="page")

            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()  # This will raise any exceptions that occurred in the thread
                    completed += 1
                    if TQDM_AVAILABLE:
                        pbar.update(1)
                    else:
                        print(
                            f"Completed {completed}/{total} fachgesellschaft pages")
                except Exception as e:
                    print(f"Error in guideline extraction thread: {e}")
                    if TQDM_AVAILABLE:
                        pbar.update(1)

            if TQDM_AVAILABLE:
                pbar.close()

        print(f"\nFound a total of {len(all_guideline_links)} guideline links")

        # Apply global test mode limit if needed
        if test_mode and max_guidelines and max_fachgesellschaft and len(all_guideline_links) > max_guidelines * max_fachgesellschaft:
            all_guideline_links = all_guideline_links[:
                                                      max_guidelines * max_fachgesellschaft]
            # Step 3: For each guideline page, extract PDF links using thread pool
            print(
                f"Test mode: limited to {len(all_guideline_links)} guidelines total")
        print(
            f"\nProcessing {len(all_guideline_links)} guideline pages with parallel workers")

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(browser_pool), max_workers)) as executor:
            # Submit tasks to the executor
            futures = []
            for idx, guideline in enumerate(all_guideline_links):
                print(
                    f"Submitting guideline {idx+1}/{len(all_guideline_links)}: {guideline['title']}")

                # Use modulo to distribute work among available browsers
                browser_idx = idx % len(browser_pool)
                future = executor.submit(
                    pdf_link_worker,
                    browser_pool[browser_idx],
                    guideline,
                    all_pdf_links,
                    idx
                )
                futures.append(future)

                # Check if we've reached the PDF limit in test mode
                if test_mode and max_pdfs and len(all_pdf_links) >= max_pdfs:
                    print(f"Test mode: reached limit of {max_pdfs} PDFs")
                    # Cancel remaining futures
                    for f in futures[idx+1:]:
                        f.cancel()
                    break

            # Wait for all futures to complete with progress bar if available
            completed = 0
            total = len(futures)

            if TQDM_AVAILABLE:
                pbar = tqdm(
                    total=total, desc="Processing guideline pages", unit="guideline")

            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()  # This will raise any exceptions that occurred in the thread
                    completed += 1
                    if TQDM_AVAILABLE:
                        pbar.update(1)
                    else:
                        print(f"Completed {completed}/{total} guideline pages")
                except Exception as e:
                    print(f"Error in PDF link extraction thread: {e}")
                    if TQDM_AVAILABLE:
                        pbar.update(1)

            if TQDM_AVAILABLE:
                pbar.close()

        # Apply test mode limit for PDFs if needed
        if test_mode and max_pdfs and len(all_pdf_links) > max_pdfs:
            all_pdf_links = all_pdf_links[:max_pdfs]
            # Step 4: Download all PDFs in parallel
            print(f"Test mode: limited to {len(all_pdf_links)} PDFs")
        print(f"\nFound a total of {len(all_pdf_links)} PDF links to download")
        print(f"\nDownloading {len(all_pdf_links)} PDFs with parallel workers")

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Enhance PDF metadata before download - find corresponding guideline info
            for pdf_info in all_pdf_links:
                guideline_info = next(
                    (g for g in all_guideline_links if g['url'] == pdf_info.get('source_page')), None)

                if guideline_info:
                    # Make sure we're using the best title available
                    if pdf_info.get('guideline_title') and pdf_info['guideline_title'] != "Unknown Title":
                        # If the PDF has a good title extracted from the detail page, use it
                        # And also update the guideline_info for future reference
                        guideline_info['title'] = pdf_info['guideline_title']
                    elif guideline_info.get('title') and guideline_info['title'] != "Unknown Title":
                        # If the guideline has a good title, use it for the PDF
                        pdf_info['guideline_title'] = guideline_info['title']

                    # Make sure register number/guideline ID is consistent
                    if guideline_info.get('register_number'):
                        pdf_info['guideline_id'] = guideline_info['register_number']
                    elif pdf_info.get('guideline_id'):
                        guideline_info['register_number'] = pdf_info['guideline_id']

            # Submit tasks to the executor
            futures = []
            for idx, pdf_info in enumerate(all_pdf_links):
                print(
                    f"Submitting PDF download {idx+1}/{len(all_pdf_links)}: {pdf_info.get('name', 'unknown.pdf')}")
                future = executor.submit(
                    pdf_download_worker,
                    pdf_info,
                    all_pdf_metadata,
                    idx
                )
                futures.append(future)

            # Wait for all futures to complete with progress bar if available
            completed = 0
            total = len(futures)
            successful = 0

            if TQDM_AVAILABLE:
                pbar = tqdm(total=total, desc="Downloading PDFs", unit="PDF")

            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()  # This will raise any exceptions that occurred in the thread
                    completed += 1
                    successful += 1 if result else 0
                    if TQDM_AVAILABLE:
                        pbar.update(1)
                        pbar.set_postfix(successful=successful)
                    else:
                        print(
                            f"Completed {completed}/{total} PDF downloads (successful: {successful})")
                except Exception as e:
                    print(f"Error in PDF download thread: {e}")
                    if TQDM_AVAILABLE:
                        pbar.update(1)

            if TQDM_AVAILABLE:
                pbar.close()

            print(
                f"\nSuccessfully downloaded {successful} out of {total} PDFs")

        # Save consolidated metadata
        save_consolidated_metadata(all_pdf_metadata)

        print(
            f"\nScraping complete. Successfully downloaded {len(all_pdf_metadata)} PDFs to {os.path.abspath(PDF_OUTPUT_DIR)}")
        return len(all_pdf_metadata)

    finally:
        # Always make sure to close all browsers
        for driver in browser_pool:
            try:
                driver.quit()
            except:
                pass


def main():
    parser = argparse.ArgumentParser(
        description='Scrape AWMF website for guideline PDFs using Selenium with multithreading')
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
    parser.add_argument('--max-workers', type=int, default=4,
                        help='Max number of worker threads (default: 4)')

    args = parser.parse_args()

    scrape_awmf_with_selenium_threaded(
        browser=args.browser,
        headless=not args.no_headless,
        test_mode=args.test,
        max_fachgesellschaft=args.max_fachgesellschaft,
        max_guidelines=args.max_guidelines,
        max_pdfs=args.max_pdfs,
        max_workers=args.max_workers
    )


if __name__ == "__main__":
    main()
