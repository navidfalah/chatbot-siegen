import requests
import json
import csv
import time
import pandas as pd
from typing import List, Dict, Optional
from datetime import datetime
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MutAtlasApiScraper:
    def __init__(self, delay_between_requests: float = 0.1, max_workers: int = 10):
        self.base_url = "https://intern.mut-foerdern.de/api/v2"
        self.delay = delay_between_requests
        self.max_workers = max_workers
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'de-DE,de;q=0.9,en;q=0.8',
            'Referer': 'https://www.mut-atlas.de/',
            'Origin': 'https://www.mut-atlas.de'
        })
        self.all_locations = []
        self.detailed_data = []
        self.failed_requests = []
        self.lock = threading.Lock()  # For thread-safe operations

    def get_locations_overview(self) -> List[Dict]:
        """
        Get overview of all locations from the zoom endpoint
        """
        url = f"{self.base_url}/angebote/zoom/12"

        try:
            logger.info("Fetching locations overview from zoom level 12")
            response = self.session.get(url, timeout=30)
            response.raise_for_status()

            locations = response.json()  # API returns list directly

            logger.info(f"Found {len(locations)} locations")
            return locations

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching locations overview: {e}")
            return []

    def get_location_details(self, guid: str) -> Optional[Dict]:
        """
        Get detailed information for a specific location using its GUID
        Thread-safe version that creates its own session
        """
        url = f"{self.base_url}/angebote/angebot/{guid}"

        # Create a new session for this thread
        session = requests.Session()
        session.headers.update(self.session.headers)

        try:
            response = session.get(url, timeout=30)
            response.raise_for_status()

            return response.json()

        except requests.exceptions.RequestException as e:
            with self.lock:
                self.failed_requests.append({'guid': guid, 'error': str(e)})
            return None
        finally:
            session.close()

    def fetch_location_with_retry(self, location_data: Dict, max_retries: int = 3) -> Optional[Dict]:
        """
        Fetch location details with retry logic
        """
        guid = location_data['guid']

        for attempt in range(max_retries):
            try:
                details = self.get_location_details(guid)
                if details:
                    # Combine overview and detailed data
                    combined_data = {**location_data, **details}
                    return combined_data
                else:
                    if attempt < max_retries - 1:
                        # Exponential backoff
                        time.sleep(self.delay * (attempt + 1))
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed for {guid}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(self.delay * (attempt + 1))

        return None

    def scrape_all_data(self) -> Dict:
        """
        Scrape all available data using multithreading for faster processing
        """
        logger.info("Starting comprehensive data scraping...")

        # Step 1: Get all locations from zoom level 12
        self.all_locations = self.get_locations_overview()

        if not self.all_locations:
            logger.error("No locations found in overview")
            return {'total_locations': 0, 'failed_requests': 0, 'data': []}

        total_locations = len(self.all_locations)
        logger.info(f"Found {total_locations} locations")
        logger.info(
            f"Starting multithreaded fetching with {self.max_workers} workers...")

        # Step 2: Get detailed data for each location using ThreadPoolExecutor
        completed = 0

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_location = {
                executor.submit(self.fetch_location_with_retry, location): location
                for location in self.all_locations
            }

            # Process completed tasks
            for future in as_completed(future_to_location):
                location = future_to_location[future]
                completed += 1

                try:
                    result = future.result()
                    if result:
                        with self.lock:
                            self.detailed_data.append(result)

                    # Log progress every 100 completions
                    if completed % 100 == 0 or completed == total_locations:
                        success_rate = len(
                            self.detailed_data) / completed * 100
                        logger.info(
                            f"Progress: {completed}/{total_locations} ({completed/total_locations*100:.1f}%) - Success rate: {success_rate:.1f}%")

                except Exception as e:
                    logger.error(
                        f"Error processing location {location.get('name', 'Unknown')}: {e}")

        logger.info(f"Multithreaded scraping completed!")
        logger.info(
            f"Successfully scraped {len(self.detailed_data)} locations with full details")
        logger.info(f"Failed requests: {len(self.failed_requests)}")

        return {
            'total_locations': len(self.detailed_data),
            'failed_requests': len(self.failed_requests),
            'data': self.detailed_data
        }

    def flatten_multifield_values(self, location_data: Dict) -> Dict:
        """
        Flatten the complex mfAngebotMultifieldValues structure into readable columns
        """
        flattened = location_data.copy()

        # Extract multifield values
        multifield_data = location_data.get('mfAngebotMultifieldValues', [])

        # Group by field name
        fields = {}
        for item in multifield_data:
            field_name = item.get('fmMultiField', {}).get('name', '')
            field_value = item.get('fmMultiFieldValue', {}).get('value', '')

            if field_name:
                if field_name not in fields:
                    fields[field_name] = []
                fields[field_name].append(field_value)

        # Add flattened fields
        for field_name, values in fields.items():
            flattened[field_name] = '; '.join(values) if len(
                values) > 1 else (values[0] if values else '')

        # Process opening hours
        opening_hours = location_data.get('openingHours', [])
        days = ['Montag', 'Dienstag', 'Mittwoch',
                'Donnerstag', 'Freitag', 'Samstag', 'Sonntag']

        for i, day_data in enumerate(opening_hours):
            if i < len(days):
                day_name = days[i]
                open1 = day_data.get('open1', '')
                close1 = day_data.get('close1', '')

                if open1 and close1:
                    flattened[f'Öffnungszeiten_{day_name}'] = f"{open1}-{close1}"
                else:
                    flattened[f'Öffnungszeiten_{day_name}'] = 'Geschlossen'

        # Remove the original complex structures
        flattened.pop('mfAngebotMultifieldValues', None)
        flattened.pop('openingHours', None)

        return flattened

    def save_to_csv(self, filename: Optional[str] = None):
        """Save scraped data to CSV with flattened structure"""
        if not self.detailed_data:
            logger.warning("No data to save")
            return

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"mut_atlas_data_{timestamp}.csv"

        # Flatten all data
        flattened_data = [self.flatten_multifield_values(
            loc) for loc in self.detailed_data]

        # Create DataFrame and save
        df = pd.DataFrame(flattened_data)
        df.to_csv(filename, index=False, encoding='utf-8')

        logger.info(f"Data saved to {filename}")
        logger.info(
            f"CSV contains {len(df)} rows and {len(df.columns)} columns")

    def save_to_json(self, filename: Optional[str] = None):
        """Save raw scraped data to JSON"""
        if not self.detailed_data:
            logger.warning("No data to save")
            return

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"mut_atlas_data_{timestamp}.json"

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump({
                'scraped_at': datetime.now().isoformat(),
                'total_locations': len(self.detailed_data),
                'locations': self.detailed_data
            }, f, ensure_ascii=False, indent=2)

        logger.info(f"Raw data saved to {filename}")

    def save_summary_report(self, filename: Optional[str] = None):
        """Generate and save a summary report"""
        if not self.detailed_data:
            logger.warning("No data to generate report")
            return

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"mut_atlas_report_{timestamp}.txt"

        # Analyze data
        total_locations = len(self.detailed_data)
        cities = {}
        categories = {}

        for location in self.detailed_data:
            # Count by city
            city = location.get('ort', 'Unknown')
            cities[city] = cities.get(city, 0) + 1

            # Count by category
            cat = location.get('katWasSum', 'Unknown')
            categories[cat] = categories.get(cat, 0) + 1

        # Write report
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"MUT-ATLAS Scraping Report\n")
            f.write(
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"=" * 50 + "\n\n")

            f.write(f"Total Locations: {total_locations}\n")
            f.write(f"Failed Requests: {len(self.failed_requests)}\n\n")

            f.write("Top 10 Cities by Number of Locations:\n")
            for city, count in sorted(cities.items(), key=lambda x: x[1], reverse=True)[:10]:
                f.write(f"  {city}: {count}\n")

            f.write("\nCategories:\n")
            for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
                f.write(f"  {cat}: {count}\n")

            if self.failed_requests:
                f.write(f"\nFailed Requests ({len(self.failed_requests)}):\n")
                for failed in self.failed_requests[:10]:  # Show first 10
                    f.write(
                        f"  GUID: {failed['guid']} - Error: {failed['error']}\n")

        logger.info(f"Summary report saved to {filename}")

    def get_sample_data(self, n: int = 5) -> List[Dict]:
        """Get a sample of the scraped data for inspection"""
        return self.detailed_data[:n] if self.detailed_data else []


def main():
    """Main function to run the scraper"""
    print("MUT-ATLAS API Scraper (Multithreaded)")
    print("=" * 40)

    # Initialize scraper with multithreading settings
    # Adjust max_workers based on your system and server tolerance
    # 10-20 workers is usually a good balance between speed and being respectful
    scraper = MutAtlasApiScraper(delay_between_requests=0.1, max_workers=15)

    start_time = time.time()

    # Scrape all data
    result = scraper.scrape_all_data()

    end_time = time.time()
    duration = end_time - start_time

    # Print summary
    print(
        f"\nScraping completed in {duration:.2f} seconds ({duration/60:.2f} minutes)!")
    print(f"Total locations: {result['total_locations']}")
    print(f"Failed requests: {result['failed_requests']}")

    if result['total_locations'] > 0:
        print(
            f"Success rate: {result['total_locations']/(result['total_locations']+result['failed_requests'])*100:.1f}%")
        print(
            f"Average time per location: {duration/len(scraper.all_locations):.3f} seconds")

    # Save data in multiple formats
    print("\nSaving data...")
    scraper.save_to_csv()
    scraper.save_to_json()
    scraper.save_summary_report()

    # Show sample data
    sample = scraper.get_sample_data(3)
    if sample:
        print(f"\nSample location data:")
        for i, location in enumerate(sample, 1):
            print(f"\n{i}. {location.get('name', 'Unknown')}")
            print(f"   City: {location.get('ort', 'Unknown')}")
            print(f"   Categories: {location.get('katWasSum', 'Unknown')}")
            print(f"   Phone: {location.get('telefon', 'Not provided')}")
            print(f"   Website: {location.get('homePage', 'Not provided')}")

    print(f"\n🎉 Scraping completed successfully!")


if __name__ == "__main__":
    main()
