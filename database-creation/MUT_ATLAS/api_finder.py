import requests
import json
from playwright.sync_api import sync_playwright
from urllib.parse import urljoin, urlparse


class ApiDiscovery:
    def __init__(self):
        self.api_calls = []
        self.base_url = "https://www.mut-atlas.de"

    def intercept_network_requests(self):
        """Intercept and analyze network requests to find API endpoints"""
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=False)
            page = browser.new_page()

            # Track network requests
            def handle_request(request):
                if any(keyword in request.url.lower() for keyword in ['api', 'graphql', 'json', 'data']):
                    self.api_calls.append({
                        'url': request.url,
                        'method': request.method,
                        'headers': dict(request.headers),
                        'post_data': request.post_data
                    })
                    print(f"API Call detected: {request.method} {request.url}")

            def handle_response(response):
                if response.request.url in [call['url'] for call in self.api_calls]:
                    try:
                        if 'json' in response.headers.get('content-type', ''):
                            content = response.text()
                            print(
                                f"Response from {response.url}: {content[:200]}...")
                    except:
                        pass

            page.on("request", handle_request)
            page.on("response", handle_response)

            # Navigate and interact with the site
            page.goto(self.base_url)
            page.wait_for_timeout(5000)

            # Try to trigger different interactions
            # Click on map, search, filters, etc.
            try:
                # Look for search inputs
                search_inputs = page.query_selector_all(
                    'input[type="search"], input[placeholder*="search"], input[placeholder*="suche"]')
                for search_input in search_inputs:
                    search_input.fill("Berlin")
                    page.keyboard.press("Enter")
                    page.wait_for_timeout(2000)

                # Click on potential filter buttons
                filter_buttons = page.query_selector_all(
                    'button[class*="filter"], .filter-button')
                for btn in filter_buttons[:3]:  # Limit to first 3
                    btn.click()
                    page.wait_for_timeout(1000)

                # Try scrolling to trigger lazy loading
                for i in range(3):
                    page.evaluate(
                        "window.scrollTo(0, document.body.scrollHeight)")
                    page.wait_for_timeout(2000)

            except Exception as e:
                print(f"Error during interaction: {e}")

            browser.close()

        return self.api_calls

    def test_api_endpoints(self):
        """Test discovered API endpoints directly"""
        results = []

        for api_call in self.api_calls:
            try:
                headers = api_call['headers'].copy()
                # Remove browser-specific headers
                headers.pop('user-agent', None)
                headers.pop('sec-ch-ua', None)
                headers.pop('sec-fetch-dest', None)

                if api_call['method'] == 'GET':
                    response = requests.get(api_call['url'], headers=headers)
                elif api_call['method'] == 'POST':
                    response = requests.post(
                        api_call['url'],
                        headers=headers,
                        data=api_call['post_data']
                    )

                if response.status_code == 200:
                    try:
                        data = response.json()
                        results.append({
                            'url': api_call['url'],
                            'method': api_call['method'],
                            'data': data
                        })
                        print(f"✅ Successfully called {api_call['url']}")
                    except:
                        print(f"❌ Non-JSON response from {api_call['url']}")
                else:
                    print(
                        f"❌ Failed {api_call['url']}: {response.status_code}")

            except Exception as e:
                print(f"Error testing {api_call['url']}: {e}")

        return results

    def save_api_data(self, api_results, filename="mut_atlas_api_data.json"):
        """Save API data to file"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(api_results, f, indent=2, ensure_ascii=False)
        print(f"API data saved to {filename}")

# Common API patterns to try


def try_common_api_patterns():
    """Try common API endpoint patterns"""
    base_url = "https://www.mut-atlas.de"

    common_endpoints = [
        "/api/locations",
        "/api/places",
        "/api/search",
        "/api/data",
        "/graphql",
        "/api/v1/locations",
        "/data/locations.json",
        "/locations.json"
    ]

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'de-DE,de;q=0.9,en;q=0.8'
    }

    for endpoint in common_endpoints:
        url = urljoin(base_url, endpoint)
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                print(f"✅ Found working endpoint: {url}")
                try:
                    data = response.json()
                    print(f"   Data preview: {str(data)[:200]}...")
                    return url, data
                except:
                    print(f"   Non-JSON response")
            else:
                print(f"❌ {url}: {response.status_code}")
        except Exception as e:
            print(f"❌ {url}: {e}")

    return None, None


if __name__ == "__main__":
    print("=== Trying common API patterns ===")
    endpoint, data = try_common_api_patterns()

    print("\n=== Intercepting network requests ===")
    discovery = ApiDiscovery()
    api_calls = discovery.intercept_network_requests()

    if api_calls:
        print(f"\nFound {len(api_calls)} API calls:")
        for call in api_calls:
            print(f"  {call['method']} {call['url']}")

        print("\n=== Testing API endpoints ===")
        results = discovery.test_api_endpoints()

        if results:
            discovery.save_api_data(results)
    else:
        print("No API calls detected")
