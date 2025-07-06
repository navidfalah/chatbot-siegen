# ==============================================================================
# SCRIPT: test_rag_api.py
#
# PURPOSE:
#   - Demonstrates how to use the RAG API
#   - Sends test queries and processes the responses
#
# HOW TO USE:
#   1. Ensure the RAG API server is running (uvicorn rag_api:app --host 0.0.0.0 --port 8000)
#   2. Set your API key as environment variable or in .env file
#   3. Run the script: python test_rag_api.py
# ==============================================================================
import os
import sys
import json
import argparse
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Default settings
DEFAULT_API_URL = "http://localhost:8000"
API_KEY = os.getenv("RAG_API_KEY")

# Check if API key is available
if not API_KEY:
    print("Error: No API key found. Please set the RAG_API_KEY environment variable or add it to .env file.")
    sys.exit(1)


def format_location_result(result):
    """Format location result for display"""
    output = []
    output.append(f"Name: {result['name']}")
    if result['address']:
        output.append(f"Address: {result['address']}")
    if result['phone']:
        output.append(f"Phone: {result['phone']}")
    if result['email']:
        output.append(f"Email: {result['email']}")
    if result['website']:
        output.append(f"Website: {result['website']}")
    output.append("\nContext:")
    output.append(result['content'])

    return '\n'.join(output)


def format_guideline_result(result):
    """Format guideline result for display"""
    output = []
    output.append(f"Guideline: {result['title']}")
    if result['url']:
        output.append(f"URL: {result['url']}")
    output.append("\nContext:")
    output.append(result['content'])

    return '\n'.join(output)


def check_api_health(api_url):
    """Check if the API is running"""
    try:
        response = requests.get(f"{api_url}/")
        if response.status_code == 200:
            print(f"API Status: {response.json().get('status', 'unknown')}")
            print(f"Message: {response.json().get('message', 'No message')}")
            return True
        else:
            print(f"Error: API returned status code {response.status_code}")
            print(response.text)
            return False
    except requests.RequestException as e:
        print(f"Error connecting to API: {str(e)}")
        return False


def query_rag_api(api_url, api_key, query, top_k=5, collection=None, debug=False, json_output=False):
    """Send a query to the RAG API and process the response"""
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": api_key
    }

    payload = {
        "query": query,
        "top_k": top_k,
        "debug": debug
    }

    if collection:
        payload["collection"] = collection

    try:
        response = requests.post(
            f"{api_url}/retrieve",
            headers=headers,
            json=payload
        )

        if response.status_code != 200:
            print(f"Error: API returned status code {response.status_code}")
            print(response.text)
            return None

        return response.json()

    except requests.RequestException as e:
        print(f"Error querying API: {str(e)}")
        return None


def main():
    """Main function to test the RAG API"""
    parser = argparse.ArgumentParser(
        description="Test the RAG API with example queries.")
    parser.add_argument("--url", type=str,
                        default=DEFAULT_API_URL, help="RAG API URL.")
    parser.add_argument("--top_k", type=int, default=5,
                        help="Number of results to display.")
    parser.add_argument("--json", action="store_true",
                        help="Output results in JSON format.")
    parser.add_argument("--collection", type=str, choices=["guideline", "location", "both"],
                        help="Force search in a specific collection.")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode in the API request.")

    args = parser.parse_args()

    # Check if the API is running
    if not check_api_health(args.url):
        print("API is not available. Please start the API server first.")
        sys.exit(1)

    print("\n" + "="*70)
    print("RAG API Test - Enter your query to search both databases via the API")
    print("Examples:")
    print(" - 'Wo finde ich psychologische Hilfe in Siegen?'")
    print(" - 'Was sind die Symptome von Diabetes?'")
    print(" - 'Beratungsstellen für Depression in Frankfurt'")
    print("="*70)

    # Interactive query loop
    while True:
        query = input("\nEnter your query (or 'exit' to quit): ")
        if query.lower() == 'exit':
            break

        # Query the API
        results = query_rag_api(
            api_url=args.url,
            api_key=API_KEY,
            query=query,
            top_k=args.top_k,
            collection=args.collection,
            debug=args.debug,
            json_output=args.json
        )

        if not results:
            print("No results received from the API.")
            continue

        # Output as JSON if requested
        if args.json:
            print(json.dumps(results, indent=2, ensure_ascii=False))
            continue

        # Print query processing information
        query_info = results['query_info']
        print(f"\nQuery: '{query_info['original_query']}'")
        if query_info['detected_locations']:
            print(
                f"Detected locations: {', '.join(query_info['detected_locations'])}")
        if query_info['detected_topics']:
            print(
                f"Detected health topics: {', '.join(query_info['detected_topics'])}")
        print(f"Search type: {query_info['search_type']}")

        # Print results
        if not results['results']:
            print("\nNo results found.")
            continue

        print(f"\n--- Top {len(results['results'])} Results ---")

        for i, result in enumerate(results['results']):
            boost_info = ""
            if 'location_boost' in result['metadata']:
                boost_info = f" | Location Boost: {result['metadata']['location_boost']:.2f}"
            if 'topic_boost' in result['metadata']:
                boost_info += f" | Topic Boost: {result['metadata']['topic_boost']:.2f}"

            print(
                f"\n--- Result {i+1} (Score: {result['metadata']['rerank_score']:.4f}{boost_info}) ---")

            if result['type'] == 'location':
                print(format_location_result(result))
            else:
                print(format_guideline_result(result))

            print("-" * 40)


if __name__ == "__main__":
    main()
