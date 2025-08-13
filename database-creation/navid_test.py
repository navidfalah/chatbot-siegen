import requests
import os
import json

def query_rag_api(api_key, query_text):
    """
    Sends a query to the RAG API and displays the results.

    Args:
        api_key (str): The API key for authentication.
        query_text (str): The question to ask the API.
    """
    # --- Configuration ---
    # UPDATED: The API URL now points to your live server.
    api_url = "https://mimir.tail84e0ec.ts.net/retrieve"
    
    # --- Request Details ---
    headers = {
        "X-API-Key": api_key,
        "Content-Type": "application/json"
    }
    
    payload = {
      "query": query_text,
      "top_k": 3,  # Ask for 3 results for this example
      "collection": "both",
      "debug": False
    }

    print(f"▶️  Sending query to {api_url}")
    print(f"   Query: '{query_text}'")

    try:
        # --- Send the POST request ---
        # Added a timeout for better network robustness
        response = requests.post(api_url, headers=headers, data=json.dumps(payload), timeout=20)

        # Raise an exception for bad status codes (4xx or 5xx)
        response.raise_for_status()

        # --- Process the successful response ---
        data = response.json()
        print("\n✅ Query Successful! Displaying results:\n")

        # Display the results
        if data.get("results"):
            for i, result in enumerate(data["results"], 1):
                print(f"--- Result {i} ---")
                print(f"  Name:    {result.get('name', 'N/A')}")
                print(f"  Type:    {result.get('type', 'N/A')}")
                print(f"  Address: {result.get('address', 'N/A')}")
                print(f"  Email:   {result.get('email', 'N/A')}")
                print(f"  Phone:   {result.get('phone', 'N/A')}")
                print(f"  Website: {result.get('website', 'N/A')}")
                # The content can be long, so we'll print a snippet
                content_snippet = result.get('content', '')[:150] + "..."
                print(f"  Content: \"{content_snippet}\"")
                print("-" * (len(f"--- Result {i} ---")))
        else:
            print("No results found for your query.")

    except requests.exceptions.HTTPError as http_err:
        print(f"❌ HTTP Error occurred: {http_err}")
        print(f"   Status Code: {response.status_code}")
        print(f"   Response: {response.text}")
    except requests.exceptions.Timeout:
        print("❌ Request timed out. The server is taking too long to respond.")
    except requests.exceptions.RequestException as req_err:
        print(f"❌ An error occurred during the request: {req_err}")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")


if __name__ == "__main__":
    # --- IMPORTANT ---
    # Get the API key. For security, it's best to use an environment variable.
    # You can set it in your terminal like this:
    # export RAG_API_KEY="your-api-key"
    
    MY_API_KEY = os.getenv("RAG_API_KEY")

    if not MY_API_KEY:
        print("Error: No API key found. Please set the RAG_API_KEY environment variable.")
    else:
        # The question we want to ask the API
        question = "Wo finde ich psychologische Hilfe in Siegen?"
        query_rag_api(api_key=MY_API_KEY, query_text=question)

