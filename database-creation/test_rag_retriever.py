# ==============================================================================
# SCRIPT: test_rag_retriever.py
#
# PURPOSE:
#   - Demonstrates how to use the unified retrieval function
#   - Tests retrieving data from both databases
#   - Shows how to integrate the retriever with a RAG application
#
# HOW TO USE:
#   1. Ensure Qdrant is running with both collections populated
#   2. Run the script: python test_rag_retriever.py
# ==============================================================================
import argparse
import json
from rag_retriever import retrieve_content

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

def main():
    """Main function to test the RAG retriever"""
    parser = argparse.ArgumentParser(
        description="Test the unified RAG retriever across both databases.")
    parser.add_argument("--host", type=str,
                        default="localhost", help="Qdrant instance host.")
    parser.add_argument("--port", type=int, default=6333,
                        help="Qdrant instance port.")
    parser.add_argument("--top_k", type=int, default=5,
                        help="Number of final results to display.")
    parser.add_argument("--json", action="store_true",
                        help="Output results in JSON format.")
    parser.add_argument("--collection", type=str, choices=["guideline", "location", "both"],
                        help="Force search in a specific collection.")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging to troubleshoot retrieval issues.")

    args = parser.parse_args()

    # Interactive query loop
    while True:
        print("\n" + "="*70)
        print("RAG Retriever Test - Enter your query to search both databases")
        print("Examples:")
        print(" - 'Wo finde ich psychologische Hilfe in Siegen?'")
        print(" - 'Was sind die Symptome von Diabetes?'")
        print(" - 'Beratungsstellen für Depression in Frankfurt'")
        print("="*70)
        
        query = input("\nEnter your query (or 'exit' to quit): ")
        if query.lower() == 'exit':
            break

        # Call the retrieval function
        results = retrieve_content(
            query=query,
            top_k=args.top_k,
            qdrant_host=args.host,
            qdrant_port=args.port,
            force_collection=args.collection,
            debug=args.debug
        )
        
        # Output as JSON if requested
        if args.json:
            print(json.dumps(results, indent=2, ensure_ascii=False))
            continue
        
        # Print query processing information
        query_info = results['query_info']
        print(f"\nQuery: '{query_info['original_query']}'")
        if query_info['detected_locations']:
            print(f"Detected locations: {', '.join(query_info['detected_locations'])}")
        if query_info['detected_topics']:
            print(f"Detected health topics: {', '.join(query_info['detected_topics'])}")
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
                
            print(f"\n--- Result {i+1} (Score: {result['metadata']['rerank_score']:.4f}{boost_info}) ---")
            
            if result['type'] == 'location':
                print(format_location_result(result))
            else:
                print(format_guideline_result(result))
                
            print("-" * 40)

if __name__ == "__main__":
    main()
