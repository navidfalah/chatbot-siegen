# ==============================================================================
# SCRIPT: example_rag_integration.py
#
# PURPOSE:
#   - Demonstrates how to integrate the unified retriever with a RAG system
#   - Shows how to format retrieved content for an LLM
#   - Provides sample prompts for different types of information
#
# HOW TO USE:
#   1. Ensure Qdrant is running with both collections populated
#   2. Install required libraries: pip install requests
#   3. Run the script: python example_rag_integration.py
# ==============================================================================
import os
import requests
import argparse
from rag_retriever import retrieve_content

# Ollama API implementation from main.py


class OllamaAPI:
    """Interface for communicating with Ollama API"""

    def __init__(self, base_url: str = "https://ollama.wineme.wiwi.uni-siegen.de", model: str = "gemma3:4b"):
        self.base_url = base_url
        self.model = model
        self.api_endpoint = f"{base_url}/api/generate"

    def generate_response(self, prompt: str, context: str = "") -> str:
        """Generate response using Ollama API"""
        try:
            # Prepare the full prompt with context if available
            full_prompt = f"{context}\n\n{prompt}" if context else prompt

            payload = {
                "model": self.model,
                "prompt": full_prompt,
                "stream": False
            }

            response = requests.post(
                self.api_endpoint, json=payload, timeout=30)
            response.raise_for_status()

            result = response.json()
            return result.get('response', '')

        except Exception as e:
            raise Exception(f"API call failed: {str(e)}")


# Templates for different response types
SYSTEM_TEMPLATE = """You are a helpful assistant that provides information about health services 
and medical guidelines based on retrieved information. Only use the information provided 
in the context to answer the question. If you don't know the answer, say so."""

LOCATION_PROMPT = """Based on the following retrieved location information, provide a helpful 
response to the user's query: {query}

Retrieved Locations:
{context}

Provide a clear, concise answer that includes the relevant location details and contact information.
"""

GUIDELINE_PROMPT = """Based on the following retrieved medical guideline information, provide a helpful 
response to the user's query: {query}

Retrieved Guidelines:
{context}

Provide a clear, concise medical explanation based solely on these guidelines. Include appropriate 
citations to the source guidelines when providing medical information.
"""

MIXED_PROMPT = """Based on the following retrieved information about both locations and medical guidelines, 
provide a helpful response to the user's query: {query}

Retrieved Information:
{context}

Provide a clear, structured response that first explains the medical information and then 
suggests relevant locations or services if available. Include appropriate citations and contact details.
"""


def format_context_for_llm(results):
    """
    Format retrieved results into a context string for the LLM prompt
    """
    # Separate results by type
    locations = [r for r in results if r['type'] == 'location']
    guidelines = [r for r in results if r['type'] == 'guideline']

    context_parts = []

    # Format location information
    if locations:
        for i, loc in enumerate(locations):
            loc_text = f"LOCATION {i+1}: {loc['name']}\n"
            loc_text += f"Address: {loc['address']}\n"
            if loc['phone']:
                loc_text += f"Phone: {loc['phone']}\n"
            if loc['email']:
                loc_text += f"Email: {loc['email']}\n"
            if loc['website']:
                loc_text += f"Website: {loc['website']}\n"
            loc_text += f"Information: {loc['content']}\n"
            context_parts.append(loc_text)

    # Format guideline information
    if guidelines:
        for i, guide in enumerate(guidelines):
            guide_text = f"GUIDELINE {i+1}: {guide['title']}\n"
            guide_text += f"Source: {guide['url']}\n"
            guide_text += f"Content: {guide['content']}\n"
            context_parts.append(guide_text)

    return "\n\n".join(context_parts)


def rag_answer(query, results, model="gemma3:4b", base_url="https://ollama.wineme.wiwi.uni-siegen.de"):
    """
    Generate an answer using Ollama API with the retrieved context
    """
    # Select the appropriate prompt template based on result types
    location_results = [r for r in results if r['type'] == 'location']
    guideline_results = [r for r in results if r['type'] == 'guideline']

    if location_results and not guideline_results:
        prompt_template = LOCATION_PROMPT
    elif guideline_results and not location_results:
        prompt_template = GUIDELINE_PROMPT
    else:
        prompt_template = MIXED_PROMPT

    # Format the context for the LLM
    context = format_context_for_llm(results)

    # Create the prompt
    prompt = prompt_template.format(
        query=query,
        context=context
    )

    # Initialize the Ollama API
    ollama = OllamaAPI(base_url=base_url, model=model)

    # Generate the response with system template as context
    response = ollama.generate_response(prompt, context=SYSTEM_TEMPLATE)
    return response


def main():
    """Main function to demonstrate RAG integration"""
    parser = argparse.ArgumentParser(
        description="Demonstrate RAG integration with the unified retriever.")
    parser.add_argument("--host", type=str,
                        default="localhost", help="Qdrant instance host.")
    parser.add_argument("--port", type=int, default=6333,
                        help="Qdrant instance port.")
    parser.add_argument("--top_k", type=int, default=3,
                        help="Number of results to retrieve.")
    parser.add_argument("--model", type=str, default="gemma3:4b",
                        help="Ollama model to use.")
    parser.add_argument("--ollama-url", type=str,
                        default="https://ollama.wineme.wiwi.uni-siegen.de",
                        help="Ollama API base URL")

    args = parser.parse_args()

    # Interactive query loop
    while True:
        print("\n" + "="*70)
        print("RAG Integration Example with Ollama")
        print(f"Using Ollama API at: {args.ollama_url}")
        print(f"Using model: {args.model}")
        print("\nExamples:")
        print(" - 'Wo finde ich psychologische Hilfe in Siegen?'")
        print(" - 'Was sind die Symptome von Diabetes?'")
        print(" - 'Beratungsstellen für Depression in Frankfurt'")
        print("="*70)

        query = input("\nEnter your query (or 'exit' to quit): ")
        if query.lower() == 'exit':
            break

        # Retrieve content from both databases
        print("\nRetrieving relevant information...")
        retrieved_data = retrieve_content(
            query=query,
            top_k=args.top_k,
            qdrant_host=args.host,
            qdrant_port=args.port
        )

        # Check if we found any results
        if not retrieved_data['results']:
            print("No relevant information found.")
            continue

        # Generate RAG answer
        print("Generating answer based on retrieved information...")
        print(f"Using Ollama model: {args.model}")
        answer = rag_answer(
            query=query,
            results=retrieved_data['results'],
            model=args.model,
            base_url=args.ollama_url
        )

        # Display the answer
        print("\n" + "="*70)
        print("ANSWER:")
        print(answer)
        print("="*70)

        # Show query analysis
        print("\nQUERY ANALYSIS:")
        if retrieved_data['query_info']['detected_locations']:
            print(
                f"Detected locations: {', '.join(retrieved_data['query_info']['detected_locations'])}")
        if retrieved_data['query_info']['detected_topics']:
            print(
                f"Detected health topics: {', '.join(retrieved_data['query_info']['detected_topics'])}")
        print(f"Search type: {retrieved_data['query_info']['search_type']}")

        # Optionally show the source of information
        show_sources = input("\nShow sources? (y/n): ").lower() == 'y'
        if show_sources:
            print("\nSOURCES:")
            for i, result in enumerate(retrieved_data['results']):
                if result['type'] == 'guideline':
                    print(f"{i+1}. {result['title']} - {result['url']}")
                else:
                    print(f"{i+1}. {result['name']} - {result['address']}")


if __name__ == "__main__":
    main()
