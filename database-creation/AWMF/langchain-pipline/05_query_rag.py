# ==============================================================================
# SCRIPT 5: 05_query_rag.py
#
# PURPOSE:
#   - Connects to the Qdrant database and loads the sentence-transformer model.
#   - Sets up a LangChain RAG (Retrieval-Augmented Generation) chain.
#   - Creates a specific prompt to ensure answers are based only on the
#     retrieved context and that sources are cited.
#   - Provides an interactive loop to ask questions and get answers from your
#     health services chatbot.
#
# HOW TO USE:
#   1. Ensure all previous stages are complete and Qdrant is running.
#   2. Make sure your Ollama server is running with the LLM model (e.g., llama3).
#   3. Install required libraries:
#      pip install -U langchain-huggingface langchain-qdrant langchain
#   4. Run this script: python 05_query_rag.py
# ==============================================================================
import qdrant_client
from langchain_qdrant import Qdrant
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- Configuration ---
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
# Must match the collection name from the upload script
QDRANT_COLLECTION_NAME = "awmf-langchain-bread"
# Must match the embedding model from the upload script
EMBEDDING_MODEL_NAME = 'mixedbread-ai/deepset-mxbai-embed-de-large-v1'
# The LLM to use for generating answers
LLM_MODEL = "hf.co/bartowski/google_gemma-3-12b-it-qat-GGUF:Q4_K_M"


def format_docs(docs):
    """
    Formats the retrieved documents into a single string for the prompt.
    Includes the source citation for each document.
    """
    formatted_docs = []
    for i, doc in enumerate(docs):
        # Extract metadata for citation
        source_title = doc.metadata.get('guideline_title', 'N/A')
        source_url = doc.metadata.get('source_page_url', 'N/A')

        # Format the document content and its source
        formatted_doc = (
            f"--- Quelle {i+1} ---\n"
            f"Titel: {source_title}\n"
            f"URL: {source_url}\n\n"
            f"Inhalt:\n{doc.page_content}\n"
            f"-------------------\n"
        )
        formatted_docs.append(formatted_doc)
    return "\n".join(formatted_docs)


def main():
    """
    Sets up the RAG chain and starts an interactive query loop.
    """
    # 1. Initialize the embedding model
    print("Initializing embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': True}
    )

    # 2. Connect to the existing Qdrant vector store
    print("Connecting to Qdrant vector store...")
    client = qdrant_client.QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    vector_store = Qdrant(
        client=client,
        collection_name=QDRANT_COLLECTION_NAME,
        embeddings=embeddings
    )
    # Create a retriever to fetch relevant documents
    retriever = vector_store.as_retriever(
        search_kwargs={"k": 4})  # Retrieve top 4 chunks

    # 3. Define the prompt template
    # This is critical for controlling the LLM's behavior.
    template = """
    Sie sind ein Assistent für Gesundheits- und Sozialdienste. Ihre Aufgabe ist es, Fragen ausschließlich auf der Grundlage der untenstehenden Quellen zu beantworten.
    Seien Sie präzise und hilfsbereit. Fassen Sie die Informationen aus den Quellen zusammen, um eine umfassende Antwort zu geben.
    Erfinden Sie keine Informationen. Wenn die Antwort in den Quellen nicht enthalten ist, sagen Sie: "Ich kann keine Informationen zu diesem Thema in meinen Dokumenten finden."
    Zitieren Sie am Ende Ihrer Antwort IMMER die Titel und URLs der verwendeten Quellen unter einer Überschrift "Quellen:".

    Quellen:
    {context}

    Frage: {question}

    Antwort:
    """
    prompt = PromptTemplate.from_template(template)

    # 4. Initialize the LLM
    print(f"Initializing LLM: {LLM_MODEL}...")
    llm = Ollama(model=LLM_MODEL)

    # 5. Create the RAG chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("\n--- Chatbot ist bereit! ---")
    print("Stellen Sie Ihre Frage oder geben Sie 'exit' ein, um das Programm zu beenden.")

    # 6. Start the interactive query loop
    while True:
        query = input("\nIhre Frage: ")
        if query.lower() == 'exit':
            break
        if not query.strip():
            continue

        print("\nAntwort wird generiert...")
        answer = rag_chain.invoke(query)
        print("\n--- Antwort ---")
        print(answer)
        print("---------------")


if __name__ == "__main__":
    main()
