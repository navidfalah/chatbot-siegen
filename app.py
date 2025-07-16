from flask import Flask, render_template, request, jsonify
# --- MODIFIED LINE: Importing from your main.py file ---
from main import RAGEnhancedConversationSystem, detect_language
import os

# Initialize the Flask application
app = Flask(__name__)

# Load the RAG API key from environment variables
rag_api_key = os.getenv("RAG_API_KEY")
if not rag_api_key:
    print("Warning: RAG_API_KEY environment variable not set.")
    # Add a default key for development if you have one, or handle the error
    # For this example, we'll proceed without it, though the RAG features will fail
    
# Instantiate your chatbot system
# This creates a single, persistent instance of the chatbot for all users
try:
    chat_system = RAGEnhancedConversationSystem(rag_api_key)
    print("RAG Enhanced Conversation System initialized successfully.")
except Exception as e:
    print(f"Failed to initialize RAGEnhancedConversationSystem: {e}")
    chat_system = None

@app.route('/')
def index():
    """
    Render the main chat page.
    This function serves the 'index.html' file, which contains the chat UI.
    """
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    """
    Handle incoming chat messages from the user.
    This endpoint receives a user's question, processes it with the chatbot,
    and returns the AI's response in JSON format.
    """
    if not chat_system:
        return jsonify({'error': 'Chat system not initialized. Check API keys and connections.'}), 500

    # Get the user's question from the POST request
    user_question = request.json.get('question', '')
    if not user_question:
        return jsonify({'error': 'No question provided'}), 400

    try:
        # Detect the language of the user's question
        language = detect_language(user_question)

        # Process the question using the existing chatbot logic
        # Note: We are simplifying the interaction loop for this web API.
        # For a more advanced implementation, you might manage conversation state (history)
        # per user session. For this example, we'll use the shared chat_history.
        
        is_in_topic, topic_category = chat_system.ai.is_question_in_topic(user_question, chat_system.chat_history, language)

        if is_in_topic:
            understanding = chat_system.ai.understand_problem(user_question, chat_system.chat_history, language)
            rag_results, rag_context = chat_system.ai.retrieve_relevant_context(user_question, language)
            
            if chat_system.ai.do_we_have_data(understanding, chat_system.chat_history, rag_results, language):
                final_answer = chat_system.ai.generate_rag_enhanced_response(
                    user_question, understanding, chat_system.chat_history, rag_results, rag_context, language
                )
            else:
                final_answer = chat_system.ai.ask_for_clarification(understanding, language)
        else:
            final_answer = "I specialize in social and health issues in the Siegen area. Could you please ask a question related to this topic?"

        # Return the chatbot's answer as a JSON response
        return jsonify({'answer': final_answer})

    except Exception as e:
        print(f"Error processing question: {e}")
        return jsonify({'error': 'An internal error occurred.'}), 500

if __name__ == '__main__':
    # Run the Flask app locally on http://127.0.0.1:5000
    # The 'debug=True' argument enables auto-reloading when you save changes
    app.run(debug=True)
