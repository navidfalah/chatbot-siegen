from flask import Flask, render_template, request, jsonify
import os
from datetime import datetime
from main import RAGEnhancedConversationSystem, detect_language, ChatMessage

# Initialize the Flask application
app = Flask(__name__)

# Load the RAG API key from environment variables
rag_api_key = os.getenv("RAG_API_KEY")
if not rag_api_key:
    print("Warning: RAG_API_KEY environment variable not set.")
    print("The system will still work but without local database access.")

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

        # Process the question using the existing chatbot logic with full context
        is_in_topic, topic_category = chat_system.ai.is_question_in_topic(user_question, chat_system.chat_history, language)

        if is_in_topic:
            understanding = chat_system.ai.understand_problem(user_question, chat_system.chat_history, language)
            rag_results, rag_context = chat_system.ai.retrieve_relevant_context(user_question, language)
            
            if chat_system.ai.do_we_have_data(understanding, chat_system.chat_history, rag_results, language):
                final_answer = chat_system.ai.generate_rag_enhanced_response(
                    user_question, understanding, chat_system.chat_history, rag_results, rag_context, language
                )
                
                # Store the complete conversation turn in history
                chat_system.chat_history.append(ChatMessage(
                    timestamp=datetime.now(),
                    question=user_question,
                    answer=final_answer,
                    context=topic_category,
                    rag_results=[chat_system._rag_result_to_dict(r) for r in rag_results]
                ))
                
            else:
                final_answer = chat_system.ai.ask_for_clarification(understanding, chat_system.chat_history, language)
                
                # Store clarification request
                chat_system.chat_history.append(ChatMessage(
                    timestamp=datetime.now(),
                    question=user_question,
                    answer=final_answer,
                    context=f"clarification_request_{topic_category}",
                    rag_results=[]
                ))
        else:
            # Handle off-topic questions with contextual response
            if language.startswith("de"):
                final_answer = f"Ich bin auf soziale und gesundheitliche Themen im Raum Siegen spezialisiert. Ihre Frage scheint sich um {topic_category} zu drehen. Könnten Sie Ihre Frage auf Gesundheit oder Soziales beziehen?"
            else:
                final_answer = f"I specialize in social and health issues in the Siegen area. Your question seems to be about {topic_category}. Could you please ask a question related to health or social concerns?"
            
            # Store off-topic question and response
            chat_system.chat_history.append(ChatMessage(
                timestamp=datetime.now(),
                question=user_question,
                answer=final_answer,
                context=f"off_topic_{topic_category}",
                rag_results=[]
            ))

        # Return the chatbot's answer as a JSON response
        return jsonify({
            'answer': final_answer,
            'language': language,
            'topic_category': topic_category,
            'is_in_topic': is_in_topic
        })

    except Exception as e:
        print(f"Error processing question: {e}")
        
        # Still store the failed interaction for context
        if chat_system:
            chat_system.chat_history.append(ChatMessage(
                timestamp=datetime.now(),
                question=user_question,
                answer=f"Error occurred: {str(e)}",
                context="error",
                rag_results=[]
            ))
        
        return jsonify({'error': f'An internal error occurred: {str(e)}'}), 500

@app.route('/clear_history', methods=['POST'])
def clear_history():
    """
    Clear the conversation history.
    This endpoint allows clearing the chat history for a fresh start.
    """
    if not chat_system:
        return jsonify({'error': 'Chat system not initialized.'}), 500
    
    try:
        chat_system.chat_history = []
        return jsonify({'message': 'Chat history cleared successfully.'})
    except Exception as e:
        return jsonify({'error': f'Failed to clear history: {str(e)}'}), 500

@app.route('/get_history', methods=['GET'])
def get_history():
    """
    Get the current conversation history.
    This endpoint returns the chat history for debugging or display purposes.
    """
    if not chat_system:
        return jsonify({'error': 'Chat system not initialized.'}), 500
    
    try:
        history = []
        for msg in chat_system.chat_history[-10:]:  # Return last 10 messages
            history.append({
                'timestamp': msg.timestamp.isoformat(),
                'question': msg.question,
                'answer': msg.answer[:200] + "..." if len(msg.answer) > 200 else msg.answer,
                'context': msg.context,
                'rag_results_count': len(msg.rag_results) if msg.rag_results else 0
            })
        
        return jsonify({
            'history': history,
            'total_messages': len(chat_system.chat_history)
        })
    except Exception as e:
        return jsonify({'error': f'Failed to get history: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint to verify the service is running.
    """
    try:
        gemini_status = "Connected" if chat_system and chat_system.ai.gemini else "Disconnected"
        rag_status = "Connected" if chat_system and chat_system.ai.rag_db and chat_system.ai.rag_db.api_key != "dummy_key" else "Limited/Disconnected"
        
        return jsonify({
            'status': 'healthy',
            'gemini_api': gemini_status,
            'rag_database': rag_status,
            'chat_history_size': len(chat_system.chat_history) if chat_system else 0,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

if __name__ == '__main__':
    # Run the Flask app locally on http://127.0.0.1:5000
    # The 'debug=True' argument enables auto-reloading when you save changes
    app.run(debug=True, host='0.0.0.0', port=5000)

