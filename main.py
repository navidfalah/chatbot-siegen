import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

@dataclass
class ChatMessage:
    """Structure for storing chat messages"""
    timestamp: datetime
    question: str
    answer: str
    context: str

class AI:
    """AI class with intelligent functions for processing conversations"""
    
    def __init__(self):
        self.knowledge_domains = {
            'programming': ['python', 'java', 'javascript', 'c++', 'algorithms', 'data structures'],
            'science': ['physics', 'chemistry', 'biology', 'mathematics'],
            'technology': ['ai', 'machine learning', 'web development', 'databases'],
            'general': ['history', 'geography', 'literature', 'arts']
        }
        self.context_window = []
        
    def analyze_relevance(self, question: str, chat_history: List[ChatMessage]) -> Tuple[bool, float, str]:
        """
        Intelligently analyze if a question is relevant/in-topic
        Returns: (is_relevant, confidence_score, detected_domain)
        """
        print("AI: Analyzing question relevance and context...")
        
        # Simulate intelligent analysis
        question_lower = question.lower()
        
        # Check against knowledge domains
        detected_domains = []
        relevance_score = 0.0
        
        for domain, keywords in self.knowledge_domains.items():
            domain_score = sum(1 for keyword in keywords if keyword in question_lower)
            if domain_score > 0:
                detected_domains.append((domain, domain_score))
                relevance_score += domain_score
        
        # Consider chat history context
        if chat_history:
            recent_context = ' '.join([msg.question + ' ' + msg.answer for msg in chat_history[-3:]])
            context_relevance = self._calculate_context_similarity(question, recent_context)
            relevance_score += context_relevance * 2  # Weight context heavily
        
        # Normalize score
        max_possible_score = 10.0
        confidence = min(relevance_score / max_possible_score, 1.0)
        
        # Determine if relevant
        is_relevant = confidence > 0.3  # 30% threshold
        main_domain = detected_domains[0][0] if detected_domains else 'unknown'
        
        return is_relevant, confidence, main_domain
    
    def _calculate_context_similarity(self, question: str, context: str) -> float:
        """Calculate similarity between question and context"""
        # Simplified similarity calculation
        common_words = set(question.lower().split()) & set(context.lower().split())
        return len(common_words) / max(len(question.split()), 1)
    
    def understand_problem(self, question: str, chat_history: List[ChatMessage]) -> Dict[str, any]:
        """
        Intelligently understand the problem from question and history
        Returns a structured understanding of the problem
        """
        print("AI: Deep analysis of the problem...")
        
        understanding = {
            'main_intent': self._extract_intent(question),
            'entities': self._extract_entities(question),
            'complexity': self._assess_complexity(question),
            'requires_context': len(chat_history) > 0,
            'suggested_approach': self._suggest_approach(question, chat_history)
        }
        
        return understanding
    
    def _extract_intent(self, question: str) -> str:
        """Extract the main intent from the question"""
        # Simulate intent extraction
        if any(word in question.lower() for word in ['how', 'explain', 'what']):
            return 'explanation'
        elif any(word in question.lower() for word in ['why', 'reason']):
            return 'reasoning'
        elif any(word in question.lower() for word in ['create', 'make', 'build']):
            return 'creation'
        elif any(word in question.lower() for word in ['fix', 'debug', 'error']):
            return 'troubleshooting'
        else:
            return 'general_query'
    
    def _extract_entities(self, question: str) -> List[str]:
        """Extract key entities from the question"""
        # Simplified entity extraction
        entities = []
        for domain, keywords in self.knowledge_domains.items():
            for keyword in keywords:
                if keyword in question.lower():
                    entities.append(keyword)
        return entities
    
    def _assess_complexity(self, question: str) -> str:
        """Assess the complexity of the question"""
        word_count = len(question.split())
        if word_count < 10:
            return 'simple'
        elif word_count < 25:
            return 'moderate'
        else:
            return 'complex'
    
    def _suggest_approach(self, question: str, history: List[ChatMessage]) -> str:
        """Suggest an approach for answering"""
        if history and len(history) > 2:
            return 'contextual_response'
        elif 'example' in question.lower():
            return 'provide_examples'
        elif 'compare' in question.lower():
            return 'comparative_analysis'
        else:
            return 'direct_answer'
    
    def check_data_availability(self, understanding: Dict, chat_history: List[ChatMessage]) -> bool:
        """
        Intelligently check if we have enough data to answer
        """
        print("AI: Checking data availability and requirements...")
        
        # Check if we have relevant context
        if understanding['requires_context'] and not chat_history:
            return False
        
        # Check if entities are recognized
        if not understanding['entities'] and understanding['complexity'] != 'simple':
            return False
        
        # Check if we can handle the intent
        supported_intents = ['explanation', 'reasoning', 'creation', 'troubleshooting', 'general_query']
        if understanding['main_intent'] not in supported_intents:
            return False
        
        return True
    
    def generate_answer(self, question: str, understanding: Dict, chat_history: List[ChatMessage], 
                       answer_type: str = 'main') -> str:
        """
        Generate an intelligent answer based on understanding
        """
        print(f"AI: Generating {answer_type} answer using neural processing...")
        
        # Simulate intelligent answer generation
        base_response = f"Based on my analysis of your question about {', '.join(understanding['entities'])}, "
        
        if answer_type == 'main':
            if understanding['main_intent'] == 'explanation':
                return base_response + f"here's a detailed explanation: [Intelligent response about {question}]"
            elif understanding['main_intent'] == 'creation':
                return base_response + f"here's how to create it: [Step-by-step guide for {question}]"
            elif understanding['main_intent'] == 'troubleshooting':
                return base_response + f"here's the solution: [Debugging steps for {question}]"
            else:
                return base_response + f"here's what you need to know: [Comprehensive answer to {question}]"
        else:
            # Side answer with additional context
            return f"Additional insights: Based on our previous discussion and current trends, [Extended context for {question}]"
    
    def enhance_with_context(self, question: str, chat_history: List[ChatMessage]) -> str:
        """
        Enhance question with relevant context from history
        """
        if not chat_history:
            return question
        
        # Extract relevant context
        recent_topics = [msg.context for msg in chat_history[-3:] if msg.context]
        context_summary = ' '.join(recent_topics)
        
        return f"{question} [Context: {context_summary}]"
    
    def generate_clarification_request(self, understanding: Dict) -> str:
        """
        Generate an intelligent clarification request
        """
        if not understanding['entities']:
            return "I notice your question lacks specific details. Could you provide more context about what specific aspect you're interested in?"
        elif understanding['complexity'] == 'complex':
            return "Your question covers multiple aspects. Could you help me understand which part is most important to you?"
        else:
            return f"I want to make sure I understand correctly. Are you asking specifically about {understanding['main_intent']} regarding {', '.join(understanding['entities'][:2])}?"
    
    def generate_examples(self, domain: str) -> List[str]:
        """
        Generate relevant examples based on domain
        """
        examples = {
            'programming': [
                "How do I implement a binary search in Python?",
                "Explain object-oriented programming concepts",
                "What's the difference between lists and tuples?"
            ],
            'science': [
                "Explain quantum entanglement in simple terms",
                "How does photosynthesis work?",
                "What causes gravitational waves?"
            ],
            'technology': [
                "How does machine learning differ from AI?",
                "Explain blockchain technology",
                "What are microservices in web development?"
            ],
            'general': [
                "What were the causes of World War I?",
                "Explain the water cycle",
                "How does the stock market work?"
            ]
        }
        
        return examples.get(domain, examples['general'])


class ConversationSystem:
    def __init__(self):
        self.chat_history: List[ChatMessage] = []
        self.ai = AI()
        self.clarification_count: int = 0
        self.current_domain: str = 'general'
        
    def start_point(self):
        """Entry point for the conversation system"""
        while True:
            # Ask a question
            question = input("\nPlease ask a question (or type 'exit' to quit): ")
            
            if question.lower() == 'exit':
                print("Goodbye!")
                break
                
            # Check if chat exists
            if self.chat_exists():
                # Understand the problem based on new + old data
                understanding = self.ai.understand_problem(question, self.chat_history)
            else:
                # Create a new chat
                self.create_new_chat()
                # Understand the problem based on the new data
                understanding = self.ai.understand_problem(question, [])
            
            # Process the question
            self.process_question(question, understanding)
    
    def chat_exists(self) -> bool:
        """Check if there's an existing chat history"""
        return len(self.chat_history) > 0
    
    def create_new_chat(self):
        """Initialize a new chat session"""
        print("AI: Initializing new conversation session with enhanced context tracking...")
        self.chat_history = []
        self.clarification_count = 0
    
    def process_question(self, question: str, understanding: Dict):
        """Main processing logic for the question"""
        # Use AI to check if the question is in topic
        is_relevant, confidence, domain = self.ai.analyze_relevance(question, self.chat_history)
        self.current_domain = domain
        
        print(f"AI: Relevance analysis complete - Confidence: {confidence:.2%}, Domain: {domain}")
        
        if is_relevant:
            # Check if we have data using AI
            if self.ai.check_data_availability(understanding, self.chat_history):
                # Add custom prompt and process
                enhanced_question = self.ai.enhance_with_context(question, self.chat_history)
                
                # Generate main answer using AI
                main_answer = self.ai.generate_answer(
                    enhanced_question, understanding, self.chat_history, 'main'
                )
                
                # Generate side answer if needed (for complex questions)
                final_answer = main_answer
                if understanding['complexity'] == 'complex':
                    side_answer = self.ai.generate_answer(
                        enhanced_question, understanding, self.chat_history, 'side'
                    )
                    final_answer = f"{main_answer}\n\n{side_answer}"
                
                # Display final answer
                print(f"\nAI Final Answer: {final_answer}")
                
                # Store in chat history
                self.chat_history.append(ChatMessage(
                    timestamp=datetime.now(),
                    question=question,
                    answer=final_answer,
                    context=domain
                ))
                
                self.clarification_count = 0  # Reset clarification count
                
            else:
                # Ask for clarification using AI
                clarification_request = self.ai.generate_clarification_request(understanding)
                print(f"\nAI: {clarification_request}")
                
                # Get clarification
                clarification = input("Your clarification: ")
                if clarification:
                    # Reprocess with clarification
                    enhanced_question = f"{question} - {clarification}"
                    new_understanding = self.ai.understand_problem(enhanced_question, self.chat_history)
                    self.process_question(enhanced_question, new_understanding)
                
        else:
            # Check if tried less than 3 times
            if self.clarification_count < 3:
                clarification_request = self.ai.generate_clarification_request(understanding)
                print(f"\nAI: I'm having difficulty understanding your question. {clarification_request}")
                self.clarification_count += 1
                
                # Get clarification
                clarification = input("Your clarification: ")
                if clarification:
                    # Reprocess with clarification
                    self.process_question(clarification, self.ai.understand_problem(clarification, self.chat_history))
            else:
                # Give contact information based on topic
                self.give_contact_information(domain)
                # Display error and give examples
                self.display_error_and_examples(domain)
                self.clarification_count = 0  # Reset for next question
    
    def give_contact_information(self, domain: str):
        """Provide contact information based on domain"""
        contacts = {
            'programming': "Programming Support Team: code-help@example.com",
            'science': "Science Department: science@example.com",
            'technology': "Tech Support: tech@example.com",
            'general': "General Inquiries: info@example.com"
        }
        
        print(f"\nAI: For specialized assistance with {domain} topics, please contact:")
        print(f"AI: {contacts.get(domain, contacts['general'])}")
    
    def display_error_and_examples(self, domain: str):
        """Display error message and provide examples using AI"""
        print("\nAI: I apologize, but I couldn't process your question after multiple attempts.")
        print(f"AI: Here are some examples of {domain} questions I can help with:")
        
        examples = self.ai.generate_examples(domain)
        for i, example in enumerate(examples, 1):
            print(f"   {i}. {example}")


# Main execution
if __name__ == "__main__":
    print("Welcome to the AI-Powered Conversation System!")
    print("This system uses intelligent analysis to understand and answer your questions.")
    
    system = ConversationSystem()
    system.start_point()
