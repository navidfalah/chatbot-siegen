import time
import json
import requests
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
            
            response = requests.post(self.api_endpoint, json=payload, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            return result.get('response', '')
            
        except Exception as e:
            raise Exception(f"API call failed: {str(e)}")

class SocialHealthAI:
    """AI class specialized for social and health issues"""
    
    def __init__(self):
        self.ollama = OllamaAPI()
        self.system_context = """
        You are a specialized AI assistant focused exclusively on social and health issues. 
        Your expertise covers:
        - Mental health and wellbeing
        - Physical health and medical information
        - Social issues and community problems
        - Healthcare systems and policies
        - Public health concerns
        - Social welfare and support systems
        - Health education and prevention
        - Social determinants of health
        - Community health initiatives
        - Healthcare accessibility and equity
        
        You should ONLY respond to questions related to these topics.
        """
        
    def understand_problem(self, question: str, chat_history: List[ChatMessage]) -> Dict[str, any]:
        """
        Use AI to understand the problem from question and history
        """
        print("AI: Analyzing problem in context of social and health issues...")
        
        # Create context from chat history
        history_context = ""
        if chat_history:
            recent_messages = chat_history[-2:]
            history_context = "Previous conversation about social/health topics:\n"
            for msg in recent_messages:
                history_context += f"Q: {msg.question}\nA: {msg.answer[:150]}...\n"
        
        understanding_prompt = f"""
        {self.system_context}
        
        Analyze this question in the context of social and health issues:
        Question: "{question}"
        
        {history_context}
        
        Please respond in this exact format only:
        INTENT: [health_inquiry/social_concern/prevention_advice/policy_question/support_seeking/general_wellness/medical_information]
        HEALTH_ENTITIES: [comma-separated health-related terms, or 'none' if no health entities]
        SOCIAL_ENTITIES: [comma-separated social-related terms, or 'none' if no social entities]
        COMPLEXITY: [simple/moderate/complex]
        REQUIRES_CONTEXT: [yes/no]
        APPROACH: [direct_answer/contextual_response/provide_resources/educational_response]
        """
        
        response = self.ollama.generate_response(understanding_prompt)
        return self._parse_understanding_response(response)
    
    def _parse_understanding_response(self, response: str) -> Dict[str, any]:
        """Parse the AI understanding response"""
        lines = response.strip().split('\n')
        
        understanding = {
            'main_intent': '',
            'health_entities': [],
            'social_entities': [],
            'complexity': '',
            'requires_context': False,
            'suggested_approach': ''
        }
        
        for line in lines:
            if "INTENT:" in line:
                understanding['main_intent'] = line.split(":", 1)[-1].strip()
            elif "HEALTH_ENTITIES:" in line:
                entities_str = line.split(":", 1)[-1].strip()
                if entities_str.lower() != 'none':
                    understanding['health_entities'] = [e.strip() for e in entities_str.split(',') if e.strip()]
            elif "SOCIAL_ENTITIES:" in line:
                entities_str = line.split(":", 1)[-1].strip()
                if entities_str.lower() != 'none':
                    understanding['social_entities'] = [e.strip() for e in entities_str.split(',') if e.strip()]
            elif "COMPLEXITY:" in line:
                understanding['complexity'] = line.split(":", 1)[-1].strip()
            elif "REQUIRES_CONTEXT:" in line:
                understanding['requires_context'] = line.split(":", 1)[-1].strip().lower() == 'yes'
            elif "APPROACH:" in line:
                understanding['suggested_approach'] = line.split(":", 1)[-1].strip()
        
        return understanding
    
    def is_question_in_topic(self, question: str, chat_history: List[ChatMessage]) -> Tuple[bool, str]:
        """
        Check if question is related to social and health issues
        Returns: (is_in_topic, topic_category)
        """
        print("AI: Checking if question relates to social and health issues...")
        
        # Create context from chat history
        history_context = ""
        if chat_history:
            recent_messages = chat_history[-3:]
            history_context = "Previous conversation context:\n"
            for msg in recent_messages:
                history_context += f"Q: {msg.question}\nA: {msg.answer[:100]}...\n"
        
        topic_check_prompt = f"""
        {self.system_context}
        
        Determine if this question is related to social and health issues:
        Question: "{question}"
        
        {history_context}
        
        Social and health topics include: mental health, physical health, medical conditions, healthcare systems, 
        social welfare, community health, public health, social issues, health education, health policy, 
        social determinants of health, healthcare accessibility, social support systems.
        
        Please respond in this exact format only:
        IN_TOPIC: [YES/NO]
        CATEGORY: [mental_health/physical_health/social_issues/healthcare_system/public_health/health_education/unrelated]
        REASONING: [brief explanation]
        """
        
        response = self.ollama.generate_response(topic_check_prompt)
        return self._parse_topic_response(response)
    
    def _parse_topic_response(self, response: str) -> Tuple[bool, str]:
        """Parse the topic relevance response"""
        lines = response.strip().split('\n')
        in_topic = False
        category = "unrelated"
        
        for line in lines:
            if "IN_TOPIC:" in line:
                in_topic = "YES" in line.upper()
            elif "CATEGORY:" in line:
                category = line.split(":", 1)[-1].strip()
        
        return in_topic, category
    
    def do_we_have_data(self, understanding: Dict, chat_history: List[ChatMessage]) -> bool:
        """
        Check if we have sufficient data to provide a helpful response
        """
        print("AI: Checking data availability for social/health response...")
        
        data_check_prompt = f"""
        {self.system_context}
        
        Can this social/health question be answered appropriately?
        
        Question Intent: {understanding['main_intent']}
        Health Entities: {', '.join(understanding['health_entities']) if understanding['health_entities'] else 'none'}
        Social Entities: {', '.join(understanding['social_entities']) if understanding['social_entities'] else 'none'}
        Complexity: {understanding['complexity']}
        Chat History Available: {'Yes' if chat_history else 'No'}
        
        Consider if this requires specialized medical advice that should be referred to professionals.
        
        Respond with only: CAN_ANSWER: [YES/NO]
        """
        
        response = self.ollama.generate_response(data_check_prompt)
        return "YES" in response.upper()
    
    def add_custom_prompt(self, question: str, chat_history: List[ChatMessage]) -> str:
        """
        Enhance question with relevant context from conversation history
        """
        if not chat_history:
            return question
        
        # Get recent context focused on social/health topics
        recent_context = ""
        for msg in chat_history[-2:]:
            recent_context += f"Previous discussion: {msg.question} -> {msg.answer[:100]}... "
        
        enhancement_prompt = f"""
        {self.system_context}
        
        Given this social/health conversation history: {recent_context}
        
        And this new question: {question}
        
        Provide an enhanced version that incorporates relevant context for a comprehensive social/health response.
        Respond with just the enhanced question, nothing else.
        """
        
        enhanced = self.ollama.generate_response(enhancement_prompt)
        return enhanced.strip()
    
    def generate_main_answer(self, question: str, understanding: Dict, chat_history: List[ChatMessage]) -> str:
        """
        Generate main answer focused on social and health issues
        """
        print("AI: Generating main answer for social/health inquiry...")
        
        # Create context from chat history
        context = ""
        if chat_history and understanding['requires_context']:
            context = "Conversation context on social/health topics:\n"
            for msg in chat_history[-3:]:
                context += f"Previous Q: {msg.question}\nPrevious A: {msg.answer[:200]}...\n\n"
        
        # Create specialized prompt based on intent
        if understanding['main_intent'] == 'health_inquiry':
            prompt = f"Provide comprehensive health information and guidance for: {question}"
        elif understanding['main_intent'] == 'social_concern':
            prompt = f"Address this social issue with practical insights and solutions: {question}"
        elif understanding['main_intent'] == 'prevention_advice':
            prompt = f"Provide prevention strategies and health promotion advice for: {question}"
        elif understanding['main_intent'] == 'policy_question':
            prompt = f"Explain relevant health/social policies and their implications for: {question}"
        elif understanding['main_intent'] == 'support_seeking':
            prompt = f"Provide supportive guidance and available resources for: {question}"
        elif understanding['main_intent'] == 'medical_information':
            prompt = f"Provide educational medical information while emphasizing professional consultation for: {question}"
        else:
            prompt = f"Provide a comprehensive response addressing the social and health aspects of: {question}"
        
        # Add important disclaimers for health-related responses
        system_prompt = f"""
        {self.system_context}
        
        Important: Always include appropriate disclaimers for medical advice and emphasize 
        consulting healthcare professionals for serious health concerns.
        
        {context}
        
        {prompt}
        """
        
        return self.ollama.generate_response(system_prompt)
    
    def generate_side_answer(self, question: str, understanding: Dict, chat_history: List[ChatMessage]) -> str:
        """
        Generate additional context and resources for complex social/health questions
        """
        print("AI: Generating additional context and resources...")
        
        side_prompt = f"""
        {self.system_context}
        
        Provide additional resources, related information, and broader context for this social/health question: {question}
        
        Include:
        - Related health/social concepts
        - Available support resources
        - Prevention strategies if applicable
        - Community or policy considerations
        
        Focus on practical, actionable information.
        """
        
        return self.ollama.generate_response(side_prompt)
    
    def ask_for_clarification(self, understanding: Dict) -> str:
        """
        Generate clarification request for unclear social/health questions
        """
        clarification_prompt = f"""
        {self.system_context}
        
        A user asked a social/health question but I need clarification.
        
        Question details:
        - Intent: {understanding['main_intent']}
        - Health entities: {', '.join(understanding['health_entities']) if understanding['health_entities'] else 'none'}
        - Social entities: {', '.join(understanding['social_entities']) if understanding['social_entities'] else 'none'}
        - Complexity: {understanding['complexity']}
        
        Generate a helpful clarification request that guides them to provide more specific 
        information about their health or social concern.
        Be empathetic and supportive. Respond with just the clarification request.
        """
        
        clarification = self.ollama.generate_response(clarification_prompt)
        return clarification.strip()
    
    def give_contact_information(self, topic_category: str) -> str:
        """
        Provide relevant contact information for social/health issues
        """
        contact_prompt = f"""
        {self.system_context}
        
        Generate appropriate contact information and resources for {topic_category} issues.
        Include relevant helplines, organizations, or professional services.
        Make contacts realistic but clearly fictional (use example.org domains).
        
        Respond with formatted contact information.
        """
        
        contact_info = self.ollama.generate_response(contact_prompt)
        return contact_info.strip()
    
    def display_error_and_examples(self, topic_category: str) -> List[str]:
        """
        Generate examples of social and health questions
        """
        examples_prompt = f"""
        {self.system_context}
        
        Generate exactly 3 example questions related to {topic_category} that I can help with.
        Make them practical, specific, and representative of real concerns people might have.
        
        Respond in this exact format:
        1. [example question]
        2. [example question]
        3. [example question]
        """
        
        response = self.ollama.generate_response(examples_prompt)
        
        # Parse examples
        examples = []
        lines = response.strip().split('\n')
        for line in lines:
            if line.strip() and (line.strip().startswith(('1.', '2.', '3.')) or line.strip().startswith('-')):
                example = line.split('.', 1)[-1].strip() if '.' in line else line.strip()
                if example:
                    examples.append(example)
        
        return examples[:3]


class SocialHealthConversationSystem:
    """Conversation system specialized for social and health issues"""
    
    def __init__(self):
        self.chat_history: List[ChatMessage] = []
        self.ai = SocialHealthAI()
        self.clarification_attempts: int = 0
        self.current_topic_category: str = ''
        
    def start_point(self):
        """Entry point following the flowchart structure"""
        print(f"AI: Connected to Ollama API at {self.ai.ollama.base_url}")
        print(f"AI: Using model: {self.ai.ollama.model}")
        print("AI: Specialized for Social and Health Issues")
        print("=" * 50)
        
        while True:
            try:
                # Ask a question
                question = input("\nPlease ask a question about social or health issues (or type 'exit' to quit): ")
                
                if question.lower() == 'exit':
                    print("Take care of yourself! Goodbye!")
                    break
                
                # Follow flowchart: Check if chat exists
                if self.the_chat_exists():
                    # Understand problem based on new + old data
                    understanding = self.ai.understand_problem(question, self.chat_history)
                else:
                    # Create new chat and understand problem based on new data
                    self.create_a_new_chat()
                    understanding = self.ai.understand_problem(question, [])
                
                # Check if question is in topic (social/health issues)
                is_in_topic, topic_category = self.ai.is_question_in_topic(question, self.chat_history)
                self.current_topic_category = topic_category
                
                if is_in_topic:
                    # Check if we have data
                    if self.ai.do_we_have_data(understanding, self.chat_history):
                        # Add custom prompt and generate response
                        enhanced_question = self.ai.add_custom_prompt(question, self.chat_history)
                        
                        # Generate main answer
                        main_answer = self.ai.generate_main_answer(enhanced_question, understanding, self.chat_history)
                        
                        # For complex questions, add side answer
                        if understanding['complexity'] == 'complex':
                            side_answer = self.ai.generate_side_answer(enhanced_question, understanding, self.chat_history)
                            final_answer = f"{main_answer}\n\n--- Additional Resources & Context ---\n{side_answer}"
                        else:
                            final_answer = main_answer
                        
                        # Display final answer
                        print(f"\nAI Response:\n{final_answer}")
                        
                        # Store in chat history
                        self.chat_history.append(ChatMessage(
                            timestamp=datetime.now(),
                            question=question,
                            answer=final_answer,
                            context=topic_category
                        ))
                        
                        self.clarification_attempts = 0
                        
                    else:
                        # Ask for clarification
                        clarification_request = self.ai.ask_for_clarification(understanding)
                        print(f"\nAI: {clarification_request}")
                        
                        clarification = input("Please provide more details: ")
                        if clarification:
                            enhanced_question = f"{question} - Additional details: {clarification}"
                            new_understanding = self.ai.understand_problem(enhanced_question, self.chat_history)
                            self.process_enhanced_question(enhanced_question, new_understanding)
                
                else:
                    # Question not related to social/health issues
                    if self.clarification_attempts < 3:
                        self.clarification_attempts += 1
                        print(f"\nAI: I specialize in social and health issues. Your question seems to be about {topic_category}.")
                        print("Could you rephrase your question to focus on health or social concerns?")
                        
                        clarification = input("Rephrased question: ")
                        if clarification:
                            new_understanding = self.ai.understand_problem(clarification, self.chat_history)
                            self.process_enhanced_question(clarification, new_understanding)
                    else:
                        # After 3 attempts, provide contact info and examples
                        print("\nAI: I apologize, but I can only assist with social and health issues.")
                        
                        # Give contact information
                        contact_info = self.ai.give_contact_information("general")
                        print(f"\nFor other topics, you might want to contact:\n{contact_info}")
                        
                        # Display examples of what I can help with
                        print(f"\nHere are examples of social and health questions I can help with:")
                        examples = self.ai.display_error_and_examples("social_and_health")
                        for i, example in enumerate(examples, 1):
                            print(f"   {i}. {example}")
                        
                        self.clarification_attempts = 0
                        
            except Exception as e:
                print(f"Error occurred: {e}")
                print("Please try again or check your connection.")
    
    def the_chat_exists(self) -> bool:
        """Check if chat history exists"""
        return len(self.chat_history) > 0
    
    def create_a_new_chat(self):
        """Initialize new chat session"""
        print("AI: Starting new conversation about social and health topics...")
        self.chat_history = []
        self.clarification_attempts = 0
    
    def process_enhanced_question(self, question: str, understanding: Dict):
        """Process question that has been enhanced with clarification"""
        try:
            is_in_topic, topic_category = self.ai.is_question_in_topic(question, self.chat_history)
            
            if is_in_topic and self.ai.do_we_have_data(understanding, self.chat_history):
                enhanced_question = self.ai.add_custom_prompt(question, self.chat_history)
                main_answer = self.ai.generate_main_answer(enhanced_question, understanding, self.chat_history)
                
                if understanding['complexity'] == 'complex':
                    side_answer = self.ai.generate_side_answer(enhanced_question, understanding, self.chat_history)
                    final_answer = f"{main_answer}\n\n--- Additional Resources & Context ---\n{side_answer}"
                else:
                    final_answer = main_answer
                
                print(f"\nAI Response:\n{final_answer}")
                
                self.chat_history.append(ChatMessage(
                    timestamp=datetime.now(),
                    question=question,
                    answer=final_answer,
                    context=topic_category
                ))
                
                self.clarification_attempts = 0
            else:
                print("AI: I still need more specific information about the health or social aspect of your concern.")
                
        except Exception as e:
            print(f"Error processing enhanced question: {e}")


# Main execution
if __name__ == "__main__":
    print("Welcome to the Social and Health Issues AI Assistant!")
    print("I specialize in providing information and support for:")
    print("• Mental health and wellbeing")
    print("• Physical health and medical information") 
    print("• Social issues and community concerns")
    print("• Healthcare systems and policies")
    print("• Public health topics")
    print("• Health education and prevention")
    print("\nNote: For serious medical emergencies, please contact emergency services immediately.")
    print("=" * 70)
    
    try:
        system = SocialHealthConversationSystem()
        system.start_point()
    except KeyboardInterrupt:
        print("\n\nTake care! Goodbye!")
    except Exception as e:
        print(f"\nSystem error: {e}")
        print("Please check your connection and try again.")
