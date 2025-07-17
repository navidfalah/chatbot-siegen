import time
import json
import requests
import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import pickle

# Install with: pip install langdetect
from langdetect import detect

@dataclass
class ChatMessage:
    timestamp: datetime
    question: str
    answer: str
    context: str
    rag_results: Optional[List[Dict]] = None
    user_intent: Optional[str] = None
    conversation_id: Optional[str] = None

@dataclass
class RAGResult:
    name: str
    type: str
    address: str
    email: Optional[str]
    phone: Optional[str]
    website: Optional[str]
    content: str
    score: Optional[float] = None

def detect_language(text: str) -> str:
    """Detect the language code ('de', 'en', etc) of the input text."""
    try:
        lang = detect(text)
    except Exception:
        lang = "en"
    return lang

class ConversationMemory:
    """Enhanced conversation memory management"""
    def __init__(self, max_history: int = 20):
        self.max_history = max_history
        self.conversation_file = "conversation_history.pkl"
        
    def save_conversation(self, chat_history: List[ChatMessage]):
        """Save conversation history to file"""
        try:
            # Convert to serializable format
            serializable_history = []
            for msg in chat_history:
                msg_dict = asdict(msg)
                msg_dict['timestamp'] = msg.timestamp.isoformat()
                serializable_history.append(msg_dict)
            
            with open(self.conversation_file, 'wb') as f:
                pickle.dump(serializable_history, f)
        except Exception as e:
            print(f"⚠️ Could not save conversation: {e}")
    
    def load_conversation(self) -> List[ChatMessage]:
        """Load conversation history from file"""
        try:
            if os.path.exists(self.conversation_file):
                with open(self.conversation_file, 'rb') as f:
                    serializable_history = pickle.load(f)
                
                chat_history = []
                for msg_dict in serializable_history:
                    msg_dict['timestamp'] = datetime.fromisoformat(msg_dict['timestamp'])
                    chat_history.append(ChatMessage(**msg_dict))
                
                return chat_history[-self.max_history:]  # Keep only recent messages
            return []
        except Exception as e:
            print(f"⚠️ Could not load conversation: {e}")
            return []

class RAGDatabaseAPI:
    def __init__(self, api_key: str = None, base_url: str = "https://mimir.tail84e0ec.ts.net"):
        self.api_key = api_key or os.getenv("RAG_API_KEY")
        self.base_url = base_url
        self.api_endpoint = f"{base_url}/retrieve"

        if not self.api_key:
            print("⚠️ Warning: RAG API key not provided. RAG features will be limited.")
            self.api_key = "dummy_key"  # Allow initialization without key

    def query_database(self, query: str, top_k: int = 5, collection: str = "both") -> List[RAGResult]:
        if self.api_key == "dummy_key":
            return []  # Return empty results if no valid key
            
        headers = {
            "X-API-Key": self.api_key,
            "Content-Type": "application/json"
        }
        payload = {
            "query": query,
            "top_k": top_k,
            "collection": collection,
            "debug": False
        }
        try:
            response = requests.post(self.api_endpoint, headers=headers, 
                                     data=json.dumps(payload), timeout=20)
            response.raise_for_status()
            data = response.json()
            results = []
            if data.get("results"):
                for result in data["results"]:
                    rag_result = RAGResult(
                        name=result.get('name', 'N/A'),
                        type=result.get('type', 'N/A'),
                        address=result.get('address', 'N/A'),
                        email=result.get('email'),
                        phone=result.get('phone'),
                        website=result.get('website'),
                        content=result.get('content', ''),
                        score=result.get('score')
                    )
                    results.append(rag_result)
            return results
        except requests.exceptions.RequestException as e:
            print(f"❌ RAG Database Error: {e}")
            return []
        except Exception as e:
            print(f"❌ Unexpected RAG error: {e}")
            return []

class GeminiAPI:
    def __init__(self, api_key: str = "AIzaSyANNUH02lwIxpBDqGFPMldCzuqZvU2KQ-0", model: str = "gemini-2.0-flash"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models"
        self.api_endpoint = f"{self.base_url}/{model}:generateContent"

    def generate_response(self, prompt: str, context: str = "", language: str = "en") -> str:
        """Generate response using Gemini API in requested language."""
        full_prompt = f"{context}\n\n{prompt}" if context else prompt
        
        headers = {
            "Content-Type": "application/json",
            "X-goog-api-key": self.api_key
        }
        
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": full_prompt
                        }
                    ]
                }
            ]
        }
        
        try:
            response = requests.post(self.api_endpoint, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            
            # Extract the response text from Gemini's response format
            if "candidates" in result and len(result["candidates"]) > 0:
                if "content" in result["candidates"][0]:
                    if "parts" in result["candidates"][0]["content"]:
                        if len(result["candidates"][0]["content"]["parts"]) > 0:
                            return result["candidates"][0]["content"]["parts"][0].get("text", "")
            
            return "Sorry, I couldn't generate a response."
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Gemini API call failed: {str(e)}")
        except Exception as e:
            raise Exception(f"Error processing Gemini response: {str(e)}")

class RAGEnhancedSocialHealthAI:
    def __init__(self, rag_api_key: str = None):
        self.gemini = GeminiAPI()
        self.rag_db = RAGDatabaseAPI(rag_api_key)
        self.memory = ConversationMemory()
        self.system_context_en = (
            "You are a specialized AI assistant for social and health issues in the Siegen area. "
            "You have access to a database of local services, organizations, and resources. "
            "Always prioritize local, specific resources when available from the database. "
            "Use the conversation history to provide contextual and personalized responses. "
            "Remember previous discussions and build upon them naturally."
        )
        self.system_context_de = (
            "Du bist ein spezialisierter KI-Assistent für soziale und gesundheitliche Themen im Raum Siegen. "
            "Du hast Zugriff auf eine Datenbank lokaler Dienste, Organisationen und Ressourcen. "
            "Bitte priorisiere immer lokale und spezifische Angebote aus der Datenbank. "
            "Nutze den Gesprächsverlauf für kontextuelle und personalisierte Antworten. "
            "Erinnere dich an vorherige Diskussionen und baue natürlich darauf auf."
        )

    def get_system_context(self, language: str) -> str:
        if language.startswith("de"):
            return self.system_context_de
        return self.system_context_en

    def build_conversation_context(self, chat_history: List[ChatMessage], language: str) -> str:
        """Build comprehensive conversation context from chat history."""
        if not chat_history:
            return ""
        
        context_parts = []
        
        if language.startswith("de"):
            context_parts.append("=== VORHERIGER GESPRÄCHSVERLAUF ===")
            for i, msg in enumerate(chat_history[-10:], 1):  # Last 10 messages
                context_parts.append(f"\nNachricht {i}:")
                context_parts.append(f"Zeit: {msg.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
                context_parts.append(f"Frage: {msg.question}")
                context_parts.append(f"Antwort: {msg.answer[:200]}..." if len(msg.answer) > 200 else f"Antwort: {msg.answer}")
                context_parts.append(f"Kontext: {msg.context}")
                if msg.rag_results:
                    context_parts.append(f"Verwendete lokale Ressourcen: {len(msg.rag_results)} Ergebnisse")
                context_parts.append("-" * 40)
            context_parts.append("=== ENDE DES GESPRÄCHSVERLAUFS ===")
        else:
            context_parts.append("=== PREVIOUS CONVERSATION HISTORY ===")
            for i, msg in enumerate(chat_history[-10:], 1):  # Last 10 messages
                context_parts.append(f"\nMessage {i}:")
                context_parts.append(f"Time: {msg.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
                context_parts.append(f"Question: {msg.question}")
                context_parts.append(f"Answer: {msg.answer[:200]}..." if len(msg.answer) > 200 else f"Answer: {msg.answer}")
                context_parts.append(f"Context: {msg.context}")
                if msg.rag_results:
                    context_parts.append(f"Used local resources: {len(msg.rag_results)} results")
                context_parts.append("-" * 40)
            context_parts.append("=== END OF CONVERSATION HISTORY ===")
        
        return "\n".join(context_parts)

    def retrieve_relevant_context(self, question: str, language: str, top_k: int = 5) -> Tuple[List[RAGResult], str]:
        print("🔍 Searching RAG database for relevant information...")
        rag_results = self.rag_db.query_database(question, top_k=top_k)
        if not rag_results:
            return [], ""
        
        context_parts = []
        context_parts.append("=== RELEVANT LOCAL SERVICES AND INFORMATION ===" if language != "de" else "=== RELEVANTE LOKALE ANGEBOTE UND INFORMATIONEN ===")
        
        for i, result in enumerate(rag_results, 1):
            if language == "de":
                context_parts.append(f"\n{i}. {result.name}")
                context_parts.append(f"   Typ: {result.type}")
                context_parts.append(f"   Adresse: {result.address}")
                if result.phone:
                    context_parts.append(f"   Telefon: {result.phone}")
                if result.email:
                    context_parts.append(f"   E-Mail: {result.email}")
                if result.website:
                    context_parts.append(f"   Webseite: {result.website}")
                context_parts.append(f"   Beschreibung: {result.content}")
                context_parts.append("   " + "-" * 40)
            else:
                context_parts.append(f"\n{i}. {result.name}")
                context_parts.append(f"   Type: {result.type}")
                context_parts.append(f"   Address: {result.address}")
                if result.phone:
                    context_parts.append(f"   Phone: {result.phone}")
                if result.email:
                    context_parts.append(f"   Email: {result.email}")
                if result.website:
                    context_parts.append(f"   Website: {result.website}")
                context_parts.append(f"   Description: {result.content}")
                context_parts.append("   " + "-" * 40)
        
        context_parts.append("\n=== END OF LOCAL SERVICES ===" if language != "de" else "\n=== ENDE DER LOKALEN ANGEBOTE ===")
        formatted_context = "\n".join(context_parts)
        print(f"✅ Found {len(rag_results)} relevant results from database")
        return rag_results, formatted_context

    def understand_problem(self, question: str, chat_history: List[ChatMessage], language: str) -> Dict[str, any]:
        print("🧠 AI: Analyzing problem in context of social and health issues...")
        
        # Build comprehensive conversation context
        conversation_context = self.build_conversation_context(chat_history, language)
        
        if language.startswith("de"):
            prompt = (
                f"{self.get_system_context(language)}\n\n"
                f"{conversation_context}\n\n"
                f"Analysiere diese Frage im Kontext von sozialen und gesundheitlichen Themen:\n"
                f"Aktuelle Frage: \"{question}\"\n\n"
                f"Berücksichtige dabei den gesamten Gesprächsverlauf oben.\n\n"
                f"Bitte antworte exakt in diesem Format:\n"
                f"INTENT: [gesundheitsanfrage/soziales_anliegen/praeventionsratgeber/gesundheitspolicy/hilfesuche/allgemeines_wohlbefinden/medizinische_information/lokale_dienste]\n"
                f"HEALTH_ENTITIES: [Komma-getrennte gesundheitsbezogene Begriffe oder 'keine']\n"
                f"SOCIAL_ENTITIES: [Komma-getrennte soziale Begriffe oder 'keine']\n"
                f"LOCATION_ENTITIES: [Komma-getrennte Ortsangaben oder 'keine']\n"
                f"COMPLEXITY: [einfach/moderate/komplex]\n"
                f"REQUIRES_CONTEXT: [ja/nein]\n"
                f"REQUIRES_RAG: [ja/nein]\n"
                f"APPROACH: [direkte_antwort/kontextuelle_antwort/ressourcen_anbieten/bildungsantwort/lokale_dienste]"
            )
        else:
            prompt = (
                f"{self.get_system_context(language)}\n\n"
                f"{conversation_context}\n\n"
                f"Analyze this question in the context of social and health issues:\n"
                f"Current question: \"{question}\"\n\n"
                f"Consider the entire conversation history above.\n\n"
                f"Please respond in this exact format only:\n"
                "INTENT: [health_inquiry/social_concern/prevention_advice/policy_question/support_seeking/general_wellness/medical_information/local_services]\n"
                "HEALTH_ENTITIES: [comma-separated health-related terms, or 'none']\n"
                "SOCIAL_ENTITIES: [comma-separated social-related terms, or 'none']\n"
                "LOCATION_ENTITIES: [comma-separated location terms, or 'none']\n"
                "COMPLEXITY: [simple/moderate/complex]\n"
                "REQUIRES_CONTEXT: [yes/no]\n"
                "REQUIRES_RAG: [yes/no]\n"
                "APPROACH: [direct_answer/contextual_response/provide_resources/educational_response/local_services]"
            )
        
        response = self.gemini.generate_response(prompt, language=language)
        return self._parse_understanding_response(response, language)

    def _parse_understanding_response(self, response: str, language: str) -> Dict[str, any]:
        # Accept both English and German key variants
        lines = response.strip().split('\n')
        understanding = {
            'main_intent': '',
            'health_entities': [],
            'social_entities': [],
            'location_entities': [],
            'complexity': '',
            'requires_context': False,
            'requires_rag': False,
            'suggested_approach': ''
        }
        
        for line in lines:
            k = line.strip().lower()
            if "intent" in k:
                understanding['main_intent'] = line.split(":", 1)[-1].strip()
            elif "health_entities" in k or "gesundheits" in k:
                entities_str = line.split(":", 1)[-1].strip()
                if entities_str.lower() not in ('none', 'keine'):
                    understanding['health_entities'] = [e.strip() for e in entities_str.split(',') if e.strip()]
            elif "social_entities" in k or "sozial" in k:
                entities_str = line.split(":", 1)[-1].strip()
                if entities_str.lower() not in ('none', 'keine'):
                    understanding['social_entities'] = [e.strip() for e in entities_str.split(',') if e.strip()]
            elif "location_entities" in k or "ort" in k:
                entities_str = line.split(":", 1)[-1].strip()
                if entities_str.lower() not in ('none', 'keine'):
                    understanding['location_entities'] = [e.strip() for e in entities_str.split(',') if e.strip()]
            elif "complexity" in k or "komplexit" in k:
                understanding['complexity'] = line.split(":", 1)[-1].strip()
            elif "requires_context" in k or "benoetigt_kontext" in k:
                val = line.split(":", 1)[-1].strip().lower()
                understanding['requires_context'] = val in ("yes", "ja")
            elif "requires_rag" in k or "benoetigt_rag" in k:
                val = line.split(":", 1)[-1].strip().lower()
                understanding['requires_rag'] = val in ("yes", "ja")
            elif "approach" in k or "vorgehen" in k:
                understanding['suggested_approach'] = line.split(":", 1)[-1].strip()
        
        return understanding

    def is_question_in_topic(self, question: str, chat_history: List[ChatMessage], language: str) -> Tuple[bool, str]:
        print("🔍 AI: Checking if question relates to social and health issues...")
        
        # Build comprehensive conversation context
        conversation_context = self.build_conversation_context(chat_history, language)
        
        if language == "de":
            prompt = (
                f"{self.get_system_context(language)}\n\n"
                f"{conversation_context}\n\n"
                "Bestimme, ob diese Frage mit sozialen und gesundheitlichen Themen verbunden ist:\n"
                f"Aktuelle Frage: \"{question}\"\n\n"
                f"Berücksichtige dabei den gesamten Gesprächsverlauf oben.\n\n"
                "Soziale und gesundheitliche Themen umfassen: psychische Gesundheit, körperliche Gesundheit, medizinische Bedingungen, Gesundheitssysteme, "
                "soziale Unterstützung, Gemeinschaftsgesundheit, öffentliche Gesundheit, soziale Probleme, "
                "Gesundheitserziehung, Gesundheitspolitik, soziale Determinanten, Zugänglichkeit zur Gesundheitsversorgung, lokale Angebote.\n"
                "Bitte antworte nur im Format:\n"
                "IN_TOPIC: [JA/NEIN]\n"
                "CATEGORY: [psychische_gesundheit/koerperliche_gesundheit/soziale_fragen/gesundheitssystem/oeffentliche_gesundheit/gesundheitserziehung/lokale_dienste/nicht_zugeordnet]\n"
                "REASONING: [kurze Erklärung]"
            )
        else:
            prompt = (
                f"{self.get_system_context(language)}\n\n"
                f"{conversation_context}\n\n"
                "Determine if this question is related to social and health issues:\n"
                f"Current question: \"{question}\"\n\n"
                f"Consider the entire conversation history above.\n\n"
                "Social and health topics include: mental health, physical health, medical conditions, healthcare systems, "
                "social welfare, community health, public health, social issues, health education, health policy, "
                "social determinants of health, healthcare accessibility, social support systems, local services.\n"
                "Please respond in this exact format only:\n"
                "IN_TOPIC: [YES/NO]\n"
                "CATEGORY: [mental_health/physical_health/social_issues/healthcare_system/public_health/health_education/local_services/unrelated]\n"
                "REASONING: [brief explanation]"
            )
        
        response = self.gemini.generate_response(prompt, language=language)
        return self._parse_topic_response(response, language)

    def _parse_topic_response(self, response: str, language: str) -> Tuple[bool, str]:
        lines = response.strip().split('\n')
        in_topic = False
        category = "unrelated"
        
        for line in lines:
            k = line.lower()
            if "in_topic" in k or "in topic" in k or "in_topic" in k or "in_thema" in k:
                if "ja" in line.lower() or "yes" in line.lower():
                    in_topic = True
            elif "category" in k or "kategorie" in k:
                category = line.split(":", 1)[-1].strip()
        
        return in_topic, category

    def do_we_have_data(self, understanding: Dict, chat_history: List[ChatMessage], rag_results: List[RAGResult], language: str) -> bool:
        print("📊 AI: Checking data availability for social/health response...")
        
        if rag_results:
            return True
        
        # Build conversation context
        conversation_context = self.build_conversation_context(chat_history, language)
        
        if language == "de":
            prompt = (
                f"{self.get_system_context(language)}\n\n"
                f"{conversation_context}\n\n"
                "Kann diese soziale/gesundheitliche Frage angemessen ohne spezielle lokale Daten beantwortet werden?\n"
                f"Frage-Intent: {understanding['main_intent']}\n"
                f"Gesundheitsbegriffe: {', '.join(understanding['health_entities']) if understanding['health_entities'] else 'keine'}\n"
                f"Soziale Begriffe: {', '.join(understanding['social_entities']) if understanding['social_entities'] else 'keine'}\n"
                f"Ortsangaben: {', '.join(understanding['location_entities']) if understanding['location_entities'] else 'keine'}\n"
                f"Komplexität: {understanding['complexity']}\n"
                f"Gesprächsverlauf vorhanden: {'Ja' if chat_history else 'Nein'}\n"
                f"Lokale Datenbank-Ergebnisse: {'Keine' if not rag_results else f'{len(rag_results)} Ergebnisse'}\n"
                "Berücksichtige den gesamten Gesprächskontext oben.\n"
                "Bitte nur antworten: CAN_ANSWER: [JA/NEIN]"
            )
        else:
            prompt = (
                f"{self.get_system_context(language)}\n\n"
                f"{conversation_context}\n\n"
                "Can this social/health question be answered appropriately without specific local data?\n"
                f"Question Intent: {understanding['main_intent']}\n"
                f"Health Entities: {', '.join(understanding['health_entities']) if understanding['health_entities'] else 'none'}\n"
                f"Social Entities: {', '.join(understanding['social_entities']) if understanding['social_entities'] else 'none'}\n"
                f"Location Entities: {', '.join(understanding['location_entities']) if understanding['location_entities'] else 'none'}\n"
                f"Complexity: {understanding['complexity']}\n"
                f"Chat History Available: {'Yes' if chat_history else 'No'}\n"
                f"Local Database Results: {'None' if not rag_results else f'{len(rag_results)} results'}\n"
                "Consider the entire conversation context above.\n"
                "Respond with only: CAN_ANSWER: [YES/NO]"
            )
        
        response = self.gemini.generate_response(prompt, language=language)
        return ("ja" in response.lower()) or ("yes" in response.lower())

    def generate_rag_enhanced_response(self, question: str, understanding: Dict, chat_history: List[ChatMessage], rag_results: List[RAGResult], rag_context: str, language: str) -> str:
        print("🤖 AI: Generating RAG-enhanced response...")
        
        # Build comprehensive conversation context
        conversation_context = self.build_conversation_context(chat_history, language)
        
        # Intent-based prompt adaptation
        intent_prompts = {
            "de": {
                "health_inquiry": f"Beantworte die Gesundheitsfrage des Nutzers: {question}",
                "social_concern": f"Hilf dem Nutzer bei seinem sozialen Anliegen: {question}",
                "support_seeking": f"Biete Unterstützung und Ressourcen für: {question}",
                "local_services": f"Hilf dem Nutzer lokale Angebote für folgendes zu finden: {question}",
                "medical_information": f"Gib medizinische Informationen zu: {question}",
                "prevention_advice": f"Gib Präventionsratschläge für: {question}",
                "general_wellness": f"Unterstütze das allgemeine Wohlbefinden bei: {question}",
                "default": f"Hilf dem Nutzer bei seinem Anliegen: {question}"
            },
            "en": {
                "health_inquiry": f"Answer the user's health question: {question}",
                "social_concern": f"Help the user with their social concern: {question}",
                "support_seeking": f"Provide support and resources for: {question}",
                "local_services": f"Help the user find local services for: {question}",
                "medical_information": f"Provide medical information about: {question}",
                "prevention_advice": f"Give prevention advice for: {question}",
                "general_wellness": f"Support general wellness regarding: {question}",
                "default": f"Help the user with their concern: {question}"
            }
        }
        
        lang_key = "de" if language.startswith("de") else "en"
        main_prompt = intent_prompts[lang_key].get(understanding['main_intent'], intent_prompts[lang_key]["default"])
        
        if language == "de":
            full_context = (
                f"{self.get_system_context(language)}\n\n"
                f"{conversation_context}\n\n"
                f"{rag_context}\n\n"
                f"Wichtige Hinweise:\n"
                f"- Nutze die lokalen Informationen oben, wenn relevant.\n"
                f"- Berücksichtige den gesamten Gesprächsverlauf für bessere Antworten.\n"
                f"- Immer Kontaktdaten angeben (Adresse, Telefon, E-Mail, Webseite), wenn verfügbar.\n"
                f"- Lokale Angebote aus Siegen bevorzugen.\n"
                f"- Nur unterstützen und keine medizinischen Diagnosen stellen.\n"
                f"- Sei empathisch und unterstützend.\n"
                f"- Praktische und umsetzbare Empfehlungen geben.\n\n"
                f"Aktuelle Nutzerfrage: {question}\n\n"
                f"AUFGABE: {main_prompt}\n\n"
                f"Bitte antworte umfassend und binde die lokalen Ressourcen und den Gesprächskontext ein."
            )
        else:
            full_context = (
                f"{self.get_system_context(language)}\n\n"
                f"{conversation_context}\n\n"
                f"{rag_context}\n\n"
                f"Important Instructions:\n"
                f"- Use the local services information provided above when relevant\n"
                f"- Consider the entire conversation history for better responses\n"
                f"- Always include contact information (address, phone, email, website) when available\n"
                f"- Prioritize local Siegen resources from the database\n"
                f"- Include appropriate medical disclaimers when needed\n"
                f"- Be supportive and empathetic\n"
                f"- Provide practical, actionable advice\n\n"
                f"Current user question: {question}\n\n"
                f"TASK: {main_prompt}\n\n"
                f"Please provide a comprehensive response that incorporates the local services information and conversation context where relevant."
            )
        
        return self.gemini.generate_response(full_context, language=language)

    def ask_for_clarification(self, understanding: Dict, chat_history: List[ChatMessage], language: str) -> str:
        # Build conversation context
        conversation_context = self.build_conversation_context(chat_history, language)
        
        if language == "de":
            prompt = (
                f"{self.get_system_context(language)}\n\n"
                f"{conversation_context}\n\n"
                "Ein:e Benutzer:in hat eine soziale/gesundheitliche Frage gestellt, "
                "aber ich brauche eine Klärung, um die besten lokalen Ressourcen anbieten zu können.\n"
                f"Frage-Details:\n"
                f"- Intent: {understanding['main_intent']}\n"
                f"- Gesundheitsbegriffe: {', '.join(understanding['health_entities']) if understanding['health_entities'] else 'keine'}\n"
                f"- Soziale Begriffe: {', '.join(understanding['social_entities']) if understanding['social_entities'] else 'keine'}\n"
                f"- Ortsangaben: {', '.join(understanding['location_entities']) if understanding['location_entities'] else 'keine'}\n"
                f"- Komplexität: {understanding['complexity']}\n"
                "Berücksichtige den gesamten Gesprächsverlauf oben.\n"
                "Formuliere eine höfliche Nachfrage, damit Nutzer:innen präziser werden, damit ich passende lokale Angebote nennen kann. "
                "Antworte bitte nur mit dieser Nachfrage."
            )
        else:
            prompt = (
                f"{self.get_system_context(language)}\n\n"
                f"{conversation_context}\n\n"
                "A user asked a social/health question but I need clarification to provide the best local resources.\n"
                f"Question details:\n"
                f"- Intent: {understanding['main_intent']}\n"
                f"- Health entities: {', '.join(understanding['health_entities']) if understanding['health_entities'] else 'none'}\n"
                f"- Social entities: {', '.join(understanding['social_entities']) if understanding['social_entities'] else 'none'}\n"
                f"- Location entities: {', '.join(understanding['location_entities']) if understanding['location_entities'] else 'none'}\n"
                f"- Complexity: {understanding['complexity']}\n"
                "Consider the entire conversation history above.\n"
                "Generate a helpful clarification request that guides them to provide more specific "
                "information about their health or social concern, focusing on what would help "
                "me find the best local resources in Siegen. Be empathetic and supportive. Respond with just the clarification request."
            )
        
        return self.gemini.generate_response(prompt, language=language).strip()

    def display_error_and_examples(self, topic_category: str, language: str) -> List[str]:
        if language == "de":
            prompt = (
                f"{self.get_system_context(language)}\n\n"
                f"Erstelle genau 3 Beispiel-Fragen zum Thema {topic_category}, die ich beantworten kann, "
                "mit Fokus auf lokale Angebote und Ressourcen in Siegen. "
                "Formuliere sie praxisnah und so, wie echte Menschen fragen würden. "
                "Antwort nur im Format:\n"
                "1. [Beispielfrage]\n2. [Beispielfrage]\n3. [Beispielfrage]"
            )
        else:
            prompt = (
                f"{self.get_system_context(language)}\n\n"
                f"Generate exactly 3 example questions related to {topic_category} that I can help with,"
                " focusing on local services and resources in Siegen."
                " Make them practical, specific, and representative of real concerns."
                " Respond in this exact format:"
                " 1. [example question]\n2. [example question]\n3. [example question]"
            )
        
        response = self.gemini.generate_response(prompt, language=language)
        examples = []
        lines = response.strip().split('\n')
        for line in lines:
            if line.strip() and (line.strip().startswith(('1.', '2.', '3.')) or line.strip().startswith('-')):
                example = line.split('.', 1)[-1].strip() if '.' in line else line.strip()
                if example:
                    examples.append(example)
        return examples[:3]

class RAGEnhancedConversationSystem:
    def __init__(self, rag_api_key: str = None):
        self.chat_history: List[ChatMessage] = []
        self.ai = RAGEnhancedSocialHealthAI(rag_api_key)
        self.memory = ConversationMemory()
        self.clarification_attempts: int = 0
        self.current_topic_category: str = ''
        self.conversation_id: str = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Load previous conversation
        self.chat_history = self.memory.load_conversation()
        if self.chat_history:
            print(f"📚 Loaded {len(self.chat_history)} previous messages from conversation history")

    def _rag_result_to_dict(self, rag_result: RAGResult) -> Dict:
        """Convert RAG result to dictionary"""
        return {
            'name': rag_result.name,
            'type': rag_result.type,
            'address': rag_result.address,
            'email': rag_result.email,
            'phone': rag_result.phone,
            'website': rag_result.website,
            'content': rag_result.content,
            'score': rag_result.score
        }

    def start_point(self):
        print(f"🤖 AI: Connected to Gemini API")
        print(f"🤖 AI: Using model: {self.ai.gemini.model}")
        print(f"🔍 RAG: Connected to database at {self.ai.rag_db.base_url}")
        print("🎯 AI: Enhanced Social and Health Issues Assistant with Full Context Memory")
        print("📝 Note: All conversations are saved and loaded for continuous context")
        print("💾 Conversation persistence: Enabled")
        print("=" * 70)
        
        while True:
            try:
                question = input("\nAsk about social/health issues or local services in Siegen (or 'exit' to quit): ").strip()
                if not question:
                    continue
                if question.lower() == 'exit':
                    self.save_conversation()
                    print("Take care of yourself! Goodbye!")
                    break

                language = detect_language(question)
                
                # Process question with full context
                self.process_question_with_context(question, language)
                
                # Save conversation after each interaction
                self.save_conversation()
                    
            except KeyboardInterrupt:
                self.save_conversation()
                print("\n\nTake care! Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error occurred: {e}")
                print("Please try again or check your connection.")

    def process_question_with_context(self, question: str, language: str):
        """Process question with full conversation context"""
        try:
            # Enhanced understanding with context
            understanding = self.ai.understand_problem(question, self.chat_history, language)
            is_in_topic, topic_category = self.ai.is_question_in_topic(question, self.chat_history, language)
            self.current_topic_category = topic_category

            if is_in_topic:
                # Get relevant context with conversation history
                rag_results, rag_context = self.ai.retrieve_relevant_context(question, language)
                
                # Generate contextual response
                final_answer = self.ai.generate_rag_enhanced_response(
                    question, understanding, self.chat_history, rag_results, rag_context, language
                )
                
                print(f"\n🤖 AI Antwort:\n" if language.startswith("de") else f"\n🤖 AI Response:\n")
                print(final_answer)
                
                # Store the complete conversation turn with enhanced info
                self.chat_history.append(ChatMessage(
                    timestamp=datetime.now(),
                    question=question,
                    answer=final_answer,
                    context=topic_category,
                    rag_results=[self._rag_result_to_dict(r) for r in rag_results],
                    user_intent=understanding.get('main_intent'),
                    conversation_id=self.conversation_id
                ))
                
                self.clarification_attempts = 0
                
            else:
                self.handle_off_topic_question(question, topic_category, language)
                
        except Exception as e:
            print(f"❌ Error processing question: {e}")
            # Still save the question for context
            self.chat_history.append(ChatMessage(
                timestamp=datetime.now(),
                question=question,
                answer=f"Error occurred: {str(e)}",
                context="error",
                conversation_id=self.conversation_id
            ))

    def handle_off_topic_question(self, question: str, topic_category: str, language: str):
        """Handle off-topic questions with conversation context"""
        if self.clarification_attempts < 2:
            self.clarification_attempts += 1
            
            if language.startswith("de"):
                off_topic_response = (
                    f"\n🤖 KI: Ich bin auf soziale und gesundheitliche Themen im Raum Siegen spezialisiert. "
                    f"Ihre Frage scheint sich um {topic_category} zu drehen. "
                    f"Könnten Sie Ihre Frage in diesem Kontext umformulieren?"
                )
            else:
                off_topic_response = (
                    f"\n🤖 AI: I specialize in social and health issues in the Siegen area. "
                    f"Your question seems to be about {topic_category}. "
                    f"Could you rephrase your question in this context?"
                )
            
            print(off_topic_response)
            
            # Store off-topic question and response
            self.chat_history.append(ChatMessage(
                timestamp=datetime.now(),
                question=question,
                answer=off_topic_response,
                context=f"off_topic_{topic_category}",
                conversation_id=self.conversation_id
            ))
            
        else:
            # Show examples relevant to conversation history
            self.show_contextual_examples(language)
            self.clarification_attempts = 0

    def show_contextual_examples(self, language: str):
        """Show examples relevant to conversation history"""
        examples_intro = (
            "\n🤖 KI: Hier sind einige Beispiele für Fragen, bei denen ich helfen kann:"
            if language.startswith("de")
            else "\n🤖 AI: Here are examples of questions I can help with:"
        )
        print(examples_intro)
        
        # Generate contextual examples
        examples = self._generate_contextual_examples(language)
        for i, example in enumerate(examples, 1):
            print(f"   {i}. {example}")

    def _generate_contextual_examples(self, language: str) -> List[str]:
        """Generate examples based on conversation history"""
        examples = self.ai.display_error_and_examples("social_and_health", language)
        return examples

    def save_conversation(self):
        """Save current conversation"""
        if self.chat_history:
            self.memory.save_conversation(self.chat_history)
            print("💾 Conversation saved for future reference")

# Main execution
if __name__ == "__main__":
    print("🏥 Welcome to the Enhanced RAG Social and Health Issues AI Assistant!")
    print("🎯 I specialize in providing information and local resources for:")
    print("   • Mental health and wellbeing services")
    print("   • Physical health and medical information") 
    print("   • Social issues and community concerns")
    print("   • Healthcare systems and local services")
    print("   • Public health topics")
    print("   • Health education and prevention")
    print("   • Local resources in Siegen and surrounding areas")
    print("\n🧠 Enhanced Features:")
    print("   • Full conversation memory and context continuity")
    print("   • Persistent conversation storage")
    print("   • Contextual understanding of follow-up questions")
    print("   • Personalized responses based on conversation history")
    print("\n🚨 Note: For medical emergencies, please contact emergency services immediately.")
    print("🔍 I have access to a local database of health and social services.")
    print("💾 All conversations are saved and loaded for continuous assistance.")
    print("=" * 70)
    
    try:
        rag_api_key = os.getenv("RAG_API_KEY")
        if not rag_api_key:
            print("⚠️  Warning: RAG_API_KEY environment variable not set.")
            print("   Please set it with: export RAG_API_KEY='your-api-key'")
            print("   The system will still work but without local database access.")
        
        system = RAGEnhancedConversationSystem(rag_api_key)
        system.start_point()
        
    except KeyboardInterrupt:
        print("\n\nTake care! Goodbye!")
    except Exception as e:
        print(f"\n❌ System error: {e}")
        print("Please check your connection and API keys, then try again.")

