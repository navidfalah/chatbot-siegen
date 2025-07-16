import time
import json
import requests
import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

# Install with: pip install langdetect
from langdetect import detect

@dataclass
class ChatMessage:
    timestamp: datetime
    question: str
    answer: str
    context: str
    rag_results: Optional[List[Dict]] = None

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

class Translator:
    """A minimal wrapper to translate prompts/text as needed. (Stub to expand with real API)"""
    @staticmethod
    def translate(text: str, dest_lang: str, src_lang: Optional[str] = None) -> str:
        # For demo: just a sketch. Could call DeepL, Google Translate, etc.
        # For now: return as-is if dest_lang is English or src_lang==dest_lang.
        if dest_lang == "en" or not text or (src_lang and src_lang == dest_lang):
            return text
        # Integrate real translation API here for production
        # e.g., use DeepL, Google Translate, or Hugging Face pipeline
        # WARNING: No translation for demo
        return text

class RAGDatabaseAPI:
    def __init__(self, api_key: str = None, base_url: str = "https://mimir.tail84e0ec.ts.net"):
        self.api_key = api_key or os.getenv("RAG_API_KEY")
        self.base_url = base_url
        self.api_endpoint = f"{base_url}/retrieve"

        if not self.api_key:
            raise ValueError("RAG API key is required. Set RAG_API_KEY environment variable.")

    def query_database(self, query: str, top_k: int = 5, collection: str = "both") -> List[RAGResult]:
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

class OllamaAPI:
    def __init__(self, base_url: str = "https://ollama.wineme.wiwi.uni-siegen.de", model: str = "gemma3:4b"):
        self.base_url = base_url
        self.model = model
        self.api_endpoint = f"{base_url}/api/generate"

    def generate_response(self, prompt: str, context: str = "", language: str = "en") -> str:
        """Generate response using Ollama API in requested language."""
        full_prompt = f"{context}\n\n{prompt}" if context else prompt
        # If prompt is not in the target language, translate it (expand this for real use).
        prompt_for_llm = full_prompt  # Translator.translate(full_prompt, language) # Uncomment/implement for prod
        payload = {
            "model": self.model,
            "prompt": prompt_for_llm,
            "stream": False
        }
        try:
            response = requests.post(self.api_endpoint, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            output_text = result.get("response", "")
            # Optionally: translate LLM output back to original language if needed
            return output_text
        except Exception as e:
            raise Exception(f"API call failed: {str(e)}")

class RAGEnhancedSocialHealthAI:
    def __init__(self, rag_api_key: str = None):
        self.ollama = OllamaAPI()
        self.rag_db = RAGDatabaseAPI(rag_api_key)
        self.system_context_en = (
            "You are a specialized AI assistant for social and health issues in the Siegen area. "
            "You have access to a database of local services, organizations, and resources. "
            "Always prioritize local, specific resources when available from the database."
        )
        self.system_context_de = (
            "Du bist ein spezialisierter KI-Assistent für soziale und gesundheitliche Themen im Raum Siegen. "
            "Du hast Zugriff auf eine Datenbank lokaler Dienste, Organisationen und Ressourcen. "
            "Bitte priorisiere immer lokale und spezifische Angebote aus der Datenbank."
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
            for i, msg in enumerate(chat_history):
                context_parts.append(f"\nNachricht {i+1}:")
                context_parts.append(f"Zeit: {msg.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
                context_parts.append(f"Frage: {msg.question}")
                context_parts.append(f"Antwort: {msg.answer}")
                context_parts.append(f"Kontext: {msg.context}")
                if msg.rag_results:
                    context_parts.append(f"Verwendete lokale Ressourcen: {len(msg.rag_results)} Ergebnisse")
                context_parts.append("-" * 40)
            context_parts.append("=== ENDE DES GESPRÄCHSVERLAUFS ===")
        else:
            context_parts.append("=== PREVIOUS CONVERSATION HISTORY ===")
            for i, msg in enumerate(chat_history):
                context_parts.append(f"\nMessage {i+1}:")
                context_parts.append(f"Time: {msg.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
                context_parts.append(f"Question: {msg.question}")
                context_parts.append(f"Answer: {msg.answer}")
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
        
        response = self.ollama.generate_response(prompt, language=language)
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
        
        response = self.ollama.generate_response(prompt, language=language)
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
        
        response = self.ollama.generate_response(prompt, language=language)
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
        
        return self.ollama.generate_response(full_context, language=language)

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
        
        return self.ollama.generate_response(prompt, language=language).strip()

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
        
        response = self.ollama.generate_response(prompt, language=language)
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
        self.clarification_attempts: int = 0
        self.current_topic_category: str = ''

    def start_point(self):
        print(f"🤖 AI: Connected to Ollama API at {self.ai.ollama.base_url}")
        print(f"🤖 AI: Using model: {self.ai.ollama.model}")
        print(f"🔍 RAG: Connected to database at {self.ai.rag_db.base_url}")
        print("🎯 AI: Specialized for Social and Health Issues with Local Resources")
        print("📝 Note: All questions and responses are stored for context continuity")
        print("=" * 70)
        
        while True:
            try:
                question = input("\nAsk about social/health issues or local services in Siegen (or 'exit' to quit): ").strip()
                if not question:
                    continue
                if question.lower() == 'exit':
                    print("Take care of yourself! Goodbye!")
                    break

                language = detect_language(question)
                # language = "de"  # For testing only; else auto-detect

                if not self.the_chat_exists():
                    self.create_a_new_chat()

                # Store the question and use full history for context
                understanding = self.ai.understand_problem(question, self.chat_history, language)
                is_in_topic, topic_category = self.ai.is_question_in_topic(question, self.chat_history, language)
                self.current_topic_category = topic_category

                if is_in_topic:
                    rag_results, rag_context = self.ai.retrieve_relevant_context(question, language)
                    if self.ai.do_we_have_data(understanding, self.chat_history, rag_results, language):
                        final_answer = self.ai.generate_rag_enhanced_response(
                            question, understanding, self.chat_history, rag_results, rag_context, language
                        )
                        print(f"\n🤖 AI Antwort:\n" if language == "de" else f"\n🤖 AI Response:\n")
                        print(final_answer)
                        
                        # Store the complete conversation turn
                        self.chat_history.append(ChatMessage(
                            timestamp=datetime.now(),
                            question=question,
                            answer=final_answer,
                            context=topic_category,
                            rag_results=[self._rag_result_to_dict(r) for r in rag_results]
                        ))
                        self.clarification_attempts = 0
                    else:
                        clarification_request = self.ai.ask_for_clarification(understanding, self.chat_history, language)
                        print(f"\n🤖 KI: {clarification_request}" if language == "de" else f"\n🤖 AI: {clarification_request}")
                        
                        # Store clarification request as well
                        self.chat_history.append(ChatMessage(
                            timestamp=datetime.now(),
                            question=question,
                            answer=clarification_request,
                            context=f"clarification_request_{topic_category}",
                            rag_results=[]
                        ))
                        
                        clarification = input("Bitte geben Sie mehr Details an: " if language == "de" else "Please provide more details: ")
                        if clarification:
                            enhanced_question = f"{question} - {clarification}"
                            self.process_enhanced_question(enhanced_question, language)
                else:
                    self.handle_off_topic_question(question, topic_category, language)
                    
            except KeyboardInterrupt:
                print("\n\nTake care! Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error occurred: {e}")
                print("Please try again or check your connection.")

    def _rag_result_to_dict(self, rag_result: RAGResult) -> Dict:
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

    def the_chat_exists(self) -> bool:
        return len(self.chat_history) > 0

    def create_a_new_chat(self):
        print("🆕 AI: Starting new conversation about social and health topics...")
        self.chat_history = []
        self.clarification_attempts = 0

    def process_enhanced_question(self, question: str, language: str):
        try:
            understanding = self.ai.understand_problem(question, self.chat_history, language)
            is_in_topic, topic_category = self.ai.is_question_in_topic(question, self.chat_history, language)
            
            if is_in_topic:
                rag_results, rag_context = self.ai.retrieve_relevant_context(question, language)
                if self.ai.do_we_have_data(understanding, self.chat_history, rag_results, language):
                    final_answer = self.ai.generate_rag_enhanced_response(
                        question, understanding, self.chat_history, rag_results, rag_context, language
                    )
                    print(f"\n🤖 KI Antwort:\n" if language == "de" else f"\n🤖 AI Response:\n")
                    print(final_answer)
                    
                    # Store the enhanced question and answer
                    self.chat_history.append(ChatMessage(
                        timestamp=datetime.now(),
                        question=question,
                        answer=final_answer,
                        context=topic_category,
                        rag_results=[self._rag_result_to_dict(r) for r in rag_results]
                    ))
                    self.clarification_attempts = 0
                else:
                    print("🤖 KI: Ich benötige noch mehr Details zu Ihrem Anliegen." if language == "de"
                          else "🤖 AI: I still need more specific information about your health or social concern.")
            else:
                print("🤖 KI: Bitte fokussieren Sie Ihre Frage auf Gesundheit oder Soziales." if language == "de"
                      else "🤖 AI: Please focus your question on health or social issues so I can help you better.")
        except Exception as e:
            print(f"❌ Error processing enhanced question: {e}")

    def handle_off_topic_question(self, question: str, topic_category: str, language: str):
        if self.clarification_attempts < 3:
            self.clarification_attempts += 1
            off_topic_response = (
                f"\n🤖 KI: Ich bin auf soziale und gesundheitliche Themen spezialisiert. Ihre Frage scheint sich um {topic_category} zu drehen."
                if language == "de"
                else f"\n🤖 AI: I specialize in social and health issues. Your question seems to be about {topic_category}."
            )
            print(off_topic_response)
            
            # Store off-topic question and response
            self.chat_history.append(ChatMessage(
                timestamp=datetime.now(),
                question=question,
                answer=off_topic_response,
                context=f"off_topic_{topic_category}",
                rag_results=[]
            ))
            
            ask_rephrase = "Könnten Sie Ihre Frage auf Gesundheit oder Soziales beziehen?" if language == "de" \
                else "Could you rephrase your question to focus on health or social concerns?"
            print(ask_rephrase)
            clarification = input("Umformulierte Frage: " if language == "de" else "Rephrased question: ")
            if clarification:
                self.process_enhanced_question(clarification, language)
        else:
            examples_intro = (
                "\n🤖 KI: Ich kann nur bei sozialen und gesundheitlichen Themen für Sie in Siegen helfen."
                "\nHier einige Beispiele für Fragen, mit denen ich helfen kann:" if language == "de"
                else "\n🤖 AI: I can only assist with social and health issues in the Siegen area."
                     "\nHere are examples of questions I can help with:"
            )
            print(examples_intro)
            
            # Store the examples response
            self.chat_history.append(ChatMessage(
                timestamp=datetime.now(),
                question=question,
                answer=examples_intro,
                context=f"examples_provided_{topic_category}",
                rag_results=[]
            ))
            
            examples = self.ai.display_error_and_examples("social_and_health", language)
            for i, example in enumerate(examples, 1):
                print(f"   {i}. {example}")
            self.clarification_attempts = 0

# Main execution
if __name__ == "__main__":
    print("🏥 Welcome to the RAG-Enhanced Social and Health Issues AI Assistant!")
    print("🎯 I specialize in providing information and local resources for:")
    print("   • Mental health and wellbeing services")
    print("   • Physical health and medical information") 
    print("   • Social issues and community concerns")
    print("   • Healthcare systems and local services")
    print("   • Public health topics")
    print("   • Health education and prevention")
    print("   • Local resources in Siegen and surrounding areas")
    print("\n🚨 Note: For medical emergencies, please contact emergency services immediately.")
    print("🔍 I have access to a local database of health and social services in your area.")
    print("📝 Context Continuity: All questions and responses are stored for better assistance.")
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

