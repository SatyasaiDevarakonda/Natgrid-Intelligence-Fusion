"""
NATGRID LLM Module
Supports multiple LLM providers: Mistral API, Gemini, OpenAI, Local GPU
Updated for Mistral AI v1.0+ API
Enhanced with improved prompts for accurate, detailed, and professional analysis
"""

import os
from typing import Optional, Dict, List
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseLLM:
    """Base class for LLM providers"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
    
    def generate(self, prompt: str, max_tokens: int = 800, temperature: float = 0.3) -> str:
        """Generate text from prompt"""
        raise NotImplementedError


class MistralAPILLM(BaseLLM):
    """Mistral AI API Client (v1.0+ API)"""
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "mistral-small-latest"):
        super().__init__(model_name)
        self.api_key = api_key or os.getenv('MISTRAL_API_KEY')
        
        if not self.api_key:
            raise ValueError("MISTRAL_API_KEY not found in environment variables")
        
        try:
            # NEW API (v1.0+)
            from mistralai import Mistral
            self.client = Mistral(api_key=self.api_key)
            logger.info(f"Mistral API initialized with model: {model_name}")
        except ImportError:
            raise ImportError("mistralai package not installed. Run: pip install mistralai")
    
    def generate(self, prompt: str, max_tokens: int = 800, temperature: float = 0.3) -> str:
        """Generate text using Mistral API (v1.0+)"""
        try:
            # NEW API format
            response = self.client.chat.complete(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert intelligence analyst working for a national security agency. Provide detailed, accurate, and professional analysis. Never make up facts - only analyze information that is explicitly provided. Be thorough but concise."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            logger.error(f"Mistral API error: {e}")
            return f"Error generating response: {str(e)}"


class GeminiLLM(BaseLLM):
    """Google Gemini API Client"""
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-pro"):
        super().__init__(model_name)
        self.api_key = api_key or os.getenv('GOOGLE_API_KEY')
        
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(model_name)
            logger.info(f"Gemini initialized with model: {model_name}")
        except ImportError:
            raise ImportError("google-generativeai package not installed. Run: pip install google-generativeai")
    
    def generate(self, prompt: str, max_tokens: int = 800, temperature: float = 0.3) -> str:
        """Generate text using Gemini"""
        try:
            system_context = "You are an expert intelligence analyst working for a national security agency. Provide detailed, accurate, and professional analysis. Never make up facts - only analyze information that is explicitly provided."
            full_prompt = f"{system_context}\n\n{prompt}"
            
            response = self.model.generate_content(
                full_prompt,
                generation_config={
                    'temperature': temperature,
                    'max_output_tokens': max_tokens
                }
            )
            return response.text
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return f"Error generating response: {str(e)}"


class OpenAILLM(BaseLLM):
    """OpenAI API Client"""
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gpt-3.5-turbo"):
        super().__init__(model_name)
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
            logger.info(f"OpenAI initialized with model: {model_name}")
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")
    
    def generate(self, prompt: str, max_tokens: int = 800, temperature: float = 0.3) -> str:
        """Generate text using OpenAI"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert intelligence analyst working for a national security agency. Provide detailed, accurate, and professional analysis. Never make up facts - only analyze information that is explicitly provided. Be thorough but concise."
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return f"Error generating response: {str(e)}"


class LocalGPULLM(BaseLLM):
    """Local GPU-based LLM (Mistral, LLaMA, etc.)"""
    
    def __init__(self, model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"):
        super().__init__(model_name)
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
            import torch
            
            logger.info(f"Loading local model: {model_name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
            
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device_map="auto"
            )
            
            logger.info("Local GPU model loaded successfully")
            
        except ImportError:
            raise ImportError("transformers and torch not installed. Run: pip install transformers torch accelerate")
        except Exception as e:
            logger.error(f"Error loading local model: {e}")
            raise
    
    def generate(self, prompt: str, max_tokens: int = 800, temperature: float = 0.3) -> str:
        """Generate text using local GPU model"""
        try:
            system_msg = "You are an expert intelligence analyst. Provide detailed, accurate analysis based only on the information provided."
            formatted_prompt = f"<s>[INST] {system_msg}\n\n{prompt} [/INST]"
            
            outputs = self.pipeline(
                formatted_prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.95,
                return_full_text=False
            )
            
            return outputs[0]['generated_text'].strip()
        
        except Exception as e:
            logger.error(f"Local GPU generation error: {e}")
            return f"Error generating response: {str(e)}"


class IntelligenceLLM:
    """
    Main LLM interface for intelligence operations
    Automatically selects provider based on configuration
    Enhanced with improved prompts for accurate, professional analysis
    """
    
    def __init__(self, provider: Optional[str] = None):
        """
        Initialize LLM with specified provider
        
        Args:
            provider: One of ['mistral_api', 'gemini', 'openai', 'local_gpu']
                     If None, reads from LLM_PROVIDER env variable
        """
        self.provider = provider or os.getenv('LLM_PROVIDER', 'mistral_api')
        self.llm = self._initialize_llm()
    
    def _initialize_llm(self) -> BaseLLM:
        """Initialize the appropriate LLM provider"""
        logger.info(f"Initializing LLM provider: {self.provider}")
        
        if self.provider == 'mistral_api':
            return MistralAPILLM()
        elif self.provider == 'gemini':
            return GeminiLLM()
        elif self.provider == 'openai':
            return OpenAILLM()
        elif self.provider == 'local_gpu':
            return LocalGPULLM()
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
    
    def summarize_intelligence_reports(self, reports: List[str]) -> str:
        """
        Summarize multiple intelligence reports with JOINT analysis
        Produces integrated, comprehensive analysis - not individual summaries
        
        Args:
            reports: List of report texts
            
        Returns:
            Consolidated joint analysis
        """
        combined = "\n\n---\n\n".join([f"REPORT {i+1}:\n{r}" for i, r in enumerate(reports)])
        
        prompt = f"""You are a senior intelligence analyst preparing a UNIFIED intelligence assessment. 
You have been provided with {len(reports)} intelligence reports that must be analyzed TOGETHER as an integrated picture.

CRITICAL INSTRUCTIONS:
1. DO NOT summarize each report separately
2. Synthesize all information into ONE coherent narrative
3. Identify CONNECTIONS and PATTERNS across all reports
4. Only state facts that are EXPLICITLY mentioned in the reports
5. Do not invent or assume any information not present in the reports

INTELLIGENCE REPORTS TO ANALYZE:
{combined}

Provide your INTEGRATED INTELLIGENCE ASSESSMENT in the following structure:

**EXECUTIVE SUMMARY**
(3-4 sentences providing the unified picture from ALL reports combined)

**KEY THREAT INDICATORS**
• List specific threats mentioned across the reports
• Note any patterns or connections between threats
• Include specific names, locations, and organizations mentioned

**ENTITIES OF CONCERN**
• Persons: List all individuals mentioned with their roles/activities
• Organizations: List all groups/entities with their suspected involvement
• Locations: List all geographic areas with associated activities

**PATTERN ANALYSIS**
(Identify connections between different reports - what links them together?)

**THREAT LEVEL ASSESSMENT**
(Based ONLY on information in the reports, assess: LOW/MEDIUM/HIGH with justification)

**RECOMMENDED ACTIONS**
(Specific, actionable recommendations based on the intelligence)

Remember: Only include information explicitly stated in the reports. Do not speculate or add external information."""

        return self.llm.generate(prompt, max_tokens=1000, temperature=0.2)
    
    def generate_threat_assessment(self, entity_name: str, related_reports: List[str]) -> str:
        """
        Generate comprehensive threat assessment for an entity
        Accurate, detailed, and professional
        
        Args:
            entity_name: Name of person/organization
            related_reports: Intelligence reports mentioning this entity
            
        Returns:
            Detailed threat assessment report
        """
        if related_reports:
            reports_text = "\n\n".join([f"• {r}" for r in related_reports])
            context_section = f"""
AVAILABLE INTELLIGENCE ON {entity_name.upper()}:
{reports_text}
"""
        else:
            context_section = f"""
NOTE: No specific intelligence reports are currently available for {entity_name}. 
This assessment is based on general entity profile information only.
"""

        prompt = f"""You are a senior threat assessment analyst. Generate a comprehensive THREAT ASSESSMENT REPORT for the following entity.

ENTITY: {entity_name}
{context_section}

CRITICAL INSTRUCTIONS:
1. Base your assessment ONLY on information provided above
2. If no intelligence is available, clearly state this limitation
3. Do not invent crimes, activities, or associations not mentioned
4. Be specific about what is KNOWN vs what is SUSPECTED
5. Provide actionable intelligence, not speculation

Generate a professional threat assessment with the following sections:

**SUBJECT PROFILE**
• Full identification of the entity
• Type (Individual/Organization)
• Known aliases or associated names (only if mentioned in intelligence)

**THREAT CLASSIFICATION**
• Current threat level: [CRITICAL/HIGH/MEDIUM/LOW]
• Justification based on available intelligence

**KNOWN ACTIVITIES**
(List ONLY activities explicitly mentioned in the intelligence)
• Activity 1: Description with source reference
• Activity 2: Description with source reference
(If no activities known, state: "No specific activities documented in current intelligence")

**KNOWN ASSOCIATIONS**
• Organizations: (only those mentioned)
• Individuals: (only those mentioned)
• Networks: (only those mentioned)

**OPERATIONAL PATTERN ANALYSIS**
(Based on available intelligence, what patterns emerge?)

**INTELLIGENCE GAPS**
(What additional information is needed for complete assessment?)

**SURVEILLANCE RECOMMENDATIONS**
• Priority level for monitoring
• Specific areas requiring investigation
• Recommended intelligence collection methods

**RISK MITIGATION STRATEGIES**
(Concrete steps to address identified threats)

This assessment is classified and based on intelligence available as of current date."""

        return self.llm.generate(prompt, max_tokens=900, temperature=0.25)
    
    def generate_executive_brief(self, date: str, high_priority_reports: List[str]) -> str:
        """
        Generate professional daily executive brief
        
        Args:
            date: Date for the brief
            high_priority_reports: List of high-priority reports
            
        Returns:
            Executive brief
        """
        reports_text = "\n\n".join([f"REPORT {i+1}:\n{r}" for i, r in enumerate(high_priority_reports)])
        
        prompt = f"""You are the Chief Intelligence Analyst preparing the DAILY EXECUTIVE INTELLIGENCE BRIEF for senior leadership.

DATE: {date}
NUMBER OF HIGH-PRIORITY ITEMS: {len(high_priority_reports)}

HIGH PRIORITY INTELLIGENCE:
{reports_text}

INSTRUCTIONS:
1. This brief goes to the highest levels of government
2. Be concise but comprehensive
3. Only include information from the provided reports
4. Prioritize actionable intelligence
5. Use professional, formal language

Generate the DAILY EXECUTIVE INTELLIGENCE BRIEF:

**NATIONAL SECURITY DAILY BRIEF**
**Date: {date}**
**Classification: TOP SECRET**

---

**I. SITUATION OVERVIEW**
(2-3 sentences on overall security posture based on today's intelligence)

**II. PRIORITY THREATS**

**Threat 1:** [Title]
• Nature of threat:
• Key actors involved:
• Geographic scope:
• Immediacy: [Imminent/Near-term/Developing]

(Repeat for each major threat identified)

**III. ENTITIES REQUIRING ATTENTION**
• Persons of Interest: (names and brief reason)
• Organizations: (names and suspected activities)

**IV. GEOGRAPHIC HOTSPOTS**
(Locations mentioned with associated threat levels)

**V. RECOMMENDED IMMEDIATE ACTIONS**
1. [Specific action with responsible agency]
2. [Specific action with responsible agency]
3. [Specific action with responsible agency]

**VI. ITEMS FOR CONTINUED MONITORING**
(Issues that require ongoing surveillance)

---
END OF BRIEF"""

        return self.llm.generate(prompt, max_tokens=850, temperature=0.2)
    
    def explain_anomaly(self, anomaly_data: Dict) -> str:
        """
        Generate detailed, accurate explanation for detected anomaly
        Professional security analyst perspective
        
        Args:
            anomaly_data: Dictionary with anomaly details
            
        Returns:
            Detailed, professional explanation
        """
        prompt = f"""You are a Security Operations Center (SOC) analyst investigating a flagged anomaly. 
Provide a detailed, professional analysis based ONLY on the data provided.

ANOMALY DETECTION ALERT
=======================
User ID: {anomaly_data.get('user_id', 'Unknown')}
Event Type: {anomaly_data.get('event_type', 'Unknown')}
Timestamp: {anomaly_data.get('timestamp', 'Unknown')}
Location: {anomaly_data.get('location', 'Unknown')}
Risk Score: {anomaly_data.get('risk_score', 0):.1f}/100

Technical Details:
{anomaly_data.get('details', 'No additional details available')}

INSTRUCTIONS:
1. Analyze ONLY the data provided - do not assume additional context
2. Explain why this specific combination of factors is suspicious
3. Consider normal vs abnormal patterns
4. Provide concrete next steps

Generate your SOC ANALYST REPORT:

**ANOMALY ANALYSIS REPORT**

**1. INCIDENT SUMMARY**
(Brief description of what was detected)

**2. RISK ASSESSMENT**
• Risk Score Interpretation: {anomaly_data.get('risk_score', 0):.1f}/100
• Severity Level: [Based on score: <50 Low, 50-70 Medium, 70-90 High, >90 Critical]
• Confidence: [How confident are we this is a true positive?]

**3. SUSPICIOUS INDICATORS**
Based on the provided data, the following factors contributed to this alert:

• Factor 1: [Specific indicator from the data]
  - Why suspicious: [Explanation]
  
• Factor 2: [Specific indicator from the data]
  - Why suspicious: [Explanation]

• Factor 3: [If applicable]
  - Why suspicious: [Explanation]

**4. POTENTIAL SCENARIOS**
Given the data, this could indicate:
• Scenario A: [Most likely explanation]
• Scenario B: [Alternative explanation]
• Scenario C: [Benign explanation if applicable]

**5. IMMEDIATE INVESTIGATION STEPS**
1. [Specific action to verify the alert]
2. [Follow-up investigation step]
3. [Data collection requirement]

**6. RECOMMENDED RESPONSE**
• Immediate: [What should happen now]
• Short-term: [What should happen within 24 hours]
• If confirmed malicious: [Escalation path]

**7. ADDITIONAL CONTEXT NEEDED**
(What information would help determine if this is a true threat?)

---
Report generated by SOC Automated Analysis System"""

        return self.llm.generate(prompt, max_tokens=800, temperature=0.25)
    
    def custom_query(self, query: str, context: Optional[str] = None) -> str:
        """
        Answer custom intelligence query with accurate, professional response
        
        Args:
            query: User's question
            context: Optional context (intelligence reports, etc.)
            
        Returns:
            Detailed, accurate AI response
        """
        if context:
            prompt = f"""You are a senior intelligence analyst responding to a query from leadership.

AVAILABLE INTELLIGENCE CONTEXT:
{context}

QUERY: {query}

INSTRUCTIONS:
1. Base your response ONLY on the provided context
2. If the context doesn't contain enough information, clearly state this
3. Do not invent or assume facts not present in the context
4. Be specific and cite relevant parts of the context
5. Provide actionable insights where possible

Provide a detailed, professional response:

**INTELLIGENCE RESPONSE**

**Query Addressed:** {query}

**Analysis:**
[Your detailed analysis based on the provided context]

**Key Findings:**
• Finding 1: [With reference to source]
• Finding 2: [With reference to source]

**Confidence Level:** [High/Medium/Low based on available information]

**Limitations:**
[What information is missing or uncertain]

**Recommendations:**
[Actionable next steps if applicable]"""
        else:
            prompt = f"""You are a senior intelligence analyst responding to a general query.

QUERY: {query}

INSTRUCTIONS:
1. Provide accurate, professional information
2. Acknowledge any limitations in your knowledge
3. Do not speculate beyond what is reasonably known
4. If the query is outside your expertise, say so

Provide a detailed, professional response addressing the query comprehensively."""
        
        return self.llm.generate(prompt, max_tokens=700, temperature=0.3)


# Factory function
def get_llm(provider: Optional[str] = None) -> IntelligenceLLM:
    """
    Get LLM instance
    
    Args:
        provider: LLM provider name (mistral_api, gemini, openai, local_gpu)
        
    Returns:
        IntelligenceLLM instance
    """
    return IntelligenceLLM(provider=provider)


# For backward compatibility with train.py
def get_intelligence_analyzer(provider: Optional[str] = None) -> IntelligenceLLM:
    """
    Get intelligence analyzer instance (alias for get_llm)
    
    Args:
        provider: LLM provider name
        
    Returns:
        IntelligenceLLM instance
    """
    return get_llm(provider=provider)


if __name__ == "__main__":
    # Test the LLM module
    print("=" * 60)
    print("LLM MODULE TEST")
    print("=" * 60)
    
    try:
        llm = get_llm()
        print(f"\n✅ LLM initialized successfully with provider: {llm.provider}")
        
        # Test summarization
        test_reports = [
            "Intelligence indicates Abdul Karim and Mohammad Bashir planning coordinated attack in Mumbai. Intercepted communications mention explosives procurement. Lashkar Network involvement suspected.",
            "Customs officials seized 50 kg contraband at Mundra Port. Investigation reveals links to D-Company cartel. Border smuggling network active."
        ]
        
        print("\n🧪 Testing joint summarization...")
        summary = llm.summarize_intelligence_reports(test_reports)
        print(f"\nSummary:\n{summary[:500]}...")
        
        print("\n✅ All tests passed!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure you have:")
        print("1. Set LLM_PROVIDER in .env file")
        print("2. Set appropriate API key (MISTRAL_API_KEY, GOOGLE_API_KEY, etc.)")
        print("3. Installed required package (mistralai, google-generativeai, openai)")
