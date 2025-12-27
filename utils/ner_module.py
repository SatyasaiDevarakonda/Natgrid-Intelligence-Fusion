"""
NATGRID Named Entity Recognition Module
Uses BERT-base-NER for extracting persons, organizations, and locations
Enhanced with better entity cleaning, deduplication, and accuracy improvements
"""

from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import re
from typing import List, Dict, Tuple, Set
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NERExtractor:
    """Named Entity Recognition for Intelligence Reports - Enhanced Version"""
    
    # Known entity lists for validation and improvement
    KNOWN_PERSONS = {
        'abdul karim', 'mohammad bashir', 'rashid ahmed', 'zaheer khan',
        'imran sheikh', 'tariq mahmood', 'junaid afridi', 'rajesh mehta',
        'priya malhotra', 'amit shah', 'sunita kapoor', 'vikram singh',
        'bilal ahmed', 'tariq siddiqui', 'javed mir', 'altaf sheikh',
        'ashfaq ali', 'zakir hussain', 'gurpreet kaur', 'mohammad iqbal'
    }
    
    KNOWN_ORGANIZATIONS = {
        'lashkar network', 'lashkar network kashmir unit', 'jaish module',
        'jaish module pulwama cell', 'hizbul mujahideen', 'hizbul mujahideen operative cell',
        'isi karachi operations', 'd-company', 'd-company dubai branch',
        'golden triangle connection', 'khalid arms syndicate', 'border weapon smuggling network',
        'infotech solutions', 'infotech solutions pvt ltd', 'global consulting group',
        'digital innovation labs', 'tech dynamics corporation', 'india business council',
        'mumbai chamber of commerce', 'coastal trading corporation', 'kashmir hawala network',
        'al-barkat money exchange', 'farmers unity forum', 'youth rights collective',
        'workers welfare association', 'student democratic front'
    }
    
    KNOWN_LOCATIONS = {
        'mumbai', 'delhi', 'bangalore', 'chennai', 'kolkata', 'hyderabad',
        'pune', 'ahmedabad', 'jaipur', 'lucknow', 'kashmir', 'wagah',
        'attari', 'mundra port', 'kandla port', 'gujarat', 'pakistan',
        'china', 'karachi', 'beijing', 'dubai', 'singapore', 'london',
        'india', 'northeast region', 'gujarat coast'
    }
    
    def __init__(self, model_name: str = "dslim/bert-base-NER"):
        """
        Initialize NER model
        
        Args:
            model_name: HuggingFace model name for NER
        """
        self.model_name = model_name
        self.ner_pipeline = None
        self._load_model()
    
    def _load_model(self):
        """Load the NER model"""
        try:
            logger.info(f"Loading NER model: {self.model_name}")
            self.ner_pipeline = pipeline(
                "ner",
                model=self.model_name,
                aggregation_strategy="simple"
            )
            logger.info("NER model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load NER model: {e}")
            raise
    
    def _clean_entity(self, entity: str) -> str:
        """Clean up entity text"""
        # Remove hashtags from subword tokenization
        entity = entity.replace('#', '')
        # Remove leading/trailing whitespace and punctuation
        entity = re.sub(r'^[\s\W]+|[\s\W]+$', '', entity)
        # Remove very short entities
        if len(entity) < 2:
            return ''
        return entity
    
    def _normalize_entity(self, entity: str) -> str:
        """Normalize entity for comparison"""
        return entity.lower().strip()
    
    def _merge_subword_entities(self, entities: List[Dict]) -> List[Dict]:
        """Merge entities that were split by the tokenizer"""
        if not entities:
            return []
        
        merged = []
        current = None
        
        for entity in entities:
            if current is None:
                current = entity.copy()
            elif (entity['entity_group'] == current['entity_group'] and 
                  entity['start'] == current['end'] + 1):
                # Merge adjacent entities of same type
                current['word'] += ' ' + entity['word']
                current['end'] = entity['end']
                current['score'] = (current['score'] + entity['score']) / 2
            else:
                merged.append(current)
                current = entity.copy()
        
        if current:
            merged.append(current)
        
        return merged
    
    def _validate_and_enhance(self, entity_text: str, entity_type: str) -> Tuple[str, bool]:
        """
        Validate entity against known lists and enhance if needed
        Returns (cleaned_entity, is_valid)
        """
        normalized = self._normalize_entity(entity_text)
        
        # Check against known lists
        if entity_type == 'PER':
            # Check if it's a known person
            for known in self.KNOWN_PERSONS:
                if known in normalized or normalized in known:
                    # Return the properly capitalized version
                    return known.title(), True
            # Check if it might be misclassified
            for known in self.KNOWN_ORGANIZATIONS:
                if normalized in known:
                    return None, False  # It's an org, not person
        
        elif entity_type == 'ORG':
            for known in self.KNOWN_ORGANIZATIONS:
                if known in normalized or normalized in known:
                    return known.title(), True
            # Check if it might be misclassified location
            for known in self.KNOWN_LOCATIONS:
                if normalized == known:
                    return None, False
        
        elif entity_type == 'LOC':
            for known in self.KNOWN_LOCATIONS:
                if known in normalized or normalized in known:
                    return known.title(), True
        
        # Return original if not in known lists but valid
        return entity_text, True
    
    def _extract_pattern_based(self, text: str) -> Dict[str, Set[str]]:
        """Extract entities using pattern matching for known entities"""
        result = {
            'persons': set(),
            'organizations': set(),
            'locations': set()
        }
        
        text_lower = text.lower()
        
        # Check for known persons
        for person in self.KNOWN_PERSONS:
            if person in text_lower:
                result['persons'].add(person.title())
        
        # Check for known organizations
        for org in self.KNOWN_ORGANIZATIONS:
            if org in text_lower:
                result['organizations'].add(org.title())
        
        # Check for known locations
        for loc in self.KNOWN_LOCATIONS:
            if loc in text_lower:
                result['locations'].add(loc.title())
        
        return result
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract named entities from text with enhanced accuracy
        
        Args:
            text: Input text to extract entities from
            
        Returns:
            Dictionary with 'persons', 'organizations', 'locations' keys
        """
        if not text or not isinstance(text, str):
            return {'persons': [], 'organizations': [], 'locations': []}
        
        result = {
            'persons': set(),
            'organizations': set(),
            'locations': set()
        }
        
        try:
            # First, use pattern-based extraction for known entities
            pattern_entities = self._extract_pattern_based(text)
            for key in result:
                result[key].update(pattern_entities[key])
            
            # Then, use NER model
            entities = self.ner_pipeline(text)
            
            # Merge subword entities
            entities = self._merge_subword_entities(entities)
            
            for entity in entities:
                entity_type = entity['entity_group']
                entity_text = self._clean_entity(entity['word'])
                score = entity['score']
                
                # Filter low confidence entities
                if score < 0.5 or not entity_text:
                    continue
                
                # Skip single characters or very short strings
                if len(entity_text) < 2:
                    continue
                
                # Skip common false positives
                if entity_text.lower() in ['the', 'a', 'an', 'of', 'in', 'at', 'to', 'for']:
                    continue
                
                # Validate and enhance
                validated, is_valid = self._validate_and_enhance(entity_text, entity_type)
                
                if not is_valid or not validated:
                    continue
                
                if entity_type == 'PER':
                    result['persons'].add(validated)
                elif entity_type == 'ORG':
                    result['organizations'].add(validated)
                elif entity_type == 'LOC':
                    result['locations'].add(validated)
            
            # Convert sets to sorted lists
            return {
                'persons': sorted(list(result['persons'])),
                'organizations': sorted(list(result['organizations'])),
                'locations': sorted(list(result['locations']))
            }
            
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            return {'persons': [], 'organizations': [], 'locations': []}
    
    def extract_entities_batch(self, texts: List[str]) -> List[Dict[str, List[str]]]:
        """
        Extract entities from multiple texts
        
        Args:
            texts: List of texts to process
            
        Returns:
            List of entity dictionaries
        """
        results = []
        for text in texts:
            entities = self.extract_entities(text)
            results.append(entities)
        return results
    
    def get_entity_counts(self, entities_list: List[Dict[str, List[str]]]) -> Dict[str, Dict[str, int]]:
        """
        Count entity occurrences across multiple documents
        
        Args:
            entities_list: List of entity dictionaries
            
        Returns:
            Nested dict with entity type -> entity -> count
        """
        counts = {
            'persons': {},
            'organizations': {},
            'locations': {}
        }
        
        for entities in entities_list:
            for entity_type in ['persons', 'organizations', 'locations']:
                for entity in entities.get(entity_type, []):
                    entity_normalized = entity.lower()
                    if entity_normalized in counts[entity_type]:
                        counts[entity_type][entity_normalized] += 1
                    else:
                        counts[entity_type][entity_normalized] = 1
        
        return counts
    
    def get_entity_summary(self, text: str) -> str:
        """
        Get a formatted summary of entities in text
        
        Args:
            text: Input text
            
        Returns:
            Formatted string with entity summary
        """
        entities = self.extract_entities(text)
        
        summary_parts = []
        
        if entities['persons']:
            summary_parts.append(f"Persons: {', '.join(entities['persons'])}")
        
        if entities['organizations']:
            summary_parts.append(f"Organizations: {', '.join(entities['organizations'])}")
        
        if entities['locations']:
            summary_parts.append(f"Locations: {', '.join(entities['locations'])}")
        
        return '; '.join(summary_parts) if summary_parts else "No entities identified"


def get_ner_extractor() -> NERExtractor:
    """Factory function to get NER extractor instance"""
    return NERExtractor()


if __name__ == "__main__":
    # Test the NER extractor
    extractor = NERExtractor()
    
    test_texts = [
        "Intelligence indicates Abdul Karim and Mohammad Bashir planning coordinated attack in Mumbai. Lashkar Network Kashmir Unit involvement suspected. ISI Karachi Operations may be providing support.",
        "Customs officials seized 50 kg contraband at Mundra Port. Suspect Rashid Ahmed arrested, investigation reveals links to D-Company Dubai Branch.",
        "Tech company InfoTech Solutions Pvt Ltd announced expansion plans in Bangalore. CEO Priya Malhotra addressed shareholders."
    ]
    
    print("=" * 60)
    print("NER EXTRACTION TEST")
    print("=" * 60)
    
    for i, text in enumerate(test_texts, 1):
        print(f"\nTest {i}:")
        print(f"Text: {text[:100]}...")
        entities = extractor.extract_entities(text)
        print(f"Persons: {entities['persons']}")
        print(f"Organizations: {entities['organizations']}")
        print(f"Locations: {entities['locations']}")
        print("-" * 40)
