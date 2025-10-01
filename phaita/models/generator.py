"""
Symptom and complaint generators using Bayesian networks and language models.
Supports both LLM-based generation (Mistral-7B) and template-based fallback.
"""

import random
import torch
import torch.nn as nn
from typing import List, Optional, Dict
from .bayesian_network import BayesianSymptomNetwork
from ..data.icd_conditions import RespiratoryConditions

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not available for generator.")

try:
    import bitsandbytes
    BITSANDBYTES_AVAILABLE = True
except ImportError:
    BITSANDBYTES_AVAILABLE = False
    print("Warning: bitsandbytes not available. 4-bit quantization disabled.")


class SymptomGenerator:
    """
    Generates realistic symptom sets for medical conditions.
    """
    
    def __init__(self):
        """Initialize symptom generator with Bayesian network."""
        self.bayesian_network = BayesianSymptomNetwork()
    
    def generate_symptoms(
        self,
        condition_code: str,
        num_symptoms: Optional[int] = None
    ) -> List[str]:
        """
        Generate symptoms for a given condition.
        
        Args:
            condition_code: ICD-10 condition code
            num_symptoms: Number of symptoms to generate (None for random)
            
        Returns:
            List of symptoms
        """
        return self.bayesian_network.sample_symptoms(condition_code, num_symptoms)


class ComplaintGenerator(nn.Module):
    """
    Generates natural language patient complaints from symptoms.
    Uses Mistral-7B-Instruct with 4-bit quantization when available,
    falls back to template-based generation otherwise.
    """
    
    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
        use_pretrained: bool = True,
        use_4bit: bool = True,
        max_new_tokens: int = 150,
        temperature: float = 0.8,
        top_p: float = 0.9,
        device: Optional[str] = None
    ):
        """
        Initialize complaint generator.
        
        Args:
            model_name: Name of the LLM to use
            use_pretrained: Whether to load pretrained LLM
            use_4bit: Whether to use 4-bit quantization
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature for diversity
            top_p: Top-p sampling parameter
            device: Device to load model on
        """
        super().__init__()
        
        self.conditions = RespiratoryConditions.get_all_conditions()
        self.use_llm = use_pretrained and TRANSFORMERS_AVAILABLE
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Try to load LLM
        if self.use_llm:
            try:
                self._load_llm(model_name, use_4bit)
            except Exception as e:
                print(f"Warning: Could not load LLM: {e}")
                print("Falling back to template-based generation")
                self.use_llm = False
                self._init_template_generator()
        else:
            self._init_template_generator()
    
    def _load_llm(self, model_name: str, use_4bit: bool):
        """Load the language model."""
        print(f"Loading {model_name}...")
        
        if use_4bit and BITSANDBYTES_AVAILABLE and torch.cuda.is_available():
            # 4-bit quantization for memory efficiency
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            # Load without quantization
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
        
        # Set padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.eval()
        print(f"✓ Loaded {model_name} successfully")
    
    def _init_template_generator(self):
        """Initialize template-based fallback generator."""
        self.model = None
        self.tokenizer = None
        
        # Add a learnable embedding for template selection (for optimizer compatibility)
        self.template_embeddings = nn.Parameter(torch.randn(8, 64))  # 8 templates
        
        # Templates for generating complaints with proper grammar
        # Use {symptoms_experiencing} for "experiencing X" form
        # Use {main_symptom_gerund} for gerund form (e.g., "wheezing")
        # Use {main_symptom_phrase} for phrase form (e.g., "trouble breathing")
        self.conditions = RespiratoryConditions.get_all_conditions()
        
        # Templates for generating complaints with proper grammar
        # Use {symptoms_experiencing} for "experiencing X" form
        # Use {main_symptom_gerund} for gerund form (e.g., "wheezing")
        # Use {main_symptom_phrase} for phrase form (e.g., "trouble breathing")
        self.templates = [
            "I've been experiencing {symptoms} for the past {time}. It's really {severity}.",
            "Doctor, I have {symptoms} and I'm {feeling}. This started {time} ago.",
            "I'm having {main_symptom_phrase}. It's been {severity} since {time}.",
            "Help, I {main_symptom_action}. I also have {other_symptom_phrase}.",
            "I've been {main_symptom_gerund} and feeling {feeling}. It's been going on for {time}.",
            "My {symptoms_phrase_form} won't go away. Started {time} and getting {severity}.",
            "Really worried about {main_symptom_phrase}. Also experiencing {other_symptom_phrase}.",
            "Can't seem to shake this {main_symptom_noun}. {other_symptom_phrase} too.",
        ]
        
        self.severity_terms = ["mild", "moderate", "severe", "terrible", "awful", "worse"]
        self.time_terms = ["a few hours", "yesterday", "two days", "this morning", "last night", "a week"]
        self.feeling_terms = ["worried", "scared", "exhausted", "panicked", "terrible", "awful"]
        
        # Grammar rules for symptom transformation
        self.symptom_grammar_rules = {
            # Map symptoms to their proper grammatical forms
            "wheezing": {
                "gerund": "wheezing",
                "noun": "wheezing",
                "phrase": "my wheezing",
                "action": "can't stop wheezing"
            },
            "shortness_of_breath": {
                "gerund": "having shortness of breath",
                "noun": "breathlessness", 
                "phrase": "shortness of breath",
                "action": "can't catch my breath"
            },
            "difficulty_breathing": {
                "gerund": "having difficulty breathing",
                "noun": "breathing difficulty",
                "phrase": "difficulty breathing",
                "action": "can't breathe properly"
            },
            "coughing": {
                "gerund": "coughing",
                "noun": "cough",
                "phrase": "my cough",
                "action": "can't stop coughing"
            },
            "cough": {
                "gerund": "coughing",
                "noun": "cough",
                "phrase": "my cough",
                "action": "can't stop coughing"
            },
            "chest_pain": {
                "gerund": "experiencing chest pain",
                "noun": "chest pain",
                "phrase": "chest pain",
                "action": "have sharp chest pain"
            },
            "chest_tightness": {
                "gerund": "experiencing chest tightness",
                "noun": "chest tightness",
                "phrase": "tight chest",
                "action": "feel chest tightness"
            },
            "tight_chest": {
                "gerund": "experiencing chest tightness",
                "noun": "chest tightness",
                "phrase": "tight chest",
                "action": "feel tightness in my chest"
            },
            "fever": {
                "gerund": "running a fever",
                "noun": "fever",
                "phrase": "my fever",
                "action": "have a fever"
            },
            "breathless": {
                "gerund": "feeling breathless",
                "noun": "breathlessness",
                "phrase": "breathlessness",
                "action": "feel breathless"
            },
            "breathlessness": {
                "gerund": "feeling breathless",
                "noun": "breathlessness",
                "phrase": "breathlessness",
                "action": "feel breathless"
            },
            "gasping_for_air": {
                "gerund": "gasping for air",
                "noun": "breathlessness",
                "phrase": "gasping for air",
                "action": "can't stop gasping for air"
            },
            "gasping for air": {
                "gerund": "gasping for air",
                "noun": "breathlessness",
                "phrase": "gasping for air",
                "action": "can't stop gasping for air"
            },
            "can't breathe": {
                "gerund": "having trouble breathing",
                "noun": "breathing difficulty",
                "phrase": "trouble breathing",
                "action": "can't breathe"
            },
            "can't catch my breath": {
                "gerund": "having trouble catching my breath",
                "noun": "breathlessness",
                "phrase": "trouble catching my breath",
                "action": "can't catch my breath"
            },
            "wheezy": {
                "gerund": "feeling wheezy",
                "noun": "wheezing",
                "phrase": "wheezing",
                "action": "feel wheezy"
            }
        }
    
    def _create_prompt(
        self,
        symptoms: List[str],
        condition_code: str
    ) -> str:
        """
        Create a prompt for the LLM to generate a patient complaint.
        
        Args:
            symptoms: List of medical symptoms
            condition_code: ICD-10 condition code
            
        Returns:
            Formatted prompt string
        """
        condition_data = self.conditions.get(condition_code, {})
        condition_name = condition_data.get("name", "respiratory condition")
        
        # Format symptoms as natural language
        symptom_list = ", ".join(s.replace("_", " ") for s in symptoms[:5])
        
        prompt = f"""[INST] You are helping generate realistic patient complaints for medical triage training.

Patient has {condition_name} with symptoms: {symptom_list}

Generate a natural, realistic patient complaint as if the patient is describing their symptoms to a doctor. Keep it:
- Brief (1-3 sentences)
- In first person
- Using everyday language (not medical jargon)
- Expressing concern or discomfort
- Natural and conversational

Patient complaint: [/INST]"""
        
        return prompt
    
    def _generate_with_llm(
        self,
        symptoms: List[str],
        condition_code: str
    ) -> str:
        """
        Generate complaint using LLM.
        
        Args:
            symptoms: List of medical symptoms
            condition_code: ICD-10 condition code
            
        Returns:
            Generated complaint string
        """
        if not self.use_llm or self.model is None:
            return self._generate_with_template(symptoms, condition_code)
        
        try:
            prompt = self._create_prompt(symptoms, condition_code)
            
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode and extract the response
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract the complaint (after the prompt)
            if "[/INST]" in generated_text:
                complaint = generated_text.split("[/INST]")[-1].strip()
            else:
                complaint = generated_text[len(prompt):].strip()
            
            # Clean up the complaint
            complaint = complaint.split('\n')[0]  # Take first line
            complaint = complaint[:500]  # Limit length
            
            return complaint if complaint else self._generate_with_template(symptoms, condition_code)
        
        except Exception as e:
            print(f"Warning: LLM generation failed: {e}")
            return self._generate_with_template(symptoms, condition_code)
    
    def _generate_with_template(
        self,
        symptoms: List[str],
        condition_code: str,
        use_lay_terms: bool = True
    ) -> str:
        """
        Generate complaint using templates (fallback method).
        
        Args:
            symptoms: List of medical symptoms
            condition_code: ICD-10 condition code
            use_lay_terms: Whether to use lay language
            
        Returns:
            Generated complaint string
        """
        if not symptoms:
            return "I'm not feeling well."
        
        # Get lay terms if available
        if use_lay_terms and condition_code in self.conditions:
            lay_terms = self.conditions[condition_code]["lay_terms"]
            if lay_terms:
                # Use lay terms for some symptoms
                display_symptoms = []
                for symptom in symptoms[:3]:  # Use first 3 symptoms
                    if random.random() < 0.5 and lay_terms:
                        display_symptoms.append(random.choice(lay_terms))
                    else:
                        display_symptoms.append(symptom.replace('_', ' '))
            else:
                display_symptoms = [s.replace('_', ' ') for s in symptoms[:3]]
        else:
            display_symptoms = [s.replace('_', ' ') for s in symptoms[:3]]
        
        # Select template
        template = random.choice(self.templates)
        
        # Fill in template with grammatically correct forms
        complaint = template
        
        # Handle {symptoms} - basic list
        if "{symptoms}" in complaint:
            symptoms_text = " and ".join(display_symptoms[:2])
            complaint = complaint.replace("{symptoms}", symptoms_text)
        
        # Handle {symptoms_phrase_form} - symptoms in phrase form (for "My X" constructions)
        if "{symptoms_phrase_form}" in complaint:
            symptom_phrases = [self._get_symptom_form(s, "phrase") for s in display_symptoms[:2]]
            symptoms_phrase = " and ".join(symptom_phrases)
            complaint = complaint.replace("{symptoms_phrase_form}", symptoms_phrase)
        
        # Handle {main_symptom_gerund} - gerund form
        if "{main_symptom_gerund}" in complaint:
            main = display_symptoms[0] if display_symptoms else "not feeling well"
            main_gerund = self._get_symptom_form(main, "gerund")
            complaint = complaint.replace("{main_symptom_gerund}", main_gerund)
        
        # Handle {main_symptom_noun} - noun form
        if "{main_symptom_noun}" in complaint:
            main = display_symptoms[0] if display_symptoms else "illness"
            main_noun = self._get_symptom_form(main, "noun")
            complaint = complaint.replace("{main_symptom_noun}", main_noun)
        
        # Handle {main_symptom_phrase} - phrase form
        if "{main_symptom_phrase}" in complaint:
            main = display_symptoms[0] if display_symptoms else "not feeling well"
            main_phrase = self._get_symptom_form(main, "phrase")
            complaint = complaint.replace("{main_symptom_phrase}", main_phrase)
        
        # Handle {main_symptom_action} - action form
        if "{main_symptom_action}" in complaint:
            main = display_symptoms[0] if display_symptoms else "feel unwell"
            main_action = self._get_symptom_form(main, "action")
            complaint = complaint.replace("{main_symptom_action}", main_action)
        
        # Handle {other_symptom_phrase} - other symptoms in phrase form
        if "{other_symptom_phrase}" in complaint:
            other = display_symptoms[1] if len(display_symptoms) > 1 else "feeling unwell"
            other_phrase = self._get_symptom_form(other, "phrase")
            complaint = complaint.replace("{other_symptom_phrase}", other_phrase)
        
        # Replace standard placeholders
        complaint = complaint.replace("{severity}", random.choice(self.severity_terms))
        complaint = complaint.replace("{time}", random.choice(self.time_terms))
        complaint = complaint.replace("{feeling}", random.choice(self.feeling_terms))
        
        return complaint
    
    def _get_symptom_form(self, symptom: str, form: str) -> str:
        """
        Get the grammatically correct form of a symptom.
        
        Args:
            symptom: Raw symptom string (e.g., "wheezing", "shortness_of_breath")
            form: Desired grammatical form ("gerund", "noun", "phrase", "action")
            
        Returns:
            Grammatically correct symptom form
        """
        # Normalize symptom
        symptom_key = symptom.lower().replace(' ', '_')
        
        # Also try with spaces for lay terms
        symptom_key_spaced = symptom.lower()
        
        # Check if we have grammar rules for this symptom
        if symptom_key in self.symptom_grammar_rules:
            return self.symptom_grammar_rules[symptom_key].get(form, symptom.replace('_', ' '))
        if symptom_key_spaced in self.symptom_grammar_rules:
            return self.symptom_grammar_rules[symptom_key_spaced].get(form, symptom)
        
        # Fallback: apply default grammar rules
        symptom_clean = symptom.replace('_', ' ')
        
        if form == "gerund":
            # Check if it's already a gerund (ends in -ing)
            if symptom_clean.endswith('ing'):
                return symptom_clean
            # Check if it starts with "can't" - convert to proper form
            if symptom_clean.startswith("can't"):
                base = symptom_clean.replace("can't ", "")
                return f"having trouble {base}"
            # For phrases with "of", prepend "having"
            if ' of ' in symptom_clean or symptom_clean.startswith('difficulty'):
                return f"having {symptom_clean}"
            # For most symptoms, add "experiencing"
            return f"experiencing {symptom_clean}"
        elif form == "noun":
            # Handle "can't X" phrases
            if symptom_clean.startswith("can't"):
                return "breathing difficulty"
            # Remove trailing -ing if present and convert to noun form
            if symptom_clean.endswith('ing'):
                return symptom_clean  # Keep as-is (e.g., "wheezing" is both verb and noun)
            # Handle adjectives
            if symptom_clean in ['breathless', 'wheezy', 'dizzy']:
                return symptom_clean + 'ness'
            return symptom_clean
        elif form == "phrase":
            # Handle "can't X" phrases
            if symptom_clean.startswith("can't"):
                base = symptom_clean.replace("can't ", "")
                return f"trouble {base if base else 'breathing'}"
            return symptom_clean
        elif form == "action":
            # Create action phrase (e.g., "can't stop X" or "have X")
            if symptom_clean.startswith("can't"):
                return symptom_clean  # Already an action
            if symptom_clean.endswith('ing'):
                return f"can't stop {symptom_clean}"
            return f"have {symptom_clean}"
        
        return symptom_clean
    
    def generate_complaint(
        self,
        symptoms: List[str],
        condition_code: str,
        use_lay_terms: bool = True
    ) -> str:
        """
        Generate a natural language patient complaint.
        Uses LLM if available, otherwise falls back to templates.
        
        Args:
            symptoms: List of medical symptoms
            condition_code: ICD-10 condition code
            use_lay_terms: Whether to use lay language (for template mode)
            
        Returns:
            Natural language complaint string
        """
        if self.use_llm:
            return self._generate_with_llm(symptoms, condition_code)
        else:
            return self._generate_with_template(symptoms, condition_code, use_lay_terms)
    
    def generate_multiple_complaints(
        self,
        condition_code: str,
        num_complaints: int = 5
    ) -> List[str]:
        """
        Generate multiple complaints for a condition.
        
        Args:
            condition_code: ICD-10 condition code
            num_complaints: Number of complaints to generate
            
        Returns:
            List of complaint strings
        """
        complaints = []
        symptom_gen = SymptomGenerator()
        
        for _ in range(num_complaints):
            symptoms = symptom_gen.generate_symptoms(condition_code)
            complaint = self.generate_complaint(symptoms, condition_code)
            complaints.append(complaint)
        
        return complaints
    
    def forward(
        self,
        symptoms_batch: List[List[str]],
        condition_codes: List[str]
    ) -> List[str]:
        """
        PyTorch-compatible forward pass for batch generation.
        
        Args:
            symptoms_batch: List of symptom lists
            condition_codes: List of condition codes
            
        Returns:
            List of generated complaints
        """
        complaints = []
        for symptoms, code in zip(symptoms_batch, condition_codes):
            complaint = self.generate_complaint(symptoms, code)
            complaints.append(complaint)
        return complaints
    
    def __call__(self, *args, **kwargs):
        """Make the generator callable."""
        if len(args) == 2 and isinstance(args[0], list):
            # Called with (symptoms, condition_code)
            return self.generate_complaint(*args, **kwargs)
        else:
            # Called as forward
            return self.forward(*args, **kwargs)
