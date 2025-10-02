"""
Interactive question generator for medical triage.
Uses Mistral-7B for dynamic question generation with template fallback.
"""

from typing import List, Optional, Dict
import random
import torch
import torch.nn as nn

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import bitsandbytes
    BITSANDBYTES_AVAILABLE = True
except ImportError:
    BITSANDBYTES_AVAILABLE = False


class QuestionGenerator(nn.Module):
    """
    Generates clarifying questions for interactive triage.
    Uses LLM when available, falls back to templates.
    """
    
    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
        use_pretrained: bool = False,  # Default False to avoid loading large model
        use_4bit: bool = True,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        device: Optional[str] = None
    ):
        """
        Initialize the question generator.
        
        Args:
            model_name: Name of the LLM to use
            use_pretrained: Whether to load pretrained LLM
            use_4bit: Whether to use 4-bit quantization
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            device: Device to load model on
        """
        super().__init__()
        
        self.use_llm = use_pretrained and TRANSFORMERS_AVAILABLE
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Try to load LLM
        if self.use_llm:
            try:
                self._load_llm(model_name, use_4bit)
            except Exception as e:
                print(f"Warning: Could not load LLM for questions: {e}")
                print("Falling back to template-based questions")
                self.use_llm = False
                self._init_template_generator()
        else:
            self._init_template_generator()
    
    def _load_llm(self, model_name: str, use_4bit: bool):
        """Load the language model."""
        print(f"Loading {model_name} for question generation...")
        
        if use_4bit and BITSANDBYTES_AVAILABLE and torch.cuda.is_available():
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
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.eval()
        print(f"✓ Loaded {model_name} successfully")
    
    def _init_template_generator(self):
        """Initialize template-based fallback generator."""
        self.model = None
        self.tokenizer = None
        
        # Add learnable parameters for optimizer compatibility
        self.template_embeddings = nn.Parameter(torch.randn(10, 32))
        
        # Question templates
        self.clarifying_questions = {
            "cough": [
                "Is your cough dry or are you bringing up mucus?",
                "How long have you had the cough?",
                "Does the cough get worse at certain times?"
            ],
            "dyspnea": [
                "Does the breathlessness happen at rest or with activity?",
                "Can you lie flat or does it get worse when lying down?",
                "How quickly did this come on?"
            ],
            "chest_pain": [
                "Where exactly is the chest pain?",
                "Does the pain get worse with breathing?",
                "Is the pain sharp or dull?"
            ],
            "fever": [
                "How high is your temperature?",
                "When did the fever start?",
                "Do you have chills or night sweats?"
            ]
        }
    
    def _create_question_prompt(
        self,
        symptoms: List[str],
        previous_answers: Optional[List[str]] = None
    ) -> str:
        """
        Create a prompt for the LLM to generate a clarifying question.
        
        Args:
            symptoms: List of symptoms
            previous_answers: Previously provided answers
            previous_questions: Questions asked earlier in the exchange
            conversation_history: Structured history of question/answer pairs
            
        Returns:
            Formatted prompt string
        """
        symptom_list = ", ".join(s.replace("_", " ") for s in symptoms[:5])
        
        prompt = f"""[INST] You are a medical triage assistant generating a clarifying question.

Patient symptoms: {symptom_list}

Generate ONE brief, clear clarifying question to better understand the patient's condition. The question should:
- Be direct and specific
- Help differentiate between possible diagnoses
- Be easy for a patient to answer
- Focus on timing, severity, or associated symptoms

Question: [/INST]"""
        
        return prompt
    
    def _generate_with_llm(
        self,
        symptoms: List[str],
        previous_answers: Optional[List[str]] = None,
        previous_questions: Optional[List[str]] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """
        Generate question using LLM.
        
        Args:
            symptoms: List of symptoms
            previous_answers: Previously provided answers
            previous_questions: Questions asked earlier in the exchange
            conversation_history: Structured history of question/answer pairs
            
        Returns:
            Generated question string
        """
        if not self.use_llm or self.model is None:
            return self._generate_with_template(
                symptoms,
                previous_answers=previous_answers,
                previous_questions=previous_questions,
                conversation_history=conversation_history,
            )
        
        try:
            prompt = self._create_question_prompt(symptoms, previous_answers)
            
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
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract the question
            if "[/INST]" in generated_text:
                question = generated_text.split("[/INST]")[-1].strip()
            else:
                question = generated_text[len(prompt):].strip()
            
            # Clean up
            question = question.split('\n')[0]  # Take first line
            question = question[:300]  # Limit length
            
            # Ensure it ends with a question mark
            if question and not question.endswith('?'):
                question += '?'
            
            return question if question else self._generate_with_template(symptoms, previous_answers)
        
        except Exception as e:
            print(f"Warning: LLM question generation failed: {e}")
            return self._generate_with_template(
                symptoms,
                previous_answers=previous_answers,
                previous_questions=previous_questions,
                conversation_history=conversation_history,
            )
    
    def _generate_with_template(
        self,
        symptoms: List[str],
        previous_answers: Optional[List[str]] = None,
        previous_questions: Optional[List[str]] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """
        Generate question using templates (fallback).
        
        Args:
            symptoms: List of symptoms
            previous_answers: Previously provided answers
            previous_questions: Questions asked earlier in the exchange
            conversation_history: Structured history of question/answer pairs
            
        Returns:
            Question string
        """
        # Find symptoms that have questions
        available_symptoms = []
        for symptom in symptoms:
            # Normalize symptom
            base_symptom = symptom.replace('_', ' ').lower()
            
            # Check if we have questions for this symptom
            for key in self.clarifying_questions:
                if key in base_symptom or base_symptom in key:
                    available_symptoms.append(key)
                    break
        
        asked = set(previous_questions or [])

        if available_symptoms:
            symptom = random.choice(available_symptoms)
            questions = [q for q in self.clarifying_questions[symptom] if q not in (previous_questions or [])]
            if questions:
                return random.choice(questions)

        default_question = "Can you tell me more about when these symptoms started?"
        if default_question in asked:
            default_question = "Are there any other symptoms we should know about?"

        return default_question
    
    def generate_clarifying_question(
        self,
        symptoms: List[str],
        previous_answers: Optional[List[str]] = None,
        previous_questions: Optional[List[str]] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """
        Generate a clarifying question based on symptoms.
        Uses LLM if available, otherwise templates.
        
        Args:
            symptoms: List of symptoms
            previous_answers: Previously provided answers
            previous_questions: Questions asked earlier in the exchange
            conversation_history: Structured history of question/answer pairs
            
        Returns:
            Clarifying question string
        """
        if self.use_llm:
            return self._generate_with_llm(
                symptoms,
                previous_answers=previous_answers,
                previous_questions=previous_questions,
                conversation_history=conversation_history,
            )
        else:
            return self._generate_with_template(
                symptoms,
                previous_answers=previous_answers,
                previous_questions=previous_questions,
                conversation_history=conversation_history,
            )
    
    def generate_followup_questions(
        self,
        condition_code: str,
        num_questions: int = 3
    ) -> List[str]:
        """
        Generate follow-up questions for a suspected condition.
        
        Args:
            condition_code: ICD-10 condition code
            num_questions: Number of questions to generate
            
        Returns:
            List of question strings
        """
        general_questions = [
            "Have you had similar symptoms before?",
            "Are you taking any medications?",
            "Do you have any other medical conditions?",
            "Does anything make the symptoms better or worse?",
            "Have you been around anyone who has been sick?"
        ]
        
        return random.sample(general_questions, min(num_questions, len(general_questions)))
