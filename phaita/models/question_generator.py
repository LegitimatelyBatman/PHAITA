"""
Interactive question generator for medical triage.
Uses Mistral-7B for dynamic question generation.
Requires transformers and bitsandbytes to be properly installed.
"""

from typing import List, Optional, Dict, Set
import random
import torch
import torch.nn as nn

# Enforce required dependencies
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
except ImportError as e:
    raise ImportError(
        "transformers is required for QuestionGenerator. "
        "Install with: pip install transformers==4.46.0\n"
        "GPU Requirements: CUDA-capable GPU with 4GB+ VRAM recommended for full functionality. "
        "CPU-only mode available but slower."
    ) from e

try:
    import bitsandbytes
except ImportError as e:
    raise ImportError(
        "bitsandbytes is required for 4-bit quantization in QuestionGenerator. "
        "Install with: pip install bitsandbytes==0.44.1\n"
        "Note: bitsandbytes requires CUDA. For CPU-only systems, this dependency must still be "
        "installed but use_4bit=False should be passed to the model."
    ) from e


class QuestionGenerator(nn.Module):
    """
    Generates clarifying questions for interactive triage.
    Uses Mistral-7B with 4-bit quantization.
    Requires transformers and bitsandbytes to be installed.
    """
    
    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
        use_pretrained: bool = True,
        use_4bit: bool = True,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        device: Optional[str] = None
    ):
        """
        Initialize the question generator.
        
        Args:
            model_name: Name of the LLM to use
            use_pretrained: Whether to load pretrained LLM (must be True)
            use_4bit: Whether to use 4-bit quantization (requires CUDA)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            device: Device to load model on
        
        Raises:
            ValueError: If use_pretrained is False
            RuntimeError: If model loading fails
        """
        super().__init__()
        
        if not use_pretrained:
            raise ValueError(
                "QuestionGenerator requires use_pretrained=True. "
                "Template-based fallback has been removed. "
                "Ensure transformers and bitsandbytes are properly installed."
            )
        
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.default_candidate_pool_size = 3
        
        # Load LLM
        self._load_llm(model_name, use_4bit)
    
    def _load_llm(self, model_name: str, use_4bit: bool):
        """
        Load the language model.
        
        Args:
            model_name: Name of the model to load
            use_4bit: Whether to use 4-bit quantization
        
        Raises:
            RuntimeError: If model loading fails
        """
        try:
            print(f"Loading {model_name} for question generation...")
            
            if use_4bit and torch.cuda.is_available():
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
        except Exception as e:
            raise RuntimeError(
                f"Failed to load model {model_name}. "
                f"Error: {e}\n"
                f"Requirements:\n"
                f"- transformers==4.46.0\n"
                f"- bitsandbytes==0.44.1 (for 4-bit quantization)\n"
                f"- torch==2.5.1\n"
                f"- CUDA GPU with 4GB+ VRAM recommended (CPU mode available with use_4bit=False)\n"
                f"- Internet connection to download model from HuggingFace Hub"
            ) from e
    
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
            
        Raises:
            RuntimeError: If LLM generation fails
        """
        if self.model is None:
            raise RuntimeError(
                "Model not loaded. Ensure QuestionGenerator was initialized with use_pretrained=True "
                "and model loaded successfully."
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
            
            if not question:
                raise RuntimeError("Model generated empty question")
            
            return question
        
        except Exception as e:
            raise RuntimeError(
                f"LLM question generation failed: {e}\n"
                f"Ensure model is properly loaded and device has sufficient memory."
            ) from e
    
    def generate_candidate_questions(
        self,
        symptoms: List[str],
        previous_answers: Optional[List[str]] = None,
        previous_questions: Optional[List[str]] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        *,
        num_candidates: Optional[int] = None,
    ) -> List[str]:
        """Return a ranked pool of candidate clarifying questions.

        Args:
            symptoms: List of observed symptoms.
            previous_answers: Previously provided answers.
            previous_questions: Questions that have already been asked.
            conversation_history: Structured history of prior exchanges.
            num_candidates: Optional cap on the number of questions to return.

        Returns:
            A list of unique question strings ordered by generation preference.
        """

        requested = num_candidates or self.default_candidate_pool_size
        seen: Set[str] = set()
        candidates: List[str] = []

        def _try_add(question: Optional[str]) -> None:
            if not question:
                return
            normalized = question.strip()
            if not normalized:
                return
            lower = normalized.lower()
            if lower in seen:
                return
            seen.add(lower)
            candidates.append(normalized)

        asked = set((previous_questions or []))

        generator = (
            self._generate_with_llm
            if self.use_llm
            else self._generate_with_template
        )

        attempts = 0
        max_attempts = max(requested * 2, requested + 1)
        while len(candidates) < requested and attempts < max_attempts:
            question = generator(
                symptoms,
                previous_answers=previous_answers,
                previous_questions=list(asked),
                conversation_history=conversation_history,
            )
            attempts += 1
            _try_add(question)

        if not candidates:
            fallback = "Can you tell me more about your symptoms?"
            if fallback not in asked:
                candidates.append(fallback)

        return candidates

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

        This helper maintains backwards compatibility with earlier code paths
        by selecting the first candidate from the broader question pool.
        """

        candidates = self.generate_candidate_questions(
            symptoms,
            previous_answers=previous_answers,
            previous_questions=previous_questions,
            conversation_history=conversation_history,
        )

        return candidates[0] if candidates else ""
    
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
