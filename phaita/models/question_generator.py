"""
Interactive question generator for medical triage.
Uses Mistral-7B for dynamic question generation.
Requires transformers and bitsandbytes to be properly installed.
"""

from typing import Any, Dict, List, Optional, Set
import random
import torch
import torch.nn as nn
from requests.exceptions import HTTPError
from ..utils.model_loader import load_model_and_tokenizer, ModelDownloadError
from ..utils.config import ModelConfig
from ..utils.dependency_versions import (
    TRANSFORMERS_VERSION,
    format_install_instruction,
    format_transformer_requirements,
)

# Enforce required dependencies
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
except ImportError as e:
    raise ImportError(
        "transformers is required for QuestionGenerator. "
        f"{format_install_instruction('transformers', TRANSFORMERS_VERSION)}\n"
        "GPU Requirements: CUDA-capable GPU with 4GB+ VRAM recommended for full functionality. "
        "CPU-only mode available but slower."
    ) from e

try:
    import bitsandbytes
    HAS_BITSANDBYTES = True
except (ImportError, ModuleNotFoundError):
    HAS_BITSANDBYTES = False
    # bitsandbytes is optional - will use CPU mode without quantization


DEFAULT_QUESTION_MODEL = ModelConfig().mistral_model


class QuestionGenerator(nn.Module):
    """
    Generates clarifying questions for interactive triage.
    Uses Mistral-7B with 4-bit quantization.
    Requires transformers and bitsandbytes to be installed.
    """
    
    def __init__(
        self,
        model_name: str = DEFAULT_QUESTION_MODEL,
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
                
                self.model, self.tokenizer = load_model_and_tokenizer(
                    model_name=model_name,
                    model_type="causal_lm",
                    max_retries=3,
                    timeout=300,
                    quantization_config=quantization_config,
                    device_map="auto",
                    trust_remote_code=True
                )
            else:
                self.model, self.tokenizer = load_model_and_tokenizer(
                    model_name=model_name,
                    model_type="causal_lm",
                    max_retries=3,
                    timeout=300,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None,
                    trust_remote_code=True
                )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model.eval()
            print(f"✓ Loaded {model_name} successfully")
        except ModelDownloadError as e:
            requirements = format_transformer_requirements(
                include_bitsandbytes=True,
                include_cuda_note=True,
                internet_note="- Internet connection to download model from HuggingFace Hub",
            )
            raise RuntimeError(
                f"Failed to load model {model_name}. "
                f"{e}\n"
                f"{requirements}"
            ) from e
        except (OSError, ValueError, HTTPError) as e:
            requirements = format_transformer_requirements(
                include_bitsandbytes=True,
                include_cuda_note=True,
                internet_note="- Internet connection to download model from HuggingFace Hub",
            )
            raise RuntimeError(
                f"Failed to load model {model_name}. "
                f"Error: {e}\n"
                f"{requirements}"
            ) from e
    
    def _create_question_prompt(
        self,
        symptoms: List[str],
        previous_answers: Optional[List[str]] = None,
        demographics: Optional[Dict[str, Any]] = None,
        history: Optional[Dict[str, Any]] = None,
        asked_topics: Optional[Set[str]] = None,
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
        demo_summary = self._summarize_demographics(demographics or {})
        history_summary = self._summarize_history(history or {})
        prior_answers = "; ".join(answer for answer in (previous_answers or []) if answer.strip())
        asked_topics_text = ", ".join(sorted(asked_topics)) if asked_topics else "none"

        prompt = f"""[INST] You are a medical triage assistant generating a clarifying question.

Patient symptoms: {symptom_list}
Known demographics: {demo_summary}
Known history: {history_summary}
Information already provided: {prior_answers if prior_answers else 'none'}
Topics already asked about: {asked_topics_text}

Generate ONE brief, clear clarifying question to better understand the patient's condition. The question should:
- Be direct and specific
- Help differentiate between possible diagnoses
- Be easy for a patient to answer
- Focus on timing, severity, associated symptoms, or missing demographic/history details
- Avoid repeating topics already discussed unless clarification is needed

Question: [/INST]"""

        return prompt

    def _generate_with_llm(
        self,
        symptoms: List[str],
        previous_answers: Optional[List[str]] = None,
        previous_questions: Optional[List[str]] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        demographics: Optional[Dict[str, Any]] = None,
        history: Optional[Dict[str, Any]] = None,
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
            asked_topics: Set[str] = set()
            if conversation_history:
                for entry in conversation_history:
                    question_text = (entry.get("question") or "").lower()
                    if "age" in question_text:
                        asked_topics.add("age")
                    if "medication" in question_text:
                        asked_topics.add("medications")
                    if "allerg" in question_text:
                        asked_topics.add("allergies")
                    if "family" in question_text:
                        asked_topics.add("family history")
                    if "work" in question_text or "occupation" in question_text:
                        asked_topics.add("occupation")

            prompt = self._create_question_prompt(
                symptoms,
                previous_answers,
                demographics=demographics,
                history=history,
                asked_topics=asked_topics,
            )
            
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
        demographics: Optional[Dict[str, Any]] = None,
        history: Optional[Dict[str, Any]] = None,
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

        attempts = 0
        max_attempts = max(requested * 2, requested + 1)
        while len(candidates) < requested and attempts < max_attempts:
            question = self._generate_with_llm(
                symptoms,
                previous_answers=previous_answers,
                previous_questions=list(asked),
                conversation_history=conversation_history,
                demographics=demographics,
                history=history,
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
        demographics: Optional[Dict[str, Any]] = None,
        history: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate a clarifying question based on symptoms using the LLM."""

        candidates = self.generate_candidate_questions(
            symptoms,
            previous_answers=previous_answers,
            previous_questions=previous_questions,
            conversation_history=conversation_history,
            demographics=demographics,
            history=history,
        )

        return candidates[0] if candidates else ""

    @staticmethod
    def _summarize_demographics(demographics: Dict[str, Any]) -> str:
        if not demographics:
            return "unknown"
        parts: List[str] = []
        inclusion = demographics.get("inclusion", {})
        age_ranges = inclusion.get("age_ranges")
        if age_ranges:
            formatted = []
            for entry in age_ranges[:2]:
                minimum = entry.get("min")
                maximum = entry.get("max")
                if minimum is None and maximum is None:
                    continue
                if minimum == maximum:
                    formatted.append(f"age {minimum}")
                else:
                    formatted.append(f"age {minimum}-{maximum}")
            if formatted:
                parts.append(", ".join(formatted))
        for key in ("sexes", "ethnicities", "occupations", "social_history", "risk_factors", "exposures"):
            values = inclusion.get(key)
            if values:
                parts.append(f"{key.replace('_', ' ')}: {', '.join(values[:3])}")
        return "; ".join(parts) if parts else "unspecified"

    @staticmethod
    def _summarize_history(history: Dict[str, Any]) -> str:
        if not history:
            return "unknown"
        inclusion = history.get("inclusion", {})
        parts: List[str] = []
        for key in (
            "past_conditions",
            "medications",
            "allergies",
            "recent_events",
            "family_history",
            "lifestyle",
            "supports",
        ):
            values = inclusion.get(key)
            if values:
                parts.append(f"{key.replace('_', ' ')}: {', '.join(values[:3])}")
        if inclusion.get("last_meal"):
            parts.append("last meal noted")
        return "; ".join(parts) if parts else "no details"
    
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
