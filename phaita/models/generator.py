"""
Symptom and complaint generators using Bayesian networks and language models.
Requires transformers and bitsandbytes to be properly installed.
"""

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from typing import List, Optional, Dict, Iterable
from .bayesian_network import BayesianSymptomNetwork
from ..data.icd_conditions import RespiratoryConditions
from ..data.template_loader import TemplateManager
from ..utils.model_loader import load_model_and_tokenizer, ModelDownloadError
from ..generation.patient_agent import (
    PatientDemographics,
    PatientHistory,
    PatientPresentation,
    PatientSimulator,
    VocabularyProfile,
)

# Enforce required dependencies
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
except ImportError as e:
    raise ImportError(
        "transformers is required for ComplaintGenerator. "
        "Install with: pip install transformers==4.46.0\n"
        "GPU Requirements: CUDA-capable GPU with 4GB+ VRAM recommended for full functionality. "
        "CPU-only mode available but slower."
    ) from e

try:
    import bitsandbytes
    HAS_BITSANDBYTES = True
except (ImportError, ModuleNotFoundError):
    HAS_BITSANDBYTES = False
    # bitsandbytes is optional - will use CPU mode without quantization


class SymptomGenerator:
    """Generates structured patient presentations for medical conditions."""

    def __init__(self, conditions: Optional[Dict[str, Dict]] = None):
        network = BayesianSymptomNetwork(conditions=conditions)
        self.simulator = PatientSimulator(network)
        self.bayesian_network = network
        RespiratoryConditions.register_reload_hook(self.reload_conditions)

    def reload_conditions(self, conditions: Optional[Dict[str, Dict]] = None) -> None:
        """Reload the symptom catalogue for long-running services."""

        if conditions is None:
            conditions = RespiratoryConditions.get_all_conditions()
        self.bayesian_network.reload(conditions=conditions)
        self.simulator.network = self.bayesian_network

    def generate_symptoms(
        self,
        condition_code: str,
        num_symptoms: Optional[int] = None,
        vocabulary_profile: Optional[VocabularyProfile] = None,
    ) -> PatientPresentation:
        """Generate a rich patient presentation for a given condition."""
        return self.simulator.sample_presentation(
            condition_code,
            num_symptoms=num_symptoms,
            vocabulary_profile=vocabulary_profile,
        )

    def get_conditional_probabilities(
        self, condition_code: str, symptoms: Optional[Iterable[str]] = None
    ) -> Dict[str, float]:
        """Expose the conditional probabilities from the underlying simulator."""
        return self.simulator.get_conditional_probabilities(condition_code, symptoms)


class ComplaintGenerator(nn.Module):
    """
    Generates natural language patient complaints from symptoms.
    Uses Mistral-7B-Instruct with 4-bit quantization.
    Requires transformers and bitsandbytes to be installed.
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
            use_pretrained: Whether to load pretrained LLM (must be True)
            use_4bit: Whether to use 4-bit quantization (requires CUDA)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature for diversity
            top_p: Top-p sampling parameter
            device: Device to load model on
        
        Raises:
            RuntimeError: If model loading fails
        """
        super().__init__()

        self.conditions = RespiratoryConditions.get_all_conditions()
        self.symptom_generator = SymptomGenerator(self.conditions)
        self.current_presentation: Optional[PatientPresentation] = None
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize template manager for template mode
        self.template_manager = TemplateManager()
        
        # Legacy term lists kept for answer_question compatibility
        self.severity_terms = [
            "mild",
            "moderate",
            "severe",
            "terrible",
            "awful",
            "worse",
        ]
        self.time_terms = [
            "a few hours",
            "yesterday",
            "two days",
            "this morning",
            "last night",
            "a week",
        ]
        self.feeling_terms = [
            "worried",
            "scared",
            "exhausted",
            "panicked",
            "terrible",
            "awful",
        ]

        RespiratoryConditions.register_reload_hook(self.reload_conditions)
        self.tokenizer = None
        self.model = None
        self.template_mode = not use_pretrained

        if use_pretrained:
            # Load LLM
            self._load_llm(model_name, use_4bit)

    def reload_conditions(self, conditions: Optional[Dict[str, Dict]] = None) -> None:
        """Refresh the condition catalogue and vocabularies at runtime."""

        self.conditions = conditions or RespiratoryConditions.get_all_conditions()
        self.symptom_generator.reload_conditions(self.conditions)
    
    def _load_llm(self, model_name: str, use_4bit: bool):
        """
        Load the language model with retry logic.
        
        Args:
            model_name: Name of the model to load
            use_4bit: Whether to use 4-bit quantization
        
        Raises:
            RuntimeError: If model loading fails
        """
        try:
            print(f"Loading {model_name}...")
            
            if use_4bit and torch.cuda.is_available():
                # 4-bit quantization for memory efficiency
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
                # Load without quantization
                self.model, self.tokenizer = load_model_and_tokenizer(
                    model_name=model_name,
                    model_type="causal_lm",
                    max_retries=3,
                    timeout=300,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None,
                    trust_remote_code=True
                )
            
            # Set padding token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model.eval()
            print(f"✓ Loaded {model_name} successfully")
        except ModelDownloadError as e:
            raise RuntimeError(
                f"Failed to load model {model_name}. "
                f"{e}\n"
                f"Requirements:\n"
                f"- transformers==4.46.0\n"
                f"- bitsandbytes==0.44.1 (for 4-bit quantization)\n"
                f"- torch==2.5.1\n"
                f"- CUDA GPU with 4GB+ VRAM recommended (CPU mode available with use_4bit=False)\n"
                f"- Internet connection to download model from HuggingFace Hub"
            ) from e
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
    
    def _create_prompt(self, presentation: PatientPresentation) -> str:
        """
        Create a prompt for the LLM to generate a patient complaint.

        Args:
            presentation: Structured patient presentation

        Returns:
            Formatted prompt string
        """
        condition_data = self.conditions.get(presentation.condition_code, {})
        condition_name = condition_data.get("name", "respiratory condition")

        profile = presentation.demographics
        history = presentation.history_profile

        demographic_summary = profile.summary()
        if profile.social_history or profile.risk_factors:
            extras = profile.social_history + profile.risk_factors
            if extras:
                demographic_summary = f"{demographic_summary} ({', '.join(extras)})"

        history_lines = []
        if history.past_conditions:
            history_lines.append(
                "Past conditions: " + ", ".join(history.past_conditions[:3])
            )
        if history.medications:
            history_lines.append(
                "Medications: " + ", ".join(history.medications[:3])
            )
        if history.allergies:
            history_lines.append(
                "Allergies: " + ", ".join(history.allergies[:2])
            )
        if history.last_meal:
            history_lines.append(f"Last meal: {history.last_meal}")
        if history.recent_events:
            history_lines.append(
                "Recent events: " + ", ".join(history.recent_events[:2])
            )
        if history.family_history:
            history_lines.append(
                "Family history: " + ", ".join(history.family_history[:2])
            )
        if history.lifestyle:
            history_lines.append(
                "Lifestyle: " + ", ".join(history.lifestyle[:2])
            )
        if history.supports:
            history_lines.append(
                "Supports: " + ", ".join(history.supports[:2])
            )

        # Format symptoms as natural language
        max_terms = presentation.vocabulary_profile.max_terms_per_response
        phrase_symptoms = [
            self._format_symptom(presentation, symptom, form="phrase")
            for symptom in presentation.symptoms[:max_terms]
        ]
        symptom_list = ", ".join(phrase_symptoms)

        prompt = f"""[INST] You are helping generate realistic patient complaints for medical triage training.

Patient profile: {demographic_summary if demographic_summary else 'unspecified demographics'}
Relevant history: {'; '.join(history_lines) if history_lines else 'No significant history provided'}
Confirmed symptoms: {symptom_list}

Generate a natural, realistic patient complaint as if the patient is describing their symptoms to a doctor. Keep it:
- Brief (1-3 sentences)
- In first person
- Using everyday language (not medical jargon)
- Expressing concern or discomfort
- Natural and conversational
- Consistent with the demographic and history details above

Patient complaint: [/INST]"""
        
        return prompt
    
    def _generate_with_llm(self, presentation: PatientPresentation) -> str:
        """
        Generate complaint using LLM.

        Args:
            presentation: Structured patient presentation

        Returns:
            Generated complaint string
            
        Raises:
            RuntimeError: If LLM generation fails
        """
        if self.model is None or self.template_mode:
            return self._generate_template_complaint(presentation)

        try:
            prompt = self._create_prompt(presentation)

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
            
            if not complaint:
                raise RuntimeError("Model generated empty complaint")
            
            return complaint

        except Exception as e:
            raise RuntimeError(
                f"LLM generation failed: {e}\n"
                f"Ensure model is properly loaded and device has sufficient memory."
            ) from e

    def _generate_template_complaint(self, presentation: PatientPresentation) -> str:
        """Deterministic fallback complaint using diverse templates."""

        profile = presentation.demographics
        history = presentation.history_profile
        
        # Format symptoms using vocabulary profile
        symptom_phrases = [
            self._format_symptom(presentation, symptom, form="phrase")
            for symptom in presentation.symptoms[: presentation.vocabulary_profile.max_terms_per_response]
        ]
        
        # Determine severity from symptom probabilities
        avg_prob = sum(
            presentation.symptom_probabilities.get(s, 0.5) 
            for s in presentation.symptoms[:3]
        ) / max(1, min(3, len(presentation.symptoms)))
        
        if avg_prob >= 0.8:
            severity = 'severe'
        elif avg_prob >= 0.5:
            severity = 'moderate'
        else:
            severity = 'mild'
        
        # Get demographics summary
        demographics_summary = profile.summary() or None
        
        # Get trigger if available
        trigger = None
        if history.recent_events:
            trigger = history.recent_events[0]
        
        # Generate complaint using template manager
        complaint = self.template_manager.generate_complaint(
            symptoms=symptom_phrases,
            age=profile.age,
            severity=severity,
            demographics_summary=demographics_summary,
            trigger=trigger
        )
        
        return complaint

    def _format_symptom(
        self, presentation: PatientPresentation, symptom: str, form: str
    ) -> str:
        """Format symptom according to vocabulary profile."""
        # Simple formatting - just clean up underscores
        base = symptom.replace('_', ' ')
        return presentation.vocabulary_profile.translate(
            symptom, form=form, default=base
        )
    
    def generate_complaint(
        self,
        condition_code: Optional[str] = None,
        presentation: Optional[PatientPresentation] = None,
        symptoms: Optional[List[str]] = None,
        num_symptoms: Optional[int] = None,
        vocabulary_profile: Optional[VocabularyProfile] = None,
        use_lay_terms: bool = True,  # Kept for API compatibility but not used
    ) -> PatientPresentation:
        """Generate a natural language patient complaint with metadata."""

        if presentation is None:
            if condition_code is None:
                raise ValueError(
                    "Either condition_code or presentation must be provided"
                )
            if symptoms is not None:
                probabilities = self.symptom_generator.get_conditional_probabilities(
                    condition_code
                )
                weights = {
                    name: max(0.0, 1.0 - probabilities.get(name, 0.0))
                    for name in probabilities.keys()
                }
                vocab = vocabulary_profile or VocabularyProfile.default_for(symptoms)
                presentation = PatientPresentation(
                    condition_code=condition_code,
                    symptoms=list(symptoms),
                    symptom_probabilities=probabilities,
                    misdescription_weights=weights,
                    vocabulary_profile=vocab,
                    demographics=PatientDemographics(),
                    history_profile=PatientHistory(),
                )
            else:
                presentation = self.symptom_generator.generate_symptoms(
                    condition_code,
                    num_symptoms=num_symptoms,
                    vocabulary_profile=vocabulary_profile,
                )

        complaint_text = self._generate_with_llm(presentation)

        presentation.complaint_text = complaint_text
        self.current_presentation = presentation
        return presentation

    def create_guidance_prompt(self, presentation: PatientPresentation) -> str:
        """Public helper to build the LLM prompt for a presentation."""

        return self._create_prompt(presentation)

    def compute_guided_log_probs(
        self,
        prompts: List[str],
        target_texts: List[str],
        max_length: int = 512,
    ) -> Dict[str, torch.Tensor]:
        """Compute log-probabilities for target sequences under provided prompts.

        Args:
            prompts: Prompt strings used to condition generation.
            target_texts: Text that should be scored token-by-token.
            max_length: Maximum combined sequence length for tokenization.

        Returns:
            Dictionary with token-level log-probabilities, sequence log-probabilities,
            the boolean mask for valid target tokens, and prompt token lengths.
        """

        if self.model is None or self.tokenizer is None:
            raise RuntimeError(
                "Model not loaded. Ensure ComplaintGenerator was initialized with use_pretrained=True "
                "and model loaded successfully."
            )

        if len(prompts) != len(target_texts):
            raise ValueError("prompts and target_texts must have the same length")

        device = self.model.device

        # Tokenize prompts and targets separately to obtain precise lengths
        prompt_encoding = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
        )
        target_encoding = self.tokenizer(
            target_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
        )

        input_sequences: List[torch.Tensor] = []
        attention_sequences: List[torch.Tensor] = []
        label_sequences: List[torch.Tensor] = []
        target_masks: List[torch.Tensor] = []
        prompt_lengths: List[int] = []

        pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

        for idx in range(len(prompts)):
            prompt_ids = prompt_encoding["input_ids"][idx]
            prompt_mask = prompt_encoding["attention_mask"][idx]
            prompt_len = int(prompt_mask.sum().item())
            prompt_ids = prompt_ids[:prompt_len]

            target_ids = target_encoding["input_ids"][idx]
            target_mask = target_encoding["attention_mask"][idx]
            target_len = int(target_mask.sum().item())
            target_ids = target_ids[:target_len]

            combined = torch.cat([prompt_ids, target_ids], dim=0)
            attention = torch.ones_like(combined)
            labels = torch.full_like(combined, fill_value=-100)
            if target_len > 0:
                labels[-target_len:] = target_ids

            token_mask = labels != -100

            input_sequences.append(combined)
            attention_sequences.append(attention)
            label_sequences.append(labels)
            target_masks.append(token_mask)
            prompt_lengths.append(prompt_len)

        padded_inputs = pad_sequence(
            [seq.to(device) for seq in input_sequences],
            batch_first=True,
            padding_value=pad_token_id,
        )
        padded_attention = pad_sequence(
            [seq.to(device) for seq in attention_sequences],
            batch_first=True,
            padding_value=0,
        )
        padded_labels = pad_sequence(
            [seq.to(device) for seq in label_sequences],
            batch_first=True,
            padding_value=-100,
        )
        padded_mask = pad_sequence(
            [seq.to(device) for seq in target_masks],
            batch_first=True,
            padding_value=0,
        ).bool()

        outputs = self.model(
            input_ids=padded_inputs,
            attention_mask=padded_attention,
        )
        logits = outputs.logits
        log_probs = F.log_softmax(logits, dim=-1)

        safe_labels = padded_labels.clone()
        safe_labels[~padded_mask] = 0
        gathered = torch.gather(log_probs, 2, safe_labels.unsqueeze(-1)).squeeze(-1)
        token_log_probs = torch.where(padded_mask, gathered, torch.zeros_like(gathered))
        sequence_log_probs = token_log_probs.sum(dim=1)

        return {
            "token_log_probs": token_log_probs,
            "sequence_log_probs": sequence_log_probs,
            "token_mask": padded_mask,
            "prompt_lengths": torch.tensor(prompt_lengths, device=device, dtype=torch.long),
        }

    def generate_multiple_complaints(
        self,
        condition_code: str,
        num_complaints: int = 5
    ) -> List[PatientPresentation]:
        """Generate multiple patient presentations with complaints."""

        presentations: List[PatientPresentation] = []

        for _ in range(num_complaints):
            presentation = self.symptom_generator.generate_symptoms(condition_code)
            presentations.append(self.generate_complaint(presentation=presentation))

        return presentations

    def answer_question(self, prompt: str, strategy: str = "default") -> str:
        """Generate a follow-up answer consistent with the active presentation."""

        if self.current_presentation is None:
            raise ValueError("No active presentation. Call generate_complaint first.")

        presentation = self.current_presentation
        combined_scores = {
            symptom: presentation.symptom_probabilities.get(symptom, 0.0)
            - presentation.misdescription_weights.get(symptom, 0.0)
            for symptom in presentation.symptom_probabilities
        }
        sorted_symptoms = sorted(
            presentation.symptoms,
            key=lambda symptom: combined_scores.get(symptom, 0.0),
            reverse=True,
        )
        max_terms = presentation.vocabulary_profile.max_terms_per_response

        selected: List[str] = []
        for symptom in sorted_symptoms:
            if len(selected) >= max_terms:
                break
            prob = presentation.symptom_probabilities.get(symptom, 0.0)
            weight = presentation.misdescription_weights.get(symptom, 0.0)
            if prob <= weight:
                continue
            selected.append(symptom)

        history = presentation.history_profile
        demographics = presentation.demographics
        lowered_prompt = prompt.lower()

        if any(keyword in lowered_prompt for keyword in ("how old", "age")):
            response = f"I'm {demographics.age} years old."
        elif any(keyword in lowered_prompt for keyword in ("sex", "gender")):
            response = f"I'm {demographics.sex}."
        elif "occupation" in lowered_prompt or "work" in lowered_prompt:
            if demographics.occupation:
                response = f"I work as {demographics.occupation}."
            else:
                response = "I'm not currently working."
        elif "medication" in lowered_prompt:
            if history.medications:
                response = "I'm taking " + ", ".join(history.medications)
            else:
                response = "I'm not on any regular medications."
        elif "allerg" in lowered_prompt:
            if history.allergies:
                response = "I'm allergic to " + ", ".join(history.allergies)
            else:
                response = "I don't have any known allergies."
        elif "last meal" in lowered_prompt or "eat" in lowered_prompt:
            if history.last_meal:
                response = f"I last ate {history.last_meal}."
            else:
                response = "I can't remember when I last ate."
        elif "history" in lowered_prompt or "past condition" in lowered_prompt:
            if history.past_conditions:
                response = "I've had " + ", ".join(history.past_conditions)
            else:
                response = "I don't have other medical conditions."
        elif "family" in lowered_prompt:
            if history.family_history:
                response = "My family has " + ", ".join(history.family_history)
            else:
                response = "No significant family history that I know of."
        elif "support" in lowered_prompt or "help" in lowered_prompt:
            if history.supports:
                response = "We're using " + ", ".join(history.supports) + " to cope."
            else:
                response = "I haven't needed extra support yet."
        elif "event" in lowered_prompt or "what happened" in lowered_prompt:
            if history.recent_events:
                response = "It started after " + ", ".join(history.recent_events)
            else:
                response = "Nothing unusual happened before this."
        else:
            if not selected:
                response = "I'm mostly feeling generally unwell, nothing specific to add."
            else:
                phrases = [
                    self._format_symptom(presentation, symptom, form="phrase")
                    for symptom in selected
                ]

                if "how long" in lowered_prompt or "when" in lowered_prompt:
                    duration = random.choice(self.time_terms)
                    response = (
                        f"It's been going on for {duration}, and I'm still dealing with "
                        f"{' and '.join(phrases)}."
                    )
                elif "severity" in lowered_prompt or "bad" in lowered_prompt:
                    severity = random.choice(self.severity_terms)
                    response = (
                        f"It feels {severity}. I'm especially bothered by "
                        f"{' and '.join(phrases)}."
                    )
                elif strategy == "brief":
                    response = f"Mostly just {' and '.join(phrases)}."
                elif strategy == "detailed":
                    severity = random.choice(self.severity_terms)
                    duration = random.choice(self.time_terms)
                    response = (
                        f"I've been dealing with {' and '.join(phrases)} for {duration}, "
                        f"and it's felt {severity} the whole time."
                    )
                else:
                    feeling = random.choice(self.feeling_terms)
                    response = (
                        f"I'm feeling {feeling} because of {' and '.join(phrases)}."
                    )

        presentation.record_response(
            prompt,
            response,
            metadata={"symptom_mentions": list(selected)},
        )
        return response

    def forward(
        self,
        presentations: List[PatientPresentation],
    ) -> List[PatientPresentation]:
        """PyTorch-compatible forward pass for batch generation."""

        return [self.generate_complaint(presentation=p) for p in presentations]

    def __call__(self, *args, **kwargs):
        """Make the generator callable."""
        if args and isinstance(args[0], PatientPresentation):
            return self.generate_complaint(*args, **kwargs)
        else:
            # Called as forward
            return self.forward(*args, **kwargs)
