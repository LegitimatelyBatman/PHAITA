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
from ..generation.patient_agent import (
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
except ImportError as e:
    raise ImportError(
        "bitsandbytes is required for 4-bit quantization in ComplaintGenerator. "
        "Install with: pip install bitsandbytes==0.44.1\n"
        "Note: bitsandbytes requires CUDA. For CPU-only systems, this dependency must still be "
        "installed but use_4bit=False should be passed to the model."
    ) from e


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
            ValueError: If use_pretrained is False
            RuntimeError: If model loading fails
        """
        super().__init__()

        if not use_pretrained:
            raise ValueError(
                "ComplaintGenerator requires use_pretrained=True. "
                "Template-based fallback has been removed. "
                "Ensure transformers and bitsandbytes are properly installed."
            )

        self.conditions = RespiratoryConditions.get_all_conditions()
        self.symptom_generator = SymptomGenerator(self.conditions)
        self.current_presentation: Optional[PatientPresentation] = None
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
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

        # Load LLM
        self._load_llm(model_name, use_4bit)

    def reload_conditions(self, conditions: Optional[Dict[str, Dict]] = None) -> None:
        """Refresh the condition catalogue and vocabularies at runtime."""

        self.conditions = conditions or RespiratoryConditions.get_all_conditions()
        self.symptom_generator.reload_conditions(self.conditions)
    
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
            print(f"Loading {model_name}...")
            
            if use_4bit and torch.cuda.is_available():
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

        # Format symptoms as natural language
        max_terms = presentation.vocabulary_profile.max_terms_per_response
        phrase_symptoms = [
            self._format_symptom(presentation, symptom, form="phrase")
            for symptom in presentation.symptoms[:max_terms]
        ]
        symptom_list = ", ".join(phrase_symptoms)

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
        if self.model is None:
            raise RuntimeError(
                "Model not loaded. Ensure ComplaintGenerator was initialized with use_pretrained=True "
                "and model loaded successfully."
            )

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

        if not selected:
            response = "I'm mostly feeling generally unwell, nothing specific to add."
        else:
            phrases = [
                self._format_symptom(presentation, symptom, form="phrase")
                for symptom in selected
            ]
            lowered_prompt = prompt.lower()

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
