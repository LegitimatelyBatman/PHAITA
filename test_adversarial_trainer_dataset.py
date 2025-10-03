import importlib
import sys
import types
from typing import Dict, List, Optional
from unittest import mock

import torch
import torch.nn as nn


class DummyTokenizer:
    def __call__(self, texts, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        batch = list(texts)
        seq_len = max(1, min(kwargs.get("max_length", 8), 8))
        input_ids = torch.ones(len(batch), seq_len, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class DummyPresentation:
    def __init__(self, condition_code: str):
        self.condition_code = condition_code
        self.complaint_text = f"placeholder complaint for {condition_code}"


class DummySymptomGenerator:
    def generate_symptoms(self, condition_code: str) -> DummyPresentation:  # type: ignore[override]
        return DummyPresentation(condition_code)


class DummyComplaintGenerator(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.dummy_param = nn.Parameter(torch.zeros(1))
        self._counter = 0

    def generate_complaint(self, presentation: DummyPresentation, **kwargs) -> DummyPresentation:  # type: ignore[override]
        presentation.complaint_text = f"synthetic complaint {self._counter} for {presentation.condition_code}"
        self._counter += 1
        return presentation

    def create_guidance_prompt(self, presentation: DummyPresentation) -> str:  # type: ignore[override]
        return f"prompt for {presentation.condition_code}"

    def compute_guided_log_probs(self, prompts, target_texts, max_length=512):  # type: ignore[override]
        if len(prompts) != len(target_texts):
            raise ValueError("prompts and target_texts length mismatch")
        batch = len(target_texts)
        device = self.dummy_param.device
        log_prob = self.dummy_param.expand(batch).clone()
        token_log_probs = log_prob.unsqueeze(1)
        mask = torch.ones(batch, 1, dtype=torch.bool, device=device)
        return {
            "token_log_probs": token_log_probs,
            "sequence_log_probs": log_prob,
            "token_mask": mask,
            "prompt_lengths": torch.ones(batch, dtype=torch.long, device=device),
        }


class DummyRealismScorer:
    def __init__(self, device: Optional[str] = None, **kwargs):
        self.device = torch.device(device or "cpu")

    def get_detailed_scores(self, complaint: str) -> Dict[str, float]:
        return {
            "fluency": 0.0,
            "coherence": 0.0,
            "medical_relevance": 0.0,
            "overall": 0.0,
        }

    def compute_realism_score(self, complaints):
        if not complaints:
            return torch.zeros(0, device=self.device)
        scores = torch.linspace(0.2, 0.8, steps=len(complaints), device=self.device)
        return scores


class DummyRealismLoss:
    def __init__(self, scorer=None, weight: float = 0.0):
        self.weight = weight

    def __call__(self, complaints: List[str]) -> torch.Tensor:
        return torch.tensor(0.0)


class DummyDiscriminator(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        from phaita.data.icd_conditions import RespiratoryConditions

        self.conditions = RespiratoryConditions.get_all_conditions()
        self.condition_codes = list(self.conditions.keys())
        self.num_conditions = len(self.condition_codes)
        self.dummy_param = nn.Parameter(torch.zeros(1))
        self.logit_scale = nn.Parameter(torch.tensor(0.4))
        self.logit_bias = nn.Parameter(torch.tensor(-0.2))
        self.tokenizer = DummyTokenizer()
        self.feature_dim = 8

    def forward(self, complaints, return_features: bool = False):  # type: ignore[override]
        batch_size = len(complaints)
        device = self.dummy_param.device
        diagnosis_logits = torch.zeros(batch_size, self.num_conditions, device=device)
        base = torch.arange(batch_size, dtype=torch.float32, device=device)
        discriminator_scores = base.unsqueeze(1) * self.logit_scale + self.logit_bias
        outputs = {
            "diagnosis_logits": diagnosis_logits,
            "discriminator_scores": discriminator_scores,
        }
        if return_features:
            feature_row = torch.arange(self.feature_dim, dtype=torch.float32, device=device)
            features = torch.stack([feature_row + idx for idx in range(batch_size)], dim=0)
            outputs["text_features"] = features
        return outputs

    def predict_diagnosis(self, complaints, top_k: int = 1):  # type: ignore[override]
        batch_predictions = []
        for _ in complaints:
            batch_predictions.append([
                {
                    "condition_code": self.condition_codes[0],
                    "probability": 0.5,
                }
            ])
        return batch_predictions


class ModulePatcher:
    def __init__(self):
        self.backups = {}
        self.stubbed = []

    def __enter__(self):
        modules_to_clear = [
            "phaita.models.generator",
            "phaita.models.discriminator",
            "phaita.utils.realism_scorer",
            "phaita.training.adversarial_trainer",
        ]
        for name in modules_to_clear:
            self.backups[name] = sys.modules.get(name)
            if name in sys.modules:
                del sys.modules[name]

        generator_module = types.ModuleType("phaita.models.generator")
        generator_module.SymptomGenerator = DummySymptomGenerator
        generator_module.ComplaintGenerator = DummyComplaintGenerator
        sys.modules["phaita.models.generator"] = generator_module
        self.stubbed.append("phaita.models.generator")

        discriminator_module = types.ModuleType("phaita.models.discriminator")
        discriminator_module.DiagnosisDiscriminator = DummyDiscriminator
        sys.modules["phaita.models.discriminator"] = discriminator_module
        self.stubbed.append("phaita.models.discriminator")

        realism_module = types.ModuleType("phaita.utils.realism_scorer")
        realism_module.create_realism_scorer = lambda *args, **kwargs: DummyRealismScorer(**kwargs)
        realism_module.RealismLoss = DummyRealismLoss
        sys.modules["phaita.utils.realism_scorer"] = realism_module
        self.stubbed.append("phaita.utils.realism_scorer")

        return self

    def __exit__(self, exc_type, exc, tb):
        for name in self.stubbed:
            sys.modules.pop(name, None)
        for name, module in self.backups.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
        return False


def test_discriminator_uses_physician_verified_corpus():
    with ModulePatcher():
        trainer_module = importlib.import_module("phaita.training.adversarial_trainer")
        importlib.reload(trainer_module)

        dataset = [
            {"text": "Verified asthma case", "label": "J45.9"},
            {"text": "Verified bronchitis case", "label": "J20.9"},
        ]

        trainer = trainer_module.AdversarialTrainer(
            use_curriculum_learning=False,
            use_forum_data=False,
            realism_weight=0.0,
            real_dataset=dataset,
        )

        assert len(trainer.real_dataset) == len(dataset)
        for entry in trainer.real_dataset:
            assert entry["text"] in {item["text"] for item in dataset}
            assert entry["tokenized"] is not None
            assert set(entry["tokenized"].keys()) == {"input_ids", "attention_mask"}

        captured_batches = []

        def fake_train_discriminator(real_complaints, real_labels, fake_complaints, label_mask=None):
            captured_batches.append(
                (
                    list(real_complaints),
                    list(fake_complaints),
                    real_labels.detach().cpu().tolist(),
                    None if label_mask is None else label_mask.detach().cpu().tolist(),
                )
            )
            trainer.disc_optimizer.zero_grad()
            trainer.disc_optimizer.step()
            return {
                "disc_loss": 0.0,
                "real_diagnosis_loss": 0.0,
                "real_adv_loss": 0.0,
                "fake_adv_loss": 0.0,
                "unsupervised_loss": 0.0,
            }

        def fake_train_generator(batch_size):
            trainer.gen_optimizer.zero_grad()
            trainer.gen_optimizer.step()
            return {
                "gen_loss": 0.0,
                "gen_adv_loss": 0.0,
                "diversity_loss": 0.0,
                "realism_loss": 0.0,
            }

        synthetic_counter = {"value": 0}

        def fake_generate_training_batch(batch_size, return_metadata=False):
            complaints = []
            codes = []
            labels = []
            for _ in range(batch_size):
                idx = synthetic_counter["value"]
                complaint = f"synthetic_{idx}"
                code = trainer.condition_codes[idx % len(trainer.condition_codes)]
                label_idx = trainer.condition_codes.index(code)
                complaints.append(complaint)
                codes.append(code)
                labels.append(label_idx)
                synthetic_counter["value"] += 1
            label_tensor = torch.tensor(labels, dtype=torch.long, device=trainer.device)
            if return_metadata:
                prompts = [f"prompt_{i}" for i in range(len(complaints))]
                presentations = [DummyPresentation(code) for code in codes]
                for pres, text in zip(presentations, complaints):
                    pres.complaint_text = text
                return complaints, codes, label_tensor, prompts, presentations
            return complaints, codes, label_tensor

        trainer.train_discriminator_step = fake_train_discriminator  # type: ignore[assignment]
        trainer.train_generator_step = fake_train_generator  # type: ignore[assignment]
        trainer.generate_training_batch = fake_generate_training_batch  # type: ignore[assignment]
        trainer.evaluate = lambda complaints, labels: {"accuracy": 1.0}  # type: ignore[assignment]
        trainer._compute_epoch_diversity_metrics = lambda complaints: {"avg_diversity": 0.0}  # type: ignore[assignment]
        trainer._compute_epoch_realism_metrics = lambda complaints: {"avg_realism": 0.0}  # type: ignore[assignment]
        trainer._compute_medical_consistency = lambda complaints, codes: 0.0  # type: ignore[assignment]
        trainer._log_adversarial_failures = lambda complaints, metrics: None  # type: ignore[assignment]
        trainer.save_models = lambda name: None  # type: ignore[assignment]

        trainer.train(num_epochs=1, batch_size=2, eval_interval=1, save_interval=5)

        assert captured_batches, "Expected discriminator to receive at least one batch"
        real_batch, fake_batch, label_indices, label_mask = captured_batches[0]

        curated_texts = {entry["text"] for entry in trainer.real_dataset}
        assert set(real_batch).issubset(curated_texts)
        assert real_batch != fake_batch

        curated_indices = {entry["label_idx"] for entry in trainer.real_dataset}
        assert set(label_indices).issubset(curated_indices)
        assert all(value == 1 for value in label_mask)


def test_forum_samples_are_marked_unlabeled():
    with ModulePatcher():
        trainer_module = importlib.import_module("phaita.training.adversarial_trainer")
        importlib.reload(trainer_module)

        trainer = trainer_module.AdversarialTrainer(
            use_curriculum_learning=True,
            use_forum_data=False,
            realism_weight=0.0,
        )

        trainer.forum_complaints = [
            "forum complaint a",
            "forum complaint b",
            "forum complaint c",
        ]

        def deterministic_batch(batch_size, return_metadata=False):
            complaints = [f"synthetic_{idx}" for idx in range(batch_size)]
            codes = [trainer.condition_codes[idx % len(trainer.condition_codes)] for idx in range(batch_size)]
            labels = torch.tensor([
                trainer.condition_codes.index(code) for code in codes
            ], dtype=torch.long, device=trainer.device)
            if return_metadata:
                prompts = [f"prompt_{code}" for code in codes]
                presentations = [DummyPresentation(code) for code in codes]
                for pres, text in zip(presentations, complaints):
                    pres.complaint_text = text
                return complaints, codes, labels, prompts, presentations
            return complaints, codes, labels

        trainer.generate_training_batch = deterministic_batch  # type: ignore[assignment]

        with mock.patch("random.randint", side_effect=AssertionError("random labels are not allowed")):
            complaints, labels, mask = trainer._sample_mixed_training_data(batch_size=6, forum_ratio=0.5)

        assert len(complaints) == len(labels) == len(mask)

        forum_count = (~mask).sum().item()
        assert forum_count == 3

        forum_complaints_in_batch = [
            complaint for complaint, is_labeled in zip(complaints, mask.tolist()) if not is_labeled
        ]
        assert set(forum_complaints_in_batch).issubset(set(trainer.forum_complaints))


def test_train_generator_step_updates_generator_parameters():
    with ModulePatcher():
        trainer_module = importlib.import_module("phaita.training.adversarial_trainer")
        importlib.reload(trainer_module)

        trainer = trainer_module.AdversarialTrainer(
            use_curriculum_learning=False,
            use_forum_data=False,
            realism_weight=0.0,
        )

        trainer.generator.dummy_param.data.fill_(1.0)
        initial_param = trainer.generator.dummy_param.detach().clone()

        losses = trainer.train_generator_step(batch_size=3)

        updated_param = trainer.generator.dummy_param.detach()

        assert "policy_loss" in losses
        assert torch.max(torch.abs(initial_param - updated_param)).item() > 1e-7
        assert torch.isfinite(torch.tensor(list(losses.values()))).all()


def test_discriminator_outputs_logits_and_loss_decreases():
    with ModulePatcher():
        trainer_module = importlib.import_module("phaita.training.adversarial_trainer")
        importlib.reload(trainer_module)

        trainer = trainer_module.AdversarialTrainer(
            use_curriculum_learning=False,
            use_forum_data=False,
            realism_weight=0.0,
        )

        real_complaints = ["real_a", "real_b", "real_c"]
        fake_complaints = ["fake_a", "fake_b", "fake_c", "fake_d"]
        real_labels = torch.tensor([0, 1, 2], dtype=torch.long, device=trainer.device)

        trainer.discriminator.eval()
        with torch.no_grad():
            real_outputs = trainer.discriminator(real_complaints)
            fake_outputs = trainer.discriminator(fake_complaints)

        real_logits = real_outputs["discriminator_scores"].squeeze(-1).cpu()
        assert real_logits.min().item() < 0
        assert real_logits.max().item() > 0

        ones = torch.ones_like(real_outputs["discriminator_scores"], device=trainer.device)
        zeros = torch.zeros_like(fake_outputs["discriminator_scores"], device=trainer.device)
        loss_fn = trainer.adversarial_loss

        initial_loss = (
            loss_fn(real_outputs["discriminator_scores"], ones).item()
            + loss_fn(fake_outputs["discriminator_scores"], zeros).item()
        )

        for _ in range(3):
            trainer.train_discriminator_step(real_complaints, real_labels, fake_complaints)

        trainer.discriminator.eval()
        with torch.no_grad():
            updated_real = trainer.discriminator(real_complaints)
            updated_fake = trainer.discriminator(fake_complaints)

        final_loss = (
            loss_fn(updated_real["discriminator_scores"], ones).item()
            + loss_fn(updated_fake["discriminator_scores"], zeros).item()
        )

        assert final_loss < initial_loss


def test_train_generator_step_does_not_update_discriminator_grads():
    with ModulePatcher():
        trainer_module = importlib.import_module("phaita.training.adversarial_trainer")
        importlib.reload(trainer_module)

        trainer = trainer_module.AdversarialTrainer(
            use_curriculum_learning=False,
            use_forum_data=False,
            realism_weight=0.0,
        )

        discriminator = trainer.discriminator

        for param in discriminator.parameters():
            param.grad = None

        with mock.patch.object(discriminator, "forward", wraps=discriminator.forward) as mocked_forward:
            trainer.train_generator_step(batch_size=2)

        assert mocked_forward.called
        assert discriminator.training, "Discriminator should return to training mode after generator step"
        for param in discriminator.parameters():
            assert param.grad is None, "Discriminator gradients should remain None during generator step"
