# Deep-Learning Upgrade Highlights

This note summarises how the deep-learning version of PHAITA differs from the lightweight template baseline.

## Major Model Components
- **DiagnosisDiscriminator (`phaita/models/discriminator.py`)**: Wraps a DeBERTa encoder and symptom Graph Attention Network, providing diagnosis logits and real/fake discrimination in a single module.
- **ComplaintGenerator (`phaita/models/generator.py`)**: Switches between quantised Mistral 7B for rich language and a CPU-friendly template fallback while keeping a consistent API.
- **SymptomGraphModule (`phaita/models/gnn_module.py`)**: Builds symptom co-occurrence graphs and delivers attention-based embeddings to the discriminator.

## Training Loop Enhancements
- Shared `AdversarialTrainer` handles alternating optimisation, gradient clipping, cosine learning-rate schedules, and curriculum sampling between synthetic and forum-style text.
- Diversity, realism, and medical-consistency losses are combined with the adversarial objectives to stabilise training.

## Supporting Utilities
- `phaita/utils/realism_scorer.py` scores complaints with transformer embeddings when available.
- `phaita/utils/metrics.py` captures accuracy plus lexical and semantic diversity metrics.
- `phaita/utils/config.py` centralises YAML-driven configuration for experiments.

## Operating Modes
- **Full stack**: Requires PyTorch, transformers, and (optionally) bitsandbytes for 4-bit Mistral; ideal for GPU experimentation.
- **Fallback mode**: Retains deterministic templates and NumPy-based logic so demos and tests run in constrained environments.

For hands-on setup instructions, hardware notes, and troubleshooting tips, refer to `DEEP_LEARNING_GUIDE.md`.
