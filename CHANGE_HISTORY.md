# Change History, Audits, and Outstanding Work

## Final Product Vision
The PHAITA program is converging on an Akinator-style triage assistant. Patients
will describe their symptoms in natural language, the system will ask
clarifying questions, and the session will conclude with a ranked list of ten
probable differential diagnoses. Each diagnosis must include likelihood
estimates, notable red flags, and guidance on whether to seek emergency care or
schedule follow-up with a family physician.

## Critical Fixes Completed
The following fixes resolved the most disruptive issues discovered during early
iterations:

- **Adversarial trainer generator reference** – introduced a
  `MockGenerator` wrapper so the trainer can expose `.parameters()`, `.train()`,
  and `.eval()` hooks without crashing when optimisers touch `self.generator`.
- **Diagnosis discriminator PyTorch compatibility** – expanded the mock
  discriminator with tensor-shaped outputs, device transfer helpers,
  `.parameters()`, and serialization methods so the adversarial loop and
  integration tests run without attribute errors.
- **Synthetic grammar corrections** – added symptom-aware grammatical forms and
  template placeholders to eliminate broken patterns such as “I've been can't
  breathe,” reducing grammar errors to below 1 % in sampling tests.
- **Forum data realism** – diversified forum-style complaints with
  condition-specific symptom pools, demographic hints, and realistic mixing so
  evaluation corpora no longer recycle identical symptom lists.

## Deep-Learning Transformation Highlights
The codebase moved from placeholder components to production-grade modules:

- **Diagnosis discriminator** – now a DeBERTa encoder fused with a symptom
  graph neural network and multi-head output heads, tallying roughly 3.8 M
  parameters and supporting gradient-based training.
- **Complaint generator** – runs in two modes: a quantised Mistral-7B
  instruction-tuned model for rich narratives and an enhanced template fallback
  with 512 learnable parameters for CPU-friendly deployment.
- **Adversarial trainer** – performs real backpropagation with gradient
  clipping, curriculum scheduling, and diversity losses instead of emitting
  random tensors.
- **Supporting modules** – question generation, synthetic data, and
  preprocessing now align with the deep-learning stack and maintain backwards
  compatibility with the original API surface.

## Recent Audit Findings
A focused audit highlighted the following improvements and remaining gaps:

- **Stable diagnostic metrics** – `compute_diagnosis_metrics` guards against
  zero-row confusion matrices to prevent `nan` values during evaluation.
- **Realism scoring overhaul** – the realism scorer now loads compatible
  transformer backbones, computes perplexity with a causal language model when
  available, and falls back gracefully when dependencies are missing.
- **Conversation loop maturity** – the current question generator lacks a
  dialogue policy for multi-turn inference. Building the Akinator-style flow
  requires richer state tracking, probability updates, and repetition avoidance.
- **Diagnosis ranking orchestration** – no module currently synthesises ten
  ranked diagnoses with red-flag context; integrating Bayesian priors, risk
  scoring, and guidance content remains open work.
- **Testing coverage gaps** – interactive triage flows and escalation guidance
  lack automated tests, and realism scoring still depends on heavy pretrained
  models without lightweight alternatives.

## Outstanding Priorities
1. Design the triage conversation engine that determines when the assistant has
   collected enough evidence to surface the ten-diagnosis slate.
2. Curate clinically reviewed red-flag messaging and escalation heuristics for
   every supported diagnosis.
3. Expand automated tests to exercise multi-turn dialogues and verify emergency
   guidance triggers.
4. Profile transformer dependencies and offer lighter, configurable fallbacks
   for resource-constrained deployments.

## Verification Snapshot
Automated regression coverage spans adversarial training integrations, grammar
checks, and data realism sampling. The most recent suite runs through `pytest`
for unit and integration targets covering the fixes above.
