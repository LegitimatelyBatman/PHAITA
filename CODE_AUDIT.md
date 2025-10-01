# Codebase Audit and Implementation Notes

## Scope and Context
This repository assembles data generators, probabilistic models, and evaluation utilities that will eventually power an Akinator-style conversational triage application. The final product is expected to converse with patients, ask clarifying questions iteratively, and surface a ranked list of ten differential diagnoses with probability estimates, red-flag guidance, and escalation advice (emergency department vs. family physician follow-up).

## Issues Fixed in this Audit
- **Division-by-zero handling in diagnostic metrics**: `compute_diagnosis_metrics` previously divided confusion matrix diagonals by row sums without guarding against empty rows, producing `nan` values and runtime warnings during evaluation. The function now masks zero rows before computing per-class accuracy statistics, ensuring stable metrics and enabling downstream calibration logic to trust the results. 【F:phaita/utils/metrics.py†L52-L70】
- **Realism scorer perplexity logic**: The realism scorer attempted to instantiate a `text-classification` pipeline with GPT-2, which is incompatible and threw initialization errors. It also lacked any true perplexity measurement. The scorer now falls back gracefully when transformers are unavailable, loads a causal language model only when possible, and blends attention-based fluency with an actual perplexity-derived score. 【F:phaita/utils/realism_scorer.py†L34-L175】

## Additional Observations and Risks
- **Conversation loop gaps**: `QuestionGenerator` defaults to a small template pool unless a massive LLM is loaded at runtime. It does not maintain dialogue state beyond symptom lists, so building an Akinator-style experience will require a richer policy layer to track answered questions, update diagnosis probabilities, and avoid repetition. 【F:phaita/models/question_generator.py†L24-L199】
- **Differential diagnosis ranking**: The current probabilistic models (Bayesian networks, discriminator/generator pair) expose sampling utilities but there is no end-to-end routine that surfaces ten ranked diagnoses with probability annotations and red-flag guidance. A dedicated triage orchestrator should integrate `EnhancedBayesianNetwork`, risk scoring, and clinical guidance content. 【F:phaita/models/enhanced_bayesian_network.py†L1-L189】
- **Evaluation coverage**: There are no automated tests for the interactive triage flow or for verifying that red-flag messaging is surfaced for high-risk symptom clusters. As the project moves toward production, integration tests should simulate multi-turn conversations and assert that escalation advice is triggered appropriately.
- **Data realism dependencies**: The realism scorer currently attempts to load large pretrained models. Without dependency management guards or lightweight fallbacks, this may hinder deployment in resource-constrained environments. Consider providing configuration flags or smaller distilled models explicitly tuned for medical complaint text.

## Recommended Next Steps
1. **Design the triage conversation engine** that will call the question generator, update Bayesian priors, and decide when enough evidence exists to output the ten-diagnosis slate.
2. **Curate red-flag content and escalation heuristics**, ensuring each diagnosis includes actionable safety guidance aligned with clinical standards.
3. **Expand automated tests** to cover conversational loops, ensuring clarifying questions adapt based on prior answers and that emergency guidance is never omitted when severe combinations are detected.
4. **Profile and optimize language model usage**, potentially swapping large general-purpose models for domain-tuned, parameter-efficient adapters to meet latency and compute constraints in a patient-facing application.
