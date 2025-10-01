# PHAITA Project Summary

## Problem
Emergency clinicians must triage patients quickly, yet labelled datasets capturing how people actually describe symptoms are scarce. Models trained on clinical notes often fail when confronted with colloquial language, edge cases, or noisy self-reports.

## Solution
PHAITA explores adversarial training for triage. A complaint generator creates human-like narratives for specific ICD-10 respiratory conditions while a diagnosis discriminator attempts to classify the condition and detect synthetic text. Iterating between both systems exposes weaknesses and drives robustness without ever touching real patient data.

## Target Experience
The end-user experience is an Akinator-style triage conversation. Patients describe their concerns, the system probes with clarifying questions, and once enough evidence is gathered it presents ten differential diagnoses with probability estimates, highlights red-flag symptoms, and advises whether to escalate to emergency services or follow up with a family physician.

## System Overview
- **Generator stack**: Bayesian symptom sampler, optional Mistral 7B complaint generation, template fallback for offline use.
- **Discriminator stack**: DeBERTa encoder fused with a symptom knowledge graph via Graph Neural Network layers.
- **Curriculum training**: Progresses from purely synthetic complaints to forum-style phrasing and includes lexical/semantic diversity objectives.
- **Questioning tools**: Template/LLM prompts that gather clarifying information during demos.

## Workflow
1. Sample a target condition and symptoms from ICD-10 priors.
2. Produce a complaint and optional follow-up questions.
3. Diagnose via the discriminator while logging realism, diversity, and failure cases.
4. Update generator and discriminator alternately using adversarial loss terms.

## Metrics & Evaluation
- Diagnosis accuracy across standard, rare, and forum-style cases.
- Diversity scores (lexical + embedding-based) for generated complaints.
- Realism scoring via transformer-based fluency checks.
- Challenge mode benchmarks that surface common failure patterns.

## Applications
- Stress-testing medical NLP pipelines.
- Generating synthetic datasets for triage research or education.
- Prototyping pre-hospital decision support workflows.

## Roadmap Highlights
- Extend coverage beyond respiratory conditions.
- Integrate richer modalities (vitals, timelines) and uncertainty estimates.
- Explore human-in-the-loop active learning and deployment guardrails.

## Disclaimer
PHAITA remains a research prototype and is unsuitable for clinical decision making without rigorous validation and regulatory clearance.
