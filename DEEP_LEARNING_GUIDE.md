# Deep-Learning Quickstart

This guide captures the essentials for running PHAITA with the full deep-learning stack.

## Prerequisites
- Python 3.10+
- GPU with ≥4GB VRAM for quantised Mistral 7B (CPU mode available with templates only).
- Packages from `requirements.txt`; install extras with `pip install torch transformers bitsandbytes` if not already present.

## Enabling the Full Stack
```python
from phaita import AdversarialTrainer
from phaita.models import ComplaintGenerator, DiagnosisDiscriminator

# Generator: enable LLM mode
generator = ComplaintGenerator(use_pretrained=True, use_4bit=True, temperature=0.7, top_p=0.9)

# Discriminator: load DeBERTa weights and symptom graph
discriminator = DiagnosisDiscriminator(use_pretrained=True, freeze_encoder=False)

trainer = AdversarialTrainer(generator=generator, discriminator=discriminator)
trainer.train(num_epochs=5, batch_size=4)
```

### Tips
- Use `freeze_encoder=True` for faster experimentation on small datasets.
- Set `use_pretrained=False` to fall back to lightweight templates or randomly initialised weights for smoke tests.

## Configuration Hooks
- `config.yaml` exposes model names, diversity weights, curriculum schedule, and learning rates.
- Override settings via `Config.from_yaml("custom.yaml")` or CLI flags (`--epochs`, `--batch-size`, etc.).

## Troubleshooting
| Issue | Suggested Fix |
|-------|---------------|
| Out-of-memory when loading Mistral | Use `use_4bit=True`, reduce batch size, or switch to template mode. |
| Missing `torch_geometric` | The GNN module automatically falls back to an MLP-based graph encoder. |
| Slow tokenisation | Cache datasets with `preprocessing.save_dataset` or disable forum mixing during early experiments. |
| Divergent adversarial training | Lower learning rates in `config.yaml` or reduce diversity weight. |

## Monitoring Training
- `trainer.train` returns history dictionaries for losses and metrics.
- Enable verbose logging with `--verbose` on CLI commands.
- `test_*` scripts provide regression checks for the data pipeline and Bayesian sampling.

## Further Reading
- `IMPLEMENTATION_DETAILS.md` for architectural highlights.
- `PROJECT_SUMMARY.md` for motivation and roadmap.
