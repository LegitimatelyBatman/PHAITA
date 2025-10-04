# PHAITA Standard Operating Procedure (SOP)

**Complete Guide to Training, Implementing, and Running PHAITA**

## Table of Contents
1. [System Requirements](#system-requirements)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Training Procedures](#training-procedures)
5. [Implementation Guide](#implementation-guide)
6. [Running the System](#running-the-system)
7. [Testing and Validation](#testing-and-validation)
8. [Troubleshooting](#troubleshooting)
9. [Deployment](#deployment)
10. [Maintenance](#maintenance)

---

## 1. System Requirements

### Minimum Requirements (CPU-Only Mode)
- **OS:** Linux, macOS, or Windows 10+
- **Python:** 3.10 or higher
- **RAM:** 8GB
- **Disk:** 2GB free space
- **Network:** Internet for initial setup (optional after)

### Recommended Requirements (GPU Mode)
- **OS:** Linux (Ubuntu 20.04+) or Windows 10+ with WSL2
- **Python:** 3.10 or higher
- **RAM:** 16GB+
- **GPU:** NVIDIA GPU with 4GB+ VRAM
- **CUDA:** 11.8 or higher
- **Disk:** 20GB free space (for models)
- **Network:** Internet for model downloads

### Check Your System
```bash
# Check Python version
python --version  # Should be 3.10+

# Check CUDA availability (if using GPU)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Check available RAM
free -h  # Linux
# or
vm_stat | grep "Pages free"  # macOS

# Check disk space
df -h
```

---

## 2. Installation

### Step 1: Clone Repository
```bash
git clone https://github.com/LegitimatelyBatman/PHAITA.git
cd PHAITA
```

### Step 2: Install Dependencies
```bash
# Install core dependencies
pip install -r requirements.txt

# Optional: Install in editable mode for development
pip install -e .
```

**Expected Installation Time:** 2-5 minutes

### Step 3: Verify Installation
```bash
# Run basic tests
python tests/test_basic.py

# Expected output: "🎉 All tests passed!"
```

### Step 4: Download Models (Optional, for GPU mode)
```bash
# Models will auto-download on first use
# Or pre-download manually:
python -c "from phaita.utils.model_loader import load_model_and_tokenizer; \
           load_model_and_tokenizer('microsoft/deberta-v3-base', 'auto')"
```

---

## 3. Configuration

### Main Configuration File: `config.yaml`

```yaml
# Model Configuration
models:
  generator:
    model_name: "mistralai/Mistral-7B-Instruct-v0.2"
    use_pretrained: true  # Set to false for CPU-only
    quantization: "4bit"  # Options: "4bit", "8bit", null
    temperature: 0.7
    max_length: 256
  
  discriminator:
    model_name: "microsoft/deberta-v3-base"
    use_pretrained: true
    hidden_dim: 768
    num_heads: 8
    dropout: 0.1
  
  bayesian:
    use_enhanced: true
    enable_age_modulation: true
    enable_severity_modulation: true
    enable_comorbidity_effects: true

# Training Configuration
training:
  batch_size: 16
  learning_rate: 1e-4
  num_epochs: 50
  gradient_clip: 1.0
  save_interval: 5

# Conversation Configuration
conversation:
  max_questions: 10
  confidence_threshold: 0.85
  min_info_gain: 0.1
  enable_red_flag_escalation: true

# Data Configuration
data:
  num_conditions: 10
  synthetic_samples: 1000
  forum_samples: 500
```

### Medical Knowledge Configuration

Located in `config/` directory:

#### `config/respiratory_conditions.yaml`
```yaml
conditions:
  J45.9:
    name: "Asthma, unspecified"
    symptoms:
      shortness_of_breath: 0.9
      wheezing: 0.8
      chest_tightness: 0.7
      cough: 0.6
```

#### `config/red_flags.yaml`
```yaml
red_flags:
  severe_respiratory_distress:
    severity: critical
    action: "Call 911 immediately"
    guidance: "Life-threatening breathing difficulty"
```

#### `config/comorbidity_effects.yaml`
```yaml
comorbidities:
  diabetes:
    affects:
      J18.9:  # Pneumonia
        symptom_multiplier: 1.3
        severity_increase: true
```

---

## 4. Training Procedures

### 4.1 Prepare Training Data

#### Option A: Use Synthetic Data
```bash
# Generate synthetic training data
python cli.py generate \
  --count 1000 \
  --output data/training_synthetic.json \
  --diverse
```

#### Option B: Scrape Forum Data (Advanced)
```bash
# Scrape Reddit medical forums
python scripts/scrape_forums.py \
  --subreddits AskDocs medical \
  --limit 500 \
  --output data/training_forum.json
```

#### Option C: Use Mixed Data (Recommended)
```bash
# Generate synthetic
python cli.py generate --count 800 --output data/synthetic.json

# Combine with forum data
python -c "
from phaita.data.synthetic_generator import combine_datasets
combine_datasets(['data/synthetic.json', 'data/forum.json'], 'data/training.json')
"
```

### 4.2 Train Models

#### Full Training (GPU Required)
```bash
# Train adversarial loop
python cli.py train \
  --epochs 50 \
  --batch-size 16 \
  --learning-rate 1e-4 \
  --data data/training.json \
  --output models/checkpoint \
  --device cuda
```

**Expected Training Time:** 2-4 hours on modern GPU

#### CPU-Only Training (Template Mode)
```bash
# Train template generator only
python cli.py train \
  --mode template \
  --epochs 20 \
  --data data/training.json \
  --output models/templates
```

**Expected Training Time:** 30-60 minutes on modern CPU

### 4.3 Monitor Training

```bash
# Training will output:
Epoch 1/50: Loss 2.134, Acc 0.456
Epoch 2/50: Loss 1.987, Acc 0.512
...
Epoch 50/50: Loss 0.432, Acc 0.874

# Checkpoints saved to models/checkpoint/
```

### 4.4 Evaluate Models

```bash
# Run evaluation on test set
python cli.py evaluate \
  --model models/checkpoint/epoch_50.pt \
  --test-data data/test.json

# Expected output:
# Accuracy: 87.3%
# Precision: 0.89
# Recall: 0.85
# F1 Score: 0.87
```

---

## 5. Implementation Guide

### 5.1 Integrate Conversation Engine

```python
from phaita.conversation.engine import ConversationEngine
from phaita.triage.diagnosis_orchestrator import DiagnosisOrchestrator

# Initialize
conversation = ConversationEngine()
orchestrator = DiagnosisOrchestrator()

# Start conversation
session = conversation.start_conversation(
    initial_complaint="I can't breathe properly"
)

# Conversation loop
while not conversation.should_stop(session):
    # Get next question
    question = conversation.get_next_question(session)
    
    # Display to user
    print(f"Q: {question['text']}")
    
    # Get user response
    response = input("Answer (yes/no): ").lower() == "yes"
    
    # Process
    conversation.process_response(session, question, response)
    
    # Check for red flags
    if session.get('red_flags'):
        print("⚠️ Emergency symptoms detected!")
        trigger_emergency_protocol(session)
        break

# Generate diagnosis
diagnosis_slate = orchestrator.generate_diagnosis_slate(
    beliefs=session['beliefs'],
    symptoms=list(session['patient_responses'].keys())
)

# Display results
for i, dx in enumerate(diagnosis_slate[:3], 1):
    print(f"{i}. {dx['name']} ({dx['confidence']:.1%})")
```

### 5.2 Implement Web API

```python
from flask import Flask, request, jsonify
from phaita.conversation.engine import ConversationEngine

app = Flask(__name__)
conversation = ConversationEngine()
sessions = {}  # In production, use Redis or database

@app.route('/api/triage/start', methods=['POST'])
def start_triage():
    """Start new triage session."""
    data = request.json
    complaint = data.get('complaint')
    
    session = conversation.start_conversation(complaint)
    session_id = session['session_id']
    sessions[session_id] = session
    
    return jsonify({
        'session_id': session_id,
        'question': conversation.get_next_question(session)
    })

@app.route('/api/triage/answer', methods=['POST'])
def process_answer():
    """Process answer and get next question."""
    data = request.json
    session_id = data.get('session_id')
    answer = data.get('answer')
    
    session = sessions[session_id]
    question = session['current_question']
    
    conversation.process_response(session, question, answer)
    
    if conversation.should_stop(session):
        # Generate diagnosis
        from phaita.triage.diagnosis_orchestrator import DiagnosisOrchestrator
        orchestrator = DiagnosisOrchestrator()
        
        diagnosis = orchestrator.generate_diagnosis_slate(
            beliefs=session['beliefs'],
            symptoms=list(session['patient_responses'].keys())
        )
        
        return jsonify({
            'status': 'complete',
            'diagnosis': diagnosis
        })
    else:
        next_question = conversation.get_next_question(session)
        return jsonify({
            'status': 'continue',
            'question': next_question
        })

if __name__ == '__main__':
    app.run(port=5000)
```

### 5.3 Build CLI Application

See `cli.py` for complete implementation. Key commands:

```bash
# Interactive demo
python cli.py demo --num-examples 5

# Generate synthetic data
python cli.py generate --count 100 --output data.json

# Diagnose complaint
python cli.py diagnose --complaint "I can't breathe"

# Train models
python cli.py train --epochs 50 --batch-size 16
```

---

## 6. Running the System

### 6.1 Quick Start (Demo Mode)

```bash
# Run interactive demo (no dependencies)
python demos/simple_demo.py

# Run dialogue demo
python demos/demo_dialogue_engine.py

# Run full system demo
python demos/demo_deep_learning.py
```

### 6.2 CLI Usage

```bash
# Interactive triage session
python cli.py diagnose --interactive

# Process batch of complaints
python cli.py batch-diagnose --input complaints.json --output diagnoses.json

# Challenge mode (test medical knowledge)
python cli.py challenge
```

### 6.3 Web Interface

```bash
# Start web server
python patient_cli.py --port 8080

# Access at http://localhost:8080
```

### 6.4 API Usage

```python
import requests

# Start session
response = requests.post('http://localhost:5000/api/triage/start', json={
    'complaint': "I have chest pain and shortness of breath"
})
session_id = response.json()['session_id']
question = response.json()['question']

# Answer questions
while True:
    answer = input(f"{question['text']} (yes/no): ")
    
    response = requests.post('http://localhost:5000/api/triage/answer', json={
        'session_id': session_id,
        'answer': answer == 'yes'
    })
    
    data = response.json()
    
    if data['status'] == 'complete':
        print("Diagnosis:", data['diagnosis'])
        break
    else:
        question = data['question']
```

---

## 7. Testing and Validation

### 7.1 Run All Tests

```bash
# Core tests (required, ~30 seconds)
python tests/test_basic.py
python tests/test_enhanced_bayesian.py
python tests/test_forum_scraping.py

# Dialogue tests
python tests/test_dialogue_engine.py
python tests/test_conversation_flow.py

# Diagnosis tests
python tests/test_diagnosis_orchestrator.py
python tests/test_escalation_guidance.py

# Model tests
pytest tests/test_model_loader.py
python tests/test_gnn_performance.py

# Integration tests (slow, requires network)
python tests/test_integration.py
```

### 7.2 Test Specific Components

```bash
# Test data layer only
python tests/test_basic.py

# Test Bayesian network
python tests/test_enhanced_bayesian.py

# Test conversation engine
python tests/test_dialogue_engine.py
```

### 7.3 Validation Checklist

- [ ] All core tests pass (test_basic.py, test_enhanced_bayesian.py)
- [ ] Dialogue engine tests pass
- [ ] Diagnosis orchestrator tests pass
- [ ] Red-flag detection works correctly
- [ ] Symptom normalization consistent
- [ ] Models load without errors
- [ ] API endpoints respond correctly
- [ ] CLI commands work
- [ ] Demo scripts run successfully

---

## 8. Troubleshooting

### 8.1 Installation Issues

#### Issue: "ModuleNotFoundError: No module named 'torch'"
**Solution:**
```bash
pip install torch
# Or for specific CUDA version:
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

#### Issue: "CUDA out of memory"
**Solution:**
```yaml
# In config.yaml, use quantization:
models:
  generator:
    quantization: "4bit"  # Reduces memory usage
  discriminator:
    use_pretrained: false  # Use lite version
```

### 8.2 Runtime Issues

#### Issue: "Cannot connect to huggingface.co"
**Solution:**
This is expected and handled gracefully. System will:
1. Retry 3 times with exponential backoff
2. Fall back to template mode (CPU-friendly)

No action needed - this is normal operation.

#### Issue: "Red-flag not detected for 'severe respiratory distress'"
**Solution:**
```python
# Symptom normalization issue - ensure consistent format
# Use underscores OR spaces, not mixed
symptom = "severe_respiratory_distress"  # Correct
# OR
symptom = "severe respiratory distress"  # Also correct

# The system normalizes automatically, but input should be consistent
```

### 8.3 Training Issues

#### Issue: "Training loss not decreasing"
**Solutions:**
1. **Lower learning rate:**
   ```bash
   python cli.py train --learning-rate 1e-5
   ```
2. **Increase batch size:**
   ```bash
   python cli.py train --batch-size 32
   ```
3. **Check data quality:**
   ```bash
   python -c "from phaita.data.synthetic_generator import validate_data; \
              validate_data('data/training.json')"
   ```

#### Issue: "Out of memory during training"
**Solutions:**
1. **Reduce batch size:**
   ```bash
   python cli.py train --batch-size 8
   ```
2. **Use gradient accumulation:**
   ```bash
   python cli.py train --accumulation-steps 4
   ```
3. **Enable mixed precision:**
   ```bash
   python cli.py train --mixed-precision
   ```

### 8.4 Testing Issues

#### Issue: "Test fails with import error"
**Solution:**
```bash
# Make sure you're running from repository root
cd /path/to/PHAITA
python tests/test_basic.py
```

#### Issue: "Integration test times out"
**Solution:**
```bash
# Integration test downloads models - increase timeout
python tests/test_integration.py  # Allows ~2 minutes for downloads
```

---

## 9. Deployment

### 9.1 Production Deployment Checklist

- [ ] **Security:**
  - [ ] Enable HTTPS
  - [ ] Set up authentication
  - [ ] Sanitize user inputs
  - [ ] Rate limiting configured
  - [ ] HIPAA compliance review (if applicable)

- [ ] **Performance:**
  - [ ] Models loaded and cached
  - [ ] Database configured (for sessions)
  - [ ] Redis/memcached for caching
  - [ ] Load balancer configured
  - [ ] Monitoring enabled

- [ ] **Reliability:**
  - [ ] Error tracking (Sentry, etc.)
  - [ ] Logging configured
  - [ ] Backup strategy in place
  - [ ] Failover tested
  - [ ] Health checks enabled

- [ ] **Compliance:**
  - [ ] Medical disclaimer displayed
  - [ ] Terms of service published
  - [ ] Privacy policy published
  - [ ] Data retention policy defined
  - [ ] User consent obtained

### 9.2 Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Install package
RUN pip install -e .

# Expose port
EXPOSE 5000

# Run application
CMD ["python", "patient_cli.py", "--port", "5000", "--host", "0.0.0.0"]
```

```bash
# Build and run
docker build -t phaita:latest .
docker run -p 5000:5000 phaita:latest
```

### 9.3 Cloud Deployment

#### AWS Lambda
```bash
# Package for Lambda
pip install -t package/ -r requirements.txt
cd package && zip -r ../deployment.zip .
cd .. && zip -g deployment.zip lambda_handler.py
aws lambda update-function-code --function-name phaita --zip-file fileb://deployment.zip
```

#### Google Cloud Run
```bash
# Deploy to Cloud Run
gcloud run deploy phaita \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

#### Azure App Service
```bash
# Deploy to Azure
az webapp up \
  --name phaita-app \
  --resource-group phaita-rg \
  --runtime "PYTHON:3.10"
```

---

## 10. Maintenance

### 10.1 Regular Maintenance Tasks

#### Daily
- [ ] Monitor error logs
- [ ] Check system health metrics
- [ ] Review red-flag alerts

#### Weekly
- [ ] Review user feedback
- [ ] Check model performance metrics
- [ ] Update medical knowledge base if needed

#### Monthly
- [ ] Retrain models with new data
- [ ] Update dependencies
- [ ] Security audit
- [ ] Backup verification

#### Quarterly
- [ ] Major version updates
- [ ] Comprehensive testing
- [ ] Performance benchmarking
- [ ] Clinical review of red-flags

### 10.2 Updating Medical Knowledge

```bash
# Edit condition definitions
nano config/respiratory_conditions.yaml

# Edit red-flags
nano config/red_flags.yaml

# Edit comorbidity effects
nano config/comorbidity_effects.yaml

# Validate changes
python -c "from phaita.utils.config import Config; Config().validate()"

# Retrain models
python cli.py train --config config.yaml
```

### 10.3 Model Retraining

```bash
# Collect new data
python scripts/scrape_forums.py --output data/new_forum.json

# Combine with existing
python -c "
from phaita.data.synthetic_generator import combine_datasets
combine_datasets(['data/training.json', 'data/new_forum.json'], 'data/updated.json')
"

# Retrain
python cli.py train \
  --data data/updated.json \
  --output models/retrained \
  --epochs 50

# Evaluate
python cli.py evaluate \
  --model models/retrained/epoch_50.pt \
  --test-data data/test.json

# Deploy if improved
cp models/retrained/epoch_50.pt models/production/model.pt
```

### 10.4 Monitoring

```python
# monitoring.py
import logging
from phaita.conversation.engine import ConversationEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/phaita.log'),
        logging.StreamHandler()
    ]
)

# Track key metrics
def log_conversation_metrics(session):
    """Log conversation statistics."""
    logging.info(f"Session {session['session_id']}")
    logging.info(f"  Questions asked: {session['question_count']}")
    logging.info(f"  Final confidence: {max(session['beliefs'].values()):.2%}")
    logging.info(f"  Red flags: {len(session.get('red_flags', []))}")
    
    # Track to monitoring system (Prometheus, Datadog, etc.)
    metrics.increment('conversations.total')
    metrics.gauge('conversations.questions', session['question_count'])
    metrics.gauge('conversations.confidence', max(session['beliefs'].values()))
```

---

## Quick Reference

### Essential Commands
```bash
# Setup
pip install -r requirements.txt

# Test
python tests/test_basic.py

# Demo
python demos/simple_demo.py

# Train
python cli.py train --epochs 50

# Run
python cli.py diagnose --interactive

# Deploy
python patient_cli.py --port 8080
```

### Key Files
- `config.yaml` - Main configuration
- `config/respiratory_conditions.yaml` - Medical knowledge
- `config/red_flags.yaml` - Emergency criteria
- `requirements.txt` - Dependencies
- `cli.py` - Command-line interface
- `patient_cli.py` - Web interface

### Support
- **Documentation:** [docs/DOCUMENTATION_INDEX.md](../DOCUMENTATION_INDEX.md)
- **Issues:** https://github.com/LegitimatelyBatman/PHAITA/issues
- **Testing:** [docs/TESTING.md](../TESTING.md)

---

## Appendix: Common Workflows

### Workflow 1: First-Time Setup
```bash
# 1. Clone and install
git clone https://github.com/LegitimatelyBatman/PHAITA.git
cd PHAITA
pip install -r requirements.txt

# 2. Verify installation
python tests/test_basic.py

# 3. Run demo
python demos/simple_demo.py

# 4. Try interactive triage
python cli.py diagnose --interactive
```

### Workflow 2: Development
```bash
# 1. Install in editable mode
pip install -e .

# 2. Make changes to code
nano phaita/conversation/dialogue_engine.py

# 3. Test changes
python tests/test_dialogue_engine.py

# 4. Run affected tests
python tests/test_conversation_flow.py
```

### Workflow 3: Production Deployment
```bash
# 1. Train production models
python cli.py train --epochs 100 --batch-size 32 --data data/production.json

# 2. Run full test suite
for test in tests/test_*.py; do python $test; done

# 3. Build Docker image
docker build -t phaita:prod .

# 4. Deploy
docker run -d -p 80:5000 phaita:prod

# 5. Monitor
tail -f logs/phaita.log
```

---

**Last Updated:** 2025-01-03  
**Version:** 1.0  
**Maintainer:** PHAITA Development Team
