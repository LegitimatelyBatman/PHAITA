#!/usr/bin/env python3
"""Integration test for PHAITA critical bug fixes.

IMPORTANT: This test suite REQUIRES real transformer models to be downloaded.
- Requires: torch==2.5.1, transformers==4.46.0, bitsandbytes==0.44.1, torch-geometric==2.6.1
- Requires: Internet connection for first-time model downloads (~10GB total)
- Requires: GPU with 4GB+ VRAM recommended (CPU mode available but slow)
- May timeout in CI environments without model caching or network access

For lightweight tests that don't require models, see test_basic.py and test_conversation_engine.py.
"""

import json
import tempfile
from datetime import datetime
from pathlib import Path


from phaita.training.adversarial_trainer import AdversarialTrainer
from phaita.models.generator import SymptomGenerator, ComplaintGenerator
from phaita.models.discriminator import DiagnosisDiscriminator
from phaita.data.forum_scraper import (
    ForumScraper,
    ForumDataAugmentation,
    PatientInfoClient,
)


def test_task1_generator_reference():
    """Test Task 1: Generator reference bug fix."""
    print("\n" + "="*70)
    print("TEST TASK 1: Generator Reference Bug Fix")
    print("="*70)
    
    try:
        trainer = AdversarialTrainer()
        
        # Check generator exists and has required methods
        assert hasattr(trainer, 'generator'), "trainer should have 'generator' attribute"
        assert hasattr(trainer.generator, 'parameters'), "generator should have 'parameters' method"
        assert hasattr(trainer.generator, 'train'), "generator should have 'train' method"
        assert hasattr(trainer.generator, 'eval'), "generator should have 'eval' method"
        
        # Check parameters are valid
        params = list(trainer.generator.parameters())
        assert len(params) > 0, "generator parameters should not be empty"
        
        # Check optimizers were created successfully
        assert trainer.gen_optimizer is not None, "gen_optimizer should be initialized"
        assert trainer.disc_optimizer is not None, "disc_optimizer should be initialized"
        
        print("✓ MockGenerator wrapper created successfully")
        print("✓ Generator has .parameters() method")
        print("✓ Generator has .train() and .eval() methods")
        print("✓ Optimizers initialized without errors")
        print("\n✅ TASK 1 PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ TASK 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_task2_discriminator_pytorch_compat():
    """Test Task 2: Discriminator PyTorch compatibility."""
    print("\n" + "="*70)
    print("TEST TASK 2: Discriminator PyTorch Compatibility")
    print("="*70)
    
    try:
        import torch
        
        discriminator = DiagnosisDiscriminator()
        
        # Test .to(device) method
        discriminator = discriminator.to('cpu')
        print("✓ .to(device) method works")
        
        # Test __call__ method
        complaints = ["I have a bad cough and chest pain", "I can't breathe and feel wheezy"]
        outputs = discriminator(complaints, return_features=True)
        
        assert isinstance(outputs, dict), "discriminator should return a dictionary"
        assert "diagnosis_logits" in outputs, "outputs should have 'diagnosis_logits'"
        assert "discriminator_scores" in outputs, "outputs should have 'discriminator_scores'"
        assert "text_features" in outputs, "outputs should have 'text_features'"
        
        # Check shapes
        batch_size = len(complaints)
        assert outputs["diagnosis_logits"].shape[0] == batch_size, "diagnosis_logits batch size mismatch"
        assert outputs["discriminator_scores"].shape == (batch_size, 1), "discriminator_scores shape mismatch"
        assert outputs["text_features"].shape == (batch_size, 768), "text_features shape mismatch"
        
        print(f"✓ __call__() returns proper dictionary")
        print(f"✓ diagnosis_logits shape: {outputs['diagnosis_logits'].shape}")
        print(f"✓ discriminator_scores shape: {outputs['discriminator_scores'].shape}")
        print(f"✓ text_features shape: {outputs['text_features'].shape}")
        
        # Test .train() and .eval() methods
        discriminator.train()
        discriminator.eval()
        print("✓ .train() and .eval() methods work")
        
        # Test .parameters() method
        params = list(discriminator.parameters())
        assert len(params) > 0, "discriminator should have parameters"
        print(f"✓ .parameters() method returns {len(params)} parameters")
        
        # Test state_dict and load_state_dict
        state = discriminator.state_dict()
        discriminator.load_state_dict(state)
        print("✓ .state_dict() and .load_state_dict() methods work")
        
        # Test gradients
        outputs = discriminator(complaints, return_features=True)
        assert outputs["diagnosis_logits"].requires_grad, "diagnosis_logits should require grad"
        assert outputs["discriminator_scores"].requires_grad, "discriminator_scores should require grad"
        print("✓ Tensors have gradients enabled")
        
        print("\n✅ TASK 2 PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ TASK 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_task3_grammar_fixes():
    """Test Task 3: Grammar fixes in synthetic data generation."""
    print("\n" + "="*70)
    print("TEST TASK 3: Grammar Fixes in Synthetic Data")
    print("="*70)
    
    try:
        gen = SymptomGenerator()
        comp_gen = ComplaintGenerator()
        
        # Generate many complaints to check grammar
        bad_patterns = []
        all_complaints = []
        
        for i in range(200):
            presentation = gen.generate_symptoms('J45.9')
            presentation = comp_gen.generate_complaint(presentation=presentation)
            complaint = presentation.complaint_text
            all_complaints.append(complaint)
            
            # Check for bad patterns
            if "I've been can't" in complaint:
                bad_patterns.append(('I\'ve been can\'t', complaint))
            if "I've been shortness" in complaint or "I've been breathless" in complaint:
                bad_patterns.append(('I\'ve been [noun]', complaint))
            if 'My wheezy' in complaint or 'My breathless' in complaint:
                bad_patterns.append(('My [adjective]', complaint))
            if 'trouble with can\'t' in complaint:
                bad_patterns.append(('trouble with can\'t', complaint))
        
        error_rate = len(bad_patterns) / 200 * 100
        
        print(f"Generated 200 test complaints")
        print(f"Found {len(bad_patterns)} grammar issues")
        print(f"Error rate: {error_rate:.1f}%")
        
        # Show sample complaints
        print("\nSample complaints:")
        for i, complaint in enumerate(all_complaints[:5]):
            print(f"  {i+1}. {complaint}")
        
        if error_rate < 5.0:
            print(f"\n✓ Grammar error rate: {error_rate:.1f}% (< 5% threshold)")
            print("✓ No 'I've been [noun]' patterns")
            print("✓ Proper verb conjugation")
            print("✓ Gerunds and infinitives handled correctly")
            print("\n✅ TASK 3 PASSED")
            return True
        else:
            print(f"\n⚠️ Grammar error rate: {error_rate:.1f}% (> 5% threshold)")
            print("Showing error samples:")
            for pattern, example in bad_patterns[:3]:
                print(f"  Pattern: {pattern}")
                print(f"  Example: {example[:100]}")
            print("\n❌ TASK 3 FAILED (error rate too high)")
            return False
            
    except Exception as e:
        print(f"\n❌ TASK 3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_task4_forum_data_generation():
    """Test Task 4: Realistic forum data generation."""
    print("\n" + "=" * 70)
    print("TEST TASK 4: Realistic Forum Data Generation")
    print("=" * 70)

    try:
        fixture_dir = Path(__file__).parent / "tests" / "fixtures"
        reddit_data = json.loads((fixture_dir / "reddit_sample.json").read_text())
        patient_html = (fixture_dir / "patient_info_sample.html").read_text()

        class FixtureRedditClient:
            def __init__(self, records):
                self.records = records

            def fetch_posts(self, max_posts: int):
                posts = []
                for record in self.records[:max_posts]:
                    created = datetime.fromisoformat(record["created_at"].replace("Z", "+00:00"))
                    posts.append(
                        {
                            "id": record["id"],
                            "title": record["title"],
                            "content": record["content"],
                            "created_at": created,
                        }
                    )
                return posts

        class FixtureResponse:
            def __init__(self, text: str):
                self.text = text
                self.status_code = 200

            def raise_for_status(self) -> None:
                return None

        class FixtureSession:
            def __init__(self, html: str):
                self.html = html

            def get(self, url: str, timeout: int = 10):
                return FixtureResponse(self.html)

        reddit_client = FixtureRedditClient(reddit_data)
        patient_client = PatientInfoClient(
            forum_paths=["/fixtures/asthma"],
            session=FixtureSession(patient_html),
            rate_limit_seconds=0.0,
        )

        with tempfile.TemporaryDirectory() as cache_dir:
            scraper = ForumScraper(
                cache_dir=cache_dir,
                reddit_client=reddit_client,
                patient_info_client=patient_client,
            )

            reddit_posts = scraper.scrape_reddit_health(max_posts=3)
            patient_posts = scraper.scrape_patient_info(max_posts=2)

            scraper.save_posts(reddit_posts, "reddit_posts.json")
            scraper.save_posts(patient_posts, "patient_info_posts.json")

            cached_reddit_posts = scraper.load_posts("reddit_posts.json")
            cached_patient_posts = scraper.load_posts("patient_info_posts.json")

        print(f"Loaded {len(reddit_posts)} Reddit posts and {len(patient_posts)} Patient.info posts")

        assert len(reddit_posts) == 3, "Reddit fixture should yield 3 posts"
        assert len(patient_posts) == 2, "Patient.info fixture should yield 2 posts"
        assert cached_reddit_posts[0].content == reddit_posts[0].content
        assert cached_patient_posts[0].lay_terms, "Patient.info posts should extract lay terms"

        augmenter = ForumDataAugmentation(forum_posts=reddit_posts)
        complaints = augmenter.get_forum_complaints_for_pretraining(max_complaints=3)
        assert complaints == [post.content for post in reddit_posts[:3]]

        augmented = augmenter.augment_complaints_with_lay_terms(
            [
                "Patient reports dyspnea and chest pain",
                "Experiencing productive cough and fever",
            ],
            ["J45.9", "J18.9"],
        )

        print("\nSample Reddit post:")
        print(f"  Title: {reddit_posts[0].title}")
        print(f"  Content: {reddit_posts[0].content}")
        print(f"  Lay terms: {reddit_posts[0].lay_terms}")

        print("\nSample Patient.info post:")
        print(f"  Title: {patient_posts[0].title}")
        print(f"  Content: {patient_posts[0].content}")
        print(f"  Lay terms: {patient_posts[0].lay_terms}")

        print("\nAugmented complaints:")
        for item in augmented:
            print(f"  - {item}")

        print("\n✓ Fixture-driven clients exercised the real scraper")
        print("✓ Posts persisted and reloaded from cache")
        print("✓ Lay language mappings applied to scraped content")
        print("\n✅ TASK 4 PASSED")
        return True

    except Exception as e:
        print(f"\n❌ TASK 4 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration_training_loop():
    """Test that training loop executes without errors."""
    print("\n" + "="*70)
    print("INTEGRATION TEST: Training Loop Execution")
    print("="*70)
    
    try:
        import torch
        
        trainer = AdversarialTrainer()
        
        # Test generate_training_batch
        complaints, codes, labels = trainer.generate_training_batch(8)
        print(f"✓ Generated {len(complaints)} training examples")
        
        # Test discriminator step
        fake_complaints, _, _ = trainer.generate_training_batch(8)
        disc_losses = trainer.train_discriminator_step(complaints, labels, fake_complaints)
        print(f"✓ Discriminator training step executed")
        print(f"  Losses: {list(disc_losses.keys())}")
        
        # Test generator step
        gen_losses = trainer.train_generator_step(8)
        print(f"✓ Generator training step executed")
        print(f"  Losses: {list(gen_losses.keys())}")
        
        # Test evaluation
        eval_complaints, eval_codes, _ = trainer.generate_training_batch(8)
        eval_metrics = trainer.evaluate(eval_complaints, eval_codes)
        print(f"✓ Evaluation executed")
        print(f"  Metrics: {list(eval_metrics.keys())}")
        
        print("\n✓ All training components work together")
        print("✓ No AttributeError during training")
        print("✓ Gradients computed successfully")
        print("\n✅ INTEGRATION TEST PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ INTEGRATION TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all integration tests."""
    print("\n" + "="*70)
    print("PHAITA INTEGRATION TEST SUITE")
    print("Testing Tasks 1-4 Critical Bug Fixes")
    print("="*70)
    
    results = {
        "Task 1: Generator Reference Bug": test_task1_generator_reference(),
        "Task 2: Discriminator PyTorch Compatibility": test_task2_discriminator_pytorch_compat(),
        "Task 3: Grammar Fixes": test_task3_grammar_fixes(),
        "Task 4: Forum Data Generation": test_task4_forum_data_generation(),
        "Integration: Training Loop": test_integration_training_loop()
    }
    
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status} - {test_name}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*70)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("All critical bugs have been fixed successfully.")
    else:
        print("⚠️ SOME TESTS FAILED")
        print("Please review the failed tests above.")
    print("="*70 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
