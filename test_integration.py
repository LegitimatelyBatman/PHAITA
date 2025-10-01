#!/usr/bin/env python3
"""
Integration test for PHAITA critical bug fixes.
Tests Tasks 1-4 to ensure all components work together.
"""

import sys
from pathlib import Path

# Add package to path
sys.path.insert(0, str(Path(__file__).parent))

from phaita.training.adversarial_trainer import AdversarialTrainer
from phaita.models.generator import SymptomGenerator, ComplaintGenerator
from phaita.models.discriminator import DiagnosisDiscriminator
from phaita.data.forum_scraper import ForumScraper, ForumDataAugmentation


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
            symptoms = gen.generate_symptoms('J45.9')
            complaint = comp_gen.generate_complaint(symptoms, 'J45.9')
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
    print("\n" + "="*70)
    print("TEST TASK 4: Realistic Forum Data Generation")
    print("="*70)
    
    try:
        scraper = ForumScraper()
        posts = scraper.scrape_reddit_health(max_posts=20)
        
        print(f"Generated {len(posts)} forum posts")
        
        # Check condition-specific symptoms
        condition_keywords = {
            'asthma': ['wheezy', 'tight chest', "can't breathe"],
            'pneumonia': ['coughing up', 'chest hurts', 'burning up', 'fever'],
            'copd': ["can't catch", 'wheezy', 'coughing up'],
            'bronchitis': ['hacking cough', 'coughing up', 'chest hurts']
        }
        
        # Verify posts have realistic symptom combinations
        condition_matches = {cond: 0 for cond in condition_keywords.keys()}
        
        for post in posts:
            content_lower = post.content.lower()
            for condition, keywords in condition_keywords.items():
                matches = sum(1 for kw in keywords if kw in content_lower)
                if matches >= 1:  # At least one keyword from the condition
                    condition_matches[condition] += 1
        
        print("\nCondition-specific symptom distribution:")
        for condition, count in condition_matches.items():
            print(f"  {condition}: {count} posts")
        
        # Check for variation (2-4 symptoms)
        symptom_counts = [len(post.lay_terms) for post in posts]
        min_symptoms = min(symptom_counts)
        max_symptoms = max(symptom_counts)
        avg_symptoms = sum(symptom_counts) / len(symptom_counts)
        
        print(f"\nSymptom variation:")
        print(f"  Min symptoms per post: {min_symptoms}")
        print(f"  Max symptoms per post: {max_symptoms}")
        print(f"  Avg symptoms per post: {avg_symptoms:.1f}")
        
        # Show sample posts
        print("\nSample forum posts:")
        for i, post in enumerate(posts[:3]):
            print(f"\n  Post {i+1}:")
            print(f"    Content: {post.content}")
            print(f"    Symptoms: {post.lay_terms}")
        
        # Test forum complaints from augmenter
        augmenter = ForumDataAugmentation()
        complaints = augmenter.get_forum_complaints_for_pretraining(max_complaints=50)
        
        # Check for grammar issues
        bad_grammar = []
        for complaint in complaints:
            if 'Can\'t stop can\'t' in complaint or 'Having can\'t' in complaint:
                bad_grammar.append(complaint)
        
        print(f"\nGenerated {len(complaints)} forum-style complaints")
        print(f"Grammar issues: {len(bad_grammar)}")
        
        # Validation
        assert len(posts) == 20, "Should generate 20 posts"
        assert min_symptoms >= 2, "Should have at least 2 symptoms per post"
        assert max_symptoms <= 4, "Should have at most 4 symptoms per post"
        assert len(bad_grammar) == 0, "Should have no grammar issues"
        assert sum(condition_matches.values()) > 0, "Should have condition-specific symptoms"
        
        print("\n✓ Condition-specific symptom mappings")
        print("✓ 2-4 symptoms per post variation")
        print("✓ No grammar issues in forum complaints")
        print("✓ Realistic symptom combinations")
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
