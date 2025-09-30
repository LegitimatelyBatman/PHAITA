#!/usr/bin/env python3
"""
Test forum scraping and lay language mapping functionality.
"""

import sys
from pathlib import Path
import tempfile
import shutil

# Add the package to Python path
sys.path.insert(0, str(Path(__file__).parent))

def test_forum_scraper():
    """Test forum scraping functionality."""
    print("🌐 Testing Forum Scraper...")
    
    try:
        from phaita.data.forum_scraper import create_forum_scraper
        
        # Create temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            scraper = create_forum_scraper(cache_dir=temp_dir)
            
            # Test mock Reddit scraping
            reddit_posts = scraper.scrape_reddit_health(max_posts=5)
            assert len(reddit_posts) > 0, "Should get Reddit posts"
            
            # Test saving/loading
            scraper.save_posts(reddit_posts, "test_posts.json")
            loaded_posts = scraper.load_posts("test_posts.json")
            assert len(loaded_posts) == len(reddit_posts), "Should load same number of posts"
            
            print(f"✅ Forum scraper tests passed - scraped {len(reddit_posts)} posts")
            return True
            
    except Exception as e:
        print(f"❌ Forum scraper test failed: {e}")
        return False

def test_lay_language_mapper():
    """Test lay language mapping functionality."""
    print("🗺️ Testing Lay Language Mapper...")
    
    try:
        from phaita.data.forum_scraper import create_lay_language_mapper, ForumPost
        
        mapper = create_lay_language_mapper()
        
        # Test predefined mappings
        medical_term = mapper.get_medical_term("can't breathe")
        assert medical_term == "dyspnea", f"Expected 'dyspnea', got {medical_term}"
        
        # Test reverse lookup
        lay_terms = mapper.get_lay_terms_for_medical("dyspnea")
        assert "can't breathe" in lay_terms, "Should find 'can't breathe' for dyspnea"
        
        # Test updating from posts
        mock_posts = [
            ForumPost(
                id="test1",
                title="Breathing issues", 
                content="I can't catch my breath when walking",
                timestamp="2024-01-01",
                forum_source="test",
                lay_terms=["can't catch my breath"],
                extracted_symptoms=["dyspnea"],
                confidence_score=0.8
            )
        ]
        
        mapper.update_mappings_from_posts(mock_posts)
        
        # Test saving/loading
        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = f"{temp_dir}/test_mappings.json"
            mapper.save_mappings(filepath)
            
            new_mapper = create_lay_language_mapper()
            new_mapper.load_mappings(filepath)
            
            assert new_mapper.get_medical_term("can't breathe") == "dyspnea"
        
        print(f"✅ Lay language mapper tests passed - {len(mapper.mappings)} mappings")
        return True
        
    except Exception as e:
        print(f"❌ Lay language mapper test failed: {e}")
        return False

def test_data_augmentation():
    """Test data augmentation functionality."""
    print("🔄 Testing Data Augmentation...")
    
    try:
        from phaita.data.forum_scraper import create_data_augmentation
        
        augmenter = create_data_augmentation()
        
        # Test complaint augmentation
        complaints = ["Patient has dyspnea and chest pain", "Cough with fever present"]
        condition_codes = ["J45.9", "J18.9"]
        
        augmented = augmenter.augment_complaints_with_lay_terms(complaints, condition_codes)
        assert len(augmented) == len(complaints), "Should return same number of complaints"
        
        # Test forum complaints retrieval
        forum_complaints = augmenter.get_forum_complaints_for_pretraining(max_complaints=10)
        assert len(forum_complaints) > 0, "Should get forum complaints"
        
        print(f"✅ Data augmentation tests passed - {len(forum_complaints)} forum complaints")
        return True
        
    except Exception as e:
        print(f"❌ Data augmentation test failed: {e}")
        return False

def main():
    """Run forum scraping tests."""
    print("🌐 Forum Scraping & Lay Language Tests")
    print("=" * 50)
    
    tests = [
        test_forum_scraper,
        test_lay_language_mapper,
        test_data_augmentation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All forum scraping tests passed!")
        return 0
    else:
        print("❌ Some forum scraping tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())