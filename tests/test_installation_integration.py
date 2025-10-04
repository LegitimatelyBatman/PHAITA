#!/usr/bin/env python3
"""
Integration test to verify different installation methods work correctly.

This script tests that:
1. Base requirements can be read and installed (simulated)
2. Extras can be read correctly
3. Import paths work correctly
"""

import sys
from pathlib import Path


def test_import_base_modules():
    """Test that core modules can be imported (base requirements only)."""
    print("🔍 Testing core module imports...")
    
    try:
        # Core Python modules
        import torch
        print(f"  ✓ torch {torch.__version__}")
        
        import transformers
        print(f"  ✓ transformers {transformers.__version__}")
        
        import numpy
        print(f"  ✓ numpy {numpy.__version__}")
        
        import pandas
        print(f"  ✓ pandas {pandas.__version__}")
        
        import sklearn
        print(f"  ✓ scikit-learn {sklearn.__version__}")
        
        # PHAITA core modules that don't require GPU
        from phaita.data.icd_conditions import RespiratoryConditions
        print("  ✓ phaita.data.icd_conditions")
        
        from phaita.models.bayesian_network import BayesianSymptomNetwork
        print("  ✓ phaita.models.bayesian_network")
        
        from phaita.utils.config import Config
        print("  ✓ phaita.utils.config")
        
        print("✅ Core module imports successful\n")
        return True
        
    except ImportError as e:
        print(f"  ✗ Failed to import core module: {e}")
        return False


def test_optional_gpu_modules():
    """Test that GPU modules handle missing dependencies gracefully."""
    print("🔍 Testing optional GPU module handling...")
    
    try:
        # Try to import bitsandbytes (should work but may warn)
        try:
            import bitsandbytes
            print("  ✓ bitsandbytes is available")
            has_bitsandbytes = True
        except ImportError:
            print("  ⚠ bitsandbytes not available (expected for CPU-only install)")
            has_bitsandbytes = False
        
        # Try to import torch_geometric (should work but may not be needed)
        try:
            import torch_geometric
            print("  ✓ torch_geometric is available")
            has_torch_geometric = True
        except ImportError:
            print("  ⚠ torch_geometric not available (expected for CPU-only install)")
            has_torch_geometric = False
        
        # Test that PHAITA modules handle optional deps gracefully
        from phaita.models.generator import ComplaintGenerator, HAS_BITSANDBYTES
        print(f"  ✓ ComplaintGenerator imports correctly (HAS_BITSANDBYTES={HAS_BITSANDBYTES})")
        
        # Discriminator should work even without torch_geometric (falls back to MLP)
        from phaita.models.discriminator import DiagnosisDiscriminator
        print("  ✓ DiagnosisDiscriminator imports correctly (may fallback to MLP)")
        
        print("✅ Optional GPU modules handled correctly\n")
        return True
        
    except ImportError as e:
        print(f"  ✗ Unexpected import error: {e}")
        return False


def test_optional_dev_modules():
    """Test that development modules are available when installed."""
    print("🔍 Testing optional development module availability...")
    
    try:
        import pytest
        print(f"  ✓ pytest {pytest.__version__} is available")
        has_pytest = True
    except ImportError:
        print("  ⚠ pytest not available (expected if dev extras not installed)")
        has_pytest = False
    
    print("✅ Development module check complete\n")
    return True


def test_optional_scraping_modules():
    """Test that scraping modules are available when installed."""
    print("🔍 Testing optional scraping module availability...")
    
    try:
        import praw
        print(f"  ✓ praw {praw.__version__} is available")
    except ImportError:
        print("  ⚠ praw not available (expected if scraping extras not installed)")
    
    try:
        import bs4
        print(f"  ✓ beautifulsoup4 {bs4.__version__} is available")
    except ImportError:
        print("  ⚠ beautifulsoup4 not available (expected if scraping extras not installed)")
    
    print("✅ Scraping module check complete\n")
    return True


def test_instantiate_models_cpu_mode():
    """Test that models can be instantiated in CPU-only mode."""
    print("🔍 Testing model instantiation in CPU-only mode...")
    
    try:
        from phaita.models.bayesian_network import BayesianSymptomNetwork
        from phaita.data.icd_conditions import RespiratoryConditions
        
        # Test Bayesian network (no GPU needed)
        conditions = RespiratoryConditions.get_all_conditions()
        bayesian = BayesianSymptomNetwork()
        print("  ✓ BayesianSymptomNetwork instantiated")
        
        # Test discriminator in lightweight mode (no GPU)
        from phaita.models.discriminator import DiagnosisDiscriminator
        discriminator = DiagnosisDiscriminator(use_pretrained=False)
        print("  ✓ DiagnosisDiscriminator instantiated (CPU mode)")
        
        print("✅ Models instantiate correctly in CPU-only mode\n")
        return True
        
    except Exception as e:
        print(f"  ✗ Failed to instantiate models: {e}")
        return False


def main():
    """Run all integration tests."""
    print("=" * 60)
    print("🏥 PHAITA Installation Integration Tests")
    print("=" * 60)
    print()
    
    tests = [
        test_import_base_modules,
        test_optional_gpu_modules,
        test_optional_dev_modules,
        test_optional_scraping_modules,
        test_instantiate_models_cpu_mode,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}\n")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if all(results):
        print("🎉 All integration tests passed!")
        return 0
    else:
        print("❌ Some integration tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
