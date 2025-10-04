#!/usr/bin/env python3
"""
Test suite for validating modular dependency installation.

Tests that:
1. Core/base requirements can be parsed correctly
2. Optional requirements (gpu, dev, scraping) can be parsed correctly
3. setup.py extras_require is configured correctly
4. All requirements files are well-formed

This test does NOT attempt to install packages, only validates the configuration.
"""

import sys
from pathlib import Path


def test_requirements_files_exist():
    """Verify all requirements files exist."""
    print("🔍 Testing requirements files exist...")
    
    repo_root = Path(__file__).parent.parent
    required_files = [
        "requirements.txt",
        "requirements-base.txt",
        "requirements-gpu.txt",
        "requirements-dev.txt",
        "requirements-scraping.txt",
    ]
    
    missing = []
    for filename in required_files:
        filepath = repo_root / filename
        if not filepath.exists():
            missing.append(filename)
        else:
            print(f"  ✓ Found {filename}")
    
    if missing:
        print(f"  ✗ Missing files: {', '.join(missing)}")
        return False
    
    print("✅ All requirements files exist\n")
    return True


def test_requirements_base_content():
    """Verify base requirements contain core dependencies."""
    print("🔍 Testing requirements-base.txt content...")
    
    repo_root = Path(__file__).parent.parent
    base_file = repo_root / "requirements-base.txt"
    
    with open(base_file) as f:
        content = f.read()
        lines = [line.strip() for line in content.splitlines() 
                if line.strip() and not line.strip().startswith("#")]
    
    required_deps = [
        "torch",
        "transformers",
        "datasets",
        "accelerate",
        "numpy",
        "pandas",
        "scikit-learn",
        "networkx",
        "scipy",
        "tqdm",
        "pyyaml",
    ]
    
    # Check that required dependencies are present
    deps_found = {dep: False for dep in required_deps}
    for line in lines:
        for dep in required_deps:
            if line.lower().startswith(dep.lower()):
                deps_found[dep] = True
    
    missing = [dep for dep, found in deps_found.items() if not found]
    
    if missing:
        print(f"  ✗ Missing required dependencies: {', '.join(missing)}")
        return False
    
    # Check that optional deps are NOT in base
    optional_deps = ["bitsandbytes", "torch-geometric", "pytest", "praw"]
    found_optional = []
    for line in lines:
        for dep in optional_deps:
            if line.lower().startswith(dep.lower()):
                found_optional.append(dep)
    
    if found_optional:
        print(f"  ✗ Optional dependencies should not be in base: {', '.join(found_optional)}")
        return False
    
    print(f"  ✓ Contains {len(lines)} core dependencies")
    print("✅ Base requirements are correct\n")
    return True


def test_requirements_gpu_content():
    """Verify GPU requirements contain GPU-specific dependencies."""
    print("🔍 Testing requirements-gpu.txt content...")
    
    repo_root = Path(__file__).parent.parent
    gpu_file = repo_root / "requirements-gpu.txt"
    
    with open(gpu_file) as f:
        content = f.read()
        lines = [line.strip() for line in content.splitlines() 
                if line.strip() and not line.strip().startswith("#")]
    
    required_deps = ["bitsandbytes", "torch-geometric"]
    
    deps_found = {dep: False for dep in required_deps}
    for line in lines:
        for dep in required_deps:
            if line.lower().startswith(dep.lower()):
                deps_found[dep] = True
    
    missing = [dep for dep, found in deps_found.items() if not found]
    
    if missing:
        print(f"  ✗ Missing GPU dependencies: {', '.join(missing)}")
        return False
    
    print(f"  ✓ Contains required GPU dependencies: {', '.join(required_deps)}")
    print("✅ GPU requirements are correct\n")
    return True


def test_requirements_dev_content():
    """Verify dev requirements contain development dependencies."""
    print("🔍 Testing requirements-dev.txt content...")
    
    repo_root = Path(__file__).parent.parent
    dev_file = repo_root / "requirements-dev.txt"
    
    with open(dev_file) as f:
        content = f.read()
        lines = [line.strip() for line in content.splitlines() 
                if line.strip() and not line.strip().startswith("#")]
    
    required_deps = ["pytest"]
    
    deps_found = {dep: False for dep in required_deps}
    for line in lines:
        for dep in required_deps:
            if line.lower().startswith(dep.lower()):
                deps_found[dep] = True
    
    missing = [dep for dep, found in deps_found.items() if not found]
    
    if missing:
        print(f"  ✗ Missing dev dependencies: {', '.join(missing)}")
        return False
    
    print(f"  ✓ Contains required dev dependencies: {', '.join(required_deps)}")
    print("✅ Dev requirements are correct\n")
    return True


def test_requirements_scraping_content():
    """Verify scraping requirements contain scraping dependencies."""
    print("🔍 Testing requirements-scraping.txt content...")
    
    repo_root = Path(__file__).parent.parent
    scraping_file = repo_root / "requirements-scraping.txt"
    
    with open(scraping_file) as f:
        content = f.read()
        lines = [line.strip() for line in content.splitlines() 
                if line.strip() and not line.strip().startswith("#")]
    
    required_deps = ["praw", "beautifulsoup4"]
    
    deps_found = {dep: False for dep in required_deps}
    for line in lines:
        for dep in required_deps:
            if line.lower().startswith(dep.lower()):
                deps_found[dep] = True
    
    missing = [dep for dep, found in deps_found.items() if not found]
    
    if missing:
        print(f"  ✗ Missing scraping dependencies: {', '.join(missing)}")
        return False
    
    print(f"  ✓ Contains required scraping dependencies: {', '.join(required_deps)}")
    print("✅ Scraping requirements are correct\n")
    return True


def test_setup_py_structure():
    """Verify setup.py has proper structure for extras_require."""
    print("🔍 Testing setup.py structure...")
    
    repo_root = Path(__file__).parent.parent
    setup_file = repo_root / "setup.py"
    
    with open(setup_file) as f:
        content = f.read()
    
    # Check for key elements
    required_elements = [
        "extras_require",
        '"gpu"',
        '"dev"',
        '"scraping"',
        '"all"',
        "requirements-base.txt",
        "requirements-gpu.txt",
        "requirements-dev.txt",
        "requirements-scraping.txt",
    ]
    
    missing = []
    for element in required_elements:
        if element not in content:
            missing.append(element)
        else:
            print(f"  ✓ Found {element}")
    
    if missing:
        print(f"  ✗ Missing elements in setup.py: {', '.join(missing)}")
        return False
    
    print("✅ setup.py structure is correct\n")
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("🏥 PHAITA Modular Dependencies Test Suite")
    print("=" * 60)
    print()
    
    tests = [
        test_requirements_files_exist,
        test_requirements_base_content,
        test_requirements_gpu_content,
        test_requirements_dev_content,
        test_requirements_scraping_content,
        test_setup_py_structure,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}\n")
            results.append(False)
    
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if all(results):
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
