"""
Comprehensive test suite runner for all tests.
Run this to execute all tests in the package.
"""

import unittest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import all test modules using absolute imports
try:
    from jackofalltrades.test import test_regression
    from jackofalltrades.test import test_classification
    from jackofalltrades.test import test_errors
    from jackofalltrades.test import test_gan
    from jackofalltrades.test import test_neighbors
    from jackofalltrades.test import test_linear_model
except ImportError:
    # Fallback for direct execution
    import test_regression
    import test_classification
    import test_errors
    import test_gan
    import test_neighbors
    import test_linear_model


def create_test_suite():
    """Create and return a test suite with all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test cases
    test_modules = [
        "jackofalltrades.test.test_regression",
        "jackofalltrades.test.test_classification",
        "jackofalltrades.test.test_errors",
        "jackofalltrades.test.test_gan",
        "jackofalltrades.test.test_neighbors",
        "jackofalltrades.test.test_linear_model",
    ]

    for module_name in test_modules:
        try:
            tests = loader.loadTestsFromName(module_name)
            suite.addTests(tests)
        except Exception as e:
            print(f"Warning: Could not load tests from {module_name}: {e}")

    return suite


def run_tests(verbosity=2):
    """Run all tests with specified verbosity."""
    suite = create_test_suite()
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    return result


if __name__ == "__main__":
    # Run all tests
    result = run_tests(verbosity=2)

    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)
