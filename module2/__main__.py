"""
Module 2 entry point for -m invocation.
Supports: python -m module2.run_template_matching
"""
import sys
from pathlib import Path

# Get the module name being invoked
if len(sys.argv) > 0 and sys.argv[0].endswith('__main__.py'):
    # Called as: python -m module2.run_template_matching
    # The actual module name is in sys.modules
    pass

# For now, just provide a helpful message
if __name__ == "__main__":
    print("Module 2 - Template Matching")
    print("Usage: python -m module2.run_template_matching --scene <path> [--threshold 0.4]")
