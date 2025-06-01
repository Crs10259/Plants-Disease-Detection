#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Plant Disease Data Collector Tool - Main Entry Point
This script serves as a simple wrapper to start the data collector application.
"""

import sys
import os

# Add this directory to the path to allow imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Try importing the app module
try:
    from app import main
    
    if __name__ == "__main__":
        sys.exit(main())
except ImportError as e:
    print(f"Error importing app module: {str(e)}")
    print("\nPlease make sure all dependencies are installed:")
    print("pip install -r requirements.txt")
    sys.exit(1)
except KeyboardInterrupt:
    print("\nApplication terminated by user.")
    sys.exit(0)
except Exception as e:
    print(f"Unexpected error: {str(e)}")
    import traceback
    print(traceback.format_exc())
    sys.exit(1)