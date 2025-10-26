#!/usr/bin/env python3
"""
Fix script for TensorFlow/DeepFace issues on Windows
Run this script if you encounter DLL loading errors or binary compatibility issues
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a command and return success status"""
    print(f"\n{description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ {description} completed successfully")
            if result.stdout:
                print(f"Output: {result.stdout.strip()}")
            return True
        else:
            print(f"✗ {description} failed")
            if result.stderr:
                print(f"Error: {result.stderr.strip()}")
            return False
    except Exception as e:
        print(f"✗ {description} failed with exception: {e}")
        return False

def main():
    print("=== DeepFace GUI Fix Script ===")
    print("This script will fix common TensorFlow and NumPy/Pandas issues on Windows\n")

    # Check Python version
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")

    # Step 1: Uninstall all problematic packages
    if not run_command("pip uninstall tensorflow tensorflow-gpu tensorflow-cpu pandas numpy tf-keras keras -y", "Uninstalling old/problematic packages"):
        return False

    # Step 2: Install NumPy first (foundation package)
    if not run_command("pip install numpy==1.25.2", "Installing NumPy (foundation package)"):
        return False

    # Step 3: Install Pandas (must come after NumPy)
    if not run_command("pip install pandas==2.1.4", "Installing Pandas (compatible with NumPy)"):
        return False

    # Step 4: Install TensorFlow CPU version
    if not run_command("pip install tensorflow-cpu==2.13.1", "Installing TensorFlow CPU version"):
        return False

    # Step 5: Install Keras (compatible version)
    if not run_command("pip install keras==2.13.1", "Installing Keras"):
        return False

    # Step 6: Install/upgrade DeepFace
    if not run_command("pip install --upgrade deepface==0.0.79", "Installing/Upgrading DeepFace"):
        return False

    # Step 7: Install remaining dependencies
    if not run_command("pip install PyQt5==5.15.10 opencv-python==4.8.1.78 pillow==10.0.1 retina-face==0.0.13 mtcnn==0.1.1", "Installing remaining dependencies"):
        return False

    # Step 8: Test installation
    print("\nTesting installation...")
    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__} imported successfully")

        import pandas as pd
        print(f"✓ Pandas {pd.__version__} imported successfully")

        import tensorflow as tf
        print(f"✓ TensorFlow {tf.__version__} imported successfully")

        from deepface import DeepFace
        print("✓ DeepFace imported successfully")

        import cv2
        print(f"✓ OpenCV {cv2.__version__} imported successfully")

        print("\n✓ All tests passed! You can now run: python deepface_gui.py")
        return True

    except ImportError as e:
        print(f"✗ Import test failed: {e}")
        print("Please check your installation and try again.")
        return False

if __name__ == "__main__":
    success = main()
    input("\nPress Enter to exit...")
    sys.exit(0 if success else 1)
