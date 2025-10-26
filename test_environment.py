#!/usr/bin/env python3
"""
Test script to verify TensorFlow and DeepFace are working correctly
"""
import sys
import os

def test_tensorflow():
    """Test basic TensorFlow functionality"""
    print("Testing TensorFlow...")
    try:
        import tensorflow as tf
        print(f"✓ TensorFlow version: {tf.__version__}")

        # Test basic operations
        a = tf.constant([1, 2, 3])
        b = tf.constant([4, 5, 6])
        c = a + b
        print(f"✓ TensorFlow operations work: {c.numpy()}")

        # Test GPU availability
        print(f"✓ Available GPUs: {len(tf.config.list_physical_devices('GPU'))}")
        print(f"✓ Available CPUs: {len(tf.config.list_physical_devices('CPU'))}")

        return True
    except Exception as e:
        print(f"✗ TensorFlow test failed: {e}")
        return False

def test_deepface():
    """Test basic DeepFace functionality"""
    print("\nTesting DeepFace...")
    try:
        from deepface import DeepFace
        print("✓ DeepFace imported successfully")

        # Test model loading (without actual images)
        models = ["VGG-Face", "Facenet", "OpenFace", "DeepFace"]
        for model in models:
            try:
                # Just test if the model can be loaded
                DeepFace.build_model(model)
                print(f"✓ {model} model loaded successfully")
            except Exception as e:
                print(f"⚠ {model} model loading failed: {e}")

        return True
    except Exception as e:
        print(f"✗ DeepFace test failed: {e}")
        return False

def test_opencv():
    """Test OpenCV functionality"""
    print("\nTesting OpenCV...")
    try:
        import cv2
        print(f"✓ OpenCV version: {cv2.__version__}")
        return True
    except Exception as e:
        print(f"✗ OpenCV test failed: {e}")
        return False

if __name__ == "__main__":
    print("=== TensorFlow and DeepFace Compatibility Test ===\n")

    success = True

    if not test_tensorflow():
        success = False

    if not test_opencv():
        success = False

    if not test_deepface():
        success = False

    print("\n=== Test Summary ===")
    if success:
        print("✓ All tests passed! The environment is ready for DeepFace GUI.")
        print("\nYou can now run: python deepface_gui.py")
    else:
        print("✗ Some tests failed. Please check the error messages above.")
        sys.exit(1)
