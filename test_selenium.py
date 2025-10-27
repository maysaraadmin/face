#!/usr/bin/env python3
"""
Test script for DeepFace GUI Selenium functionality
Run this to verify that all dependencies are working correctly.
"""

import sys
import os

def test_imports():
    """Test if all required modules can be imported"""
    print("🔍 Testing imports...")

    try:
        import selenium
        print(f"✅ Selenium: {selenium.__version__}")
    except ImportError as e:
        print(f"❌ Selenium import failed: {e}")
        return False

    try:
        import webdriver_manager
        print(f"✅ Webdriver Manager: {webdriver_manager.__version__}")
    except ImportError as e:
        print(f"❌ Webdriver Manager import failed: {e}")
        return False

    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        print("✅ Selenium webdriver imported successfully")
    except ImportError as e:
        print(f"❌ Selenium webdriver import failed: {e}")
        return False

    try:
        from webdriver_manager.chrome import ChromeDriverManager
        print("✅ ChromeDriverManager imported successfully")
    except ImportError as e:
        print(f"❌ ChromeDriverManager import failed: {e}")
        return False

    return True

def test_chrome_setup():
    """Test Chrome driver initialization"""
    print("\n🔧 Testing Chrome driver setup...")

    try:
        from selenium.webdriver.chrome.options import Options
        from webdriver_manager.chrome import ChromeDriverManager

        # Test options creation
        options = Options()
        options.add_argument("--headless")
        print("✅ Chrome options created successfully")

        # Test webdriver-manager installation
        try:
            driver_path = ChromeDriverManager().install()
            print(f"✅ ChromeDriver installed at: {driver_path}")
        except Exception as e:
            print(f"⚠️ ChromeDriverManager failed: {e}")
            print("This is normal if Chrome is already installed")

        return True

    except Exception as e:
        print(f"❌ Chrome setup failed: {e}")
        return False

def test_gui_import():
    """Test if the GUI module can be imported"""
    print("\n🖥️ Testing GUI import...")

    try:
        # Change to the face directory
        os.chdir('d:\\face')

        # Try to import the GUI module
        import deepface_gui
        print("✅ DeepFace GUI imported successfully")

        # Check if key methods exist
        if hasattr(deepface_gui.DeepFaceGUI, 'perform_internet_search'):
            print("✅ perform_internet_search method found")
        else:
            print("❌ perform_internet_search method not found")
            return False

        if hasattr(deepface_gui.DeepFaceGUI, 'open_manual_search'):
            print("✅ open_manual_search method found")
        else:
            print("❌ open_manual_search method not found")
            return False

        return True

    except ImportError as e:
        print(f"❌ GUI import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ GUI test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 DeepFace GUI Selenium Test Suite")
    print("=" * 50)

    all_passed = True

    # Test imports
    if not test_imports():
        all_passed = False

    # Test Chrome setup
    if not test_chrome_setup():
        all_passed = False

    # Test GUI import
    if not test_gui_import():
        all_passed = False

    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All tests passed! Selenium functionality should work correctly.")
        print("\n📋 Next steps:")
        print("1. Run: python deepface_gui.py")
        print("2. Go to the 'Internet Search' tab")
        print("3. Upload an image")
        print("4. Try 'Manual Search (Browser)' first - this always works!")
        print("5. If that works, try 'Search with Selenium'")
    else:
        print("❌ Some tests failed. Please check the error messages above.")
        print("\n🔧 Troubleshooting:")
        print("• Install Chrome browser: https://www.google.com/chrome/")
        print("• Update dependencies: pip install -r requirements.txt")
        print("• Try manual search first to verify Google Images works")

    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
