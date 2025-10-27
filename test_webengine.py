#!/usr/bin/env python3
"""
WebEngine Test Script
Test if PyQt WebEngine is working correctly
"""

import sys
import os

def test_webengine():
    """Test WebEngine functionality"""
    print("🌐 PyQt WebEngine Test")
    print("=" * 50)

    try:
        from PyQt5.QtWebEngineWidgets import QWebEngineView
        from PyQt5.QtWebEngine import QtWebEngine
        print("✅ WebEngine imports successful")
    except ImportError as e:
        print(f"❌ WebEngine import failed: {e}")
        print("Install with: pip install PyQtWebEngine")
        return False

    try:
        # Test WebEngine initialization (without GUI)
        from PyQt5.QtWidgets import QApplication
        from PyQt5.QtCore import QUrl

        app = QApplication(sys.argv) if not QApplication.instance() else QApplication.instance()

        # Initialize WebEngine
        QtWebEngine.initialize()
        print("✅ WebEngine initialized successfully")

        # Test creating a WebEngine view
        view = QWebEngineView()
        print("✅ WebEngineView created successfully")

        # Test setting URL
        view.setUrl(QUrl("https://www.google.com"))
        print("✅ URL set successfully")

        return True

    except Exception as e:
        print(f"❌ WebEngine test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run the test"""
    success = test_webengine()

    print("\n" + "=" * 50)
    if success:
        print("🎉 WebEngine is working correctly!")
        print("\n📋 Features available:")
        print("• In-app web browsing")
        print("• Integrated Google Images search")
        print("• Enhanced user experience")
    else:
        print("❌ WebEngine has issues")
        print("\n💡 Solutions:")
        print("• Install PyQtWebEngine: pip install PyQtWebEngine")
        print("• Check PyQt5 version compatibility")
        print("• Try restarting your Python environment")

    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
