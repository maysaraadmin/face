#!/usr/bin/env python3
"""
Install WebEngine Script
Install PyQtWebEngine for in-app web browsing
"""

import subprocess
import sys

def install_webengine():
    """Install PyQtWebEngine"""
    print("🌐 Installing PyQtWebEngine...")
    print("=" * 50)

    try:
        # Try to install PyQtWebEngine
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "PyQtWebEngine==5.15.6"
        ], capture_output=True, text=True)

        if result.returncode == 0:
            print("✅ PyQtWebEngine installed successfully!")
            print("\n📋 You can now use in-app web browsing in DeepFace GUI")
            print("• Web Browser tab will be available")
            print("• Manual search opens in the app instead of external browser")
            print("• Enhanced user experience with integrated browsing")
            return True
        else:
            print(f"❌ Installation failed: {result.stderr}")
            print("\n💡 Try installing manually:")
            print("pip install PyQtWebEngine==5.15.6")
            return False

    except Exception as e:
        print(f"❌ Error during installation: {e}")
        return False

def main():
    """Run the installation"""
    print("PyQtWebEngine Installation Script")
    print("This will enable in-app web browsing in DeepFace GUI")
    print()

    # Ask for confirmation
    response = input("Install PyQtWebEngine? (y/n): ").lower().strip()
    if response not in ['y', 'yes']:
        print("Installation cancelled.")
        return 1

    success = install_webengine()

    if success:
        print("\n🎉 Installation complete!")
        print("Restart DeepFace GUI to use the new web browsing features.")
    else:
        print("\n❌ Installation failed.")
        print("You may need to install PyQtWebEngine manually.")

    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
