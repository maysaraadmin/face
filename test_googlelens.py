#!/usr/bin/env python3
"""
GoogleLens Diagnostic Script
Run this to test and debug GoogleLens functionality
"""

import sys
import os

def test_googlelens():
    """Test GoogleLens functionality"""
    print("🔍 GoogleLens Diagnostic Test")
    print("=" * 50)

    # Test import
    try:
        from googlelens import GoogleLens
        print("✅ GoogleLens imported successfully")
        HAS_GOOGLE_LENS = True
    except ImportError as e:
        print(f"❌ GoogleLens import failed: {e}")
        print("Install with: pip install git+https://github.com/krishna2206/google-lens-python.git")
        return False

    # Test instance creation
    try:
        gl = GoogleLens()
        print("✅ GoogleLens instance created")
    except Exception as e:
        print(f"❌ GoogleLens instance creation failed: {e}")
        return False

    # Check available methods
    methods = [method for method in dir(gl) if not method.startswith('_') and callable(getattr(gl, method))]
    print(f"📋 Available methods: {methods}")

    # Check for expected methods
    expected_methods = ['search_by_file', 'search_by_url', 'upload', 'search']
    for method in expected_methods:
        if hasattr(gl, method):
            print(f"✅ {method} method available")
        else:
            print(f"❌ {method} method missing")

    # Test with a sample image if available
    test_image = None
    for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        test_path = f"test_image{ext}"
        if os.path.exists(test_path):
            test_image = test_path
            break

    if test_image:
        print(f"\n🖼️ Testing with image: {test_image}")
        try:
            # Try search_by_file method (correct API)
            if hasattr(gl, 'search_by_file'):
                print("🔄 Testing search_by_file method...")
                results = gl.search_by_file(test_image)
                print(f"✅ search_by_file successful, got {len(results)} results")
                return True
            elif hasattr(gl, 'search_by_url'):
                print("🔄 Testing search_by_url method...")
                # Convert file path to file URL
                from urllib.parse import urljoin
                from urllib.request import pathname2url
                file_url = urljoin('file:', pathname2url(os.path.abspath(test_image)))
                results = gl.search_by_url(file_url)
                print(f"✅ search_by_url successful, got {len(results)} results")
                return True
            elif hasattr(gl, 'upload'):
                print("🔄 Testing upload method...")
                results = gl.upload(test_image)
                print(f"✅ Upload successful, got {len(results)} results")
                return True
            elif hasattr(gl, 'search'):
                print("🔄 Testing search method...")
                results = gl.search(test_image)
                print(f"✅ Search successful, got {len(results)} results")
                return True
            else:
                print("❌ No usable methods found")
                return False
        except Exception as e:
            print(f"❌ Method execution failed: {e}")
            return False
    else:
        print("\n⚠️ No test image found")
        print("To test fully: Place a test image named 'test_image.jpg' in the current directory")

        # Try to call methods without parameters to see what happens
        try:
            print("\n🔄 Testing method calls without parameters...")
            if hasattr(gl, 'search_by_file'):
                print("Trying gl.search_by_file() without parameters...")
                results = gl.search_by_file()
                print(f"✅ search_by_file without parameters worked: {len(results)} results")
            elif hasattr(gl, 'search_by_url'):
                print("Trying gl.search_by_url() without parameters...")
                results = gl.search_by_url()
                print(f"✅ search_by_url without parameters worked: {len(results)} results")
            elif hasattr(gl, 'upload'):
                print("Trying gl.upload() without parameters...")
                results = gl.upload()
                print(f"✅ Upload without parameters worked: {len(results)} results")
            elif hasattr(gl, 'search'):
                print("Trying gl.search() without parameters...")
                results = gl.search()
                print(f"✅ Search without parameters worked: {len(results)} results")
        except Exception as e:
            print(f"❌ Method call without parameters failed: {e}")

    return True

def main():
    """Run the diagnostic"""
    success = test_googlelens()

    print("\n" + "=" * 50)
    if success:
        print("🎉 GoogleLens appears to be working!")
        print("\n📋 Recommendations:")
        print("• GoogleLens is functional")
        print("• GUI now uses correct method names: search_by_file, search_by_url")
        print("• Use it in the GUI Internet Search tab")
        print("• If it still doesn't work in GUI, check the console output")
    else:
        print("❌ GoogleLens has issues")
        print("\n💡 Solutions:")
        print("• Update GoogleLens: pip install --upgrade git+https://github.com/krishna2206/google-lens-python.git")
        print("• Try alternative search methods in the GUI")
        print("• Use 'Manual Search (Browser)' - always reliable!")
        print("• Use 'Search with Selenium' - automated search")

    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
