"""
deploy.py - M.A.N.T.R.A. Deployment Check Script
==============================================
Optional script to validate your setup before deployment.
Run: python deploy.py
"""

import sys
import os
import importlib.util

def check_file_exists(filename):
    """Check if a file exists"""
    if os.path.exists(filename):
        print(f"✅ {filename} - Found")
        return True
    else:
        print(f"❌ {filename} - Missing")
        return False

def check_module_syntax(filename):
    """Check if a Python module has valid syntax"""
    try:
        spec = importlib.util.spec_from_file_location("module", filename)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        print(f"✅ {filename} - Syntax OK")
        return True
    except Exception as e:
        print(f"❌ {filename} - Syntax Error: {e}")
        return False

def main():
    print("🔱 M.A.N.T.R.A. Deployment Check")
    print("=" * 40)
    
    # Required files
    required_files = [
        'app.py',
        'constants.py', 
        'data_loader.py',
        'signals.py',
        'ui_components.py',
        'requirements.txt'
    ]
    
    print("\n📁 Checking Files...")
    all_files_exist = True
    for file in required_files:
        if not check_file_exists(file):
            all_files_exist = False
    
    if not all_files_exist:
        print("\n❌ Missing files detected. Please ensure all required files are present.")
        return False
    
    print("\n🐍 Checking Python Syntax...")
    python_files = ['app.py', 'constants.py', 'data_loader.py', 'signals.py', 'ui_components.py']
    all_syntax_ok = True
    for file in python_files:
        if not check_module_syntax(file):
            all_syntax_ok = False
    
    if not all_syntax_ok:
        print("\n❌ Syntax errors detected. Please fix before deployment.")
        return False
    
    print("\n📦 Checking Requirements...")
    try:
        with open('requirements.txt', 'r') as f:
            reqs = f.read().strip()
            if 'streamlit' in reqs and 'pandas' in reqs and 'plotly' in reqs:
                print("✅ requirements.txt - Contains essential packages")
            else:
                print("⚠️ requirements.txt - Missing essential packages")
    except Exception as e:
        print(f"❌ requirements.txt - Error reading: {e}")
    
    print("\n🎯 Testing Basic Import...")
    try:
        import constants
        import data_loader
        import signals
        import ui_components
        print("✅ All modules import successfully")
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False
    
    print("\n" + "=" * 40)
    print("🚀 DEPLOYMENT READY!")
    print("=" * 40)
    print("\nNext steps:")
    print("1. Push all files to GitHub")
    print("2. Go to share.streamlit.io")
    print("3. Connect your repository")
    print("4. Set main file: app.py")
    print("5. Deploy!")
    print("\n🔱 M.A.N.T.R.A. will be live in minutes!")
    
    return True

if __name__ == "__main__":
    main()
