#!/usr/bin/env python3
"""
Test script for OAuth2 Google Drive setup using Streamlit Authenticator

This script helps verify that OAuth2 configuration is working correctly.
Run this before using the Google Drive export feature.
"""

import os
import sys
import json
import yaml

def test_oauth_config():
    """Test OAuth2 configuration setup."""
    print("🔍 Testing OAuth2 Configuration...")
    print("=" * 50)
    
    # Test 1: Check for configuration sources
    print("1. Checking configuration sources:")
    
    config_found = False
    
    # Check environment variable (JSON content)
    if os.environ.get('STREAMLIT_OAUTH_CONFIG'):
        print("   ✅ STREAMLIT_OAUTH_CONFIG environment variable found")
        try:
            json.loads(os.environ['STREAMLIT_OAUTH_CONFIG'])
            print("   ✅ Environment variable contains valid JSON")
            config_found = True
        except json.JSONDecodeError:
            print("   ❌ Environment variable contains invalid JSON")
    
    # Check file path environment variable
    oauth_file = os.environ.get('STREAMLIT_OAUTH_FILE')
    if oauth_file:
        print(f"   📁 STREAMLIT_OAUTH_FILE points to: {oauth_file}")
        if os.path.exists(oauth_file):
            print("   ✅ OAuth config file exists")
            config_found = True
        else:
            print("   ❌ OAuth config file not found")
    
    # Check common file locations
    config_files = [
        'auth/oauth_config.yaml',
        'oauth_config.yaml'
    ]
    
    for config_file in config_files:
        if os.path.exists(config_file):
            print(f"   ✅ Found config file: {config_file}")
            config_found = True
            try:
                with open(config_file, 'r') as f:
                    yaml.safe_load(f)
                print(f"   ✅ {config_file} is valid YAML")
            except Exception as e:
                print(f"   ❌ Error parsing {config_file}: {e}")
            break
    
    if not config_found:
        print("   ❌ No OAuth2 configuration found")
        print("\n📋 Setup Instructions:")
        print("   1. Create auth/oauth_config.yaml with your configuration")
        print("   2. See docs/oauth2_setup_guide.md for detailed instructions")
        print("   3. Configure Streamlit Authenticator with user credentials")
        return False
    
    # Test 2: Try importing the OAuth module
    print("\n2. Testing OAuth module import:")
    try:
        from src.epistemx.ee_config import (
            setup_google_drive_oauth,
            is_user_authenticated,
            get_authenticated_user,
            get_google_drive_service,
            logout_user
        )
        print("   ✅ OAuth module imported successfully")
    except ImportError as e:
        print(f"   ❌ Failed to import OAuth module: {e}")
        return False
    
    # Test 3: Initialize OAuth handler
    print("\n3. Testing Streamlit Authenticator initialization:")
    try:
        authenticator = setup_google_drive_oauth()
        if authenticator:
            print("   ✅ Streamlit Authenticator initialized successfully")
        else:
            print("   ❌ Failed to initialize Streamlit Authenticator")
            print("   Note: This is expected if running outside of Streamlit context")
    except Exception as e:
        print(f"   ⚠️  Warning during initialization: {e}")
        print("   Note: This may be expected if running outside of Streamlit context")
    
    print("\n🎉 OAuth2 configuration test completed!")
    print("\n📝 Next steps:")
    print("   1. Configure auth/oauth_config.yaml with your user credentials")
    print("   2. Start the Streamlit application: streamlit run home.py")
    print("   3. Navigate to Module 1 (Generate Image Mosaic)")
    print("   4. Select 'Google Drive' as export destination")
    print("   5. Log in with your credentials")
    print("   6. Authorize access to Google Drive")
    
    return True

def test_dependencies():
    """Test required dependencies."""
    print("\n🔍 Testing Dependencies...")
    print("=" * 50)
    required_packages = [
        'google.auth',
        'google_auth_oauthlib',
        'googleapiclient',
        'streamlit'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} (missing)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("   Run: pip install -r requirements.txt")
        return False
    
    print("\n✅ All dependencies are installed")
    return True

if __name__ == "__main__":
    print("🚀 EpistemX OAuth2 Setup Test")
    print("=" * 50)
    
    # Test dependencies first
    if not test_dependencies():
        sys.exit(1)
    
    # Test OAuth configuration
    if not test_oauth_config():
        sys.exit(1)
    
    print("\n🎉 All tests passed! OAuth2 setup is ready.")