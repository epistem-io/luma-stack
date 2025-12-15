#!/usr/bin/env python3
"""
Test script for OAuth2 Google Drive setup

This script helps verify that OAuth2 configuration is working correctly.
Run this before using the Google Drive export feature.
"""

import os
import sys
import json

def test_oauth_config():
    """Test OAuth2 configuration setup."""
    print("🔍 Testing OAuth2 Configuration...")
    print("=" * 50)
    
    # Test 1: Check for configuration sources
    print("1. Checking configuration sources:")
    
    config_found = False
    
    # Check environment variable (JSON content)
    if os.environ.get('GOOGLE_OAUTH_CLIENT_CONFIG'):
        print("   ✅ GOOGLE_OAUTH_CLIENT_CONFIG environment variable found")
        try:
            json.loads(os.environ['GOOGLE_OAUTH_CLIENT_CONFIG'])
            print("   ✅ Environment variable contains valid JSON")
            config_found = True
        except json.JSONDecodeError:
            print("   ❌ Environment variable contains invalid JSON")
    
    # Check environment variable (Base64)
    if os.environ.get('GOOGLE_OAUTH_CLIENT_CONFIG_B64'):
        print("   ✅ GOOGLE_OAUTH_CLIENT_CONFIG_B64 environment variable found")
        config_found = True
    
    # Check file path environment variable
    oauth_file = os.environ.get('GOOGLE_OAUTH_CLIENT_FILE')
    if oauth_file:
        print(f"   📁 GOOGLE_OAUTH_CLIENT_FILE points to: {oauth_file}")
        if os.path.exists(oauth_file):
            print("   ✅ OAuth config file exists")
            config_found = True
        else:
            print("   ❌ OAuth config file not found")
    
    # Check common file locations
    config_files = [
        'oauth_client_config.json',
        'client_secret.json',
        'auth/oauth_client_config.json',
        'auth/client_secret.json'
    ]
    
    for config_file in config_files:
        if os.path.exists(config_file):
            print(f"   ✅ Found config file: {config_file}")
            config_found = True
            break
    
    if not config_found:
        print("   ❌ No OAuth2 configuration found")
        print("\n📋 Setup Instructions:")
        print("   1. Create OAuth2 credentials in Google Cloud Console")
        print("   2. Download the client configuration JSON")
        print("   3. Save as 'oauth_client_config.json' or set environment variable")
        print("   4. See docs/google_drive_setup.md for detailed instructions")
        return False
    
    # Test 2: Try importing the OAuth module
    print("\n2. Testing OAuth module import:")
    try:
        from src.epistemx.ee_config import GoogleDriveAuth
        print("   ✅ OAuth module imported successfully")
    except ImportError as e:
        print(f"   ❌ Failed to import OAuth module: {e}")
        return False
    
    # Test 3: Initialize OAuth handler
    print("\n3. Testing OAuth initialization:")
    try:
        auth = GoogleDriveAuth()
        if auth.is_configured():
            print("   ✅ OAuth2 is properly configured")
        else:
            print("   ❌ OAuth2 configuration is invalid or missing")
            return False
    except Exception as e:
        print(f"   ❌ Failed to initialize OAuth: {e}")
        return False
    
    # Test 4: Generate auth URL
    print("\n4. Testing auth URL generation:")
    try:
        auth_url = auth.get_auth_url()
        if auth_url:
            print("   ✅ Auth URL generated successfully")
            print(f"   🔗 URL: {auth_url[:50]}...")
        else:
            print("   ❌ Failed to generate auth URL")
            return False
    except Exception as e:
        print(f"   ❌ Error generating auth URL: {e}")
        return False
    
    print("\n🎉 OAuth2 setup test completed successfully!")
    print("\n📝 Next steps:")
    print("   1. Start the Streamlit application")
    print("   2. Navigate to Module 1")
    print("   3. Select 'Google Drive' as export destination")
    print("   4. Click 'Login Google' to authenticate")
    
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