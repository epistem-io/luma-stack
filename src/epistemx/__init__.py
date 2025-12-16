"""
EpistemX - Earth Observation Data Processing Package

A comprehensive package for processing and analyzing Earth observation data
using Google Earth Engine and other remote sensing tools.
"""

from .ee_config import (
    initialize_earth_engine, 
    ensure_ee_initialized, 
    is_ee_initialized,
    initialize_with_service_account,
    authenticate_manually,
    get_auth_status,
    print_auth_instructions,
    setup_earth_engine,
    reset_ee_initialization
)

from .ee_config import (
    setup_google_drive_oauth,
    is_user_authenticated,
    get_authenticated_user,
    get_google_drive_service,
    logout_user,
    get_google_oauth_config,
    initiate_google_oauth_login,
    handle_google_oauth_callback,
    show_oauth_setup_instructions,
    print_oauth_setup_instructions
)
import os
import warnings

# Do not automatically initialize Earth Engine at import time
# Users should call setup_earth_engine() or ensure_ee_initialized() when needed

# Check for service account file in environment or common locations
def _find_service_account_file():
    """Look for service account file in common locations."""
    # Debug: Check what environment variables are available
    print("DEBUG: Checking environment variables...")
    print(f"DEBUG: GOOGLE_SERVICE_ACCOUNT_JSON exists: {bool(os.environ.get('GOOGLE_SERVICE_ACCOUNT_JSON'))}")
    print(f"DEBUG: GOOGLE_APPLICATION_CREDENTIALS: {os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')}")
    
    # Check for Base64 encoded service account JSON first (most robust)
    service_account_b64 = os.environ.get('GOOGLE_SERVICE_ACCOUNT_JSON_B64')
    if service_account_b64:
        import tempfile
        import json
        import base64
        try:
            print("DEBUG: Found GOOGLE_SERVICE_ACCOUNT_JSON_B64, decoding...")
            # Decode from Base64
            raw_json = base64.b64decode(service_account_b64).decode('utf-8')
            # Validate JSON
            service_account_info = json.loads(raw_json)
            # Create temporary file
            temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
            json.dump(service_account_info, temp_file, indent=2)
            temp_file.close()
            print(f"DEBUG: Created temporary service account file from Base64: {temp_file.name}")
            return temp_file.name
        except Exception as e:
            print(f"Warning: Failed to decode GOOGLE_SERVICE_ACCOUNT_JSON_B64: {e}")
    
    # Check if service account JSON is provided as environment variable content
    service_account_json = os.environ.get('GOOGLE_SERVICE_ACCOUNT_JSON')
    if service_account_json:
        # Create temporary file from environment variable content
        import tempfile
        import json
        try:
            # Validate JSON format
            service_account_info = json.loads(service_account_json)
            # Create temporary file with properly formatted JSON
            temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
            json.dump(service_account_info, temp_file, indent=2)
            temp_file.close()
            print(f"DEBUG: Created temporary service account file: {temp_file.name}")
            return temp_file.name
        except json.JSONDecodeError as e:
            print(f"Warning: GOOGLE_SERVICE_ACCOUNT_JSON is not valid JSON: {e}")
            print(f"DEBUG: Error at position {e.pos}")
            print(f"DEBUG: Content around error: {service_account_json[max(0, e.pos-50):e.pos+50]}")
            print("DEBUG: This usually means the JSON contains unescaped newlines in the private_key field")
            print("DEBUG: Please use GOOGLE_SERVICE_ACCOUNT_JSON_B64 instead for robust handling")
    
    # Check environment variable for file path
    env_file = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
    if env_file and os.path.exists(env_file):
        print(f"DEBUG: Using GOOGLE_APPLICATION_CREDENTIALS: {env_file}")
        return env_file
    
    # Check auth/ directory first (project-specific location)
    # Use absolute path to ensure it works regardless of working directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))  # Go up from src/epistemx to project root
    auth_dir = os.path.join(project_root, 'auth')
    
    print(f"DEBUG: Checking auth directory: {auth_dir}")
    if os.path.exists(auth_dir):
        print(f"DEBUG: Auth directory exists, listing files...")
        for file in os.listdir(auth_dir):
            print(f"DEBUG: Found file in auth/: {file}")
            if file.endswith('.json'):
                auth_file = os.path.join(auth_dir, file)
                if os.path.exists(auth_file):
                    print(f"DEBUG: Using auth file: {auth_file}")
                    return auth_file
    else:
        print(f"DEBUG: Auth directory does not exist: {auth_dir}")
    
    # Also check relative auth/ directory (fallback)
    auth_dir_relative = 'auth'
    print(f"DEBUG: Checking relative auth directory: {auth_dir_relative}")
    if os.path.exists(auth_dir_relative):
        print(f"DEBUG: Relative auth directory exists, listing files...")
        for file in os.listdir(auth_dir_relative):
            print(f"DEBUG: Found file in relative auth/: {file}")
            if file.endswith('.json'):
                auth_file = os.path.join(auth_dir_relative, file)
                if os.path.exists(auth_file):
                    print(f"DEBUG: Using relative auth file: {auth_file}")
                    return auth_file
    else:
        print(f"DEBUG: Relative auth directory does not exist: {auth_dir_relative}")
    
    # Check for Docker container specific path
    docker_auth_dir = '/home/user/app/auth'
    print(f"DEBUG: Checking Docker auth directory: {docker_auth_dir}")
    if os.path.exists(docker_auth_dir):
        print(f"DEBUG: Docker auth directory exists, listing files...")
        for file in os.listdir(docker_auth_dir):
            print(f"DEBUG: Found file in Docker auth/: {file}")
            if file.endswith('.json'):
                auth_file = os.path.join(docker_auth_dir, file)
                if os.path.exists(auth_file):
                    print(f"DEBUG: Using Docker auth file: {auth_file}")
                    return auth_file
    else:
        print(f"DEBUG: Docker auth directory does not exist: {docker_auth_dir}")
    
    # Check common file names in current directory
    common_names = [
        'service-account.json',
        'gee-service-account.json',
        'earth-engine-service-account.json',
        'credentials.json'
    ]
    
    for name in common_names:
        if os.path.exists(name):
            return name
    
    return None

# Helper function for manual initialization (can be called by users)
def auto_initialize():
    """Attempt automatic initialization with fallback options."""
    try:
        print("DEBUG: Starting auto_initialize...")
        print(f"DEBUG: All environment variables containing 'GOOGLE': {[k for k in os.environ.keys() if 'GOOGLE' in k]}")
        
        # First, try to find and use service account
        service_account_file = _find_service_account_file()
        if service_account_file:
            print(f"DEBUG: Found service account file: {service_account_file}")
            if setup_earth_engine(service_account_file=service_account_file):
                print(f"Earth Engine initialized with service account: {service_account_file}")
                return True
        else:
            print("DEBUG: No service account file found")
        
        # If no service account, try regular initialization
        print("DEBUG: Trying regular initialization...")
        if setup_earth_engine():
            print("Earth Engine initialized successfully")
            return True
        
        print("DEBUG: All initialization methods failed")
        return False
        
    except Exception as e:
        print(f"DEBUG: Exception in auto_initialize: {e}")
        warnings.warn(
            f"Could not automatically initialize Earth Engine: {e}. "
            "You may need to authenticate manually. Use print_auth_instructions() for help.",
            UserWarning
        )
        return False

__version__ = "0.1.0"
__author__ = "EpistemX Team"

# Make key functions available at package level
__all__ = [
    'initialize_earth_engine',
    'ensure_ee_initialized', 
    'is_ee_initialized',
    'initialize_with_service_account',
    'authenticate_manually',
    'get_auth_status',
    'print_auth_instructions',
    'setup_earth_engine',
    'reset_ee_initialization',
    'auto_initialize',
    'setup_google_drive_oauth',
    'is_user_authenticated',
    'get_authenticated_user',
    'get_google_drive_service',
    'logout_user',
    'get_google_oauth_config',
    'initiate_google_oauth_login',
    'handle_google_oauth_callback',
    'show_oauth_setup_instructions',
    'print_oauth_setup_instructions'
]