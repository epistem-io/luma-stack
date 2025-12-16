"""
Earth Engine and Google Drive Configuration Module

Centralized Earth Engine authentication and initialization for the epistemx package.
This module ensures Earth Engine is properly set up before any GEE operations.

Supports both service account authentication and manual user authentication.
Uses Streamlit Authenticator for OAuth2 Google Drive authentication.
"""

import ee
import logging
import os
import json
import yaml
from typing import Optional, Dict, Any
import streamlit as st
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
import streamlit_authenticator as stauth

# Configure logging
logger = logging.getLogger(__name__)

# Global flag to track initialization status
_ee_initialized = False

def initialize_with_service_account(
    service_account_file: str, 
    project: Optional[str] = None
) -> bool:
    """
    Initialize Earth Engine using a service account.
    
    Parameters
    ----------
    service_account_file : str
        Path to the service account JSON key file.
    project : str, optional
        GEE project ID. If None, uses project from service account.
        
    Returns
    -------
    bool
        True if initialization successful, False otherwise.
        
    Example
    -------
    >>> from epistemx.ee_config import initialize_with_service_account
    >>> initialize_with_service_account('path/to/service-account.json')
    """
    global _ee_initialized
    
    # Initialize service_account_info variable
    service_account_info = None
    
    try:
        # Validate service account file exists
        if not os.path.exists(service_account_file):
            logger.error(f"Service account file not found: {service_account_file}")
            return False
        
        # Load service account credentials
        with open(service_account_file, 'r') as f:
            try:
                service_account_info = json.load(f)
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error in service account file: {e}")
                logger.error(f"File: {service_account_file}")
                return False
        
        # Check if we successfully parsed the service account info
        if not service_account_info:
            logger.error("Failed to parse service account JSON")
            return False
        
        # Extract project ID if not provided
        if not project:
            project = service_account_info.get('project_id')
        
        logger.info(f"Attempting to initialize Earth Engine with service account for project: {project}")
        
        # Set the environment variable for Google Application Credentials
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = service_account_file
        
        # Initialize Earth Engine with the project
        if project:
            ee.Initialize(project=project)
        else:
            ee.Initialize()
        
        _ee_initialized = True
        logger.info(f"Earth Engine initialized successfully with service account for project: {project}")
        return True
        
    except Exception as e:
        logger.error(f"Service account initialization failed: {e}")
        # Try alternative method with explicit credentials only if we have service account info
        if service_account_info:
            try:
                logger.info("Trying alternative authentication method...")
                credentials = ee.ServiceAccountCredentials(
                    email=service_account_info['client_email'],
                    key_file=service_account_file
                )
                
                if project:
                    ee.Initialize(credentials, project=project)
                else:
                    ee.Initialize(credentials)
                
                _ee_initialized = True
                logger.info(f"Earth Engine initialized with alternative method for project: {project}")
                return True
                
            except Exception as e2:
                logger.error(f"Alternative authentication method also failed: {e2}")
                return False
        else:
            logger.error("Cannot try alternative method - service account info not available")
            return False

def authenticate_manually(project: Optional[str] = None) -> bool:
    """
    Perform manual Earth Engine authentication.
    
    This will open a browser window for authentication.
    
    Parameters
    ----------
    project : str, optional
        GEE project ID. If None, uses default project.
        
    Returns
    -------
    bool
        True if authentication and initialization successful, False otherwise.
        
    Example
    -------
    >>> from epistemx.ee_config import authenticate_manually
    >>> authenticate_manually()
    """
    global _ee_initialized
    
    try:
        logger.info("Starting manual Earth Engine authentication...")
        ee.Authenticate()
        
        if project:
            ee.Initialize(project=project)
        else:
            ee.Initialize()
        
        _ee_initialized = True
        logger.info("Earth Engine authenticated and initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Manual authentication failed: {e}")
        return False

def _print_manual_auth_instructions() -> None:
    """Print step-by-step manual authentication instructions."""
    instructions = """
    EARTH ENGINE AUTHENTICATION NOTES:
    
    1. Make sure you already have a google cloud project that has enable the Earth Engine API and registered to 
       commercial or non-commercial use. For more information visit: https://developers.google.com/earth-engine/guides/access 
    
    2. you can authenticate programmatically by calling: from epistemx.ee_config import authenticate_manually
       authenticate_manually()
    
    3. This will open a web browser. Sign in with your Google account that has Earth Engine access.
    
    4. Copy the authorization code from the browser and paste it in the terminal.
    
    
    For more details, visit: https://developers.google.com/earth-engine/guides/python_install
    """
    print(instructions)

def initialize_earth_engine(
    project: Optional[str] = None, 
    service_account_file: Optional[str] = None,
    force_reinit: bool = False
) -> bool:
    """
    Initialize Google Earth Engine with authentication.
    
    Parameters
    ----------
    project : str, optional
        GEE project ID. If None, uses default project.
    service_account_file : str, optional
        Path to service account JSON file. If provided, uses service account auth.
    force_reinit : bool, default False
        Force re-initialization even if already initialized.
        
    Returns
    -------
    bool
        True if initialization successful, False otherwise.
        
    Example
    -------
    >>> from epistemx.ee_config import initialize_earth_engine
    >>> # Manual authentication
    >>> initialize_earth_engine()
    >>> # Service account authentication
    >>> initialize_earth_engine(service_account_file='service-account.json')
    """
    global _ee_initialized
    
    if _ee_initialized and not force_reinit:
        logger.debug("Earth Engine already initialized")
        return True
    
    # Use service account if provided
    if service_account_file:
        return initialize_with_service_account(service_account_file, project)
    
    try:
        # Try to initialize without authentication first (for already authenticated users)
        if project:
            ee.Initialize(project=project)
        else:
            ee.Initialize()
        
        _ee_initialized = True
        logger.info("Earth Engine initialized successfully")
        return True
        
    except ee.EEException as e:
        if "not authenticated" in str(e).lower():
            logger.warning("Earth Engine authentication required. Please run manual authentication.")
            logger.info("To authenticate manually, follow these steps:")
            _print_manual_auth_instructions()
            return False
        else:
            logger.error(f"Earth Engine initialization failed: {e}")
            return False
    
    except Exception as e:
        logger.error(f"Unexpected error during Earth Engine initialization: {e}")
        return False

def ensure_ee_initialized(
    project: Optional[str] = None, 
    service_account_file: Optional[str] = None
) -> None:
    """
    Ensure Earth Engine is initialized, raising an exception if it fails.
    
    Parameters
    ----------
    project : str, optional
        GEE project ID. If None, uses default project.
    service_account_file : str, optional
        Path to service account JSON file. If provided, uses service account auth.
        
    Raises
    ------
    RuntimeError
        If Earth Engine initialization fails.
    """
    if not initialize_earth_engine(project=project, service_account_file=service_account_file):
        raise RuntimeError(
            "Failed to initialize Google Earth Engine. "
            "Please check your authentication and internet connection. "
            "Run authenticate_manually() or provide valid service account credentials."
        )

def is_ee_initialized() -> bool:
    """
    Check if Earth Engine is initialized.
    
    Returns
    -------
    bool
        True if Earth Engine is initialized, False otherwise.
    """
    return _ee_initialized

def get_auth_status() -> Dict[str, Any]:
    """
    Get detailed authentication status information.
    
    Returns
    -------
    dict
        Dictionary containing authentication status details.
    """
    status = {
        'initialized': _ee_initialized,
        'authenticated': False,
        'project': None,
        'user_info': None
    }
    
    if _ee_initialized:
        try:
            # Try a simple operation to verify authentication
            ee.Number(1).getInfo()
            status['authenticated'] = True
            
            # Try to get project info
            try:
                # This might not work in all cases, but worth trying
                status['project'] = ee.data.getAssetRoots()[0]['id'] if ee.data.getAssetRoots() else None
            except:
                pass
                
        except Exception as e:
            logger.debug(f"Authentication check failed: {e}")
            status['authenticated'] = False
    
    return status

def print_auth_instructions() -> None:
    """
    Print comprehensive authentication instructions.
    """
    _print_manual_auth_instructions()

def reset_ee_initialization() -> None:
    """
    Reset the initialization flag. Useful for testing or troubleshooting.
    """
    global _ee_initialized
    _ee_initialized = False
    logger.debug("Earth Engine initialization flag reset")

def setup_earth_engine(
    project: Optional[str] = None,
    service_account_file: Optional[str] = None,
    auto_authenticate: bool = False
) -> bool:
    """
    Comprehensive Earth Engine setup function.
    
    Parameters
    ----------
    project : str, optional
        GEE project ID.
    service_account_file : str, optional
        Path to service account JSON file.
    auto_authenticate : bool, default False
        If True, attempt manual authentication if needed.
        
    Returns
    -------
    bool
        True if setup successful, False otherwise.
        
    Example
    -------
    >>> from epistemx.ee_config import setup_earth_engine
    >>> # Try automatic setup
    >>> setup_earth_engine()
    >>> # Setup with service account
    >>> setup_earth_engine(service_account_file='service-account.json')
    """
    # First try normal initialization
    if initialize_earth_engine(project=project, service_account_file=service_account_file):
        return True
    
    # If that fails and auto_authenticate is True, try manual auth
    if auto_authenticate and not service_account_file:
        logger.info("Attempting manual authentication...")
        return authenticate_manually(project=project)
    
    return False


# ============================================================================
# Google Drive OAuth2 Authentication using Streamlit Authenticator
# ============================================================================

# OAuth2 scopes for Google Drive access
OAUTH_SCOPES = [
    'https://www.googleapis.com/auth/drive.file',
    'https://www.googleapis.com/auth/earthengine'
]


def _load_oauth_config() -> Optional[Dict]:
    """
    Load OAuth2 configuration from environment or file.
    
    Priority:
    1. STREAMLIT_OAUTH_CONFIG environment variable (JSON string)
    2. STREAMLIT_OAUTH_FILE environment variable (path to YAML file)
    3. auth/oauth_config.yaml (default file location)
    4. oauth_config.yaml (root directory)
    
    Returns
    -------
    dict or None
        OAuth configuration dictionary or None if not found
    """
    # Try environment variable first (JSON string)
    oauth_config_json = os.environ.get('STREAMLIT_OAUTH_CONFIG')
    if oauth_config_json:
        try:
            return json.loads(oauth_config_json)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in STREAMLIT_OAUTH_CONFIG: {e}")
    
    # Try file path from environment variable
    oauth_config_file = os.environ.get('STREAMLIT_OAUTH_FILE')
    if oauth_config_file and os.path.exists(oauth_config_file):
        try:
            with open(oauth_config_file, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load OAuth config from {oauth_config_file}: {e}")
    
    # Try common file locations
    config_files = [
        'auth/oauth_config.yaml',
        'oauth_config.yaml',
    ]
    
    for config_file in config_files:
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    return yaml.safe_load(f)
            except Exception as e:
                logger.error(f"Failed to load OAuth config from {config_file}: {e}")
    
    return None


def setup_google_drive_oauth(
    config: Optional[Dict] = None,
    cookie_name: str = "epistemx_oauth",
    cookie_key: str = "epistemx_key",
    cookie_expiry_days: int = 30
) -> Optional[Any]:
    """
    Initialize Streamlit Authenticator for Google Drive OAuth2.
    
    Parameters
    ----------
    config : dict, optional
        OAuth configuration. If None, loads from file or environment.
    cookie_name : str, default "epistemx_oauth"
        Name for session cookie
    cookie_key : str, default "epistemx_key"
        Key for session cookie encryption
    cookie_expiry_days : int, default 30
        Cookie expiration time in days
        
    Returns
    -------
    Authenticator object or None
        Initialized Streamlit Authenticator or None if config not found
        
    Example
    -------
    >>> from epistemx.ee_config import setup_google_drive_oauth
    >>> authenticator = setup_google_drive_oauth()
    >>> if authenticator:
    ...     authenticator.login()
    """
    # Load config if not provided
    if config is None:
        config = _load_oauth_config()
    
    if not config:
        logger.error("OAuth configuration not found. Please check your config file or environment variables.")
        return None
    
    try:
        # Initialize Streamlit Authenticator
        authenticator = stauth.Authenticate(
            credentials=config.get('credentials', {}),
            cookie_name=cookie_name,
            cookie_key=cookie_key,
            cookie_expiry_days=cookie_expiry_days,
            pre_authorized=config.get('pre_authorized', [])
        )
        
        logger.info("Streamlit Authenticator initialized successfully")
        return authenticator
        
    except Exception as e:
        logger.error(f"Failed to initialize Streamlit Authenticator: {e}")
        return None


def get_google_oauth_config() -> Optional[Dict]:
    """
    Get Google OAuth2 configuration from loaded config.
    
    Returns
    -------
    dict or None
        Google OAuth2 configuration if available and enabled, None otherwise
    """
    config = _load_oauth_config()
    if not config:
        return None
    
    google_oauth = config.get('google_oauth', {})
    if google_oauth.get('enabled', False):
        return google_oauth
    
    return None


def initiate_google_oauth_login() -> Optional[str]:
    """
    Generate Google OAuth2 authorization URL for Streamlit app.
    
    This creates the URL that users should click to authenticate with Google.
    
    Returns
    -------
    str or None
        Authorization URL or None if OAuth2 not configured
        
    Example
    -------
    >>> from epistemx.ee_config import initiate_google_oauth_login
    >>> auth_url = initiate_google_oauth_login()
    >>> if auth_url:
    ...     st.markdown(f'[Login with Google]({auth_url})')
    """
    try:
        google_config = get_google_oauth_config()
        if not google_config:
            logger.error("Google OAuth2 not configured or not enabled")
            return None
        
        from google_auth_oauthlib.flow import Flow
        
        # Create OAuth2 flow
        flow = Flow.from_client_config(
            {
                "installed": {
                    "client_id": google_config.get('client_id'),
                    "client_secret": google_config.get('client_secret'),
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                    "redirect_uris": [google_config.get('redirect_uri')]
                }
            },
            scopes=google_config.get('scopes', OAUTH_SCOPES),
            redirect_uri=google_config.get('redirect_uri')
        )
        
        # Generate authorization URL
        auth_url, state = flow.authorization_url(
            access_type='offline',
            include_granted_scopes='true',
            prompt='consent'
        )
        
        # Store flow in session state for later use
        st.session_state['oauth_flow'] = flow
        st.session_state['oauth_state'] = state
        
        logger.info("Google OAuth2 authorization URL generated")
        return auth_url
        
    except Exception as e:
        logger.error(f"Failed to generate Google OAuth2 URL: {e}")
        return None


def handle_google_oauth_callback(code: str) -> bool:
    """
    Handle Google OAuth2 callback and save credentials to session.
    
    Parameters
    ----------
    code : str
        Authorization code from Google OAuth2 callback
        
    Returns
    -------
    bool
        True if authentication successful, False otherwise
    """
    try:
        flow = st.session_state.get('oauth_flow')
        if not flow:
            logger.error("OAuth flow not found in session state")
            return False
        
        # Exchange authorization code for credentials
        flow.fetch_token(code=code)
        credentials = flow.credentials
        
        # Store credentials in session state
        st.session_state['oauth_credentials'] = {
            'token': credentials.token,
            'refresh_token': credentials.refresh_token,
            'token_uri': credentials.token_uri,
            'client_id': credentials.client_id,
            'client_secret': credentials.client_secret,
            'scopes': credentials.scopes
        }
        
        # Get user info
        try:
            from googleapiclient.discovery import build
            oauth2_service = build('oauth2', 'v2', credentials=credentials)
            user_info = oauth2_service.userinfo().get().execute()
            st.session_state['authenticated_user'] = user_info.get('email', 'Unknown')
            st.session_state['authentication_status'] = True
        except Exception as e:
            logger.warning(f"Could not retrieve user info: {e}")
            st.session_state['authenticated_user'] = 'Google User'
            st.session_state['authentication_status'] = True
        
        logger.info("Google OAuth2 authentication successful")
        return True
        
    except Exception as e:
        logger.error(f"Failed to handle OAuth2 callback: {e}")
        return False


def is_user_authenticated() -> bool:
    """
    Check if user is currently authenticated.
    
    Returns
    -------
    bool
        True if user is authenticated, False otherwise
    """
    return st.session_state.get('authentication_status', False) is True


def get_authenticated_user() -> Optional[str]:
    """
    Get the current authenticated username.
    
    Returns
    -------
    str or None
        Username if authenticated, None otherwise
    """
    if is_user_authenticated():
        return st.session_state.get('username')
    return None


def get_google_drive_service() -> Optional[Any]:
    """
    Get an authenticated Google Drive service instance.
    
    Requires user to be authenticated via setup_google_drive_oauth().
    
    Returns
    -------
    googleapiclient.discovery.Resource or None
        Authenticated Drive service or None if not authenticated
        
    Example
    -------
    >>> from epistemx.ee_config import get_google_drive_service
    >>> service = get_google_drive_service()
    >>> if service:
    ...     files = service.files().list().execute()
    """
    if not is_user_authenticated():
        logger.warning("User not authenticated. Cannot create Drive service.")
        return None
    
    try:
        # Get credentials from session state (set by Streamlit Authenticator)
        credentials_data = st.session_state.get('oauth_credentials')
        if not credentials_data:
            logger.error("OAuth credentials not found in session state")
            return None
        
        # Create credentials object
        credentials = Credentials(
            token=credentials_data.get('token'),
            refresh_token=credentials_data.get('refresh_token'),
            token_uri=credentials_data.get('token_uri'),
            client_id=credentials_data.get('client_id'),
            client_secret=credentials_data.get('client_secret'),
            scopes=credentials_data.get('scopes', OAUTH_SCOPES)
        )
        
        # Refresh if needed
        if credentials.expired and credentials.refresh_token:
            credentials.refresh(Request())
            # Update session state
            st.session_state['oauth_credentials']['token'] = credentials.token
        
        # Build and return Drive service
        service = build('drive', 'v3', credentials=credentials)
        return service
        
    except Exception as e:
        logger.error(f"Failed to create Google Drive service: {e}")
        return None


def logout_user() -> None:
    """
    Log out the current user and clear authentication data.
    
    Example
    -------
    >>> from epistemx.ee_config import logout_user
    >>> if st.button("Logout"):
    ...     logout_user()
    ...     st.rerun()
    """
    try:
        # Clear authentication-related session state
        auth_keys_to_remove = [
            'authentication_status',
            'username',
            'oauth_credentials',
            'oauth_flow'
        ]
        
        for key in auth_keys_to_remove:
            if key in st.session_state:
                del st.session_state[key]
        
        logger.info("User logged out successfully")
        
    except Exception as e:
        logger.error(f"Error during logout: {e}")