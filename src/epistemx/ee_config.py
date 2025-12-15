"""
Earth Engine and Google Drive Configuration Module

Centralized Earth Engine authentication and initialization for the epistemx package.
This module ensures Earth Engine is properly set up before any GEE operations.

Supports both service account authentication and manual user authentication.
"""

import ee
import logging
import os
import json
from typing import Optional, Dict, Any
import streamlit as st
import json
import os
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
import logging
from typing import Optional, Dict, Any

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

logger = logging.getLogger(__name__)

# OAuth2 scopes for Google Drive access
SCOPES = [
    'https://www.googleapis.com/auth/drive.file',
    'https://www.googleapis.com/auth/earthengine'
]

class GoogleDriveAuth:
    """Handle Google Drive OAuth2 authentication for Streamlit apps."""
    
    def __init__(self, client_config: Optional[Dict] = None):
        """
        Initialize Google Drive authentication.
        
        Parameters
        ----------
        client_config : dict, optional
            OAuth2 client configuration. If None, loads from environment.
        """
        self.client_config = client_config or self._load_client_config()
        self.credentials = None
        
    def _load_client_config(self) -> Optional[Dict]:
        """Load OAuth2 client configuration from environment or file."""
        # Try environment variable first (JSON string)
        client_config_json = os.environ.get('GOOGLE_OAUTH_CLIENT_CONFIG')
        if client_config_json:
            try:
                return json.loads(client_config_json)
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in GOOGLE_OAUTH_CLIENT_CONFIG: {e}")
        
        # Try file path from environment
        client_config_file = os.environ.get('GOOGLE_OAUTH_CLIENT_FILE')
        if client_config_file and os.path.exists(client_config_file):
            try:
                with open(client_config_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load OAuth config from {client_config_file}: {e}")
        
        # Try common file locations
        config_files = [
            'oauth_client_config.json',
            'client_secret.json',
            'auth/oauth_client_config.json',
            'auth/client_secret.json'
        ]
        
        for config_file in config_files:
            if os.path.exists(config_file):
                try:
                    with open(config_file, 'r') as f:
                        return json.load(f)
                except Exception as e:
                    logger.error(f"Failed to load OAuth config from {config_file}: {e}")
        
        return None
    
    def is_configured(self) -> bool:
        """Check if OAuth2 is properly configured."""
        return self.client_config is not None
    
    def get_auth_url(self, redirect_uri: str = "http://localhost:7860") -> Optional[str]:
        """
        Get OAuth2 authorization URL.
        
        Parameters
        ----------
        redirect_uri : str
            OAuth2 redirect URI
            
        Returns
        -------
        str or None
            Authorization URL or None if not configured
        """
        if not self.is_configured():
            return None
            
        try:
            flow = Flow.from_client_config(
                self.client_config,
                scopes=SCOPES,
                redirect_uri=redirect_uri
            )
            
            auth_url, _ = flow.authorization_url(
                access_type='offline',
                include_granted_scopes='true',
                prompt='consent'
            )
            
            # Store flow in session state for later use
            st.session_state['oauth_flow'] = flow
            
            return auth_url
            
        except Exception as e:
            logger.error(f"Failed to create auth URL: {e}")
            return None
    
    def handle_auth_code(self, auth_code: str, redirect_uri: str = "http://localhost:7860") -> bool:
        """
        Handle OAuth2 authorization code and get credentials.
        
        Parameters
        ----------
        auth_code : str
            Authorization code from OAuth2 callback
        redirect_uri : str
            OAuth2 redirect URI
            
        Returns
        -------
        bool
            True if authentication successful
        """
        if not self.is_configured():
            return False
            
        try:
            # Use flow from session state or create new one
            flow = st.session_state.get('oauth_flow')
            if not flow:
                flow = Flow.from_client_config(
                    self.client_config,
                    scopes=SCOPES,
                    redirect_uri=redirect_uri
                )
            
            # Exchange code for credentials
            flow.fetch_token(code=auth_code)
            self.credentials = flow.credentials
            
            # Store credentials in session state
            st.session_state['drive_credentials'] = {
                'token': self.credentials.token,
                'refresh_token': self.credentials.refresh_token,
                'token_uri': self.credentials.token_uri,
                'client_id': self.credentials.client_id,
                'client_secret': self.credentials.client_secret,
                'scopes': self.credentials.scopes
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to handle auth code: {e}")
            return False
    
    def load_credentials_from_session(self) -> bool:
        """Load credentials from Streamlit session state."""
        creds_data = st.session_state.get('drive_credentials')
        if not creds_data:
            return False
            
        try:
            self.credentials = Credentials(
                token=creds_data['token'],
                refresh_token=creds_data['refresh_token'],
                token_uri=creds_data['token_uri'],
                client_id=creds_data['client_id'],
                client_secret=creds_data['client_secret'],
                scopes=creds_data['scopes']
            )
            
            # Refresh if needed
            if self.credentials.expired and self.credentials.refresh_token:
                self.credentials.refresh(Request())
                # Update session state with new token
                st.session_state['drive_credentials']['token'] = self.credentials.token
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load credentials from session: {e}")
            return False
    
    def is_authenticated(self) -> bool:
        """Check if user is authenticated and credentials are valid."""
        if not self.credentials:
            return self.load_credentials_from_session()
        
        if self.credentials.expired and self.credentials.refresh_token:
            try:
                self.credentials.refresh(Request())
                # Update session state
                if 'drive_credentials' in st.session_state:
                    st.session_state['drive_credentials']['token'] = self.credentials.token
                return True
            except Exception as e:
                logger.error(f"Failed to refresh credentials: {e}")
                return False
        
        return self.credentials.valid
    
    def get_user_info(self) -> Optional[Dict[str, Any]]:
        """Get authenticated user information."""
        if not self.is_authenticated():
            return None
            
        try:
            # Use OAuth2 API to get user info
            service = build('oauth2', 'v2', credentials=self.credentials)
            user_info = service.userinfo().get().execute()
            return user_info
        except Exception as e:
            logger.error(f"Failed to get user info: {e}")
            return None
    
    def logout(self):
        """Clear authentication credentials."""
        self.credentials = None
        if 'drive_credentials' in st.session_state:
            del st.session_state['drive_credentials']
        if 'oauth_flow' in st.session_state:
            del st.session_state['oauth_flow']
    
    def get_drive_service(self):
        """Get authenticated Google Drive service."""
        if not self.is_authenticated():
            return None
            
        try:
            return build('drive', 'v3', credentials=self.credentials)
        except Exception as e:
            logger.error(f"Failed to create Drive service: {e}")
            return None

def create_oauth_config_template() -> Dict:
    """Create a template for OAuth2 client configuration."""
    return {
        "web": {
            "client_id": "your-client-id.apps.googleusercontent.com",
            "project_id": "your-project-id",
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "client_secret": "your-client-secret",
            "redirect_uris": ["http://localhost:7860"]
        }
    }

def show_oauth_setup_instructions():
    """Display OAuth2 setup instructions."""
    st.markdown("""
    ### Google Drive OAuth2 Setup Required
    
    To export to Google Drive, you need to set up OAuth2 authentication:
    
    1. **Create OAuth2 Credentials:**
       - Go to [Google Cloud Console](https://console.cloud.google.com/)
       - Enable Google Drive API and Earth Engine API
       - Create OAuth2 credentials (Web application)
    - Add `http://localhost:7860` to authorized redirect URIs
    
    2. **Configure Credentials:**
       - Download the client configuration JSON
       - Save as `oauth_client_config.json` in the project root or `auth/` folder
       - Or set `GOOGLE_OAUTH_CLIENT_CONFIG` environment variable with JSON content
    
    3. **Required Scopes:**
       - `https://www.googleapis.com/auth/drive.file`
       - `https://www.googleapis.com/auth/earthengine`
    """)