import ee
from .ee_config import ensure_ee_initialized
import logging
from typing import List, Optional
import geopandas as gpd
from shapely.geometry import shape
# Configure logger
logger = logging.getLogger(__name__)
class GEE_Asset_Manager:
    """
    Manager class for Google Earth Engine asset operations.
    
    This class provides methods for loading and processing administrative boundary
    data from GEE asset
    
    Attributes:
        asset_path: Path to the GEE asset containing administrative boundaries
        name_property: Property name containing regency/city names
        asset: Loaded GEE FeatureCollection (cached after first load)
    
    Example:
        >>> manager = GEEAssetManager()
        >>> if manager.load_asset():
        ...     names = manager.get_regency_names()
        ...     geometry = manager.get_regency_geometry(names[0])
    """
    
    def __init__(
        self,
        asset_path: str = 'projects/ee-rg2icraf/assets/Indonesian_Regency_Area', #can be change or added here
        name_property: str = "WADMKK"
    ):
        """
        Initialize GEE Asset Manager.
        
        Args:
            asset_path: Path to the GEE asset containing regency boundaries
            name_property: Property name containing regency names (default: 'WADMKK')
        """
        self.asset_path = asset_path
        self.name_property = name_property
        self.asset: Optional[ee.FeatureCollection] = None
    
    def load_asset(self) -> bool:
        """
        Load GEE FeatureCollection containing administrative boundaries.
        
        Returns:
            True if successful, False if asset not found or error occurs
        
        Example:
            >>> manager = GEEAssetManager()
            >>> if manager.load_asset():
            ...     print("Asset loaded successfully")
        """
        try:
            # Load the asset
            self.asset = ee.FeatureCollection(self.asset_path)
            
            # Verify asset is not empty by checking size
            size = self.asset.size().getInfo()
            
            if size == 0:
                logger.error(f"Asset at {self.asset_path} is empty")
                self.asset = None
                return False
            
            logger.info(f"Successfully loaded asset with {size} features")
            return True
        
        except ee.EEException as e:
            logger.error(f"Earth Engine error loading asset: {e}")
            self.asset = None
            return False
        
        except Exception as e:
            logger.error(f"Unexpected error loading asset: {e}")
            self.asset = None
            return False
    
    def get_regency_names(self) -> List[str]:
        """
        Extract regency names from loaded GEE FeatureCollection.
        
        Returns:
            List of regency names sorted alphabetically, empty list if error occurs
        
        Example:
            >>> manager = GEEAssetManager()
            >>> manager.load_asset()
            >>> names = manager.get_regency_names()
            >>> print(f"Found {len(names)} regencies")
        """
        if self.asset is None:
            logger.error("Asset not loaded. Call load_asset() first.")
            return []
        
        try:
            # Extract names from the asset
            names = self.asset.aggregate_array(self.name_property).getInfo()
            
            if not names:
                logger.warning("No regency names found in asset")
                return []
            
            # Remove duplicates and sort
            unique_names = sorted(list(set(names)))
            
            logger.info(f"Extracted {len(unique_names)} unique regency names")
            return unique_names
        
        except ee.EEException as e:
            logger.error(f"Earth Engine error extracting names: {e}")
            return []
        
        except Exception as e:
            logger.error(f"Unexpected error extracting names: {e}")
            return []
    
    def get_regency_geometry(self, regency_name: str) -> Optional[ee.Geometry]:
        """
        Get geometry for a selected regency from loaded GEE asset.
        
        Args:
            regency_name: Name of the regency to retrieve
        
        Returns:
            ee.Geometry if regency found, None otherwise
        
        Example:
            >>> manager = GEEAssetManager()
            >>> manager.load_asset()
            >>> geometry = manager.get_regency_geometry("Kabupaten Bandung")
            >>> if geometry:
            ...     print("Geometry retrieved successfully")
        """
        if self.asset is None:
            logger.error("Asset not loaded. Call load_asset() first.")
            return None
        
        try:
            # Filter asset by regency name
            filtered = self.asset.filter(ee.Filter.eq(self.name_property, regency_name))
            
            # Check if any features match
            size = filtered.size().getInfo()
            
            if size == 0:
                logger.error(f"No regency found with name: {regency_name}")
                return None
            
            if size > 1:
                logger.warning(f"Multiple regencies found with name: {regency_name}, using first match")
            
            # Get geometry from first matching feature
            geometry = filtered.first().geometry()
            
            # Verify geometry is valid
            if geometry is None:
                logger.error(f"Geometry is None for regency: {regency_name}")
                return None
            
            logger.info(f"Successfully retrieved geometry for: {regency_name}")
            return geometry
        
        except ee.EEException as e:
            logger.error(f"Earth Engine error retrieving geometry: {e}")
            return None
        
        except Exception as e:
            logger.error(f"Unexpected error retrieving geometry: {e}")
            return None
    
    def convert_to_geodataframe(self, regency_name: str) -> Optional[gpd.GeoDataFrame]:
        """
        Create GeoDataFrame from GEE feature for a selected regency.
        
        Args:
            regency_name: Name of the regency to convert
        
        Returns:
            GeoDataFrame if successful, None otherwise
        
        Example:
            >>> manager = GEEAssetManager()
            >>> manager.load_asset()
            >>> gdf = manager.convert_to_geodataframe("Kabupaten Bandung")
            >>> if gdf is not None:
            ...     print(f"GeoDataFrame created with {len(gdf)} features")
        """
        if self.asset is None:
            logger.error("Asset not loaded. Call load_asset() first.")
            return None
        
        try:
            # Filter asset by regency name
            filtered = self.asset.filter(ee.Filter.eq(self.name_property, regency_name))
            
            # Check if any features match
            size = filtered.size().getInfo()
            
            if size == 0:
                logger.error(f"No regency found with name: {regency_name}")
                return None
            
            # Get the feature as GeoJSON
            feature = filtered.first()
            geojson = feature.getInfo()
            
            if not geojson:
                logger.error(f"Failed to get GeoJSON for regency: {regency_name}")
                return None
            
            # Convert geometry to shapely geometry
            geometry = shape(geojson['geometry'])
            
            # Create GeoDataFrame
            gdf = gpd.GeoDataFrame(
                [geojson['properties']],
                geometry=[geometry],
                crs='EPSG:4326'
            )
            
            logger.info(f"Successfully created GeoDataFrame for: {regency_name}")
            return gdf
        
        except ee.EEException as e:
            logger.error(f"Earth Engine error converting to GeoDataFrame: {e}")
            return None
        
        except Exception as e:
            logger.error(f"Unexpected error converting to GeoDataFrame: {e}")
            return None

#############################  Area of Interest  ###########################
def get_aoi_from_gaul(country="Indonesia", province="Sumatera Selatan"):
    """
    Get Area of Interest geometry from GAUL administrative boundaries.
    
    Parameters:
    -----------
    country : str
        Country name (default: "Indonesia")
    province : str
        Province/state name (default: "Sumatera Selatan")
        
    Returns:
    --------
    ee.Geometry : Area of interest geometry
    """
    ensure_ee_initialized()
    admin = ee.FeatureCollection("FAO/GAUL/2015/level1")
    aoi_fc = admin.filter(ee.Filter.eq('ADM0_NAME', country)).filter(
        ee.Filter.eq('ADM1_NAME', province)
    )
    return aoi_fc.geometry()