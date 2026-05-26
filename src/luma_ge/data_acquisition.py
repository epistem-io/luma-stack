import ee
from datetime import datetime
import logging
from .ee_config import ensure_ee_initialized
import geopandas as gpd
from shapely.geometry import shape
from typing import List, Optional
# Do not initialize Earth Engine at import time. Initialize when an instance is created.

#Configure root for global functions
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
"""
GEE Asset Utility Functions

This module provides utility functions for working with Google Earth Engine assets,
specifically for loading and processing administrative boundary data. 

Classes:
- GEEAssetManager: Manager class for GEE asset operations
"""
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

# Module 1: Cloudless Image Mosaic
## System Response 1.2: Search and Filter Imagery
class Reflectance_Data:
    """Class for fetching and pre-processing Landsat image collection from Google Earth Engine API."""
    #Define the optical datasets. The band reflectances used is from Collection 2 Surface Reflectancce Data
    OPTICAL_DATASETS = {
        'L1_RAW': {
            'collection': 'LANDSAT/LM01/C02/T1',
            'cloud_property': 'CLOUD_COVER_LAND',
            'type': 'landsat_raw',
            'sensor': 'L1',
            'description': 'Landsat 1 Multispectral Scanner Raw Collection'
        },
        'L2_RAW': {
            'collection': 'LANDSAT/LM02/C02/T1',
            'cloud_property': 'CLOUD_COVER_LAND',
            'type': 'landsat_raw',
            'sensor': 'L2',
            'description': 'Landsat 2 Multispectral Scanner Raw Collection'            
        },
        'L3_RAW':{
            'collection': 'LANDSAT/LM03/C02/T1',
            'cloud_property': 'CLOUD_COVER_LAND',
            'type': 'landsat_raw',
            'sensor': 'L3',
            'description': 'Landsat 3 Multispectral Scanner Raw Collection'
        },
        'L4_SR': {
            'collection': 'LANDSAT/LT04/C02/T1_L2',
            'cloud_property': 'CLOUD_COVER_LAND',
            'type': 'landsat_sr',
            'sensor': 'L4',
            'description': 'Landsat 4 Thematic Mapper Surface Reflectance Collection',           
        },
        'L5_SR': {
            'collection': 'LANDSAT/LT05/C02/T1_L2',
            'cloud_property': 'CLOUD_COVER_LAND',
            'type': 'landsat_sr',
            'sensor': 'L5',
            'description': 'Landsat 5 Thematic Mapper Surface Reflectance Collection',
        },
        'L7_SR': {
            'collection': 'LANDSAT/LE07/C02/T1_L2', 
            'cloud_property': 'CLOUD_COVER_LAND',
            'type': 'landsat_sr',
            'sensor': 'L7',
            'description': 'Landsat 7 Enhance Thematic Mapper + Surface Reflectance'
        },
        'L8_SR': {
            'collection': 'LANDSAT/LC08/C02/T1_L2',
            'cloud_property': 'CLOUD_COVER_LAND',
            'type': 'landsat_sr',
            'sensor': 'L8',
            'description': 'Landsat 8 Operational Land Imager Surface Reflectance'
        },
        'L9_SR': {
            'collection': 'LANDSAT/LC09/C02/T1_L2',
            'cloud_property': 'CLOUD_COVER_LAND', 
            'type': 'landsat_sr',
            'sensor': 'L9',
            'description': 'Landsat 9 Operational Land Imager-2 Surface Reflectance'
        }
    }
    #Define the Sentinel-2 datasets. Uses Level-2A Surface Reflectance (Harmonized) collection
    #Cloud masking is performed via Cloud Score+ (GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED)
    S2_DATASETS = {
        'S2_SR': {
            'collection': 'COPERNICUS/S2_SR_HARMONIZED',
            'cloud_property': 'CLOUDY_PIXEL_PERCENTAGE',
            'type': 's2_sr',
            'sensor': 'S2',
            'description': 'Sentinel-2 Level-2A Surface Reflectance (Harmonized)',
        },
    }
    #Define the thermal datasets. The thermal bands used is from Collection 2 Top-of-atmosphere data 
    #The TOA data provide consistent result and contain minimum missing pixel data
    #Note: Landsat 1-3 MSS sensors did not have thermal bands, so they are not included
    THERMAL_DATASETS = {
        'L4_TOA': {
            'collection': 'LANDSAT/LT04/C02/T1_TOA',
            'cloud_property': 'CLOUD_COVER_LAND',
            'type': 'landsat_toa',
            'sensor': 'L4',
            'description': 'Landsat 4 Top-of-atmosphere reflectance',            
        },
        'L5_TOA': {
            'collection': 'LANDSAT/LT05/C02/T1_TOA',
            'cloud_property': 'CLOUD_COVER_LAND',
            'type': 'landsat_toa',
            'sensor': 'L5',
            'description': 'Landsat 5 Top-of-atmosphere reflectance',
        },
        'L7_TOA': {
            'collection': 'LANDSAT/LE07/C02/T1_TOA',
            'cloud_property': 'CLOUD_COVER_LAND',
            'type': 'landsat_toa',
            'sensor': 'L7',
            'description': 'Landsat 7 Top-of-atmosphere reflectance',
        },
        'L8_TOA': {
            'collection': 'LANDSAT/LC08/C02/T1_TOA',
            'cloud_property': 'CLOUD_COVER_LAND',
            'type': 'landsat_toa',
            'sensor': 'L8',
            'description': 'Landsat 8 Top-of-atmosphere reflectance'  
        },
        'L9_TOA': {
            'collection': 'LANDSAT/LC09/C02/T1_TOA',
            'cloud_property': 'CLOUD_COVER_LAND',
            'type': 'landsat_toa',
            'sensor': 'L9',
            'description': 'Landsat 9 Top-of-atmosphere reflectance'      
        }
    }
    #Initialize the class
    def __init__(self, log_level=logging.INFO):
        """
        Initialize the ReflectanceData object and set up a class-specific logger.
        Ensure Earth Engine is initialized lazily (avoids import-time failures).
        """
        # Ensure Earth Engine is initialized when first used (raises helpful error if not)
        ensure_ee_initialized()
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)

        self.logger.info("ReflectanceData initialized.")
    #check if landsat mission contain thermal band
    def has_thermal_capability(self, optical_data):
        """
        Check if a given optical dataset has corresponding thermal bands.
        
        Parameters
        ----------
        optical_data : str
            Optical dataset code (e.g., 'L8_SR', 'L1_RAW')
            
        Returns
        -------
        bool
            True if thermal bands are available, False otherwise
        """
        # Landsat 1-3 MSS sensors did not have thermal bands
        if optical_data in ['L1_RAW', 'L2_RAW', 'L3_RAW']:
            return False
        
        # Check if corresponding thermal dataset exists
        thermal_data = optical_data.replace('_SR', '_TOA')
        return thermal_data in self.THERMAL_DATASETS
    #Function to mask clouds, shadow, and cirrus. Using QA Bands
    def mask_landsat_sr(self, image,cloud_conf_thresh=2, shadow_conf_thresh=2, cirrus_conf_thresh=2):
        """
        Mask clouds, shadows and cirrus for Landsat Collection 2 SR using QA_PIXEL band.
                
        Parameters:
        -----------
        image : ee.Image: Landsat SR image
        cloud_conf_thresh : int. Cloud confidence threshold (0=None, 1=Low, 2=Med, 3=High)
        shadow_conf_thresh : int. Shadow confidence threshold (0=None, 1=Low, 2=Med, 3=High)
        cirrus_conf_thresh : int. Cirrus confidence threshold (0=None, 1=Low, 2=Med, 3=High)

        Returns:
        --------
        ee.Image : Masked image (ee.)

        References
        --------
        https://www.usgs.gov/landsat-missions/landsat-collection-2-quality-assessment-bands 

        Example
        --------
        >>> get_landsat = Reflectance_Data()
        #Implementation on image collection
        >>> collection = (collection.map(lambda img: get_landsat.mask_landsat_sr(img))
        #Implementatio on Image
        >>> masked_image = get_landsat.mask_landsat_sr(image)
        """
        qa = image.select('QA_PIXEL')
        #Deterministic bits
        #fyi, (bit 3 is set to 1) and so on
        cloud_bit = 1 << 3
        shadow_bit = 1 << 4
        cirrus_bit = 1 << 2
        #bitwise operations to get the masks
        cloud_mask = qa.bitwiseAnd(cloud_bit).eq(0)
        shadow_mask = qa.bitwiseAnd(shadow_bit).eq(0)
        cirrus_mask = qa.bitwiseAnd(cirrus_bit).eq(0)
        #Confidence bits ---
        cloud_conf = qa.rightShift(8).bitwiseAnd(3)     # Bits 8–9
        shadow_conf = qa.rightShift(10).bitwiseAnd(3)   # Bits 10–11
        cirrus_conf = qa.rightShift(14).bitwiseAnd(3)   # Bits 14–15
        #Keep pixels below thresholds
        conf_mask = (cloud_conf.lt(cloud_conf_thresh)
                        .And(shadow_conf.lt(shadow_conf_thresh))
                        .And(cirrus_conf.lt(cirrus_conf_thresh)))
        #Final mask
        final_mask = cloud_mask.And(shadow_mask).And(cirrus_mask).And(conf_mask)
        return image.updateMask(final_mask).copyProperties(image, image.propertyNames())
    #Functions to rename Landsat bands 
    def rename_landsat_bands(self, image, sensor_type):
        """
        Standardize Landsat Surface Reflectance (SR) band names based on sensor type. From 'SR_B*' or 'B*' to 'NIR', 'GREEN', etc.
        Used in get multispectral data function

        Parameters
        ----------
        image : ee.Image. Landsat SR image
        sensor_type : str. Sensor type ('L5', 'L7', 'L8', 'L9')

        Returns
        -------
        ee.Image : Image with standardized band names

        Example
        -------
        >>> rd = Reflectance_Data()
        >>> img8 = ee.Image('LANDSAT/LC08/C02/T1_L2/LC08_044034_20200716')
        >>> renamed8 = rd.rename_landsat_bands(img8, 'L8')
        """
        if sensor_type in ['L4','L5', 'L7']:
            # Landsat 5/7 SR bands
            return image.select(
                ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7'], 
                ['BLUE', 'GREEN', 'RED', 'NIR', 'SWIR1', 'SWIR2']
            )
        elif sensor_type in ['L1', 'L2', 'L3']:
            return image.select(
                ['B4', 'B5', 'B6', 'B7'],
                ['GREEN', 'RED', 'NIR1', 'NIR2']
            )
        elif sensor_type in ['L8', 'L9']:
            # Landsat 8/9 SR bands
            return image.select(
                ['SR_B1','SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7'], 
                ['AEROSOL','BLUE', 'GREEN', 'RED', 'NIR', 'SWIR1', 'SWIR2']
            )
        else:
            raise ValueError(f"Unsupported sensor type for SR data: {sensor_type}")
    #function to implement Landsat Collection 2 Tier 1 SR scale factor
    def apply_scale_factors(self, image):
        """
        Apply Landsat collection 2 scalling factors using the following formula: Digital Number (DN) * scale_factor + offset.
        Allowing the user to get the data back to its original floating point value, a scale factor and offset.

        Parameters
        ----------
        image : ee.Image (Landsat SR image) with DN value
        sensor_type : str. Sensor type ('L4', 'L5', 'L7', 'L8', 'L9')

        Returns
        -------
        ee.Image : Image with floating point, corresponding to surface reflectance value
        
        References
        -------
        https://www.usgs.gov/faqs/how-do-i-use-a-scale-factor-landsat-level-2-science-products
        
        Example
        --------
        >>> get_landsat = Reflectance_Data()
        #Implementation on image collection
        >>> collection = (collection.map(lambda img: get_landsat.apply_scale_factors(img))
        #Implementatio on Image
        >>> masked_image = get_landsat.apply_scale_factors(image)
        """           
        optical_bands = image.select('SR_B.').multiply(0.0000275).add(-0.2)
        #thermal_bands = image.select('ST_B.*').multiply(0.00341802).add(149.0)
        return image.addBands(optical_bands, None, True)
    #Cloud masking for Sentinel-2 via Cloud Score+
    #Cloud Score+ provide improve cloud masking approach compared with quality control band pixel based masking
    def mask_s2_clouds(self, image: ee.Image, threshold: float = 0.60) -> ee.Image:
        """
        Mask cloudy, hazy, and cirrus-affected pixels in a Sentinel-2 image using Cloud Score+.

        The image must already have the ``cs_cdf`` band joined from the
        ``GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED`` collection (e.g. via
        ``linkCollection``).  Pixels whose ``cs_cdf`` value is below
        *threshold* are masked out; all original image properties are
        preserved.
        further reading: https://medium.com/google-earth/all-clear-with-cloud-score-bd6ee2e2235e

        Parameters
        ----------
        image : ee.Image
            Sentinel-2 SR image with the ``cs_cdf`` band already joined.
        threshold : float, optional
            Minimum ``cs_cdf`` score a pixel must meet to be considered
            clear.  Values range from 0 (fully occluded) to 1 (fully
            clear).  Default is ``0.60``.

        Returns
        -------
        ee.Image
            Cloud-masked image with all original properties copied, or
            ``None`` if an error occurs.

        References
        ----------
        https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_CLOUD_SCORE_PLUS_V1_S2_HARMONIZED

        Example
        -------
        >>> rd = Reflectance_Data()
        >>> masked = rd.mask_s2_clouds(image, threshold=0.65)
        >>> # Map over a collection
        >>> masked_col = collection.map(lambda img: rd.mask_s2_clouds(img))
        """
        try:
            masked = image.updateMask(image.select('cs_cdf').gte(threshold))
            return masked.copyProperties(image, image.propertyNames())
        except ee.EEException as e:
            self.logger.error(f"EEException in mask_s2_clouds: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error in mask_s2_clouds: {e}")
            return None

    #Band renaming for Sentinel-2. Standarized band name for later use
    def rename_s2_bands(self, image: ee.Image) -> ee.Image:
        """
        Rename Sentinel-2 SR bands to standardized semantic names.

        Maps the raw Sentinel-2 band identifiers to a common naming
        convention shared across the luma-ge pipeline, enabling downstream
        modules to reference bands by semantic name rather than sensor-
        specific identifiers.

        Parameters
        ----------
        image : ee.Image
            Sentinel-2 SR image containing the source bands
            ``['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12']``.

        Returns
        -------
        ee.Image
            Image with bands renamed to
            ``['AEROSOL','BLUE','GREEN','RED','RED_EDGE1','RED_EDGE2',
            'RED_EDGE3','NIR','NIR_NARROW','SWIR1','SWIR2']``.

        Example
        -------
        >>> rd = Reflectance_Data()
        >>> renamed = rd.rename_s2_bands(image)
        >>> # Map over a collection
        >>> renamed_col = collection.map(lambda img: rd.rename_s2_bands(img))
        """
        return image.select(
            ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12'],
            ['AEROSOL', 'BLUE', 'GREEN', 'RED', 'RED_EDGE1', 'RED_EDGE2', 'RED_EDGE3',
             'NIR', 'NIR_NARROW', 'SWIR1', 'SWIR2']
        )

    #Scalled the sentinel 2 band according to the data scale factor
    def apply_s2_scale_factors(self, image: ee.Image) -> ee.Image:
        """
        Convert Sentinel-2 DN values to surface reflectance by applying a scale factor.

        Multiplies all optical bands by ``0.0001``, replacing the original
        DN bands in-place.  Non-optical bands (e.g. ``cs_cdf``, QA bands)
        are preserved unchanged.

        Parameters
        ----------
        image : ee.Image
            Sentinel-2 SR image with DN optical bands
            ``['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12']``

        Returns
        -------
        ee.Image
            Image with optical bands scaled to the ``[0, 1]`` reflectance
            range, replacing the original DN bands.  All other bands and
            image properties are preserved.

        References
        ----------
        https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED

        Example
        -------
        >>> rd = Reflectance_Data()
        >>> scaled = rd.apply_s2_scale_factors(image)
        >>> # Map over a collection
        >>> scaled_col = collection.map(lambda img: rd.apply_s2_scale_factors(img))
        """
        optical_bands = image.select('B.*').multiply(0.0001)
        return image.addBands(optical_bands, None, True)

    #Sharpening the 20m bands using multi resolution analysis, namely High Pass Filter (HPF)
    def sharpen_s2_bands(self, image: ee.Image) -> ee.Image:
        """
        Sharpen 20 m Sentinel-2 bands to 10 m using High-Pass Filter (HPF) approach

        Expects an image with standardized band names already applied (output
        of :meth:`rename_s2_bands`).  The NIR band (10 m native) is used as
        the high-resolution panchromatic reference.  A high-frequency detail
        component is extracted from NIR via a Gaussian blur and added to each
        bilinearly-resampled 20 m band.  The four 10 m native bands
        (``BLUE``, ``GREEN``, ``RED``, ``NIR``) are passed through unchanged.

        Algorithm
        ---------
        1. Extract HPF component from NIR (10 m reference)::

               hpf = NIR - gaussian_smooth(NIR)   # kernel: radius=1, sigma=1

        2. For each 20 m band::

               sharpened = resample(band_20m, bilinear) + hpf

        3. 10 m native bands pass through unchanged.

        Parameters
        ----------
        image : ee.Image
            Sentinel-2 SR image with standardized band names
            ``['BLUE','GREEN','RED','RED_EDGE1','RED_EDGE2','RED_EDGE3',
            'NIR','NIR_NARROW','SWIR1','SWIR2']`` already applied.

        Returns
        -------
        ee.Image
            Image with all 10 bands at 10 m resolution and all original
            image properties preserved, or ``None`` if an error occurs.

        References
        ----------
        https://developers.google.com/earth-engine/guides/resample

        Example
        -------
        >>> rd = Reflectance_Data()
        >>> sharpened = rd.sharpen_s2_bands(renamed_image)
        >>> # Map over a collection
        >>> sharpened_col = collection.map(lambda img: rd.sharpen_s2_bands(img))
        """
        try:
            # --- 1. Extract high-frequency detail from NIR (10 m reference) ---
            nir = image.select('NIR')
            gaussian_kernel = ee.Kernel.gaussian(radius=1, sigma=1, units='pixels')
            nir_smooth = nir.reduceNeighborhood(
                reducer=ee.Reducer.mean(),
                kernel=gaussian_kernel,
            )
            hpf = nir.subtract(nir_smooth)

            # --- 2. Pass 10 m native bands through unchanged ---
            bands_10m = image.select(['BLUE', 'GREEN', 'RED', 'NIR'])

            # --- 3. Resample each 20 m band to 10 m and inject HPF detail ---
            bands_20m_names = ['RED_EDGE1', 'RED_EDGE2', 'RED_EDGE3', 'NIR_NARROW', 'SWIR1', 'SWIR2']
            sharpened_bands = image.select(bands_20m_names).resample('bilinear').add(hpf)

            # --- 4. Recombine all bands and preserve image properties ---
            result = bands_10m.addBands(sharpened_bands)
            return result.copyProperties(image, image.propertyNames())

        except ee.EEException as e:
            self.logger.error(f"EEException in sharpen_s2_bands: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error in sharpen_s2_bands: {e}")
            return None

    #Core methods to retrieve sentinel 2 data
    def get_s2_optical_data(self, aoi, start_date, end_date, cloud_cover: int = 30, cs_threshold: float = 0.60,
        verbose: bool = True,
        compute_detailed_stats: bool = True,):
        """
        Retrieve a filtered and pre-processed Sentinel-2 image collection.

        Fetches ``COPERNICUS/S2_SR_HARMONIZED`` for the given AOI and date
        range, joins Cloud Score+ scores, applies cloud masking, scale
        factors, and band renaming, then returns the collection together
        with a statistics dictionary.

        Parameters
        ----------
        aoi : ee.Geometry or ee.FeatureCollection
            Area of interest.  Must not be ``None``.
        start_date : int or str
            Start date as ``'YYYY-MM-DD'`` or a 4-digit year (expanded to
            ``YYYY-01-01``).
        end_date : int or str
            End date as ``'YYYY-MM-DD'`` or a 4-digit year (expanded to
            ``YYYY-12-31``).
        cloud_cover : int, optional
            Maximum ``CLOUDY_PIXEL_PERCENTAGE`` for pre-filtering images.
            Default is ``30``.
        cs_threshold : float, optional
            Minimum ``cs_cdf`` score for pixel-level cloud masking via
            Cloud Score+.  Default is ``0.60``.
        verbose : bool, optional
            Log collection details when ``True``.  Default is ``True``.
        compute_detailed_stats : bool, optional
            Compute full statistics (cloud cover range, dates, etc.) when
            ``True``.  Default is ``True``.

        Returns
        -------
        tuple : (ee.ImageCollection, dict)
            * ``ee.ImageCollection`` — filtered and pre-processed collection
              with standardized band names
              ``['AEROSOL','BLUE','GREEN','RED','RED_EDGE1','RED_EDGE2',
              'RED_EDGE3','NIR','NIR_NARROW','SWIR1','SWIR2']``.
            * ``dict`` — statistics and metadata with keys:
              ``dataset``, ``sensor``, ``date_range_requested``,
              ``cloud_cover_threshold``, ``initial_collection``,
              ``filtered_collection``, ``detailed_stats_computed``.

        Raises
        ------
        ValueError
            If ``aoi`` is ``None``.

        References
        ----------
        https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED

        Example
        -------
        >>> rd = Reflectance_Data()
        >>> collection, stats = rd.get_s2_optical_data(aoi, 2020, 2023)
        >>> collection, stats = rd.get_s2_optical_data(aoi, '2021-01-01', '2022-12-31', cloud_cover=20)
        """
        # Validate AOI before any GEE call (Requirement 9.2)
        if aoi is None:
            raise ValueError("aoi must not be None — provide an ee.Geometry or ee.FeatureCollection.")

        # Helper: expand 4-digit year to full date string (Requirement 6.2)
        def parse_year_or_date(date_input, is_start=True):
            if isinstance(date_input, int):
                return f"{date_input}-01-01" if is_start else f"{date_input}-12-31"
            elif isinstance(date_input, str):
                if len(date_input) == 4 and date_input.isdigit():
                    return f"{date_input}-01-01" if is_start else f"{date_input}-12-31"
                else:
                    return date_input
            else:
                raise ValueError("Date must be either YYYY or YYYY-MM-DD format")

        start_date = parse_year_or_date(start_date, is_start=True)
        end_date   = parse_year_or_date(end_date, is_start=False)

        config = self.S2_DATASETS['S2_SR']

        try:
            if verbose:
                self.logger.info(f"Starting data fetch for {config['description']}")
                self.logger.info(f"Date range: {start_date} to {end_date}")
                self.logger.info(f"Cloud cover threshold (image-level): {cloud_cover}%")
                self.logger.info(f"Cloud Score+ pixel threshold: {cs_threshold}")
                if not compute_detailed_stats:
                    self.logger.info("Detailed statistics will not be computed")

            # --- Initial collection (before cloud cover filter) ---
            initial_collection = (
                ee.ImageCollection(config['collection'])
                .filterBounds(aoi)
                .filterDate(start_date, end_date)
            )
            stats_object = Reflectance_Stats()
            initial_stats = stats_object.get_collection_statistics(
                initial_collection,
                compute_detailed_stats,
                cloud_property='CLOUDY_PIXEL_PERCENTAGE',
            )

            if verbose and compute_detailed_stats and initial_stats.get('total_images', 0) > 0:
                self.logger.info(
                    f"Initial collection (before cloud filtering): "
                    f"{initial_stats['total_images']} images"
                )
                self.logger.info(
                    f"Date range of available images: {initial_stats['date_range']}"
                )

            #Filter by cloud cover, limit scene collection, join the main collection with Cloud Score+
            collection = (
                initial_collection
                .filter(ee.Filter.lt(config['cloud_property'], cloud_cover))
                .limit(350)
            )

            #Join Cloud Score+ 'cs_cdf' band via linkCollection
            cs_plus = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED')
            collection = collection.linkCollection(cs_plus, ['cs_cdf'])

            #Apply processing pipeline: mask → scale → rename
            collection = (
                collection
                .map(lambda img: self.mask_s2_clouds(img, threshold=cs_threshold))
                .map(lambda img: self.apply_s2_scale_factors(img))
                .map(lambda img: self.rename_s2_bands(img))
            )

            #Filtered stats (after cloud cover filter + processing)
            filtered_stats = stats_object.get_collection_statistics(
                collection,
                compute_detailed_stats,
                cloud_property='CLOUDY_PIXEL_PERCENTAGE',
            )

            #Verbose logging
            if verbose and compute_detailed_stats:
                if filtered_stats.get('total_images', 0) > 0:
                    self.logger.info(
                        f"After cloud filtering (<{cloud_cover}%): "
                        f"{filtered_stats['total_images']} images"
                    )
                    self.logger.info(
                        f"Cloud cover of selected images: "
                        f"{filtered_stats['cloud_cover']['min']:.1f}% - "
                        f"{filtered_stats['cloud_cover']['max']:.1f}%"
                    )
                    self.logger.info(
                        f"Average cloud cover: {filtered_stats['cloud_cover']['mean']:.1f}%"
                    )
                    if len(filtered_stats['individual_dates']) <= 20:
                        self.logger.info(
                            f"Image dates: {', '.join(filtered_stats['individual_dates'])}"
                        )
                    else:
                        self.logger.info(
                            f"Images span from "
                            f"{min(filtered_stats['individual_dates'])} "
                            f"to {max(filtered_stats['individual_dates'])}"
                        )
                else:
                    self.logger.warning(
                        f"No images found after cloud filtering (CLOUDY_PIXEL_PERCENTAGE < {cloud_cover}%)"
                    )
                    if initial_stats.get('total_images', 0) > 0:
                        self.logger.info(
                            "Consider increasing the cloud_cover threshold or expanding the date range. "
                            f"Available cloud cover range: "
                            f"{initial_stats['cloud_cover']['min']:.1f}% - "
                            f"{initial_stats['cloud_cover']['max']:.1f}%"
                        )
            elif verbose:
                self.logger.info(
                    "Filtered collection created (use compute_detailed_stats=True for more information)"
                )

            return collection, {
                'dataset': config['description'],
                'sensor': config['sensor'],
                'date_range_requested': f"{start_date} to {end_date}",
                'cloud_cover_threshold': cloud_cover,
                'initial_collection': initial_stats,
                'filtered_collection': filtered_stats,
                'detailed_stats_computed': compute_detailed_stats,
            }

        except ee.EEException as e:
            self.logger.error(f"EEException in get_s2_optical_data: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error in get_s2_optical_data: {e}")
            raise

    #core method to retrieve landsat data
    def get_optical_data(self, aoi, start_date, end_date, optical_data='L8_SR',
                        cloud_cover=30,
                        verbose=True, compute_detailed_stats=True):
        """
        Get multispectral image collection for Landsat 1-9 with detailed information logging.

        Parameters
        ----------
        aoi :  ee.FeatureCollection. Area of interest.
        start_date : str. Start date in format 'YYYY-MM-DD' or year.
        end_date : str. End date in format 'YYYY-MM-DD' or year.
        optical_data : str. Dataset type: i.e 'L5_SR', 'L7_SR', 'L8_SR', 'L9_SR'.
        cloud_cover : int. Maximum cloud cover percentage on land (default: 30).
        verbose : bool. Print detailed information about the collection (default: True).
        compute_detailed_stats : bool
            If True, compute detailed statistics 
            If False, return only basic information (default: True).

        Returns
        -------
        tuple : (ee.ImageCollection, dict)
            Filtered and preprocessed image collection with statistics.
        
        References
        -------
        https://developers.google.com/earth-engine/datasets/catalog/landsat

        Example
        --------
        >>> get_landsat = Reflectance_Data()
        >>> collection, stats = get_landsat.get_multispectral_data(aoi, 2020, 2023, 'L8_SR', 30, True, True)
        >>> # With AOI cloud filtering
        >>> collection, stats = get_landsat.get_multispectral_data(aoi, 2020, 2023, 'L8_SR', cloud_cover=30)
        """
        #Helper function so that the user only input year or specific date range
        def parse_year_or_date(date_input, is_start=True):
            if isinstance(date_input, int):  # User gave integer year like 2024
                return f"{date_input}-01-01" if is_start else f"{date_input}-12-31"
            elif isinstance(date_input, str):
                if len(date_input) == 4 and date_input.isdigit():
                    return f"{date_input}-01-01" if is_start else f"{date_input}-12-31"
                else:
                    return date_input  # Already full date
            else:
                raise ValueError("Date must be either YYYY or YYYY-MM-DD format")
        # Parse inputs (handles both year and full date)
        start_date = parse_year_or_date(start_date, is_start=True)
        end_date   = parse_year_or_date(end_date, is_start=False)

        if optical_data not in self.OPTICAL_DATASETS:
            raise ValueError(f"optical_data must be one of: {list(self.OPTICAL_DATASETS.keys())}")

        config = self.OPTICAL_DATASETS[optical_data]

        # Warn if using legacy Landsat 1-3 data
        if optical_data in ['L1_RAW', 'L2_RAW', 'L3_RAW']:
            self.logger.warning(
                f"WARNING: You are using {config['description']}"
                f"Data avaliability and coverage for Landsat 1-3 (1972 - 1984) is very limited in some locations."
                f"Consider using Landsat 4-9 for better data coverage."
                f"Moreover, Landsat 1-3 data are DN values, representing scaled, calibrated at-sensor radiance. Further pre-processing are required. "
            )

        #Use verbose to import detailed logging information
        if verbose:
            self.logger.info(f"Starting data fetch for {config['description']}")
            self.logger.info(f"Date range: {start_date} to {end_date}")
            self.logger.info(f"Cloud cover threshold: {cloud_cover}%")
            if not compute_detailed_stats:
                self.logger.info("detailed statistics will not be computed")

        #Initial collection
        initial_collection = (ee.ImageCollection(config['collection'])
                            .filterBounds(aoi)
                            .filterDate(start_date, end_date))
        #initial_stats = self.get_collection_statistics(initial_collection, compute_detailed_stats)
        stats_object = Reflectance_Stats()
        initial_stats = stats_object.get_collection_statistics(initial_collection, compute_detailed_stats)
        if verbose and compute_detailed_stats and initial_stats.get('total_images', 0) > 0:
            self.logger.info(f"Initial collection (before cloud filtering): {initial_stats['total_images']} images")
            self.logger.info(f"Date range of available images: {initial_stats['date_range']}")

        #Collection after cloud cover filter
        collection = initial_collection.filter(ee.Filter.lt(config['cloud_property'], cloud_cover))
        #Limit to 250 scenes to avoid Earth Engine computation limits
        collection = collection.limit(250)
        filtered_stats = stats_object.get_collection_statistics(collection, compute_detailed_stats)
        #Computing image statistics
        if verbose and compute_detailed_stats:
            if filtered_stats.get('total_images', 0) > 0:
                self.logger.info(f"After cloud filtering (<{cloud_cover}%): {filtered_stats['total_images']} images")
                self.logger.info(f"Cloud cover of selected images: "
                                f"{filtered_stats['cloud_cover']['min']:.1f}% - "
                                f"{filtered_stats['cloud_cover']['max']:.1f}%")
                self.logger.info(f"Average cloud cover: {filtered_stats['cloud_cover']['mean']:.1f}%")
                if len(filtered_stats['individual_dates']) <= 20:
                    self.logger.info(f"Image dates: {', '.join(filtered_stats['individual_dates'])}")
                else:
                    self.logger.info(f"Images span from {min(filtered_stats['individual_dates'])} "
                                    f"to {max(filtered_stats['individual_dates'])}")

                if filtered_stats['path_row_tiles']:
                    path_row_str = ', '.join([f"{p}/{r}" for p, r in filtered_stats['path_row_tiles'][:10]])
                    if len(filtered_stats['path_row_tiles']) > 10:
                        path_row_str += f" ... (+{len(filtered_stats['path_row_tiles']) - 10} more)"
                    self.logger.info(f"Path/Row tiles: {path_row_str}")
            else:
                self.logger.warning(f"No images found matching criteria (cloud cover < {cloud_cover}%)")
                if initial_stats.get('total_images', 0) > 0:
                    self.logger.info(f"Consider increasing cloud cover threshold, or expand the year of acqusition "
                                    f"Available range: {initial_stats['cloud_cover']['min']:.1f}% - "
                                    f"{initial_stats['cloud_cover']['max']:.1f}%")
        elif verbose:
            self.logger.info("Filtered collection created (use compute_detailed_stats=True for more information)")

        #Apply masking and band renaming to image collection after filtering
        collection = (collection
                    .map(lambda img: self.mask_landsat_sr(img))
                    .map(lambda img: self.apply_scale_factors(img))
                    .map(lambda img: self.rename_landsat_bands(img, config['sensor'])))

        #Return results
        return collection, {
            'dataset': config['description'],
            'sensor': config['sensor'],
            'date_range_requested': f"{start_date} to {end_date}",
            'cloud_cover_threshold': cloud_cover,
            'initial_collection': initial_stats,
            'filtered_collection': filtered_stats,
            'detailed_stats_computed': compute_detailed_stats
        }
    #TOA-based Thermal Bands
    def get_thermal_bands(self, aoi, start_date, end_date, thermal_data = 'L8_TOA', cloud_cover=30,
                        verbose=True, compute_detailed_stats=True):
        """
        Get the thermal bands from landsat TOA data
    
        Parameters
        ----------
        aoi :  ee.FeatureCollection. Area of interest.
        start_date : str. Start date in format 'YYYY-MM-DD' or year.
        end_date : str. End date in format 'YYYY-MM-DD' or year.
        optical_data : str. Dataset type: 'L5_SR', 'L7_SR', 'L8_SR', 'L9_SR'.
        cloud_cover : int. Maximum cloud cover percentage on land (default: 30).
        verbose : bool. Print detailed information about the collection (default: True).
        compute_detailed_stats : bool
            If True, compute detailed statistics 
            If False, return only basic information (default: True).
            
        Returns
        -------
        tuple : (ee.ImageCollection, dict)
            Filtered and preprocessed image collection with statistics.
        References
        -------
        https://developers.google.com/earth-engine/datasets/catalog/landsat

        Example
        --------
        >>> get_landsat = Reflectance_Data()
        >>> collection, stats = get_landsat.get_thermal_data(aoi, 2020, 2023, 'L8_TOA', 30, True, True)
        >>> # With AOI cloud filtering
        >>> collection, stats = get_landsat.get_thermal_data(aoi, 2020, 2023, 'L8_TOA', cloud_cover=30)
        """
        #Helper function to parse the date so that the user only input year 
        def parse_year_or_date(date_input, is_start=True):
            if isinstance(date_input, int):  # User gave integer year like 2024
                return f"{date_input}-01-01" if is_start else f"{date_input}-12-31"
            elif isinstance(date_input, str):
                if len(date_input) == 4 and date_input.isdigit():
                    return f"{date_input}-01-01" if is_start else f"{date_input}-12-31"
                else:
                    return date_input  # Already full date
            else:
                raise ValueError("Date must be either YYYY or YYYY-MM-DD format")
        # Parse inputs (handles both year and full date)
        start_date = parse_year_or_date(start_date, is_start=True)
        end_date   = parse_year_or_date(end_date, is_start=False)
        #Helper function to rename the bands
        def rename_thermal_band(img):
            sensor = config['sensor']
            thermal_band_map = {
                'L4': ['B6'],
                'L5': ['B6'],
                'L7': ['B6_VCID_2'],
                'L8': ['B10'],
                'L9': ['B10']
            }
            band_names = thermal_band_map.get(sensor, [])
            if len(band_names) == 1:
                return img.select(band_names).rename(['THERMAL'])
            else:
                return img
        #The core function for thermal bnd
        if thermal_data not in self.THERMAL_DATASETS:
                raise ValueError(f"thermal_data must be one of: {list(self.THERMAL_DATASETS.keys())}")

        config = self.THERMAL_DATASETS[thermal_data]
        #Decide which thermal band to select
        sensor = config['sensor']
        if sensor in ['L4', 'L5']:
            thermal_band = 'B6'
        elif sensor == 'L7':
            thermal_band = 'B6_VCID_2'
        elif sensor in ['L8', 'L9']:
            thermal_band = 'B10'
        else:
            raise ValueError(f"Unsupported sensor type: {sensor}")
        stats = Reflectance_Stats()
        #Logging
        if verbose:
            self.logger.info(f"Starting thermal data fetch for {config['description']}")
            self.logger.info(f"Date range: {start_date} to {end_date}")
            self.logger.info(f"Cloud cover threshold: {cloud_cover}%")
            if not compute_detailed_stats:
                self.logger.info("Fast mode enabled - detailed statistics will not be computed")
        #Initial collection
        initial_collection = (ee.ImageCollection(config['collection'])
                            .filterBounds(aoi)
                            .filterDate(start_date, end_date))
        initial_stats = stats.get_collection_statistics(initial_collection, compute_detailed_stats)

        if verbose and compute_detailed_stats and initial_stats.get('total_images', 0) > 0:
            self.logger.info(f"Initial collection (before cloud filtering): {initial_stats['total_images']} images")
            self.logger.info(f"Date range of available images: {initial_stats['date_range']}")
        #Apply cloud cover filter
        collection = initial_collection.filter(ee.Filter.lt(config['cloud_property'], cloud_cover))
        #Limit to 250 scenes to avoid Earth Engine computation limits
        collection = collection.limit(250)
        filtered_stats = stats.get_collection_statistics(collection, compute_detailed_stats)
        if verbose and compute_detailed_stats:
            if filtered_stats.get('total_images', 0) > 0:
                self.logger.info(f"After cloud filtering (<{cloud_cover}%): {filtered_stats['total_images']} images")
                self.logger.info(f"Cloud cover range: {filtered_stats['cloud_cover']['min']:.1f}% - {filtered_stats['cloud_cover']['max']:.1f}%")
                self.logger.info(f"Average cloud cover: {filtered_stats['cloud_cover']['mean']:.1f}%")
            else:
                self.logger.warning(f"No images found matching criteria (cloud cover < {cloud_cover}%)")
        elif verbose:
            self.logger.info("Filtered collection created (use compute_detailed_stats=True for detailed info)")
        
        #Apply masking (QA-based)
        collection = collection.map(lambda img: self.mask_landsat_sr(img))
        collection = collection.select(thermal_band)
        collection = collection.map(rename_thermal_band)

        #Return collection and stats
        return collection, {
            'dataset': config['description'],
            'sensor': sensor,
            'thermal_band': thermal_band,
            'date_range_requested': f"{start_date} to {end_date}",
            'cloud_cover_threshold': cloud_cover,
            'initial_collection': initial_stats,
            'filtered_collection': filtered_stats,
            'detailed_stats_computed': compute_detailed_stats
        }
class Reflectance_Stats:
    """
    Class for fetching image collection statistics from Reflectance_Data class
    """
    def __init__(self, log_level=logging.INFO):
        """
        Initialize the ReflectanceStats object and set up a class-specific logger.
        Ensure Earth Engine is initialized lazily (avoids import-time failures).
        """
        # Ensure Earth Engine is initialized when first used (raises helpful error if not)
        ensure_ee_initialized()
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)

        self.logger.info("Reflectance Stats initialized.")
    def get_collection_statistics(self, collection, compute_stats=True, print_report=False, cloud_property='CLOUD_COVER_LAND'):
        """
        Get comprehensive statistics about an Earth Engine image collection retrival.

        Parameters
        ----------
        collection : ee.ImageCollection
            The Earth Engine ImageCollection (from get multispectral function).
        compute_stats : bool, optional
            If True (default) the function will call ``getInfo()`` to compute
            detailed statistics (counts, cloud cover numbers, dates, WRS tiles).
            If False the function returns a minimal, server-side friendly
            summary object and avoids client-side network calls.
        print_report : bool, optional
            If True the function will print formatted report of collection retrival
            (default: False).
        cloud_property : str, optional
            The image property name used to retrieve cloud cover values
            (default: ``'CLOUD_COVER_LAND'`` for Landsat). Pass
            ``'CLOUDY_PIXEL_PERCENTAGE'`` for Sentinel-2 collections.

        Returns
        -------
        dict
            A dictionary containing either detailed statistics (when
            ``compute_stats=True``) or a lightweight summary with the server
            side objects (when ``compute_stats=False``). Typical keys when
            detailed stats are returned:
            - 'total_images'
            - 'date_range'
            - 'cloud_cover' (dict with min/max/mean/values)
            - 'aoi_cloud_cover' (dict with min/max/mean/values, if available)
            - 'path_row_tiles'
            - 'unique_tiles'
            - 'individual_dates'
            - 'Scene_ids'

        Example
        -------
        >>> stats = Reflectance_Stats().get_collection_statistics(collection, compute_stats=True)
        >>> print(stats['total_images'], stats['date_range'])
        """
        #Get the number of image used 
        try:
            size = collection.size()
            if compute_stats:
                total_images = size.getInfo() #Client side operation, produce number of image collection
                if total_images > 0:
                    #Get the cloud cover percentage, and image aqcusition date
                    cloud_values = collection.aggregate_array(cloud_property).getInfo()
                    dates = collection.aggregate_array('system:time_start').getInfo()
                    dates_readable = [datetime.fromtimestamp(d/1000).strftime('%Y-%m-%d') for d in dates]
                    date_range = f"{min(dates_readable)} to {max(dates_readable)}"
                    #add lines to identify collection ID
                    scene_id = collection.aggregate_array('system:index').getInfo()
                    date_range = f"{min(dates_readable)} to {max(dates_readable)}"
                    #Get information regarding image's WRS path and row
                    try:
                        first_img = collection.first()
                        has_path_row = first_img.propertyNames().contains('WRS_PATH').getInfo()
                        if has_path_row:
                            paths = collection.aggregate_array('WRS_PATH').getInfo()
                            rows = collection.aggregate_array('WRS_ROW').getInfo()
                            path_rows = list(set(zip(paths, rows)))
                            path_rows.sort()
                        else:
                            path_rows = []
                    except Exception:
                        path_rows = []
                    #Image collections information 
                    stats = {
                        'total_images': total_images,
                        'date_range': date_range,
                        'cloud_cover': {
                            'min': min(cloud_values) if cloud_values else None,
                            'max': max(cloud_values) if cloud_values else None,
                            'mean': sum(cloud_values) / len(cloud_values) if cloud_values else None,
                            'values': cloud_values
                        },
                        'path_row_tiles': path_rows,
                        'unique_tiles': len(path_rows),
                        'individual_dates': dates_readable,
                        'Scene_ids': scene_id
                    }
                    if print_report:
                        self.print_collection_report(stats)
                else:
                    stats = {
                        'total_images': 0,
                        'date_range': "No images found",
                        'cloud_cover': {'min': None, 'max': None, 'mean': None, 'values': []},
                        'path_row_tiles': [],
                        'unique_tiles': 0,
                        'individual_dates': [],
                        'Scene_ids':[]
                    }
                    if print_report:
                        print("="*60)
                        print("           LANDSAT COLLECTION REPORT")
                        print("="*60)
                        print("No images found matching the specified criteria.")
                        print("="*60)
            else:
                stats = {
                    'total_images': 'computed_on_demand',
                    'size_object': size,
                    'collection_object': collection,
                    'computed': False
                }
            return stats

        except Exception as e:
            self.logger.error(f"Error getting collection statistics: {str(e)}")
            return {'error': str(e)} 
    def print_collection_report(self, stats):
        """
        Print a formatted report of collection statistics.
        
        Parameters
        ----------
        stats : dict
            Statistics dictionary from get_collection_statistics
        """
        print("="*60)
        print("           Landsat Data Collection Retrival Report")
        print("="*60)
        #Basic Information
        print(f"Total Images Found: {stats['total_images']}")
        print(f"Date Range: {stats['date_range']}")
        print(f"Unique WRS Tiles: {stats['unique_tiles']}")
        print()
        #Cloud Cover Statistics
        print("Scene Cloud Cover Statistics:")
        print("-" * 30)
        if stats['cloud_cover']['mean'] is not None:
            print(f"Average Cloud Cover: {stats['cloud_cover']['mean']:.1f}%")
            print(f"Minimum Cloud Cover: {stats['cloud_cover']['min']:.1f}%")
            print(f"Maximum Cloud Cover: {stats['cloud_cover']['max']:.1f}%")
        else:
            print("No cloud cover data available")
        print()
        #WRS Path/Row Information
        if stats['path_row_tiles']:
            print("WRS Path/Row Tiles:")
            print("-" * 30)
            if len(stats['path_row_tiles']) <= 10:
                for path, row in stats['path_row_tiles']:
                    print(f"Path {path:03d}/Row {row:03d}")
            else:
                # Show first 8 tiles
                for path, row in stats['path_row_tiles'][:8]:
                    print(f"Path {path:03d}/Row {row:03d}")
                print(f"... and {len(stats['path_row_tiles']) - 8} more tiles")
            print()
        #Available Dates
        if stats['individual_dates']:
            print("Available Acqusition Date:")
            print("-" * 30)
            if len(stats['individual_dates']) <= 20:
                # Group dates by month for better readability
                dates_by_month = {}
                for date in sorted(stats['individual_dates']):
                    month_key = date[:7]  # YYYY-MM
                    if month_key not in dates_by_month:
                        dates_by_month[month_key] = []
                    dates_by_month[month_key].append(date[8:])  # Just the day
                
                for month, days in dates_by_month.items():
                    print(f"{month}: {', '.join(days)}")
            else:
                print(f"Date range: {min(stats['individual_dates'])} to {max(stats['individual_dates'])}")
                print(f"({len(stats['individual_dates'])} total acquisition dates)")
            print()
        # Sample Scene IDs
        if stats['Scene_ids']:
            print("Scene IDs (first 10):")
            print("-" * 30)
            if len(stats['Scene_ids']) <= 10:
                for scene_id in stats['Scene_ids']:
                    print(f"• {scene_id}")
            else:
                for scene_id in stats['Scene_ids'][:10]:
                    print(f"• {scene_id}")
                print(f"... and {len(stats['Scene_ids']) - 10} more scenes")
            print()
        
        print("="*60)
#Class for converting image collection into single image either using mosaic or temporal compositing
class final_Image:
    """
    Class for combining image collections to get the final images.
    Supports quality mosaics (direct stack) and temporal aggregation.
    """
    
    def __init__(self, log_level=logging.INFO):
        """
        Initialize the final_Image object and set up a class-specific logger.
        """
        ensure_ee_initialized()
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)
        self.logger.info("final_Image creation initialized.")
    #Function to calculate data coverage (pixel validity) within AOI
    #adjust the scale if needed
    def calculate_data_coverage(self, composite_image, aoi, scale=30, max_pixels=1e9, verbose=True):
        """
        Calculate data coverage (valid pixels) percentage within AOI for temporal aggregation composite image.
        This shows how much an AOI has valid data after compositing.
        
        Parameters
        ----------
        composite_image : ee.Image
            Composite image (from get_temporal_composite or get_quality_mosaic)
        aoi : ee.Geometry or ee.FeatureCollection
            Area of interest for coverage calculation
        scale : int
            Pixel scale for computation in meters (default: 30m for Landsat)
        max_pixels : float
            Maximum number of pixels to process (default: 1e9)
        verbose : bool
            If True, log detailed information (default: True)
        
        Returns
        -------
        dict
            Dictionary containing:
            - 'data_coverage_percent': Percentage of AOI with valid data (0-100)
            - 'data_gap_percent': Percentage of AOI with missing/masked data (0-100)
            - 'total_area_km2': Total AOI area in km²
            - 'valid_area_km2': Area with valid data in km²
            - 'gap_area_km2': Area with data gaps in km²
        
        Example
        -------
        >>> image_processor = final_Image()
        >>> composite = image_processor.get_temporal_composite(collection, aoi)
        >>> coverage = image_processor.calculate_data_coverage(composite, aoi)
        >>> print(f"Data coverage: {coverage['data_coverage_percent']:.1f}%")
        """
        if isinstance(aoi, ee.FeatureCollection):
            geometry = aoi.geometry()
        else:
            geometry = aoi
        
        if verbose:
            self.logger.info("Calculating data coverage within AOI...")
        
        #Create a mask of valid (non-masked) pixels
        #Use the first band to check for valid data (all bands should have same mask)
        first_band = composite_image.select(0)
        valid_mask = first_band.mask()
        #Calculate pixel area
        pixel_area = ee.Image.pixelArea()
        #Total area in AOI
        total_area = pixel_area.reduceRegion(reducer=ee.Reducer.sum(),geometry=geometry,scale=scale,maxPixels=max_pixels,
            bestEffort=True,
            tileScale=4
        )
        #Valid (unmasked) area
        valid_area_img = valid_mask.multiply(pixel_area).rename('valid_area')
        valid_area = valid_area_img.reduceRegion(reducer=ee.Reducer.sum(),geometry=geometry,scale=scale,maxPixels=max_pixels,
            bestEffort=True,
            tileScale=4
        )
        #Calculate coverage percentage
        total = ee.Number(total_area.get('area')).max(1)  # Avoid division by zero
        valid = ee.Number(valid_area.get('valid_area')).max(0)  # Default to 0 if null
        coverage_percent = valid.multiply(100).divide(total)
        gap_percent = ee.Number(100).subtract(coverage_percent)
        # Clamp between 0-100 (make sure its 0 - 100)
        coverage_percent = coverage_percent.max(0).min(100)
        gap_percent = gap_percent.max(0).min(100)
        #Get values (pull from server)
        total_val = total.getInfo()
        valid_val = valid.getInfo()
        coverage_val = coverage_percent.getInfo()
        gap_val = gap_percent.getInfo()
        #Convert to km²
        total_km2 = total_val / 1e6
        valid_km2 = valid_val / 1e6
        gap_km2 = (total_val - valid_val) / 1e6
        #Data logging if needed
        if verbose:
            self.logger.info(f"Data coverage: {coverage_val:.1f}% ({valid_km2:.2f} km² of {total_km2:.2f} km²)")
            self.logger.info(f"Data gaps: {gap_val:.1f}% ({gap_km2:.2f} km²)")
        return {
            'data_coverage_percent': round(coverage_val, 2),
            'data_gap_percent': round(gap_val, 2),
            'total_area_km2': round(total_km2, 2),
            'valid_area_km2': round(valid_km2, 2),
            'gap_area_km2': round(gap_km2, 2)
        }
    #Quality mosaic for stacking multiple scene and then clip them. This procedure stacked all of the imagery regardless of the pixel value
    def get_quality_mosaic(self, collection, aoi, quality_band='NDVI', 
                          calculate_coverage=False, coverage_scale=30, verbose=True):
        """
        Create a mosaic that selects the best available pixels across the AOI.
        Uses qualityMosaic to automatically select pixels with highest quality metric.
        
        Parameters
        ----------
        collection : ee.ImageCollection
            Filtered image collection from Reflectance_Data
        aoi : ee.Geometry or ee.FeatureCollection
            Area of interest for clipping
        quality_band : str
            Band to use for quality assessment. Options:
            - 'NDVI': Normalized Difference Vegetation Index (Select pixels with high NDVI value)
            - 'NIR': Near-infrared band (general purpose, select pixel with highest NIR reflectance)
        calculate_coverage : bool
            If True, calculate data coverage within AOI (default: False)
            Note: This triggers a client-side computation and may be slow for large areas
        coverage_scale : int
            Pixel scale in meters for coverage calculation (default: 30m)
            Use larger values (90m+) for faster computation on large areas
        verbose : bool
            If True, log detailed information (default: True)
            
        Returns
        -------
        tuple : (ee.Image, dict) if calculate_coverage=True, else ee.Image
            - Quality mosaic image clipped to AOI with best available pixels
            - Coverage statistics dict (only if calculate_coverage=True)
            
        Example
        -------
        >>> data_fetcher = Reflectance_Data()
        >>> collection, stats = data_fetcher.get_optical_data(aoi, 2020, 2020, 'L8_SR')
        >>> image_processor = final_Image()
        >>> quality_image = image_processor.get_quality_mosaic(collection, aoi, quality_band='NDVI')
        >>> # With coverage calculation
        >>> quality_image, coverage = image_processor.get_quality_mosaic(
        ...     collection, aoi, quality_band='NDVI', calculate_coverage=True
        ... )
        """
        #safety checks, make sure the AOI is ee feature collection
        if isinstance(aoi, ee.FeatureCollection):
            geometry = aoi.geometry()
        else:
            geometry = aoi
        
        if verbose:
            size = collection.size().getInfo()
            if size == 0:
                raise ValueError("Collection is empty, cannot create quality mosaic")
            self.logger.info(f"Creating quality mosaic from {size} images using {quality_band} as quality metric")
        
        #Add quality band based on selection
        def add_quality_band(img):
            if quality_band == 'NDVI':
                #NDVI: (NIR - RED) / (NIR + RED)
                ndvi = img.normalizedDifference(['NIR', 'RED']).rename('quality')
                return img.addBands(ndvi)
            elif quality_band == 'NIR':
                #If NIR band is selected (higher, better)
                return img.addBands(img.select('NIR').rename('quality'))
            else:
                # Use specified band directly
                return img.addBands(img.select(quality_band).rename('quality'))
        
        #Add quality band to collection
        collection_with_quality = collection.map(add_quality_band)
        #automatically selects pixels with highest quality values
        mosaic = collection_with_quality.qualityMosaic('quality')
        #Remove the quality band from output
        original_bands = collection.first().bandNames()
        mosaic = mosaic.select(original_bands)
        # Clip to AOI
        clipped = mosaic.clip(geometry)
        
        if verbose:
            self.logger.info(f"Quality mosaic created covering AOI with best available pixels")
        
        #Add metadata
        first_img = collection.first()
        last_img = collection.sort('system:time_start', False).first()
        
        start_date = ee.Date(first_img.get('system:time_start')).format('YYYY-MM-dd')
        end_date = ee.Date(last_img.get('system:time_start')).format('YYYY-MM-dd')
        
        clipped = clipped.set({
            'mosaic_type': 'quality_mosaic',
            'quality_metric': quality_band,
            'date_range_start': start_date,
            'date_range_end': end_date,
            'image_count': collection.size()
        })
        
        # Cast to ee.Image
        clipped = ee.Image(clipped)
        
        if verbose:
            start_str = start_date.getInfo()
            end_str = end_date.getInfo()
            self.logger.info(f"Mosaic date range: {start_str} to {end_str}")
        
        # Calculate data coverage if requested
        if calculate_coverage:
            coverage_stats = self.calculate_data_coverage(
                clipped, aoi, scale=coverage_scale, verbose=verbose
            )
            return clipped, coverage_stats
        else:
            return clipped
    #Temporal composite computes statistics across pixels
    #logic behind cloud 'removal' is that cloud typically have higher pixel value due to high reflectance,
    #thus when median composite is used cloud get 'remove' from the final image
    #adjust scale if needed
    def get_temporal_composite(self, collection, aoi, reducer='median',calculate_coverage=False, 
                              coverage_scale=30, verbose=True):
        """
        Create a temporal composite from image collection using specified reducer.
        Output is always clipped to the AOI.
        
        Parameters
        ----------
        collection : ee.ImageCollection. Filtered image collection from Reflectance_Data
        aoi :  ee.FeatureCollection. Area of interest for clipping
        reducer : str or ee.Reducer. Reduction method: 'median', 'mean', 'min', 'max', 'percentile_'
        add_band_stats : bool, If True, add additional bands with stdDev and count (default: False)
        calculate_coverage : bool. If True, calculate data coverage within AOI (default: False)
            Note: This triggers a client-side computation and may be slow for large areas
        coverage_scale : int. Pixel scale in meters for coverage calculation (default: 30m)
            Use larger values (90m+) for faster computation on large areas
        verbose : bool
            If True, log detailed information (default: True)
            
        Returns
        -------
        tuple : (ee.Image, dict) if calculate_coverage=True, else ee.Image
            - Composite image clipped to AOI with original band names (NIR, RED, etc.)
            - Coverage statistics dict (only if calculate_coverage=True)
            
        Example
        -------
        >>> data_fetcher = Reflectance_Data()
        >>> collection, stats = data_fetcher.get_optical_data(aoi, 2020, 2020, 'L8_SR')
        >>> image_processor = final_Image()
        >>> composite = image_processor.get_temporal_composite(collection, aoi, reducer='median')
        >>> # With coverage calculation
        >>> composite, coverage = image_processor.get_temporal_composite(
        ...     collection, aoi, reducer='median', calculate_coverage=True
        ... )
        """
        #Make sure that AOi is feature collection
        if isinstance(aoi, ee.FeatureCollection):
            geometry = aoi.geometry()
        else:
            geometry = aoi
        #Get original band names before reduction (for renaming)
        original_bands = collection.first().bandNames()
        #Resolve reducer argument to an Earth Engine reducer
        if isinstance(reducer, str):
            reducer_lower = reducer.lower()
            if reducer_lower == 'median':
                ee_reducer = ee.Reducer.median()
            elif reducer_lower == 'mean':
                ee_reducer = ee.Reducer.mean()
            elif reducer_lower == 'min':
                ee_reducer = ee.Reducer.min()
            elif reducer_lower == 'max':
                ee_reducer = ee.Reducer.max()
            elif reducer_lower.startswith('percentile_'):
                try:
                    percentile = int(reducer_lower.split('_')[1])
                except Exception:
                    raise ValueError(f"Invalid percentile format: {reducer}")
                ee_reducer = ee.Reducer.percentile([percentile])
            else:
                raise ValueError(f"Unsupported reducer: {reducer}")
        else:
            ee_reducer = reducer

        #Check collection size - only if verbose to avoid unnecessary getInfo()
        if verbose:
            size = collection.size().getInfo()
            if size == 0:
                raise ValueError("Collection is empty, cannot create composite")
            self.logger.info(f"Creating {reducer} composite from {size} images")

        #Single reducer composite - rename main bands back to original names
        composite = collection.reduce(ee_reducer).rename(original_bands)
        
        # Clip to AOI (always required)
        composite = composite.clip(geometry)
        
        if verbose:
            self.logger.info("Composite clipped to AOI")
        
        # Add metadata 
        first_img = collection.first()
        last_img = collection.sort('system:time_start', False).first()
        
        # Store dates as server-side strings to avoid getInfo() calls
        start_date = ee.Date(first_img.get('system:time_start')).format('YYYY-MM-dd')
        end_date = ee.Date(last_img.get('system:time_start')).format('YYYY-MM-dd')
        size_server = collection.size()
        
        composite = composite.set({
            'composite_start_date': start_date,
            'composite_end_date': end_date,
            'composite_count': size_server,
            'composite_reducer': str(reducer)
        })
        
        if verbose:
            #Only call the client side info if needed
            start_date_str = start_date.getInfo()
            end_date_str = end_date.getInfo()
            self.logger.info(f"Composite created from {start_date_str} to {end_date_str}")
        
        # Calculate data coverage if requested
        if calculate_coverage:
            coverage_stats = self.calculate_data_coverage(
                composite, aoi, scale=coverage_scale, verbose=verbose
            )
            return composite, coverage_stats
        else:
            return composite