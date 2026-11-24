"""
Module 1: Data Acquisition 

Provides:
- Data retrieval and preprocessing (non correction) from Google Earth Engine
- Support for handling various satellite image collections. Currently supports Landsat 1-9, Sentinel-2, and NASA Harmonized Landsat Sentinel-2 (HLS v2.0) datasets.
"""
import ee
from datetime import datetime
import logging
from typing import Union, Optional, List
from .ee_config import ensure_ee_initialized
#Do not initialize Earth Engine at import time. Initialize when an instance is created.
#input date pasring function, used by data retrieval functions to allow flexible date input (year or full date)
def parse_date_input(date_input: Union[int, str], is_start: bool = True) -> str:
    """
    Parse date input to YYYY-MM-DD format.

    Accepts either a year (int or 4-digit string) or a full date string.
    For year inputs, returns Jan 1 for start dates or Dec 31 for end dates.

    Parameters
    ----------
    date_input : int or str
        Year (e.g., 2020 or "2020") or full date (e.g., "2020-06-15")
    is_start : bool, default=True
        If True and input is a year, return January 1.
        If False and input is a year, return December 31.

    Returns
    -------
    str
        Date in YYYY-MM-DD format

    Raises
    ------
    ValueError
        If date_input is not int, str, or valid format

    Examples
    --------
    >>> parse_date_input(2020, is_start=True)
    '2020-01-01'
    >>> parse_date_input(2020, is_start=False)
    '2020-12-31'
    >>> parse_date_input("2020-06-15")
    '2020-06-15'
    >>> parse_date_input("2020", is_start=False)
    '2020-12-31'
    """
    if isinstance(date_input, int):
        return f"{date_input}-01-01" if is_start else f"{date_input}-12-31"
    elif isinstance(date_input, str):
        if len(date_input) == 4 and date_input.isdigit():
            return f"{date_input}-01-01" if is_start else f"{date_input}-12-31"
        else:
            # Validate full date format (basic check)
            if len(date_input) == 10 and date_input[4] == '-' and date_input[7] == '-':
                return date_input
            else:
                raise ValueError(
                    f"Invalid date format: {date_input}. Expected YYYY or YYYY-MM-DD"
                )
    else:
        raise ValueError(f"Date must be int or str, got {type(date_input)}")

#MODULE-LEVEL CONSTANTS
#Maximum number of images to retrieve from Earth Engine, prevents computation timeout errors for large collections
MAX_COLLECTION_SIZE = 300
#Default pixel scale for Landsat imagery (meters), Landsat 4-9: 30m, Landsat 1-3 MSS: 60m, Sentinel-2 10/20m
DEFAULT_COVERAGE_SCALE = 30
#Maximum pixels for reduce operations. Prevents memory errors on large area computations
DEFAULT_MAX_PIXELS = 1e9
#Tile scale for large area computations, Higher values = faster but less accurate
DEFAULT_TILE_SCALE = 4
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

# Module 1: Cloudless Image Mosaic
## System Response 1.2: Search and Filter Imagery
class Reflectance_Data:
    """Class for fetching and pre-processing satellite image collection from Google Earth Engine API."""
    #Define the Landsat optical datasets. The band reflectances used is from Collection 2 Surface Reflectancce Data
    OPTICAL_DATASETS = {
        'L1_RAW': {
            'collection': 'LANDSAT/LM01/C02/T1',
            'satellite_family': 'landsat',
            'cloud_property': 'CLOUD_COVER_LAND',
            'type': 'landsat_raw',
            'sensor': 'L1',
            'stats_properties': ['CLOUD_COVER_LAND', 'system:time_start', 'system:index', 'WRS_PATH', 'WRS_ROW'],
            'description': 'Landsat 1 Multispectral Scanner Raw Collection'
        },
        'L2_RAW': {
            'collection': 'LANDSAT/LM02/C02/T1',
            'satellite_family': 'landsat',
            'cloud_property': 'CLOUD_COVER_LAND',
            'type': 'landsat_raw',
            'sensor': 'L2',
            'stats_properties': ['CLOUD_COVER_LAND', 'system:time_start', 'system:index', 'WRS_PATH', 'WRS_ROW'],
            'description': 'Landsat 2 Multispectral Scanner Raw Collection'            
        },
        'L3_RAW':{
            'collection': 'LANDSAT/LM03/C02/T1',
            'satellite_family': 'landsat',
            'cloud_property': 'CLOUD_COVER_LAND',
            'type': 'landsat_raw',
            'sensor': 'L3',
            'stats_properties': ['CLOUD_COVER_LAND', 'system:time_start', 'system:index', 'WRS_PATH', 'WRS_ROW'],
            'description': 'Landsat 3 Multispectral Scanner Raw Collection'
        },
        'L4_SR': {
            'collection': 'LANDSAT/LT04/C02/T1_L2',
            'satellite_family': 'landsat',
            'cloud_property': 'CLOUD_COVER_LAND',
            'type': 'landsat_sr',
            'sensor': 'L4',
            'stats_properties': ['CLOUD_COVER_LAND', 'system:time_start', 'system:index', 'WRS_PATH', 'WRS_ROW'],
            'description': 'Landsat 4 Thematic Mapper Surface Reflectance Collection',           
        },
        'L5_SR': {
            'collection': 'LANDSAT/LT05/C02/T1_L2',
            'satellite_family': 'landsat',
            'cloud_property': 'CLOUD_COVER_LAND',
            'type': 'landsat_sr',
            'sensor': 'L5',
            'stats_properties': ['CLOUD_COVER_LAND', 'system:time_start', 'system:index', 'WRS_PATH', 'WRS_ROW'],
            'description': 'Landsat 5 Thematic Mapper Surface Reflectance Collection',
        },
        'L7_SR': {
            'collection': 'LANDSAT/LE07/C02/T1_L2', 
            'satellite_family': 'landsat',
            'cloud_property': 'CLOUD_COVER_LAND',
            'type': 'landsat_sr',
            'sensor': 'L7',
            'stats_properties': ['CLOUD_COVER_LAND', 'system:time_start', 'system:index', 'WRS_PATH', 'WRS_ROW'],
            'description': 'Landsat 7 Enhance Thematic Mapper + Surface Reflectance'
        },
        'L8_SR': {
            'collection': 'LANDSAT/LC08/C02/T1_L2',
            'satellite_family': 'landsat',
            'cloud_property': 'CLOUD_COVER_LAND',
            'type': 'landsat_sr',
            'sensor': 'L8',
            'stats_properties': ['CLOUD_COVER_LAND', 'system:time_start', 'system:index', 'WRS_PATH', 'WRS_ROW'],
            'description': 'Landsat 8 Operational Land Imager Surface Reflectance'
        },
        'L9_SR': {
            'collection': 'LANDSAT/LC09/C02/T1_L2',
            'satellite_family': 'landsat',
            'cloud_property': 'CLOUD_COVER_LAND', 
            'type': 'landsat_sr',
            'sensor': 'L9',
            'stats_properties': ['CLOUD_COVER_LAND', 'system:time_start', 'system:index', 'WRS_PATH', 'WRS_ROW'],
            'description': 'Landsat 9 Operational Land Imager-2 Surface Reflectance'
        }
    }
    #Define the Sentinel-2 datasets, SR and TOA
    #Cloud masking is performed via Cloud Score+ (GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED)
    S2_DATASETS = {
        'S2_SR': {
            'collection': 'COPERNICUS/S2_SR_HARMONIZED',
            'cloud_property': 'CLOUDY_PIXEL_PERCENTAGE',
            'type': 's2_sr',
            'sensor': 'S2',
            'description': 'Sentinel-2 Level-2A Surface Reflectance (Harmonized)',
        },
        'S2_TOA': {
            'collection': 'COPERNICUS/S2_HARMONIZED',
            'cloud_property': 'CLOUDY_PIXEL_PERCENTAGE',
            'type': 's2_toa',
            'sensor': 'S2',
            'description': 'Sentinel-2 Level-1C Top-of-Atmosphere (Harmonized)',
        }
    }
    #Define NASA Harmonized Landsat Sentinel-2 (HLS v2.0) datasets (2013-present, 30m MGRS grid)
    #alternative solution to sentinel for multitemporal analysis
    HLS_DATASETS = {
        'HLS': {
            'l30_collection': 'NASA/HLS/HLSL30/v002',
            's30_collection': 'NASA/HLS/HLSS30/v002',
            'cloud_property': 'CLOUD_COVERAGE',
            'type': 'hls_merged',
            'sensor': 'HLS',
            'start_date': '2013-04-01',
            'description': 'NASA Harmonized Landsat Sentinel-2 (HLSL30 + HLSS30 v2.0, 30m)',
        },
        'HLS_L30': {
            'collection': 'NASA/HLS/HLSL30/v002',
            'cloud_property': 'CLOUD_COVERAGE',
            'type': 'hls_l30',
            'sensor': 'HLS_L30',
            'start_date': '2013-04-01',
            'description': 'NASA Harmonized Landsat-8/9 OLI Surface Reflectance (HLSL30 v2.0, 30m)',
        },
        'HLS_S30': {
            'collection': 'NASA/HLS/HLSS30/v002',
            'cloud_property': 'CLOUD_COVERAGE',
            'type': 'hls_s30',
            'sensor': 'HLS_S30',
            'start_date': '2015-11-28',
            'description': 'NASA Harmonized Sentinel-2 MSI Surface Reflectance (HLSS30 v2.0, 30m)',
        },
    }
    #Roy et al. (2016) Table 2 OLS coefficients transforming Landsat TM/ETM+ (L4/L5/L7) SR to OLI (L8/L9) SR
    #BRDF correction coefficient
    ROY2016_COEFFICIENTS = {
        'slopes': {
            'BLUE': 0.9747,
            'GREEN': 0.9822,
            'RED': 0.9739,
            'NIR': 1.0113,
            'SWIR1': 1.0071,
            'SWIR2': 1.0106,
        },
        'intercepts': {
            'BLUE': 0.0003,
            'GREEN': 0.0033,
            'RED': 0.0010,
            'NIR': 0.0051,
            'SWIR1': 0.0043,
            'SWIR2': 0.0030,
        },
    }
    #Define the thermal datasets. The thermal bands used is from Collection 2 Top-of-atmosphere data 
    #The TOA data provide consistent result and contain minimum missing pixel data
    #Note: Landsat 1-3 MSS sensors did not have thermal bands, so they are excluded
    THERMAL_DATASETS = {
        'L4_TOA': {
            'collection': 'LANDSAT/LT04/C02/T1_TOA',
            'cloud_property': 'CLOUD_COVER_LAND',
            'type': 'landsat_toa',
            'sensor': 'L4',
            'thermal_band': 'B6',
            'description': 'Landsat 4 Top-of-atmosphere reflectance',            
        },
        'L5_TOA': {
            'collection': 'LANDSAT/LT05/C02/T1_TOA',
            'cloud_property': 'CLOUD_COVER_LAND',
            'type': 'landsat_toa',
            'sensor': 'L5',
            'thermal_band': 'B6',
            'description': 'Landsat 5 Top-of-atmosphere reflectance',
        },
        'L7_TOA': {
            'collection': 'LANDSAT/LE07/C02/T1_TOA',
            'cloud_property': 'CLOUD_COVER_LAND',
            'type': 'landsat_toa',
            'sensor': 'L7',
            'thermal_band': 'B6_VCID_2',
            'description': 'Landsat 7 Top-of-atmosphere reflectance',
        },
        'L8_TOA': {
            'collection': 'LANDSAT/LC08/C02/T1_TOA',
            'cloud_property': 'CLOUD_COVER_LAND',
            'type': 'landsat_toa',
            'sensor': 'L8',
            'thermal_band': 'B10',
            'description': 'Landsat 8 Top-of-atmosphere reflectance'  
        },
        'L9_TOA': {
            'collection': 'LANDSAT/LC09/C02/T1_TOA',
            'cloud_property': 'CLOUD_COVER_LAND',
            'type': 'landsat_toa',
            'sensor': 'L9',
            'thermal_band': 'B10',
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
    #check thermal capability of the dataset. Only Landsat 4-9 have thermal bands, and they are only available in the TOA collection, not the SR collection. 
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
        #Implementation on Image
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
        return ee.Image(image.updateMask(final_mask).copyProperties(image, image.propertyNames()))
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
        preserved. further reading: https://medium.com/google-earth/all-clear-with-cloud-score-bd6ee2e2235e

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
            return ee.Image(masked.copyProperties(image, image.propertyNames()))
        except ee.EEException as e:
            self.logger.error(f"EEException in mask_s2_clouds: {e}")
            return None # type: ignore
        except Exception as e:
            self.logger.error(f"Unexpected error in mask_s2_clouds: {e}")
            return None # type: ignore
    #Band renaming for Sentinel-2. Standarized band name for later use
    def rename_s2_bands(self, image: ee.Image) -> ee.Image:
        """
        Select and rename Sentinel-2 optical bands to standardized semantic names.

        Selects raw Sentinel-2 optical bands and renames them to the common
        semantic convention used across the luma-ge pipeline.  Raw DN values
        (0–10000) are preserved — scaling to reflectance (0.0–1.0) is a
        separate, standalone operation performed by ``apply_s2_scale_factors``.
        B1 (coastal aerosol, 60m) is excluded.

        Parameters
        ----------
        image : ee.Image
            Sentinel-2 image containing source bands
            ``['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12']``.

        Returns
        -------
        ee.Image
            Image with renamed bands retaining raw DN values:
            ``['BLUE','GREEN','RED','RED_EDGE1','RED_EDGE2',
            'RED_EDGE3','NIR','RED_EDGE4','SWIR1','SWIR2']``.
            Note: B1 (coastal aerosol, 60m) is excluded.

        Example
        -------
        >>> rd = Reflectance_Data()
        >>> renamed = rd.rename_s2_bands(image)
        >>> renamed_col = collection.map(lambda img: rd.rename_s2_bands(img))
        """
        renamed = image.select(
            ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12'],
            ['BLUE', 'GREEN', 'RED', 'RED_EDGE1', 'RED_EDGE2', 'RED_EDGE3',
             'NIR', 'RED_EDGE4', 'SWIR1', 'SWIR2']
        )
        return ee.Image(renamed.copyProperties(image, image.propertyNames()))

    #Apply Sentinel-2 scale factors (standalone, post-collection operation)
    def apply_s2_scale_factors(self, image: ee.Image) -> ee.Image:
        """
        Convert Sentinel-2 DN values to surface reflectance in the [0, 1] range.

        Parameters
        ----------
        image : ee.Image Sentinel-2 image with standardized band names (output of ``rename_s2_bands``).  Band values are raw DN (0–10000).

        Returns
        -------
        ee.Image
            Image with optical bands scaled to reflectance (0.0–1.0).

        Example
        -------
        >>> rd = Reflectance_Data()
        >>> collection, _ = rd.get_s2_optical_data(aoi, 2022, 2023)
        >>> composite = final_Image().get_temporal_composite(collection, aoi)
        >>> sharpened = rd.sharpen_s2_bands(composite, aoi=aoi)   # requires raw DN
        >>> reflectance = rd.apply_s2_scale_factors(sharpened)    # scale after sharpen
        """
        optical_bands = image.multiply(0.0001).toFloat()
        return ee.Image(optical_bands.copyProperties(image, image.propertyNames()))

    #Sharpening the 20m bands using multi resolution analysis, namely High Pass Filter (HPF)
    def sharpen_s2_bands(self, image: ee.Image, aoi=None,
        crs: str = 'EPSG:3857',  # legacy; overridden when crs_transform is None
        crs_transform: Optional[List] = None,
        mod: float = 0.25,
        best_effort: bool = True,
        tile_scale: float = 4.0,
        max_pixels: Optional[int] = None,
    ) -> Optional[ee.Image]:
        """
        Sharpen 20m Sentinel-2 bands to 10m using High Pass Filter (HPF) Multi Resolution Analysis (MRA).

        Summary of the HPF algorithm:
        ---------
        1. Build pseudo-panchromatic band (PAN) from mean of 10m Vis-NIR bands.
        2. Convolve PAN with a 5×5 Laplacian HPF kernel.
        3. Compute per-band injection weight from native 20m stdDev and HPF stdDev:
               W = σ_native20m / σ_HPF × mod
        4. Bilinear-resample 20m bands to 10m.
        5. Inject: ``out = MS_resampled + HPF × W``.
        6. Linear histogram stretch to restore original 20m mean/stdDev:
               sharpened = (out − μ_out) / σ_out × σ_native20m + μ_native20m
           All stats are computed server-side via ``reduceRegion → toImage``.
        7. Stack sharpened 20m bands with original 10m bands; copy image properties.

        Notes
        -----
        ``reproject()`` is intentionally avoided on intermediate images to prevent
        GEE from materializing full-resolution pixel grids in memory before
        downstream operations.  The UTM CRS and scale are instead passed
        directly to ``reduceRegion`` so GEE can apply lazy evaluation.  The
        final reproject at step 4 is the only anchor, applied only to the
        lightweight bilinear-resampled 20m image immediately before injection.

        Parameters
        ----------
        image : ee.Image
            Sentinel-2 image with standardized band names (output of
            ``rename_s2_bands``).  Band values may be raw DN (0–10 000) or
            scaled reflectance (0.0–1.0) — the algorithm is scale-agnostic.
        aoi : ee.Geometry or ee.FeatureCollection, optional
            Region for stat reductions. Falls back to ``image.geometry()`` if None.
        crs : str, optional
            EPSG code of the target UTM projection. Ignored when
            ``crs_transform`` is ``None`` — UTM zone is auto-derived from the
            AOI centroid in that case.
        crs_transform : list, optional
            Affine transform ``[xScale, xShear, xOrig, yShear, yScale, yOrig]``
            anchoring the pixel grid. When ``None`` (default), auto-derived from
            AOI centroid with origin ``[10, 0, 300000, 0, -10, 300000]``.
            Supply both ``crs`` and ``crs_transform`` to fully override.
        mod : float, optional
            HPF modulating factor (sharpening strength). Default: ``0.25``.
        best_effort : bool, optional
            If the geometry exceeds ``max_pixels``, auto-increase scale to fit.
            Default: ``True``.
        tile_scale : float, optional
            Aggregation tile-size scaling factor for ``reduceRegion``.
            Increase (e.g. 4 or 8) for large AOIs to avoid memory errors.
            Default: ``4.0``.

        Returns
        -------
        ee.Image or None
            Sharpened image with bands
            ``['BLUE','GREEN','RED','RED_EDGE1','RED_EDGE2','RED_EDGE3',
            'NIR','RED_EDGE4','SWIR1','SWIR2']``, or ``None`` on error.

        References
        ----------
        https://leclab.wixsite.com/spatial/post/pansharpening-sentinel-2-imagery-in-google-earth-engine

        Example
        -------
        >>> rd = Reflectance_Data()
        >>> collection, _ = rd.get_s2_optical_data(aoi, 2022, 2023)
        >>> img = final_Image().get_temporal_composite(collection, aoi)
        >>> # Auto UTM zone derived from AOI centroid (recommended)
        >>> sharpened = rd.sharpen_s2_bands(img, aoi=aoi)
        >>> # For a very large AOI increase tile_scale to avoid memory errors
        >>> sharpened = rd.sharpen_s2_bands(img, aoi=aoi, tile_scale=8.0)
        >>> # Force a specific CRS and tile-grid origin (advanced override)
        >>> sharpened = rd.sharpen_s2_bands(img, aoi=aoi, crs='EPSG:32749',
        ...                                  crs_transform=[10, 0, 300000, 0, -10, 300000])
        """
        try:
            # Resolve geometry for stat reductions
            if aoi is not None:
                geometry = aoi.geometry() if isinstance(aoi, ee.FeatureCollection) else aoi
            else:
                geometry = image.geometry()

            # Derive UTM zone from AOI centroid (single cheap getInfo call)
            if crs_transform is None:
                centroid_coords = geometry.centroid(maxError=1000).getInfo()['coordinates'] # type: ignore
                lon, lat = centroid_coords[0], centroid_coords[1]
                zone = int((lon + 180) / 6) + 1
                epsg_code = 32600 + zone if lat >= 0 else 32700 + zone
                crs = f'EPSG:{epsg_code}'
                crs_transform = [10, 0, 300_000, 0, -10, 300_000]
                self.logger.info(
                    f"Auto-derived UTM zone: {crs} "
                    f"(centroid lon={lon:.4f}, lat={lat:.4f}, zone={zone})"
                )
            else:
                self.logger.info(f"Using caller-supplied crs_transform with CRS: {crs}")

            bands_10m = ['BLUE', 'GREEN', 'RED', 'NIR']
            bands_20m = ['RED_EDGE1', 'RED_EDGE2', 'RED_EDGE3',
                         'RED_EDGE4', 'SWIR1', 'SWIR2']

            # Common reduceRegion kwargs — crs/scale passed directly (no reproject anchor)
            rr_kwargs = dict(
                geometry   = geometry,
                bestEffort = best_effort,
                tileScale  = tile_scale,
            )
            if max_pixels is not None:
                rr_kwargs['maxPixels'] = max_pixels

            # 1. Pseudo-panchromatic band from mean of 10m Vis-NIR bands.
            #    No reproject here — let GEE evaluate lazily.
            pan = (
                image.select(bands_10m)
                     .reduce(ee.Reducer.mean())
                     .rename('PAN')
            )

            # 2. HPF kernel: 5×5 Laplacian centred at (-2, -2)
            kernel = ee.Kernel.fixed(
                5, 5,
                [[-1, -1, -1, -1, -1],
                 [-1, -1, -1, -1, -1],
                 [-1, -1, 24, -1, -1],
                 [-1, -1, -1, -1, -1],
                 [-1, -1, -1, -1, -1]],
                -2, -2, False,
            )
            hpf = pan.convolve(kernel).rename('highPassFilter')

            # 3. Native 20m band statistics — pass crs + scale to reduceRegion directly.
            #    Avoid reproject() here to prevent premature full-resolution materialization.
            stats_20m = (
                image.select(bands_20m)
                .reduceRegion(
                    reducer = ee.Reducer.stdDev().combine(
                        ee.Reducer.mean(), sharedInputs=True
                    ),
                    crs   = crs,
                    scale = 20,
                    **rr_kwargs,
                )
                .toImage()
            )
            mean_20m    = stats_20m.select('.*_mean').regexpRename('(.*)_mean', '$1')
            std_dev_20m = stats_20m.select('.*_stdDev').regexpRename('(.*)_stdDev', '$1')

            # 4. HPF stdDev at 10m — server-side scalar, no materialization
            std_dev_hpf = (
                hpf
                .reduceRegion(
                    reducer = ee.Reducer.stdDev(),
                    crs     = crs,
                    scale   = 10,
                    **rr_kwargs,
                )
                .getNumber('highPassFilter')
            )

            # 5. Bilinear resample 20m → 10m.
            #    reproject() is applied only here — to the lightweight resampled
            #    image, immediately before injection, minimising the materialized footprint.
            crs_transform_20m = [20, 0, crs_transform[2], 0, -20, crs_transform[5]]
            resampled_20m = (
                image.select(bands_20m)
                     .reproject(crs=crs, crsTransform=crs_transform_20m)
                     .resample('bilinear')
                     .reproject(crs=crs, crsTransform=crs_transform)
            )

            # 6. HPF injection per band (Python loop — avoids ee.List.map serialization overhead)
            # W = σ_native20m / σ_HPF × mod;  out = MS_resampled + HPF × W
            injected_bands = []
            for band in bands_20m:
                w = ee.Image().expression(
                    'std_dev_20m / std_dev_hpf * mod', {
                        'std_dev_20m' : std_dev_20m.select([band]),
                        'std_dev_hpf' : std_dev_hpf,
                        'mod'         : ee.Number(mod),
                    }
                )
                injected = (
                    ee.Image().expression(
                        'ms + hpf * w', {
                            'ms'  : resampled_20m.select([band]),
                            'hpf' : hpf,
                            'w'   : w,
                        }
                    )
                    .rename([band])
                )
                injected_bands.append(injected)

            output = ee.Image.cat(injected_bands)

            # 7. Linear histogram stretch — crs + scale passed directly, no reproject
            #    sharpened = (output − μ_out) / σ_out × σ_native20m + μ_native20m
            stats_output = (
                output
                .reduceRegion(
                    reducer = ee.Reducer.stdDev().combine(
                        ee.Reducer.mean(), sharedInputs=True
                    ),
                    crs   = crs,
                    scale = 10,
                    **rr_kwargs,
                )
                .toImage()
            )
            mean_output    = stats_output.select('.*_mean').regexpRename('(.*)_mean', '$1')
            std_dev_output = stats_output.select('.*_stdDev').regexpRename('(.*)_stdDev', '$1')

            sharpened = (
                ee.Image().expression(
                    '(output - mean_output) / std_dev_output * std_dev_20m + mean_20m', {
                        'output'        : output,
                        'mean_output'   : mean_output,
                        'std_dev_output': std_dev_output,
                        'std_dev_20m'   : std_dev_20m,
                        'mean_20m'      : mean_20m,
                    }
                )
            )

            # 8. Stack sharpened 20m bands with original 10m bands and copy properties
            result = image.select(bands_10m).addBands(sharpened)
            return ee.Image(result.copyProperties(image, image.propertyNames()))

        except ee.EEException as e:
            self.logger.error(f"EEException in sharpen_s2_bands: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error in sharpen_s2_bands: {e}")
            return None

    #Core methods to retrieve sentinel 2 data
    def get_s2_optical_data(self, aoi, start_date, end_date, cloud_cover: int = 30, cs_threshold: float = 0.7,
        verbose: bool = True,
        compute_detailed_stats: bool = True,):
        """
        Retrieve a filtered and pre-processed Sentinel-2 image collection. 
        Joins Cloud Score+ scores, applies cloud masking, scale
        factors, and band renaming

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
              ``['BLUE','GREEN','RED','RED_EDGE1','RED_EDGE2',
              'RED_EDGE3','NIR','RED_EDGE4','SWIR1','SWIR2']``.
            * ``dict`` — statistics and metadata with keys:
              ``dataset``, ``sensor``, ``date_range_requested``,
              ``cloud_cover_threshold``, ``collection_type``,
              ``initial_collection``,
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
        # Validate AOI before any GEE call
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

        # SR_BOUNDARY: first date for which COPERNICUS/S2_SR_HARMONIZED is available in GEE
        SR_BOUNDARY = '2017-03-28'

        # --- Three-branch date routing (Requirements 13.1, 13.2, 13.3) ---
        if end_date < SR_BOUNDARY:
            # TOA-only path — SR archive does not exist before this date
            collection_type = 'S2_TOA'
            config = self.S2_DATASETS['S2_TOA']
            dataset_description = config['description']
        elif start_date >= SR_BOUNDARY:
            # SR-only path — existing behavior
            collection_type = 'S2_SR'
            config = self.S2_DATASETS['S2_SR']
            dataset_description = config['description']
        else:
            # Split/merge path — date range straddles the SR availability boundary
            collection_type = 'S2_MERGED'
            config = self.S2_DATASETS['S2_SR']  # used for cloud_property field
            dataset_description = 'Sentinel-2 Merged TOA+SR (auto-fallback)'

        try:
            if verbose:
                self.logger.info(f"Date range: {start_date} to {end_date}")
                self.logger.info(f"Cloud cover threshold (image-level): {cloud_cover}%")
                self.logger.info(f"Cloud Score+ pixel threshold: {cs_threshold}")
                if not compute_detailed_stats:
                    self.logger.info("Detailed statistics will not be computed")

                # Verbose logging for collection selection reason (Requirements 16.1, 16.2, 16.3, 16.4)
                if collection_type == 'S2_TOA':
                    self.logger.info(
                        f"S2_SR data is unavailable before {SR_BOUNDARY}. "
                        f"Using Sentinel-2 Level 1 TOA instead."
                    )
                elif collection_type == 'S2_SR':
                    self.logger.info(
                        f"Using Sentinel-2 Level 2A SR data."
                    )
                else:  # S2_MERGED
                    self.logger.info(
                        f"Date range spans the Sentinel-2 SR availability boundary ({SR_BOUNDARY}). "
                        f"Sentinel-2 Level 1C TOA will be used for the pre-boundary portion "
                        f"[{start_date} to 2017-03-27] and Level 2A SR for "
                        f"the post-boundary portion [2017-03-28 to {end_date}]. "
                        f"Split point: {SR_BOUNDARY}."
                    )

            # Cloud Score+ collection (shared across all paths)
            cs_plus = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED')

            # Build the collection based on the routing branch
            if collection_type == 'S2_TOA':
                # TOA-only path
                initial_collection = (
                    ee.ImageCollection(self.S2_DATASETS['S2_TOA']['collection'])
                    .filterBounds(aoi)
                    .filterDate(start_date, end_date)
                )
                collection = (
                    initial_collection
                    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_cover))
                    .limit(250)
                    .linkCollection(cs_plus, ['cs_cdf'])
                    .map(lambda img: self.mask_s2_clouds(img, threshold=cs_threshold))
                    .map(lambda img: self.rename_s2_bands(img))
                )

            elif collection_type == 'S2_SR':
                # SR-only path (existing behavior)
                initial_collection = (
                    ee.ImageCollection(self.S2_DATASETS['S2_SR']['collection'])
                    .filterBounds(aoi)
                    .filterDate(start_date, end_date)
                )
                collection = (
                    initial_collection
                    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_cover))
                    .limit(250)
                    .linkCollection(cs_plus, ['cs_cdf'])
                    .map(lambda img: self.mask_s2_clouds(img, threshold=cs_threshold))
                    .map(lambda img: self.rename_s2_bands(img))
                )

            else:  # S2_MERGED
                # Split/merge path: TOA covers pre-boundary, SR covers post-boundary
                toa_sub = (
                    ee.ImageCollection(self.S2_DATASETS['S2_TOA']['collection'])
                    .filterBounds(aoi)
                    .filterDate(start_date, '2017-03-27')
                    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_cover))
                    .limit(250)
                    .linkCollection(cs_plus, ['cs_cdf'])
                    .map(lambda img: self.mask_s2_clouds(img, threshold=cs_threshold))
                    .map(lambda img: self.rename_s2_bands(img))
                )
                sr_sub = (
                    ee.ImageCollection(self.S2_DATASETS['S2_SR']['collection'])
                    .filterBounds(aoi)
                    .filterDate('2017-03-28', end_date)
                    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_cover))
                    .limit(250)
                    .linkCollection(cs_plus, ['cs_cdf'])
                    .map(lambda img: self.mask_s2_clouds(img, threshold=cs_threshold))
                    .map(lambda img: self.rename_s2_bands(img))
                )
                initial_collection = toa_sub.merge(sr_sub)
                collection = initial_collection

            #Sentinel 2 retrieval statistics
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

            # Filtered stats (after cloud cover filter + processing)
            filtered_stats = stats_object.get_collection_statistics(
                collection,
                compute_detailed_stats,
                cloud_property='CLOUDY_PIXEL_PERCENTAGE',
            )

            # Verbose logging of filtered collection details
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

            #Stamp the requested date range so downstream modules (e.g. SAR predictors) can inherit the Module 1 date selection
            collection = collection.set({
                'query_start_date': str(parse_date_input(start_date, True)),
                'query_end_date': str(parse_date_input(end_date, False)),
            })

            return collection, {
                'dataset': dataset_description,
                'sensor': 'S2',
                'date_range_requested': f"{start_date} to {end_date}",
                'cloud_cover_threshold': cloud_cover,
                'collection_type': collection_type,
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
        Get image collection for Landsat 1-9 with detailed information logging.

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
        tuple : (ee.ImageCollection, dict) Filtered and preprocessed image collection with statistics.
        
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
        #Parse inputs (handles both year and full date)
        start_date = parse_date_input(start_date, is_start=True)
        end_date   = parse_date_input(end_date, is_start=False)

        if optical_data not in self.OPTICAL_DATASETS:
            raise ValueError(f"optical_data must be one of: {list(self.OPTICAL_DATASETS.keys())}")

        config = self.OPTICAL_DATASETS[optical_data]
        #Warn if using legacy Landsat 1-3 data
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
        #Limit to MAX_COLLECTION_SIZE scenes to avoid Earth Engine computation limits
        collection = collection.limit(MAX_COLLECTION_SIZE)
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

        # Stamp the requested date range so downstream modules (e.g. SAR predictors)
        # can inherit the Module 1 date selection
        collection = collection.set({
            'query_start_date': str(parse_date_input(start_date, True)),
            'query_end_date': str(parse_date_input(end_date, False)),
        })

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
        # Parse inputs (handles both year and full date)
        start_date = parse_date_input(start_date, is_start=True)
        end_date   = parse_date_input(end_date, is_start=False)
        #Helper function to rename the bands
        def rename_thermal_band(img):
            return img.select([thermal_band]).rename(['THERMAL'])
        #The core function for thermal bnd
        if thermal_data not in self.THERMAL_DATASETS:
                raise ValueError(f"thermal_data must be one of: {list(self.THERMAL_DATASETS.keys())}")

        config = self.THERMAL_DATASETS[thermal_data]
        #Decide which thermal band to select
        sensor = config['sensor']
        thermal_band = config['thermal_band']
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
        #Limit to MAX_COLLECTION_SIZE scenes to avoid Earth Engine computation limits
        collection = collection.limit(MAX_COLLECTION_SIZE)
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

    # HLS v2.0 (Harmonized Landsat Sentinel-2) & Cross-Sensor Harmonization
    #support both Landsat and Sentinel-2
    #cloud mask for HLS
    def mask_hls_fmask(self, image: ee.Image, mask_high_aerosol: bool = True) -> ee.Image:
        """
        Mask clouds, cloud adjacency, cloud shadows, snow/ice, and optionally high
        aerosol in NASA HLS v2.0 imagery (`HLSL30` and `HLSS30`) using the `Fmask` band.

        Bit layout of `Fmask`:
        - Bit 0: Cirrus
        - Bit 1: Cloud
        - Bit 2: Adjacent to cloud/shadow
        - Bit 3: Cloud shadow
        - Bit 4: Snow/ice
        - Bits 6-7: Aerosol level (`11` = 3 = high aerosol)
        """
        qa = image.select('Fmask')
        cirrus = qa.bitwiseAnd(1 << 0).eq(0)
        cloud = qa.bitwiseAnd(1 << 1).eq(0)
        adj_cloud = qa.bitwiseAnd(1 << 2).eq(0)
        shadow = qa.bitwiseAnd(1 << 3).eq(0)
        snow = qa.bitwiseAnd(1 << 4).eq(0)

        mask = cirrus.And(cloud).And(adj_cloud).And(shadow).And(snow)
        if mask_high_aerosol:
            low_med_aerosol = qa.rightShift(6).bitwiseAnd(3).neq(3)
            mask = mask.And(low_med_aerosol)

        return ee.Image(image.updateMask(mask).copyProperties(image, image.propertyNames()))
    #rename HLS bands according to Luma name conventions
    def rename_hls_bands(self, image: ee.Image, hls_type: str = 'HLS_L30') -> ee.Image:
        """
        Select and rename NASA HLS v2.0 optical bands to standardized semantic names:
        `['BLUE', 'GREEN', 'RED', 'NIR', 'SWIR1', 'SWIR2']`.

        - `HLS_L30` (Landsat 8/9 OLI): `['B2', 'B3', 'B4', 'B5', 'B6', 'B7']`
        - `HLS_S30` (Sentinel-2 MSI): `['B2', 'B3', 'B4', 'B8A', 'B11', 'B12']`
        """
        target_names = ['BLUE', 'GREEN', 'RED', 'NIR', 'SWIR1', 'SWIR2']
        if hls_type == 'HLS_L30':
            src_bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7']
        elif hls_type == 'HLS_S30':
            src_bands = ['B2', 'B3', 'B4', 'B8A', 'B11', 'B12']
        else:
            raise ValueError(f"Unsupported hls_type '{hls_type}'. Expected 'HLS_L30' or 'HLS_S30'.")

        renamed = image.select(src_bands, target_names)
        return ee.Image(renamed.copyProperties(image, image.propertyNames()))
    #scale factor for HLS
    def apply_hls_scale_factors(self, image: ee.Image) -> ee.Image:
        """
        Apply the NASA HLS v2.0 surface reflectance scale factor (`0.0001`).

        Note: In Google Earth Engine, `NASA/HLS/HLSL30/v002` and `NASA/HLS/HLSS30/v002`
        optical bands are stored with a scale factor of `0.0001` (or pre-scaled float
        in some GEE catalog views). If `multiply(0.0001)` is requested via
        `scale_reflectance=True` in `get_hls_optical_data`, this converts integer DN
        to `[0.0, 1.0]` surface reflectance.
        """
        scaled = image.multiply(0.0001).toFloat()
        return ee.Image(scaled.copyProperties(image, image.propertyNames()))
    #core function to retrieve HLS data
    def get_hls_optical_data(
        self,
        aoi,
        start_date,
        end_date,
        dataset: str = 'HLS',
        cloud_cover: int = 30,
        mask_high_aerosol: bool = True,
        scale_reflectance: bool = False,
        verbose: bool = True,
        compute_detailed_stats: bool = True,
    ):
        """
        Retrieve a filtered, Fmask-masked, and band-standardized NASA Harmonized
        Landsat Sentinel-2 (HLS v2.0) collection (`HLS`, `HLS_L30`, or `HLS_S30`).

        Parameters
        ----------
        aoi : ee.Geometry or ee.FeatureCollection
            Area of interest (must not be None).
        start_date, end_date : int or str
            Date bounds (`YYYY` or `YYYY-MM-DD`). Must not end before `2013-04-01`
            (`2015-11-28` for `HLS_S30`).
        dataset : str, optional
            One of `'HLS'` (merged L30 + S30), `'HLS_L30'`, or `'HLS_S30'`. Default `'HLS'`.
        cloud_cover : int, optional
            Maximum `CLOUD_COVERAGE` percentage (0-100). Default `30`.
        mask_high_aerosol : bool, optional
            Mask pixels with high aerosol (`Fmask` bits 6-7 == 3). Default `True`.
        scale_reflectance : bool, optional
            Multiply optical bands by `0.0001` to convert DN to `[0, 1]` reflectance.
            Default `True`.
        verbose : bool, optional
            Log collection details. Default `True`.
        compute_detailed_stats : bool, optional
            Compute detailed collection statistics. Default `True`.

        Returns
        -------
        tuple : (ee.ImageCollection, dict)
            Pre-processed collection with bands `['BLUE','GREEN','RED','NIR','SWIR1','SWIR2']`
            and metadata/statistics dictionary.
        """
        if aoi is None:
            raise ValueError("aoi must not be None — provide an ee.Geometry or ee.FeatureCollection.")

        if dataset not in self.HLS_DATASETS:
            raise ValueError(f"dataset must be one of {list(self.HLS_DATASETS.keys())}, got '{dataset}'")

        start_date = parse_date_input(start_date, is_start=True)
        end_date = parse_date_input(end_date, is_start=False)
        config = self.HLS_DATASETS[dataset]

        if end_date < config['start_date']:
            raise ValueError(
                f"{dataset} is only available from {config['start_date']} onward "
                f"(requested: {start_date} to {end_date}). Use Landsat 4-7 for earlier periods."
            )

        def _prepare_subcollection(col_id: str, hls_type: str):
            init_col = (
                ee.ImageCollection(col_id)
                .filterBounds(aoi)
                .filterDate(start_date, end_date)
            )
            filtered = (
                init_col
                .filter(ee.Filter.lt(config['cloud_property'], cloud_cover))
                .limit(MAX_COLLECTION_SIZE)
                .map(lambda img: self.mask_hls_fmask(img, mask_high_aerosol=mask_high_aerosol))
                .map(lambda img: self.rename_hls_bands(img, hls_type=hls_type))
            )
            if scale_reflectance:
                filtered = filtered.map(lambda img: self.apply_hls_scale_factors(img))
            return init_col, filtered

        if dataset == 'HLS':
            init_l30, col_l30 = _prepare_subcollection(config['l30_collection'], 'HLS_L30')
            init_s30, col_s30 = _prepare_subcollection(config['s30_collection'], 'HLS_S30')
            initial_collection = init_l30.merge(init_s30)
            collection = col_l30.merge(col_s30).limit(MAX_COLLECTION_SIZE)
        elif dataset == 'HLS_L30':
            initial_collection, collection = _prepare_subcollection(config['collection'], 'HLS_L30')
        else:
            initial_collection, collection = _prepare_subcollection(config['collection'], 'HLS_S30')

        stats_object = Reflectance_Stats()
        hls_props = [config['cloud_property'], 'system:time_start', 'system:index']
        initial_stats = stats_object.get_collection_statistics(
            initial_collection,
            compute_detailed_stats,
            cloud_property=config['cloud_property'],
            stats_properties=hls_props,
        )
        filtered_stats = stats_object.get_collection_statistics(
            collection,
            compute_detailed_stats,
            cloud_property=config['cloud_property'],
            stats_properties=hls_props,
        )

        if verbose:
            self.logger.info(f"Retrieved {config['description']} for {start_date} to {end_date}")

        collection = collection.set({
            'query_start_date': str(start_date),
            'query_end_date': str(end_date),
        })

        return collection, {
            'dataset': config['description'],
            'sensor': config['sensor'],
            'date_range_requested': f"{start_date} to {end_date}",
            'cloud_cover_threshold': cloud_cover,
            'collection_type': dataset,
            'initial_collection': initial_stats,
            'filtered_collection': filtered_stats,
            'detailed_stats_computed': compute_detailed_stats,
        }
    #core execution of Bidirectional Reflectance Distribution Function (BRDF) Harmonization
    #harmonize spectral reflectance between two imagery sensor 
    def apply_brdf_harmonization(self, image: ee.Image, sensor_type: str) -> ee.Image:
        """
        Apply Roy et al. (2016) Bidirectional Reflectance Distribution Function (BRDF) 
        Table 2 OLS linear regression coefficients to transform
        Landsat 4/5/7 TM/ETM+ Surface Reflectance (`BLUE..SWIR2`) into Landsat 8/9 OLI
        equivalent Surface Reflectance:

            rho_OLI = slope * rho_ETM + intercept

        For sensors other than `'L4'`, `'L5'`, `'L7'`, returns ``image`` unchanged.
        """
        if sensor_type not in ('L4', 'L5', 'L7'):
            return image

        bands = ['BLUE', 'GREEN', 'RED', 'NIR', 'SWIR1', 'SWIR2']
        slopes = [self.ROY2016_COEFFICIENTS['slopes'][b] for b in bands]
        intercepts = [self.ROY2016_COEFFICIENTS['intercepts'][b] for b in bands]

        slope_img = ee.Image.constant(slopes).rename(bands)
        itcp_img = ee.Image.constant(intercepts).rename(bands)

        harmonized = (
            image.select(bands)
            .multiply(slope_img)
            .add(itcp_img)
            .toFloat()
            .rename(bands)
        )
        return ee.Image(
            image
            .addBands(harmonized, bands, True)
            .copyProperties(image, image.propertyNames())
        )
    #new function
    #entry point for multitemporal anlaysis
    def get_multitemporal_optical_data(
        self,
        aoi,
        periods,
        harmonize_landsat: bool = True,
        apply_brdf: bool = False,
        apply_topo_correction: bool = False,
        dem: Optional[ee.Image] = None,
        include_thermal: bool = False,
        verbose: bool = True,
        compute_detailed_stats: bool = True,
    ) -> dict:
        """
        Acquire optical image collections across N >= 2 mapping periods.

        Supports any mix of Landsat (`L1_RAW`..`L9_SR`), Sentinel-2 (`S2_SR`, `S2_TOA`, `S2`),
        and NASA HLS (`HLS`, `HLS_L30`, `HLS_S30`). Optionally applies Roy et al. (2016)
        TM/ETM+ -> OLI spectral adjustment, BRDF c-factor normalization (`BRDF_Correction`),
        and SCSc topographic correction (`Topographic_Correction`).

        Parameters
        ----------
        aoi : ee.Geometry or ee.FeatureCollection
            Area of interest.
        periods : dict or list of dict/PeriodConfig
            Either `{label: {'start_date', 'end_date', 'sensor', 'cloud_cover', ...}}`
            or a list of period dicts/objects each carrying a `'label'`.
            Must contain at least 2 distinct periods.
        harmonize_landsat : bool, optional
            Apply Roy et al. (2016) OLS harmonization to Landsat 4/5/7 SR collections.
            Default `True`.
        apply_brdf : bool, optional
            Apply `BRDF_Correction` to Landsat 4-9 and Sentinel-2 collections.
            Skipped automatically for HLS (already NBAR) and Landsat 1-3 MSS.
            Default `False`.
        apply_topo_correction : bool, optional
            Apply `Topographic_Correction` (SCSc) to collections (except MSS).
            Default `False`.
        dem : ee.Image, optional
            Custom DEM for topographic correction (defaults to NASADEM).
        include_thermal : bool, optional
            Also fetch corresponding Landsat TOA thermal collection when available.
            Default `False`.
        verbose : bool, optional
            Log progress details. Default `True`.
        compute_detailed_stats : bool, optional
            Compute detailed collection statistics per period. Default `True`.

        Returns
        -------
        dict
            `{'periods': {label: {...}}, 'period_order': [label, ...]}` sorted
            chronologically by `(order_index, start_date)`.
        """
        if aoi is None:
            raise ValueError("aoi must not be None.")

        # Normalize periods into a list of dicts
        normalized = []
        if isinstance(periods, dict):
            for idx, (label, cfg) in enumerate(periods.items()):
                item = dict(cfg) if isinstance(cfg, dict) else vars(cfg).copy()
                item['label'] = label
                item.setdefault('order_index', idx)
                normalized.append(item)
        elif isinstance(periods, (list, tuple)):
            for idx, cfg in enumerate(periods):
                item = dict(cfg) if isinstance(cfg, dict) else vars(cfg).copy()
                if 'label' not in item or not item['label']:
                    raise ValueError(f"Period at index {idx} is missing a 'label'.")
                item.setdefault('order_index', idx)
                normalized.append(item)
        else:
            raise ValueError("periods must be a dict or a list/tuple of period configurations.")

        if len(normalized) < 2:
            raise ValueError(f"Multitemporal analysis requires at least 2 periods, got {len(normalized)}.")

        labels = [p['label'] for p in normalized]
        if len(set(labels)) != len(labels):
            raise ValueError(f"Period labels must be unique, got: {labels}")

        # Sort chronologically by parsed start_date, breaking ties with order_index
        for p in normalized:
            p['_parsed_start'] = parse_date_input(p['start_date'], is_start=True)
            p['_parsed_end'] = parse_date_input(p['end_date'], is_start=False)

        normalized.sort(key=lambda p: (p['_parsed_start'], p.get('order_index', 0)))
        period_order = [p['label'] for p in normalized]

        brdf_corrector = None
        topo_corrector = None
        if apply_brdf or apply_topo_correction:
            from .imagery_correction import BRDF_Correction, Topographic_Correction
            if apply_brdf:
                brdf_corrector = BRDF_Correction(log_level=self.logger.level)
            if apply_topo_correction:
                topo_corrector = Topographic_Correction(log_level=self.logger.level)

        results = {}
        for p in normalized:
            label = p['label']
            sensor_key = p.get('sensor') or p.get('optical_data') or 'L8_SR'
            cloud_cover = p.get('cloud_cover', 30)
            start_d = p['_parsed_start']
            end_d = p['_parsed_end']
            thermal_col = None

            if sensor_key in self.OPTICAL_DATASETS:
                col, stats = self.get_optical_data(
                    aoi=aoi,
                    start_date=start_d,
                    end_date=end_d,
                    optical_data=sensor_key,
                    cloud_cover=cloud_cover,
                    verbose=verbose,
                    compute_detailed_stats=compute_detailed_stats,
                )
                sensor_code = self.OPTICAL_DATASETS[sensor_key]['sensor']
                if harmonize_landsat and sensor_code in ('L4', 'L5', 'L7'):
                    def _make_harmonize_fn(sc):
                        return lambda img: self.apply_brdf_harmonization(img, sc)

                    col = col.map(_make_harmonize_fn(sensor_code))

                if apply_brdf and brdf_corrector is not None and sensor_code in ('L4', 'L5', 'L7', 'L8', 'L9'):
                    col = brdf_corrector.correct_collection(col, verbose=verbose)
                if apply_topo_correction and topo_corrector is not None and sensor_code in ('L4', 'L5', 'L7', 'L8', 'L9'):
                    col = topo_corrector.correct_collection(col, dem=dem, scale=30, verbose=verbose)

                if include_thermal and self.has_thermal_capability(sensor_key):
                    thermal_key = sensor_key.replace('_SR', '_TOA')
                    thermal_col, _ = self.get_thermal_bands(
                        aoi=aoi,
                        start_date=start_d,
                        end_date=end_d,
                        thermal_data=thermal_key,
                        cloud_cover=cloud_cover,
                        verbose=False,
                        compute_detailed_stats=False,
                    )

            elif sensor_key in self.S2_DATASETS or sensor_key == 'S2':
                cs_thresh = p.get('cs_threshold', 0.7)
                col, stats = self.get_s2_optical_data(
                    aoi=aoi,
                    start_date=start_d,
                    end_date=end_d,
                    cloud_cover=cloud_cover,
                    cs_threshold=cs_thresh,
                    verbose=verbose,
                    compute_detailed_stats=compute_detailed_stats,
                )
                if apply_brdf and brdf_corrector is not None:
                    col = brdf_corrector.correct_collection(
                        col, band_list=brdf_corrector.S2_OPTICAL_BANDS, verbose=verbose
                    )
                if apply_topo_correction and topo_corrector is not None:
                    col = topo_corrector.correct_collection(col, dem=dem, scale=10, band_list=topo_corrector.S2_OPTICAL_BANDS, verbose=verbose)

            elif sensor_key in self.HLS_DATASETS:
                col, stats = self.get_hls_optical_data(
                    aoi=aoi,
                    start_date=start_d,
                    end_date=end_d,
                    dataset=sensor_key,
                    cloud_cover=cloud_cover,
                    verbose=verbose,
                    compute_detailed_stats=compute_detailed_stats,
                )
                # Note: HLS is already NBAR BRDF-normalized; only apply topo correction if requested
                if apply_topo_correction and topo_corrector is not None:
                    col = topo_corrector.correct_collection(col, dem=dem, scale=30, verbose=verbose)
            else:
                raise ValueError(
                    f"Unsupported sensor '{sensor_key}' for period '{label}'. "
                    f"Supported: {list(self.OPTICAL_DATASETS.keys()) + list(self.S2_DATASETS.keys()) + list(self.HLS_DATASETS.keys())}"
                )

            if hasattr(col, 'set'):
                col = col.set({
                    'query_start_date': str(start_d),
                    'query_end_date': str(end_d),
                })

            results[label] = {
                'collection': col,
                'thermal_collection': thermal_col,
                'stats': stats,
                'sensor': sensor_key,
                'start_date': start_d,
                'end_date': end_d,
                'cloud_cover': cloud_cover,
            }

        return {
            'periods': results,
            'period_order': period_order,
        }
class Reflectance_Stats:
    """
    Core methods for fetching image collection statistics and reporting
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
    def get_collection_statistics(
        self,
        collection,
        compute_stats=True,
        print_report=False,
        cloud_property='CLOUD_COVER_LAND',
        stats_properties=None,
        dataset_key=None,
    ):
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
        stats_properties : list[str], optional
            Image properties to aggregate. When omitted, the standard
            properties for ``dataset_key`` are used, or the default set.
        dataset_key : str, optional
            Optical dataset key used to select its configured statistics
            properties.

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
            - 'path_row_tiles'
            - 'unique_tiles'
            - 'individual_dates'
            - 'Scene_ids'

        Example
        -------
        >>> stats = Reflectance_Stats().get_collection_statistics(collection, compute_stats=True)
        >>> print(stats['total_images'], stats['date_range'])
        """
        # Resolve the list of properties to aggregate in a single batched call.
        # Priority: explicit stats_properties > dataset_key lookup > defaults.
        if stats_properties is not None:
            props = stats_properties
        elif dataset_key is not None:
            config = Reflectance_Data.OPTICAL_DATASETS.get(dataset_key, {})
            props = config.get('stats_properties', [
                cloud_property, 'system:time_start', 'system:index',
                'WRS_PATH', 'WRS_ROW'
            ])
        else:
            props = [cloud_property, 'system:time_start', 'system:index',
                     'WRS_PATH', 'WRS_ROW']

        try:
            size = collection.size()
            if compute_stats:
                total_images = size.getInfo()
                if total_images > 0:
                    #Improve API calls, reducing from 5 to 1 by using aggregate_array for all properties at once
                    agg_dict = {prop: collection.aggregate_array(prop) for prop in props}
                    aggregated = ee.Dictionary(agg_dict).getInfo()

                    #Unpack with safe defaults (empty list if a property is missing)
                    raw = {prop: aggregated.get(prop, []) for prop in props} # type: ignore

                    #Derive the fields expected by the rest of the codebase
                    cloud_prop = next(
                        (p for p in props if 'cloud' in p.lower()), None
                    )
                    cloud_values = raw.get(cloud_prop, []) if cloud_prop else []

                    dates_raw = raw.get('system:time_start', [])
                    dates_readable = [
                        datetime.fromtimestamp(d / 1000).strftime('%Y-%m-%d')
                        for d in dates_raw
                    ] if dates_raw else []
                    date_range = (
                        f"{min(dates_readable)} to {max(dates_readable)}"
                        if dates_readable else "No date information"
                    )

                    scene_id = raw.get('system:index', [])

                    # Build path/row pairs when both properties are present
                    paths = raw.get('WRS_PATH', [])
                    rows = raw.get('WRS_ROW', [])
                    if paths and rows:
                        path_rows = sorted(set(zip(paths, rows)))
                    else:
                        path_rows = []

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
                        'Scene_ids': []
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
    #Function to calculate data coverage (pixel validity) within AOI, serve as alternative to calculate cloud cover within AOI
    def calculate_data_coverage(self, composite_image, aoi, scale=DEFAULT_COVERAGE_SCALE, max_pixels=DEFAULT_MAX_PIXELS, verbose=True):
        """
        Calculate data coverage (valid pixels) percentage within AOI.
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
        
        try:
            #Create a mask of valid (non-masked) pixels
            #Use the first band to check for valid data (all bands should have same mask)
            first_band = composite_image.select(0)
            valid_mask = first_band.mask()
            #Calculate pixel area
            pixel_area = ee.Image.pixelArea()
            #Total area in AOI
            total_area = pixel_area.reduceRegion(reducer=ee.Reducer.sum(),geometry=geometry,scale=scale,maxPixels=max_pixels, # type: ignore
                bestEffort=True,
                tileScale=DEFAULT_TILE_SCALE
            )
            #Valid (unmasked) area
            valid_area_img = valid_mask.multiply(pixel_area).rename('valid_area')
            valid_area = valid_area_img.reduceRegion(reducer=ee.Reducer.sum(),geometry=geometry,scale=scale,maxPixels=max_pixels,
                bestEffort=True,
                tileScale=DEFAULT_TILE_SCALE
            )
            #Optimize API Calls
            area_results = ee.Dictionary({
                'area': total_area.get('area'),
                'valid_area': valid_area.get('valid_area')
            }).getInfo()

            total_val = area_results.get('area') # type: ignore
            valid_val = area_results.get('valid_area') # type: ignore 

            #Null valid area handling: default to 0 and warn
            if valid_val is None:
                self.logger.warning("No valid pixels found in AOI - setting to 0")
                valid_val = 0

            #Calculate coverage percentage (client-side)
            total_val = max(total_val, 1)  # Avoid division by zero
            coverage_val = (valid_val / total_val) * 100
            gap_val = 100 - coverage_val
            #Make sure the value 0-100
            coverage_val = max(0, min(100, coverage_val))
            gap_val = max(0, min(100, gap_val))
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

        except ee.EEException as e:
            error_msg = str(e).lower()
            if 'timeout' in error_msg or 'deadline' in error_msg:
                self.logger.error(
                    f"Coverage calculation timed out (scale={scale}m): {e}"
                )
                raise TimeoutError(
                    f"Coverage calculation timed out. Try using a larger scale "
                    f"(current: {scale}m) or smaller AOI."
                ) from e
            else:
                self.logger.error(
                    f"Earth Engine error during coverage calculation: {e}"
                )
                raise RuntimeError(
                    f"Earth Engine error during coverage calculation: {e}"
                ) from e

        except Exception as e:
            self.logger.error(
                f"Unexpected error in calculate_data_coverage(): {e}"
            )
            raise
    #Quality mosaic for stacking multiple scene and then clip them. This procedure stacked all of the imagery regardless of the pixel value
    #if not used in the future, this function should be removed
    def get_quality_mosaic(self, collection, aoi, quality_band='NDVI',
                          calculate_coverage=False, coverage_scale=30,
                          sharpen: bool = False, verbose=True):
        """
        Create a mosaic that selects the best available pixels across the AOI.
        Uses qualityMosaic to automatically select pixels with highest quality metric.
        
        Parameters
        ----------
        collection : ee.ImageCollection
            Filtered image collection from Reflectance_Data.
        aoi : ee.Geometry or ee.FeatureCollection
            Area of interest for clipping.
        quality_band : str
            Band to use for quality assessment. Options:
            - 'NDVI': Normalized Difference Vegetation Index (selects pixels with highest NDVI)
            - 'NIR': Near-infrared band (general purpose)
        calculate_coverage : bool
            If True, calculate data coverage within AOI (default: False).
            Note: This triggers a client-side computation and may be slow for large areas.
        coverage_scale : int
            Pixel scale in meters for coverage calculation (default: 30m).
            Use larger values (90m+) for faster computation on large areas.
        sharpen : bool
            If True, apply HPF pan-sharpening to upsample 20m Sentinel-2 bands
            (RED_EDGE1–SWIR2) to 10m after mosaicking. Only effective for
            Sentinel-2 collections; ignored with a warning for Landsat. Default is False.
        verbose : bool
            If True, log detailed information (default: True).
            
        Returns
        -------
        tuple : (ee.Image, dict) if calculate_coverage=True, else ee.Image
            - Quality mosaic image clipped to AOI with best available pixels.
            - Coverage statistics dict (only if calculate_coverage=True).
            
        Example
        -------
        >>> data_fetcher = Reflectance_Data()
        >>> collection, stats = data_fetcher.get_optical_data(aoi, 2020, 2020, 'L8_SR')
        >>> image_processor = final_Image()
        >>> quality_image = image_processor.get_quality_mosaic(collection, aoi, quality_band='NDVI')
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

        # Optional HPF sharpening (Sentinel-2 only) ---
        if sharpen:
            s2_bands = ['RED_EDGE1', 'RED_EDGE2', 'RED_EDGE3', 'RED_EDGE4', 'SWIR1', 'SWIR2']
            band_names = clipped.bandNames().getInfo()
            if all(b in band_names for b in s2_bands): # type: ignore
                rd = Reflectance_Data.__new__(Reflectance_Data)
                rd.logger = self.logger
                clipped = rd.sharpen_s2_bands(clipped, aoi=aoi)
                if clipped is None:
                    raise RuntimeError("sharpen_s2_bands failed — check logs for details")
                if verbose:
                    self.logger.info("HPF sharpening applied to 20m Sentinel-2 bands")
            else:
                if verbose:
                    self.logger.warning(
                        "sharpen=True ignored: collection does not contain Sentinel-2 20m bands "
                        f"({', '.join(s2_bands)})"
                    )

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
    def get_temporal_composite(self, collection, aoi, reducer='median', calculate_coverage=False,
                              coverage_scale=30, sharpen: bool = False, verbose=True,
                              return_image_only: bool = False):
        """
        Create a temporal composite from image collection using specified reducer.
        Output is always clipped to the AOI.
        
        Parameters
        ----------
        collection : ee.ImageCollection
            Filtered image collection from Reflectance_Data.
        aoi : ee.Geometry or ee.FeatureCollection
            Area of interest for clipping.
        reducer : str or ee.Reducer
            Reduction method: 'median', 'mean', 'min', 'max', 'percentile_<n>'.
        calculate_coverage : bool
            If True, calculate data coverage within AOI (default: False).
            Note: This triggers a client-side computation and may be slow for large areas.
        coverage_scale : int
            Pixel scale in meters for coverage calculation (default: 30m).
            Use larger values (90m+) for faster computation on large areas.
        sharpen : bool
            If True, apply HPF pan-sharpening to upsample 20m Sentinel-2 bands
            (RED_EDGE1–SWIR2) to 10m after compositing. Only effective for
            Sentinel-2 collections that contain those band names; ignored silently
            for Landsat. Default is False.
        verbose : bool
            If True, log detailed information (default: True).
            
        Returns
        -------
        dict
            Always returns a dictionary with keys:
            - 'image' : ee.Image
                Composite image clipped to AOI with original band names (NIR, RED, etc.)
            - 'metadata' : dict
                Dictionary with composite_start_date, composite_end_date,
                composite_count, composite_reducer
            - 'coverage' : dict or None
                Coverage statistics if calculate_coverage=True, else None
            
        Example
        -------
        >>> data_fetcher = Reflectance_Data()
        >>> collection, stats = data_fetcher.get_optical_data(aoi, 2020, 2020, 'L8_SR')
        >>> image_processor = final_Image()
        >>> composite = image_processor.get_temporal_composite(collection, aoi, reducer='median')
        >>> # Sentinel-2 with sharpening
        >>> s2_col, _ = Reflectance_Data().get_s2_optical_data(aoi, 2022, 2023)
        >>> sharpened = image_processor.get_temporal_composite(s2_col, aoi, sharpen=True)
        >>> # With coverage calculation
        >>> result = image_processor.get_temporal_composite(
        ...     collection, aoi, reducer='median', calculate_coverage=True
        ... )
        >>> composite = result['image']
        >>> coverage = result['coverage']
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

        metadata = {
            'composite_start_date': start_date,
            'composite_end_date': end_date,
            'composite_count': size_server,
            'composite_reducer': str(reducer),
        }
        
        if verbose:
            #Only call the client side info if needed
            start_date_str = start_date.getInfo()
            end_date_str = end_date.getInfo()
            self.logger.info(f"Composite created from {start_date_str} to {end_date_str}")

        # --- Optional HPF sharpening (Sentinel-2 only) ---
        if sharpen:
            s2_bands = ['RED_EDGE1', 'RED_EDGE2', 'RED_EDGE3', 'RED_EDGE4', 'SWIR1', 'SWIR2']
            band_names = composite.bandNames().getInfo()
            if all(b in band_names for b in s2_bands):
                rd = Reflectance_Data.__new__(Reflectance_Data)
                rd.logger = self.logger
                composite = rd.sharpen_s2_bands(composite, aoi=aoi)
                if composite is None:
                    raise RuntimeError("sharpen_s2_bands failed — check logs for details")
                if verbose:
                    self.logger.info("HPF sharpening applied to 20m Sentinel-2 bands")
            else:
                if verbose:
                    self.logger.warning(
                        "sharpen=True ignored: collection does not contain Sentinel-2 20m bands "
                        f"({', '.join(s2_bands)})"
                    )

        # Calculate data coverage if requested
        coverage_stats = None
        if calculate_coverage:
            coverage_stats = self.calculate_data_coverage(
                composite, aoi, scale=coverage_scale, verbose=verbose
            )

        if return_image_only:
            return composite

        return {
            'image': composite,
            'metadata': metadata,
            'coverage': coverage_stats
        }

    #CDF Histogram Matching & Multitemporal Composite Harmonization
    #multitemporal analysis function: histogram matching
    def histogram_match(self, source_image: ee.Image, reference_image: ee.Image, 
                        region, bands: Optional[List[str]] = None, scale: int = 30, max_buckets: int = 256,
    ) -> ee.Image:
        """
        Perform server-side piecewise-linear Cumulative Distribution Function (CDF)
        histogram matching of ``source_image`` onto ``reference_image`` within ``region``.

        Uses a single ``reduceRegion(ee.Reducer.histogram())`` call per image across
        all target ``bands`` (avoiding per-band network or aggregation loops) and
        maps each source band through ``interpolate(src_vals, src_cdf)`` followed by
        ``interpolate(ref_cdf, ref_vals)``.

        Parameters
        ----------
        source_image : ee.Image
            Historical or target composite image to be radiometrically matched.
        reference_image : ee.Image
            Baseline/reference composite image defining the target radiometric distribution.
        region : ee.Geometry or ee.FeatureCollection
            Area of interest over which histograms are computed.
        bands : list of str, optional
            Optical bands to histogram-match. If None, defaults to
            `['BLUE', 'GREEN', 'RED', 'NIR', 'SWIR1', 'SWIR2']`.
        scale : int, optional
            Sampling scale in meters for histogram computation. Default `30`.
        max_buckets : int, optional
            Maximum histogram bins. Default `256`.

        Returns
        -------
        ee.Image
            ``source_image`` with ``bands`` replaced by their histogram-matched
            counterparts and all non-matched bands & properties preserved.
        Reference
        https://developers.google.com/earth-engine/tutorials/community/histogram-matching
        """
        if bands is None:
            bands = ['BLUE', 'GREEN', 'RED', 'NIR', 'SWIR1', 'SWIR2']

        if not bands:
            return source_image

        geometry = region.geometry() if isinstance(region, ee.FeatureCollection) else region

        src_hist_dict = source_image.select(bands).reduceRegion(
            reducer=ee.Reducer.histogram(maxBuckets=max_buckets),
            geometry=geometry,
            scale=scale,
            maxPixels=DEFAULT_MAX_PIXELS,
            bestEffort=True,
            tileScale=DEFAULT_TILE_SCALE,
        )
        ref_hist_dict = reference_image.select(bands).reduceRegion(
            reducer=ee.Reducer.histogram(maxBuckets=max_buckets),
            geometry=geometry,
            scale=scale,
            maxPixels=DEFAULT_MAX_PIXELS,
            bestEffort=True,
            tileScale=DEFAULT_TILE_SCALE,
        )

        default_hist = ee.Dictionary({
            'bucketMeans': [0.0, 1.0],
            'histogram': [1.0, 1.0],
        })

        def _extract_vals_and_cdf(hist_dict: ee.Dictionary, band_name: str):
            raw_hist = ee.Dictionary(
                ee.List([hist_dict.get(band_name), default_hist]).reduce(ee.Reducer.firstNonNull())
            )
            means = ee.List(
                ee.List([raw_hist.get('bucketMeans'), [0.0, 1.0]]).reduce(ee.Reducer.firstNonNull())
            )
            counts_list = ee.List(
                ee.List([raw_hist.get('histogram'), [1.0, 1.0]]).reduce(ee.Reducer.firstNonNull())
            )
            # Enforce at least 2 bins and strictly positive counts (1e-6 floor)
            # so cumulative sum is strictly monotonic for ee.Image.interpolate()
            safe_means = ee.List(ee.Algorithms.If(means.size().gte(2), means, [0.0, 1.0]))
            safe_counts = ee.List(ee.Algorithms.If(counts_list.size().gte(2), counts_list, [1.0, 1.0]))

            counts_arr = ee.Array(safe_counts).max(1e-6)
            cum_arr = counts_arr.accum(0)
            total = cum_arr.get([-1]).max(1e-6)
            cdf_list = cum_arr.divide(total).toList()
            return safe_means, cdf_list

        matched_bands = []
        for band in bands:
            src_vals, src_cdf = _extract_vals_and_cdf(src_hist_dict, band)
            ref_vals, ref_cdf = _extract_vals_and_cdf(ref_hist_dict, band)

            matched = (
                source_image.select(band)
                .interpolate(src_vals, src_cdf, 'clamp')
                .interpolate(ref_cdf, ref_vals, 'clamp')
                .toFloat()
                .rename(band)
            )
            matched_bands.append(matched)

        matched_stack = ee.Image.cat(matched_bands)
        return ee.Image(
            source_image
            .addBands(matched_stack.select(bands), bands, True)
            .copyProperties(source_image, source_image.propertyNames())
        )

    @staticmethod
    def _expected_bands_for_sensor(sensor_key: Optional[str]) -> Optional[List[str]]:
        """Return known standardized optical band names for a sensor key."""
        if sensor_key in ('L8_SR', 'L9_SR'):
            return ['AEROSOL', 'BLUE', 'GREEN', 'RED', 'NIR', 'SWIR1', 'SWIR2']
        if sensor_key in ('L4_SR', 'L5_SR', 'L7_SR', 'HLS', 'HLS_L30', 'HLS_S30'):
            return ['BLUE', 'GREEN', 'RED', 'NIR', 'SWIR1', 'SWIR2']
        if sensor_key in ('S2_SR', 'S2_TOA', 'S2'):
            return [
                'BLUE', 'GREEN', 'RED', 'RED_EDGE1', 'RED_EDGE2',
                'RED_EDGE3', 'NIR', 'RED_EDGE4', 'SWIR1', 'SWIR2',
            ]
        if sensor_key in ('L1_RAW', 'L2_RAW', 'L3_RAW'):
            return ['GREEN', 'RED', 'NIR1', 'NIR2']
        return None
    #multitemporal analysis function: batch process composites
    def get_multitemporal_composites(self, collections_by_period: dict, aoi,
        reference_period: str = 'baseline', #use for histogram macthing reference 
        reducer: str = 'median', #median reducer for both
        drop_aerosol: bool = True, #drop aerosol from landsat 8/9, not usefull for older sensor
        harmonize_sensors: bool = True, 
        histogram_match: bool = True,
        radiometric_matching: Optional[str] = None,
        pif_metric: str = 'sid',
        pif_percentile: float = 10.0,
        scale: int = 30,
        max_buckets: int = 256,
        sharpen_s2: bool = False,
        calculate_coverage: bool = False,
        coverage_scale: int = 30, #change to 10m/20m for sentinel
        verbose: bool = True,
    ) -> dict:
        """
        Build, harmonize, and optionally histogram-match temporal composites across
        N >= 2 mapping periods.

        Cross-sensor harmonization steps performed when ``harmonize_sensors=True``:
        1. Drops ``'AEROSOL'`` from Landsat 8/9 composites when ``drop_aerosol=True``.
        2. Scales ``S2_SR`` composites (`0-10000` DN) to `[0, 1]` surface reflectance
           when paired with reflectance-scaled datasets (Landsat SR, HLS, `S2_TOA`).
        3. Aliases Landsat 1-3 MSS ``'NIR2'`` (0.8-1.1 um) -> ``'NIR'`` when paired
           with non-MSS sensors.
        4. Computes the ordered ``common_bands`` intersection across all periods and
           selects ``common_bands`` on every period's composite.
        5. When ``histogram_match=True``, matches all common optical bands of each
           non-reference period's composite onto ``reference_period`` within ``aoi``.
        """
        # Unwrap output of get_multitemporal_optical_data() if passed directly
        if 'periods' in collections_by_period and isinstance(collections_by_period['periods'], dict):
            period_order = collections_by_period.get(
                'period_order', list(collections_by_period['periods'].keys())
            )
            period_entries = collections_by_period['periods']
        else:
            period_entries = collections_by_period
            period_order = list(period_entries.keys())

        if len(period_order) < 2:
            raise ValueError(
                f"get_multitemporal_composites requires at least 2 periods, got {len(period_order)}."
            )

        if reference_period not in period_entries:
            fallback_ref = period_order[-1]
            self.logger.warning(
                f"reference_period '{reference_period}' not found in periods {period_order}; "
                f"defaulting to '{fallback_ref}'."
            )
            reference_period = fallback_ref

        geometry = aoi.geometry() if isinstance(aoi, ee.FeatureCollection) else aoi

        raw_composites = {}
        metadata_by_period = {}
        coverage_by_period = {}
        sensor_by_period = {}
        bands_by_period = {}

        # Detect whether any non-MSS or reflectance-scaled sensor is in the mix
        for label in period_order:
            entry = period_entries[label]
            if isinstance(entry, dict) and 'collection' in entry:
                sensor_by_period[label] = entry.get('sensor')
            else:
                sensor_by_period[label] = None

        has_non_mss = any(
            s not in ('L1_RAW', 'L2_RAW', 'L3_RAW') for s in sensor_by_period.values() if s is not None
        )
        has_reflectance_sensor = any(
            s in ('L4_SR', 'L5_SR', 'L7_SR', 'L8_SR', 'L9_SR', 'HLS', 'HLS_L30', 'HLS_S30', 'S2_TOA')
            for s in sensor_by_period.values() if s is not None
        )

        for label in period_order:
            entry = period_entries[label]
            if isinstance(entry, dict) and 'collection' in entry:
                col = entry['collection']
                thermal_col = entry.get('thermal_collection')
                sensor_key = entry.get('sensor')
            else:
                col = entry
                thermal_col = None
                sensor_key = None

            comp_res = self.get_temporal_composite(
                collection=col,
                aoi=aoi,
                reducer=reducer,
                calculate_coverage=calculate_coverage,
                coverage_scale=coverage_scale,
                sharpen=sharpen_s2,
                verbose=verbose,
            )
            comp_img = comp_res['image'] # type: ignore

            # Scale S2_SR DN (0-10000) to [0, 1] reflectance when harmonized with SR datasets
            if harmonize_sensors and sensor_key in ('S2_SR', 'S2') and (has_reflectance_sensor or histogram_match or (radiometric_matching in ('cdf', 'pif'))):
                comp_img = ee.Image(
                    comp_img.multiply(0.0001).toFloat().copyProperties(comp_img, comp_img.propertyNames())
                )

            expected_bands = self._expected_bands_for_sensor(sensor_key)
            if expected_bands is None:
                expected_bands = list(comp_img.bandNames().getInfo()) # type: ignore
            else:
                expected_bands = list(expected_bands)

            # Drop AEROSOL band in multitemporal stacks
            if drop_aerosol and 'AEROSOL' in expected_bands:
                expected_bands = [b for b in expected_bands if b != 'AEROSOL']
                comp_img = comp_img.select(expected_bands)

            # Alias MSS NIR2 -> NIR when paired with non-MSS sensors
            if harmonize_sensors and sensor_key in ('L1_RAW', 'L2_RAW', 'L3_RAW') and has_non_mss:
                if 'NIR2' in expected_bands and 'NIR' not in expected_bands:
                    nir_alias = comp_img.select(['NIR2'], ['NIR'])
                    comp_img = comp_img.addBands(nir_alias)
                    expected_bands.append('NIR')

            # Attach thermal median if provided
            if thermal_col is not None:
                thermal_median = thermal_col.median().clip(geometry).rename('THERMAL')
                comp_img = comp_img.addBands(thermal_median)
                if 'THERMAL' not in expected_bands:
                    expected_bands.append('THERMAL')

            raw_composites[label] = comp_img.toFloat()
            metadata_by_period[label] = comp_res['metadata'] # type: ignore
            coverage_by_period[label] = comp_res['coverage'] # type: ignore
            bands_by_period[label] = expected_bands

        # Compute canonical ordered common band intersection across all periods
        canonical_order = [
            'BLUE', 'GREEN', 'RED', 'RED_EDGE1', 'RED_EDGE2',
            'RED_EDGE3', 'NIR', 'RED_EDGE4', 'SWIR1', 'SWIR2',
            'NIR1', 'NIR2', 'THERMAL',
        ]
        common_set = set(bands_by_period[period_order[0]])
        for label in period_order[1:]:
            common_set &= set(bands_by_period[label])

        common_bands = [b for b in canonical_order if b in common_set]
        for b in bands_by_period[period_order[0]]:
            if b in common_set and b not in common_bands:
                common_bands.append(b)

        if harmonize_sensors and not common_bands:
            raise ValueError(
                f"No common optical bands found across periods: {bands_by_period}"
            )

        harmonized_composites = {}
        for label in period_order:
            img = raw_composites[label]
            if harmonize_sensors and common_bands:
                img = img.select(common_bands)
            harmonized_composites[label] = img

        # Resolve relative radiometric matching mode ('cdf', 'pif', or 'none')
        if radiometric_matching is not None:
            rm_mode = str(radiometric_matching).strip().lower()
            if rm_mode not in ('cdf', 'pif', 'none'):
                raise ValueError(
                    f"Unsupported radiometric_matching '{radiometric_matching}'. Expected 'cdf', 'pif', or 'none'."
                )
        else:
            rm_mode = 'cdf' if histogram_match else 'none'

        if rm_mode in ('cdf', 'pif'):
            ref_img = harmonized_composites[reference_period]
            match_bands = [b for b in common_bands if b != 'THERMAL']
            pif_norm = None
            if rm_mode == 'pif':
                from .imagery_correction import PIF_Normalization
                pif_norm = PIF_Normalization(log_level=self.logger.level)
            for label in period_order:
                if label == reference_period:
                    continue
                if rm_mode == 'cdf':
                    harmonized_composites[label] = self.histogram_match(
                        source_image=harmonized_composites[label],
                        reference_image=ref_img,
                        region=geometry,
                        bands=match_bands,
                        scale=scale,
                        max_buckets=max_buckets,
                    )
                else:
                    assert pif_norm is not None
                    harmonized_composites[label] = pif_norm.normalize_image(
                        source_image=harmonized_composites[label],
                        reference_image=ref_img,
                        region=geometry,
                        bands=match_bands,
                        metric=pif_metric,
                        pif_percentile=pif_percentile,
                        scale=scale,
                        verbose=verbose,
                    )

        return {
            'composites': harmonized_composites,
            'raw_composites': raw_composites,
            'metadata': metadata_by_period,
            'coverage': coverage_by_period,
            'common_bands': common_bands,
            'reference_period': reference_period,
            'period_order': period_order,
            'histogram_matched': rm_mode == 'cdf',
            'radiometric_matching': rm_mode,
        }
    #wrapper for landtrendr temporal segmentation
    def get_landtrendr_composites(self, aoi, periods, reference_period: str = 'baseline', 
                                  driver_index: str = 'NBR',
        min_years: int = 6,
        seasonal_window=None,
        cloud_cover: float = 40.0,
        reducer: str = 'median',
        apply_brdf: bool = False,
        apply_topo_correction: bool = False,
        dem=None,
        lt_params=None,
        lt_image=None,
        verbose: bool = True,
    ) -> dict:
        """Thin Module 1 entry point delegating to ``LandTrendrCompositor``."""
        from .temporal_segmentation import LandTrendr_fit
        compositor = LandTrendr_fit(log_level=self.logger.level)
        return compositor.generate_multitemporal_composites(
            aoi=aoi,
            periods=periods,
            reference_period=reference_period,
            driver_index=driver_index,
            min_years=min_years,
            seasonal_window=seasonal_window,
            cloud_cover=cloud_cover,
            reducer=reducer,
            apply_brdf=apply_brdf,
            apply_topo_correction=apply_topo_correction,
            dem=dem,
            lt_params=lt_params,
            lt_image=lt_image,
            verbose=verbose,
        )
#Class helper for managing visualization parameters and band combinations originally
class Vis_Params:
    """
    Manages band combination presets and visualization parameters for Landsat composites.
    """
    #Preset band combinations with default visualization parameters.
    BAND_COMBINATIONS = {
        "True Color (RGB)": {
            'bands': ['RED', 'GREEN', 'BLUE'],
            'min': 0.0,
            'max': 0.3,
            'gamma': 1.4
        },
        "False Color Infrared (NIR/Red/Green)": {
            'bands': ['NIR', 'RED', 'GREEN'],
            'min': 0.0,
            'max': 0.4,
            'gamma': 1.1
        },
        "Short-wave Infrared (SWIR2/NIR/RED)": {
            'bands': ['SWIR2', 'NIR', 'RED'],
            'min': 0.0,
            'max': 0.4,
            'gamma': 1.2
        },
        "Land/Water (NIR/SWIR1/RED)": {
            'bands': ['NIR', 'SWIR1', 'RED'],
            'min': 0.0,
            'max': 0.4,
            'gamma': [0.95, 1.1, 1.0]
        },
        "Buat kombinasi saluran": {
            'bands': ['RED', 'GREEN', 'BLUE'],
            'min': 0.0,
            'max': 0.4,
            'gamma': 1.0
        }
    }

    #Default thermal visualization parameters.
    THERMAL_VIS = {
        'min': 286,
        'max': 300,
        'gamma': 0.4
    }
    #class method since it interact with class level variables (band combinations)
    @classmethod
    def get_combination_names(cls):
        """Return the list of available preset combination names.

        Returns
        -------
        list[str]
            Ordered list of preset names.
        """
        return list(cls.BAND_COMBINATIONS.keys())

    @classmethod
    def get_vis_params(cls, combination_name: str) -> dict:
        """Return a copy of the visualization parameters for a given preset.

        Parameters
        ----------
        combination_name : str
            One of the keys in :attr:`BAND_COMBINATIONS`.

        Returns
        -------
        dict
            Visualization parameters with keys ``bands``, ``min``, ``max``, ``gamma``.

        Raises
        ------
        KeyError
            If ``combination_name`` is not a recognised preset.
        """
        if combination_name not in cls.BAND_COMBINATIONS:
            raise KeyError(
                f"Unknown combination '{combination_name}'. "
                f"Available: {cls.get_combination_names()}"
            )
        return cls.BAND_COMBINATIONS[combination_name].copy()
    #static method since it does not interact with class level variables and is a utility function for building custom visualization parameters
    @staticmethod
    def build_custom_vis_params(
        bands: list,
        min_val: float = 0.0,
        max_val: float = 0.4,
        gamma=1.0
    ) -> dict:
        """Build a visualization parameter dict for a custom band selection.

        Parameters
        ----------
        bands : list[str]
            1 or 3 band names (grayscale or RGB).
        min_val : float
            Minimum display value.
        max_val : float
            Maximum display value.
        gamma : float or list[float]
            Single gamma or per-channel list for RGB.

        Returns
        -------
        dict
            Visualization parameters ready for ``geemap.addLayer``.

        Raises
        ------
        ValueError
            If ``bands`` does not contain 1 or 3 entries.
        """
        if len(bands) not in (1, 3):
            raise ValueError("bands must contain exactly 1 (grayscale) or 3 (RGB) entries")
        return {
            'bands': bands,
            'min': min_val,
            'max': max_val,
            'gamma': gamma
        }

    @classmethod
    def get_thermal_vis_params(cls) -> dict:
        """Return the default thermal visualization parameters.

        Returns
        -------
        dict
            Thermal visualization parameters with keys ``min``, ``max``, ``gamma``.
        """
        return cls.THERMAL_VIS.copy()
