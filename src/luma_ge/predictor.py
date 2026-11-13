"""
Revised Predictors Module for Luma Geospatial Engine

This module provides backend services for calculating terrain metrics, spectral indices, and distance metrics
to enhance land cover classification. This version is revised to follow the latest update in Luma-lite

"""
import ee
import logging
import datetime
from typing import List, Optional, Dict, Any, Union, Set
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from .ee_config import ensure_ee_initialized
from .sar import get_sar_data

# Configure logging for development
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)
#=================================== Terrain Metrics ===================================
class terrain_calculator:
    """
    Calculate terrain metrics for creating input covariates in land cover mapping.
    This class provides methods to calculate elevation, slope, and aspect from various
    Digital Elevation Model (DEM) sources. It supports multiple DEM datasets and
    includes fallback mechanisms for reliability.
    
    Supported DEM Sources:
    - NASADEM: NASA NASADEM HGT (default) 
    - COPERNICUS: Copernicus DEM GLO30
    - ALOS: ALOS World 3D DSM 
    
    Usage Example:
        calc = terrain_calculator()
        #Calculate individual layers
        elevation = calc.calculate_elevation(aoi_geometry)
        slope = calc.calculate_slope(aoi_geometry)
        aspect = calc.calculate_aspect(aoi_geometry)
        #All layers use the same DEM source for consistency
        elevation = calc.calculate_elevation(aoi_geometry, dem_source="COPERNICUS")
        slope = calc.calculate_slope(aoi_geometry, dem_source="COPERNICUS")

    """
    
    def __init__(self):
        """
        Initialize the terrain calculator.

        """
        ensure_ee_initialized()
        logger.info("Terrain calculator initialized")
    #Fetch elevation data from earth engine catalog
    def calculate_elevation(self, aoi: ee.Geometry, dem_source: str = "NASADEM") -> ee.Image:
        """
        Calculate elevation layer from various DEM sources.
        This method loads a Digital Elevation Model from the specified source,
        clips it to the area of interest, and returns a standardized elevation layer.
        
        Parameters:
        -----------
        aoi : ee.Geometry
            Area of interest geometry for clipping the DEM
        dem_source : str, default="NASADEM"
            DEM source to use. Supported options:
            - "NASADEM": NASA NASADEM HGT/001 
            - "COPERNICUS": Copernicus DEM GLO30 
            - "ALOS": JAXA ALOS AW3D30 V4.1
            
        Returns:
        --------
        ee.Image or None
            Elevation layer with band name 'elevation' in meters above sea level.
            Returns None if calculation fails completely.
        
        Example:
        --------
        >>> calc = terrain_calculator()
        >>> aoi = ee.Geometry.Rectangle([106.0, -6.5, 107.0, -6.0])
        >>> elevation = calc.calculate_elevation(aoi, dem_source="COPERNICUS")
        >>> print(elevation.bandNames().getInfo())  # ['elevation']
        """
        try:
            logger.info(f"Calculating elevation layer using {dem_source} DEM...")
            
            # Define DEM datasets and their band names
            # Each dataset has different band naming conventions
            dem_datasets = {
                "NASADEM": {
                    "asset": "NASA/NASADEM_HGT/001",
                    "band": "elevation",
                    "is_collection": False
                },
                "COPERNICUS": {
                    "asset": "COPERNICUS/DEM/GLO30",
                    "band": "DEM", #The band is called DEM, but documentation called it DSM
                    "is_collection": True  #GLO30 is an ImageCollection
                },
                "ALOS": {
                    "asset": "JAXA/ALOS/AW3D30/V4_1",
                    "band": "DSM",  # Digital Surface Model
                    "is_collection": False
                }
            }
            
            # Validate DEM source and provide fallback
            if dem_source not in dem_datasets:
                logger.warning(f"Unknown DEM source '{dem_source}', using NASADEM as fallback")
                dem_source = "NASADEM"
            
            # Get DEM configuration
            dem_config = dem_datasets[dem_source]
            #Load and process DEM, with special treatment for Copernicus DEM
            if dem_config["is_collection"]:
                dem = ee.ImageCollection(dem_config["asset"]).select(dem_config["band"]).mosaic().clip(aoi)
            else:
                dem = ee.Image(dem_config["asset"]).select(dem_config["band"]).clip(aoi)
            # Rename band to 'elevation' for consistency across all DEM sources
            dem = dem.rename('elevation')
            logger.info(f"Successfully calculated elevation layer using {dem_source} DEM")
            return dem
            
        except Exception as e:
            logger.error(f"Failed to calculate elevation using {dem_source}: {str(e)}")
            # Try fallback to NASADEM if different source failed
            if dem_source != "NASADEM":
                logger.info("Attempting fallback to NASADEM...")
                try:
                    dem = ee.Image("NASA/NASADEM_HGT/001").select('elevation').clip(aoi)
                    logger.info("Successfully calculated elevation using NASADEM fallback")
                    return dem
                except Exception as fallback_e:
                    logger.error(f"Fallback to NASADEM also failed: {str(fallback_e)}")
            return None

    #calculate slope from elevation data 
    def calculate_slope(self, aoi: ee.Geometry, dem_source: str = "NASADEM") -> ee.Image:
        """
        Calculate slope layer from various DEM sources.
        
        Slope represents the steepness or gradient of the terrain, calculated as the
        maximum rate of change in elevation between a pixel and its neighbors.
        This is useful for distinguishing flat areas (agriculture, urban) from
        steep areas (forests, mountains).
        
        Parameters:
        -----------
        aoi : ee.Geometry
            Area of interest geometry for clipping the result
        dem_source : str, default="NASADEM"
            DEM source to use. Must match the source used for elevation
            to ensure consistency. 
            
        Returns:
        --------
        ee.Image or None
            Slope layer with band name 'slope' in degrees (0-90°).
            Returns None if calculation fails.
        Example:
        --------
        >>> calc = terrain_calculator()
        >>> aoi = ee.Geometry.Rectangle([106.0, -6.5, 107.0, -6.0])
        >>> slope = calc.calculate_slope(aoi, dem_source="NASADEM")
        >>> # Typical slope ranges: 0-3° (flat), 3-15° (moderate), >15° (steep)
        """
        try:
            logger.info(f"Calculating slope layer using {dem_source} DEM...")
            
            # Get elevation using the same DEM source for consistency
            elevation = self.calculate_elevation(aoi, dem_source)
            if elevation is None:
                logger.error("Failed to get elevation for slope calculation")
                return None
            
            # Calculate slope from elevation using Earth Engine's terrain algorithm
            # ee.Terrain.slope() uses the Horn (1981) algorithm for slope calculation
            slope = ee.Terrain.slope(elevation).rename('slope')
            logger.info(f"Successfully calculated slope layer using {dem_source} DEM")
            return slope
            
        except Exception as e:
            logger.error(f"Failed to calculate slope using {dem_source}: {str(e)}")
            return None
    #Calculate aspect from elevation data
    def calculate_aspect(self, aoi: ee.Geometry, dem_source: str = "NASADEM") -> ee.Image:
        """
        Calculate aspect layer from various DEM sources.
        
        Aspect represents the compass direction that a slope faces, which affects
        solar radiation exposure, moisture retention, and vegetation patterns.
        This is particularly useful in mountainous areas where slope orientation
        significantly influences land cover types.
        
        Parameters:
        -----------
        aoi : ee.Geometry
            Area of interest geometry for clipping the result
        dem_source : str, default="NASADEM"
            DEM source to use. Must match the source used for elevation
            to ensure consistency.
            
        Returns:
        --------
        ee.Image or None
            Aspect layer with band name 'aspect' in degrees (0-360°).
            Returns None if calculation fails.

        Example:
        --------
        >>> calc = terrain_calculator()
        >>> aoi = ee.Geometry.Rectangle([106.0, -6.5, 107.0, -6.0])
        >>> aspect = calc.calculate_aspect(aoi, dem_source="NASADEM")
        >>> # North: 315-45°, East: 45-135°, South: 135-225°, West: 225-315°
        """
        try:
            logger.info(f"Calculating aspect layer using {dem_source} DEM...")
            
            # Get elevation using the same DEM source for consistency
            elevation = self.calculate_elevation(aoi, dem_source)
            if elevation is None:
                logger.error("Failed to get elevation for aspect calculation")
                return None
            
            #Calculate aspect from elevation using Earth Engine's terrain algorithm
            #ee.Terrain.aspect() computes the compass direction of the steepest slope
            aspect = ee.Terrain.aspect(elevation).rename('aspect')
            logger.info(f"Successfully calculated aspect layer using {dem_source} DEM")
            return aspect
            
        except Exception as e:
            logger.error(f"Failed to calculate aspect using {dem_source}: {str(e)}")
            return None
    def get_landform(self, aoi: ee.Geometry, landform_source: str = "ALOS") -> ee.Image:
        """
        Retrieve global landform classification data.

        Landform data describes the shape and structure of the terrain surface
        (e.g. peak, ridge, slope, valley, plain) derived from DEM analysis.

        Parameters
        ----------
        aoi : ee.Geometry
            Area of interest geometry for clipping the result.
        landform_source : str, default="ALOS"
            Landform dataset to use. Supported options:

            - ``"ALOS"``  : Global ALOS Landform
              (``CSP/ERGo/1_0/Global/ALOS_landforms``, ~90 m)
            - ``"SRTM"``  : Global SRTM Landform
              (``CSP/ERGo/1_0/Global/SRTM_landforms``, ~90 m)

        Returns
        -------
        ee.Image or None
            Single-band image with band name ``'landform'`` containing
            landform class codes (11–62). Returns ``None`` if retrieval fails.

        Example
        -------
        >>> calc = terrain_calculator()
        >>> aoi = ee.Geometry.Rectangle([106.0, -6.5, 107.0, -6.0])
        >>> lf = calc.get_landform(aoi, landform_source="ALOS")
        >>> # Class 11 = Peak/Ridge top, 41 = Slope, 51 = Valley, etc.

        References
        ----------
        Theobald et al. (2015) *Ecologically-Relevant Maps of Landforms and
        Physiographic Diversity for Climate Adaptation Planning.*
        https://doi.org/10.1371/journal.pone.0143619
        """
        # Map human-readable keys to GEE asset IDs
        LANDFORM_SOURCES = {
            "ALOS": "CSP/ERGo/1_0/Global/ALOS_landforms",
            "SRTM": "CSP/ERGo/1_0/Global/SRTM_landforms",
        }

        source_upper = landform_source.upper()
        if source_upper not in LANDFORM_SOURCES:
            logger.error(
                f"Unsupported landform_source '{landform_source}'. "
                f"Choose from: {list(LANDFORM_SOURCES.keys())}"
            )
            return None

        asset_id = LANDFORM_SOURCES[source_upper]

        try:
            logger.info(f"Retrieving landform layer from {asset_id}...")

            landform = (
                ee.Image(asset_id)
                .select("constant")          # only band in both datasets
                .clip(aoi)
                .rename("landform")
            )

            logger.info(
                f"Successfully retrieved landform layer using {landform_source} source"
            )
            return landform

        except ee.EEException as e:
            logger.error(f"Earth Engine error retrieving landform ({landform_source}): {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Failed to retrieve landform ({landform_source}): {str(e)}")
            return None

#=================================== Sensor Identification & Predictor Catalog ===================================
class sensor_detector:
    """
    Utility for detecting satellite sensor family and resolving sensor-specific band mappings.

    Supported sensor families:
    - 'LANDSAT': Landsat 4, 5, 7, 8, 9 (and legacy MSS 1-3)
    - 'SENTINEL': Sentinel-2 MSI (S2A, S2B, Harmonized)
    - 'UNKNOWN': Unrecognized sensor family
    """

    # Semantic band name to spyndex parameter abbreviations
    BAND_TO_SPYNDEX = {
        "BLUE": "B",
        "GREEN": "G",
        "RED": "R",
        "NIR": "N",
        "SWIR1": "S1",
        "SWIR2": "S2",
        "RED_EDGE1": "RE1",
        "RED_EDGE2": "RE2",
        "RED_EDGE3": "RE3",
        "RED_EDGE4": "RE4",
        "THERMAL": "T",
        "ST_B10": "T",
        "B6": "T",
        "B6_VCID_2": "T",
        "B2": "B",
        "B3": "G",
        "B4": "R",
        "B8": "N",
        "B8A": "RE4",
        "B5": "RE1",
        "B6": "RE2",
        "B7": "RE3",
        "B11": "S1",
        "B12": "S2",
    }

    # Reverse lookup: spyndex code to primary semantic band name
    SPYNDEX_TO_BAND = {
        "B": "BLUE",
        "G": "GREEN",
        "R": "RED",
        "N": "NIR",
        "S1": "SWIR1",
        "S2": "SWIR2",
        "RE1": "RED_EDGE1",
        "RE2": "RED_EDGE2",
        "RE3": "RED_EDGE3",
        "RE4": "RED_EDGE4",
        "T": "THERMAL",
    }

    SENTINEL_INDICATOR_BANDS = {"RED_EDGE1", "RED_EDGE2", "RED_EDGE3", "RED_EDGE4", "B5", "B6", "B7", "B8A"}
    LANDSAT_INDICATOR_BANDS = {"THERMAL", "ST_B10", "SR_B1", "AEROSOL", "B6_VCID_2", "QA_PIXEL", "NIR1", "NIR2"}

    @classmethod
    def normalize_sensor_name(cls, sensor_hint: Optional[str]) -> str:
        """
        Normalize a sensor string hint into a canonical sensor family.

        Parameters
        ----------
        sensor_hint : str, optional
            Sensor identifier string (e.g. 'Landsat', 'Sentinel', 'L8_SR', 'S2_SR')

        Returns
        -------
        str : 'LANDSAT', 'SENTINEL', or 'UNKNOWN'
        """
        if not sensor_hint or not isinstance(sensor_hint, str):
            return "UNKNOWN"

        s = sensor_hint.strip().upper()
        if any(token in s for token in ["SENTINEL", "S2", "COPERNICUS/S2", "MSI"]):
            return "SENTINEL"
        if any(token in s for token in ["LANDSAT", "LC08", "LC09", "LE07", "LT05", "LT04",
                                        "LM01", "LM02", "LM03", "L8", "L9", "L7", "L5", "L4", "L1", "L2", "L3"]):
            return "LANDSAT"
        return "UNKNOWN"

    @classmethod
    def detect_sensor_from_bands(cls, band_names: List[str]) -> str:
        """
        Detect sensor family from a list of band names.

        Parameters
        ----------
        band_names : List[str]
            List of band names from the image or collection

        Returns
        -------
        str : 'SENTINEL', 'LANDSAT', or 'UNKNOWN'
        """
        if not band_names:
            return "UNKNOWN"

        band_set = set(band_names)
        if band_set.intersection(cls.SENTINEL_INDICATOR_BANDS):
            return "SENTINEL"
        if band_set.intersection(cls.LANDSAT_INDICATOR_BANDS):
            return "LANDSAT"

        # Check standard 6 optical bands: if NIR and SWIR1 present without Red-Edge, standard Landsat convention
        if {"NIR", "RED", "GREEN", "BLUE"}.issubset(band_set):
            return "LANDSAT"

        return "UNKNOWN"

    @classmethod
    def detect_sensor(
        cls,
        sensor_hint: Optional[str] = None,
        band_names: Optional[List[str]] = None,
        composite: Optional[ee.Image] = None,
        collection: Optional[ee.ImageCollection] = None
    ) -> str:
        """
        Comprehensive sensor detection strategy combining hints, band names, and Earth Engine objects.
        """
        # 1. Try explicit hint first
        norm = cls.normalize_sensor_name(sensor_hint)
        if norm != "UNKNOWN":
            return norm

        # 2. Try explicit band names list
        if band_names:
            detected = cls.detect_sensor_from_bands(band_names)
            if detected != "UNKNOWN":
                return detected

        # 3. Try composite band names / metadata
        if composite is not None:
            try:
                bands = composite.bandNames().getInfo()
                detected = cls.detect_sensor_from_bands(bands)
                if detected != "UNKNOWN":
                    return detected
            except Exception as e:
                logger.debug(f"Could not read bandNames from composite: {e}")

        # 4. Try collection metadata or first image
        if collection is not None:
            try:
                first_img = collection.first()
                if first_img is not None:
                    bands = first_img.bandNames().getInfo()
                    detected = cls.detect_sensor_from_bands(bands)
                    if detected != "UNKNOWN":
                        return detected
            except Exception as e:
                logger.debug(f"Could not read bandNames from collection: {e}")

        # Default fallback
        return "LANDSAT"

#centralize predictor catalog to ease up addition or substraction of predictor
class predictor_catalog:
    """
    Central registry and metadata catalog for all supported predictors in Luma-GE,
    including spectral indices, topography, radar backscatter, and distance based metrics.
    """
    PREDICTORS = {
        # --- Vegetation Indices ---
        "NDVI": {
            "name": "NDVI",
            "category": "Vegetasi",
            "long_name": "Normalized Difference Vegetation Index",
            "sensor": "ALL",
            "required_bands": ["N", "R"],
            "type": "spectral",
            "description": "General canopy greenness and biomass",
        },
        "EVI": {
            "name": "EVI",
            "category": "Vegetasi",
            "long_name": "Enhanced Vegetation Index",
            "sensor": "ALL",
            "required_bands": ["N", "R", "B"],
            "type": "spectral",
            "description": "High-biomass canopy greenness with aerosol correction",
        },
        "SAVI": {
            "name": "SAVI",
            "category": "Vegetasi",
            "long_name": "Soil Adjusted Vegetation Index",
            "sensor": "ALL",
            "required_bands": ["N", "R"],
            "type": "spectral",
            "description": "Soil-brightness compensated vegetation index",
        },
        "MSAVI": {
            "name": "MSAVI",
            "category": "Vegetasi",
            "long_name": "Modified Soil Adjusted Vegetation Index",
            "sensor": "ALL",
            "required_bands": ["N", "R"],
            "type": "spectral",
            "description": "Vegetation index for sparse canopy / arid areas",
        },
        "GNDVI": {
            "name": "GNDVI",
            "category": "Vegetasi",
            "long_name": "Green Normalized Difference Vegetation Index",
            "sensor": "ALL",
            "required_bands": ["N", "G"],
            "type": "spectral",
            "description": "Chlorophyll concentration and plant senescence",
        },
        "MCARI": {
            "name": "MCARI",
            "category": "Vegetasi",
            "long_name": "Modified Chlorophyll Absorption in Reflectance Index",
            "sensor": "SENTINEL",
            "required_bands": ["RE1", "R", "G"],
            "type": "spectral",
            "description": "Canopy chlorophyll absorption (requires Sentinel-2 Red-Edge 1)",
            "sensor_note": "Requires Sentinel-2 Red-Edge band (RE1)",
        },
        "RENDVI": {
            "name": "RENDVI",
            "category": "Vegetasi",
            "long_name": "Red Edge Normalized Difference Vegetation Index",
            "sensor": "SENTINEL",
            "required_bands": ["RE2", "RE1"],
            "type": "spectral",
            "description": "Red-Edge vegetation index (requires Sentinel-2 Red-Edge 1 & 2)",
            "sensor_note": "Requires Sentinel-2 Red-Edge bands (RE1, RE2)",
        },
        "MCARI1": {
            "name": "MCARI1",
            "category": "Vegetasi",
            "long_name": "Modified Chlorophyll Absorption in Reflectance Index 1",
            "sensor": "ALL",
            "required_bands": ["N", "R", "G"],
            "type": "spectral",
            "description": "Chlorophyll variation without red-edge requirement",
        },

        # --- Water Indices ---
        "MNDWI": {
            "name": "MNDWI",
            "category": "Air & Kelembaban",
            "long_name": "Modified Normalized Difference Water Index",
            "sensor": "ALL",
            "required_bands": ["G", "S1"],
            "type": "spectral",
            "description": "Open water delineation with shadow suppression",
        },
        "NDMI": {
            "name": "NDMI",
            "category": "Air & Kelembaban",
            "long_name": "Normalized Difference Moisture Index",
            "sensor": "ALL",
            "required_bands": ["N", "S1"],
            "type": "spectral",
            "description": "Canopy water content and wetland soil moisture",
        },
        "AWEIsh": {
            "name": "AWEIsh",
            "category": "Air & Kelembaban",
            "long_name": "Automated Water Extraction Index (with shadow)",
            "sensor": "ALL",
            "required_bands": ["B", "G", "N", "S1", "S2"],
            "type": "spectral",
            "description": "Water extraction in shadow-heavy mountainous/urban regions",
        },
        "AWEInsh": {
            "name": "AWEInsh",
            "category": "Air & Kelembaban",
            "long_name": "Automated Water Extraction Index (no shadow)",
            "sensor": "ALL",
            "required_bands": ["G", "S1"],
            "type": "spectral",
            "description": "Water extraction without shadow compensation",
        },
        "NDTI": {
            "name": "NDTI",
            "category": "Air & Kelembaban",
            "long_name": "Normalized Difference Turbidity Index",
            "sensor": "ALL",
            "required_bands": ["R", "G"],
            "type": "spectral",
            "description": "Water turbidity and suspended sediment detection",
        },
        "ANDWI": {
            "name": "ANDWI",
            "category": "Air & Kelembaban",
            "long_name": "Augmented Normalized Difference Water Index",
            "sensor": "ALL",
            "required_bands": ["B", "G", "R", "N", "S1", "S2"],
            "type": "spectral",
            "description": "Multi-band augmented water extraction",
        },

        # --- Build-up and Bare Soil ---
        "NDBI": {
            "name": "NDBI",
            "category": "Tanah dan Lahan Terbangun",
            "long_name": "Normalized Difference Built-Up Index",
            "sensor": "ALL",
            "required_bands": ["S1", "N"],
            "type": "spectral",
            "description": "Impervious surface and built-up land",
        },
        "DBSI": {
            "name": "DBSI",
            "category": "Tanah dan Lahan Terbangun",
            "long_name": "Dry Bare-Soil Index",
            "sensor": "ALL",
            "required_bands": ["S1", "G", "N", "R"],
            "type": "spectral",
            "description": "Dry bare soil separation from urban areas",
        },
        "BLFEI": {
            "name": "BLFEI",
            "category": "Tanah dan Lahan Terbangun",
            "long_name": "Built-Up Land Features Extraction Index",
            "sensor": "ALL",
            "required_bands": ["G", "R", "S2", "S1"],
            "type": "spectral",
            "description": "Built-up footprint extraction",
        },
        "MBI": {
            "name": "MBI",
            "category": "Tanah dan Lahan Terbangun",
            "long_name": "Modified Bare Soil Index",
            "sensor": "ALL",
            "required_bands": ["S1", "S2", "N"],
            "type": "spectral",
            "description": "Bare soil and fallow agricultural land discrimination",
        },
        "EBBI": {
            "name": "EBBI",
            "category": "Tanah dan Lahan Terbangun",
            "long_name": "Enhanced Built-Up and Bareness Index",
            "sensor": "LANDSAT",
            "required_bands": ["S1", "N", "T"],
            "type": "spectral",
            "description": "Built-up and bareness extraction (requires Landsat Thermal band)",
            "sensor_note": "Requires Landsat Thermal band (not available on Sentinel-2)",
        },

        # --- Topography ---
        "Elevation": {
            "name": "Elevation",
            "category": "Topografi",
            "long_name": "Digital Elevation Model",
            "sensor": "ALL",
            "type": "terrain",
            "description": "Elevation above sea level in meters (NASADEM / Copernicus / ALOS)",
        },
        "Slope": {
            "name": "Slope",
            "category": "Topografi",
            "long_name": "Terrain Slope",
            "sensor": "ALL",
            "type": "terrain",
            "description": "Terrain gradient in degrees (0-90°)",
        },
        "Aspect": {
            "name": "Aspect",
            "category": "Topografi",
            "long_name": "Terrain Aspect",
            "sensor": "ALL",
            "type": "terrain",
            "description": "Compass direction of steepest slope (0-360°)",
        },
        "Landform": {
            "name": "Landform",
            "category": "Topografi",
            "long_name": "Landform Classification",
            "sensor": "ALL",
            "type": "terrain",
            "description": "Discrete landform morphological classes (ALOS / SRTM)",
        },

        # --- Radar Polarization ---
        "VV": {
            "name": "VV",
            "category": "Polarisasi Radar",
            "long_name": "Sentinel-1 VV Backscatter",
            "sensor": "SENTINEL",
            "type": "radar",
            "min_date": "2014-10-03",
            "description": "Vertical-transmit Vertical-receive C-band backscatter (10m, dB)",
            "sensor_note": "Available when Sentinel workflow is selected (active >= Oct 2014)",
        },
        "VH": {
            "name": "VH",
            "category": "Polarisasi Radar",
            "long_name": "Sentinel-1 VH Backscatter",
            "sensor": "SENTINEL",
            "type": "radar",
            "min_date": "2014-10-03",
            "description": "Vertical-transmit Horizontal-receive C-band backscatter (10m, dB)",
            "sensor_note": "Available when Sentinel workflow is selected (active >= Oct 2014)",
        },
        "VV-VH Ratio": {
            "name": "VV-VH Ratio",
            "category": "Polarisasi Radar",
            "long_name": "Sentinel-1 VV/VH Polarization Ratio",
            "sensor": "SENTINEL",
            "type": "radar",
            "min_date": "2014-10-03",
            "description": "Sentinel-1 cross-polarization ratio (VV - VH in dB)",
            "sensor_note": "Available when Sentinel workflow is selected (active >= Oct 2014)",
        },
        "HH": {
            "name": "HH",
            "category": "Polarisasi Radar",
            "long_name": "ALOS PALSAR HH Backscatter",
            "sensor": "LANDSAT",
            "type": "radar",
            "valid_years": [2007, 2008, 2009, 2010, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
            "description": "Horizontal-transmit Horizontal-receive L-band backscatter (25m, dB)",
            "sensor_note": "Available when Landsat workflow is selected (2007-2010, 2015-2024)",
        },
        "HV": {
            "name": "HV",
            "category": "Polarisasi Radar",
            "long_name": "ALOS PALSAR HV Backscatter",
            "sensor": "LANDSAT",
            "type": "radar",
            "valid_years": [2007, 2008, 2009, 2010, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
            "description": "Horizontal-transmit Vertical-receive L-band backscatter (25m, dB)",
            "sensor_note": "Available when Landsat workflow is selected (2007-2010, 2015-2024)",
        },
        "HV-HH Ratio": {
            "name": "HV-HH Ratio",
            "category": "Polarisasi Radar",
            "long_name": "ALOS PALSAR HH/HV Polarization Ratio",
            "sensor": "LANDSAT",
            "type": "radar",
            "valid_years": [2007, 2008, 2009, 2010, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
            "description": "ALOS PALSAR cross-polarization ratio (HH - HV in dB)",
            "sensor_note": "Available when Landsat workflow is selected (2007-2010, 2015-2024)",
        },
        # --- Distance & Accessibility Predictors ---
        "dist_roads": {
            "name": "dist_roads",
            "category": "Jarak & Aksesibilitas",
            "long_name": "Distance to Main Roads (GRIP4)",
            "sensor": "ALL",
            "type": "distance",
            "description": "Euclidean distance to nearest main road (GRIP4 highways, primary & secondary roads)",
        },
        "dist_coast": {
            "name": "dist_coast",
            "category": "Jarak & Aksesibilitas",
            "long_name": "Distance to Coastline (S2COAST-2023)",
            "sensor": "ALL",
            "type": "distance",
            "description": "Euclidean distance to nearest S2COAST-2023 shoreline",
        },
        "dist_settlement": {
            "name": "dist_settlement",
            "category": "Jarak & Aksesibilitas",
            "long_name": "Distance to Settlement (HRSL)",
            "sensor": "ALL",
            "type": "distance",
            "description": "Euclidean distance to nearest High Resolution Settlement Layer (HRSL) built-up pixel",
        },
    }

    @classmethod
    def get_all_catalog(cls) -> dict:
        """Return a copy of the entire predictor catalog dictionary."""
        return cls.PREDICTORS.copy()

    @classmethod
    def check_availability(
        cls,
        sensor: str = "LANDSAT",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate availability for all registered predictors based on sensor and temporal range.
        """
        sensor_norm = sensor_detector.normalize_sensor_name(sensor)
        if sensor_norm == "UNKNOWN":
            sensor_norm = "LANDSAT"

        # Parse date range
        start_dt = None
        end_dt = None
        year_set = set()
        if start_date:
            try:
                start_dt = datetime.datetime.strptime(str(start_date)[:10], "%Y-%m-%d").date()
            except Exception:
                pass
        if end_date:
            try:
                end_dt = datetime.datetime.strptime(str(end_date)[:10], "%Y-%m-%d").date()
            except Exception:
                pass

        if start_dt and end_dt:
            year_set = set(range(start_dt.year, end_dt.year + 1))
        elif start_dt:
            year_set = {start_dt.year}
        elif end_dt:
            year_set = {end_dt.year}

        results = {}
        for name, meta in cls.PREDICTORS.items():
            item_sensor = meta.get("sensor", "ALL")
            is_avail = True
            reason = ""
            disabled = False

            # Sensor compatibility check
            if item_sensor != "ALL" and item_sensor != sensor_norm:
                is_avail = False
                disabled = True
                reason = meta.get("sensor_note", f"Requires {item_sensor} sensor")

            # Temporal range check
            if is_avail and "min_date" in meta and end_dt:
                min_dt = datetime.datetime.strptime(meta["min_date"], "%Y-%m-%d").date()
                if end_dt < min_dt:
                    is_avail = False
                    disabled = True
                    reason = f"Only available from {meta['min_date']} onwards"

            if is_avail and "valid_years" in meta and year_set:
                valid_set = set(meta["valid_years"])
                if not year_set.intersection(valid_set):
                    is_avail = False
                    disabled = True
                    reason = f"Unavailable for year(s) {sorted(list(year_set))} (active years: {meta['valid_years'][0]}-{meta['valid_years'][-1]})"

            results[name] = {
                "name": name,
                "category": meta["category"],
                "long_name": meta.get("long_name", name),
                "type": meta.get("type", "spectral"),
                "description": meta.get("description", ""),
                "available": is_avail,
                "disabled": disabled,
                "reason": reason,
                "sensor_compatibility": item_sensor
            }

        return results

#=================================== Spectral Calculator Backend ===================================
#reference for the use of spectral index in LC mapping: https://doi.org/10.3390/rs12183062
class SpectralCalculator:
    """
    Backend service for spectral index calculations using spyndex library.
    Migrated from Streamlit UI to follow clean architecture patterns.
    Enhanced to support dynamic sensor band mapping (Red-Edge, Thermal) and sensor-aware availability.
    """
    
    def __init__(self):
        """
        Initialize the spectral calculator backend.
        Ensure Earth Engine is initialized lazily.
        """
        ensure_ee_initialized()
        
        # Import spyndex library for spectral index calculations
        try:
            import spyndex
            self.spyndex = spyndex
            self.spyndex_indices = spyndex.indices
            logger.info("SpectralCalculator initialized")
        except ImportError as e:
            logger.error(f"Failed to import spyndex library: {str(e)}")
            raise ImportError("spyndex library is required for spectral calculations")
        
        # Define supported indices from PredictorCatalog
        self.supported_indices = {
            name: meta["category"]
            for name, meta in predictor_catalog.PREDICTORS.items()
            if meta.get("type", "spectral") == "spectral"
        }

        # Define default coefficients for indices that require them
        self.default_coefficients = {
            "EVI": {
                "g": 2.5,      # Gain factor
                "C1": 6.0,     # Coefficient for aerosol resistance term 
                "C2": 7.5,     # Coefficient for aerosol resistance term 
                "L": 1.0       # Canopy background adjustment factor
            },
            "SAVI": {
                "L": 0.5       # Soil brightness correction factor (0-1)
            },
            "MSAVI": {
                "L": 0.5       # Soil adjustment factor
            },
            "EVI2": {
                "g": 2.5,
                "L": 1.0
            },
            "ARVI": {
                "gamma": 1.0
            }
        }
        
        self.calculated_count = 0
    
    def get_available_indices(
        self,
        sensor: Optional[str] = None,
        band_names: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get available spectral indices with metadata, filtered by sensor and/or available bands.
        
        Parameters:
        -----------
        sensor : str, optional
            Satellite sensor family ('LANDSAT' or 'SENTINEL')
        band_names : List[str], optional
            Available band names to filter compatible indices
            
        Returns:
        --------
        Dict[str, Dict[str, Any]] : Dictionary of available indices with metadata
        """
        try:
            available_indices = {}
            sensor_norm = sensor_detector.normalize_sensor_name(sensor) if sensor else None
            
            for idx_name, category in self.supported_indices.items():
                cat_meta = predictor_catalog.PREDICTORS.get(idx_name, {})
                idx_sensor = cat_meta.get("sensor", "ALL")

                # Filter by sensor compatibility if sensor is provided
                if sensor_norm and sensor_norm != "UNKNOWN" and idx_sensor != "ALL" and idx_sensor != sensor_norm:
                    continue

                # Filter by band names if band_names list is provided
                if band_names and not self.validate_index_requirements(idx_name, band_names):
                    continue

                if idx_name in self.spyndex_indices:
                    idx_info = self.spyndex_indices[idx_name]
                    available_indices[idx_name] = {
                        "long_name": getattr(idx_info, 'long_name', cat_meta.get("long_name", idx_name)),
                        "formula": getattr(idx_info, 'formula', ''),
                        "reference": getattr(idx_info, 'reference', ''),
                        "category": category,
                        "sensor_compatibility": idx_sensor,
                        "required_bands": cat_meta.get("required_bands", getattr(idx_info, 'bands', [])),
                    }
                else:
                    available_indices[idx_name] = {
                        "long_name": cat_meta.get("long_name", idx_name),
                        "formula": "Formula not available",
                        "reference": "",
                        "category": category,
                        "sensor_compatibility": idx_sensor,
                        "required_bands": cat_meta.get("required_bands", []),
                    }
            
            logger.info(f"Found {len(available_indices)} available spectral indices")
            return available_indices
            
        except Exception as e:
            logger.error(f"Failed to get available indices: {str(e)}")
            return {}
    
    def get_index_coefficients(self, index_name: str) -> Dict[str, float]:
        """
        Get default coefficient values for a spectral index.
        """
        try:
            if index_name in self.default_coefficients:
                coeffs = self.default_coefficients[index_name].copy()
                logger.info(f"Retrieved default coefficients for {index_name}: {coeffs}")
                return coeffs
            else:
                logger.debug(f"Index {index_name} does not require coefficients")
                return {}
        except Exception as e:
            logger.error(f"Failed to get coefficients for {index_name}: {str(e)}")
            return {}
    
    def validate_coefficients(
        self, 
        index_name: str, 
        coefficients: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Validate coefficient values for a spectral index.
        """
        validation_result = {
            "valid": True,
            "warnings": []
        }
        try:
            coefficient_ranges = {
                "g": (0.1, 10.0),
                "C1": (0.0, 20.0),
                "C2": (0.0, 20.0),
                "L": (0.0, 2.0),
                "gamma": (0.1, 5.0)
            }
            
            for coeff_name, coeff_value in coefficients.items():
                if not isinstance(coeff_value, (int, float)):
                    validation_result["warnings"].append(
                        f"Coefficient '{coeff_name}' has non-numeric value: {coeff_value}"
                    )
                    validation_result["valid"] = False
                    continue
                
                if coeff_name in coefficient_ranges:
                    min_val, max_val = coefficient_ranges[coeff_name]
                    if coeff_value < min_val or coeff_value > max_val:
                        validation_result["warnings"].append(
                            f"Coefficient '{coeff_name}' = {coeff_value} is outside typical range "
                            f"[{min_val}, {max_val}] for {index_name}"
                        )
            
            if validation_result["warnings"]:
                logger.warning(f"Coefficient validation warnings for {index_name}: {validation_result['warnings']}")
            else:
                logger.info(f"Coefficients validated successfully for {index_name}")
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Error validating coefficients for {index_name}: {str(e)}")
            return {
                "valid": False,
                "warnings": [f"Validation error: {str(e)}"]
            }

    def validate_index_requirements(self, index_name: str, available_bands: List[str]) -> bool:
        """
        Validate that required bands are available for index computation.
        Maps available semantic band names to spyndex abbreviations.
        """
        try:
            avail_spyndex_symbols = set()
            for b in available_bands:
                b_u = str(b).upper()
                if b_u in sensor_detector.BAND_TO_SPYNDEX:
                    avail_spyndex_symbols.add(sensor_detector.BAND_TO_SPYNDEX[b_u])
                avail_spyndex_symbols.add(b_u)

            if index_name in predictor_catalog.PREDICTORS:
                required = predictor_catalog.PREDICTORS[index_name].get("required_bands", [])
            elif index_name in self.spyndex_indices:
                required = getattr(self.spyndex_indices[index_name], 'bands', [])
            else:
                logger.warning(f"Index '{index_name}' not found in catalog or spyndex")
                return False

            missing = [band for band in required if band not in avail_spyndex_symbols]
            if missing:
                logger.debug(f"Index '{index_name}' requires bands {missing} not in available bands {available_bands}")
                return False

            return True
            
        except Exception as e:
            logger.error(f"Failed to validate requirements for {index_name}: {str(e)}")
            return False

    def _build_spyndex_params(
        self,
        image: ee.Image,
        available_bands: Optional[List[str]] = None,
        base_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Dynamically build the spyndex parameter dictionary by inspecting available bands in the image.
        Supports standard optical bands (B, G, R, N, S1, S2), Red-Edge bands (RE1, RE2, RE3, RE4),
        and Thermal band (T).
        """
        params = {}
        bands_set = set(available_bands) if available_bands else set()

        # Primary semantic band to spyndex parameter mappings
        primary_mappings = [
            ("NIR", "N"),
            ("RED", "R"),
            ("GREEN", "G"),
            ("BLUE", "B"),
            ("SWIR1", "S1"),
            ("SWIR2", "S2"),
            ("RED_EDGE1", "RE1"),
            ("RED_EDGE2", "RE2"),
            ("RED_EDGE3", "RE3"),
            ("RED_EDGE4", "RE4"),
            ("THERMAL", "T"),
            ("ST_B10", "T"),
            ("B6", "T"),
            ("B2", "B"),
            ("B3", "G"),
            ("B4", "R"),
            ("B8", "N"),
            ("B8A", "RE4"),
            ("B5", "RE1"),
            ("B6", "RE2"),
            ("B7", "RE3"),
            ("B11", "S1"),
            ("B12", "S2"),
        ]

        for band_name, spyndex_code in primary_mappings:
            if not bands_set or band_name in bands_set:
                if spyndex_code not in params:
                    try:
                        params[spyndex_code] = image.select(band_name)
                    except Exception:
                        pass

        # Fallback for standard names if band_set was empty
        fallbacks = [
            ("N", "NIR"),
            ("R", "RED"),
            ("G", "GREEN"),
            ("B", "BLUE"),
            ("S1", "SWIR1"),
            ("S2", "SWIR2"),
        ]
        for spyndex_code, band_name in fallbacks:
            if spyndex_code not in params:
                try:
                    params[spyndex_code] = image.select(band_name)
                except Exception:
                    pass

        if base_params:
            params.update(base_params)

        return params

    def calculate_indices(
        self,
        image: ee.Image,
        index_list: List[str],
        coefficients: Optional[Dict[str, Dict[str, float]]] = None,
        sensor: Optional[str] = None
    ) -> Optional[ee.Image]:
        """
        Calculate spectral indices for a single image using spyndex with dynamic band mapping.
        
        Parameters:
        -----------
        image : ee.Image
            Input image containing optical/thermal/red-edge bands
        index_list : List[str]
            List of spectral index names to calculate
        coefficients : Dict[str, Dict[str, float]], optional
            Custom coefficient values for indices that require them
        sensor : str, optional
            Sensor family hint ('LANDSAT' or 'SENTINEL')
            
        Returns:
        --------
        ee.Image or None : Multi-band image containing calculated indices
        """
        try:
            if not index_list:
                return None

            try:
                available_bands = image.bandNames().getInfo()
            except Exception:
                available_bands = []

            # Filter valid indices for this image
            valid_indices = []
            for idx in index_list:
                if not available_bands or self.validate_index_requirements(idx, available_bands):
                    valid_indices.append(idx)
                else:
                    logger.warning(f"Skipping index '{idx}': required bands not present in image ({available_bands})")

            if not valid_indices:
                logger.warning("No requested spectral indices are compatible with the image bands")
                return None

            base_params = {}
            if coefficients:
                for idx_name, index_coeffs in coefficients.items():
                    if idx_name in valid_indices:
                        validation = self.validate_coefficients(idx_name, index_coeffs)
                        if validation["valid"]:
                            base_params.update(index_coeffs)
            else:
                for idx_name in valid_indices:
                    default_coeffs = self.get_index_coefficients(idx_name)
                    if default_coeffs:
                        base_params.update(default_coeffs)

            params = self._build_spyndex_params(image, available_bands=available_bands, base_params=base_params)
            spectral_indices = self.spyndex.computeIndex(
                index=valid_indices,
                params=params
            )
            self.calculated_count += len(valid_indices)
            return spectral_indices

        except Exception as e:
            logger.error(f"Failed to calculate spectral indices on single image: {str(e)}")
            return None

    def calculate_indices_with_collection(
        self, 
        collection: ee.ImageCollection, 
        aoi: ee.Geometry,
        index_list: List[str],
        coefficients: Optional[Dict[str, Dict[str, float]]] = None,
        reducer_method: str = "mean",
        sensor: Optional[str] = None
    ) -> Optional[ee.Image]:
        """
        Calculate spectral indices over an image collection using map and reducer.
        Preserves per-scene spectral characteristics before composite reduction.
        """
        try:
            logger.info(f"Calculating spectral indices using map and {reducer_method} reducer on collection")
            
            valid_reducers = ["mean", "median", "max", "min"]
            if reducer_method not in valid_reducers:
                logger.warning(f"Unsupported reducer method: {reducer_method}, using mean")
                reducer_method = "mean"

            # Inspect available bands from the collection
            try:
                available_bands = collection.first().bandNames().getInfo()
            except Exception:
                available_bands = []

            # Filter indices that are compatible with available bands
            valid_indices = []
            for idx in index_list:
                if not available_bands or self.validate_index_requirements(idx, available_bands):
                    valid_indices.append(idx)
                else:
                    logger.warning(f"Skipping index '{idx}': required bands not present in collection ({available_bands})")

            if not valid_indices:
                logger.warning("No requested spectral indices are compatible with collection bands")
                return None

            # Prepare base parameters for spyndex
            base_params = {}
            if coefficients:
                for idx_name, index_coeffs in coefficients.items():
                    if idx_name in valid_indices:
                        validation = self.validate_coefficients(idx_name, index_coeffs)
                        if not validation["valid"]:
                            raise ValueError(f"Invalid coefficients for {idx_name}: {validation['warnings']}")
                        base_params.update(index_coeffs)
            else:
                for idx_name in valid_indices:
                    default_coeffs = self.get_index_coefficients(idx_name)
                    if default_coeffs:
                        base_params.update(default_coeffs)
            
            # Map calculation over collection
            def calculate_indices_for_image(image):
                params = self._build_spyndex_params(image, available_bands=available_bands, base_params=base_params)
                return self.spyndex.computeIndex(
                    index=valid_indices,
                    params=params
                )
            
            indices_collection = collection.map(calculate_indices_for_image)
            
            # Reducer
            logger.info(f"Applying {reducer_method} reducer to collection...")
            if reducer_method == "mean":
                reduced_indices = indices_collection.mean()
            elif reducer_method == "median":
                reduced_indices = indices_collection.median()
            elif reducer_method == "max":
                reduced_indices = indices_collection.max()
            elif reducer_method == "min":
                reduced_indices = indices_collection.min()
            
            reduced_indices = reduced_indices.clip(aoi)
            
            if reduced_indices is not None:
                band_count = reduced_indices.bandNames().size().getInfo()
                if band_count > 0:
                    self.calculated_count += len(valid_indices)
                    logger.info(f"Successfully calculated {band_count} spectral indices using {reducer_method} reducer")
                    return reduced_indices
                else:
                    logger.error("No valid bands in reduced indices")
                    return None
            else:
                logger.error("Failed to calculate spectral indices - no valid results")
                return None
            
        except Exception as e:
            logger.error(f"Failed to calculate indices with collection: {str(e)}")
            return None
    
    def get_calculation_summary(self) -> Dict[str, Any]:
        """
        Get summary of calculation process.
        """
        return {
            'indices_calculated': self.calculated_count,
            'supported_indices': len(self.supported_indices),
            'categories': list(set(self.supported_indices.values()))
        }

#=================================== Distance Metrics ===================================
class distance_calculator:
    """
    Computes fast Euclidean distance-based predictors measuring proximity to:
    - Main Roads (`dist_roads`): On-the-fly from Global Roads Inventory Project (GRIP4, main roads
      GP_RTP in [1, 2, 3]) or from a pre-computed raster asset.
    - Coastline (`dist_coast`): Derived from S2COAST-2023 (`projects/sat-io/open-datasets/S2COAST-2023`)
      with outward spatial buffering to preserve high/low-tide shoreline continuity.
    - Settlements (`dist_settlement`): Derived from High Resolution Settlement Layer (HRSL).

    Uses Earth Engine's linear-time `fastDistanceTransform()` instead of `cumulativeCost()`
    to avoid neighborhood size limits and timeouts.
    """

    S2COAST_ASSET = "projects/sat-io/open-datasets/S2COAST-2023"

    # GRIP4 regional collections covering Southeast Asia & PNG/Oceania
    GRIP4_REGIONAL_ASSETS = [
        "projects/sat-io/open-datasets/GRIP4/South-East-Asia",
        "projects/sat-io/open-datasets/GRIP4/Oceania",
    ]

    # GRIP4 road types (GP_RTP): 1=Highway, 2=Primary, 3=Secondary (excludes 4=Tertiary, 5=Local)
    MAIN_ROAD_TYPES = [1, 2, 3]

    HRSL_ASSET = "projects/sat-io/open-datasets/hrsl/hrslpop"

    def __init__(self):
        """
        Initialize the distance calculator lazily without heavy GEE computations.
        """
        ensure_ee_initialized()
        logger.info("Distance calculator initialized")

    def _compute_fast_distance(
        self,
        source_binary_mask: ee.Image,
        aoi: ee.Geometry,
        band_name: str,
        max_dist: float = 50000,
        in_meters: bool = True,
        neighborhood_px: int = 1024
    ) -> ee.Image:
        """
        Compute Euclidean distance from a binary source mask (1=feature, 0=background)
        using `fastDistanceTransform` and clip to `aoi`.
        """
        clean_mask = source_binary_mask.unmask(0).gt(0)
        dist_px = clean_mask.fastDistanceTransform(
            neighborhood=neighborhood_px,
            units="pixels",
            metric="squared_euclidean"
        ).sqrt()

        if in_meters:
            dist_img = dist_px.multiply(ee.Image.pixelArea().sqrt())
        else:
            dist_img = dist_px

        return (
            dist_img
            .clamp(0, max_dist)
            .unmask(max_dist)
            .clip(aoi)
            .toFloat()
            .rename(band_name)
        )

    def extract_s2_coastline(
        self,
        aoi: ee.Geometry,
        buffer_m: float = 10000
    ) -> ee.FeatureCollection:
        """
        Extract coastline polyline segments from S2COAST-2023 within the user-defined AOI.

        Applies an outward spatial buffer (`buffer_m`, default 10 km) around `aoi` so that
        shoreline segments (and high/low-tide lines) just outside the AOI boundary are
        captured when computing proximity.

        Parameters
        ----------
        aoi : ee.Geometry
            User-defined area of interest geometry.
        buffer_m : float, default=10000
            Outward search buffer in meters around `aoi` to capture nearby coastlines.
        """
        s2_coast = ee.FeatureCollection(self.S2COAST_ASSET)
        target_buffer = aoi.buffer(buffer_m)
        return s2_coast.filter(ee.Filter.bounds(target_buffer))

    def extract_grip4_roads(
        self,
        aoi: ee.Geometry,
        buffer_m: float = 50000,
        main_roads_only: bool = True,
        road_types: Optional[List[int]] = None,
        grip4_assets: Optional[List[str]] = None
    ) -> ee.FeatureCollection:
        """
        Extract road network features from the Global Roads Inventory Project (GRIP4).

        Parameters
        ----------
        aoi : ee.Geometry
            Area of interest.
        buffer_m : float, default=50000
            Outward search buffer in meters around `aoi` so nearby roads outside the
            AOI boundary are accounted for.
        main_roads_only : bool, default=True
            If True, filters `GP_RTP` to main roads (`[1, 2, 3]`: Highways, Primary, Secondary).
        road_types : list of int, optional
            Custom list of `GP_RTP` road type codes.
        grip4_assets : list of str, optional
            Custom GRIP4 FeatureCollection asset IDs.
        """
        search_geom = aoi.buffer(buffer_m)
        assets = grip4_assets or self.GRIP4_REGIONAL_ASSETS
        target_types = road_types or self.MAIN_ROAD_TYPES

        collections = []
        for asset_id in assets:
            fc = ee.FeatureCollection(asset_id).filter(ee.Filter.bounds(search_geom))
            if main_roads_only:
                fc = fc.filter(ee.Filter.inList("GP_RTP", target_types))
            collections.append(fc)

        merged_roads = collections[0]
        for fc in collections[1:]:
            merged_roads = merged_roads.merge(fc)

        return merged_roads

    def calculate_distance_to_coast(
        self,
        aoi: ee.Geometry,
        max_dist: float = 50000,
        in_meters: bool = True,
        neighborhood_px: int = 1024
    ) -> ee.Image:
        """
        Compute Euclidean distance to nearest S2COAST-2023 shoreline (`dist_coast`),
        clipped to the user-defined `aoi`.
        """
        search_buffer = max(10000.0, float(max_dist))
        filtered_coast = self.extract_s2_coastline(
            aoi=aoi,
            buffer_m=search_buffer
        )
        coast_mask = (
            filtered_coast
            .map(lambda ft: ft.set("constant", 1))
            .reduceToImage(["constant"], ee.Reducer.first())
            .unmask(0)
        )
        return self._compute_fast_distance(
            source_binary_mask=coast_mask,
            aoi=aoi,
            band_name="dist_coast",
            max_dist=max_dist,
            in_meters=in_meters,
            neighborhood_px=neighborhood_px
        )

    def calculate_distance_to_roads(
        self,
        aoi: ee.Geometry,
        max_dist: float = 50000,
        in_meters: bool = True,
        road_source: str = "grip4",
        main_roads_only: bool = True,
        road_types: Optional[List[int]] = None,
        precomputed_asset: Optional[str] = None,
        neighborhood_px: int = 1024
    ) -> ee.Image:
        """
        Compute distance to roads (`dist_roads`) using either:
        1. `road_source='grip4'` (default): On-the-fly computation from GRIP4 main roads.
        2. `road_source='precomputed'`: Pre-computed distance raster asset (`precomputed_asset`).
        """
        source_mode = (road_source or "grip4").lower()
        if source_mode == "precomputed" and precomputed_asset:
            logger.info(f"Loading pre-computed road distance asset: {precomputed_asset}")
            return (
                ee.Image(precomputed_asset)
                .select([0])
                .clamp(0, max_dist)
                .unmask(max_dist)
                .clip(aoi)
                .toFloat()
                .rename("dist_roads")
            )

        search_buffer = max(10000.0, float(max_dist))
        roads_fc = self.extract_grip4_roads(
            aoi=aoi,
            buffer_m=search_buffer,
            main_roads_only=main_roads_only,
            road_types=road_types
        )
        roads_mask = (
            roads_fc
            .map(lambda ft: ft.set("constant", 1))
            .reduceToImage(["constant"], ee.Reducer.max())
            .unmask(0)
        )
        return self._compute_fast_distance(
            source_binary_mask=roads_mask,
            aoi=aoi,
            band_name="dist_roads",
            max_dist=max_dist,
            in_meters=in_meters,
            neighborhood_px=neighborhood_px
        )

    def calculate_distance_to_settlement(
        self,
        aoi: ee.Geometry,
        max_dist: float = 50000,
        in_meters: bool = True,
        neighborhood_px: int = 1024
    ) -> ee.Image:
        """
        Compute Euclidean distance to nearest HRSL settlement pixel (`dist_settlement`).
        """
        search_buffer = max(10000.0, float(max_dist))
        hrsl = (
            ee.ImageCollection(self.HRSL_ASSET)
            .filterBounds(aoi.buffer(search_buffer))
            .mosaic()
        )
        settlement_mask = hrsl.gt(0).unmask(0)
        return self._compute_fast_distance(
            source_binary_mask=settlement_mask,
            aoi=aoi,
            band_name="dist_settlement",
            max_dist=max_dist,
            in_meters=in_meters,
            neighborhood_px=neighborhood_px
        )

    def calculate_distance_metrics(
        self,
        aoi: ee.Geometry,
        max_dist: float = 50000,
        in_meters: bool = True,
        include_roads: bool = True,
        include_coast: bool = True,
        include_settlement: bool = True,
        road_source: str = "grip4",
        main_roads_only: bool = True,
        road_types: Optional[List[int]] = None,
        precomputed_road_asset: Optional[str] = None,
        neighborhood_px: int = 1024
    ) -> Optional[ee.Image]:
        """
        Calculate stacked Euclidean distance predictors (`dist_roads`, `dist_coast`, `dist_settlement`)
        using `fastDistanceTransform`, clipped to the user-defined `aoi`.

        Parameters
        ----------
        aoi : ee.Geometry
            User-defined area of interest geometry for filtering and clipping results.
        max_dist : float, default=50000
            Maximum search/cap distance in meters (default 50 km) or pixels.
        in_meters : bool, default=True
            If True, output distance in meters; if False, output in pixel units.
        include_roads : bool, default=True
            Compute `dist_roads` (GRIP4 main roads or pre-computed asset).
        include_coast : bool, default=True
            Compute `dist_coast` (S2COAST-2023).
        include_settlement : bool, default=True
            Compute `dist_settlement` (HRSL).
        road_source : str, default='grip4'
            Road data source: `'grip4'` (on-the-fly main roads) or `'precomputed'`.
        main_roads_only : bool, default=True
            When `road_source='grip4'`, filter `GP_RTP` to `[1, 2, 3]` (Highways, Primary, Secondary).
        road_types : list of int, optional
            Custom `GP_RTP` codes for GRIP4 road filtering.
        precomputed_road_asset : str, optional
            GEE Image asset ID when `road_source='precomputed'`.
        neighborhood_px : int, default=1024
            Neighborhood window in pixels for `fastDistanceTransform`.

        Returns
        -------
        ee.Image or None
            Stacked distance image containing requested bands (`dist_roads`, `dist_coast`, `dist_settlement`).
        """
        try:
            logger.info(
                f"Calculating fast distance metrics (max_dist={max_dist}, in_meters={in_meters}, "
                f"road_source={road_source}, main_roads_only={main_roads_only})..."
            )
            layers = []

            if include_roads:
                dist_roads = self.calculate_distance_to_roads(
                    aoi=aoi,
                    max_dist=max_dist,
                    in_meters=in_meters,
                    road_source=road_source,
                    main_roads_only=main_roads_only,
                    road_types=road_types,
                    precomputed_asset=precomputed_road_asset,
                    neighborhood_px=neighborhood_px
                )
                layers.append(dist_roads)

            if include_coast:
                dist_coast = self.calculate_distance_to_coast(
                    aoi=aoi,
                    max_dist=max_dist,
                    in_meters=in_meters,
                    neighborhood_px=neighborhood_px
                )
                layers.append(dist_coast)

            if include_settlement:
                dist_settlement = self.calculate_distance_to_settlement(
                    aoi=aoi,
                    max_dist=max_dist,
                    in_meters=in_meters,
                    neighborhood_px=neighborhood_px
                )
                layers.append(dist_settlement)

            if not layers:
                logger.warning("No distance layers were requested")
                return None

            distance_metrics = ee.Image.cat(*layers) if len(layers) > 1 else layers[0]
            logger.info("Successfully constructed fast distance predictor stack")
            return distance_metrics

        except Exception as e:
            logger.error(f"Failed to calculate distance metrics: {str(e)}")
            return None

#=================================== Predictor Calculation ===================================
class PredictorCalculation:
    """
    Main orchestration for predictor computation workflows.
    This class coordinates between terrain_calculator, SpectralCalculator, SAR.py, 
    and distance_calculator to provide a unified interface for predictor computation.
    """
    
    def __init__(self):
        """Initialize the enhanced predictor service with all calculator backends."""
        ensure_ee_initialized()
        
        # Initialize calculator backends
        self.terrain_calc = terrain_calculator()
        self.spectral_calc = SpectralCalculator()
        self.sar_calc = get_sar_data()
        self.distance_calc = distance_calculator()
        
        logger.info("Predictor Calculation initialized with terrain, spectral, SAR, and distance calculators")
    def detect_sensor(
        self,
        sensor_hint: Optional[str] = None,
        composite: Optional[ee.Image] = None,
        collection: Optional[ee.ImageCollection] = None
    ) -> str:
        """
        Detect sensor family from explicit hint, composite image, or collection.
        Returns: 'LANDSAT', 'SENTINEL', or 'UNKNOWN'
        """
        return sensor_detector.detect_sensor(
            sensor_hint=sensor_hint,
            composite=composite,
            collection=collection
        )

    def get_available_predictors(
        self,
        sensor: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        composite: Optional[ee.Image] = None,
        collection: Optional[ee.ImageCollection] = None
    ) -> Dict[str, Any]:
        """
        Query availability of all registered predictors based on sensor and temporal criteria.

        Parameters
        ----------
        sensor : str, optional
            Sensor identifier or hint (e.g. 'Landsat', 'Sentinel-2', 'L8_SR')
        start_date : str, optional
            Start date string ('YYYY-MM-DD')
        end_date : str, optional
            End date string ('YYYY-MM-DD')
        composite : ee.Image, optional
            Composite image to infer sensor or date range from if not provided
        collection : ee.ImageCollection, optional
            Collection to infer sensor or date range from if not provided

        Returns
        -------
        Dict[str, Any] : Dictionary containing detected sensor, resolved dates, and predictor availability mapping.
        """
        detected_sensor = self.detect_sensor(sensor_hint=sensor, composite=composite, collection=collection)

        if not start_date or not end_date:
            resolved_start, resolved_end = self._resolve_date_range({}, collection=collection, composite=composite)
            start_date = start_date or resolved_start
            end_date = end_date or resolved_end

        predictors = predictor_catalog.check_availability(
            sensor=detected_sensor,
            start_date=start_date,
            end_date=end_date
        )

        categories = sorted(list(set(p["category"] for p in predictors.values())))
        available_count = sum(1 for p in predictors.values() if p["available"])

        return {
            "sensor": detected_sensor,
            "start_date": start_date,
            "end_date": end_date,
            "categories": categories,
            "predictors": predictors,
            "total_count": len(predictors),
            "available_count": available_count
        }

    #Input validation
    def validate_prerequisites(self, composite: ee.Image, aoi: ee.Geometry, 
                             collection: ee.ImageCollection = None) -> Dict[str, Any]:
        """
        Validate that required inputs are available for predictor computation.
        
        Parameters:
        -----------
        composite : ee.Image
            Composite image from Module 1
        aoi : ee.Geometry
            Area of interest geometry
        collection : ee.ImageCollection, optional
            Image collection for spectral indices
            
        Returns:
        --------
        Dict[str, Any] : Validation result with status and details
        """
        try:
            validation_result = {
                "valid": True,
                "errors": [],
                "warnings": []
            }
            
            # Validate composite image
            if composite is None:
                validation_result["errors"].append("Composite image is required")
                validation_result["valid"] = False
            else:
                try:
                    # Test if composite is accessible. Used for stacking with the predictor
                    band_names = composite.bandNames().getInfo()
                    if not band_names:
                        validation_result["errors"].append("Composite image has no bands")
                        validation_result["valid"] = False
                except Exception as e:
                    validation_result["errors"].append(f"Cannot access composite image: {str(e)}")
                    validation_result["valid"] = False
            
            # Validate AOI
            if aoi is None:
                validation_result["errors"].append("Area of interest (AOI) is required")
                validation_result["valid"] = False
            
            # Validate collection for spectral indices (warning only)
            if collection is None:
                validation_result["warnings"].append("Image collection not provided - spectral indices will use composite only")
            
            logger.info(f"Prerequisite validation completed: valid={validation_result['valid']}")
            return validation_result
            
        except Exception as e:
            logger.error(f"Error during prerequisite validation: {str(e)}")
            return {
                "valid": False,
                "errors": [f"Validation failed: {str(e)}"],
                "warnings": []
            }
    def _resolve_date_range(self, predictor_config: Dict[str, Any],
                            collection: ee.ImageCollection = None,
                            composite: ee.Image = None):
        """
        Resolve the (start_date, end_date) selected in Module 1 / Step 1.

        Priority: predictor_config -> collection 'query_start_date'/'query_end_date'
        -> composite 'composite_start_date'/'composite_end_date' ->
        min/max 'system:time_start' of the collection.

        Returns:
        --------
        tuple : (start_date, end_date) as strings, or (None, None) if unresolved.
        """
        start = predictor_config.get("start_date")
        end = predictor_config.get("end_date")
        if start is not None and end is not None:
            return str(start), str(end)

        def _get_props(obj, start_key, end_key):
            try:
                info = ee.Dictionary({
                    "s": obj.get(start_key),
                    "e": obj.get(end_key),
                }).getInfo()
                if info.get("s") and info.get("e"):
                    return str(info["s"]), str(info["e"])
            except Exception as e:
                logger.debug(f"Could not read date properties {start_key}/{end_key}: {e}")
            return None

        if collection is not None:
            found = _get_props(collection, "query_start_date", "query_end_date")
            if found:
                return found
        if composite is not None:
            found = _get_props(composite, "composite_start_date", "composite_end_date")
            if found:
                return found
            found = _get_props(composite, "date_range_start", "date_range_end")
            if found:
                return found
        if collection is not None:
            try:
                rng = collection.reduceColumns(
                    ee.Reducer.minMax(), ["system:time_start"]
                ).getInfo()
                if rng and rng.get("min") is not None and rng.get("max") is not None:
                    fmt = lambda ms: ee.Date(ms).format("YYYY-MM-dd").getInfo()
                    return fmt(rng["min"]), fmt(rng["max"])
            except Exception as e:
                logger.debug(f"Could not derive date range from collection: {e}")
        return None, None

    def _resolve_sar_selection(self, predictor_config: Dict[str, Any], spectral_indices: list):
        """
        Determine which Sentinel-1 and PALSAR bands were requested.

        SAR predictors can be requested via:
        - predictor_config["sentinel1"] / predictor_config["individual_predictors"]["sentinel1"]
        - predictor_config["palsar"] / predictor_config["individual_predictors"]["palsar"]
        - band names (S1_VV, S1_VH, S1_VV_VH, PALSAR_HH, PALSAR_HV, PALSAR_HH_HV) mixed
          into the spectral index list (luma-be forwards non-terrain predictors in "spectral_indices")

        Returns:
        --------
        tuple : (selected_s1_bands: list, selected_palsar_bands: list, remaining_spectral_indices: list)
        """
        all_s1 = self.sar_calc.SAR_DATASETS["S1_GRD"]["output_bands"]
        all_palsar = self.sar_calc.SAR_DATASETS["PALSAR_EPOCH"]["output_bands"]
        selected_s1 = []
        selected_palsar = []

        def _collect(value, target_list, valid_bands):
            if value is True:
                target_list.extend(valid_bands)
            elif isinstance(value, (list, tuple, set)):
                target_list.extend(str(v).upper() for v in value)

        _collect(predictor_config.get("sentinel1"), selected_s1, all_s1)
        _collect(predictor_config.get("individual_predictors", {}).get("sentinel1"), selected_s1, all_s1)
        _collect(predictor_config.get("palsar"), selected_palsar, all_palsar)
        _collect(predictor_config.get("individual_predictors", {}).get("palsar"), selected_palsar, all_palsar)

        remaining = []
        for idx in spectral_indices or []:
            idx_u = str(idx).upper()
            if idx_u in all_s1:
                selected_s1.append(idx_u)
            elif idx_u in all_palsar:
                selected_palsar.append(idx_u)
            else:
                remaining.append(idx)

        selected_s1 = [b for b in all_s1 if b in set(selected_s1)]
        selected_palsar = [b for b in all_palsar if b in set(selected_palsar)]
        return selected_s1, selected_palsar, remaining

    #core function: Compute Predictor
    def compute_predictors(self, composite: ee.Image, aoi: ee.Geometry, predictor_config: Dict[str, Any], collection: ee.ImageCollection = None,
                          progress_callback: callable = None, sensor: Optional[str] = None) -> Dict[str, Any]:
        """
        Compute predictors based on configuration using backend calculators.
        This method orchestrates the computation workflow:
        1. Validate inputs
        2. Compute terrain metrics (if selected) and spectral indices (if selected)
        3. Stack all predictors and returned them
        
        Parameters:
        -----------
        composite : ee.Image
            Composite image from Module 1
        aoi : ee.Geometry
            Area of interest geometry
        predictor_config : Dict[str, Any]
            Configuration with terrain and spectral_indices selections
        collection : ee.ImageCollection, optional
            Image collection for spectral indices
        progress_callback : callable, optional
            Callback function for progress updates
            
        Returns:
        --------
        Dict[str, Any] : Computation results with predictors and metadata
        """
        try:
            # Initialize result structure
            result = {
                "success": False,
                "predictors": {},
                "stacked_predictors": None,
                "predictor_info": [],
                "total_bands": 0,
                "errors": [],
                "warnings": []
            }
            
            # Step 1: Validate prerequisites
            if progress_callback:
                progress_callback(5, "Validating prerequisites...")
            
            validation = self.validate_prerequisites(composite, aoi, collection)
            if not validation["valid"]:
                result["errors"] = validation["errors"]
                return result
            
            result["warnings"].extend(validation["warnings"])
            
            # Step 1b: Detect sensor family
            detected_sensor = self.detect_sensor(
                sensor_hint=sensor or predictor_config.get("sensor"),
                composite=composite,
                collection=collection
            )
            logger.info(f"Predictor computation utilizing detected sensor: {detected_sensor}")
            result["sensor"] = detected_sensor

            # Step 2: Add original multispectral bands
            if progress_callback:
                progress_callback(15, "Processing original multispectral bands...")
            
            try:
                original_bands = composite
                result["predictors"]["original_bands"] = original_bands
                band_names = original_bands.bandNames().getInfo()
                result["predictor_info"].append(f"Kanal Multispektral ({len(band_names)} band)")
                logger.info(f"Added {len(band_names)} original multispectral bands")
            except Exception as e:
                error_msg = f"Failed to process original bands: {str(e)}"
                result["errors"].append(error_msg)
                logger.error(error_msg)
            
            # Step 3: Compute terrain metrics (if selected)
            terrain_layers = []
            
            # Check for individual terrain layer selections
            if "individual_predictors" in predictor_config:
                individual_preds = predictor_config["individual_predictors"]
                
                if individual_preds.get("elevation", False):
                    if progress_callback:
                        progress_callback(25, "Computing elevation...")
                    try:
                        elevation = self.terrain_calc.calculate_elevation(aoi)
                        if elevation is not None:
                            terrain_layers.append(elevation)
                            logger.info("Successfully computed elevation")
                    except Exception as e:
                        result["warnings"].append(f"Failed to compute elevation: {str(e)}")
                
                if individual_preds.get("slope", False):
                    if progress_callback:
                        progress_callback(30, "Computing slope...")
                    try:
                        slope = self.terrain_calc.calculate_slope(aoi)
                        if slope is not None:
                            terrain_layers.append(slope)
                            logger.info("Successfully computed slope")
                    except Exception as e:
                        result["warnings"].append(f"Failed to compute slope: {str(e)}")
                
                if individual_preds.get("aspect", False):
                    if progress_callback:
                        progress_callback(35, "Computing aspect...")
                    try:
                        aspect = self.terrain_calc.calculate_aspect(aoi)
                        if aspect is not None:
                            terrain_layers.append(aspect)
                            logger.info("Successfully computed aspect")
                    except Exception as e:
                        result["warnings"].append(f"Failed to compute aspect: {str(e)}")

                if individual_preds.get("landform", False):
                    if progress_callback:
                        progress_callback(38, "Computing landform...")
                    try:
                        landform = self.terrain_calc.get_landform(aoi)
                        if landform is not None:
                            terrain_layers.append(landform)
                            logger.info("Successfully computed landform")
                    except Exception as e:
                        result["warnings"].append(f"Failed to compute landform: {str(e)}")
                
                # Stack individual terrain layers if any were computed
                if terrain_layers:
                    if len(terrain_layers) == 1:
                        terrain_metrics = terrain_layers[0]
                    else:
                        terrain_metrics = terrain_layers[0]
                        for layer in terrain_layers[1:]:
                            terrain_metrics = terrain_metrics.addBands(layer)
                    
                    result["predictors"]["terrain"] = terrain_metrics
                    terrain_names = []
                    if individual_preds.get("elevation", False):
                        terrain_names.append("Elevation")
                    if individual_preds.get("slope", False):
                        terrain_names.append("Slope")
                    if individual_preds.get("aspect", False):
                        terrain_names.append("Aspect")
                    if individual_preds.get("landform", False):
                        terrain_names.append("Landform")
                    
                    result["predictor_info"].append(f"Data Topografi ({len(terrain_layers)} layer: {', '.join(terrain_names)})")
                    
            elif predictor_config.get("terrain", False):
                # Fallback to original terrain calculation for compatibility
                if progress_callback:
                    progress_callback(35, "Computing terrain metrics...")
                
                try:
                    terrain_metrics = self.terrain_calc.calculate_terrain(aoi)
                    if terrain_metrics is not None:
                        result["predictors"]["terrain"] = terrain_metrics
                        result["predictor_info"].append("Data Topografi (3 band: Elevation, Slope, Aspect)")
                        logger.info("Successfully computed terrain metrics")
                    else:
                        result["warnings"].append("Failed to compute terrain metrics")
                except Exception as e:
                    error_msg = f"Error computing terrain metrics: {str(e)}"
                    result["warnings"].append(error_msg)
                    logger.warning(error_msg)
            
            # Step 4: Compute spectral indices (if selected)
            spectral_indices = []
            index_coefficients = None
            
            # Check for individual predictor format first
            if "individual_predictors" in predictor_config:
                spectral_indices = predictor_config["individual_predictors"].get("spectral_indices", [])
                index_coefficients = predictor_config["individual_predictors"].get("index_coefficients", None)
            else:
                # Fallback to original format for compatibility
                spectral_indices = predictor_config.get("spectral_indices", [])
                index_coefficients = predictor_config.get("index_coefficients", None)
            
            # Separate SAR (Sentinel-1 & PALSAR) band selections from spectral indices
            s1_bands, palsar_bands, spectral_indices = self._resolve_sar_selection(predictor_config, spectral_indices)
            
            if spectral_indices:
                if progress_callback:
                    progress_callback(65, "Computing spectral indices...")
                
                try:
                    if collection is not None:
                        # Use collection with reducer
                        spectral_result = self.spectral_calc.calculate_indices_with_collection(
                            collection=collection,
                            aoi=aoi,
                            index_list=spectral_indices,
                            coefficients=index_coefficients,
                            reducer_method="mean",
                            sensor=detected_sensor
                        )
                    else:
                        # Use composite directly
                        spectral_result = self.spectral_calc.calculate_indices(
                            image=composite,
                            index_list=spectral_indices,
                            coefficients=index_coefficients,
                            sensor=detected_sensor
                        )
                    
                    if spectral_result is not None:
                        result["predictors"]["spectral"] = spectral_result
                        num_indices = len(spectral_indices)
                        method_used = "rata-rata koleksi" if collection else "komposit"
                        result["predictor_info"].append(f"Indeks Spektral ({num_indices} band - {method_used})")
                        logger.info(f"Successfully computed {num_indices} spectral indices")
                    else:
                        result["warnings"].append("Failed to compute spectral indices")
                except Exception as e:
                    error_msg = f"Error computing spectral indices: {str(e)}"
                    result["warnings"].append(error_msg)
                    logger.warning(error_msg)
            
            # Step 4b: Compute Sentinel-1 SAR predictors (if selected)
            if s1_bands:
                if progress_callback:
                    progress_callback(80, "Computing Sentinel-1 SAR predictors...")
                try:
                    start_date, end_date = self._resolve_date_range(
                        predictor_config, collection=collection, composite=composite
                    )
                    if start_date is None or end_date is None:
                        result["warnings"].append(
                            "Sentinel-1 skipped: could not resolve date range from Module 1"
                        )
                    else:
                        s1_opts = predictor_config.get("sentinel1_options", {}) or {}
                        s1_image = self.sar_calc.get_sentinel1(
                            aoi=aoi,
                            start_date=start_date,
                            end_date=end_date,
                            orbit_pass=s1_opts.get("orbit_pass", "BOTH"),
                            reducer=s1_opts.get("reducer", "median"),
                            speckle_filter=s1_opts.get("speckle_filter", "focal_median"),
                            filter_radius_m=s1_opts.get("filter_radius_m", 30.0),
                            include_ratio=True,
                        )
                        if s1_image is None:
                            result["warnings"].append(
                                f"Sentinel-1 data unavailable for {start_date} to {end_date}"
                            )
                        else:
                            result["predictors"]["sentinel1"] = s1_image.select(s1_bands)
                            result["predictor_info"].append(
                                f"Sentinel-1 SAR ({len(s1_bands)} band: {', '.join(s1_bands)})"
                            )
                            logger.info(f"Added Sentinel-1 bands: {s1_bands}")
                except Exception as e:
                    error_msg = f"Error computing Sentinel-1 predictors: {str(e)}"
                    result["warnings"].append(error_msg)
                    logger.warning(error_msg)

            # Step 4c: Compute ALOS PALSAR / PALSAR-2 L-band SAR predictors (if selected)
            if palsar_bands:
                if progress_callback:
                    progress_callback(85, "Computing ALOS PALSAR SAR predictors...")
                try:
                    start_date, end_date = self._resolve_date_range(
                        predictor_config, collection=collection, composite=composite
                    )
                    if start_date is None or end_date is None:
                        result["warnings"].append(
                            "PALSAR skipped: could not resolve date range from Module 1"
                        )
                    else:
                        palsar_opts = predictor_config.get("palsar_options", {}) or {}
                        palsar_image = self.sar_calc.get_palsar(
                            aoi=aoi,
                            start_date=start_date,
                            end_date=end_date,
                            reducer=palsar_opts.get("reducer", "median"),
                            speckle_filter=palsar_opts.get("speckle_filter", "focal_median"),
                            filter_radius_m=palsar_opts.get("filter_radius_m", 30.0),
                            include_ratio=True,
                            mask_layover_shadow=palsar_opts.get("mask_layover_shadow", True),
                            allow_nearest_year_fallback=palsar_opts.get("allow_nearest_year_fallback", True),
                        )
                        if palsar_image is None:
                            result["warnings"].append(
                                f"PALSAR data unavailable for {start_date} to {end_date}"
                            )
                        else:
                            result["predictors"]["palsar"] = palsar_image.select(palsar_bands)
                            result["predictor_info"].append(
                                f"ALOS PALSAR SAR ({len(palsar_bands)} band: {', '.join(palsar_bands)})"
                            )
                            logger.info(f"Added ALOS PALSAR bands: {palsar_bands}")
                except Exception as e:
                    error_msg = f"Error computing PALSAR predictors: {str(e)}"
                    result["warnings"].append(error_msg)
                    logger.warning(error_msg)
            
            # Step 4d: Compute Distance-based predictors (if selected)
            distance_layers = []
            distance_names = []
            dist_opts = predictor_config.get("distance_options", {}) or {}
            max_dist_val = dist_opts.get("max_dist", 50000)
            in_meters_val = dist_opts.get("in_meters", True)
            road_source_val = dist_opts.get("road_source", "grip4")
            main_roads_only_val = dist_opts.get("main_roads_only", True)
            precomputed_road_asset_val = dist_opts.get("precomputed_road_asset", None)

            if "individual_predictors" in predictor_config:
                ind_p = predictor_config["individual_predictors"]
                want_all_dist = bool(predictor_config.get("distance", False) or ind_p.get("distance", False))
                want_roads = want_all_dist or bool(ind_p.get("dist_roads", False))
                want_coast = want_all_dist or bool(ind_p.get("dist_coast", False))
                want_settlement = want_all_dist or bool(ind_p.get("dist_settlement", False))
            else:
                want_all_dist = bool(predictor_config.get("distance", False))
                want_roads = want_all_dist
                want_coast = want_all_dist
                want_settlement = want_all_dist

            if want_roads or want_coast or want_settlement:
                if progress_callback:
                    progress_callback(88, "Computing distance predictors...")
                try:
                    dist_img = self.distance_calc.calculate_distance_metrics(
                        aoi=aoi,
                        max_dist=max_dist_val,
                        in_meters=in_meters_val,
                        include_roads=want_roads,
                        include_coast=want_coast,
                        include_settlement=want_settlement,
                        road_source=road_source_val,
                        main_roads_only=main_roads_only_val,
                        precomputed_road_asset=precomputed_road_asset_val,
                    )
                    if dist_img is not None:
                        result["predictors"]["distance"] = dist_img
                        if want_roads:
                            distance_names.append("dist_roads")
                        if want_coast:
                            distance_names.append("dist_coast")
                        if want_settlement:
                            distance_names.append("dist_settlement")
                        result["predictor_info"].append(
                            f"Jarak & Aksesibilitas ({len(distance_names)} band: {', '.join(distance_names)})"
                        )
                        logger.info(f"Added distance bands: {distance_names}")
                except Exception as e:
                    error_msg = f"Error computing distance predictors: {str(e)}"
                    result["warnings"].append(error_msg)
                    logger.warning(error_msg)

            # Step 5: Stack all predictors together
            if result["predictors"]:
                if progress_callback:
                    progress_callback(90, "Stacking predictors...")
                
                try:
                    predictor_list = list(result["predictors"].values())
                    stacked_predictors = predictor_list[0]
                    
                    for predictor in predictor_list[1:]:
                        stacked_predictors = stacked_predictors.addBands(predictor)
                    
                    # Get total band count
                    total_bands = stacked_predictors.bandNames().size().getInfo()
                    
                    result["stacked_predictors"] = stacked_predictors
                    result["total_bands"] = total_bands
                    result["success"] = True
                    
                    if progress_callback:
                        progress_callback(100, f"Completed! {total_bands} predictors ready.")
                    
                    logger.info(f"Successfully stacked {total_bands} predictor bands")
                    
                except Exception as e:
                    error_msg = f"Error stacking predictors: {str(e)}"
                    result["errors"].append(error_msg)
                    logger.error(error_msg)
            else:
                result["errors"].append("No predictors were successfully computed")
            
            return result
            
        except Exception as e:
            error_msg = f"Unexpected error in predictor computation: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {
                "success": False,
                "predictors": {},
                "stacked_predictors": None,
                "predictor_info": [],
                "total_bands": 0,
                "errors": [error_msg],
                "warnings": []
            }
    #Predictor calculation summary
    #This summary will be used to determine which predictor is used for classification
    def prepare_module6_summary(self, computation_result: Dict[str, Any], 
                               predictor_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare summary metadata for Module 6 consumption.
        
        Parameters:
        -----------
        computation_result : Dict[str, Any]
            Results from compute_predictors()
        predictor_config : Dict[str, Any]
            Original predictor configuration
            
        Returns:
        --------
        Dict[str, Any] : Module 6 summary with metadata
        """
        try:
            import datetime
            
            summary = {
                "status": "completed" if computation_result["success"] else "failed",
                "total_bands": computation_result["total_bands"],
                "components": computation_result["predictor_info"],
                "predictor_config": predictor_config,
                "computation_errors": computation_result["errors"],
                "computation_warnings": computation_result["warnings"],
                "timestamp": datetime.datetime.now().isoformat(),
                "service_version": "enhanced_predictors_v1.0"
            }
            
            logger.info("Prepared Module 6 summary with metadata")
            return summary
            
        except Exception as e:
            logger.error(f"Error preparing Module 6 summary: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.datetime.now().isoformat()
            }
#=================================== Correlation Analysis ===================================
class EECorrelationAnalysis:
    """
    Minimal correlation analysis using Earth Engine native operations.
    
    This class computes correlation matrices for multi-band Earth Engine images
    using random sampling and server-side correlation computation.
    
    Supported Correlation Methods:
    - Spearman: Robust to outliers, detects monotonic relationships (default)
    - Pearson: Faster computation, detects linear relationships only
    
    Usage Example:
        analyzer = EECorrelationAnalysis(predictor_image, aoi_geometry)
        corr_matrix = analyzer.compute_correlation(num_samples=5000)
        fig = analyzer.visualize()
    
    """
    
    def __init__(self, image: ee.Image, aoi: ee.Geometry, method: str = 'spearman'):
        """
        Initialize the correlation analysis.
        
        Parameters:
        -----------
        image : ee.Image
            Multi-band Earth Engine image containing predictor layers
        aoi : ee.Geometry
            Area of Interest geometry for analysis
        method : str, default='spearman'
            Correlation method to use:
            - 'spearman': Robust to outliers, detects monotonic relationships
            - 'pearson': Faster, detects linear relationships only
            
        Raises:
        -------
        ValueError
            If method is not 'spearman' or 'pearson'
        """
        ensure_ee_initialized()
        
        self.image = image
        self.aoi = aoi
        self.method = method.lower()
        self.correlation_matrix = None
        
        # Validate method
        if self.method not in ['spearman', 'pearson']:
            logger.error(f"Invalid correlation method: {method}")
            raise ValueError(f"Method must be 'spearman' or 'pearson', got '{method}'")
        
        logger.info(f"Correlation analysis initialized with {method} method")

    def generate_random_samples(self, num_samples: int = 2500, seed: int = 42) -> ee.FeatureCollection:
        """
        Generate random sample points within the Area of Interest.
        
        Uses Earth Engine's randomPoints to create a specified number of 
        random points within the AOI boundary for correlation analysis.
        
        Parameters:
        -----------
        num_samples : int, default=2500
            Number of random points to generate
        seed : int, default=42
            Random seed for reproducibility
            
        Returns:
        --------
        ee.FeatureCollection
            Collection of random points within the AOI
            
        Raises:
        -------
        Exception
            If random point generation fails
            
        Example:
        --------
        >>> samples = analyzer.generate_random_samples(num_samples=1000, seed=123)
        >>> print(samples.size().getInfo())
        1000
        """
        try:
            logger.info(f"Generating {num_samples} random sample points with seed {seed}")
            
            samples = ee.FeatureCollection.randomPoints(
                region=self.aoi,
                points=num_samples,
                seed=seed
            )
            
            logger.info(f"Successfully generated {num_samples} random sample points")
            return samples
            
        except Exception as e:
            logger.error(f"Failed to generate random samples: {str(e)}")
            raise

    def compute_correlation(self, num_samples: int = 2500, scale: int = 30, seed: int = 42) -> pd.DataFrame:
        """
        Compute correlation matrix using the specified correlation method.
        
        This method performs server-side correlation analysis by:
        1. Generating random sample points in the AOI
        2. Extracting pixel values at those points for all bands
        3. Computing pairwise correlations using Earth Engine reducers
        4. Building a symmetric correlation matrix
        5. Converting to pandas DataFrame with band names as labels
        
        Parameters:
        -----------
        num_samples : int, default=2500
            Number of random samples to use for correlation computation
        scale : int, default=30
            Scale in meters for sampling pixel values
        seed : int, default=42
            Random seed for reproducibility
            
        Returns:
        --------
        pd.DataFrame
            Correlation matrix with band names as index and columns
            
            
        Example:
        --------
        >>> corr_matrix = analyzer.compute_correlation(num_samples=5000, scale=30)
        >>> print(corr_matrix.shape)
        (10, 10)
        """
        try:
            logger.info(f"Computing {self.method} correlation matrix with {num_samples} samples at {scale}m scale")
            
            # Generate random samples
            samples = self.generate_random_samples(num_samples=num_samples, seed=seed)
            
            # Get band names from the image
            band_names = self.image.bandNames().getInfo()
            n_bands = len(band_names)
            
            logger.info(f"Processing {n_bands} bands: {band_names}")
            
            # Initialize correlation matrix with ones on diagonal
            corr_matrix = np.ones((n_bands, n_bands))
            
            # Select the appropriate reducer
            if self.method == 'spearman':
                reducer = ee.Reducer.spearmansCorrelation()
            else:  # pearson
                reducer = ee.Reducer.pearsonsCorrelation()
            
            # Compute pairwise correlations for upper triangle
            for i in range(n_bands):
                for j in range(i + 1, n_bands):
                    logger.debug(f"Computing correlation between {band_names[i]} and {band_names[j]}")
                    
                    # Select the pair of bands
                    pair_image = self.image.select([band_names[i], band_names[j]])
                    
                    # Sample the pair at random points
                    pair_samples = pair_image.sampleRegions(
                        collection=samples,
                        scale=scale,
                        tileScale=4
                    )
                    
                    # Compute correlation using selected reducer
                    stats = pair_samples.reduceColumns(
                        reducer=reducer,
                        selectors=[band_names[i], band_names[j]]
                    ).getInfo()
                    
                    # Extract correlation value
                    corr_value = stats.get('correlation', 0)
                    
                    # Fill symmetric matrix
                    corr_matrix[i, j] = corr_value
                    corr_matrix[j, i] = corr_value
            
            # Convert to DataFrame with band names as labels
            correlation_df = pd.DataFrame(
                corr_matrix,
                index=band_names,
                columns=band_names
            )
            
            # Store in instance variable
            self.correlation_matrix = correlation_df
            
            logger.info(f"Successfully computed {self.method} correlation matrix for {n_bands} bands")
            return correlation_df
            
        except Exception as e:
            logger.error(f"Failed to compute correlation matrix: {str(e)}")
            raise

    def visualize(self) -> go.Figure:
        """
        Create an interactive heatmap visualization of the correlation matrix.
        
        The visualization displays only the lower triangle of the correlation matrix
        to avoid redundancy, with an interactive Plotly heatmap using a diverging
        colorscale centered at zero.
        
        Returns:
        --------
        plotly.graph_objects.Figure
            Interactive Plotly figure with correlation heatmap
            
        Raises:
        -------
        ValueError
            If correlation matrix has not been computed yet
        Exception
            If visualization creation fails
            
        Example:
        --------
        >>> fig = analyzer.visualize()
        >>> fig.show()
        """
        try:
            # Check if correlation matrix exists
            if self.correlation_matrix is None:
                logger.error("Correlation matrix not computed. Call compute_correlation() first.")
                raise ValueError("Correlation matrix not computed. Call compute_correlation() first.")
            
            logger.info("Creating correlation matrix visualization")
            
            # Create a copy and mask upper triangle with NaN
            corr_data = self.correlation_matrix.copy()
            mask = np.triu(np.ones_like(corr_data, dtype=bool), k=0)
            corr_data[mask] = np.nan
            
            # Create hover text with predictor names and correlation values
            hover_text = []
            for i in range(len(corr_data)):
                hover_row = []
                for j in range(len(corr_data.columns)):
                    if np.isnan(corr_data.iloc[i, j]):
                        hover_row.append("N/A")
                    else:
                        hover_row.append(
                            f"{corr_data.index[i]} ↔ {corr_data.columns[j]}<br>"
                            f"Correlation: {corr_data.iloc[i, j]:.4f}"
                        )
                hover_text.append(hover_row)
            
            # Create Plotly heatmap
            fig = go.Figure(data=go.Heatmap(
                z=corr_data.values,
                x=corr_data.columns,
                y=corr_data.index,
                colorscale='RdBu_r',
                zmid=0,
                zmin=-1,
                zmax=1,
                hovertext=hover_text,
                hoverinfo='text',
                colorbar=dict(title="Correlation")
            ))
            
            # Set layout with title and axis labels
            method_name = self.method.capitalize()
            fig.update_layout(
                title=f"Predictor Correlation Matrix ({method_name})",
                xaxis_title="Predictors",
                yaxis_title="Predictors",
                width=800,
                height=800
            )
            
            logger.info("Successfully created correlation matrix visualization")
            return fig
            
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Failed to create visualization: {str(e)}")
            raise