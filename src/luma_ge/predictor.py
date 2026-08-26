"""
Revised Predictors Module for Luma Geospatial Engine

This module provides backend services for calculating terrain metrics, spectral indices, and distance metrics
to enhance land cover classification. This version is revised to follow the latest update in Luma-lite

"""
import ee
import logging
from typing import List, Optional, Dict, Any, Tuple
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from .ee_config import ensure_ee_initialized
import matplotlib.pyplot as plt
# Configure logging for development
logging.basicConfig(level=logging.INFO,
format='%(asctime)s - %(levelname)s - %(message)s')

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
            return None # type: ignore

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

#=================================== Spectral Calculator Backend ===================================
#reference for the use of spectral index in LC mapping: https://doi.org/10.3390/rs12183062
class SpectralCalculator:
    """
    Backend service for spectral index calculations using spyndex library.
    Migrated from Streamlit UI to follow clean architecture patterns.
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
        
        #Define supported indices with categories
        #check for more index here: https://awesome-ee-spectral-indices.readthedocs.io/en/latest/list.html
        #main reference: https://doi.org/10.3390/rs12183062 and remap
        #can be added more
        self.supported_indices = {
            #Vegetation Indices
            "NDVI": "🌿 Vegetasi", 
            "EVI": "🌿 Vegetasi", 
            "SAVI": "🌿 Vegetasi", #https://doi.org/10.1080/24749508.2019.1608409
            "MSAVI": "🌿 Vegetasi",
            "OSAVI": "🌿 Vegetasi",
            "ARVI": "🌿 Vegetasi",
            "GBNDVI": "🌿 Vegetasi",
            "GNDVI": "🌿 Vegetasi",
            #Water Indices
            "MNDWI": "💧 Air & Kelembaban",
            "NDMI": "💧 Air & Kelembaban",
            "AWEInsh" : "💧 Air & Kelembaban",
            #Soil and build-up Indices
            "NDBI": "🟫 Tanah dan Lahan Terbangun",
            "DBSI": "🟫 Tanah dan Lahan Terbangun",
            "MBI": "🟫 Tanah dan Lahan Terbangun",
        }
        #Define default coefficients for indices that require them
        #ref: https://www.usgs.gov/landsat-missions/landsat-enhanced-vegetation-index
        #https://doi.org/10.1016/0034-4257(94)90018-3
        self.default_coefficients = {
            "EVI": {
                "g": 2.5,      # Gain factor
                "C1": 6.0,     # Coefficient for aerosol resistance term 
                "C2": 7.5,     # Coefficient for aerosol resistance term 
                "L": 1.0       # Canopy background adjustment factor
            },
            "SAVI": {
                "L": 0.5       #Soil brightness correction factor (0-1)
            },                 #ref https://doi.org/10.3390/agronomy11040652 and https://www.usgs.gov/landsat-missions/landsat-soil-adjusted-vegetation-index
            "MSAVI": {
                "L": 0.5       #Soil adjustment factor
            },                
            "EVI2": {
                "g": 2.5,      #Gain factor
                "L": 1.0       #Canopy background adjustment 
            },                 #ref https://www.usgs.gov/landsat-missions/landsat-enhanced-vegetation-index
            "ARVI": {
                "gamma": 1.0   #Atmospheric resistance coefficient
            }                  #ref https://doi.org/10.1109/36.134076
        }
        
        self.calculated_count = 0
    
    #Get the list of avaliable indices from previously defined list
    def get_available_indices(self, band_names: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Get available spectral indices with metadata 
        
        Parameters:
        -----------
        band_names : List[str], optional
            Available band names to filter compatible indices
            
        Returns:
        --------
        Dict[str, Dict[str, Any]] : Dictionary of available indices with metadata
        """
        try:
            available_indices = {}
            
            for idx_name in self.supported_indices.keys():
                if idx_name in self.spyndex_indices:
                    idx_info = self.spyndex_indices[idx_name]
                    available_indices[idx_name] = {
                        "long_name": idx_info.long_name,
                        "formula": getattr(idx_info, 'formula', ''),
                        "reference": getattr(idx_info, 'reference', ''),
                        "category": self.supported_indices[idx_name],
                    }
                else:
                    # If index not found in spyndex, still add it with warning
                    logger.warning(f"Index '{idx_name}' not found in spyndex library")
                    available_indices[idx_name] = {
                        "long_name": idx_name,
                        "formula": "Formula not available",
                        "reference": "",
                        "category": self.supported_indices[idx_name],
                    }
            
            logger.info(f"Found {len(available_indices)} available spectral indices")
            return available_indices
            
        except Exception as e:
            logger.error(f"Failed to get available indices: {str(e)}")
            return {}
    
    #Used to handle spectral index that have coefficient
    def get_index_coefficients(self, index_name: str) -> Dict[str, float]:
        """
        Get default coefficient values for a spectral index.
        Some spectral indices require additional coefficients beyond the standard
        band inputs (e.g., EVI requires gain factor 'g' and correction coefficients).
        This method returns the default coefficient values from literature.
        
        Parameters:
        -----------
        index_name : str
            Name of the spectral index (e.g., "EVI", "SAVI")
            
        Returns:
        --------
        Dict[str, float] : Dictionary of coefficient names and their default values.
                          Returns empty dict if index doesn't require coefficients.
        
        Example:
        --------
        >>> calc = SpectralCalculator()
        >>> evi_coeffs = calc.get_index_coefficients("EVI")
        >>> print(evi_coeffs)
        {'g': 2.5, 'C1': 6.0, 'C2': 7.5, 'L': 1.0}
        >>> 
        >>> ndvi_coeffs = calc.get_index_coefficients("NDVI")
        >>> print(ndvi_coeffs)
        {}  # NDVI doesn't require coefficients
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
    
    #validate the input coefficient for spectral index calc
    def validate_coefficients(
        self, 
        index_name: str, 
        coefficients: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Validate coefficient values for a spectral index.
        
        Checks that provided coefficients are within reasonable ranges
        and provides warnings for unusual values that might indicate errors.
        
        Parameters:
        -----------
        index_name : str
            Name of the spectral index
        coefficients : Dict[str, float]
            User-provided coefficient values to validate
            
        Returns:
        --------
        Dict[str, Any] : Validation result with 'valid' flag and any warnings
        
        Example:
        --------
        >>> calc = SpectralCalculator()
        >>> result = calc.validate_coefficients("EVI", {"g": 2.5, "C1": 6.0, "C2": 7.5, "L": 1.0})
        >>> print(result)
        {'valid': True, 'warnings': []}
        """
        validation_result = {
            "valid": True,
            "warnings": []
        }
        
        try:
            # Define reasonable ranges for common coefficients
            coefficient_ranges = {
                "g": (0.1, 10.0),      # Gain factor range
                "C1": (0.0, 20.0),     # Aerosol correction range
                "C2": (0.0, 20.0),     # Aerosol correction range
                "L": (0.0, 2.0),       # Canopy/soil adjustment range
                "gamma": (0.1, 5.0)    # Atmospheric resistance range
            }
            
            for coeff_name, coeff_value in coefficients.items():
                # Check if coefficient value is numeric
                if not isinstance(coeff_value, (int, float)):
                    validation_result["warnings"].append(
                        f"Coefficient '{coeff_name}' has non-numeric value: {coeff_value}"
                    )
                    validation_result["valid"] = False
                    continue
                
                # Check if coefficient is within reasonable range
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
    #validate the required band
    def validate_index_requirements(self, index_name: str, available_bands: List[str]) -> bool:
        """
        Validate that required bands are available for index computation.
        
        Parameters:
        -----------
        index_name : str
            Name of the spectral index
        available_bands : List[str]
            List of available band names
            
        Returns:
        --------
        bool : True if requirements are met, False otherwise
        """
        try:
            if index_name not in self.spyndex_indices:
                logger.warning(f"Index '{index_name}' not found in spyndex library")
                return False
            
            # Get required bands from spyndex
            idx_info = self.spyndex_indices[index_name]
            required_bands = getattr(idx_info, 'bands', [])
            
            # Check if all required bands are available
            missing_bands = [band for band in required_bands if band not in available_bands]
            
            if missing_bands:
                logger.warning(f"Index '{index_name}' requires bands {missing_bands} which are not available")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to validate requirements for {index_name}: {str(e)}")
            return False
        
    #calculate index across the collection, composite them into an image using mean reducer
    def calculate_indices_with_collection(
        self, 
        collection: ee.ImageCollection, 
        aoi: ee.Geometry,
        index_list: List[str],
        coefficients: Optional[Dict[str, Dict[str, float]]] = None,
        reducer_method: str = "mean"
    ) -> ee.Image:
        """
        This method applies the spectral index calculation to each image in the collection
        using the map function, then reduces the results using the specified reducer method.
        preserving the spectral characteristics of each individual image.
        
        Parameters:
        -----------
        collection : ee.ImageCollection. Earth Engine image collection with required spectral bands
        aoi : ee.Geometry. Area of interest geometry for clipping results
        index_list : List[str]. List of spectral index names to calculate
        coefficients : Dict[str, Dict[str, float]], optional
            Custom coefficient values for indices that require them.
            Format: {"EVI": {"g": 2.5, "C1": 6.0, "C2": 7.5, "L": 1.0}}
            If not provided, default coefficients will be used.
        reducer_method : str, default="mean". Supported reducer: min, max, median, mean

        Returns:
        --------
        ee.Image : Calculated and reduced spectral indices, or None if calculation fails
        
        Example:
        --------
        >>> calc = SpectralCalculator()
        >>> collection = ee.ImageCollection("COPERNICUS/S2_SR").filterDate("2023-01-01", "2023-12-31")
        >>> aoi = ee.Geometry.Rectangle([106.0, -6.5, 107.0, -6.0])
        >>> 
        >>> # Using default coefficients
        >>> indices = calc.calculate_indices_with_collection(
        ...     collection=collection,
        ...     aoi=aoi,
        ...     index_list=["NDVI", "EVI"],
        ...     reducer_method="median"
        ... )
        >>> 
        >>> # Using custom coefficients
        >>> custom_coeffs = {"EVI": {"g": 3.0, "C1": 6.0, "C2": 7.5, "L": 1.0}}
        >>> indices = calc.calculate_indices_with_collection(
        ...     collection=collection,
        ...     aoi=aoi,
        ...     index_list=["EVI"],
        ...     coefficients=custom_coeffs,
        ...     reducer_method="mean"
        ... )
        """
        try:
            logger.info(f"Calculating spectral indices using map and {reducer_method} reducer on collection")
            
            # Validate reducer method
            valid_reducers = ["mean", "median", "max", "min"]
            if reducer_method not in valid_reducers:
                logger.warning(f"Unsupported reducer method: {reducer_method}, using mean")
                reducer_method = "mean"
            
            # Prepare base parameters for spyndex
            base_params = {}
            
            # Add coefficients to params if provided
            if coefficients:
                logger.info(f"Using custom coefficients for indices: {list(coefficients.keys())}")
                for index_name, index_coeffs in coefficients.items():
                    if index_name in index_list:
                        # Validate coefficients
                        validation = self.validate_coefficients(index_name, index_coeffs)
                        if not validation["valid"]:
                            logger.error(f"Invalid coefficients for {index_name}: {validation['warnings']}")
                            raise ValueError(f"Invalid coefficients for {index_name}: {validation['warnings']}")
                        
                        if validation["warnings"]:
                            logger.warning(f"Coefficient warnings for {index_name}: {validation['warnings']}")
                        
                        # Add coefficients to base params
                        base_params.update(index_coeffs)
                        logger.info(f"Added coefficients for {index_name}: {index_coeffs}")
            else:
                # Use default coefficients for indices that require them
                logger.info("Using default coefficients where required")
                for index_name in index_list:
                    default_coeffs = self.get_index_coefficients(index_name)
                    if default_coeffs:
                        base_params.update(default_coeffs)
                        logger.info(f"Added default coefficients for {index_name}: {default_coeffs}")
            
            # Step 1: Map the index calculation function over each image in the collection. Define as server side computation
            logger.info(f"Mapping index calculation over collection...")
            
            def calculate_indices_for_image(image):
                """
                Calculate spectral indices for a single image using spyndex.
                
                This function is mapped over each image in the collection.
                Since spyndex.computeIndex returns an ee.Image, this works within
                Earth Engine's server-side execution model.
                """
                # Prepare params with band selections for this image
                params = {
                    "N": image.select("NIR"),
                    "R": image.select("RED"),
                    "G": image.select("GREEN"),
                    "B": image.select("BLUE"),
                    "S1": image.select("SWIR1"),
                    "S2": image.select("SWIR2")
                }
                
                # Add coefficients to params
                params.update(base_params)
                
                # Use spyndex to compute indices for this image. spyndex.computeIndex returns an ee.Image that can be used in map
                spectral_indices = self.spyndex.computeIndex(
                    index=index_list,
                    params=params
                )
                return spectral_indices
            
            # Apply the function to each image in the collection
            indices_collection = collection.map(calculate_indices_for_image)
            
            # Step 2: Apply the specified reducer to the collection
            logger.info(f"Applying {reducer_method} reducer to collection...")
            if reducer_method == "mean":
                reduced_indices = indices_collection.mean()
            elif reducer_method == "median":
                reduced_indices = indices_collection.median()
            elif reducer_method == "max":
                reduced_indices = indices_collection.max()
            elif reducer_method == "min":
                reduced_indices = indices_collection.min()
            
            # Step 3: Clip to AOI
            reduced_indices = reduced_indices.clip(aoi) # type: ignore
            
            # Validate results
            if reduced_indices is not None:
                band_count = reduced_indices.bandNames().size().getInfo()
                if band_count > 0:
                    logger.info(f"Successfully calculated {band_count} spectral indices using map and {reducer_method} reducer")
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
        
        Returns:
        --------
        Dict[str, Any] : Summary information
        """
        return {
            'indices_calculated': self.calculated_count,
            'supported_indices': len(self.supported_indices),
            'categories': list(set(self.supported_indices.values()))
        }

#=================================== Distance Metrics ===================================
class distance_calculator:
    """
    This class computes distance-based predictors that measure proximity to various
    geographic features like roads, coastlines, and settlements.
    Supported Distance Metrics:
    - Roads: Distance to road networks 
    - Coastline: Distance to coastline features
    - Settlements: Distance to populated areas
    Data Sources:
    - Natural Earth Coastline: https://www.naturalearthdata.com/
    - OpenStreetMap (OSM) Roads: Community-contributed road data
    - RBI Roads: Regional road network data
    - HRSL Settlements: High Resolution Settlement Layer

    Usage Example:
        calc = distance_calculator()
        
        # Calculate all distance metrics
        distances = calc.calculate_distance_metrics(
            aoi=aoi_geometry,
            max_dist=50000,  # 50km maximum distance
            in_meters=True   # Output in meters
        )
        
        # Result contains: dist_roads, dist_coast, dist_settlement bands
    """
    
    def __init__(self):
        """
        Initialize the distance calculator.
        
        Ensures Earth Engine is initialized and sets up logging.
        No heavy computations are performed during initialization.
        """
        ensure_ee_initialized()
        logger.info("Distance calculator initialized")
    
    def calculate_distance_metrics(self, 
                                   aoi: ee.Geometry, 
                                   max_dist: float = 500000, 
                                   in_meters: bool = False) -> ee.Image:
        """
        Create distance images based on predefined datasets.
        
        This method calculates cumulative cost distance from various geographic
        features to create spatial predictors for land cover classification using Earth Engine's cumulativeCost().
        Distance metrics help capture accessibility patterns and spatial
        relationships that influence land use decisions.
        
        Parameters:
        -----------
        aoi : ee.Geometry
            Area of interest geometry for clipping results
        max_dist : float, default=500000
            Maximum cumulative cost distance in meters or pixels.
            Larger values increase computation time but provide more complete coverage.
        in_meters : bool, default=False
            Distance unit selection:
            - True: Output distance in meters (more accurate, slower computation)
            - False: Output distance in pixels (faster, less precise)
            
        Returns:
        --------
        ee.Image or None
            Stacked distance metrics with bands:
            - 'dist_roads': Distance to nearest road network
            - 'dist_coast': Distance to nearest coastline
            - 'dist_settlement': Distance to nearest settlement
            Returns None if calculation fails completely.
            
        Example:
        --------
        >>> calc = distance_calculator()
        >>> aoi = ee.Geometry.Rectangle([106.0, -6.5, 107.0, -6.0])
        >>> 
        >>> # Fast pixel-based calculation
        >>> distances_px = calc.calculate_distance_metrics(aoi, max_dist=100000, in_meters=False)
        >>> 
        >>> # Accurate meter-based calculation  
        >>> distances_m = calc.calculate_distance_metrics(aoi, max_dist=50000, in_meters=True)
        
        """
        try:
            logger.info(f"Calculating distance metrics (max_dist={max_dist}, in_meters={in_meters})...")
            
            # Set up cost image based on distance unit preference
            if in_meters:
                # Distance in meters: Use pixel area square root as cost
                # This accounts for varying pixel sizes due to map projection
                cost_image = ee.Image.pixelArea().sqrt()
                logger.info("Using meter-based distance calculation (slower but more accurate)")
            else:
                # Distance in pixel units: Uniform cost of 1 per pixel
                # Faster computation but less accurate for large areas
                cost_image = ee.Image(1)
                logger.info("Using pixel-based distance calculation (faster but less precise)")
            
            # Helper function to calculate distance from source mask
            def distance_image(source_mask):
                """
                Calculate cumulative cost distance from binary source mask.
                
                Parameters:
                -----------
                source_mask : ee.Image
                    Binary mask where value 1 indicates source locations (starting points)
                    
                Returns:
                --------
                ee.Image : Distance image with cumulative cost values
                """
                return cost_image.cumulativeCost(source_mask, max_dist)
            
            # 1. Natural Earth Coastline Data Processing
            # Source: https://www.naturalearthdata.com/downloads/10m-physical-vectors/10m-coastline/
            logger.info("Processing coastline data...")
            try:
                ne_coastline = ee.FeatureCollection(
                    "users/hadicu06/IIASA/RESTORE/vector_datasets/coastline_ne_10m"
                ).filterBounds(aoi)
                
                # Convert vector features to raster mask
                # Each feature gets a constant value of 1, then reduce to image
                coast_dist = ne_coastline.map(lambda ft: ft.set('constant', 1)) \
                                        .reduceToImage(['constant'], ee.Reducer.first()) \
                                        .mask()  # Create binary mask (1 where coastline exists)
                
            except Exception as e:
                logger.warning(f"Failed to load coastline data: {str(e)}")
                # Create empty mask if coastline data unavailable
                coast_dist = ee.Image(0).mask(ee.Image(0))
            
            # 2. Roads Data Processing (OSM + RBI)
            # Combines OpenStreetMap and Rupa Bumi Indonesia (RBI) road networks
            logger.info("Processing roads data...")
            try:
                #OpenStreetMap roads (use legacy asset from RESTORE+, update might be needed)
                road_osm = ee.FeatureCollection(
                    "users/hadicu06/IIASA/RESTORE/vector_datasets/road_osm"
                ).filterBounds(aoi)
                
                #RBI roads (use legacy asset from RESTORE+, update might be needed)
                road_rbi = ee.FeatureCollection(
                    "users/hadicu06/IIASA/RESTORE/vector_datasets/road_rbi"
                ).filterBounds(aoi)
                
                # Merge road datasets and convert to raster mask
                roads_dist = road_osm.merge(road_rbi) \
                                    .map(lambda ft: ft.set('constant', 1)) \
                                    .reduceToImage(['constant'], ee.Reducer.max()) \
                                    .mask()  # Use max reducer to handle overlapping roads
                
            except Exception as e:
                logger.warning(f"Failed to load roads data: {str(e)}")
                # Create empty mask if roads data unavailable
                roads_dist = ee.Image(0).mask(ee.Image(0))
            
            # 3. Settlement Data Processing (HRSL)
            # Source: Facebook High Resolution Settlement Layer
            # https://dataforgood.facebook.com/dfg/tools/high-resolution-population-density-maps
            #calculation is based on RESTORE+ codebase
            logger.info("Processing settlement data...")
            try:
                # Load HRSL population density data
                hrsl = ee.ImageCollection("projects/sat-io/open-datasets/hrsl/hrslpop") \
                        .mosaic() \
                        .clip(aoi)
                
                # Apply connected component analysis to identify settlement clusters
                # This groups adjacent populated pixels into coherent settlements
                hrsl_connected = hrsl.int().connectedPixelCount(maxSize=100, eightConnected=True)
                
                # Create settlement mask: areas with >3 connected populated pixels
                # This filters out isolated pixels and focuses on actual settlements
                hrsl_masked = hrsl_connected.unmask().gt(3)
                
            except Exception as e:
                logger.warning(f"Failed to load settlement data: {str(e)}")
                # Create empty mask if settlement data unavailable
                hrsl_masked = ee.Image(0).mask(ee.Image(0))
            
            # 4. Calculate Distance Transformations
            logger.info("Computing distance transformations...")
            
            # Calculate cumulative cost distance for each feature type
            dist_roads = distance_image(roads_dist).rename('dist_roads')
            dist_coast = distance_image(coast_dist).rename('dist_coast')
            dist_settlement = distance_image(hrsl_masked).rename('dist_settlement')
            
            # Stack all distance metrics into single multi-band image
            distance_metrics = ee.Image.cat(dist_roads, dist_coast, dist_settlement)
            
            logger.info("Successfully calculated distance metrics: dist_roads, dist_coast, dist_settlement")
            return distance_metrics
            
        except Exception as e:
            logger.error(f"Failed to calculate distance metrics: {str(e)}")
            logger.error("This may be due to:")
            logger.error("- Missing or inaccessible datasets")
            logger.error("- Earth Engine quota limits")
            logger.error("- Network connectivity issues")
            logger.error("- AOI outside dataset coverage area")
            return None

#=================================== Predictor Calculation ===================================
class PredictorCalculation:
    """
    Main orchestration service for predictor computation workflows.
    This class coordinates between terrain_calculator, SpectralCalculator, 
    and distance_calculator to provide a unified interface for predictor computation.
    """
    
    def __init__(self):
        """Initialize the enhanced predictor service with all calculator backends."""
        ensure_ee_initialized()
        
        # Initialize calculator backends
        self.terrain_calc = terrain_calculator()
        self.spectral_calc = SpectralCalculator()
        
        logger.info("Predictor Calculation initialized with terrain and spectral calculators")
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
    #core function: Compute Predictor
    def compute_predictors(self, composite: ee.Image,aoi: ee.Geometry,predictor_config: Dict[str, Any],collection: ee.ImageCollection = None,
                          progress_callback: callable = None) -> Dict[str, Any]:
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
                            reducer_method="mean"
                        )
                    else:
                        # Use composite directly
                        spectral_result = self.spectral_calc.calculate_indices(
                            image=composite,
                            index_list=spectral_indices,
                            coefficients=index_coefficients
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

#=================================== Recursive Feature Elimination ===================================
class RFEGEE:
    """
    Recursive Feature Elimination (RFE) using
    Random Forest variable importance in Google Earth Engine.
    Inspired by https://github.com/DanGeospatial/GEE_Wetland_Classification
    """

    def __init__(
        self,
        training_data: ee.FeatureCollection,
        class_property: str,
        features: List[str],
        test_data: Optional[ee.FeatureCollection] = None,
        n_features_to_select: Optional[int] = None,
        step: int = 1,
        n_trees: int = 50,
        seed: int = 42,
        track_accuracy: bool = False,
        verbose: bool = True
    ):
        self.training_data = training_data
        self.test_data = test_data
        self.class_property = class_property
        self.features = features.copy()
        self.n_features_to_select = (
            n_features_to_select or len(features) // 2
        )
        self.step = step
        self.n_trees = n_trees
        self.seed = seed
        self.track_accuracy = track_accuracy
        self.verbose = verbose

        # Results
        self.selected_features_ = None
        self.elimination_order_ = []
        self.accuracy_history_ = []
        self.n_features_history_ = []
        self.feature_importance_history_ = []

        self._validate_inputs()

    #Utilities
    def _log(self, msg: str):
        if self.verbose:
            print(msg)
    #Function to validate input, log error if something happen
    def _validate_inputs(self):
        if not isinstance(self.training_data, ee.FeatureCollection):
            raise TypeError("training_data must be ee.FeatureCollection")

        if self.test_data is not None and not isinstance(self.test_data, ee.FeatureCollection):
            raise TypeError("test_data must be ee.FeatureCollection or None")

        if self.track_accuracy and self.test_data is None:
            raise ValueError("test_data is required when track_accuracy=True")

        if self.n_features_to_select < 1:
            raise ValueError("n_features_to_select must be >= 1")

        if self.n_features_to_select > len(self.features):
            raise ValueError("n_features_to_select exceeds feature count")

        if self.step < 1:
            raise ValueError("step must be >= 1")


    #Pre-requisites for implementation
    #1. Train classifier (RF)
    def _train_classifier(self, feature_list: List[str]) -> ee.Classifier:
        return ee.Classifier.smileRandomForest(
            numberOfTrees=self.n_trees,
            seed=self.seed
        ).train(
            features=self.training_data,
            classProperty=self.class_property,
            inputProperties=feature_list
        )

    #2. Get variable importance AND optional accuracy in a single getInfo() call
    def _get_importance_and_accuracy(
        self, classifier: ee.Classifier, feature_list: List[str]
    ) -> Tuple[Dict[str, float], Optional[float]]:
        """
        Retrieve feature importance (and optionally test accuracy) in a
        single round-trip to the Earth Engine servers.

        Batching both server-side objects into one getInfo() call avoids
        the cost of a second blocking HTTP request per RFE iteration.
        """
        try:
            if self.track_accuracy and self.test_data is not None:
                # Build both EE objects server-side, then fetch in one call.
                accuracy_val = (
                    self.test_data
                    .classify(classifier)
                    .errorMatrix(self.class_property, "classification")
                    .accuracy()
                )
                result = ee.Dictionary({
                    "explanation": classifier.explain(),
                    "accuracy": accuracy_val,
                }).getInfo()

                explanation = result["explanation"]
                accuracy = result["accuracy"]
            else:
                explanation = classifier.explain().getInfo()
                accuracy = None

            if "importance" not in explanation:
                raise ValueError("Feature importance not available in model explanation")

            return explanation["importance"], accuracy

        except Exception as e:
            raise RuntimeError(f"RF importance/accuracy extraction failed: {str(e)}")

    def _adaptive_step(self, n_current: int) -> int:
        """
        Return the number of features to drop this iteration.

        Uses a larger step when far from the target (to reduce total
        iterations and therefore EE API calls) and falls back to the
        user-configured ``self.step`` once within 2× of the target.

        This is a conservative heuristic: it never drops more than half
        the surplus in one go, so the importance ordering sampled at a
        large-step iteration is still representative.
        """
        surplus = n_current - self.n_features_to_select
        # Within the final stretch — use the configured fine-grained step.
        if surplus <= 2 * self.step:
            return min(self.step, surplus)
        # Coarse phase: drop up to half the surplus per iteration, but
        # always respect the user-configured step as a lower bound so
        # that if step=1 is explicitly requested we honour it.
        coarse = max(self.step, surplus // 2)
        return min(coarse, surplus)

    #Implement Recursive Feature Elimination
    #Remove feature one by one, evaluate its accuracy after each removal
    #backward selection by starting with all of the input features 
    #iteratively remove each feature that did not contribute to the increase of accuracy
    def fit(self):
        current_features = self.features.copy()

        self._log(f"Starting RFE with {len(current_features)} features")
        self._log(f"Target: {self.n_features_to_select}")
        self._log("-" * 50)

        while len(current_features) > self.n_features_to_select:

            classifier = self._train_classifier(current_features)

            # Single getInfo() call retrieves importance + accuracy together
            importance, acc = self._get_importance_and_accuracy(classifier, current_features)

            # Store importance history for visualization
            self.feature_importance_history_.append(importance.copy())

            if self.track_accuracy and acc is not None:
                self.accuracy_history_.append(acc)
                self.n_features_history_.append(len(current_features))
                self._log(f"Features: {len(current_features)} | Acc: {acc:.4f}")
            else:
                self._log(f"Features: {len(current_features)}")

            # Sort features by ascending importance
            ranked = sorted(importance.items(), key=lambda x: x[1])

            n_remove = self._adaptive_step(len(current_features))
            to_remove = [f for f, _ in ranked[:n_remove]]

            self._log(f"Removing: {to_remove}")

            for f in to_remove:
                current_features.remove(f)
                self.elimination_order_.append(f)

        self.selected_features_ = current_features
        self._log("-" * 50)
        self._log(f"Selected features ({len(current_features)}):")
        for f in current_features:
            self._log(f"  - {f}")

        return self
    # ------------------------------------------------------------------
    #Get the selected features
    def get_selected_features(self) -> List[str]:
        if self.selected_features_ is None:
            raise RuntimeError("Call fit() first")
        return self.selected_features_.copy()
    #Rank the result 
    def get_ranking(self) -> Dict[str, int]:
        """
        Rank 1 = selected features
        Higher rank = eliminated earlier
        """
        if self.selected_features_ is None:
            raise RuntimeError("Call fit() first")

        ranking = {}
        for f in self.features:
            if f in self.selected_features_:
                ranking[f] = 1
            else:
                ranking[f] = self.elimination_order_.index(f) + 2

        return ranking

    def get_support(self) -> Dict[str, bool]:
        if self.selected_features_ is None:
            raise RuntimeError("Call fit() first")

        return {f: f in self.selected_features_ for f in self.features}

    # ------------------------------------------------------------------
    # Visualization methods for Streamlit
    # ------------------------------------------------------------------
    def plot_accuracy_curve(self, figsize: Tuple[int, int] = (10, 6)):
        """
        Plot accuracy vs number of features curve.
        Returns matplotlib figure for Streamlit integration.
        """
        if not self.accuracy_history_:
            raise RuntimeError("No accuracy data. Set track_accuracy=True and call fit() first")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(
            self.n_features_history_, 
            self.accuracy_history_, 
            marker='o',
            linewidth=2,
            markersize=6,
            color='#1f77b4'
        )
        
        # Highlight selected number of features
        ax.axvline(
            x=self.n_features_to_select,
            color='red',
            linestyle='--',
            linewidth=2,
            label=f'Selected: {self.n_features_to_select} features'
        )
        
        ax.set_xlabel('Number of Features', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('RFE: Accuracy vs Number of Features', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        return fig

    def plot_feature_rankings(self, figsize: Tuple[int, int] = (10, 8)):
        """
        Plot feature rankings as horizontal bar chart.
        Returns matplotlib figure for Streamlit integration.
        """
        if self.selected_features_ is None:
            raise RuntimeError("Call fit() first")
        
        ranking = self.get_ranking()
        ranking_sorted = sorted(ranking.items(), key=lambda x: x[1])
        features, ranks = zip(*ranking_sorted)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Color selected features differently
        colors = ['#2ca02c' if r == 1 else '#d62728' for r in ranks]
        
        bars = ax.barh(range(len(features)), ranks, color=colors, alpha=0.7)
        
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features, fontsize=10)
        ax.set_xlabel('Rank (1 = Selected)', fontsize=12)
        ax.set_title('Feature Rankings', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#2ca02c', alpha=0.7, label='Selected'),
            Patch(facecolor='#d62728', alpha=0.7, label='Eliminated')
        ]
        ax.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        return fig

    def plot_rfe_summary(self, figsize: Tuple[int, int] = (15, 6)):
        """
        Combined plot showing both accuracy curve and feature rankings.
        Returns matplotlib figure for Streamlit integration.
        """
        if self.selected_features_ is None:
            raise RuntimeError("Call fit() first")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot 1: Accuracy curve (if available)
        if self.accuracy_history_:
            ax1.plot(
                self.n_features_history_, 
                self.accuracy_history_, 
                marker='o',
                linewidth=2,
                markersize=6,
                color='#1f77b4'
            )
            ax1.axvline(
                x=self.n_features_to_select,
                color='red',
                linestyle='--',
                linewidth=2,
                label=f'Selected: {self.n_features_to_select} features'
            )
            ax1.set_xlabel('Number of Features', fontsize=12)
            ax1.set_ylabel('Accuracy', fontsize=12)
            ax1.set_title('Accuracy vs Number of Features', fontsize=14)
            ax1.grid(True, alpha=0.3)
            ax1.legend()
        else:
            ax1.text(0.5, 0.5, 'No accuracy data\n(set track_accuracy=True)', 
                    ha='center', va='center', transform=ax1.transAxes, fontsize=12)
            ax1.set_title('Accuracy Tracking Disabled', fontsize=14)
        
        # Plot 2: Feature rankings
        ranking = self.get_ranking()
        ranking_sorted = sorted(ranking.items(), key=lambda x: x[1])
        features, ranks = zip(*ranking_sorted)
        
        colors = ['#2ca02c' if r == 1 else '#d62728' for r in ranks]
        ax2.barh(range(len(features)), ranks, color=colors, alpha=0.7)
        ax2.set_yticks(range(len(features)))
        ax2.set_yticklabels(features, fontsize=9)
        ax2.set_xlabel('Rank', fontsize=12)
        ax2.set_title('Feature Rankings', fontsize=14)
        ax2.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        return fig

    def get_results_dataframe(self) -> pd.DataFrame:
        """
        Get comprehensive results as pandas DataFrame for Streamlit display.
        """
        if self.selected_features_ is None:
            raise RuntimeError("Call fit() first")
        
        ranking = self.get_ranking()
        support = self.get_support()
        
        # Create base DataFrame
        results_df = pd.DataFrame({
            'Feature': list(ranking.keys()),
            'Rank': list(ranking.values()),
            'Selected': list(support.values()),
            'Status': ['✅ Selected' if s else '❌ Eliminated' for s in support.values()]
        })
        
        # Add final importance scores if available
        if self.feature_importance_history_:
            final_importance = self.feature_importance_history_[-1]
            results_df['Final_Importance'] = results_df['Feature'].map(
                lambda x: final_importance.get(x, 0)
            )
        
        # Sort by rank
        results_df = results_df.sort_values('Rank').reset_index(drop=True)
        
        return results_df

    def get_accuracy_dataframe(self) -> pd.DataFrame:
        """
        Get accuracy history as pandas DataFrame for Streamlit display.
        """
        if not self.accuracy_history_:
            raise RuntimeError("No accuracy data. Set track_accuracy=True and call fit() first")
        
        return pd.DataFrame({
            'Iteration': range(1, len(self.accuracy_history_) + 1),
            'Number_of_Features': self.n_features_history_,
            'Accuracy': self.accuracy_history_
        })


#=================================== Recursive Feature Elimination ===================================
class RFEGEE:
    """
    Recursive Feature Elimination (RFE) using
    Random Forest variable importance in Google Earth Engine.
    Inspired by https://github.com/DanGeospatial/GEE_Wetland_Classification
    """

    def __init__(
        self,
        training_data: ee.FeatureCollection,
        class_property: str,
        features: List[str],
        test_data: Optional[ee.FeatureCollection] = None,
        n_features_to_select: Optional[int] = None,
        step: int = 1,
        n_trees: int = 50,
        seed: int = 42,
        track_accuracy: bool = False,
        verbose: bool = True
    ):
        self.training_data = training_data
        self.test_data = test_data
        self.class_property = class_property
        self.features = features.copy()
        self.n_features_to_select = (
            n_features_to_select or len(features) // 2
        )
        self.step = step
        self.n_trees = n_trees
        self.seed = seed
        self.track_accuracy = track_accuracy
        self.verbose = verbose

        # Results
        self.selected_features_ = None
        self.elimination_order_ = []
        self.accuracy_history_ = []
        self.n_features_history_ = []
        self.feature_importance_history_ = []

        self._validate_inputs()

    #Utilities
    def _log(self, msg: str):
        if self.verbose:
            print(msg)
    #Function to validate input, log error if something happen
    def _validate_inputs(self):
        if not isinstance(self.training_data, ee.FeatureCollection):
            raise TypeError("training_data must be ee.FeatureCollection")

        if self.test_data is not None and not isinstance(self.test_data, ee.FeatureCollection):
            raise TypeError("test_data must be ee.FeatureCollection or None")

        if self.track_accuracy and self.test_data is None:
            raise ValueError("test_data is required when track_accuracy=True")

        if self.n_features_to_select < 1:
            raise ValueError("n_features_to_select must be >= 1")

        if self.n_features_to_select > len(self.features):
            raise ValueError("n_features_to_select exceeds feature count")

        if self.step < 1:
            raise ValueError("step must be >= 1")
    #Pre-requisites for implementation
    #1. Train classifier (RF)
    def _train_classifier(self, feature_list: List[str]) -> ee.Classifier:
        return ee.Classifier.smileRandomForest(
            numberOfTrees=self.n_trees,
            seed=self.seed
        ).train(
            features=self.training_data,
            classProperty=self.class_property,
            inputProperties=feature_list
        )

    #2. Get variable importance AND optional accuracy in a single getInfo() call
    def _get_importance_and_accuracy(
        self, classifier: ee.Classifier, feature_list: List[str]
    ) -> Tuple[Dict[str, float], Optional[float]]:
        """
        Retrieve feature importance (and optionally test accuracy) in a
        single round-trip to the Earth Engine servers.

        Batching both server-side objects into one getInfo() call avoids
        the cost of a second blocking HTTP request per RFE iteration.
        """
        try:
            if self.track_accuracy and self.test_data is not None:
                # Build both EE objects server-side, then fetch in one call.
                accuracy_val = (
                    self.test_data
                    .classify(classifier)
                    .errorMatrix(self.class_property, "classification")
                    .accuracy()
                )
                result = ee.Dictionary({
                    "explanation": classifier.explain(),
                    "accuracy": accuracy_val,
                }).getInfo()

                explanation = result["explanation"]
                accuracy = result["accuracy"]
            else:
                explanation = classifier.explain().getInfo()
                accuracy = None

            if "importance" not in explanation:
                raise ValueError("Feature importance not available in model explanation")

            return explanation["importance"], accuracy

        except Exception as e:
            raise RuntimeError(f"RF importance/accuracy extraction failed: {str(e)}")
    #Fallback mechanism
    def _adaptive_step(self, n_current: int) -> int:
        """
        Return the number of features to drop this iteration.

        Uses a larger step when far from the target (to reduce total
        iterations and therefore EE API calls) and falls back to the
        user-configured ``self.step`` once within 2× of the target.

        This is a conservative heuristic: it never drops more than half
        the surplus in one go, so the importance ordering sampled at a
        large-step iteration is still representative.
        """
        surplus = n_current - self.n_features_to_select
        # Within the final stretch — use the configured fine-grained step.
        if surplus <= 2 * self.step:
            return min(self.step, surplus)
        # Coarse phase: drop up to half the surplus per iteration, but
        # always respect the user-configured step as a lower bound so
        # that if step=1 is explicitly requested we honour it.
        coarse = max(self.step, surplus // 2)
        return min(coarse, surplus)

    #3. Implement Recursive Feature Elimination
    #Remove feature one by one, evaluate its accuracy after each removal
    #backward selection by starting with all of the input features 
    #iteratively remove each feature that did not contribute to the increase of accuracy
    def fit(self):
        current_features = self.features.copy()
        self._log(f"Starting RFE with {len(current_features)} features")
        self._log(f"Target: {self.n_features_to_select}")
        self._log("-" * 50)
        while len(current_features) > self.n_features_to_select:
            classifier = self._train_classifier(current_features)
            #Single getInfo() call retrieves importance + accuracy together
            #optimize API calls 
            importance, acc = self._get_importance_and_accuracy(classifier, current_features)

            # Store importance history for visualization
            self.feature_importance_history_.append(importance.copy())

            if self.track_accuracy and acc is not None:
                self.accuracy_history_.append(acc)
                self.n_features_history_.append(len(current_features))
                self._log(f"Features: {len(current_features)} | Acc: {acc:.4f}")
            else:
                self._log(f"Features: {len(current_features)}")

            # Sort features by ascending importance
            ranked = sorted(importance.items(), key=lambda x: x[1])

            n_remove = self._adaptive_step(len(current_features))
            to_remove = [f for f, _ in ranked[:n_remove]]

            self._log(f"Removing: {to_remove}")

            for f in to_remove:
                current_features.remove(f)
                self.elimination_order_.append(f)

        self.selected_features_ = current_features
        self._log("-" * 50)
        self._log(f"Selected features ({len(current_features)}):")
        for f in current_features:
            self._log(f"  - {f}")

        return self
    #Get the selected features
    def get_selected_features(self) -> List[str]:
        if self.selected_features_ is None:
            raise RuntimeError("Call fit() first")
        return self.selected_features_.copy()
    #Rank the result 
    def get_ranking(self) -> Dict[str, int]:
        """
        Rank 1 = selected features
        Higher rank = eliminated earlier
        """
        if self.selected_features_ is None:
            raise RuntimeError("Call fit() first")

        ranking = {}
        for f in self.features:
            if f in self.selected_features_:
                ranking[f] = 1
            else:
                ranking[f] = self.elimination_order_.index(f) + 2

        return ranking

    def get_support(self) -> Dict[str, bool]:
        if self.selected_features_ is None:
            raise RuntimeError("Call fit() first")

        return {f: f in self.selected_features_ for f in self.features}

    #Visualization
    #accuracy curve
    def plot_accuracy_curve(self, figsize: Tuple[int, int] = (10, 6)):
        """
        Plot accuracy vs number of features curve.
        Returns matplotlib figure for Streamlit integration.
        """
        if not self.accuracy_history_:
            raise RuntimeError("No accuracy data. Set track_accuracy=True and call fit() first")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(
            self.n_features_history_, 
            self.accuracy_history_, 
            marker='o',
            linewidth=2,
            markersize=6,
            color='#1f77b4'
        )
        #Highlight selected number of features
        ax.axvline(
            x=self.n_features_to_select,
            color='red',
            linestyle='--',
            linewidth=2,
            label=f'Selected: {self.n_features_to_select} features'
        )
        
        ax.set_xlabel('Number of Features', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('RFE: Accuracy vs Number of Features', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        return fig
    #visualization for feature ranking
    def plot_feature_rankings(self, figsize: Tuple[int, int] = (10, 8)):
        """
        Plot feature rankings as horizontal bar chart.
        Returns matplotlib figure for Streamlit integration.
        """
        if self.selected_features_ is None:
            raise RuntimeError("Call fit() first")
        
        ranking = self.get_ranking()
        ranking_sorted = sorted(ranking.items(), key=lambda x: x[1])
        features, ranks = zip(*ranking_sorted)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Color selected features differently
        colors = ['#2ca02c' if r == 1 else '#d62728' for r in ranks]
        
        bars = ax.barh(range(len(features)), ranks, color=colors, alpha=0.7)
        
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features, fontsize=10)
        ax.set_xlabel('Rank (1 = Selected)', fontsize=12)
        ax.set_title('Feature Rankings', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#2ca02c', alpha=0.7, label='Selected'),
            Patch(facecolor='#d62728', alpha=0.7, label='Eliminated')
        ]
        ax.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        return fig

    def plot_rfe_summary(self, figsize: Tuple[int, int] = (15, 6)):
        """
        Combined plot showing both accuracy curve and feature rankings.
        Returns matplotlib figure for Streamlit integration.
        """
        if self.selected_features_ is None:
            raise RuntimeError("Call fit() first")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot 1: Accuracy curve (if available)
        if self.accuracy_history_:
            ax1.plot(
                self.n_features_history_, 
                self.accuracy_history_, 
                marker='o',
                linewidth=2,
                markersize=6,
                color='#1f77b4'
            )
            ax1.axvline(
                x=self.n_features_to_select,
                color='red',
                linestyle='--',
                linewidth=2,
                label=f'Selected: {self.n_features_to_select} features'
            )
            ax1.set_xlabel('Number of Features', fontsize=12)
            ax1.set_ylabel('Accuracy', fontsize=12)
            ax1.set_title('Accuracy vs Number of Features', fontsize=14)
            ax1.grid(True, alpha=0.3)
            ax1.legend()
        else:
            ax1.text(0.5, 0.5, 'No accuracy data\n(set track_accuracy=True)', 
                    ha='center', va='center', transform=ax1.transAxes, fontsize=12)
            ax1.set_title('Accuracy Tracking Disabled', fontsize=14)
        
        # Plot 2: Feature rankings
        ranking = self.get_ranking()
        ranking_sorted = sorted(ranking.items(), key=lambda x: x[1])
        features, ranks = zip(*ranking_sorted)
        
        colors = ['#2ca02c' if r == 1 else '#d62728' for r in ranks]
        ax2.barh(range(len(features)), ranks, color=colors, alpha=0.7)
        ax2.set_yticks(range(len(features)))
        ax2.set_yticklabels(features, fontsize=9)
        ax2.set_xlabel('Rank', fontsize=12)
        ax2.set_title('Feature Rankings', fontsize=14)
        ax2.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        return fig

    def get_results_dataframe(self) -> pd.DataFrame:
        """
        Get comprehensive results as pandas DataFrame for Streamlit display.
        """
        if self.selected_features_ is None:
            raise RuntimeError("Call fit() first")
        
        ranking = self.get_ranking()
        support = self.get_support()
        
        # Create base DataFrame
        results_df = pd.DataFrame({
            'Feature': list(ranking.keys()),
            'Rank': list(ranking.values()),
            'Selected': list(support.values()),
            'Status': ['✅ Selected' if s else '❌ Eliminated' for s in support.values()]
        })
        
        # Add final importance scores if available
        if self.feature_importance_history_:
            final_importance = self.feature_importance_history_[-1]
            results_df['Final_Importance'] = results_df['Feature'].map(
                lambda x: final_importance.get(x, 0)
            )
        
        # Sort by rank
        results_df = results_df.sort_values('Rank').reset_index(drop=True)
        
        return results_df

    def get_accuracy_dataframe(self) -> pd.DataFrame:
        """
        Get accuracy history as pandas DataFrame for Streamlit display.
        """
        if not self.accuracy_history_:
            raise RuntimeError("No accuracy data. Set track_accuracy=True and call fit() first")
        
        return pd.DataFrame({
            'Iteration': range(1, len(self.accuracy_history_) + 1),
            'Number_of_Features': self.n_features_history_,
            'Accuracy': self.accuracy_history_
        })
