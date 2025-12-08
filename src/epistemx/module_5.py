import ee
import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
from .ee_config import ensure_ee_initialized
# Do not initialize Earth Engine at import time. Initialize when classes are instantiated.
#=================================== THIS CODE IS USED FOR PHASE 2!!!!!!!!!
#Configure logging globally
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

class indexcategory(Enum):
    """
    categories of avaliable spectal indices
    """
    vegetation = 'vegetation'
    moisture = 'moisture'
    burn = 'burn'
    water = 'water'
    soil_build_up = 'soil_buildup'
    tascap = 'tascap'

@dataclass
class indexdef:
    """
    Defining the spectral transformation
    """
    name: str
    formula: str
    description: str
    category: indexcategory
    reference: Optional[str] = None
    required_bands : Optional[List[str]] = None

class spectral_transformation_calcultator:
    """
    Calculate spectral transformation for creating an input covariates in land cover mapping
    """
    def __init__(self):
        """
        Initialize the spectral transformation calculator
        Ensure Earth Engine is initialized lazily (avoids import-time failures).
        """
        # Ensure Earth Engine is initialized when first used (raises helpful error if not)
        ensure_ee_initialized()
        
        self.avaliable_indices = self.initialize_index()
        self.calculated_count = 0
    #Initialize the supported index. The list here is tailored with landsat mission
    def initialize_index(self) -> Dict[str, indexdef]:
        """
        Initialize all avaliable spectral transformation definitions
        """
        indices = {
            #Vegetation Indices
            'NDVI': indexdef(
                'NDVI',  #Name
                'normalizedDifference(NIR, RED)',   #Functions in GEE
                'Normalized Difference Vegetation Index',   #Index name
                indexcategory.vegetation, #append to the category
                reference = 'https://doi.org/10.1016/0034-4257(79)90013-0', #Reference (optional)
                required_bands=['NIR', 'RED'] #Bands to used
            ),
            'GNDVI': indexdef(
                'GNDVI', 
                'normalizedDifference(NIR, GREEN)',
                'Green Normalized Difference Vegetation Index',
                indexcategory.vegetation,
                'https://doi.org/10.1016/S0034-4257(96)00072-7 ',
                required_bands = ['NIR', 'GREEN']
            ),
            'MSAVI': indexdef(
                'MSAVI', 
                '(2 * NIR + 1 - sqrt(pow(2 * NIR + 1, 2) - 8 * (NIR - RED))) / 2', #Index Formula
                'Modified Soil Adjusted Vegetation Index',
                indexcategory.vegetation,
                'https://doi.org/10.1016/0034-4257(94)90134-1',
                required_bands = ['NIR', 'RED']
            ),
            'OSAVI': indexdef(
                'OSAVI', 
                '(NIR - RED) / (NIR + RED + 0.16)',
                'Optimized Soil Adjusted Vegetation Index',
                indexcategory.vegetation,
                'https://doi.org/10.1016/0034-4257(95)00186-7',
                required_bands = ['NIR', 'RED']
            ),
            'ARVI': indexdef(
                'ARVI', 
                '(NIR - RED - 2 * (RED - BLUE)) / (NIR + RED - 2 * (RED - BLUE))',
                'Atmospherically Resistant Vegetation Index',
                indexcategory.vegetation,
                'https://doi.org/10.1109/36.134076',
                required_bands = ['NIR','RED', 'BLUE']
            ),
            'CVI': indexdef(
                'CVI', 
                '(NIR * RED) / pow(GREEN, 2)',
                'Chlorophyll Vegetation Index',
                indexcategory.vegetation,
                'https://doi.org/10.1007/s11119-010-9204-3',
                required_bands = ['NIR', 'RED', 'GREEN']
            ),
            'EVI': indexdef(
                'EVI', 
                '2.5 * (NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1)',
                'Enhanced Vegetation Index',
                indexcategory.vegetation,
                'https://doi.org/10.1016/S0034-4257(96)00112-5',
                required_bands =['NIR', 'RED', 'BLUE']
            ),
            'GBNDVI': indexdef(
                'GBNDVI',  
                '(NIR - (GREEN + BLUE)) / (NIR + (GREEN + BLUE))',
                'Green-Blue NDVI',
                indexcategory.vegetation,
                'https://doi.org/10.1016/S1672-6308(07)60027-4',
                required_bands = ['NIR', 'GREEN', 'BLUE']
            ),
            'MTVI1': indexdef(
                'MTVI1', 
                '(1.2 * (1.2 * (NIR - GREEN) - 2.5 * (RED - GREEN)))',
                'Modified Triangular Vegetation Index 1',
                indexcategory.vegetation,
                'https://doi.org/10.1016/j.rse.2003.12.013',
                required_bands = ['NIR', 'GREEN', 'RED']
            ),
            #Moisture based index
            'NDMI': indexdef(
                'NDMI', 
                'normalizedDifference(NIR, SWIR1)' ,
                'Normalized Difference Moisture Index',
                indexcategory.moisture,
                'https://doi.org/10.1016/S0034-4257(01)00318-2',
                required_bands = ['NIR', 'SWIR1']
            ),
            'RVMI': indexdef(
                'RVMI', 
                '(NDVI - NDMI) / (NDVI + NDMI)',
                'Renormalization of Vegetation Mositure Index',
                indexcategory.moisture,
               'https://doi.org/10.1080/10106049.2019.1687592',
               required_bands=['NIR', 'RED', 'SWIR1']
            ),
            #Burn Index
            'NBR': indexdef(
                'NBR', 
                'normalizedDifference(NIR, SWIR2)',
                'Normalized Burn Ratio',
                indexcategory.burn,
                'https://doi.org/10.3133/ofr0211',
                required_bands = ['NIR', 'SWIR2']
            ),
            'NBR2': indexdef(
                'NBR2', 
                'normalizedDifference(SWIR1, SWIR2)',
                'Normalized Burn Ratio 2',
                indexcategory.burn,
                'https://www.usgs.gov/landsat-missions/landsat-normalized-burn-ratio-2',
                required_bands = ['SWIR1', 'SWIR2']
            ),
            'NBRSWIR': indexdef(
                'NBRSWIR', 
                '(SWIR2 - SWIR1 - 0.002)/(SWIR2 + SWIR1 + 0.1)',
                'Normalized Burn Ratio with SWIR Bands',
                indexcategory.burn,
                'https://doi.org/10.1080/22797254.2020.1738900',
                required_bands = ['SWIR1', 'SWIR2']
            ),
            #Water index
            'MNDWI': indexdef(
                'MNDWI',
                'normalizedDifference(GREEN, SWIR1)',
                'Modified Normalized Difference Water Index',
                indexcategory.water,
                'https://doi.org/10.1080/01431160600589179',
                required_bands = ['SWIR1', 'GREEN']
            ),
            'AWEI': indexdef(
                'AWEI', 
                'BLUE + 2.5 * GREEN - 1.5 * (NIR + SWIR1) - 0.25 * SWIR2',
                'Automatic Water Extraction Index',
                indexcategory.water,
                'https://doi.org/10.1016/j.rse.2013.08.029',
                required_bands=['BLUE', 'GREEN', 'NIR', 'SWIR1', 'SWIR2']
            ),
            #Soil/Built-up
            'BLFEI': indexdef(
                'BLFEI', 
                '(((GREEN + RED + SWIR2)/3) - SWIR1)/(((GREEN + RED + SWIR2)/3)+SWIR1)',
                'Built-up Land Feature Extraction Index',
                indexcategory.soil_build_up,
                'https://doi.org/10.1080/10106049.2018.1497094',
                required_bands=['GREEN', 'RED', 'SWIR1', 'SWIR2']
            ),
            'NSDS': indexdef(
                'NSDS', 
                '(SWIR1-SWIR2)/SWIR1',
                'Normalized Shortwave-Infrared Difference Bare Soil Moisture Index 1',
                indexcategory.soil_build_up,
                'https://doi.org/10.1016/j.isprsjprs.2019.06.012',
                required_bands = ['SWIR1', 'SWIR2']
            ),
            'DBSI': indexdef(
                'DBSI',
                '((SWIR1 - GREEN)/(SWIR1 + GREEN)) - ((NIR - RED)/(NIR + RED))',
                'Dry Bareness Soil Index',
                indexcategory.soil_build_up,
                'https://doi.org/10.3390/land7030081',
                required_bands=['SWIR1', 'GREEN', 'RED', 'NIR']
            ),
            #Tasseled Cap (only works for Landsat 8/9)
            'Tascap': indexdef(
                'Tascap', 'Matrix multiplication with TC coefficients',
                'Tasseled Cap Transformation (Brightness, Greeness, Wetness)',
                indexcategory.tascap,
                'https://doi.org/10.1016/j.rse.2022.112992',
                required_bands=['BLUE', 'GREEN', 'RED', 'NIR', 'SWIR1', 'SWIR2']
            )
        }
        return indices
    #Probide information regarding indices categories
    def indices_list(self, category:Optional[indexcategory]):
        """
        List of supported indices, can be filtered by category
        """
        if category:
            indices = {k: v for k, v in self.avaliable_indices.items() if v.category == category}
            logger.info(f"avaliable indices in category '{category.value}':")
        else: 
            indices = self.avaliable_indices
            logger.info('Supported spectral transformation:')
        for idx_name, idx_def in indices.items():
            ref_info = f"(Reference: {idx_def.reference})" if idx_def.reference else ""
            logger.info (f" - {idx_name}: {idx_def.description}{ref_info}")
    def list_category(self) -> None:
        """
        Indices category
        """
        logger.info("Spectral transformation category:")
        for category in indexcategory:
            count = sum(1 for idx in self.avaliable_indices.values()if idx.category == category)
            logger.info(f"  - {category.value}: {count} indices")
            
    #Perform Calculations
    def calculate_individual_index(self, image: ee.Image, index_name: str) -> ee.Image:
        """Calculate a single spectral index"""
        idx_def = self.avaliable_indices[index_name]
        try:
            if index_name == 'NDVI':
                result = image.normalizedDifference(['NIR', 'RED']).rename('NDVI')
            elif index_name == 'GNDVI':
                result = image.normalizedDifference(['NIR', 'GREEN']).rename('GNDVI')
            elif index_name == 'NDMI':
                result = image.normalizedDifference(['NIR', 'SWIR1']).rename('NDMI')
            elif index_name == 'NBR':
                result = image.normalizedDifference(['NIR', 'SWIR2']).rename('NBR')
            elif index_name == 'NBR2':
                result = image.normalizedDifference(['SWIR1', 'SWIR2']).rename('NBR2')
            elif index_name == 'MNDWI':
                result = image.normalizedDifference(['GREEN', 'SWIR1']).rename('MNDWI')
            elif index_name == 'MSAVI':
                result = image.expression(
                    '(2 * NIR + 1 - sqrt(pow(2 * NIR + 1, 2) - 8 * (NIR - RED))) / 2', 
                    {'NIR': image.select('NIR'), 'RED': image.select('RED')}
                ).rename('MSAVI')
            elif index_name == 'NSDS':
                result = image.expression(
                    '(SWIR1-SWIR2)/SWIR1', 
                    {'SWIR1': image.select('SWIR1'),'SWIR2': image.select('SWIR2')}
                    ).rename('NSDS')   
            elif index_name == 'MTVI1':
                result = image.expression(
                    '(1.2 * (1.2 * (NIR - GREEN) - 2.5 * (RED - GREEN)))',
                    {'NIR': image.select('NIR'), 'RED': image.select('RED'), 'GREEN': image.select('GREEN')}
                ).rename('MTVI1')
            elif index_name == 'OSAVI':
                result = image.expression(
                    '(NIR - RED) / (NIR + RED + 0.16)',
                    {'NIR': image.select('NIR'), 'RED': image.select('RED')}
                ).rename('OSAVI')
            elif index_name == 'ARVI':
                result = image.expression(
                    '(NIR - RED - 2 * (RED - BLUE)) / (NIR + RED - 2 * (RED - BLUE))',
                    {'NIR': image.select('NIR'), 'RED': image.select('RED'), 'BLUE': image.select('BLUE')}
                ).rename('ARVI')
            elif index_name == 'CVI':
                result = image.expression(
                    '(NIR * RED) / pow(GREEN, 2)',
                    {'NIR': image.select('NIR'), 'RED': image.select('RED'), 'GREEN': image.select('GREEN')}
                ).rename('CVI')
            elif index_name == 'EVI':
                result = image.expression(
                    '2.5 * (NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1)',
                    {'NIR': image.select('NIR'), 'RED': image.select('RED'), 'BLUE': image.select('BLUE')}
                ).rename('EVI')
            elif index_name == 'GBNDVI':
                result = image.expression(
                    '(NIR - (GREEN + BLUE)) / (NIR + (GREEN + BLUE))',
                    {'NIR': image.select('NIR'), 'GREEN': image.select('GREEN'), 'BLUE': image.select('BLUE')}
                ).rename('GBNDVI')
            elif index_name == 'RVMI':
                ndvi = image.normalizedDifference(['NIR', 'RED'])
                ndmi = image.normalizedDifference(['NIR', 'SWIR1'])
                result = ndvi.subtract(ndmi).divide(ndvi.add(ndmi)).rename('RVMI')
            elif index_name == 'NBRSWIR':
                result = image.expression(
                    '(SWIR2 - SWIR1 - 0.002)/(SWIR2 + SWIR1 + 0.1)',
                    {'SWIR1': image.select('SWIR1'), 'SWIR2': image.select('SWIR2')}
                ).rename('NBRSWIR')
            elif index_name == 'BLFEI':
                result = image.expression(
                    '(((GREEN + RED + SWIR2)/3) - SWIR1)/(((GREEN + RED + SWIR2)/3)+SWIR1)',
                    {'GREEN': image.select('GREEN'), 'RED': image.select('RED'), 
                    'SWIR1': image.select('SWIR1'), 'SWIR2': image.select('SWIR2')}        
                ).rename('BLFEI')
            elif index_name == 'DBSI':
                result = image.expression(
                    '((SWIR1 - GREEN)/(SWIR1 + GREEN)) - ((NIR - RED)/(NIR + RED))',
                {'SWIR1': image.select('SWIR1'), 'GREEN': image.select('GREEN'),
                'NIR': image.select('NIR'), 'RED': image.select('RED')}                    
                ).rename('DBSI')
            elif index_name == 'AWEI':
                result = image.expression(
                    'BLUE + 2.5 * GREEN - 1.5 * (NIR + SWIR1) - 0.25 * SWIR2',
                {'BLUE': image.select('BLUE'), 'GREEN': image.select('GREEN'), 'NIR': image.select('NIR'),
                'SWIR1': image.select('SWIR1'), 'SWIR2': image.select('SWIR2')}                   
                ).rename('AWEI')
            elif index_name == 'Tascap':
                #Tasseled Cap Transformation coefficient (works for Landsat 8 and 9 Surface Reflectance Data)
                tc_coeffs = ee.Array([
                    [0.3690, 0.4271, 0.4689, 0.5073, 0.3824, 0.2406],    # Brightness
                    [-0.2870, -0.2685, -0.4087, 0.8145, 0.0637, -0.1052], # Greenness  
                    [0.0382, 0.2137, 0.3536, 0.2270, -0.6108, -0.6351]    # Wetness
                ])
                bands_array = image.select(['BLUE', 'GREEN', 'RED', 'NIR', 'SWIR1', 'SWIR2']).toArray()
                result = ee.Image(tc_coeffs).matrixMultiply(bands_array.toArray(1)).arrayProject([0]).arrayFlatten([['brightness', 'greenness', 'wetness']])
            else:
                raise ValueError(f"Unknown index: {index_name}")
            
            logger.info(f"Successfully calculated {index_name}: {idx_def.description}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to calculate {index_name}: {str(e)}")
            return None
    def calculate_index(self, image: ee.Image,
                        index: Optional[List[str]] = None,
                        categories: Optional[List[indexcategory]] = None,
                        ):
        """
        Perform spectral transformation on specified spectral index
        """

        logger.info('Starting spectral indices calculation')
        if index is not None:
            invalid_index = [idx for idx in index if idx not in self.avaliable_indices]
            if invalid_index:
                logger.error(f"Invalid indices: {invalid_index}")
                raise ValueError(f"Invalid indices: {invalid_index}")
            indices_to_calculate = index
        elif categories is not None:
            indices_to_calculate = [
                idx_name for idx_name, idx_def in self.avaliable_indices.items()
                if idx_def.category in categories
            ]
        else:
            indices_to_calculate = list(self.avaliable_indices.keys())
        logger.info(f"Calculating {len(indices_to_calculate)} indices: {indices_to_calculate}")
        transformed = []
        self.calculated_count = 0

        for idx_name in indices_to_calculate:
            result = self.calculate_individual_index(image, idx_name)
            if result is not None:
                transformed.append(result)
                self.calculated_count+= 1
        if not transformed:
            logger.info("No indices were successfully calculated")
            return ee.Image([])
        
        stacked = ee.Image.cat(transformed)
        logger.info(f"Calculatation sucessfull on {self.calculated_count} spectral indices")
        return stacked
    def calculation_summary (self) -> Dict[str, Any]:
        """
        Summary of the calculation process
        """
        return {
            'Indices Calculated': self.calculated_count,
            'Avaliable Indices': len(self.avaliable_indices),
            'Categories': [cat.value for cat in indexcategory]
        }
#=================================== Terrain Metrics ===================================
class terrain_calculator:
    """
    Calculate terrain metrics for creating input covariates in land cover mapping.
    Provides elevation, slope, and aspect data.
    """
    def __init__(self):
        """
        Initialize the terrain calculator.
        Ensure Earth Engine is initialized lazily.
        """
        ensure_ee_initialized()
        logger.info("Terrain calculator initialized")
    
    def calculate_terrain(self, aoi: ee.Geometry) -> ee.Image:
        """
        Return stacked terrain metric consisting of elevation, slope, and aspect data.
        
        Parameters:
        -----------
        aoi : ee.Geometry
            Area of interest geometry
            
        Returns:
        --------
        ee.Image : Stacked terrain metrics (elevation, slope, aspect)
        
        Notes:
        ------
        - Further development should use Indonesian National DEM (DEMNAS) for higher spatial resolution
        - Future additions: Global ALOS Landforms 
          (https://developers.google.com/earth-engine/datasets/catalog/CSP_ERGo_1_0_Global_ALOS_landforms#bands)
        """
        try:
            logger.info("Calculating terrain metrics...")
            
            #Get elevation data
            #Use NASADEM since it provide better height accuracy compared to other DEM data
            dem = ee.Image("NASA/NASADEM_HGT/001").select('elevation').clip(aoi)
            #Calculate slope (units are degrees, range is [0,90))
            slope = ee.Terrain.slope(dem).rename('slope')
            #Calculate aspect (units are degrees where 0=N, 90=E, 180=S, 270=W)
            aspect = ee.Terrain.aspect(dem).rename('aspect')
            #Stack the terrain metrics
            terrain_metric = ee.Image.cat(dem, slope, aspect)
            logger.info("Successfully calculated terrain metrics: elevation, slope, aspect")
            return terrain_metric
            
        except Exception as e:
            logger.error(f"Failed to calculate terrain metrics: {str(e)}")
            return None
#=================================== Distance Metrics ===================================
class distance_calculator:
    """
    Calculate distance metrics based on predefined datasets (roads, coastline, settlements, etc.)
    for creating input covariates in land cover mapping.
    """
    def __init__(self):
        """
        Initialize the distance calculator.
        Ensure Earth Engine is initialized lazily.
        """
        ensure_ee_initialized()
        logger.info("Distance calculator initialized")
    
    def calculate_distance_metrics(self, 
                                   aoi: ee.Geometry, 
                                   max_dist: float = 500000, 
                                   in_meters: bool = False) -> ee.Image:
        """
        Create distance images based on predefined datasets (roads, coastline, settlements).
        
        Parameters:
        -----------
        aoi : ee.Geometry
            Area of interest geometry
        max_dist : float
            Maximum cumulative cost distance (in meters or pixels). Default: 500000
        in_meters : bool
            If True, output distance in meters instead of pixels. Default: False
            Note: Using meters requires more computation time
            
        Returns:
        --------
        ee.Image : Stacked distance metrics (dist_roads, dist_coast, dist_settlement)
        
        Notes:
        ------
        - The datasets used for distance metrics may be outdated and require updates
        - Datasets include: Natural Earth Coastline, OSM roads, RBI roads, HRSL settlements
        """
        try:
            logger.info(f"Calculating distance metrics (max_dist={max_dist}, in_meters={in_meters})...")
            
            # Set up cost image based on distance unit preference
            if in_meters:
                # Distance in meters (requires more computation time)
                cost_image = ee.Image.pixelArea().sqrt()
                logger.info("Using meter-based distance calculation")
            else:
                # Distance in pixel units (more efficient storage)
                cost_image = ee.Image(1)
                logger.info("Using pixel-based distance calculation")
            
            # Helper function to calculate distance from source mask
            def distance_image(source_mask):
                """Source mask is binary mask where value 1 is unmasked data (starting location)"""
                return cost_image.cumulativeCost(source_mask, max_dist)
            
            # Natural Earth Coastline data
            # (https://www.naturalearthdata.com/downloads/10m-physical-vectors/10m-coastline/)
            logger.info("Processing coastline data...")
            ne_coastline = ee.FeatureCollection(
                "users/hadicu06/IIASA/RESTORE/vector_datasets/coastline_ne_10m"
            ).filterBounds(aoi)
            coast_dist = ne_coastline.map(lambda ft: ft.set('constant', 1)) \
                                    .reduceToImage(['constant'], ee.Reducer.first()) \
                                    .mask()
            
            # Roads data (OSM and RBI)
            # Note: May require updates
            logger.info("Processing roads data...")
            road_osm = ee.FeatureCollection(
                "users/hadicu06/IIASA/RESTORE/vector_datasets/road_osm"
            ).filterBounds(aoi)
            road_rbi = ee.FeatureCollection(
                "users/hadicu06/IIASA/RESTORE/vector_datasets/road_rbi"
            ).filterBounds(aoi)
            roads_dist = road_osm.merge(road_rbi) \
                                .map(lambda ft: ft.set('constant', 1)) \
                                .reduceToImage(['constant'], ee.Reducer.max()) \
                                .mask()
            
            # Settlement data from High Resolution Settlement Layer (HRSL)
            # (https://dataforgood.facebook.com/dfg/tools/high-resolution-population-density-maps)
            logger.info("Processing settlement data...")
            hrsl = ee.ImageCollection("projects/sat-io/open-datasets/hrsl/hrslpop") \
                    .mosaic() \
                    .clip(aoi)
            hrsl_connected = hrsl.int().connectedPixelCount(maxSize=100, eightConnected=True)
            hrsl_masked = hrsl_connected.unmask().gt(3)
            
            # Calculate distance metrics
            logger.info("Computing distance transformations...")
            dist_roads = distance_image(roads_dist).rename('dist_roads')
            dist_coast = distance_image(coast_dist).rename('dist_coast')
            dist_settlement = distance_image(hrsl_masked).rename('dist_settlement')
            
            # Stack into one image
            distance_metrics = ee.Image.cat(dist_roads, dist_coast, dist_settlement)
            
            logger.info("Successfully calculated distance metrics: dist_roads, dist_coast, dist_settlement")
            return distance_metrics
            
        except Exception as e:
            logger.error(f"Failed to calculate distance metrics: {str(e)}")
            return None