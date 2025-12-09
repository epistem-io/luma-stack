import ee
from datetime import datetime
import logging
from .ee_config import ensure_ee_initialized

# Do not initialize Earth Engine at import time. Initialize when an instance is created.

#Configure root for global functions
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
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
    #Function to check if thermal bands are available for a given dataset
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
        #Landsat 1-3 MSS sensors did not have thermal bands
        if optical_data in ['L1_RAW', 'L2_RAW', 'L3_RAW']:
            return False
        
        #Check if corresponding thermal dataset exists
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
    #Functions standarize Landsat bands 
    def rename_landsat_bands(self, image, sensor_type):
        """
        Standardize Landsat Surface Reflectance (SR) band names based on sensor type. From 'SR_B*' or 'B*' to 'NIR', 'GREEN', etc.

        Parameters
        ----------
        image : ee.Image. Landsat SR image
        sensor_type : str. Sensor type ('L5', 'L7', 'L8', 'L9')

        Returns
        -------
        ee.Image : Image with standardized band names

        Example
        --------

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
        """        
        optical_bands = image.select('SR_B.').multiply(0.0000275).add(-0.2)
        #thermal_bands = image.select('ST_B.*').multiply(0.00341802).add(149.0)
        return image.addBands(optical_bands, None, True)
    
        #Add new code for calculating cloud cover within AOI
    def add_aoi_cloud_cover(self, img, aoi, scale=90, max_pixels=1e9,
                            cloud_conf_thresh=2, shadow_conf_thresh=2, cirrus_conf_thresh=2,
                            debug=False):
        """
        Calculate cloud/shadow/cirrus coverage percentage within AOI.
        Uses the same QA_PIXEL logic as mask_landsat_sr() for consistency.
        
        Parameters
        ----------
        img : ee.Image
            Landsat image with QA_PIXEL band
        aoi : ee.Geometry or ee.FeatureCollection
            Area of interest for cloud calculation
        scale : int
            Pixel scale for computation in meters (default: 90m for efficiency)
            Use 30m for accurate results on small areas, 90m+ for large regions
        max_pixels : float
            Maximum number of pixels to process (default: 1e9)
        cloud_conf_thresh : int
            Cloud confidence threshold (0=None, 1=Low, 2=Med, 3=High)
        shadow_conf_thresh : int
            Shadow confidence threshold (0=None, 1=Low, 2=Med, 3=High)
        cirrus_conf_thresh : int
            Cirrus confidence threshold (0=None, 1=Low, 2=Med, 3=High)
        debug : bool
            If True, print debug information about cloud calculation (default: False)
        
        Returns
        -------
        ee.Image
            Input image with added 'CLOUDY_PERC_AOI' property
        
        Example
        -------
        >>> collection = collection.map(lambda img: optical_reflectance.add_aoi_cloud_cover(img, aoi, scale=90))
        >>> collection = collection.filter(ee.Filter.lt('CLOUDY_PERC_AOI', 30))
        """
        #Get QA band and clip to AOI
        qa = img.select('QA_PIXEL').clip(aoi)
        #=== SAME LOGIC AS mask_landsat_sr() ===
        #Deterministic bits
        cloud_bit = 1 << 3
        shadow_bit = 1 << 4
        cirrus_bit = 1 << 2
        #Bitwise operations to get the masks
        cloud_mask = qa.bitwiseAnd(cloud_bit).eq(0)
        shadow_mask = qa.bitwiseAnd(shadow_bit).eq(0)
        cirrus_mask = qa.bitwiseAnd(cirrus_bit).eq(0)
        #Confidence bits
        cloud_conf = qa.rightShift(8).bitwiseAnd(3)
        shadow_conf = qa.rightShift(10).bitwiseAnd(3)
        cirrus_conf = qa.rightShift(14).bitwiseAnd(3)
        #Keep pixels below thresholds
        conf_mask = (cloud_conf.lt(cloud_conf_thresh)
                    .And(shadow_conf.lt(shadow_conf_thresh))
                    .And(cirrus_conf.lt(cirrus_conf_thresh)))
        
        #Final clear mask (1 = clear, 0 = cloudy/masked)
        clear_mask_aoi = cloud_mask.And(shadow_mask).And(cirrus_mask).And(conf_mask)
        #Calculate areas
        pixel_area = ee.Image.pixelArea()
        #Total area in AOI
        total_area = pixel_area.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=aoi,
            scale=scale,
            maxPixels=max_pixels,
            bestEffort=True,
            tileScale=4
        )
        #Clear (unmasked) area and renamed
        clear_area_img = clear_mask_aoi.multiply(pixel_area).rename('clear_area')
        clear_area = clear_area_img.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=aoi,
            scale=scale,
            maxPixels=max_pixels,
            bestEffort=True,
            tileScale=4
        )
        #Calculate cloud percentage with null handling
        total = ee.Number(total_area.get('area')).max(1)  # Avoid division by zero
        clear = ee.Number(clear_area.get('clear_area')).max(0)  # Default to 0 if null
        
        clear_percent = clear.multiply(100).divide(total)
        cloud_percent = ee.Number(100).subtract(clear_percent).round()
        #Clamp between 0 and 100
        cloud_percent = cloud_percent.max(0).min(100)
        
        # Debug logging if requested
        if debug:
            scene_id = img.get('system:index')
            cloud_val = cloud_percent.getInfo()
            total_val = total.getInfo()
            clear_val = clear.getInfo()
            self.logger.info(f"Scene {scene_id.getInfo()}: AOI Cloud={cloud_val}%, Total={total_val/1e6:.2f}km², Clear={clear_val/1e6:.2f}km²")
        
        return img.set('CLOUDY_PERC_AOI', cloud_percent)

    #Function to retrive Landsat multispectral bands
    def get_optical_data(self, aoi, start_date, end_date, optical_data='L8_SR',
                        cloud_cover=30, apply_cloud_mask=True, verbose=True, compute_detailed_stats=True):
        """
        Get optical image collection for Landsat 1-9 SR data with detailed information logging.

        Parameters
        ----------
        aoi :  ee.FeatureCollection. Area of interest.
        start_date : str. Start date in format 'YYYY-MM-DD' or year.
        end_date : str. End date in format 'YYYY-MM-DD' or year.
        optical_data : str. Dataset type: i.e 'L5_SR', 'L7_SR', 'L8_SR', 'L9_SR'.
        cloud_cover : int. Maximum cloud cover percentage on land (default: 30).
        apply_cloud_mask : bool. If True, apply QA-based cloud masking (default: True).
            Set to False to skip cloud masking and preserve all pixels.
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
        >>> collection, stats = get_landsat.get_optical_data(aoi, 2020, 2023, 'L8_SR', 30, True, True)
        >>> # Without cloud masking
        >>> collection, stats = get_landsat.get_optical_data(aoi, 2020, 2023, 'L8_SR', 
        ...                                                   cloud_cover=30, apply_cloud_mask=False)
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

        #Use verbose to import detailed logging information
        if verbose:
            self.logger.info(f"Starting data fetch for {config['description']}")
            self.logger.info(f"Date range: {start_date} to {end_date}")
            self.logger.info(f"Cloud cover threshold: {cloud_cover}%")
            if not compute_detailed_stats:
                self.logger.info("detailed statistics will not be computed")

        #Initial collection statistic
        initial_collection = (ee.ImageCollection(config['collection'])
                            .filterBounds(aoi)
                            .filterDate(start_date, end_date))
        #Used the class Reflectance stats for improved report 
        stats_object = Reflectance_Stats()
        initial_stats = stats_object.get_collection_statistics(initial_collection, compute_detailed_stats)
        if verbose and compute_detailed_stats and initial_stats.get('total_images', 0) > 0:
            self.logger.info(f"Initial collection (before cloud filtering): {initial_stats['total_images']} images")
            self.logger.info(f"Date range of available images: {initial_stats['date_range']}")

        #Collection after cloud cover filter
        collection = initial_collection.filter(ee.Filter.lt(config['cloud_property'], cloud_cover))
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
        #Apply masking and band renaming to image collection after filtering. Masking is now optional 
        if apply_cloud_mask:
            collection = collection.map(lambda img: self.mask_landsat_sr(img))
            if verbose:
                self.logger.info("Cloud masking applied using QA_PIXEL band")
        else:
            if verbose:
                self.logger.info("Cloud masking skipped - all pixels preserved")
        
        collection = (collection
                    .map(lambda img: self.apply_scale_factors(img))
                    .map(lambda img: self.rename_landsat_bands(img, config['sensor'])))

        #Return results
        return collection, {
            'dataset': config['description'],
            'sensor': config['sensor'],
            'date_range_requested': f"{start_date} to {end_date}",
            'cloud_cover_threshold': cloud_cover,
            'cloud_mask_applied': apply_cloud_mask,
            'initial_collection': initial_stats,
            'filtered_collection': filtered_stats,
            'detailed_stats_computed': compute_detailed_stats
        }
    #TOA-based Thermal Bands
    def get_thermal_bands(self, aoi, start_date, end_date, thermal_data = 'L8_TOA', cloud_cover=30,
                        apply_cloud_mask=True, verbose=True, compute_detailed_stats=True):
        """
        Get the thermal bands from landsat TOA data
    
        Parameters
        ----------
        aoi :  ee.FeatureCollection. Area of interest.
        start_date : str. Start date in format 'YYYY-MM-DD' or year.
        end_date : str. End date in format 'YYYY-MM-DD' or year.
        thermal_data : str. Dataset type: 'L4_TOA', 'L5_TOA', 'L7_TOA', 'L8_TOA', 'L9_TOA'.
        cloud_cover : int. Maximum cloud cover percentage on land (default: 30).
        verbose : bool. Print detailed information about the collection (default: True).
        compute_detailed_stats : bool
            If True, compute detailed statistics 
            If False, return only basic information (default: True).
            
        Returns
        -------
        tuple : (ee.ImageCollection, dict)
            Filtered and preprocessed image collection with statistics.
        """
        #Helper function to parse the date so that the user can only input the year
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
        
        #Apply masking (QA-based) - optional
        if apply_cloud_mask:
            collection = collection.map(lambda img: self.mask_landsat_sr(img))
            if verbose:
                self.logger.info("Cloud masking applied using QA_PIXEL band")
        else:
            if verbose:
                self.logger.info("Cloud masking skipped - all pixels preserved")
        
        collection = collection.select(thermal_band)
        collection = collection.map(rename_thermal_band)

        #Return collection and stats
        return collection, {
            'dataset': config['description'],
            'sensor': sensor,
            'thermal_band': thermal_band,
            'date_range_requested': f"{start_date} to {end_date}",
            'cloud_cover_threshold': cloud_cover,
            'cloud_mask_applied': apply_cloud_mask,
            'initial_collection': initial_stats,
            'filtered_collection': filtered_stats,
            'detailed_stats_computed': compute_detailed_stats
        }
class Reflectance_Stats:
    """
    Class for fetching image collection statistics
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
    def get_collection_statistics(self, collection, compute_stats=True, print_report=False):
        """
        Get comprehensive statistics about an image collection.
        """
        #Get the number of image used 
        try:
            size = collection.size()
            if compute_stats:
                total_images = size.getInfo() #Client side operation, produce number of image collection
                if total_images > 0:
                    #Get the cloud cover percentage, and image aqcusition date
                    cloud_values = collection.aggregate_array('CLOUD_COVER_LAND').getInfo()
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
    Supports single-date mosaics and time series compositing.
    """
    
    def __init__(self, log_level=logging.INFO):
        """
        Initialize the final_Image object and set up a class-specific logger.
        """
        ensure_ee_initialized()
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)
        self.logger.info("final_Image initialized.")
    #Function to calculate data coverage (pixel validity) within AOI
    def calculate_data_coverage(self, composite_image, aoi, scale=30, max_pixels=1e9, verbose=True):
        """
        Calculate data coverage (valid pixels) percentage within AOI for a composite image.
        This shows how much of your AOI has valid data after compositing.
        
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
            - 'BLUE': Blue band (inverted - selects clearest pixels, since lower blue value correspond to haze)
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
            elif quality_band == 'BLUE':
                #Invert BLUE band (lower blue = clearer, so negate it)
                inverted_blue = img.select('BLUE').multiply(-1).rename('quality')
                return img.addBands(inverted_blue)
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
        
        # Add metadata
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
    def get_temporal_composite(self, collection, aoi, reducer='median', 
                              add_band_stats=False, calculate_coverage=False, 
                              coverage_scale=30, verbose=True):
        """
        Create a temporal composite from image collection using specified reducer.
        Output is always clipped to the AOI.
        
        Parameters
        ----------
        collection : ee.ImageCollection
            Filtered image collection from Reflectance_Data
        aoi : ee.Geometry or ee.FeatureCollection
            Area of interest for clipping
        reducer : str or ee.Reducer
            Reduction method: 'median', 'mean', 'min', 'max', 'percentile_'
            (default: 'median')
        add_band_stats : bool
            If True, add additional bands with stdDev and count (default: False)
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
        if isinstance(aoi, ee.FeatureCollection):
            geometry = aoi.geometry()
        else:
            geometry = aoi
        
        # Get original band names before reduction (server-side ee.List)
        original_bands = collection.first().bandNames()

        # Resolve reducer argument to an Earth Engine reducer
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

        # Check collection size - only if verbose to avoid unnecessary getInfo()
        if verbose:
            size = collection.size().getInfo()
            if size == 0:
                raise ValueError("Collection is empty, cannot create composite")
            self.logger.info(f"Creating {reducer} composite from {size} images")

        # Create composite
        if add_band_stats:
            # Compute main composite (using the requested reducer) and two additional stats: stdDev and count
            main_comp = collection.reduce(ee_reducer).rename(original_bands)
            std_comp = collection.reduce(ee.Reducer.stdDev()).rename(
                original_bands.map(lambda b: ee.String(b).cat('_stdDev'))
            )
            count_comp = collection.reduce(ee.Reducer.count()).rename(
                original_bands.map(lambda b: ee.String(b).cat('_count'))
            )

            # Combine bands in a predictable order: main bands, stdDev bands, count bands
            composite = main_comp.addBands(std_comp).addBands(count_comp)
        else:
            # Single reducer composite - rename main bands back to original names
            composite = collection.reduce(ee_reducer).rename(original_bands)
        
        # Clip to AOI (always required)
        composite = composite.clip(geometry)
        
        if verbose:
            self.logger.info("Composite clipped to AOI")
        
        # Add metadata - use server-side operations, avoid getInfo()
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