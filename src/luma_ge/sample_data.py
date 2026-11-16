"""
Module 3 Backend - Training Data Management

This module provides training data loading and processing functionality.
"""
import pandas as pd
import geopandas as gpd
import numpy as np
import ee
from typing import Dict, List, Optional, Tuple, Any, Union
import warnings
import logging
from shapely.geometry import shape

# Configure logging
logger = logging.getLogger(__name__)

try:
    import ee
    
    class SyncTrainData:
        """Training data synchronization functionality."""
        
        @staticmethod
        def LoadTrainData(landcover_df, aoi_geometry, training_shp_path=None, training_ee_path=None):
            """Load training data from EE asset or shapefile."""
            try:
                if training_ee_path:
                    logger.info(f"Loading training data from EE asset: {training_ee_path}")
                    
                    # Test Earth Engine authentication and asset access
                    try:
                        # Load from Earth Engine asset
                        training_fc = ee.FeatureCollection(training_ee_path)
                        
                        # Get initial count
                        initial_count = training_fc.size().getInfo()
                        logger.info(f"Initial feature count: {initial_count}")
                        
                        if initial_count == 0:
                            raise Exception(f"Earth Engine asset '{training_ee_path}' contains 0 features")
                            
                    except Exception as ee_error:
                        logger.error(f"Failed to access Earth Engine asset: {ee_error}")
                        raise Exception(f"Cannot access Earth Engine asset '{training_ee_path}': {str(ee_error)}")
                    
                    # Filter by AOI if provided
                    if aoi_geometry:
                        logger.info("Filtering by AOI bounds...")
                        logger.info(f"AOI geometry type: {type(aoi_geometry)}")
                        try:
                            # Convert AOI to EE geometry based on type
                            if hasattr(aoi_geometry, 'geometry') and hasattr(aoi_geometry.geometry, 'iloc'):
                                # It's a GeoDataFrame - get the first geometry
                                geom = aoi_geometry.geometry.iloc[0]
                                ee_geom = ee.Geometry(geom.__geo_interface__)
                                training_fc = training_fc.filterBounds(ee_geom)
                            elif hasattr(aoi_geometry, '__geo_interface__'):
                                # It's a shapely geometry
                                ee_geom = ee.Geometry(aoi_geometry.__geo_interface__)
                                training_fc = training_fc.filterBounds(ee_geom)
                            else:
                                # Assume it's already an EE geometry
                                training_fc = training_fc.filterBounds(aoi_geometry)
                            
                            filtered_count = training_fc.size().getInfo()
                            logger.info(f"Features after AOI filter: {filtered_count}")
                            
                            if filtered_count == 0:
                                logger.warning("AOI filtering resulted in 0 features - using original dataset")
                                # Reload original dataset without AOI filter
                                training_fc = ee.FeatureCollection(training_ee_path)
                        except Exception as filter_error:
                            logger.error(f"AOI filtering failed: {filter_error}")
                            logger.info("Using original dataset without AOI filter")
                            # Keep original training_fc without filtering
                    
                    # Manual conversion to GeoDataFrame with size limit
                    logger.info("Converting to GeoDataFrame...")
                    
                    # Check collection size and implement stratified sampling for large datasets
                    collection_size = training_fc.size().getInfo()
                    logger.info(f"Collection size: {collection_size}")
                    
                    # If collection is larger than 5000, use stratified sampling to ensure class representation
                    if collection_size > 5000:
                        logger.info(f"Collection has {collection_size} features, implementing stratified sampling for class representation")
                        
                        try:
                            # Get unique classes and their counts
                            class_field = 'kelas'  # The field containing class information
                            unique_classes = training_fc.aggregate_array(class_field).distinct().getInfo()
                            logger.info(f"Found {len(unique_classes)} unique classes: {unique_classes}")
                            
                            # Calculate samples per class (aim for 5000 total)
                            target_total = 5000
                            base_samples_per_class = target_total // len(unique_classes)
                            remaining_samples = target_total % len(unique_classes)
                            
                            logger.info(f"Target: {target_total} samples, Base per class: {base_samples_per_class}")
                            
                            all_features = []
                            class_counts = {}
                            
                            for i, class_value in enumerate(unique_classes):
                                try:
                                    # Calculate samples for this class
                                    samples_for_class = base_samples_per_class
                                    if i < remaining_samples:  # Distribute remaining samples to first few classes
                                        samples_for_class += 1
                                    
                                    logger.info(f"Sampling class {class_value}: targeting {samples_for_class} samples")
                                    
                                    # Filter by class and get count
                                    class_fc = training_fc.filter(ee.Filter.eq(class_field, class_value))
                                    class_size = class_fc.size().getInfo()
                                    
                                    if class_size == 0:
                                        logger.warning(f"Class {class_value} has no features, skipping")
                                        continue
                                    
                                    # Sample from this class
                                    actual_samples = min(samples_for_class, class_size)
                                    if class_size <= samples_for_class:
                                        # Take all features if class has fewer than target
                                        sampled_fc = class_fc
                                        logger.info(f"Class {class_value}: taking all {class_size} features")
                                    else:
                                        # Random sample if class has more than target
                                        sampled_fc = class_fc.randomColumn('random', 42).sort('random').limit(actual_samples)
                                        logger.info(f"Class {class_value}: sampling {actual_samples} from {class_size} features")
                                    
                                    # Get the features
                                    class_info = sampled_fc.getInfo()
                                    class_features = class_info['features']
                                    
                                    all_features.extend(class_features)
                                    class_counts[class_value] = len(class_features)
                                    
                                    logger.info(f"Class {class_value}: collected {len(class_features)} features")
                                    
                                except Exception as class_error:
                                    logger.error(f"Error sampling class {class_value}: {class_error}")
                                    continue
                            
                            # Create combined feature info
                            info = {
                                'type': 'FeatureCollection',
                                'features': all_features
                            }
                            features = all_features
                            
                            # Log final class distribution
                            logger.info(f"Stratified sampling complete: {len(features)} total features")
                            for class_val, count in class_counts.items():
                                percentage = (count / len(features)) * 100 if len(features) > 0 else 0
                                logger.info(f"  Class {class_val}: {count} features ({percentage:.1f}%)")
                            
                        except Exception as stratified_error:
                            logger.error(f"Stratified sampling failed: {stratified_error}")
                            logger.info("Falling back to simple random sampling")
                            # Fallback to simple random sampling
                            training_fc = training_fc.randomColumn('random', 42).sort('random').limit(5000)
                            info = training_fc.getInfo()
                            features = info['features']
                        
                    else:
                        # For collections <= 5000, load normally
                        logger.info(f"Collection size ({collection_size}) is within normal limits, loading directly")
                        info = training_fc.getInfo()
                        features = info['features']
                    
                    if collection_size == 0:
                        logger.warning("No features found in collection")
                        return {
                            'training_data': gpd.GeoDataFrame(columns=['kelas', 'geometry']),
                            'landcover_df': landcover_df,
                            'class_field': 'kelas',
                            'validation_results': {
                                'total_points': 0,
                                'valid_points': 0,
                                'points_after_class_filter': 0,
                                'invalid_classes': [],
                                'outside_aoi': [],
                                'insufficient_samples': [],
                                'warnings': ['No training data found in AOI']
                            }
                        }
                    

                    logger.info(f"Features to convert: {len(features)}")
                    
                    data = []
                    for f in features:
                        try:
                            geom = shape(f['geometry'])
                            props = f['properties']
                            props['geometry'] = geom
                            data.append(props)
                        except Exception as geom_error:
                            logger.warning(f"Error processing feature geometry: {geom_error}")
                            continue

                    logger.info(f"Successfully processed {len(data)} features")
                    training_gdf = gpd.GeoDataFrame(data, geometry='geometry', crs='EPSG:4326')
                    
                    # Log class field info
                    if 'kelas' in training_gdf.columns:
                        unique_classes = training_gdf['kelas'].unique()
                        logger.info(f"Unique classes in training data: {unique_classes}")
                        logger.info(f"Class counts: {training_gdf['kelas'].value_counts().to_dict()}")
                    else:
                        logger.warning("'kelas' field not found in training data")
                        logger.info(f"Available columns: {training_gdf.columns.tolist()}")

                    return {
                        'training_data': training_gdf,
                        'landcover_df': landcover_df,
                        'class_field': 'kelas',
                        'validation_results': {
                            'total_points': len(training_gdf),
                            'valid_points': len(training_gdf),
                            'points_after_class_filter': len(training_gdf),
                            'invalid_classes': [],
                            'outside_aoi': [],
                            'insufficient_samples': [],
                            'warnings': []
                        }
                    }
                elif training_shp_path:
                    logger.info(f"Loading training data from shapefile: {training_shp_path}")
                    data = gpd.read_file(training_shp_path)
                    training_gdf = gpd.GeoDataFrame(data, geometry='geometry', crs='EPSG:4326')
                    
                    # Log class field info
                    if 'kelas' in training_gdf.columns:
                        unique_classes = training_gdf['kelas'].unique()
                        logger.info(f"Unique classes in training data: {unique_classes}")
                        logger.info(f"Class counts: {training_gdf['kelas'].value_counts().to_dict()}")
                    else:
                        logger.warning("'kelas' field not found in training data")
                        logger.info(f"Available columns: {training_gdf.columns.tolist()}")
                    
                    return {
                        'training_data': training_gdf,
                        'landcover_df': landcover_df,
                        'class_field': 'kelas',
                        'validation_results': {
                            'total_points': len(training_gdf),
                            'valid_points': len(training_gdf),
                            'points_after_class_filter': len(training_gdf),
                            'invalid_classes': [],
                            'outside_aoi': [],
                            'insufficient_samples': [],
                            'warnings': []
                        }
                    }
                else:
                    raise ValueError("Either training_ee_path or training_shp_path must be provided")
                    
            except Exception as e:
                logger.error(f"Error loading training data: {str(e)}")
                import traceback
                logger.error(f"Full traceback: {traceback.format_exc()}")
                return {
                    'training_data': None,
                    'landcover_df': landcover_df,
                    'class_field': 'kelas',
                    'validation_results': {
                        'total_points': 0,
                        'valid_points': 0,
                        'points_after_class_filter': 0,
                        'invalid_classes': [],
                        'outside_aoi': [],
                        'insufficient_samples': [],
                        'warnings': [str(e)]
                    }
                }
        
        @staticmethod
        def SetClassField(train_data_dict, class_field):
            """Set the class field for training data."""
            if train_data_dict and 'training_data' in train_data_dict:
                train_data_dict['class_field'] = class_field
            return train_data_dict
        
        @staticmethod
        def ValidClass(train_data_dict, use_class_ids=False):
            """Validate classes in training data."""
            if train_data_dict and train_data_dict.get('training_data') is not None:
                training_data = train_data_dict['training_data']
                class_field = train_data_dict.get('class_field', 'kelas')
                landcover_df = train_data_dict.get('landcover_df')
                
                logger.info(f"Validating classes with use_class_ids={use_class_ids}")
                logger.info(f"Class field: {class_field}")
                logger.info(f"Training data type: {type(training_data)}")
                
                if landcover_df is not None:
                    logger.info(f"Landcover DF columns: {landcover_df.columns.tolist()}")
                    if use_class_ids and 'ID' in landcover_df.columns:
                        logger.info(f"Valid IDs in landcover_df: {landcover_df['ID'].tolist()}")
                    elif 'LULC_Type' in landcover_df.columns:
                        logger.info(f"Valid LULC_Types in landcover_df: {landcover_df['LULC_Type'].tolist()}")
                
                valid_classes = []
                invalid_classes = []
                
                if isinstance(training_data, gpd.GeoDataFrame):
                    logger.info(f"Processing GeoDataFrame with {len(training_data)} features")
                    if class_field in training_data.columns:
                        classes = training_data[class_field].unique()
                        logger.info(f"Unique classes in training data: {classes}")
                        
                        for cls in classes:
                            if pd.isna(cls):
                                continue
                            if use_class_ids:
                                if cls in landcover_df['ID'].values:
                                    valid_classes.append(cls)
                                    logger.info(f"Valid class ID: {cls}")
                                else:
                                    invalid_classes.append(cls)
                                    logger.warning(f"Invalid class ID: {cls}")
                            else:
                                if cls in landcover_df['LULC_Type'].values:
                                    valid_classes.append(cls)
                                    logger.info(f"Valid class type: {cls}")
                                else:
                                    invalid_classes.append(cls)
                                    logger.warning(f"Invalid class type: {cls}")
                        
                        logger.info(f"Valid classes: {valid_classes}")
                        logger.info(f"Invalid classes: {invalid_classes}")
                        
                        filtered_data = training_data[training_data[class_field].isin(valid_classes)]
                        logger.info(f"Features after class validation: {len(filtered_data)}")
                        
                        train_data_dict['training_data'] = filtered_data
                        train_data_dict['validation_results']['points_after_class_filter'] = len(filtered_data)
                        train_data_dict['validation_results']['invalid_classes'] = invalid_classes
                    else:
                        logger.error(f"Class field '{class_field}' not found in training data columns: {training_data.columns.tolist()}")
                
                elif isinstance(training_data, ee.FeatureCollection):
                    logger.info("Processing Earth Engine FeatureCollection")
                    # Filter non-null first
                    non_null_fc = training_data.filter(ee.Filter.notNull([class_field]))
                    # Get distinct classes
                    classes = non_null_fc.aggregate_array(class_field).distinct().getInfo()
                    logger.info(f"Unique classes in EE FeatureCollection: {classes}")
                    
                    for cls in classes:
                        if use_class_ids:
                            if cls in landcover_df['ID'].values.tolist():
                                valid_classes.append(cls)
                            else:
                                invalid_classes.append(cls)
                        else:
                            if cls in landcover_df['LULC_Type'].values.tolist():
                                valid_classes.append(cls)
                            else:
                                invalid_classes.append(cls)
                    
                    logger.info(f"Valid classes: {valid_classes}")
                    logger.info(f"Invalid classes: {invalid_classes}")
                    
                    # Filter to valid classes
                    if valid_classes:
                        filter_valid = ee.Filter.inList(class_field, valid_classes)
                        filtered_fc = non_null_fc.filter(filter_valid)
                    else:
                        filtered_fc = ee.FeatureCollection([])
                    
                    filtered_count = filtered_fc.size().getInfo()
                    logger.info(f"Features after class validation: {filtered_count}")
                    
                    train_data_dict['training_data'] = filtered_fc
                    train_data_dict['validation_results']['points_after_class_filter'] = filtered_count
                    train_data_dict['validation_results']['invalid_classes'] = invalid_classes
            
            return train_data_dict
        
        @staticmethod
        def CheckSufficiency(train_data_dict, min_samples=20):
            """Check if there are sufficient samples per class."""
            if train_data_dict and train_data_dict.get('training_data') is not None:
                training_data = train_data_dict['training_data']
                class_field = train_data_dict.get('class_field', 'kelas')
                
                if class_field in training_data.columns:
                    class_counts = training_data[class_field].value_counts()
                    insufficient_classes = class_counts[class_counts < min_samples].index.tolist()
                    train_data_dict['validation_results']['insufficient_samples'] = insufficient_classes
            
            return train_data_dict
        
        @staticmethod
        def FilterTrainAoi(train_data_dict):
            """Filter training data by AOI."""
            if train_data_dict and train_data_dict.get('training_data') is not None:
                # Simplified AOI filtering
                training_data = train_data_dict['training_data']
                aoi_geometry = train_data_dict.get('aoi_geometry')
                
                if aoi_geometry is not None and hasattr(aoi_geometry, 'geometry'):
                    # Perform spatial filter (simplified)
                    try:
                        # Ensure both GeoDataFrames have the same CRS
                        if training_data.crs != aoi_geometry.crs:
                            logger.info(f"Reprojecting training data from {training_data.crs} to {aoi_geometry.crs}")
                            training_data = training_data.to_crs(aoi_geometry.crs)
                        
                        # Use 'intersects' instead of 'within' to catch polygons that overlap with AOI
                        # This is more appropriate for polygon training data
                        filtered_data = gpd.sjoin(training_data, aoi_geometry, how='inner', predicate='intersects')
                        
                        # Remove duplicate columns from the join (index_right, etc.)
                        cols_to_keep = [col for col in filtered_data.columns if not col.startswith('index_')]
                        if 'index_right' in filtered_data.columns:
                            cols_to_keep = [col for col in cols_to_keep if col != 'index_right']
                        filtered_data = filtered_data[cols_to_keep]
                        
                        # Remove duplicate rows that may result from multiple AOI polygons
                        filtered_data = filtered_data.drop_duplicates()
                        
                        logger.info(f"AOI filtering: {len(training_data)} -> {len(filtered_data)} features")
                        
                        train_data_dict['training_data'] = filtered_data
                        train_data_dict['validation_results']['valid_points'] = len(filtered_data)
                    except Exception as e:
                        logger.warning(f"AOI filtering failed: {str(e)}")
                        # If filtering fails, keep original data
                        train_data_dict['validation_results']['valid_points'] = len(training_data)
                        train_data_dict['validation_results']['warnings'].append(f"AOI filtering skipped: {str(e)}")
            
            return train_data_dict
        
        @staticmethod
        def TrainDataRaw(training_data, landcover_df, class_field):
            """Create raw training data summary."""
            if training_data is None or training_data.empty:
                return pd.DataFrame(), 0, pd.DataFrame()
            
            try:
                # Create summary table
                summary_data = []
                total_samples = len(training_data)
                
                if class_field in training_data.columns:
                    class_counts = training_data[class_field].value_counts()
                    
                    # Create mapping from ID to LULC_Type
                    id_to_lulc_type = {}
                    if landcover_df is not None and 'ID' in landcover_df.columns and 'LULC_Type' in landcover_df.columns:
                        id_to_lulc_type = dict(zip(landcover_df['ID'], landcover_df['LULC_Type']))
                    
                    for class_id, count in class_counts.items():
                        percentage = (count / total_samples * 100) if total_samples > 0 else 0
                        
                        # Map ID to LULC_Type name if available
                        if class_id in id_to_lulc_type:
                            lulc_class_name = id_to_lulc_type[class_id]
                        else:
                            lulc_class_name = str(class_id)  # Fallback to ID if mapping not found
                        
                        summary_data.append({
                            'ID': class_id,
                            'LULC_class': lulc_class_name,
                            'Sample_Count': count,
                            'Percentage': percentage
                        })
                
                summary_df = pd.DataFrame(summary_data)
                
                # Create insufficient samples table
                insufficient_data = []
                for _, row in summary_df.iterrows():
                    if row['Sample_Count'] < 20:
                        insufficient_data.append({
                            'ID': row['ID'],
                            'LULC_class': row['LULC_class'],
                            'Sample_Count': row['Sample_Count'],
                            'Needed': 20 - row['Sample_Count'],
                            'Percentage': row['Percentage'],
                            'Status': 'Insufficient' if row['Sample_Count'] > 0 else 'No Samples'
                        })
                
                insufficient_df = pd.DataFrame(insufficient_data)
                
                return summary_df, total_samples, insufficient_df
                
            except Exception as e:
                logger.error(f"Error creating training data summary: {str(e)}")
                return pd.DataFrame(), 0, pd.DataFrame()

        @staticmethod
        def LoadMultitemporalTrainData(
            training_sources: Dict[str, Any],
            landcover_df: Optional[pd.DataFrame],
            aoi_geometry: Any = None,
            period_order: Optional[List[str]] = None,
            reference_period: str = 'baseline',
            class_field: Optional[str] = None,
            use_class_ids: bool = True,
            min_samples: int = 20,
        ) -> Dict[str, Any]:
            """
            Load and validate training data across multitemporal periods against a
            shared classification scheme, routing the workflow to either:

            - ``'model_transfer'`` when only ``reference_period`` has training data
            - ``'batch'`` when every period in ``period_order`` has training data

            Parameters
            ----------
            training_sources : dict[str, Any]
                Mapping ``{period_label: source}`` where each source may be:
                - ``str`` path to a shapefile (`.shp`) or Earth Engine FeatureCollection asset (`users/` / `projects/`),
                - ``geopandas.GeoDataFrame``,
                - ``ee.FeatureCollection``, or
                - ``None`` (no training data for that period).
            landcover_df : pandas.DataFrame or None
                Shared classification scheme DataFrame containing `'ID'` (when ``use_class_ids=True``)
                and/or `'LULC_Type'` / `'Class Name'`.
            aoi_geometry : Any, optional
                Area of interest (`GeoDataFrame`, Shapely geometry, or `ee.Geometry`).
            period_order : list[str], optional
                Ordered list of all mapping period labels. Defaults to ``list(training_sources.keys())``.
            reference_period : str, default='baseline'
                Reference period label used as the training source in Model Transfer mode.
            class_field : str, optional
                Column/property containing class labels. Auto-detects `'class_id'` or `'kelas'`
                for GeoDataFrames when omitted, defaulting to `'kelas'`.
            use_class_ids : bool, default=True
                Validate class labels against ``landcover_df['ID']`` when True.
            min_samples : int, default=20
                Minimum sample count per class for sufficiency warnings.

            Returns
            -------
            dict
                ``{'mode', 'reference_period', 'period_order', 'provided_periods',
                'training_by_period', 'validation_summary'}``.
            """
            if not isinstance(training_sources, dict) or not training_sources:
                raise ValueError("training_sources must be a non-empty dictionary keyed by period label.")

            if period_order is None:
                period_order = list(training_sources.keys())
            else:
                period_order = list(period_order)

            if len(period_order) < 2:
                raise ValueError(
                    f"Multitemporal training validation requires at least 2 periods, got {len(period_order)}."
                )

            unknown_keys = [k for k in training_sources.keys() if k not in period_order]
            if unknown_keys:
                raise ValueError(
                    f"training_sources contains unknown period label(s) {unknown_keys}; "
                    f"expected subset of {period_order}."
                )

            if reference_period not in period_order:
                raise ValueError(
                    f"reference_period '{reference_period}' is not in period_order {period_order}."
                )

            def _is_non_empty_source(src: Any) -> bool:
                if src is None:
                    return False
                if isinstance(src, gpd.GeoDataFrame):
                    return not src.empty
                if isinstance(src, str):
                    return bool(src.strip())
                return True

            provided_periods = [
                p for p in period_order if _is_non_empty_source(training_sources.get(p))
            ]

            if len(provided_periods) == 0:
                raise ValueError(
                    f"No training data provided. Supply training data for '{reference_period}' "
                    "(Model Transfer) or for all periods (Batch)."
                )
            elif len(provided_periods) == 1:
                if provided_periods[0] != reference_period:
                    raise ValueError(
                        f"Single training dataset must belong to reference_period '{reference_period}', "
                        f"but was provided for '{provided_periods[0]}'."
                    )
                mode = 'model_transfer'
            elif len(provided_periods) == len(period_order):
                mode = 'batch'
            else:
                missing = [p for p in period_order if p not in provided_periods]
                raise ValueError(
                    f"Partial training data provided for {provided_periods}; missing periods {missing}. "
                    f"Provide training data for '{reference_period}' only (Model Transfer) or all periods (Batch)."
                )

            training_by_period: Dict[str, Dict[str, Any]] = {}
            validation_summary: Dict[str, Dict[str, Any]] = {}

            for label in provided_periods:
                src = training_sources[label]
                if isinstance(src, str):
                    is_ee_asset = src.startswith(('users/', 'projects/'))
                    train_dict = SyncTrainData.LoadTrainData(
                        landcover_df=landcover_df,
                        aoi_geometry=aoi_geometry,
                        training_shp_path=None if is_ee_asset else src,
                        training_ee_path=src if is_ee_asset else None,
                    )
                elif isinstance(src, gpd.GeoDataFrame):
                    gdf_copy = src.copy()
                    if gdf_copy.crs is None:
                        gdf_copy = gdf_copy.set_crs('EPSG:4326')
                    elif str(gdf_copy.crs).upper() != 'EPSG:4326':
                        gdf_copy = gdf_copy.to_crs('EPSG:4326')
                    detected_field = class_field
                    if detected_field is None:
                        if 'class_id' in gdf_copy.columns:
                            detected_field = 'class_id'
                        elif 'kelas' in gdf_copy.columns:
                            detected_field = 'kelas'
                        else:
                            detected_field = 'kelas'
                    train_dict = {
                        'training_data': gdf_copy,
                        'landcover_df': landcover_df,
                        'class_field': detected_field,
                        'validation_results': {
                            'total_points': len(gdf_copy),
                            'valid_points': len(gdf_copy),
                            'points_after_class_filter': len(gdf_copy),
                            'invalid_classes': [],
                            'outside_aoi': [],
                            'insufficient_samples': [],
                            'warnings': [],
                        },
                    }
                elif isinstance(src, ee.FeatureCollection):
                    fc = src
                    # Filter by AOI bounds if AOI is provided
                    if aoi_geometry is not None:
                        if isinstance(aoi_geometry, (ee.Geometry, ee.FeatureCollection)):
                            fc = fc.filterBounds(aoi_geometry)
                        elif hasattr(aoi_geometry, 'geometry') and hasattr(aoi_geometry.geometry, 'iloc'):
                            fc = fc.filterBounds(ee.Geometry(aoi_geometry.geometry.iloc[0].__geo_interface__))
                        elif hasattr(aoi_geometry, '__geo_interface__'):
                            fc = fc.filterBounds(ee.Geometry(aoi_geometry.__geo_interface__))

                    props = list(fc.first().propertyNames().getInfo() or [])
                    if class_field is not None:
                        if class_field not in props:
                            raise ValueError(
                                f"class_field '{class_field}' not found in EE FeatureCollection for period '{label}'. "
                                f"Available properties: {props}"
                            )
                        resolved_field = class_field
                    else:
                        resolved_field = next(
                            (c for c in ('class_id', 'ID_Kelas', 'kelas', 'ID') if c in props),
                            'class_id',
                        )

                    fc_size = fc.size().getInfo()
                    train_dict = {
                        'training_data': fc,
                        'landcover_df': landcover_df,
                        'class_field': resolved_field,
                        'validation_results': {
                            'total_points': fc_size,
                            'valid_points': fc_size,
                            'points_after_class_filter': fc_size,
                            'invalid_classes': [],
                            'outside_aoi': [],
                            'insufficient_samples': [],
                            'warnings': [],
                        },
                    }
                else:
                    raise ValueError(
                        f"Unsupported training source type for period '{label}': {type(src).__name__}"
                    )

                if class_field is not None:
                    train_dict = SyncTrainData.SetClassField(train_dict, class_field)

                if aoi_geometry is not None:
                    train_dict['aoi_geometry'] = aoi_geometry
                    if isinstance(train_dict.get('training_data'), gpd.GeoDataFrame) and isinstance(
                        aoi_geometry, gpd.GeoDataFrame
                    ):
                        train_dict = SyncTrainData.FilterTrainAoi(train_dict)

                if landcover_df is not None:
                    train_dict = SyncTrainData.ValidClass(train_dict, use_class_ids=use_class_ids)

                # CheckSufficiency expects DataFrame-like .columns; handle FeatureCollection safely
                td_obj = train_dict.get('training_data')
                if isinstance(td_obj, gpd.GeoDataFrame):
                    train_dict = SyncTrainData.CheckSufficiency(train_dict, min_samples=min_samples)
                elif isinstance(td_obj, ee.FeatureCollection):
                    cf = train_dict.get('class_field', 'class_id')
                    hist = td_obj.aggregate_histogram(cf).getInfo() or {}
                    insufficient = [
                        k for k, v in hist.items() if int(v) < min_samples
                    ]
                    train_dict['validation_results']['insufficient_samples'] = insufficient

                training_by_period[label] = train_dict
                validation_summary[label] = train_dict['validation_results']

            return {
                'mode': mode,
                'reference_period': reference_period,
                'period_order': period_order,
                'provided_periods': provided_periods,
                'training_by_period': training_by_period,
                'validation_summary': validation_summary,
            }
    
    class SplitTrainData:
        """Data splitting functionality."""
        
        @staticmethod
        def SplitProcess(train_data, TrainSplitPct=0.7, random_state=123):
            """Split training data into train and validation sets."""
            try:
                if train_data is None or train_data.empty:
                    return gpd.GeoDataFrame(), gpd.GeoDataFrame()
                
                # Simple train/validation split
                from sklearn.model_selection import train_test_split
                
                if hasattr(train_data, 'index'):
                    # For GeoDataFrames, split by index to maintain spatial relationships
                    train_indices, val_indices = train_test_split(
                        train_data.index, 
                        train_size=TrainSplitPct, 
                        random_state=random_state,
                        stratify=train_data.get('kelas', None) if 'kelas' in train_data.columns else None
                    )
                    train_split = train_data.loc[train_indices].copy()
                    val_split = train_data.loc[val_indices].copy()
                    return train_split, val_split
                else:
                    return gpd.GeoDataFrame(), gpd.GeoDataFrame()
                
            except Exception as e:
                logger.error(f"Error splitting training data: {str(e)}")
                return gpd.GeoDataFrame(), gpd.GeoDataFrame()

        @staticmethod
        def SplitMultitemporalProcess(
            training_by_period: Dict[str, Any],
            TrainSplitPct: float = 0.7,
            random_state: int = 123,
            class_field: Optional[str] = None,
        ) -> Dict[str, Tuple[Any, Any]]:
            """
            Split training sets (``geopandas.GeoDataFrame`` or ``ee.FeatureCollection``)
            across one or more periods into ``(train_split, val_split)`` pairs.
            """
            splits: Dict[str, Tuple[Any, Any]] = {}
            for label, entry in training_by_period.items():
                if isinstance(entry, dict) and 'training_data' in entry:
                    data = entry['training_data']
                    cf = class_field or entry.get('class_field', 'kelas')
                else:
                    data = entry
                    cf = class_field or 'kelas'

                if data is None:
                    splits[label] = (gpd.GeoDataFrame(), gpd.GeoDataFrame())
                    continue

                if isinstance(data, ee.FeatureCollection):
                    classes = data.aggregate_array(cf).distinct()

                    def _make_split_fn(fc, prop, ratio, seed):
                        # EE CustomFunction.create inspects co_argcount; keep callback unary.
                        def _split_class(c):
                            subset = fc.filter(ee.Filter.eq(prop, c)).randomColumn('random', seed=seed)
                            tr = subset.filter(ee.Filter.lt('random', ratio)).map(
                                lambda f: f.set('fraction', 'training')
                            )
                            te = subset.filter(ee.Filter.gte('random', ratio)).map(
                                lambda f: f.set('fraction', 'testing')
                            )
                            return tr.merge(te)

                        return _split_class

                    split_fc = ee.FeatureCollection(
                        classes.map(_make_split_fn(data, cf, TrainSplitPct, random_state))
                    ).flatten()
                    train_fc = split_fc.filter(ee.Filter.eq('fraction', 'training'))
                    test_fc = split_fc.filter(ee.Filter.eq('fraction', 'testing'))
                    splits[label] = (train_fc, test_fc)
                    continue

                if not isinstance(data, gpd.GeoDataFrame) or data.empty:
                    splits[label] = (gpd.GeoDataFrame(), gpd.GeoDataFrame())
                    continue

                from sklearn.model_selection import train_test_split
                strat_col = data[cf] if cf in data.columns else None
                try:
                    train_idx, val_idx = train_test_split(
                        data.index,
                        train_size=TrainSplitPct,
                        random_state=random_state,
                        stratify=strat_col,
                    )
                except ValueError:
                    # Fallback without stratification if a rare class has 1 sample
                    train_idx, val_idx = train_test_split(
                        data.index,
                        train_size=TrainSplitPct,
                        random_state=random_state,
                    )
                splits[label] = (data.loc[train_idx].copy(), data.loc[val_idx].copy())
            return splits

except ImportError as e:
    logger.warning(f"Some functionality not available: {str(e)}")
    
    class SyncTrainData:
        @staticmethod
        def LoadTrainData(*args, **kwargs):
            raise NotImplementedError("SyncTrainData not available")
        
        @staticmethod
        def SetClassField(*args, **kwargs):
            raise NotImplementedError("SyncTrainData not available")
        
        @staticmethod
        def ValidClass(*args, **kwargs):
            raise NotImplementedError("SyncTrainData not available")
        
        @staticmethod
        def CheckSufficiency(*args, **kwargs):
            raise NotImplementedError("SyncTrainData not available")
        
        @staticmethod
        def FilterTrainAoi(*args, **kwargs):
            raise NotImplementedError("SyncTrainData not available")
        
        @staticmethod
        def TrainDataRaw(*args, **kwargs):
            raise NotImplementedError("SyncTrainData not available")
    
    class SplitTrainData:
        @staticmethod
        def SplitProcess(data, TrainSplitPct=0.7, random_state=123):
            """Simple split functionality."""
            try:
                if data is None or data.empty:
                    return gpd.GeoDataFrame(), gpd.GeoDataFrame()
                
                from sklearn.model_selection import train_test_split
                
                # Simple train/validation split
                train_indices, val_indices = train_test_split(
                    data.index, 
                    train_size=TrainSplitPct, 
                    random_state=random_state
                )
                train_split = data.loc[train_indices].copy()
                val_split = data.loc[val_indices].copy()
                return train_split, val_split
            except Exception as e:
                logger.error(f"Error in split: {str(e)}")
                return gpd.GeoDataFrame(), gpd.GeoDataFrame()