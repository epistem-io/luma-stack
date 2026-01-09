"""
Module 3 Backend - Training Data Management

This module provides training data loading and processing functionality.
"""

import streamlit as st
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