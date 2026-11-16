"""
Module 4 GE - Sample analysis via Separability and cluster analysis

This module provides training data loading and processing functionality.
"""
import ee
import pandas as pd
import numpy as np
from .ee_config import ensure_ee_initialized
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import plotly.graph_objects as go
import plotly.express as px
from .ee_config import ensure_ee_initialized

# Do not initialize Earth Engine at import time. Initialize when classes are instantiated.
# Module 4: Region of Interest Separability Analysis
## System Response 4.1 Separability Analysis
class sample_quality:
    """
    methods which contain several functions used to conduct training data/sample analysis 
    
    """
    def __init__(self, training_data, image, class_property, region, class_name_property=None):
        """
        Initialize the tools for conducting the analysis
        Ensure Earth Engine is initialized lazily (avoids import-time failures).
        Args:
            training_data: ee.FeatureCollection - Training polygons/points
            image: ee.Image - The image to extract spectral values
            class_property: str - Property/columen containing class ID (unique)
            region: ee.Geometry - Optional region to limit analysis
            class_name_property: str - Property/column containing class name
        """
        # Ensure Earth Engine is initialized when first used (raises helpful error if not)
        ensure_ee_initialized()
        
        self.training_data = training_data
        self.image = image
        self.class_property = class_property
        self.class_name_property = class_name_property
        self.region = region
        self.band_names = self.image.bandNames().getInfo()
        self.class_mapping = None
    def get_display_property(self):
        """
        Helper functions to determine which properties (column in the training data) to display
        """
        return self.class_name_property if self.class_name_property else self.class_property
    def class_renaming(self):
        """
        Function to mapped between class ID and class names
        """
        if self.class_mapping is None and self.class_name_property:
            try:
                #mapped the combination between ID and names
                pairs = self.training_data.distinct([self.class_property, self.class_name_property])
                pair_info = pairs.getInfo()
                mapping = {}
                for features in pair_info['features']:
                    prop = features['properties']
                    class_id = prop[self.class_property]
                    class_name = prop[self.class_name_property]
                    # Convert float to int for consistent mapping keys
                    if isinstance(class_id, (int, float)):
                        mapping[int(class_id)] = class_name
                    else:
                        mapping[class_id] = class_name
                return mapping
            except Exception as e:
                print(f"Warning: Could not create class mapping: {e}")
                return {}
        return self.class_mapping or {}
    def add_class_names(self, df):
        """
        Add class names to dataframe based on class ids
        """ 
        id_column = self.class_property
        class_name = self.class_name_property  # Add this line
        mapping = self.class_renaming()
        if mapping:
            df[class_name] = df[id_column].astype(float).astype(int).map(lambda x: mapping.get(x, f"Class {x}"))
            cols = df.columns.tolist()
            if class_name in cols and id_column in cols:  
                id_idx = cols.index(id_column)
                cols.remove(class_name)  
                cols.insert(id_idx + 1, class_name) 
                df = df[cols]
        return df

    #Basic statistic of the training data:
    #Note that this process applied to training data before pixel value is extracted
    def sample_stats(self):
        """
        Get basic statistics about the training dataset. 
        """
        try:
            # Perfom in server side to minimize computational load
            class_counts = self.training_data.aggregate_histogram(self.class_property)
            total_samples = self.training_data.size()
            unique_classes = self.training_data.aggregate_array(self.class_property).distinct()
            # Get class names if available
            class_names = None
            if self.class_name_property:
                class_names = self.training_data.aggregate_histogram(self.class_name_property)            
            #return the dictionary
            results = ee.Dictionary({
                'class_counts': class_counts,
                'total_samples': total_samples,
                'unique_classes': unique_classes
            }).getInfo()
            if class_names:
                results['class_names'] = class_names            
            # Process results client-side
            class_counts_dict = results['class_counts']
            total_count = results['total_samples']
            classes_list = results['unique_classes']
            stats_dict = {
                'total_samples': total_count,
                'num_classes': len(classes_list),
                'class_counts': class_counts_dict,
                'classes': classes_list,
                'class_balance': {str(k): v/total_count for k, v in class_counts_dict.items()}
            }
            # Add class names mapping if available
            if 'class_names' in results:
                stats_dict['class_names'] = results['class_names']            
            return stats_dict
            
        except Exception as e:
            print(f"Error in get_basic_statistics: {e}")
            return None
    def get_sample_stats_df(self):
        """
        Get sample statistics as a formatted DataFrame
        """
        stats = self.sample_stats()
        if not stats:
            return pd.DataFrame()
        
        # Create DataFrame from class counts
        stats_data = []
        for class_id, count in stats['class_counts'].items():
            proportion = stats['class_balance'][str(class_id)]
            stats_data.append({
                'Class_ID': class_id,
                'Sample_Count': count,
                'Proportion': proportion,
                'Percentage': proportion * 100
            })
        
        df = pd.DataFrame(stats_data)
        df = df.sort_values('Class_ID').reset_index(drop=True)
        df = df.rename(columns={'Class_ID': self.class_property})
        # Add class names if available
        df = self.add_class_names(df)
        
        # Format percentage column
        df['Percentage'] = df['Percentage'].round(2)
        df['Proportion'] = df['Proportion'].round(4)
        
        return df
    #This function extract the pixel for all of the samples. 
    def extract_spectral_values(self, scale=30, max_pixels_per_class=7000):
        """
        Extract spectral values for all training samples. 
        
        Args:
            scale: int - Scale for sampling (meters)
            max_pixels_per_class: int - Maximum pixels per class to prevent memory issues
            
        Returns:
            pandas.DataFrame - Spectral values with class labels
        """
        try:
            # Limit Sample Size to prevent computational time out
            def limit_samples_per_class(class_value):
                class_samples = self.training_data.filter(
                    ee.Filter.eq(self.class_property, class_value)
                )
                # Use ee.Algorithms.If for server-side conditional logic
                sample_count = class_samples.size()
                limited_samples = ee.Algorithms.If(
                    sample_count.gt(max_pixels_per_class),
                    class_samples.randomColumn('random', 42).sort('random').limit(max_pixels_per_class),
                    class_samples
                )
                return ee.FeatureCollection(limited_samples)
            
            # Get unique classes
            unique_classes = self.training_data.aggregate_array(self.class_property).distinct()
            # Create balanced sample
            balanced_samples = ee.FeatureCollection(
                unique_classes.map(limit_samples_per_class)
            ).flatten()
            # Define properties to extract
            properties_to_extract = [self.class_property]
            if self.class_name_property:
                properties_to_extract.append(self.class_name_property)            
            # Extract the Pixel Value based on the training data
            training_sample = self.image.sampleRegions(
                collection=balanced_samples,
                properties=properties_to_extract,
                scale=scale,
                geometries=False,
                tileScale=16  # Increase tile scale to handle larger computations
            )
            # Single getInfo() call to get all data
            sample_data = training_sample.getInfo()
            if 'features' not in sample_data or len(sample_data['features']) == 0:
                print("Warning: No spectral data extracted. Check your training data quantity. Increase pixel size to reduce the size of sample")
                return pd.DataFrame()
            # Convert to panda
            df = pd.DataFrame([feat['properties'] for feat in sample_data['features']])
            
            # Data Cleaning
            # only keep the resulting 
            spectral_columns = [col for col in df.columns
                                if col !=self.class_property and col in self.band_names]
            # Convert only spectral columns to numeric
            for col in spectral_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')            
            df = df.dropna(subset=spectral_columns, how='all')
            print(f"Extracted spectral values for {len(df)} samples across {df[self.class_property].nunique()} classes")  
            return df
            
        except Exception as e:
            print(f"Error in extract_spectral_values: {e}")
            return pd.DataFrame() 
   #Calculate statistics of the training data after pixel value extraction               
    def sample_pixel_stats (self, df):
        """
        Compute statistics for sample extracted from the satellite imagery
        """
         #If the panda dataframe from previous step failed
        #if df.empty:
         #  return {}
         #Define the properties which include band pixel value   
        spectral_bands = [col for col in df.columns if col != self.class_property and col in self.band_names]
        classes = df[self.class_property].unique()
        properties_to_exclude = [self.class_property]
         #Define the properties which include band pixel value   
        if self.class_name_property:
            properties_to_exclude.append(self.class_name_property)        
        stats_byclass = {}
        #mapped them accross the class
        for class_name in classes:
            class_data = df[df[self.class_property] == class_name]
            stats_class ={}
            for band in spectral_bands:
                if band in class_data.columns:
                    band_values = class_data[band].dropna()
                    if len(band_values) > 0:
                        stats_class[band] = {
                            'mean': float(band_values.mean()),
                            'std' : float(band_values.std()),
                            'min': float(band_values.min()),
                            'max': float(band_values.max()),
                            'median': float(band_values.median()),
                            'count': int(len(band_values))
                        }
            stats_byclass[str(class_name)] = stats_class
        return stats_byclass
    def get_sample_pixel_stats_df(self, df):
        """
        Get pixel statistics as a formatted DataFrame
        Parameters:
        -----------
        df : pandas.DataFrame
            The dataframe from extract_spectral_values
        statistic : str
            Which statistic to display ('mean', 'std', 'min', 'max', 'median', 'count')
        Returns:
        --------
        pandas.DataFrame
            Formatted statistics with class names if available
        """
        stats = self.sample_pixel_stats(df)
        if not stats:
            return pd.DataFrame()
        
        # Convert to long format DataFrame
        stats_data = []
        for class_id, bands in stats.items():
            for band, band_stats in bands.items():
                stats_data.append({
                    'Class_ID': int(class_id),
                    'Band': band,
                    'Mean': round(band_stats['mean'], 2),
                    'Std': round(band_stats['std'], 2),
                    'Min': round(band_stats['min'], 2),
                    'Max': round(band_stats['max'], 2),
                    'Median': round(band_stats['median'], 2),
                    'Count': band_stats['count']
                })
        
        result_df = pd.DataFrame(stats_data)
        result_df = result_df.sort_values(['Class_ID', 'Band']).reset_index(drop=True)
        result_df = result_df.rename(columns={'Class_ID': self.class_property})
        
        # Add class names if available
        result_df = self.add_class_names(result_df)
        
        return result_df
    #Calculate class separability using jeffries matusita distance
    #Note: This approach is not ideal for machine learning classifier, since JM distance works best with gaussian distribution
    def check_class_separability(self, df, method='TD'):
        """
        Calculate class separability using Transformed Divergence.
        Method: 'TD' (Transformed Divergence) - hardcoded for consistency
        """
        spectral_bands = [col for col in df.columns if col != self.class_property and col in self.band_names]
        classes = df[self.class_property].unique()

        if len(classes) < 2:
            print("Warning: Required at least 2 class for separability analysis")
            return {}

        separability_matrix = {}

        for i, class1 in enumerate(classes):
            separability_matrix[str(class1)] = {}
            for j, class2 in enumerate(classes):
                if i != j:
                    try:
                        data1 = df[df[self.class_property] == class1][spectral_bands].dropna().values
                        data2 = df[df[self.class_property] == class2][spectral_bands].dropna().values

                        if len(data1) == 0 or len(data2) == 0:
                            separability_matrix[str(class1)][str(class2)] = 0.0
                            continue

                        # Always use Transformed Divergence
                        sep = self.transform_divergence(data1, data2)

                        separability_matrix[str(class1)][str(class2)] = float(sep)

                    except Exception as e:
                        print(f"Error calculating separability for classes {class1}-{class2}: {e}")
                        separability_matrix[str(class1)][str(class2)] = 0.0
                else:
                    separability_matrix[str(class1)][str(class2)] = 0.0

        return separability_matrix

    def get_separability_df(self, df, method='TD'):
        """
        Get separability results as a DataFrame.
        method: 'TD' (Transformed Divergence) - hardcoded for consistency
        """
        separability = self.check_class_separability(df, method=method)
        if not separability:
            return pd.DataFrame()

        pairs = []
        class_mapping = self.class_renaming()
        metric_name = "TD_Distance"  #Always use Transformed Divergence. Fyi JM avaliable in source code
        processed_pairs = set()  # Track processed pairs to avoid duplicates

        for class1_id, class1_data in separability.items():
            for class2_id, value in class1_data.items():
                if value > 0:
                    # Create a sorted tuple to ensure unique pairs (avoid Class1-Class2 and Class2-Class1)
                    pair_key = tuple(sorted([class1_id, class2_id]))
                    
                    #Skip if pair is already process
                    if pair_key in processed_pairs:
                        continue
                    
                    processed_pairs.add(pair_key)
                    
                    # Use consistent key type for mapping
                    try:
                        class1_key = int(class1_id)
                    except ValueError:
                        class1_key = class1_id
                    try:
                        class2_key = int(class2_id)
                    except ValueError:
                        class2_key = class2_id

                    class1_name = class_mapping.get(class1_key, f"Class {class1_id}")
                    class2_name = class_mapping.get(class2_key, f"Class {class2_id}")
                    pairs.append({
                        'Class1_ID': class1_id,
                        'Class1_Name': class1_name,
                        'Class2_ID': class2_id,
                        'Class2_Name': class2_name,
                        metric_name: round(value, 3),
                        'Separability_Level': self.separability_level(value)
                    })

        pairs_df = pd.DataFrame(pairs)
        if not pairs_df.empty:
            pairs_df = pairs_df.sort_values(metric_name).reset_index(drop=True)
        return pairs_df

    def lowest_separability(self, df, top_n=10, method='TD'):
        """
        Get the class pairs with the lowest separability value.
        Args:
            df: DataFrame with spectral values
            top_n: Number of lowest pairs to return
            method: 'TD' (Transformed Divergence) - hardcoded
        Returns:
            DataFrame of lowest separability pairs with interpretation
        """
        separability_df = self.get_separability_df(df, method=method)
        if separability_df.empty:
            return pd.DataFrame()
        metric_name = "TD_Distance"  # Always use Transformed Divergence
        lowest_pairs = separability_df.head(top_n).copy()
        lowest_pairs['Interpretation'] = lowest_pairs[metric_name].apply(
            lambda x: self.separability_level(x, method=method)
        )
        return lowest_pairs

    def separability_level(self, value, method='TD'):
        """
        Categorize separability value for TD (Transformed Divergence) distance.
        Args:
            value: Separability metric value
            method: 'TD' (Transformed Divergence) - now hardcoded
        Returns:
            String interpretation with detailed explanation
        """
        # Thresholds for TD (range 0–2)
        if value >= 1.8:
            return "🟢 Good Separability (TD ≥ 1.8)"
        elif 1.0 <= value < 1.8:
            return "🟡 Weak/Marginal Separability (1.0 ≤ TD < 1.8)"
        else:
            return "🔴 Poor Separability (TD < 1.0)"
    #Summarize the separability
    def sum_separability(self, df):
        """
        summarize the separability result
        """
        sum_df = self.get_separability_df(df)
        if sum_df.empty:
            return pd.DataFrame()
        
        #Count each separability level for summary report
        good_count = len(sum_df[sum_df["Separability_Level"].str.contains("Good Separability", na=False)])
        weak_count = len(sum_df[sum_df["Separability_Level"].str.contains("Weak/Marginal Separability", na=False)])
        poor_count = len(sum_df[sum_df["Separability_Level"].str.contains("Poor Separability", na=False)])
        
        summary = {
            'Total Pairs': len(sum_df),
            'Good Separability Pairs': good_count,
            'Weak Separability Pairs': weak_count,
            'Poor Separability Pairs': poor_count
        }
        return pd.DataFrame(summary, index=[0])

    #Core function to calculate JM distance and Transformed Divergence
    def _jeffries_matusita_distance(self, class1_data, class2_data):
        """Calculate Jeffries-Matusita distance between two classes"""
        try:
            if len(class1_data) < 2 or len(class2_data) < 2:
                return 0.0
                
            mean1 = np.mean(class1_data, axis=0)
            mean2 = np.mean(class2_data, axis=0)
            
            # Handle single band case
            if class1_data.shape[1] == 1:
                var1 = np.var(class1_data)
                var2 = np.var(class2_data)
                cov1 = np.array([[var1]]) if var1 > 0 else np.array([[1e-6]])
                cov2 = np.array([[var2]]) if var2 > 0 else np.array([[1e-6]])
            else:
                cov1 = np.cov(class1_data.T)
                cov2 = np.cov(class2_data.T)
                
                # Ensure covariance matrices are 2D
                if cov1.ndim == 0:
                    cov1 = np.array([[cov1]])
                if cov2.ndim == 0:
                    cov2 = np.array([[cov2]])
            
            # Add regularization for numerical stability
            reg_factor = 1e-6
            cov1 += np.eye(cov1.shape[0]) * reg_factor
            cov2 += np.eye(cov2.shape[0]) * reg_factor
            
            cov_mean = (cov1 + cov2) / 2
            diff_mean = mean1 - mean2
            
            # Calculate Bhattacharyya distance
            try:
                term1 = 0.125 * np.dot(np.dot(diff_mean.T, np.linalg.inv(cov_mean)), diff_mean)
            except np.linalg.LinAlgError:
                term1 = 0.125 * np.dot(np.dot(diff_mean.T, np.linalg.pinv(cov_mean)), diff_mean)
            
            try:
                det_cov_mean = np.linalg.det(cov_mean)
                det_cov1 = np.linalg.det(cov1)
                det_cov2 = np.linalg.det(cov2)
                
                if det_cov_mean <= 0 or det_cov1 <= 0 or det_cov2 <= 0:
                    term2 = 0.0
                else:
                    term2 = 0.5 * np.log(det_cov_mean / np.sqrt(det_cov1 * det_cov2))
            except:
                term2 = 0.0
            
            bhatt_dist = term1 + term2
            
            #Calculate Jeffries Matusita Distance
            jm_dist = 2 * (1 - np.exp(-bhatt_dist))
            
            return float(np.clip(jm_dist, 0, 2))
            
        except Exception as e:
            print(f"Error in JM distance calculation: {e}")
            return 0.0
        
    def transform_divergence(self, class1_data, class2_data):
        """
        Calculate Transformed Divergence between two classes.
        
        Transformed Divergence (TD) is a statistical measure that quantifies the separability
        between two spectral classes by analyzing differences in their mean values and 
        covariance structures. It ranges from 0 to 2, where:
        
        - TD ≥ 1.8: Good separability (classes are well-separated)
        - 1.0 ≤ TD < 1.8: Weak/marginal separability (some overlap exists)
        - TD < 1.0: Poor separability (significant overlap, high misclassification risk)
        
        The method considers both the distance between class centers and the spread
        (variance) of each class in multidimensional spectral space.
        """
        try:
            #Compute mean and covariance
            mean1 = np.mean(class1_data, axis=0)
            mean2 = np.mean(class2_data, axis=0)
            cov1 = np.cov(class1_data.T) + np.eye(class1_data.shape[1]) * 1e-6
            cov2 = np.cov(class2_data.T) + np.eye(class2_data.shape[1]) * 1e-6
            diff_mean  = mean1 - mean2
            #inverse covariance matrix
            try:
                inv_cov1 = np.linalg.inv(cov1)
            except np.linalg.LinAlgError:
                inv_cov1 = np.linalg.pinv(cov1)
            try:
                inv_cov2 = np.linalg.inv(cov2)
            except np.linalg.LinAlgError:
                inv_cov2 = np.linalg.pinv(cov2)
            #compute Divergence
            term1 = 0.5 * np.trace((cov1 - cov2) @ (inv_cov2 - inv_cov1))
            term2 = 0.5 * diff_mean.T @ (inv_cov1 + inv_cov2) @ diff_mean
            divergence = term1 + term2
            #Apply transformation
            td = 2 * (1 - np.exp(-divergence/8))
            return float(np.clip(td, 0, 2))
        except Exception as e:
            print(f"Error in Transform Divergence calculation: {e}")
            return 0.0
    #Plotting data distribution for evaluation
    #Data Histogram

    def print_analysis_summary(self, df):
        """
        Print a comprehensive analysis summary with all key results
        """
        # Class Separability
        print("\n\n1. Class Separability Summary:")
        print("-" * 40)
        sep_summary = self.sum_separability(df)
        if not sep_summary.empty:
            print(sep_summary.to_string(index=False))
        
        # 4. Most Problematic Class Pairs
        print("\n\n2. Lowest Separability Between Class:")
        print("-" * 40)
        lowest_pairs = self.lowest_separability(df, top_n=10)
        if not lowest_pairs.empty:
            display_cols = ['Class1_Name', 'Class2_Name', 'TD_Distance', 'Separability_Level', 'Interpretation']
            print(lowest_pairs[display_cols].to_string(index=False))
        
        print("\n" + "="*80)

    @staticmethod
    def run_multitemporal_separability(
        composites_by_period,
        training_by_period,
        class_property='class_id',
        region=None,
        class_name_property='class_name',
        scale=30,
        max_pixels_per_class=5000,
        method='TD',
        period_order=None,
    ):
        """
        Run separability analysis across multitemporal periods.

        Only periods present with non-null training data in ``training_by_period``
        are analyzed (e.g., only ``reference_period`` in Model Transfer mode, or all
        periods in Batch mode); periods without training data are recorded under
        ``'skipped_periods'``.

        Parameters
        ----------
        composites_by_period : dict[str, ee.Image] or dict
            Mapping of ``{period_label: ee.Image}`` or the result dict returned by
            ``final_Image.get_multitemporal_composites()``.
        training_by_period : dict[str, Any]
            Mapping of ``{period_label: ee.FeatureCollection | train_data_dict | GeoDataFrame}``.
        class_property : str, default='class_id'
            Property/column name holding numeric class IDs.
        region : ee.Geometry, optional
            Optional AOI geometry limiting extraction.
        class_name_property : str or None, default='class_name'
            Optional property/column holding human-readable class names.
        scale : int, default=30
            Pixel sampling scale in meters.
        max_pixels_per_class : int, default=5000
            Maximum sampled pixels per class.
        method : str, default='TD'
            Separability metric (`'TD'`).
        period_order : list[str], optional
            Explicit chronological period order.

        Returns
        -------
        dict
            ``{'per_period': {label: {'spectral_df', 'separability_df', 'summary_df', 'analyzer'}},
            'analyzed_periods': list[str], 'skipped_periods': list[str]}``.
        """
        if isinstance(composites_by_period, dict) and isinstance(
            composites_by_period.get('composites'), dict
        ):
            if period_order is None:
                period_order = composites_by_period.get('period_order')
            composites = composites_by_period['composites']
        else:
            composites = composites_by_period

        if period_order is None:
            period_order = list(composites.keys())

        per_period = {}
        analyzed_periods = []
        skipped_periods = []

        for label in period_order:
            if label not in composites:
                raise ValueError(f"Period '{label}' missing from composites_by_period.")

            train_entry = (training_by_period or {}).get(label)
            if isinstance(train_entry, dict) and 'training_data' in train_entry:
                train_obj = train_entry['training_data']
            else:
                train_obj = train_entry

            if train_obj is None:
                skipped_periods.append(label)
                continue

            if isinstance(train_obj, pd.DataFrame) and train_obj.empty:
                skipped_periods.append(label)
                continue

            # Convert GeoDataFrame to ee.FeatureCollection if needed
            if not isinstance(train_obj, ee.FeatureCollection) and hasattr(train_obj, 'geometry'):
                from .input_utils import EE_converter
                train_fc = EE_converter(verbose=False).convert_roi_gdf(train_obj)
            else:
                train_fc = train_obj

            analyzer = sample_quality(
                training_data=train_fc,
                image=composites[label],
                class_property=class_property,
                region=region,
                class_name_property=class_name_property,
            )
            spectral_df = analyzer.extract_spectral_values(
                scale=scale,
                max_pixels_per_class=max_pixels_per_class,
            )
            separability_df = analyzer.get_separability_df(spectral_df, method=method)
            summary_df = analyzer.sum_separability(spectral_df)

            per_period[label] = {
                'spectral_df': spectral_df,
                'separability_df': separability_df,
                'summary_df': summary_df,
                'analyzer': analyzer,
            }
            analyzed_periods.append(label)

        return {
            'per_period': per_period,
            'analyzed_periods': analyzed_periods,
            'skipped_periods': skipped_periods,
        }
        
# Do not initialize Earth Engine at import time. Initialize when classes are instantiated.

# Module 4: Region of Interest Separability Analysis
## System Response 4.2 Sample Visualization
#Class container for plotting Region of Interest 
class spectral_plotter:
    """
    Class container to plot region of interest
    """
    #Initialize the class. 
    def __init__(self, sample_quality):
        """
        Initialize using functions from sample_quality class
        Ensure Earth Engine is initialized lazily (avoids import-time failures).
        """
        # Ensure Earth Engine is initialized when first used (raises helpful error if not)
        ensure_ee_initialized()
        
        self.sq = sample_quality
        self.band_names = self.sq.band_names
        self.class_property = self.sq.class_property
    
    #-----------------OVERLAID HISTOGRAM--------------------------------
    def plot_histogram(self, df, bands=None, max_bands = 3, bins=30, opacity = 0.6):
        """
        Plot overlaid histograms using Plotly for better interactivity.
        All classes shown on same plot for easy comparison.

        Parameters
        ----------
        df : pandas.DataFrame
        bands : list, optional. Bands to plot. If None, take the first 'max_bands'.
        max_bands : int. Maximum number of bands to plot.
        bins : int. Number of bins for histogram
        opacity : float. Transparency of bars (0-1)
        
        Returns
        -------
        list of plotly.graph_objects.Figure
        """
        #Print error message if dataframe from sample analysis is empty
        if df.empty:
            print("No data avaliable for creating histogram")
            return []
        #print error message if bands are empty 
        if bands is None:
            bands = [b for b in self.band_names if b in df.columns][:max_bands]
        #Empty figures list for storing the result
        figures = []
        classes = sorted(df[self.class_property].unique())
        class_mapping = self.sq.class_renaming()
        #Histogram plotting function
        for band in bands:
            figs = go.Figure()
            for class_id in classes:
                class_data = df[df[self.class_property] == class_id][band]
                
                #Get display name
                if class_mapping and class_id in class_mapping:
                    display_name = f"{class_mapping[class_id]} (ID: {class_id})"
                else:
                    display_name = f"Class {class_id}"
                
                figs.add_trace(go.Histogram(
                    x=class_data,
                    name=display_name,
                    opacity=opacity,
                    nbinsx=bins,
                    hovertemplate='<b>%{fullData.name}</b><br>' +
                                  f'{band}: %{{x:.4f}}<br>' +
                                  'Count: %{y}<br>' +
                                  '<extra></extra>'
                ))
            #Modified the annotation for the final plot
            figs.update_layout(
                title = f'Distribution of {band} Reflectance by Class',
                xaxis_title = f'{band} Reflectance',
                yaxis_title = 'Frequency',
                barmode = 'overlay',
                hovermode = 'closest',
                height = 500, 
                legend = dict(
                    title = 'Land Cover Class',
                    orientation = 'v',
                    yanchor = "top",
                    y=1,
                    xanchor = "left",
                    x=1.02
                ),
                template = 'plotly_white'
            )
            figures.append(figs)
        #Return plotly object, therefore it is customizeable during the visualization
        return figures
    
    #-----------------BOX Plot--------------------------------
    def plot_boxplot(self, df, bands=None, max_bands = 5):
        """
        Plot boxplot interactively using plotly
        All classes shown on same plot for easy comparison.
        
        Parameters
        ----------
        df : pandas.DataFrame
        bands : list, optional
            Bands to plot. If None, take the first `max_bands`.
        max_bands : int
            Maximum number of bands to plot.
        bins : int
            Number of bins for histogram
        opacity : float
            Transparency of bars (0-1)
            
        Returns
        -------
        list of plotly.graph_objects.Figure
        """
        #Print error message if dataframe from sample analysis is empty
        if df.empty:
            print("No data avaliable for creating histogram")
            return []
        #print error message if bands are empty 
        if bands is None:
            bands = [b for b in self.band_names if b in df.columns][:max_bands]
        #Empty list to store the figures
        figures = []
        class_mapping = self.sq.class_renaming()
        #get the dataframe from sample_quality analysis and display them on box plot
        df_plot = df.copy()
        if class_mapping:
            df_plot ['Class_Display'] = df_plot[self.class_property].map(
                lambda x: f"{class_mapping.get(x, f'Class {x}')}(ID: {x})"
            )
        else:
            df_plot['Class_Display'] = df_plot[self.class_property].map(lambda x: f"Class {x}")
        #core function for box plot visualiazation
        for band in bands:
            fig = px.box(
                df_plot, 
                x='Class_Display', 
                y=band,
                color='Class_Display',
                title=f'Boxplot of {band} by Class',
                labels={'Class_Display': 'Class', band: f'{band} Reflectance'},
                hover_data={self.class_property: True, band: ':.4f'}
            )
            
            fig.update_layout(
                height=500,
                showlegend=False,
                xaxis_tickangle=-45,
                template='plotly_white',
                hovermode='closest'
            )
            
            fig.update_traces(
                hovertemplate='<b>%{x}</b><br>' +
                              f'{band}: %{{y:.4f}}<br>' +
                              '<extra></extra>'
            )
            
            figures.append(fig)
        
        return figures
  
    #-----------------Interactive Scatter Plot--------------------------------
    #More interactive option of plotting using plotly
    def interactive_scatter_plot(self, df, x_band=None, y_band=None, marker_size=6, 
                    opacity=0.6):
        """
        Create interactive scatter plot using Plotly.

        Parameters
        ----------
        df : pandas.DataFrame
        bands : list, optional
            Bands to plot. If None, take the first `max_bands`.
        max_bands : int
            Maximum number of bands to plot.
            
        Returns
        -------
        list of plotly.graph_objects.Figure
        """
        if df.empty:
            print("No data available for plotting")
            return None
        
        # Get available spectral bands
        available_bands = [col for col in df.columns 
                          if col != self.class_property and col in self.band_names]
        
        if len(available_bands) < 2:
            print("Need at least 2 bands for scatter plot.")
            return None
        
        # Set default bands if not provided
        if x_band is None:
            x_band = available_bands[0]
        if y_band is None:
            y_band = available_bands[1]
        
        # Check if specified bands exist
        if x_band not in available_bands or y_band not in available_bands:
            print(f"Specified bands not found. Available: {available_bands}")
            return None
        
        # Prepare data with display names
        df_plot = df.copy()
        class_mapping = self.sq.class_renaming()
        
        if class_mapping:
            df_plot['Class_Display'] = df_plot[self.class_property].map(
                lambda x: f"{class_mapping.get(x, f'Class {x}')} (ID: {x})"
            )
        else:
            df_plot['Class_Display'] = df_plot[self.class_property].map(lambda x: f"Class {x}")
        
        # Create scatter plot
        fig = px.scatter(
            df_plot,
            x=x_band,
            y=y_band,
            color='Class_Display',
            title=f'Spectral Scatter Plot: {y_band} vs {x_band}',
            labels={
                x_band: f'{x_band} Reflectance',
                y_band: f'{y_band} Reflectance',
                'Class_Display': 'Land Cover Class'
            },
            hover_data={
                self.class_property: True,
                x_band: ':.4f',
                y_band: ':.4f',
                'Class_Display': False
            },

            opacity=opacity
        )
        
        fig.update_traces(
            marker=dict(size=marker_size, line=dict(width=0.5, color='white')),
            hovertemplate='<b>%{fullData.name}</b><br>' +
                         f'{x_band}: %{{x:.4f}}<br>' +
                         f'{y_band}: %{{y:.4f}}<br>' +
                         '<extra></extra>'
        )
        
        fig.update_layout(
            height=600,
            template='plotly_white',
            hovermode='closest',
            legend=dict(
                title='Land Cover Classes',
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02
            )
        )
        
        return fig    
    #-----------------Static Scatter Plot--------------------------------
    #static plotting using matplotlib, but able to compute confidence elipse
    def static_scatter_plot(self, df, x_band=None, y_band=None, alpha=0.6, figsize=(10, 8), 
                         color_palette='tab10', add_legend=True, add_ellipse=False):
        """
        Plot region of interest in a feature space between two bands
        Parameters:
            df : pandas.DataFrame. The dataframe from extract_spectral_values containing spectral data
            x_band : str, optional. Band name for x-axis. If None, uses first available band
            y_band : str, optional. Band name for y-axis. If None, uses second available band
            alpha : float. Transparency of points (0-1)
            figsize : tuple. Figure size (width, height)
            color_palette : str. Color palette for different classes
            add_legend : bool. Whether to add legend
            add_ellipse : bool. Whether to add confidence ellipses for each class
        Returns:
        scatter plot figures
                """
        if df.empty:
            print("No data avaliable for plotting")
        # Get available spectral bands
        available_bands = [col for col in df.columns 
                        if col != self.class_property and col in self.band_names]
        if len(available_bands) < 2:
            print("Need at least 2 bands for scatter plot.")
            return None
        # Set default bands if not provided
        if x_band is None:
            x_band = available_bands[0]
        if y_band is None:
            y_band = available_bands[1] if len(available_bands) > 1 else available_bands[0]    
        # Check if specified bands exist
        if x_band not in available_bands:
            print(f"Band {x_band} not found. Available bands: {available_bands}")
            return None
        if y_band not in available_bands:
            print(f"Band {y_band} not found. Available bands: {available_bands}")
            return None
        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)
        # Get unique classes and create color mapping
        classes = sorted(df[self.class_property].unique())
        colors = plt.cm.get_cmap(color_palette)(np.linspace(0, 1, len(classes)))
        # Get class names if available
        class_mapping = self.sq.class_renaming()
        # Plot each class
        for i, class_id in enumerate(classes):
            class_data = df[df[self.class_property] == class_id]
            # Get display name for class
            if class_mapping and class_id in class_mapping:
                display_name = f"{class_mapping[class_id]} (ID: {class_id})"
            else:
                display_name = f"Class {class_id}"
            # Create scatter plot
            ax.scatter(
                class_data[x_band], 
                class_data[y_band],
                c=[colors[i]], 
                alpha=alpha,
                label=display_name,
                s=20,  # Point size
                edgecolors='white',
                linewidths=0.5
            )
                # Add confidence ellipse if requested
            if add_ellipse and len(class_data) > 2:
                self.add_elipse(
                    ax, class_data[x_band], class_data[y_band], 
                    colors[i], alpha=0.2
                )            
        # Customize plot
        ax.set_xlabel(f'{x_band} Reflectance', fontsize=12)
        ax.set_ylabel(f'{y_band} Reflectance', fontsize=12)
        ax.set_title(f'Spectral Scatter Plot: {y_band} vs {x_band}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        # Add legend
        if add_legend:
            legend = ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
            legend.set_title('Land Cover Classes', prop={'size': 11, 'weight': 'bold'})
        
        # Adjust layout to prevent legend cutoff
        plt.tight_layout()
        
        return fig
    #-----------------Scatter Plot Elipse--------------------------------
    #only able to be visualize using matplotlib (maybe plotly, but with more code)
    def add_elipse(self, ax, x, y, color, n_std = 2, alpha=0.2):
        """
        """
        try:
            cov = np.cov(x, y)
            pearson = cov[0,1]/np.sqrt(cov[0,0] * cov[1,1])
            ell_radius_x = np.sqrt(1 + pearson)
            ell_radius_y = np.sqrt(1 - pearson)
            # Create ellipse
            ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                            facecolor=color, alpha=alpha, edgecolor=color, linewidth=1.5)
            
            # Transform ellipse to data coordinates
            scale_x = np.sqrt(cov[0, 0]) * n_std
            mean_x = np.mean(x)
            scale_y = np.sqrt(cov[1, 1]) * n_std
            mean_y = np.mean(y)  
            transf = transforms.Affine2D() \
            .scale(scale_x, scale_y) \
            .translate(mean_x, mean_y)
        
            ellipse.set_transform(transf + ax.transData)
            ax.add_patch(ellipse)
        except Exception as e:
            print(f"Warning: Could not add confidence ellipse: {e}")
    
    #-----------------3D Scatter Plot--------------------------------
    #This is used for exploratory purposes
    def scatter_plot_3d(self, df, x_band=None, y_band=None, z_band=None, 
                       marker_size=4, opacity=0.7):
        """
        Create interactive 3D scatter plot using Plotly.
        Useful for exploring 3-band feature space.
        
        Parameters
        ----------
        df : pandas.DataFrame
        x_band, y_band, z_band : str
            Band names for each axis
        marker_size : int
            Size of scatter points
        opacity : float
            Transparency of points (0-1)
            
        Returns
        -------
        plotly.graph_objects.Figure
        """
        if df.empty:
            print("No data available for plotting")
            return None
        
        available_bands = [col for col in df.columns 
                          if col != self.class_property and col in self.band_names]
        
        if len(available_bands) < 3:
            print("Need at least 3 bands for 3D scatter plot.")
            return None
        
        # Set default bands
        if x_band is None:
            x_band = available_bands[0]
        if y_band is None:
            y_band = available_bands[1]
        if z_band is None:
            z_band = available_bands[2]
        
        # Prepare data
        df_plot = df.copy()
        class_mapping = self.sq.class_renaming()
        
        if class_mapping:
            df_plot['Class_Display'] = df_plot[self.class_property].map(
                lambda x: f"{class_mapping.get(x, f'Class {x}')} (ID: {x})"
            )
        else:
            df_plot['Class_Display'] = df_plot[self.class_property].map(lambda x: f"Class {x}")
        
        fig = px.scatter_3d(
            df_plot,
            x=x_band,
            y=y_band,
            z=z_band,
            color='Class_Display',
            title=f'3D Spectral Feature Space: {x_band}, {y_band}, {z_band}',
            labels={
                x_band: f'{x_band} Reflectance',
                y_band: f'{y_band} Reflectance',
                z_band: f'{z_band} Reflectance',
                'Class_Display': 'Land Cover Class'
            },
            opacity=opacity
        )
        
        fig.update_traces(
            marker=dict(size=marker_size, line=dict(width=0.3, color='white')),
            hovertemplate='<b>%{fullData.name}</b><br>' +
                         f'{x_band}: %{{x:.4f}}<br>' +
                         f'{y_band}: %{{y:.4f}}<br>' +
                         f'{z_band}: %{{z:.4f}}<br>' +
                         '<extra></extra>'
        )
        
        fig.update_layout(
            height=700,
            legend=dict(
                title='Land Cover Classes',
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02
            )
        )
        
        return fig       
