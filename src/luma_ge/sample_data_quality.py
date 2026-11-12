import ee
import pandas as pd
import numpy as np
from .ee_config import ensure_ee_initialized
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional, Any
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples
from .ee_config import ensure_ee_initialized

# Do not initialize Earth Engine at import time. Initialize when classes are instantiated.
# Module 4: Region of Interest Separability Analysis
## System Response 4.1 Separability Analysis
class sample_quality:
    """
    class which contain several functions used to conduct training data/sample analysis 
    
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
    def _get_feature_columns(self, df: pd.DataFrame, bands: Optional[List[str]] = None) -> List[str]:
        """
        Resolve valid numeric feature/predictor columns from the DataFrame.
        Supports explicit band subsets (e.g. post-RFECV) or defaults to image band_names.
        """
        excluded = {self.class_property, self.class_name_property, 'Class_Display', 'Class_Name'}
        if bands is not None:
            cols = [col for col in bands if col in df.columns and col not in excluded]
            if cols:
                return cols
        cols = [col for col in df.columns if col not in excluded and col in self.band_names]
        if cols:
            return cols
        # Fallback to numeric columns if df contains additional predictor bands not in initial image
        return [
            col for col in df.select_dtypes(include=[np.number]).columns
            if col not in excluded
        ]

    def _stratified_subsample(
        self,
        df: pd.DataFrame,
        max_samples: Optional[int] = 5000,
        random_state: int = 42
    ) -> pd.DataFrame:
        """
        Perform stratified subsampling across classes so non-parametric metrics and
        embeddings execute quickly on large extracted pixel tables.
        """
        if df.empty or max_samples is None or len(df) <= max_samples:
            return df

        rng = np.random.RandomState(random_state)
        total = len(df)
        sampled_parts = []
        for _, group in df.groupby(self.class_property, group_keys=False):
            n_group = len(group)
            if n_group == 0:
                continue
            target_n = max(2, int(round(max_samples * (n_group / total))))
            target_n = min(n_group, target_n)
            idx = rng.choice(group.index, size=target_n, replace=False)
            sampled_parts.append(group.loc[idx])

        if not sampled_parts:
            return df.sample(n=min(len(df), max_samples), random_state=random_state)

        result = pd.concat(sampled_parts, axis=0)
        if len(result) > max_samples:
            result = result.sample(n=max_samples, random_state=random_state)
        return result.reset_index(drop=True)

    def compute_silhouette_analysis(
        self,
        df: pd.DataFrame,
        bands: Optional[List[str]] = None,
        metric: str = 'euclidean',
        max_samples: Optional[int] = 5000,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """
        Compute non-parametric Silhouette Score analysis on standardized predictor features.

        Evaluates how tightly grouped each class is relative to its nearest neighboring
        class in [-1.0, +1.0] without assuming Gaussian distributions or inverting
        covariance matrices.

        Parameters
        ----------
        df : pd.DataFrame
            Extracted sample pixel values containing class_property and predictor columns.
        bands : list of str, optional
            Specific predictor subset to evaluate (e.g. RFECV selected bands).
        metric : str, default='euclidean'
            Distance metric for silhouette calculation ('euclidean', 'cosine', 'manhattan').
        max_samples : int or None, default=5000
            Maximum total samples (stratified across classes) for fast execution.
        random_state : int, default=42
            Random seed for reproducible stratified subsampling.

        Returns
        -------
        dict with keys:
            - 'overall_score': float in [-1, 1]
            - 'per_class_df': pd.DataFrame with per-class silhouette statistics & Negative_Ratio
            - 'pairwise_matrix': dict[str, dict[str, float]] of pairwise class silhouette scores
            - 'bands_used': list of str
            - 'samples_analyzed': int
        """
        feature_cols = self._get_feature_columns(df, bands=bands)
        if not feature_cols or df.empty or self.class_property not in df.columns:
            return {
                'overall_score': 0.0,
                'per_class_df': pd.DataFrame(),
                'pairwise_matrix': {},
                'bands_used': feature_cols,
                'samples_analyzed': 0
            }

        clean_df = df[[self.class_property] + feature_cols].dropna().copy()
        classes = clean_df[self.class_property].unique()
        if len(classes) < 2 or len(clean_df) <= len(classes):
            return {
                'overall_score': 0.0,
                'per_class_df': pd.DataFrame(),
                'pairwise_matrix': {},
                'bands_used': feature_cols,
                'samples_analyzed': len(clean_df)
            }

        sub_df = self._stratified_subsample(clean_df, max_samples=max_samples, random_state=random_state)
        X = StandardScaler().fit_transform(sub_df[feature_cols].values.astype(float))
        y = sub_df[self.class_property].values

        sample_scores = silhouette_samples(X, y, metric=metric)
        overall_score = float(np.mean(sample_scores))

        class_mapping = self.class_renaming() or {}
        per_class_rows = []
        unique_classes = sorted(sub_df[self.class_property].unique())

        for cls_val in unique_classes:
            mask = (y == cls_val)
            cls_scores = sample_scores[mask]
            if len(cls_scores) == 0:
                continue
            try:
                cls_key = int(cls_val)
            except (ValueError, TypeError):
                cls_key = cls_val
            cls_name = class_mapping.get(cls_key, f"Class {cls_val}")
            mean_s = float(np.mean(cls_scores))
            neg_ratio = float(np.mean(cls_scores < 0.0))
            per_class_rows.append({
                'Class_ID': cls_val,
                'Class_Name': cls_name,
                'Sample_Count': int(len(cls_scores)),
                'Mean_Silhouette': round(mean_s, 3),
                'Std_Silhouette': round(float(np.std(cls_scores)), 3),
                'Min_Silhouette': round(float(np.min(cls_scores)), 3),
                'Negative_Ratio': round(neg_ratio, 3),
                'Separability_Level': self.separability_level(mean_s, method='SILHOUETTE')
            })

        per_class_df = pd.DataFrame(per_class_rows)
        if not per_class_df.empty:
            per_class_df = per_class_df.sort_values('Mean_Silhouette').reset_index(drop=True)

        # Compute pairwise silhouette scores on each 2-class subset
        pairwise_matrix: Dict[str, Dict[str, float]] = {str(c): {} for c in unique_classes}
        for i, c1 in enumerate(unique_classes):
            for j, c2 in enumerate(unique_classes):
                if i == j:
                    pairwise_matrix[str(c1)][str(c2)] = 0.0
                elif j < i:
                    pairwise_matrix[str(c1)][str(c2)] = pairwise_matrix[str(c2)][str(c1)]
                else:
                    pair_mask = (y == c1) | (y == c2)
                    X_pair = X[pair_mask]
                    y_pair = y[pair_mask]
                    if len(np.unique(y_pair)) < 2 or len(y_pair) < 3:
                        pair_score = 0.0
                    else:
                        try:
                            pair_score = float(silhouette_score(X_pair, y_pair, metric=metric))
                        except Exception:
                            pair_score = 0.0
                    pairwise_matrix[str(c1)][str(c2)] = pair_score

        return {
            'overall_score': round(overall_score, 4),
            'per_class_df': per_class_df,
            'pairwise_matrix': pairwise_matrix,
            'bands_used': feature_cols,
            'samples_analyzed': int(len(sub_df))
        }

    def get_class_silhouette_df(
        self,
        df: pd.DataFrame,
        bands: Optional[List[str]] = None,
        metric: str = 'euclidean',
        max_samples: Optional[int] = 5000,
        random_state: int = 42
    ) -> pd.DataFrame:
        """Return the per-class Silhouette summary DataFrame sorted from lowest to highest."""
        res = self.compute_silhouette_analysis(
            df, bands=bands, metric=metric, max_samples=max_samples, random_state=random_state
        )
        return res.get('per_class_df', pd.DataFrame())

    def compare_predictor_sets(
        self,
        df: pd.DataFrame,
        predictor_sets: Dict[str, List[str]],
        metric: str = 'euclidean',
        max_samples: Optional[int] = 5000,
        random_state: int = 42
    ) -> pd.DataFrame:
        """
        Compare multiple predictor subsets (e.g. Optical Only vs. Full Stack vs. RFECV)
        using non-parametric Silhouette analysis.

        Parameters
        ----------
        df : pd.DataFrame
            Extracted sample pixel values containing all candidate predictor bands.
        predictor_sets : dict[str, list[str]]
            Mapping of subset name to list of band names.
        metric : str, default='euclidean'
            Distance metric for silhouette calculation.
        max_samples : int or None, default=5000
            Maximum samples (stratified) for evaluation.
        random_state : int, default=42
            Random seed for reproducible subsampling across sets.

        Returns
        -------
        pd.DataFrame summarizing each predictor set's quality.
        """
        rows = []
        for set_name, band_list in predictor_sets.items():
            valid_bands = self._get_feature_columns(df, bands=band_list)
            if not valid_bands:
                continue
            analysis = self.compute_silhouette_analysis(
                df,
                bands=valid_bands,
                metric=metric,
                max_samples=max_samples,
                random_state=random_state
            )
            sep_df = self.get_separability_df(
                df,
                method='SILHOUETTE',
                bands=valid_bands,
                metric=metric,
                max_samples=max_samples,
                random_state=random_state
            )
            per_cls_df = analysis['per_class_df']
            neg_ratio = float(per_cls_df['Negative_Ratio'].mean()) if not per_cls_df.empty else 0.0

            if not sep_df.empty and 'Silhouette_Score' in sep_df.columns:
                mean_pair = float(sep_df['Silhouette_Score'].mean())
                min_row = sep_df.iloc[0]
                min_pair = float(min_row['Silhouette_Score'])
                worst_pair = f"{min_row['Class1_Name']} vs {min_row['Class2_Name']}"
            else:
                mean_pair = 0.0
                min_pair = 0.0
                worst_pair = "N/A"

            rows.append({
                'Predictor_Set': set_name,
                'Num_Bands': len(valid_bands),
                'Overall_Silhouette': round(analysis['overall_score'], 3),
                'Mean_Pairwise_Silhouette': round(mean_pair, 3),
                'Min_Pairwise_Silhouette': round(min_pair, 3),
                'Worst_Class_Pair': worst_pair,
                'Negative_Sample_Ratio': round(neg_ratio, 3),
                'Bands': ', '.join(valid_bands)
            })

        result_df = pd.DataFrame(rows)
        if not result_df.empty:
            result_df = result_df.sort_values('Overall_Silhouette', ascending=False).reset_index(drop=True)
        return result_df

    def compute_tsne_coordinates(
        self,
        df: pd.DataFrame,
        bands: Optional[List[str]] = None,
        n_components: int = 2,
        perplexity: float = 30.0,
        max_samples: Optional[int] = 3000,
        random_state: int = 42
    ) -> pd.DataFrame:
        """
        Compute 2D or 3D t-SNE embedding coordinates on standardized predictor features.
        Returns a DataFrame containing TSNE_1, TSNE_2 (, TSNE_3) and class metadata.
        """
        feature_cols = self._get_feature_columns(df, bands=bands)
        if not feature_cols or df.empty or self.class_property not in df.columns:
            return pd.DataFrame()

        clean_df = df[[self.class_property] + feature_cols].dropna().copy()
        if len(clean_df) < 4:
            return pd.DataFrame()

        sub_df = self._stratified_subsample(clean_df, max_samples=max_samples, random_state=random_state)
        n_samples = len(sub_df)
        effective_perplexity = float(min(perplexity, max(2.0, (n_samples - 1) / 3.0)))

        X = StandardScaler().fit_transform(sub_df[feature_cols].values.astype(float))
        tsne = TSNE(
            n_components=n_components,
            perplexity=effective_perplexity,
            random_state=random_state,
            init='pca',
            learning_rate='auto'
        )
        coords = tsne.fit_transform(X)

        out_df = pd.DataFrame()
        out_df[self.class_property] = sub_df[self.class_property].values
        class_mapping = self.class_renaming() or {}

        def _fmt_display(val):
            try:
                key = int(val)
            except (ValueError, TypeError):
                key = val
            if class_mapping and key in class_mapping:
                return f"{class_mapping[key]} (ID: {val})"
            return f"Class {val}"

        out_df['Class_Display'] = [
            _fmt_display(v) for v in sub_df[self.class_property].values
        ]
        for i in range(n_components):
            out_df[f'TSNE_{i + 1}'] = coords[:, i]

        return out_df

    def compute_pca_coordinates(
        self,
        df: pd.DataFrame,
        bands: Optional[List[str]] = None,
        n_components: int = 2,
        max_samples: Optional[int] = 5000,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """
        Compute PCA projection coordinates, explained variance ratios, and feature loadings
        on standardized predictor features.
        """
        feature_cols = self._get_feature_columns(df, bands=bands)
        if not feature_cols or df.empty or self.class_property not in df.columns:
            return {'coordinates_df': pd.DataFrame(), 'explained_variance_ratio': [], 'loadings_df': pd.DataFrame()}

        clean_df = df[[self.class_property] + feature_cols].dropna().copy()
        n_comp = min(n_components, len(feature_cols), len(clean_df))
        if n_comp < 2:
            return {'coordinates_df': pd.DataFrame(), 'explained_variance_ratio': [], 'loadings_df': pd.DataFrame()}

        sub_df = self._stratified_subsample(clean_df, max_samples=max_samples, random_state=random_state)
        X = StandardScaler().fit_transform(sub_df[feature_cols].values.astype(float))

        pca = PCA(n_components=n_comp, random_state=random_state)
        coords = pca.fit_transform(X)

        out_df = pd.DataFrame()
        out_df[self.class_property] = sub_df[self.class_property].values
        class_mapping = self.class_renaming() or {}

        def _fmt_display(val):
            try:
                key = int(val)
            except (ValueError, TypeError):
                key = val
            if class_mapping and key in class_mapping:
                return f"{class_mapping[key]} (ID: {val})"
            return f"Class {val}"

        out_df['Class_Display'] = [
            _fmt_display(v) for v in sub_df[self.class_property].values
        ]
        pc_cols = []
        for i in range(n_comp):
            col_name = f'PC_{i + 1}'
            pc_cols.append(col_name)
            out_df[col_name] = coords[:, i]

        loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
        loadings_df = pd.DataFrame(loadings, index=feature_cols, columns=pc_cols)
        loadings_df['Magnitude'] = np.linalg.norm(loadings[:, :min(2, n_comp)], axis=1)
        loadings_df = loadings_df.sort_values('Magnitude', ascending=False)

        return {
            'coordinates_df': out_df,
            'explained_variance_ratio': [float(v) for v in pca.explained_variance_ratio_],
            'loadings_df': loadings_df,
            'bands_used': feature_cols
        }

    #Calculate class separability using jeffries matusita distance
    #Note: This approach is not ideal for machine learning classifier, since JM distance works best with gaussian distribution
    def check_class_separability(self, df, method='TD', bands=None, **kwargs):
        """
        Calculate class separability using Transformed Divergence ('TD'),
        Jeffries-Matusita ('JM'), or non-parametric Silhouette Score ('SILHOUETTE').
        Default method is 'TD' for backward compatibility.
        """
        method_upper = (method or 'TD').upper()
        if method_upper == 'SILHOUETTE':
            analysis = self.compute_silhouette_analysis(df, bands=bands, **kwargs)
            return analysis.get('pairwise_matrix', {})

        spectral_bands = self._get_feature_columns(df, bands=bands)
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

                        if method_upper == 'JM':
                            sep = self._jeffries_matusita_distance(data1, data2)
                        else:
                            sep = self.transform_divergence(data1, data2)

                        separability_matrix[str(class1)][str(class2)] = float(sep)

                    except Exception as e:
                        print(f"Error calculating separability for classes {class1}-{class2}: {e}")
                        separability_matrix[str(class1)][str(class2)] = 0.0
                else:
                    separability_matrix[str(class1)][str(class2)] = 0.0

        return separability_matrix

    def _get_metric_column_name(self, method='TD') -> str:
        method_upper = (method or 'TD').upper()
        if method_upper == 'SILHOUETTE':
            return 'Silhouette_Score'
        if method_upper == 'JM':
            return 'JM_Distance'
        return 'TD_Distance'

    def get_separability_df(self, df, method='TD', bands=None, **kwargs):
        """
        Get separability results as a DataFrame.
        Supports method='TD' (default), 'JM', or 'SILHOUETTE'.
        """
        method_upper = (method or 'TD').upper()
        separability = self.check_class_separability(df, method=method_upper, bands=bands, **kwargs)
        if not separability:
            return pd.DataFrame()

        pairs = []
        class_mapping = self.class_renaming() or {}
        metric_name = self._get_metric_column_name(method_upper)
        processed_pairs = set()  # Track processed pairs to avoid duplicates

        for class1_id, class1_data in separability.items():
            for class2_id, value in class1_data.items():
                if str(class1_id) == str(class2_id):
                    continue
                if method_upper != 'SILHOUETTE' and value <= 0:
                    continue
                # Create a sorted tuple to ensure unique pairs (avoid Class1-Class2 and Class2-Class1)
                pair_key = tuple(sorted([str(class1_id), str(class2_id)]))

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
                    metric_name: round(float(value), 3),
                    'Separability_Level': self.separability_level(value, method=method_upper)
                })

        pairs_df = pd.DataFrame(pairs)
        if not pairs_df.empty:
            pairs_df = pairs_df.sort_values(metric_name).reset_index(drop=True)
        return pairs_df

    def lowest_separability(self, df, top_n=10, method='TD', bands=None, **kwargs):
        """
        Get the class pairs with the lowest separability value.
        Args:
            df: DataFrame with spectral values
            top_n: Number of lowest pairs to return
            method: 'TD', 'JM', or 'SILHOUETTE'
        Returns:
            DataFrame of lowest separability pairs with interpretation
        """
        separability_df = self.get_separability_df(df, method=method, bands=bands, **kwargs)
        if separability_df.empty:
            return pd.DataFrame()
        metric_name = self._get_metric_column_name(method)
        lowest_pairs = separability_df.head(top_n).copy()
        lowest_pairs['Interpretation'] = lowest_pairs[metric_name].apply(
            lambda x: self.separability_level(x, method=method)
        )
        return lowest_pairs

    def separability_level(self, value, method='TD'):
        """
        Categorize separability value for TD/JM (0-2) or Silhouette (-1 to 1).
        Args:
            value: Separability metric value
            method: 'TD', 'JM', or 'SILHOUETTE'
        Returns:
            String interpretation with detailed explanation
        """
        method_upper = (method or 'TD').upper()
        if method_upper == 'SILHOUETTE':
            if value >= 0.50:
                return "🟢 Good Separability (Silhouette ≥ 0.50)"
            elif 0.25 <= value < 0.50:
                return "🟡 Weak/Marginal Separability (0.25 ≤ Silhouette < 0.50)"
            else:
                return "🔴 Poor Separability (Silhouette < 0.25)"

        # Thresholds for TD / JM (range 0–2)
        if value >= 1.8:
            return "🟢 Good Separability (TD ≥ 1.8)"
        elif 1.0 <= value < 1.8:
            return "🟡 Weak/Marginal Separability (1.0 ≤ TD < 1.8)"
        else:
            return "🔴 Poor Separability (TD < 1.0)"
    #Summarize the separability
    def sum_separability(self, df, method='TD', bands=None, **kwargs):
        """
        summarize the separability result
        """
        sum_df = self.get_separability_df(df, method=method, bands=bands, **kwargs)
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

    #t-SNE & PCA Dimensionality Reduction Plots
    #t-SNE (t-distributed stochastic neighbor embedding)
    #PCA (Principal Component Analysis)
    def plot_tsne(
        self,
        df: pd.DataFrame,
        bands: Optional[List[str]] = None,
        n_components: int = 2,
        perplexity: float = 30.0,
        max_samples: Optional[int] = 3000,
        random_state: int = 42,
        interactive: bool = True,
        title: Optional[str] = None,
        marker_size: int = 6,
        opacity: float = 0.75,
        figsize: tuple = (10, 8)
    ):
        """
        Visualize high-dimensional predictor stacks in 2D or 3D using t-SNE.

        Standardizes all selected predictors before computing t-SNE and performs
        stratified subsampling so non-linear multi-band stacks render quickly.

        Parameters
        ----------
        df : pd.DataFrame
            Extracted sample pixel values.
        bands : list of str, optional
            Predictor subset to include (e.g. RFECV selected bands).
        n_components : int, default=2
            Embedding dimension (2 or 3).
        perplexity : float, default=30.0
            t-SNE perplexity parameter (automatically clamped for small sample sizes).
        max_samples : int or None, default=3000
            Maximum samples (stratified across classes) to embed.
        random_state : int, default=42
            Random seed for reproducible layout.
        interactive : bool, default=True
            If True, returns an interactive Plotly Figure; if False, returns a Matplotlib Figure.
        title : str, optional
            Custom plot title.
        marker_size : int, default=6
            Point size.
        opacity : float, default=0.75
            Point opacity.
        figsize : tuple, default=(10, 8)
            Figure size when interactive=False.

        Returns
        -------
        plotly.graph_objects.Figure or matplotlib.figure.Figure or None
        """
        if df.empty:
            print("No data available for t-SNE visualization")
            return None

        n_comp = 3 if n_components == 3 else 2
        coords_df = self.sq.compute_tsne_coordinates(
            df,
            bands=bands,
            n_components=n_comp,
            perplexity=perplexity,
            max_samples=max_samples,
            random_state=random_state
        )
        if coords_df.empty:
            print("Insufficient data or bands for t-SNE visualization")
            return None

        feature_cols = self.sq._get_feature_columns(df, bands=bands)
        plot_title = title or f"t-SNE {n_comp}D Projection ({len(feature_cols)} Predictors, n={len(coords_df)})"

        if interactive:
            if n_comp == 3:
                fig = px.scatter_3d(
                    coords_df,
                    x='TSNE_1',
                    y='TSNE_2',
                    z='TSNE_3',
                    color='Class_Display',
                    title=plot_title,
                    labels={'Class_Display': 'Land Cover Class'},
                    opacity=opacity
                )
            else:
                fig = px.scatter(
                    coords_df,
                    x='TSNE_1',
                    y='TSNE_2',
                    color='Class_Display',
                    title=plot_title,
                    labels={'Class_Display': 'Land Cover Class'},
                    opacity=opacity
                )
            fig.update_traces(
                marker=dict(size=marker_size, line=dict(width=0.4, color='white'))
            )
            fig.update_layout(
                template='plotly_white',
                height=650,
                legend=dict(
                    title='Land Cover Classes',
                    orientation='v',
                    yanchor='top',
                    y=1,
                    xanchor='left',
                    x=1.02
                )
            )
            return fig

        # Static Matplotlib fallback
        if n_comp == 3:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection='3d')
            for cls_label, grp in coords_df.groupby('Class_Display'):
                ax.scatter(
                    grp['TSNE_1'], grp['TSNE_2'], grp['TSNE_3'],
                    label=cls_label, s=marker_size * 4, alpha=opacity
                )
            ax.set_xlabel('t-SNE 1')
            ax.set_ylabel('t-SNE 2')
            ax.set_zlabel('t-SNE 3')
        else:
            fig, ax = plt.subplots(figsize=figsize)
            for cls_label, grp in coords_df.groupby('Class_Display'):
                ax.scatter(
                    grp['TSNE_1'], grp['TSNE_2'],
                    label=cls_label, s=marker_size * 5, alpha=opacity
                )
            ax.set_xlabel('t-SNE 1')
            ax.set_ylabel('t-SNE 2')
            ax.grid(True, alpha=0.3)

        ax.set_title(plot_title)
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.tight_layout()
        return fig

    def plot_pca(
        self,
        df: pd.DataFrame,
        bands: Optional[List[str]] = None,
        n_components: int = 2,
        max_samples: Optional[int] = 5000,
        show_loadings: bool = True,
        top_n_loadings: int = 8,
        interactive: bool = True,
        title: Optional[str] = None,
        marker_size: int = 6,
        opacity: float = 0.75,
        figsize: tuple = (10, 8)
    ):
        """
        Visualize predictor stack in 2D or 3D Principal Component space with optional
        feature loading vectors (biplot) showing which predictors drive separation.

        Parameters
        ----------
        df : pd.DataFrame
            Extracted sample pixel values.
        bands : list of str, optional
            Predictor subset to include.
        n_components : int, default=2
            Number of principal components to plot (2 or 3).
        max_samples : int or None, default=5000
            Maximum samples (stratified across classes) to project.
        show_loadings : bool, default=True
            If True (in 2D mode), overlays predictor loading vectors on the plot.
        top_n_loadings : int, default=8
            Maximum number of top predictor loading arrows to display for readability.
        interactive : bool, default=True
            If True, returns an interactive Plotly Figure; if False, returns a Matplotlib Figure.
        title : str, optional
            Custom plot title.
        marker_size : int, default=6
            Point size.
        opacity : float, default=0.75
            Point opacity.
        figsize : tuple, default=(10, 8)
            Figure size when interactive=False.

        Returns
        -------
        plotly.graph_objects.Figure or matplotlib.figure.Figure or None
        """
        if df.empty:
            print("No data available for PCA visualization")
            return None

        n_comp = 3 if n_components == 3 else 2
        pca_res = self.sq.compute_pca_coordinates(
            df,
            bands=bands,
            n_components=n_comp,
            max_samples=max_samples
        )
        coords_df = pca_res.get('coordinates_df', pd.DataFrame())
        if coords_df.empty:
            print("Insufficient data or bands for PCA visualization")
            return None

        evr = pca_res.get('explained_variance_ratio', [])
        loadings_df = pca_res.get('loadings_df', pd.DataFrame())
        bands_used = pca_res.get('bands_used', [])

        pc1_pct = evr[0] * 100 if len(evr) > 0 else 0.0
        pc2_pct = evr[1] * 100 if len(evr) > 1 else 0.0
        pc3_pct = evr[2] * 100 if len(evr) > 2 else 0.0
        total_pct = sum(evr[:n_comp]) * 100

        plot_title = title or f"PCA {n_comp}D Projection ({len(bands_used)} Predictors, {total_pct:.1f}% Variance Explained)"
        x_label = f"PC 1 ({pc1_pct:.1f}%)"
        y_label = f"PC 2 ({pc2_pct:.1f}%)"
        z_label = f"PC 3 ({pc3_pct:.1f}%)"

        # Scale loading vectors to fit nicely within the score range
        top_loadings = loadings_df.head(top_n_loadings) if not loadings_df.empty else pd.DataFrame()
        scale_factor = 1.0
        if not top_loadings.empty and 'PC_1' in coords_df.columns and 'PC_2' in coords_df.columns:
            max_score = max(
                float(np.percentile(np.abs(coords_df['PC_1']), 95)),
                float(np.percentile(np.abs(coords_df['PC_2']), 95)),
                1.0
            )
            max_loading = float(top_loadings['Magnitude'].max()) or 1.0
            scale_factor = (max_score * 0.85) / max_loading

        if interactive:
            if n_comp == 3 and 'PC_3' in coords_df.columns:
                fig = px.scatter_3d(
                    coords_df,
                    x='PC_1',
                    y='PC_2',
                    z='PC_3',
                    color='Class_Display',
                    title=plot_title,
                    labels={'PC_1': x_label, 'PC_2': y_label, 'PC_3': z_label, 'Class_Display': 'Land Cover Class'},
                    opacity=opacity
                )
            else:
                fig = px.scatter(
                    coords_df,
                    x='PC_1',
                    y='PC_2',
                    color='Class_Display',
                    title=plot_title,
                    labels={'PC_1': x_label, 'PC_2': y_label, 'Class_Display': 'Land Cover Class'},
                    opacity=opacity
                )
                if show_loadings and not top_loadings.empty:
                    for band_name, row in top_loadings.iterrows():
                        lx = float(row['PC_1']) * scale_factor
                        ly = float(row['PC_2']) * scale_factor
                        fig.add_annotation(
                            x=lx,
                            y=ly,
                            ax=0,
                            ay=0,
                            xref='x',
                            yref='y',
                            axref='x',
                            ayref='y',
                            text=f"<b>{band_name}</b>",
                            showarrow=True,
                            arrowhead=2,
                            arrowsize=1.2,
                            arrowwidth=1.8,
                            arrowcolor='#222222',
                            font=dict(size=11, color='#111111'),
                            bgcolor='rgba(255,255,255,0.75)'
                        )

            fig.update_traces(
                marker=dict(size=marker_size, line=dict(width=0.4, color='white'))
            )
            fig.update_layout(
                template='plotly_white',
                height=650,
                legend=dict(
                    title='Land Cover Classes',
                    orientation='v',
                    yanchor='top',
                    y=1,
                    xanchor='left',
                    x=1.02
                )
            )
            return fig

        # Static Matplotlib fallback
        if n_comp == 3 and 'PC_3' in coords_df.columns:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection='3d')
            for cls_label, grp in coords_df.groupby('Class_Display'):
                ax.scatter(
                    grp['PC_1'], grp['PC_2'], grp['PC_3'],
                    label=cls_label, s=marker_size * 4, alpha=opacity
                )
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_zlabel(z_label)
        else:
            fig, ax = plt.subplots(figsize=figsize)
            for cls_label, grp in coords_df.groupby('Class_Display'):
                ax.scatter(
                    grp['PC_1'], grp['PC_2'],
                    label=cls_label, s=marker_size * 5, alpha=opacity
                )
            if show_loadings and not top_loadings.empty:
                for band_name, row in top_loadings.iterrows():
                    lx = float(row['PC_1']) * scale_factor
                    ly = float(row['PC_2']) * scale_factor
                    ax.annotate(
                        band_name,
                        xy=(lx, ly),
                        xytext=(lx * 1.08, ly * 1.08),
                        arrowprops=dict(arrowstyle='<-', color='#222222', lw=1.5),
                        fontsize=9,
                        fontweight='bold',
                        color='#111111'
                    )
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.grid(True, alpha=0.3)

        ax.set_title(plot_title)
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.tight_layout()
        return fig

