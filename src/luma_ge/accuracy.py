"""
Module 7: Thematic Accuracy Assessment

Provides core functionality for assessing thematic accuracy of land cover
classification maps using an independent reference data. Adapted from AcATaMa QGIS Plugins (https://github.com/SMByC/AcATaMa)
"""
from scipy import stats
import numpy as np
import ee
from typing import Dict, List, Tuple, Any, Optional
import logging
import time
import io
import zipfile
import tempfile
import os
import geopandas as gpd

# Earth Engine initialization with fallback
try:
    from luma_ge import ensure_ee_initialized
except ImportError:
    def ensure_ee_initialized():
        """Basic Earth Engine initialization fallback."""
        try:
            ee.data.getAssetRoots()
        except Exception:
            ee.Initialize()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#Sampling Design

class sample_size_calculator:
    """Calculator for reference sample sizes using stratified random sampling.

    Distributes samples across strata (classes) following Cochran (1977).
    """

    def __init__(self):
        ensure_ee_initialized()

    def get_pixel_counts_per_class(
        self,
        classification_map: ee.Image,
        class_ids: List[int],
        geometry: ee.Geometry,
        max_retries: int = 3,
        initial_backoff: float = 1.0
    ) -> Dict[int, int]:
        """Extract pixel counts for each class with retry logic.

        Parameters
        ----------
        classification_map : ee.Image
            Classified image containing integer class labels.
        class_ids : list[int]
            List of class IDs to retrieve counts for.
        geometry : ee.Geometry
            Area of interest over which to count pixels.
        max_retries : int, optional
            Maximum number of retry attempts for Earth Engine calls.
        initial_backoff : float, optional
            Initial backoff time in seconds between retries.

        Returns
        -------
        dict[int, int]
            Mapping of class ID to pixel count.
        """
        last_error = None
        backoff = initial_backoff

        for attempt in range(max_retries):
            try:
                histogram = classification_map.reduceRegion(
                    reducer=ee.Reducer.frequencyHistogram(),
                    geometry=geometry,
                    scale=30,
                    maxPixels=1e9,
                    tileScale=4
                )

                band_names = classification_map.bandNames().getInfo()
                if not band_names:
                    raise RuntimeError("classification_map has no bands")

                histogram_dict = (
                    histogram.get('classification').getInfo()
                    if 'classification' in band_names
                    else histogram.get(band_names[0]).getInfo()
                )

                if histogram_dict is None:
                    raise RuntimeError("Failed to retrieve histogram from Earth Engine")

                return {class_id: int(histogram_dict.get(str(class_id), 0))
                        for class_id in class_ids}

            except ee.EEException as e:
                last_error = e
                if "User memory limit exceeded" in str(e) or "Computation timed out" in str(e):
                    if attempt < max_retries - 1:
                        time.sleep(backoff)
                        backoff *= 2
                else:
                    raise RuntimeError(f"Earth Engine API error: {str(e)}") from e

            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    time.sleep(backoff)
                    backoff *= 2

        raise RuntimeError(f"Failed after {max_retries} attempts: {last_error}")

    @staticmethod
    def validate_sample_size_inputs(
        expected_accuracies: Dict[int, float],
        standard_error: float
    ) -> Tuple[bool, List[str]]:
        """Validate inputs for sample size calculation.

        Parameters
        ----------
        expected_accuracies : dict[int, float]
            Expected per-class accuracies (values in (0, 1)).
        standard_error : float
            Desired standard error for the overall estimate.

        Returns
        -------
        tuple[bool, list[str]]
            Tuple of (is_valid, errors). If valid, errors is an empty list.
        """
        errors = []

        for class_id, accuracy in expected_accuracies.items():
            if not (0.01 <= accuracy <= 0.99):
                errors.append(f"Accuracy for class {class_id} must be in [0.01, 0.99]")

        if not (0.001 <= standard_error <= 0.1):
            errors.append("Standard error must be in [0.001, 0.1]")

        return len(errors) == 0, errors

    @staticmethod
    def calculate_strata_sample(
        pixel_counts: Dict[int, int],
        expected_accuracies: Dict[int, float],
        standard_error: float
    ) -> Dict[str, Any]:
        """Calculate required sample sizes using stratified random sampling.

        Cochran Formula (1977):
            Si = sqrt(Ui * (1 - Ui))           # Stratum std dev
            Wi = Ni / N                         # Stratum weight
            n  = (sum(Wi * Si) / SE)^2          # Total samples
            ni = n * (Wi * Si) / sum(Wi * Si)   # Samples per stratum

        Parameters
        ----------
        pixel_counts : dict[int, int]
            Number of pixels per class in the study area.
        expected_accuracies : dict[int, float]
            Expected accuracy for each class (values in (0, 1)).
        standard_error : float
            Desired standard error of the overall accuracy estimate.

        Returns
        -------
        dict[str, Any]
            Dictionary containing total_samples, samples_per_class,
            stratum_std_devs, and class_proportions.
        """
        if not pixel_counts or not expected_accuracies:
            raise ValueError("pixel_counts and expected_accuracies cannot be empty")

        if set(pixel_counts.keys()) != set(expected_accuracies.keys()):
            raise ValueError("pixel_counts and expected_accuracies must have same class IDs")

        for class_id, count in pixel_counts.items():
            if count <= 0:
                raise ValueError(f"Pixel count for class {class_id} must be positive")

        total_pixels = sum(pixel_counts.values())

        class_proportions = {
            cid: count / total_pixels
            for cid, count in pixel_counts.items()
        }

        stratum_std_devs = {
            cid: (expected_accuracies[cid] * (1 - expected_accuracies[cid])) ** 0.5
            for cid in pixel_counts.keys()
        }

        sum_wi_si = sum(class_proportions[cid] * stratum_std_devs[cid]
                        for cid in pixel_counts.keys())

        total_samples = int(np.ceil((sum_wi_si / standard_error) ** 2))

        samples_per_class = {
            cid: max(1, int(round(total_samples * class_proportions[cid] *
                                  stratum_std_devs[cid] / sum_wi_si)))
            for cid in pixel_counts.keys()
        }

        return {
            'total_samples': sum(samples_per_class.values()),
            'samples_per_class': samples_per_class,
            'stratum_std_devs': stratum_std_devs,
            'class_proportions': class_proportions
        }

    def generate_stratified_samples(
        self,
        classification_map: ee.Image,
        samples_per_class: Dict[int, int],
        geometry: ee.Geometry,
        scale: int = 30,
        seed: int = 42
    ) -> ee.FeatureCollection:
        """Generate stratified random sample points using Earth Engine.

        Parameters
        ----------
        classification_map : ee.Image
            Classified image containing integer class labels.
        samples_per_class : dict[int, int]
            Desired number of samples per class.
        geometry : ee.Geometry
            Area of interest to sample within.
        scale : int, optional
            Pixel scale in meters (default 30).
        seed : int, optional
            Random seed for reproducibility (default 42).

        Returns
        -------
        ee.FeatureCollection
            FeatureCollection of points with classification and sample_id properties.

        Notes
        -----
        Requires Earth Engine API versions 1.13, 1.14, or 1.16 for stratifiedSample.
        """
        if not samples_per_class:
            raise ValueError("samples_per_class cannot be empty")

        for class_id, count in samples_per_class.items():
            if count <= 0:
                raise ValueError(f"Sample count for class {class_id} must be positive")

        class_values = list(samples_per_class.keys())
        class_points = list(samples_per_class.values())

        # Resolve band name to a Python string before passing to EE
        band_names = classification_map.bandNames().getInfo()
        if not band_names:
            raise ValueError("classification_map has no bands")
        class_band = 'classification' if 'classification' in band_names else band_names[0]

        samples = classification_map.stratifiedSample(
            numPoints=sum(class_points),
            classBand=class_band,
            region=geometry,
            scale=scale,
            classValues=class_values,
            classPoints=class_points,
            seed=seed,
            geometries=True,
            tileScale=4
        )

        return samples.map(lambda f: f.set('sample_id', f.id()))

    def export_samples_to_shp(
        self,
        samples: ee.FeatureCollection,
        filename: str = "validation_samples"
    ) -> bytes:
        """Export sample points as a zipped shapefile.

        Parameters
        ----------
        samples : ee.FeatureCollection
            FeatureCollection of sampled points (must include geometries).
        filename : str, optional
            Base filename for the shapefile (without extension).

        Returns
        -------
        bytes
            Bytes of a ZIP archive containing the shapefile components.

        Notes
        -----
        Current limitation: only class IDs are written to the attribute table,
        not class names.
        """
        geojson = samples.getInfo()
        gdf = gpd.GeoDataFrame.from_features(geojson['features'])
        gdf.set_crs(epsg=4326, inplace=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            shapefile_path = os.path.join(tmpdir, filename)
            gdf.to_file(shapefile_path + ".shp", driver="ESRI Shapefile")

            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                for ext in ['.shp', '.shx', '.dbf', '.prj', '.cpg']:
                    file_path = shapefile_path + ext
                    if os.path.exists(file_path):
                        zip_file.write(file_path, arcname=filename + ext)

            zip_buffer.seek(0)
            return zip_buffer.read()


#Validation Point Flagging

class validation_error_flag:
    """Flags validation points as correct/incorrect and generates map popups.

    All methods are stateless utilities grouped here for organisational clarity
    and future extensibility (e.g. adding per-class error breakdown).
    """

    def __init__(self, class_names: Dict[int, str] = None):
        """
        Parameters
        ----------
        class_names : dict[int, str], optional
            Mapping from class ID to human-readable name, used by
            :meth:`generate_popup_html`.
        """
        self.class_names = class_names or {}

    def classify_validation_points(
        self,
        validation_gdf: gpd.GeoDataFrame,
        class_property: str = 'CLASS_ID',
        predicted_property: str = 'classification'
    ) -> gpd.GeoDataFrame:
        """Add correctness flags to validation points.

        Parameters
        ----------
        validation_gdf : geopandas.GeoDataFrame
            Ground truth validation points with class labels.
        class_property : str, optional
            Column name for actual class labels.
        predicted_property : str, optional
            Column name for predicted class labels.

        Returns
        -------
        geopandas.GeoDataFrame
            GeoDataFrame with added columns: actual_class, predicted_class,
            is_correct, error_type ('correct' or 'incorrect').
        """
        if not isinstance(validation_gdf, gpd.GeoDataFrame) or validation_gdf.empty:
            raise ValueError("validation_gdf must be a non-empty GeoDataFrame")

        if class_property not in validation_gdf.columns:
            raise ValueError(f"Column '{class_property}' not found")

        if predicted_property not in validation_gdf.columns:
            raise ValueError(f"Column '{predicted_property}' not found")

        result_gdf = validation_gdf.copy()
        result_gdf['actual_class'] = result_gdf[class_property].astype(int)
        result_gdf['predicted_class'] = result_gdf[predicted_property].astype(int)
        result_gdf['is_correct'] = result_gdf['predicted_class'] == result_gdf['actual_class']
        result_gdf['error_type'] = result_gdf['is_correct'].map({True: 'correct', False: 'incorrect'})

        return result_gdf

    def generate_popup_html(
        self,
        predicted_class: int,
        actual_class: int,
        coordinates: tuple
    ) -> str:
        """Generate an HTML popup string for a validation point.

        Parameters
        ----------
        predicted_class : int
            Predicted class label.
        actual_class : int
            Actual (reference) class label.
        coordinates : tuple
            (longitude, latitude) tuple.

        Returns
        -------
        str
            HTML string suitable for map popup display.
        """
        is_correct = predicted_class == actual_class
        status = "✓ Correct" if is_correct else "✗ Error"
        color = "#2ecc71" if is_correct else "#e74c3c"

        actual_name = self.class_names.get(actual_class, f"Class {actual_class}")
        predicted_name = self.class_names.get(predicted_class, f"Class {predicted_class}")

        lon, lat = coordinates

        return f"""
        <div style="font-family: Arial, sans-serif; padding: 10px; min-width: 200px;">
            <h4 style="margin: 0 0 10px 0; color: {color}; font-size: 16px; border-bottom: 2px solid {color}; padding-bottom: 5px;">
                {status}
            </h4>
            <div style="margin: 8px 0;">
                <p style="margin: 5px 0; font-size: 14px;"><strong>Actual:</strong> {actual_name}</p>
                <p style="margin: 5px 0; font-size: 14px;"><strong>Predicted:</strong> {predicted_name}</p>
            </div>
            <div style="margin-top: 10px; padding-top: 8px; border-top: 1px solid #ddd;">
                <p style="margin: 5px 0; font-size: 12px; color: #7f8c8d;">
                    <strong>Location:</strong><br/>Lat: {lat:.4f}, Lon: {lon:.4f}
                </p>
            </div>
        </div>
        """.strip()

#Thematic Accuracy Assessment
class thematic_accuracy:
    """Thematic Accuracy Assessment manager.

    Provides comprehensive accuracy metrics including overall accuracy, kappa
    coefficient, producer's/user's accuracies, F1 scores, and confusion matrix.
    """

    def __init__(self):
        ensure_ee_initialized()
        self.supported_metrics = [
            'overall_accuracy', 'kappa', 'producer_accuracy',
            'user_accuracy', 'f1_scores', 'confusion_matrix'
        ]

    def validate_assessment_inputs(
        self,
        lcmap: ee.Image,
        validation_data: ee.FeatureCollection,
        class_property: str,
        scale: int
    ) -> Tuple[bool, Optional[str]]:
        """Validate input parameters for accuracy assessment.

        Parameters
        ----------
        lcmap : ee.Image
            Land cover classification image (must include a 'classification' band).
        validation_data : ee.FeatureCollection
            Validation points with ground truth labels.
        class_property : str
            Property name in validation_data containing the reference class.
        scale : int
            Sampling scale in meters.

        Returns
        -------
        tuple[bool, str | None]
            Tuple of (is_valid, error_message). If valid, error_message is None.
        """
        try:
            band_names = lcmap.bandNames().getInfo()
            if 'classification' not in band_names:
                return False, "Land cover map must contain 'classification' band"

            properties = validation_data.first().propertyNames().getInfo()
            if class_property not in properties:
                return False, f"Class property '{class_property}' not found in validation data"

            if not isinstance(scale, (int, float)) or scale <= 0:
                return False, "Scale must be a positive number"

            return True, None

        except Exception as e:
            return False, f"Validation error: {str(e)}"

    def _calculate_confidence_interval(
        self,
        n_correct: int,
        n_total: int,
        confidence: float = 0.95
    ) -> Tuple[float, float]:
        """Calculate confidence interval for overall accuracy.

        Parameters
        ----------
        n_correct : int
            Number of correctly classified samples.
        n_total : int
            Total number of samples.
        confidence : float, optional
            Confidence level (default 0.95).

        Returns
        -------
        tuple[float, float]
            Lower and upper bounds of the confidence interval.
        """
        if n_total == 0:
            return 0.0, 0.0

        p = n_correct / n_total
        se = np.sqrt((p * (1 - p)) / n_total)
        z = stats.norm.ppf((1 + confidence) / 2)
        margin = z * se

        return max(0.0, p - margin), min(1.0, p + margin)

    def _calculate_f1_scores(
        self,
        producer_accuracy: List[float],
        user_accuracy: List[float]
    ) -> List[float]:
        """Calculate F1 scores for each class.

        Parameters
        ----------
        producer_accuracy : list[float]
            Producer accuracy per class.
        user_accuracy : list[float]
            User accuracy per class.

        Returns
        -------
        list[float]
            F1 scores per class.
        """
        return [
            2 * (pa * ua) / (pa + ua) if (pa + ua) > 0 else 0.0
            for pa, ua in zip(producer_accuracy, user_accuracy)
        ]

    def _extract_confusion_matrix_data(
        self,
        confusion_matrix: ee.ConfusionMatrix
    ) -> Dict[str, Any]:
        """Extract metrics from an Earth Engine confusion matrix.

        Parameters
        ----------
        confusion_matrix : ee.ConfusionMatrix
            Earth Engine confusion matrix object.

        Returns
        -------
        dict[str, Any]
            Dictionary with overall_accuracy, kappa, producer/user accuracies,
            and the confusion matrix array.

        Notes
        -----
        Earth Engine uses the term "consumers accuracy"; this is mapped to
        "user_accuracy" to align with standard remote sensing terminology.
        """
        try:
            overall_accuracy = confusion_matrix.accuracy().getInfo()
            kappa = confusion_matrix.kappa().getInfo()

            producers_accuracy = np.array(
                confusion_matrix.producersAccuracy().getInfo()
            ).flatten().tolist()

            user_accuracy = np.array(
                confusion_matrix.consumersAccuracy().getInfo()
            ).flatten().tolist()

            cm_info = confusion_matrix.getInfo()
            cm_array = cm_info['array'] if isinstance(cm_info, dict) else cm_info

            return {
                'overall_accuracy': overall_accuracy,
                'kappa': kappa,
                'producers_accuracy': producers_accuracy,
                'user_accuracy': user_accuracy,
                'cm_array': cm_array
            }

        except Exception as e:
            raise RuntimeError(f"Failed to extract confusion matrix data: {str(e)}")

    def run_accuracy_assessment(
        self,
        lcmap: ee.Image,
        validation_data: ee.FeatureCollection,
        class_property: str,
        scale: int = 30,
        confidence: float = 0.95
    ) -> Tuple[bool, Dict[str, Any]]:
        """Perform a thematic accuracy assessment from validation data.

        Parameters
        ----------
        lcmap : ee.Image
            Land cover classification image (must include a 'classification' band).
        validation_data : ee.FeatureCollection
            Validation points containing ground truth class labels.
        class_property : str
            Property name in validation_data for ground truth class labels.
        scale : int, optional
            Sampling scale in meters (default 30).
        confidence : float, optional
            Confidence level for overall accuracy interval (default 0.95).

        Returns
        -------
        tuple[bool, dict]
            (success, results) where results contains accuracy metrics or an error key.
        """
        try:
            is_valid, error_msg = self.validate_assessment_inputs(
                lcmap, validation_data, class_property, scale
            )
            if not is_valid:
                return False, {"error": error_msg}

            validation_sample = lcmap.select('classification').sampleRegions(
                collection=validation_data,
                properties=[class_property],
                scale=scale,
                geometries=False,
                tileScale=4
            )

            confusion_matrix = validation_sample.errorMatrix(class_property, 'classification')
            cm_data = self._extract_confusion_matrix_data(confusion_matrix)

            n_correct = int(np.trace(np.array(cm_data['cm_array'])))
            n_total = int(np.sum(np.array(cm_data['cm_array'])))
            ci_lower, ci_upper = self._calculate_confidence_interval(n_correct, n_total, confidence)

            f1_scores = self._calculate_f1_scores(
                cm_data['producers_accuracy'],
                cm_data['user_accuracy']
            )

            return True, {
                'overall_accuracy': cm_data['overall_accuracy'],
                'kappa': cm_data['kappa'],
                'producer_accuracy': cm_data['producers_accuracy'],
                'user_accuracy': cm_data['user_accuracy'],
                'f1_scores': f1_scores,
                'confusion_matrix': cm_data['cm_array'],
                'overall_accuracy_ci': (ci_lower, ci_upper),
                'confidence_level': confidence,
                'n_total': n_total,
                'n_correct': n_correct,
                'scale': scale
            }

        except Exception as e:
            error_msg = f"Accuracy assessment failed: {str(e)}"
            logger.error(error_msg)
            return False, {"error": error_msg}

    @staticmethod
    def format_accuracy_summary(results: Dict[str, Any]) -> Dict[str, str]:
        """Format accuracy results for display.

        Parameters
        ----------
        results : dict
            Metrics returned by :meth:`run_accuracy_assessment`.

        Returns
        -------
        dict[str, str]
            Formatted strings for display (percentages and summary values).
        """
        if 'error' in results:
            return results

        return {
            'overall_accuracy': f"{results['overall_accuracy']*100:.2f}%",
            'kappa': f"{results['kappa']:.3f}",
            'confidence_interval': (
                f"{results['overall_accuracy_ci'][0]*100:.2f}% - "
                f"{results['overall_accuracy_ci'][1]*100:.2f}%"
            ),
            'sample_size': str(results['n_total'])
        }
