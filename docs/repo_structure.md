<!------------------------------------------------------------------------------------
    This document serves as a tags for connecting Quarto Technical Documentation with python codes 
    in this project
------------------------------------------------------------------------------------->

```text
luma-stack/
├── pyproject.toml                       # Python project metadata and dependencies tracking
├── Dockerfile                           # Docker configuration for reproducible environments
├── .dockerignore
├── .gitignore
├── LICENSE
├── README.md                            # Project overview, setup guide, and references
│
├── src/luma_ge/                         # Core Luma Geospatial Engine Python package
│   ├── __init__.py                      # Package exports, lazy imports, and credential auto-discovery
│   │   ├── def _find_service_account_file()
│   │   ├── def auto_initialize()
│   │   └── def __getattr__()
│   │
│   ├── ee_config.py                     # Module 0: Google Earth Engine authentication & initialization
│   │   ├── def initialize_with_service_account()
│   │   ├── def authenticate_manually()
│   │   ├── def _print_manual_auth_instructions()
│   │   ├── def initialize_earth_engine()
│   │   ├── def ensure_ee_initialized()
│   │   ├── def is_ee_initialized()
│   │   ├── def get_auth_status()
│   │   ├── def print_auth_instructions()
│   │   ├── def reset_ee_initialization()
│   │   └── def setup_earth_engine()
│   │
│   ├── helpers.py                       # Shared GEE asset & administrative boundary utilities
│   │   ├── class GEE_Asset_Manager
│   │   │   ├── def __init__()
│   │   │   ├── def load_asset()
│   │   │   ├── def get_regency_names()
│   │   │   ├── def get_regency_geometry()
│   │   │   └── def convert_to_geodataframe()
│   │   └── def get_aoi_from_gaul()
│   │
│   ├── input_utils.py                   # Vector geometry validation & Earth Engine conversion
│   │   ├── class GeometryValidator
│   │   │   ├── def __init__()
│   │   │   ├── def log()
│   │   │   ├── def _is_valid_coordinate()
│   │   │   ├── def _count_vertices()
│   │   │   ├── def _fix_crs()
│   │   │   ├── def _clean_geometries()
│   │   │   ├── def _validate_points()
│   │   │   ├── def _validate_polygons()
│   │   │   └── def _final_validation()
│   │   ├── class shapefile_validator(GeometryValidator)
│   │   │   └── def validate_and_fix_geometry()
│   │   ├── class kml_validator(GeometryValidator)
│   │   │   ├── def load_and_validate()
│   │   │   ├── def validate_and_fix_geometry()
│   │   │   ├── def _verify_kml_crs()
│   │   │   ├── def _clean_geometries_kml()
│   │   │   └── def _final_validation_kml()
│   │   └── class EE_converter
│   │       ├── def __init__()
│   │       ├── def log()
│   │       ├── def convert_aoi_gdf()
│   │       └── def convert_roi_gdf()
│   │
│   ├── data_acquisition.py              # Module 1: Acquisition of Near-Cloud-Free Satellite Imagery
│   │   ├── def parse_date_input()
│   │   ├── class Reflectance_Data
│   │   │   ├── OPTICAL_DATASETS
│   │   │   ├── S2_DATASETS
│   │   │   ├── HLS_DATASETS
│   │   │   ├── THERMAL_DATASETS
│   │   │   ├── ROY2016_COEFFICIENTS
│   │   │   ├── def __init__()
│   │   │   ├── def has_thermal_capability()
│   │   │   ├── def mask_landsat_sr()
│   │   │   ├── def rename_landsat_bands()
│   │   │   ├── def apply_scale_factors()
│   │   │   ├── def mask_s2_clouds()
│   │   │   ├── def rename_s2_bands()
│   │   │   ├── def apply_s2_scale_factors()
│   │   │   ├── def sharpen_s2_bands()
│   │   │   ├── def get_s2_optical_data()
│   │   │   ├── def get_optical_data()
│   │   │   ├── def get_thermal_bands()
│   │   │   ├── def mask_hls_fmask()
│   │   │   ├── def rename_hls_bands()
│   │   │   ├── def apply_hls_scale_factors()
│   │   │   ├── def get_hls_optical_data()
│   │   │   ├── def apply_brdf_harmonization()
│   │   │   └── def get_multitemporal_optical_data()
│   │   ├── class Reflectance_Stats
│   │   │   ├── def __init__()
│   │   │   ├── def get_collection_statistics()
│   │   │   └── def print_collection_report()
│   │   ├── class final_Image
│   │   │   ├── def __init__()
│   │   │   ├── def calculate_data_coverage()
│   │   │   ├── def get_quality_mosaic()
│   │   │   ├── def get_temporal_composite()
│   │   │   ├── def histogram_match()
│   │   │   ├── def _expected_bands_for_sensor()
│   │   │   └── def get_multitemporal_composites()
│   │   └── class Vis_Params
│   │       ├── BAND_COMBINATIONS
│   │       ├── THERMAL_VIS
│   │       ├── def get_combination_names()
│   │       ├── def get_vis_params()
│   │       ├── def build_custom_vis_params()
│   │       └── def get_thermal_vis_params()
│   │
│   ├── imagery_correction.py            # Auxiliary Module: Topographic (SCSc) & BRDF Corrections
│   │   ├── class Topographic_Correction
│   │   │   ├── DEFAULT_DEM_ASSET
│   │   │   ├── DEFAULT_DEM_BAND
│   │   │   ├── DEFAULT_OPTICAL_BANDS
│   │   │   ├── S2_OPTICAL_BANDS
│   │   │   ├── def __init__()
│   │   │   ├── def compute_illumination_condition()
│   │   │   ├── def apply_scsc_correction()
│   │   │   ├── def correct_image()
│   │   │   └── def correct_collection()
│   │   └── class BRDF_Correction
│   │       ├── DEFAULT_OPTICAL_BANDS
│   │       ├── S2_OPTICAL_BANDS
│   │       ├── BAND_COEFFICIENTS
│   │       ├── def __init__()
│   │       ├── def _line_from_coords()
│   │       ├── def _invert_mask()
│   │       ├── def _where()
│   │       ├── def _value()
│   │       ├── def compute_sun_angles()
│   │       ├── def compute_view_angles()
│   │       ├── def compute_kvol()
│   │       ├── def _correct_band()
│   │       ├── def correct_image()
│   │       └── def correct_collection()
│   │
│   ├── classification_scheme.py         # Module 2: LULC Classification Scheme Definition
│   │   └── class LULC_Scheme_Manager
│   │       ├── def __init__()
│   │       ├── def get_state()
│   │       ├── def set_state()
│   │       ├── def has_classes()
│   │       ├── def get_class_count()
│   │       ├── def validate_class_input()
│   │       ├── def add_class()
│   │       ├── def _reset_edit_mode()
│   │       ├── def _sort_and_update_next_id()
│   │       ├── def _generate_random_color()
│   │       ├── def _generate_distinct_colors()
│   │       ├── def edit_class()
│   │       ├── def delete_class()
│   │       ├── def cancel_edit()
│   │       ├── def process_csv_upload()
│   │       ├── def finalize_csv_upload()
│   │       ├── def get_csv_data()
│   │       ├── def get_dataframe()
│   │       ├── def load_default_scheme()
│   │       ├── def get_default_schemes()
│   │       ├── def auto_detect_csv_columns()
│   │       └── def store_classes_of_interest()
│   │
│   ├── sample_data.py                   # Module 3: Sample Data Generation & Management
│   │   ├── class SyncTrainData
│   │   │   ├── def LoadTrainData()
│   │   │   ├── def SetClassField()
│   │   │   ├── def ValidClass()
│   │   │   ├── def CheckSufficiency()
│   │   │   ├── def FilterTrainAoi()
│   │   │   ├── def TrainDataRaw()
│   │   │   └── def LoadMultitemporalTrainData()
│   │   └── class SplitTrainData
│   │       └── def SplitProcess()
│   │
│   ├── sample_data_quality.py           # Module 4: Sample Data Quality & Separability Analysis
│   │   ├── class sample_quality
│   │   │   ├── def __init__()
│   │   │   ├── def get_display_property()
│   │   │   ├── def class_renaming()
│   │   │   ├── def add_class_names()
│   │   │   ├── def sample_stats()
│   │   │   ├── def get_sample_stats_df()
│   │   │   ├── def extract_spectral_values()
│   │   │   ├── def sample_pixel_stats()
│   │   │   ├── def get_sample_pixel_stats_df()
│   │   │   ├── def _get_feature_columns()
│   │   │   ├── def _stratified_subsample()
│   │   │   ├── def compute_silhouette_analysis()
│   │   │   ├── def get_class_silhouette_df()
│   │   │   ├── def compare_predictor_sets()
│   │   │   ├── def compute_tsne_coordinates()
│   │   │   ├── def compute_pca_coordinates()
│   │   │   ├── def check_class_separability()
│   │   │   ├── def _get_metric_column_name()
│   │   │   ├── def get_separability_df()
│   │   │   ├── def lowest_separability()
│   │   │   ├── def separability_level()
│   │   │   ├── def sum_separability()
│   │   │   ├── def _jeffries_matusita_distance()
│   │   │   ├── def transform_divergence()
│   │   │   ├── def print_analysis_summary()
│   │   │   └── def run_multitemporal_separability()
│   │   └── class spectral_plotter
│   │       ├── def __init__()
│   │       ├── def plot_histogram()
│   │       ├── def plot_boxplot()
│   │       ├── def interactive_scatter_plot()
│   │       ├── def static_scatter_plot()
│   │       ├── def add_elipse()
│   │       ├── def scatter_plot_3d()
│   │       ├── def plot_tsne()
│   │       └── def plot_pca()
│   │
│   ├── predictor.py                     # Module 5: Feature Engineering, Predictors & Feature Selection
│   │   ├── class terrain_calculator
│   │   │   ├── def __init__()
│   │   │   ├── def calculate_elevation()
│   │   │   ├── def calculate_slope()
│   │   │   ├── def calculate_aspect()
│   │   │   └── def get_landform()
│   │   ├── class sensor_detector
│   │   │   ├── BAND_TO_SPYNDEX
│   │   │   ├── SPYNDEX_TO_BAND
│   │   │   ├── def normalize_sensor_name()
│   │   │   ├── def detect_sensor_from_bands()
│   │   │   └── def detect_sensor()
│   │   ├── class predictor_catalog
│   │   │   ├── PREDICTORS
│   │   │   ├── def get_all_catalog()
│   │   │   ├── def check_availability()
│   │   │   └── def check_multitemporal_availability()
│   │   ├── class SpectralCalculator
│   │   │   ├── def __init__()
│   │   │   ├── def get_available_indices()
│   │   │   ├── def get_index_coefficients()
│   │   │   ├── def validate_coefficients()
│   │   │   ├── def validate_index_requirements()
│   │   │   ├── def _build_spyndex_params()
│   │   │   ├── def calculate_indices()
│   │   │   ├── def calculate_indices_with_collection()
│   │   │   └── def get_calculation_summary()
│   │   ├── class distance_calculator
│   │   │   ├── S2COAST_ASSET
│   │   │   ├── GRIP4_REGIONAL_ASSETS
│   │   │   ├── HRSL_ASSET
│   │   │   ├── def __init__()
│   │   │   ├── def _resolve_geometry()
│   │   │   ├── def _resolve_compute_scale()
│   │   │   ├── def _compute_fast_distance()
│   │   │   ├── def extract_s2_coastline()
│   │   │   ├── def extract_grip4_roads()
│   │   │   ├── def calculate_distance_to_coast()
│   │   │   ├── def calculate_distance_to_roads()
│   │   │   ├── def calculate_distance_to_settlement()
│   │   │   └── def calculate_distance_metrics()
│   │   ├── class PredictorCalculation
│   │   │   ├── def __init__()
│   │   │   ├── def detect_sensor()
│   │   │   ├── def get_available_predictors()
│   │   │   ├── def validate_prerequisites()
│   │   │   ├── def _resolve_date_range()
│   │   │   ├── def _resolve_sar_selection()
│   │   │   ├── def compute_predictors()
│   │   │   ├── def prepare_module6_summary()
│   │   │   └── def compute_multitemporal_predictors()
│   │   ├── class EECorrelationAnalysis
│   │   │   ├── def __init__()
│   │   │   ├── def generate_random_samples()
│   │   │   ├── def compute_correlation()
│   │   │   └── def visualize()
│   │   ├── class RFEGEE
│   │   │   ├── def __init__()
│   │   │   ├── def _log()
│   │   │   ├── def _validate_inputs()
│   │   │   ├── def _train_classifier()
│   │   │   ├── def _get_importance_and_accuracy()
│   │   │   ├── def _calculate_step()
│   │   │   ├── def fit()
│   │   │   ├── def get_selected_features()
│   │   │   ├── def get_ranking()
│   │   │   ├── def get_support()
│   │   │   ├── def plot_accuracy_curve()
│   │   │   ├── def plot_feature_rankings()
│   │   │   ├── def plot_rfe_summary()
│   │   │   ├── def get_results_dataframe()
│   │   │   └── def get_accuracy_dataframe()
│   │   ├── class ProgressRF(BaseEstimator, ClassifierMixin)
│   │   │   ├── def __init__()
│   │   │   ├── def fit()
│   │   │   ├── def predict()
│   │   │   ├── def predict_proba()
│   │   │   └── def feature_importances_()
│   │   └── class RFE_CV
│   │       ├── def __init__()
│   │       ├── def _log()
│   │       ├── def _validate_inputs()
│   │       ├── def _extract_dataframe()
│   │       ├── def _estimate_total_fits()
│   │       ├── def fit()
│   │       ├── def get_selected_features()
│   │       ├── def get_ranking()
│   │       ├── def get_support()
│   │       ├── def select_bands()
│   │       ├── def get_results_dataframe()
│   │       └── def plot_feature_importance()
│   │
│   ├── sar.py                           # Auxiliary Module: Synthetic Aperture Radar (Sentinel-1 & ALOS PALSAR)
│   │   └── class get_sar_data
│   │       ├── SAR_DATASETS
│   │       ├── def __init__()
│   │       ├── def _db_to_power()
│   │       ├── def _power_to_db()
│   │       ├── def _reduce()
│   │       ├── def apply_speckle_filter()
│   │       ├── def mask_s1_border_and_angle()
│   │       ├── def get_sentinel1()
│   │       ├── def mask_and_calibrate_palsar()
│   │       ├── def _resolve_palsar_years()
│   │       └── def get_palsar()
│   │
│   ├── classification.py                # Module 6: Model Training & LULC Map Generation
│   │   ├── def extract_class_name_map()
│   │   ├── def build_prob_band_names()
│   │   ├── def compute_log_loss_from_arrays()
│   │   ├── class FeatureExtraction
│   │   │   ├── def __init__()
│   │   │   ├── def random_split()
│   │   │   ├── def stratified_split()
│   │   │   └── def extract_multitemporal_samples()
│   │   └── class Generate_LULC
│   │       ├── def __init__()
│   │       ├── def hard_classification()
│   │       ├── def soft_classification()
│   │       ├── def classify_with_model()
│   │       ├── def get_feature_importance()
│   │       ├── def compute_log_loss()
│   │       ├── def evaluate_model()
│   │       ├── def reclassify_map_by_classes()
│   │       ├── def classify_from_prebuilt()
│   │       ├── def classify_from_prebuilt_multitemporal()
│   │       └── def classify_multitemporal()
│   │
│   ├── accuracy.py                      # Module 7: Thematic Accuracy Assessment
│   │   ├── class sample_size_calculator
│   │   │   ├── def __init__()
│   │   │   ├── def get_pixel_counts_per_class()
│   │   │   ├── def validate_sample_size_inputs()
│   │   │   ├── def calculate_strata_sample()
│   │   │   ├── def generate_stratified_samples()
│   │   │   └── def export_samples_to_shp()
│   │   ├── class validation_error_flag
│   │   │   ├── def __init__()
│   │   │   ├── def classify_validation_points()
│   │   │   └── def generate_popup_html()
│   │   └── class thematic_accuracy
│   │       ├── def __init__()
│   │       ├── def validate_assessment_inputs()
│   │       ├── def _calculate_confidence_interval()
│   │       ├── def _calculate_f1_scores()
│   │       ├── def _extract_confusion_matrix_data()
│   │       ├── def run_accuracy_assessment()
│   │       ├── def format_accuracy_summary()
│   │       ├── def run_multitemporal_assessment()
│   │       └── def compare_accuracy()
│   │
│   ├── multitemporal.py                 # Auxiliary Module: Multitemporal LULC Analysis Orchestration
│   │   ├── def filter_compatible_indices()
│   │   ├── def compute_single_image_indices()
│   │   ├── def _extract_predictor_selections()
│   │   ├── def compute_multitemporal_predictors()
│   │   ├── def normalize_period_configs()
│   │   ├── class PeriodConfig
│   │   │   └── def to_dict()
│   │   └── class MultitemporalAnalysis
│   │       ├── def __init__()
│   │       ├── def acquire_imagery()
│   │       ├── def load_training_data()
│   │       ├── def check_predictor_availability()
│   │       ├── def check_separability()
│   │       ├── def compute_predictors()
│   │       ├── def extract_features()
│   │       ├── def run_feature_selection()
│   │       ├── def run_classification()
│   │       ├── def assess_accuracy()
│   │       └── def run_pipeline()
│   │
│   └── post_classification.py           # Auxiliary Module: Post-Classification Consistency & Change Analysis
│       ├── def _parse_int_id()
│       ├── def _normalize_cell_weight()
│       ├── def _normalize_rules()
│       └── class ConsistencyAnalyzer
│           ├── def __init__()
│           ├── def _load_matrix_source()
│           ├── def validate_transition_matrix()
│           ├── def _require_valid_matrix()
│           ├── def _compute_multiplier()
│           ├── def _resolve_prob_bands()
│           ├── def _extract_aoi_geometry()
│           ├── def cross_tabulate()
│           ├── def _emission_log_probs()
│           ├── def _log_transition_matrix()
│           ├── def _apply_rules_override()
│           ├── def _run_hmm_anchored_pair()
│           ├── def _run_viterbi_sequence()
│           ├── def _run_rule_reclassify_pair()
│           ├── def _summarize_correction()
│           ├── def correct_inconsistencies()
│           └── def correct_multitemporal_sequence()
│
├── tests/                               # Unit and integration test suite
│   ├── test_multitemporal_phase1.py
│   ├── test_multitemporal_phase2.py
│   ├── test_multitemporal_phase3_4.py
│   ├── test_multitemporal_phase5.py
│   ├── test_multitemporal_phase6.py
│   ├── test_rfe_local.py
│   ├── test_sample_quality_nonparametric.py
│   ├── test_sar_predictors.py
│   └── test_sensor_predictors.py
│
├── docs/                                # Supporting documentation and assets
│   ├── img/                             # Documentation images and GIFs
│   ├── js/                              # Supporting JavaScript resources
│   ├── MULTITEMPORAL_WORKFLOW.md        # Multitemporal analysis & consistency workflow guide
│   ├── repo_structure.md                # Repository navigation structure (this file)
│   └── Visio_and_code_tags.md           # MS Visio workflow to Python code mapping tags
│
├── data/                                # Example datasets for testing and tutorials
│   ├── AOI_Kab_PagarAlam.kmz
│   ├── area_of_interest.shp             # (+ .dbf, .shx, .prj, .cpg, .sbn, .sbx, .zip)
│   ├── MODUL2_Skema_Klasifikasi_Kambera.csv
│   ├── Tes_classification_scheme.csv
│   ├── training_points.shp              # (+ .dbf, .shx, .prj, .cpg, .sbn, .sbx, .shp.xml, .zip)
│   └── validation_samples - labeled.shp # (+ .dbf, .shx, .prj)
│
└── notebooks/                           # Jupyter notebooks for tutorials and exploratory analysis
    ├── Module_implementation.ipynb
    ├── LumaGE_Phase2_implementation.ipynb
    ├── Module5_Predictors_POC.ipynb
    ├── Multitemporal_Live_EE_Test.ipynb
    ├── Multitemporal_Model_Transfer_Example.ipynb
    ├── RFE_testing.ipynb
    └── SAR_Predictors_POC.ipynb
```
