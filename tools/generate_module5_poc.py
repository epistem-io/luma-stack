"""
Tool to automatically generate or update the Module 5 Predictors & Sample Quality Proof of Concept (PoC) Jupyter Notebook.

Whenever new predictors (indices, terrain, radar) or sample quality methods are added to `src/luma_ge/`,
running this script:
    python tools/generate_module5_poc.py

will inspect the live PredictorCatalog and regenerate `notebooks/Module5_Predictors_POC.ipynb`
with verified Earth Engine execution code, non-parametric Silhouette & t-SNE/PCA quality checks,
and geemap visualizers.
"""

import os
import nbformat as nbf
from luma_ge.predictor import predictor_catalog


def generate_module5_notebook(output_path="notebooks/Module5_Predictors_POC.ipynb"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    catalog = predictor_catalog.get_all_catalog()
    spectral_indices = [k for k, v in catalog.items() if v.get("type") == "spectral"]

    nb = nbf.v4.new_notebook()
    nb['metadata'] = {
        'language_info': {'name': 'python'},
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
    }

    cells = []

    def add_md(text):
        cells.append(nbf.v4.new_markdown_cell(text.strip()))

    def add_code(text):
        cells.append(nbf.v4.new_code_cell(text.strip()))

    # ==============================================================================
    # Title & Overview
    # ==============================================================================
    add_md(f"""
# Module 5 & 4: Predictor Calculation & Predictor Stack Quality Analysis

## Overview
This Jupyter Notebook serves as the **executable specification and Proof of Concept (PoC)** for **Module 5 (Predictor Calculation)** and **Module 4 (Non-Parametric Predictor Stack Separability & Visual Diagnosis)** in the Luma Geospatial Engine (`luma-ge`). 

### Live Catalog Status
- **Total Predictors Registered**: {len(catalog)}
- **Spectral Indices Registered**: {len(spectral_indices)} ({', '.join(spectral_indices)})
- **Categories Covered**: Vegetation, Water, Built-up & Bare Soil, Topography, Radar Polarization

### Key Features Demonstrated:
1. **Sensor-Aware Predictor Catalog**: Dynamic filtering based on optical sensor (`LANDSAT` vs. `SENTINEL-2`) and temporal windows.
2. **Spectral Indices Engine**: Automated computation via `spyndex` and `luma-ge.predictor`.
3. **Terrain Predictors**: Elevation, Slope, Aspect, and Landform morphometry derived from DEMs (`NASADEM`, `ALOS`).
4. **SAR Radar Predictors**: Sentinel-1 C-band ($VV$, $VH$, ratio) and ALOS PALSAR / PALSAR-2 L-band ($HH$, $HV$, ratio).
5. **End-to-End Orchestration**: High-level execution via `PredictorCalculation.compute_predictors()`, maintaining complete contract compatibility with `luma-be`.
6. **Predictor Stack Quality & Separability Analysis (`sample_data_quality.py`)**:
   - **Non-Parametric Silhouette Score (`method='SILHOUETTE'`)**: Overall, per-class (`Negative_Ratio` outlier detection), and pairwise class separability.
   - **Predictor Set Comparison (`compare_predictor_sets`)**: Quantitative comparison of *Optical Only* vs. *Full Predictor Stack* vs. *Custom / RFECV Subset*.
   - **Dimensionality Reduction (`plot_tsne` & `plot_pca`)**: Interactive 2D/3D t-SNE cluster maps and PCA biplots with predictor loading arrows.
7. **Interactive Map Visualization**: Multi-layer rendering using `geemap`.
""")

    # ==============================================================================
    # Section 1: Setup & Initialization
    # ==============================================================================
    add_md("""
---
## 1. Environment Setup & Earth Engine Initialization
We initialize Google Earth Engine using `luma-ge`'s lazy initialization mechanism and import required geospatial libraries.
""")

    add_code(r"""
import ee
import geemap
import pandas as pd
from luma_ge.ee_config import ensure_ee_initialized
from luma_ge.data_acquisition import Reflectance_Data, final_Image
from luma_ge.predictor import (
    PredictorCalculation,
    predictor_catalog,
    SpectralCalculator,
    sensor_detector,
    terrain_calculator,
    distance_calculator
)
from luma_ge.sar import get_sar_data
from luma_ge.sample_data_quality import sample_quality, spectral_plotter
#add code to manually initialize the GEE
luma_ge.authenticate_manually()
# Initialize Earth Engine
ensure_ee_initialized()
print("Google Earth Engine initialized successfully!")
""")

    # ==============================================================================
    # Section 2: Area of Interest (AOI), Time Window & Sample Data Placeholders
    # ==============================================================================
    add_md("""
---
## 2. Area of Interest (AOI), Time Window & Training Sample Placeholders
Replace the placeholders below with your target **AOI** and **Training Sample Data** (Shapefile, GeoDataFrame, or GEE FeatureCollection asset) when running your live verification.
""")

    add_code(r"""
#Define Area of Interest (AOI)
#Kambera, NTT
aoi = ee.FeatureCollection('projects/ee-agilfahrezy60/assets/Sesi1_AOI_Kambera_NTT')
#time period
start_date = '2025-01-01'
end_date = '2025-12-31'

#Training data
roi_path = ee.FeatureCollection('projects/ee-agilfahrezy60/assets/Sesi1_Data_Latih_Kambera')   
class_property = 'ID_Kelas' #custom define (class id)                          
class_name_property = 'KelasTupla' #custom define (class name)                
""")

    # ==============================================================================
    # Section 3: Sensor-Aware Predictor Availability
    # ==============================================================================
    add_md("""
---
## 3. Predictor Catalog & Sensor-Aware Availability Check
`PredictorCatalog` dynamically defines compatibility rules for all predictors.
- **Landsat-Specific**: e.g., `EBBI` (requires Thermal and SWIR).
- **Sentinel-2 Specific**: e.g., `RENDVI`, `MCARI` (require Red-Edge bands).
- **Radar**: Sentinel-1 ($VV$, $VH$ available from Oct 2014) vs. ALOS PALSAR ($HH$, $HV$).
""")

    add_code(r"""
predictor_service = PredictorCalculation()

# Query available predictors for Sentinel-2
s2_catalog = predictor_service.get_available_predictors(
    start_date=start_date,
    end_date=end_date,
    sensor='SENTINEL'
)

# Query available predictors for Landsat
landsat_catalog = predictor_service.get_available_predictors(
    start_date=start_date,
    end_date=end_date,
    sensor='LANDSAT'
)

# Access the inner predictors dictionaries
s2_preds = s2_catalog.get('predictors', {})
landsat_preds = landsat_catalog.get('predictors', {})

# Convert to a comparison DataFrame for inspection
df_comparison = pd.DataFrame([
    {
        "Predictor": name,
        "Category": meta.get('category', ''),
        "Sensor Target": meta.get('sensor_compatibility', meta.get('sensor', 'ALL')),
        "Sentinel-2": "Available" if s2_preds.get(name, {}).get('available') else f"Excluded: {s2_preds.get(name, {}).get('reason')}",
        "Landsat": "Available" if landsat_preds.get(name, {}).get('available') else f"Excluded: {landsat_preds.get(name, {}).get('reason')}"
    }
    for name, meta in s2_preds.items()
])

print(f"Total Predictors in Catalog: {len(df_comparison)}")
df_comparison
""")

    # ==============================================================================
    # Section 4: Optical Data Acquisition
    # ==============================================================================
    add_md("""
---
## 4. Optical Imagery Acquisition (Module 1 Handshake)
We acquire standard cloud-masked, harmonized optical composites:
1. **Sentinel-2 Level-2A** via `ref_data.get_s2_optical_data()`
2. **Landsat 8 Level-2** via `ref_data.get_optical_data()`
""")

    add_code(r"""
ref_data = Reflectance_Data()
processor = final_Image()

# 1. Acquire Sentinel-2 Surface Reflectance Collection & Composite
print("Fetching Sentinel-2 collection...")
s2_col, s2_stats = ref_data.get_s2_optical_data(
    aoi=aoi,
    start_date=start_date,
    end_date=end_date,
    cloud_cover=40,
    compute_detailed_stats=False
)
s2_comp_res = processor.get_temporal_composite(s2_col, aoi, reducer='median')
s2_composite = s2_comp_res['image']
print("Sentinel-2 Bands:", s2_composite.bandNames().getInfo())

# 2. Acquire Landsat 8 Surface Reflectance Collection & Composite
print("\nFetching Landsat 8 collection...")
l8_col, l8_stats = ref_data.get_optical_data(
    aoi=aoi,
    start_date=start_date,
    end_date=end_date,
    optical_data='L8_SR',
    cloud_cover=40,
    compute_detailed_stats=False
)
l8_comp_res = processor.get_temporal_composite(l8_col, aoi, reducer='median')
l8_composite = l8_comp_res['image']
print("Landsat 8 Bands:", l8_composite.bandNames().getInfo())
""")

    # ==============================================================================
    # Section 5: Individual Predictor Modules
    # ==============================================================================
    add_md("""
---
## 5. Individual Predictor Components
Standalone computation of each predictor component:
1. **Spectral Indices** (`SpectralCalculator`)
2. **Terrain Predictors** (`terrain_calculator`)
3. **Radar SAR Polarizations** (`sar.py`)
4. **Distance & Accessibility Predictors** (`distance_calculator`)
""")

    add_md("""
### 5.1 Spectral Indices Calculation
Computing standard indices alongside sensor-specific indices:
- `RENDVI` and `MCARI` computed on Sentinel-2
- `DBSI` and `MBI` computed on Landsat-8
""")

    add_code(r"""
# Sentinel-2 Indices (Testing Red-Edge index: RENDVI, MCARI)
s2_calc = SpectralCalculator()
s2_indices = s2_calc.calculate_indices(
    image=s2_composite,
    index_list=['NDVI', 'EVI', 'MNDWI', 'RENDVI', 'MCARI'],
    sensor='SENTINEL'
)
print("Sentinel-2 Computed Indices:", s2_indices.bandNames().getInfo())

# Landsat-8 Indices (Testing Soil and Water: DBSI, MBI, MNDWI)
l8_calc = SpectralCalculator()
l8_indices = l8_calc.calculate_indices(
    image=l8_composite,
    index_list=['NDVI', 'MNDWI', 'DBSI', 'MBI'],
    sensor='LANDSAT'
)
print("Landsat-8 Computed Indices:", l8_indices.bandNames().getInfo())
""")

    add_md("""
### 5.2 Terrain Predictors (Topography & Geomorphology)
Using `terrain_calculator` to extract Elevation, Slope, Aspect, and morphological Landform classes.
""")

    add_code(r"""
tc = terrain_calculator()

# 1. Elevation (from NASADEM)
dem_elevation = tc.calculate_elevation(aoi)
print("Elevation Bands:", dem_elevation.bandNames().getInfo())

# 2. Morphological Landforms (Penumbral, Ridges, Valleys, Plains)
landform_image = tc.get_landform(aoi)
print("Landform Bands:", landform_image.bandNames().getInfo())
""")

    add_md("""
### 5.3 Radar SAR Backscatter (Sentinel-1 & ALOS PALSAR)
Using `get_sar_data()` to extract C-band ($VV$, $VH$) and L-band ($HH$, $HV$).
""")

    add_code(r"""
sar_service = get_sar_data()

# Sentinel-1 C-Band
s1_sar = sar_service.get_sentinel1(
    aoi=aoi,
    start_date=start_date,
    end_date=end_date,
    orbit_pass='BOTH',
    reducer='median',
    include_ratio=True
)
print("Sentinel-1 SAR Bands:", s1_sar.bandNames().getInfo())

# ALOS PALSAR L-Band (with automatic gap fallback)
palsar_sar = sar_service.get_palsar(
    aoi=aoi,
    start_date=start_date,
    end_date=end_date,
    allow_nearest_year_fallback=True
)
print("ALOS PALSAR SAR Bands:", palsar_sar.bandNames().getInfo())
""")

    add_md("""
### 5.4 Distance & Accessibility Predictors (GRIP4 Main Roads, S2COAST-2023, HRSL)
Using `distance_calculator` with Earth Engine's linear-time `fastDistanceTransform()` clipped to the user-defined `aoi`:
- `dist_roads`: On-the-fly from Global Roads Inventory Project (`GRIP4`, main roads `GP_RTP` in `[1, 2, 3]`) or pre-computed asset (`road_source='precomputed'`)
- `dist_coast`: S2COAST-2023 (`projects/sat-io/open-datasets/S2COAST-2023`) clipped to `aoi`
- `dist_settlement`: High Resolution Settlement Layer (`HRSL`)
""")

    add_code(r"""
dc = distance_calculator()

# Compute all 3 distance predictors on-the-fly clipped to AOI
distance_stack = dc.calculate_distance_metrics(
    aoi=aoi,
    max_dist=50000,
    in_meters=True,
    include_roads=True,
    include_coast=True,
    include_settlement=True,
    road_source="grip4",
    main_roads_only=True
)
print("Distance Predictor Bands:", distance_stack.bandNames().getInfo())
""")

    # ==============================================================================
    # Section 6: End-to-End Orchestrated Pipeline
    # ==============================================================================
    add_md("""
---
## 6. End-to-End Pipeline Orchestration: `compute_predictors()`
This is the primary method called by the backend service (`lulc_map.py` and `thematic.py`).

Notice that the configuration dynamically pulls **all registered spectral indices directly from `PredictorCatalog`**, so any new index added to `predictor.py` is automatically computed and tested here!
""")

    add_code(r"""
# Dynamically pull all registered spectral indices from the catalog
all_indices = [
    name for name, meta in predictor_catalog.get_all_catalog().items()
    if meta.get("type") == "spectral"
]

print(f"Testing with all {len(all_indices)} registered spectral indices:")
print(all_indices)

# Define full predictor configuration (matching backend payload schema)
predictor_config = {
    "individual_predictors": {
        "elevation": True,
        "slope": True,
        "aspect": True,
        "landform": True,
        "spectral_indices": all_indices,
        "sentinel1": True,
        "palsar": True
    }
}

orchestrator = PredictorCalculation()

# Execute on Sentinel-2
print("\n=== Running compute_predictors on Sentinel-2 ===")
s2_result = orchestrator.compute_predictors(
    composite=s2_composite,
    aoi=aoi,
    predictor_config=predictor_config,
    collection=s2_col
)

print(f"Status: Success={s2_result['success']}")
print(f"Detected Sensor: {s2_result['sensor']}")
print(f"Total Output Bands: {s2_result['total_bands']}")
print("Output Band Names:")
print(s2_result['stacked_predictors'].bandNames().getInfo())

if s2_result['warnings']:
    print("\nExpected Warnings (graceful skipping of Landsat-only indices):")
    for w in s2_result['warnings']:
        print(f"  - {w}")
""")

    add_code(r"""
# Execute on Landsat 8
print("\n=== Running compute_predictors on Landsat-8 ===")
l8_result = orchestrator.compute_predictors(
    composite=l8_composite,
    aoi=aoi,
    predictor_config=predictor_config,
    collection=l8_col
)

print(f"Status: Success={l8_result['success']}")
print(f"Detected Sensor: {l8_result['sensor']}")
print(f"Total Output Bands: {l8_result['total_bands']}")
print("Output Band Names:")
print(l8_result['stacked_predictors'].bandNames().getInfo())

if l8_result['warnings']:
    print("\nExpected Warnings (graceful skipping of Sentinel-only indices):")
    for w in l8_result['warnings']:
        print(f"  - {w}")
""")

    # ==============================================================================
    # Section 7: Predictor Stack & Sample Quality Analysis (Silhouette, t-SNE, PCA)
    # ==============================================================================
    add_md("""
---
## 7. Predictor Stack & Sample Quality Analysis (`sample_data_quality.py`)
Once the full predictor stack (`s2_result['stacked_predictors']` or `l8_result['stacked_predictors']`) is computed, we evaluate how well the selected predictors separate our training classes using:
1. **Non-Parametric Silhouette Score (`method='SILHOUETTE'`)**: Evaluates cluster cohesion and separation in standardized predictor space without assuming Gaussian normality.
2. **Predictor Subset Comparison (`compare_predictor_sets`)**: Compares *Optical Only* vs. *Optical + Indices* vs. *Full Stack* vs. *RFECV / Custom Subset*.
3. **Dimensionality Reduction (`plot_tsne` & `plot_pca`)**: Visualizes 20+ predictor dimensions in 2D/3D space to inspect overlapping classes, multi-modal clusters, and predictor loading arrows.
""")

    add_md("""
### 7.1 Extract Predictor Values at Training Sample Locations
Select which stacked image to analyze (`target_stacked_image`) and extract pixel values for `labeled_roi`.
""")

    add_code(r"""
# Choose which stacked predictor image to evaluate (Sentinel-2 or Landsat-8)
target_stacked_image = s2_result['stacked_predictors']  # or l8_result['stacked_predictors']
sampling_scale = 30  # Resolution in meters (e.g., 10 or 30)
max_pixels_per_class = 5000

if labeled_roi is None:
    raise ValueError(
        "Please set `labeled_roi` (and `class_property` / `class_name_property`) in Section 2 "
        "before running Section 7."
    )

analyzer = sample_quality(
    training_data=labeled_roi,
    image=target_stacked_image,
    class_property=class_property,
    region=aoi,
    class_name_property=class_name_property
)

print("Extracting predictor stack values at sample locations on GEE...")
pixel_extract = analyzer.extract_spectral_values(
    scale=sampling_scale,
    max_pixels_per_class=max_pixels_per_class
)
print(f"Extracted {len(pixel_extract)} sample pixels across {len(analyzer.band_names)} predictor bands.")
pixel_extract.head()
""")

    add_md("""
### 7.2 Per-Class & Pairwise Non-Parametric Silhouette Analysis
- **Per-Class Table**: Inspect `Mean_Silhouette` and `Negative_Ratio` (fraction of samples closer to another class than their own, indicating mislabeled polygons or class overlap).
- **Pairwise Table**: Inspect which specific class pairs have low separation (`Silhouette_Score < 0.25`).
""")

    add_code(r"""
# 1. Per-Class Silhouette Summary
class_silhouette_df = analyzer.get_class_silhouette_df(pixel_extract)
print("=== Per-Class Silhouette Quality Summary ===")
display(class_silhouette_df)

# 2. Pairwise Class Separability using Non-Parametric Silhouette Score
pairwise_silhouette_df = analyzer.get_separability_df(pixel_extract, method='SILHOUETTE')
print("\n=== Pairwise Class Separability (Silhouette Score) ===")
display(pairwise_silhouette_df)

# 3. Overall Separability Summary Count (Good / Weak / Poor pairs)
sep_summary_df = analyzer.sum_separability(pixel_extract, method='SILHOUETTE')
print("\n=== Silhouette Separability Summary ===")
display(sep_summary_df)
""")

    add_md("""
### 7.3 Compare Candidate Predictor Subsets (`compare_predictor_sets`)
Compare different combinations of predictors (e.g., raw optical bands vs. full predictor stack vs. your RFECV-selected subset) to verify whether adding terrain, radar, or specific indices improves class separation.
""")

    add_code(r"""
available_stack_bands = [b for b in analyzer.band_names if b in pixel_extract.columns]

# Define candidate predictor sets to compare (customize `rfecv_selected_bands` with your RFECV output)
optical_only_bands = [
    b for b in ['AEROSOL', 'BLUE', 'GREEN', 'RED', 'RED_EDGE1', 'RED_EDGE2', 'RED_EDGE3', 'NIR', 'RED_EDGE4', 'SWIR1', 'SWIR2']
    if b in available_stack_bands
]
optical_plus_indices = [
    b for b in available_stack_bands
    if b not in ['elevation', 'slope', 'aspect', 'landform', 'S1_VV', 'S1_VH', 'S1_VV_VH', 'PALSAR_HH', 'PALSAR_HV', 'PALSAR_HH_HV']
]

# PLACEHOLDER: Replace with your RFECV selected band list if available
rfecv_selected_bands = [
    b for b in ['RED', 'NIR', 'SWIR1', 'NDVI', 'MNDWI', 'RENDVI', 'DBSI', 'elevation', 'slope', 'S1_VV', 'S1_VH']
    if b in available_stack_bands
]

predictor_sets_to_compare = {
    "1. Optical Bands Only": optical_only_bands,
    "2. Optical + Spectral Indices": optical_plus_indices,
    "3. Full Predictor Stack (All Bands)": available_stack_bands,
    "4. RFECV / Custom Selected Subset": rfecv_selected_bands,
}

comparison_df = analyzer.compare_predictor_sets(
    df=pixel_extract,
    predictor_sets=predictor_sets_to_compare,
    max_samples=5000
)
print("=== Predictor Subset Quality Comparison ===")
display(comparison_df)
""")

    add_md("""
### 7.4 Dimensionality Reduction & Visual Diagnosis (`plot_tsne` & `plot_pca`)
- **`plot_tsne()`**: Non-linearly projects all selected predictors into 2D/3D space to reveal cluster separation, multi-modal subclasses, and isolated mislabeled points.
- **`plot_pca()`**: Projects predictors onto the top Principal Components and overlays **predictor loading vectors** showing which bands pull which classes apart.
""")

    add_code(r"""
plotter = spectral_plotter(analyzer)

# Choose which band subset to visualize (e.g., available_stack_bands or rfecv_selected_bands)
bands_to_visualize = available_stack_bands

# 1. Interactive 2D t-SNE Projection
fig_tsne_2d = plotter.plot_tsne(
    df=pixel_extract,
    bands=bands_to_visualize,
    n_components=2,
    perplexity=30.0,
    max_samples=3000,
    interactive=True
)
fig_tsne_2d.show()
""")

    add_code(r"""
# 2. Interactive 3D t-SNE Projection (Optional exploratory view)
fig_tsne_3d = plotter.plot_tsne(
    df=pixel_extract,
    bands=bands_to_visualize,
    n_components=3,
    perplexity=30.0,
    max_samples=3000,
    interactive=True
)
fig_tsne_3d.show()
""")

    add_code(r"""
# 3. Interactive 2D PCA Biplot with Top Predictor Loading Arrows
fig_pca_2d = plotter.plot_pca(
    df=pixel_extract,
    bands=bands_to_visualize,
    n_components=2,
    max_samples=5000,
    show_loadings=True,
    top_n_loadings=8,
    interactive=True
)
fig_pca_2d.show()
""")

    # ==============================================================================
    # Section 8: Interactive Map Visualization
    # ==============================================================================
    add_md("""
---
## 8. Interactive Multi-Layer Visualization (`geemap`)
Visual inspection of predictors over the AOI to verify radiometric quality, spatial resolution alignment, and edge contrasts.
""")

    add_code(r"""
# Initialize Geemap canvas centered on AOI
Map = geemap.Map()
Map.centerObject(aoi, 11)

# 1. Base Optical Imagery (Natural Color & False Color NIR)
s2_stacked = s2_result['stacked_predictors']
Map.addLayer(
    s2_stacked,
    {'bands': ['RED', 'GREEN', 'BLUE'], 'min': 0, 'max': 3000},
    '1. S2 True Color (RGB)'
)
Map.addLayer(
    s2_stacked,
    {'bands': ['NIR', 'RED', 'GREEN'], 'min': 0, 'max': 4500},
    '2. S2 False Color Infrared (CIR)'
)

# 3. Spectral Indices
# NDVI (Vegetation Greenness)
ndvi_vis = {'bands': ['NDVI'], 'min': 0.0, 'max': 0.85, 'palette': ['blue', 'white', 'green']}
Map.addLayer(s2_stacked, ndvi_vis, '3. NDVI (Vegetation)')

# MNDWI (Water / Moisture)
mndwi_vis = {'bands': ['MNDWI'], 'min': -0.5, 'max': 0.5, 'palette': ['brown', 'yellow', 'cyan', 'blue']}
Map.addLayer(s2_stacked, mndwi_vis, '4. MNDWI (Water/Moisture)')

# RENDVI (Sentinel-2 Red Edge NDVI)
rendvi_vis = {'bands': ['RENDVI'], 'min': 0.0, 'max': 0.7, 'palette': ['red', 'yellow', 'darkgreen']}
Map.addLayer(s2_stacked, rendvi_vis, '5. RENDVI (Red Edge)')

# 4. Topography & Terrain
elev_vis = {'bands': ['elevation'], 'min': 0, 'max': 150, 'palette': ['darkblue', 'teal', 'lightgreen', 'brown', 'white']}
Map.addLayer(s2_stacked, elev_vis, '6. NASADEM Elevation')

landform_vis = {'bands': ['landform'], 'min': 11, 'max': 42, 'palette': ['141414', '383838', '808080', 'EBEBEB']}
Map.addLayer(s2_stacked, landform_vis, '7. Morphological Landforms')

# 5. Radar Backscatter (Sentinel-1 SAR)
s1_sar_vis = {'bands': ['S1_VV', 'S1_VH', 'S1_VV_VH'], 'min': [-18, -25, 0.2], 'max': [-4, -10, 0.9]}
Map.addLayer(s2_stacked, s1_sar_vis, '8. Sentinel-1 SAR (VV/VH/Ratio)')

# Add AOI boundary outline
aoi_outline = ee.Image().paint(ee.FeatureCollection([ee.Feature(aoi)]), color=1, width=2)
Map.addLayer(aoi_outline, {'palette': ['FF0000']}, 'AOI Boundary')

# Overlay training sample polygons/points if loaded
if labeled_roi is not None:
    Map.addLayer(labeled_roi, {}, 'Training Samples (ROI)')

# Render the interactive map
Map
""")

    # ==============================================================================
    # Section 9: Technical Contract for Backend & Frontend Teams
    # ==============================================================================
    add_md(f"""
---
## 9. Technical Specification & Contract for Backend & Frontend

### 9.1 Backend Implementation Contract (`luma-be`)
When the frontend submits the predictor configuration, the payload structured below must be stored in `luma.predictor_config` and passed into `compute_predictors()`:

```json
{{
  "individual_predictors": {{
    "elevation": true,
    "slope": true,
    "aspect": true,
    "landform": true,
    "spectral_indices": {spectral_indices},
    "sentinel1": true,
    "palsar": false
  }}
}}
```

> **Important**: The backend whitelist in `application/logic/luma/predictor.py` (`set_predictor_config()`) should accept all predictors defined in `PredictorCatalog.get_all_catalog().keys()`.

### 9.2 Frontend Layer Styling Guide (`luma-fe`)
Recommended layer visualization palettes for frontend map rendering:

| Predictor Type | Band Name | Min | Max | Recommended Palette |
| :--- | :--- | :--- | :--- | :--- |
| **Vegetation** | `NDVI`, `EVI`, `GNDVI` | `0.0` | `0.85` | `['#0000FF', '#FFFFFF', '#00AA00']` |
| **Red Edge** | `RENDVI`, `MCARI` | `0.0` | `0.70` | `['#FF0000', '#FFFF00', '#006400']` |
| **Water** | `MNDWI`, `NDMI`, `AWEIsh` | `-0.5` | `0.50` | `['#8B4513', '#FFFFE0', '#00FFFF', '#0000FF']` |
| **Built-up / Soil** | `EBBI`, `DBSI`, `BLFEI`, `MBI`, `BI` | `-0.5` | `0.80` | `['#228B22', '#FFFF00', '#FF0000', '#800080']` |
| **Elevation** | `elevation` | `0` | `2000` | `['#00008B', '#008080', '#90EE90', '#8B4513', '#FFFFFF']` |
| **SAR Radar** | `S1_VV` / `S1_VH` | `-25 dB` | `0 dB` | Grayscale `['#000000', '#FFFFFF']` |
""")

    nb['cells'] = cells

    with open(output_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)

    print(f"Successfully generated {output_path} with {len(cells)} cells and {len(spectral_indices)} indices!")


if __name__ == '__main__':
    generate_module5_notebook()
