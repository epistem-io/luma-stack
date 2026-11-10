import nbformat as nbf

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
add_md("""
# Module 5: Predictor Calculation Implementation Blueprint

## Overview
This Jupyter Notebook serves as the **executable specification and Proof of Concept (PoC)** for **Module 5: Predictor Calculation** in the Luma Geospatial Engine (`luma-ge`). 
### What Module 5 Does:
Module 5 transforms raw optical and radar satellite imagery, alongside digital elevation models, into an expanded stack of geospatial predictors for Land Use / Land Cover (LULC) classification (Module 6).

### Key Features Demonstrated in this Notebook:
1. **Sensor-Aware Predictor Catalog**: Dynamic filtering of indices based on optical sensor (`LANDSAT` vs. `SENTINEL-2`) and temporal windows.
2. **Spectral Indices Engine**: Automated computation of Vegetation, Water, and Built-up/Soil indices via `spyndex` and `luma-ge.predictor`.
3. **Terrain Predictors**: Elevation, Slope, Aspect, and Landform morphometry derived from DEMs (`NASADEM`, `Copernicus`).
4. **SAR Radar Predictors**: Sentinel-1 C-band backscatter ($VV$, $VH$) and ALOS PALSAR / PALSAR-2 L-band backscatter ($HH$, $HV$) with gap fallback.
5. **End-to-End Orchestration**: High-level execution via `PredictorCalculation.compute_predictors()`, maintaining complete contract compatibility with `luma-be`.
6. **Interactive Visualization**: Multi-layer rendering using `geemap` for visual inspection of feature boundaries and gradients.
7. **BE / FE Contract Specs**: Complete JSON payloads and layer styling palettes for direct web application implementation.
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
from luma_ge.data_acquisition import Reflectance_Data
from luma_ge.predictor import (
    PredictorCalculation,
    PredictorCatalog,
    SpectralCalculator,
    SensorDetector,
    terrain_calculator
)
from luma_ge.sar import get_sar_data

# Initialize Earth Engine
ensure_ee_initialized()
print("Google Earth Engine initialized successfully!")
""")

# ==============================================================================
# Section 2: Area of Interest (AOI) & Date Range
# ==============================================================================
add_md("""
---
## 2. Area of Interest (AOI) & Time Window
We define a representative testing region in **South Sumatra, Indonesia** featuring a diverse mix of:
- Dense natural & secondary forest
- Agricultural plantations (oil palm / rubber)
- Peatlands, inland water bodies, and coastal wetlands
- Urban / built-up settlements
""")

add_code(r"""
# Define Area of Interest (AOI) - Banyuasin / Palembang, South Sumatra
aoi = ee.Geometry.Rectangle([104.60, -3.10, 104.90, -2.80])

# Temporal window: Annual composite window for 2021
start_date = '2021-01-01'
end_date = '2021-12-31'

print("AOI Coordinates:", aoi.coordinates().getInfo())
print(f"Temporal Window: {start_date} to {end_date}")
""")

# ==============================================================================
# Section 3: Sensor-Aware Predictor Availability
# ==============================================================================
add_md("""
---
## 3. Predictor Catalog & Sensor-Aware Availability Check
`PredictorCatalog` defines compatibility rules for all predictors.
- **Landsat-Specific**: `EBBI` (Enhanced Built-Up and Bareness Index requires Thermal and SWIR).
- **Sentinel-2 Specific**: `RENDVI` (Red-Edge NDVI), `MCARI` (Modified Chlorophyll Absorption in Reflectance Index require Red-Edge bands).
- **Radar**: Sentinel-1 ($VV$, $VH$ available from Oct 2014) vs. ALOS PALSAR ($HH$, $HV$ active 2007-2010, 2015-present).

The frontend calls the backend, which queries `PredictorCalculation.get_available_predictors()` to know which indices to display or grey out.
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

# Convert to a comparison DataFrame for inspection
df_comparison = pd.DataFrame([
    {
        "Predictor": name,
        "Category": meta['category'],
        "Sensor Target": meta['sensor'],
        "Sentinel-2": "Available" if s2_catalog[name]['available'] else f"Excluded: {s2_catalog[name]['reason']}",
        "Landsat": "Available" if landsat_catalog[name]['available'] else f"Excluded: {landsat_catalog[name]['reason']}"
    }
    for name, meta in s2_catalog.items()
])

print(f"Total Predictors in Catalog: {len(df_comparison)}")
df_comparison
""")

# ==============================================================================
# Section 4: Optical Data Acquisition (Module 1 Handshake)
# ==============================================================================
add_md("""
---
## 4. Optical Imagery Acquisition (Module 1 Handshake)
We acquire standard cloud-masked, harmonized optical composites:
1. **Sentinel-2 Level-2A** (`S2_SR`)
2. **Landsat 8 Level-2** (`L8_SR`)
""")

add_code(r"""
ref_data = Reflectance_Data()

# 1. Acquire Sentinel-2 Surface Reflectance Collection
print("Fetching Sentinel-2 collection...")
s2_col, s2_stats = ref_data.get_optical_data(
    aoi=aoi,
    start_date=start_date,
    end_date=end_date,
    optical_data='S2_SR',
    cloud_cover=40,
    compute_detailed_stats=False
)
s2_composite = ref_data.get_temporal_composite(s2_col, reducer='median', aoi=aoi)
print("Sentinel-2 Bands:", s2_composite.bandNames().getInfo())

# 2. Acquire Landsat 8 Surface Reflectance Collection
print("Fetching Landsat 8 collection...")
l8_col, l8_stats = ref_data.get_optical_data(
    aoi=aoi,
    start_date=start_date,
    end_date=end_date,
    optical_data='L8_SR',
    cloud_cover=40,
    compute_detailed_stats=False
)
l8_composite = ref_data.get_temporal_composite(l8_col, reducer='median', aoi=aoi)
print("Landsat 8 Bands:", l8_composite.bandNames().getInfo())
""")

# ==============================================================================
# Section 5: Individual Predictor Modules
# ==============================================================================
add_md("""
---
## 5. Individual Predictor Components
Before testing full pipeline orchestration, we demonstrate the standalone computation of each predictor component:
1. **Spectral Indices** (`SpectralCalculator`)
2. **Terrain Predictors** (`terrain_calculator`)
3. **Radar SAR Polarizations** (`sar.py`)
""")

add_md("""
### 5.1 Spectral Indices Calculation
Computing standard indices (`NDVI`, `MNDWI`) alongside sensor-specific indices:
- `RENDVI` and `MCARI` computed on Sentinel-2
- `EBBI` computed on Landsat-8 (using Thermal band)
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

# Landsat-8 Indices (Testing Thermal-assisted index: EBBI, plus DBSI, MBI)
l8_calc = SpectralCalculator()
l8_indices = l8_calc.calculate_indices(
    image=l8_composite,
    index_list=['NDVI', 'MNDWI', 'EBBI', 'DBSI', 'MBI'],
    sensor='LANDSAT'
)
print("Landsat-8 Computed Indices:", l8_indices.bandNames().getInfo())
""")

add_md("""
### 5.2 Terrain Predictors (Topography & Geomorphology)
Using `terrain_calculator` to extract Elevation, Slope, Aspect, and morphological Landform classes.
""")

add_code(r"""
tc = terrain_calculator(aoi=aoi)

# 1. Elevation, Slope, Aspect (from NASADEM)
dem_image = tc.get_terrain(dem_source='NASADEM')
print("DEM Bands:", dem_image.bandNames().getInfo())

# 2. Morphological Landforms (Penumbral, Ridges, Valleys, Plains)
landform_image = tc.get_landform()
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

# ==============================================================================
# Section 6: End-to-End Orchestrated Pipeline
# ==============================================================================
add_md("""
---
## 6. End-to-End Pipeline Orchestration: `compute_predictors()`
This is the primary method called by the backend service (`lulc_map.py` and `thematic.py`).

### Key Capabilities Verified:
1. **Automatic Sensor Detection**: Infers `SENTINEL` vs `LANDSAT` from the composite bands if `sensor` is omitted.
2. **Date Range Inheritance**: Automatically extracts start and end dates from `collection` metadata to pull synchronized SAR data.
3. **Graceful Skipping**: Safely ignores incompatible indices with warnings instead of failing the pipeline.
4. **Stacked Output**: Appends all predictors directly onto the optical composite.
""")

add_code(r"""
# Define full predictor configuration (matching backend payload schema)
predictor_config = {
    "individual_predictors": {
        "elevation": True,
        "slope": True,
        "aspect": True,
        "landform": True,
        "spectral_indices": [
            "NDVI", "EVI", "MNDWI", "GNDVI", "MSAVI",
            "RENDVI", "MCARI",  # Sentinel-specific
            "EBBI", "DBSI", "MBI"  # EBBI is Landsat-specific
        ]
    },
    "sar": {
        "sentinel1": True,
        "palsar": True
    }
}

orchestrator = PredictorCalculation()

# Execute on Sentinel-2
print("=== Running compute_predictors on Sentinel-2 ===")
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
    print("Expected Warnings (graceful skipping of Landsat-only indices):")
    for w in s2_result['warnings']:
        print(f"  - {w}")
""")

add_code(r"""
# Execute on Landsat 8
print("=== Running compute_predictors on Landsat-8 ===")
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
    print("Expected Warnings (graceful skipping of Sentinel-only indices):")
    for w in l8_result['warnings']:
        print(f"  - {w}")
""")

# ==============================================================================
# Section 7: Interactive Map Visualization (Geemap)
# ==============================================================================
add_md("""
---
## 7. Interactive Multi-Layer Visualization (`geemap`)
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

# Landsat EBBI (Enhanced Built-up and Bareness Index)
l8_stacked = l8_result['stacked_predictors']
ebbi_vis = {'bands': ['EBBI'], 'min': -0.5, 'max': 0.8, 'palette': ['green', 'yellow', 'red', 'purple']}
Map.addLayer(l8_stacked, ebbi_vis, '6. EBBI (Landsat Built-up/Bare Soil)')

# 4. Topography & Terrain
elev_vis = {'bands': ['elevation'], 'min': 0, 'max': 150, 'palette': ['darkblue', 'teal', 'lightgreen', 'brown', 'white']}
Map.addLayer(s2_stacked, elev_vis, '7. NASADEM Elevation')

landform_vis = {'bands': ['landform'], 'min': 11, 'max': 42, 'palette': ['141414', '383838', '808080', 'EBEBEB']}
Map.addLayer(s2_stacked, landform_vis, '8. Morphological Landforms')

# 5. Radar Backscatter (Sentinel-1 SAR)
s1_sar_vis = {'bands': ['S1_VV', 'S1_VH', 'S1_ratio'], 'min': [-18, -25, 0.2], 'max': [-4, -10, 0.9]}
Map.addLayer(s2_stacked, s1_sar_vis, '9. Sentinel-1 SAR (VV/VH/Ratio)')

# Add AOI boundary outline
aoi_outline = ee.Image().paint(ee.FeatureCollection([ee.Feature(aoi)]), color=1, width=2)
Map.addLayer(aoi_outline, {'palette': ['FF0000']}, 'AOI Boundary')

# Render the interactive map
Map
""")

# ==============================================================================
# Section 8: Technical Contract for Backend & Frontend Teams
# ==============================================================================
add_md("""
---
## 8. Technical Specification & Contract for Backend & Frontend

### 8.1 Backend Implementation Contract (`luma-be`)
When the frontend submits the predictor configuration, the payload structured below must be stored in `luma.predictor_config` and passed into `compute_predictors()`:

```json
{
  "individual_predictors": {
    "elevation": true,
    "slope": true,
    "aspect": true,
    "landform": true,
    "spectral_indices": [
      "NDVI", "EVI", "MNDWI", "MSAVI", "MCARI", "RENDVI", "GNDVI",
      "NDMI", "AWEIsh", "NDTI", "EBBI", "DBSI", "BLFEI", "MBI"
    ]
  },
  "sar": {
    "sentinel1": true,
    "palsar": false
  }
}
```

> **Important**: The backend whitelist in `application/logic/luma/predictor.py` (`set_predictor_config()`) should accept all predictors defined in `PredictorCatalog.get_all_catalog().keys()`.

### 8.2 Frontend Layer Styling Guide (`luma-fe`)
Recommended layer visualization palettes for frontend map rendering:

| Predictor Type | Band Name | Min | Max | Recommended Palette |
| :--- | :--- | :--- | :--- | :--- |
| **Vegetation** | `NDVI`, `EVI`, `GNDVI` | `0.0` | `0.85` | `['#0000FF', '#FFFFFF', '#00AA00']` |
| **Red Edge** | `RENDVI`, `MCARI` | `0.0` | `0.70` | `['#FF0000', '#FFFF00', '#006400']` |
| **Water** | `MNDWI`, `NDMI`, `AWEIsh` | `-0.5` | `0.50` | `['#8B4513', '#FFFFE0', '#00FFFF', '#0000FF']` |
| **Built-up / Soil** | `EBBI`, `DBSI`, `BLFEI`, `MBI` | `-0.5` | `0.80` | `['#228B22', '#FFFF00', '#FF0000', '#800080']` |
| **Elevation** | `elevation` | `0` | `2000` | `['#00008B', '#008080', '#90EE90', '#8B4513', '#FFFFFF']` |
| **SAR Radar** | `S1_VV` / `S1_VH` | `-25 dB` | `0 dB` | Grayscale `['#000000', '#FFFFFF']` |
""")

nb['cells'] = cells

output_path = 'notebooks/Module5_Predictors_POC.ipynb'
with open(output_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Successfully generated {output_path} with {len(cells)} cells!")
