import ee
import geemap

ee.Authenticate()  # use force=True for re-authentication
ee.Initialize()

# =============================================================================
# IMPORTANT NOTES ON GLAD GLCLU2020 AS AN ELEMENT/PROPERTY SOURCE
# =============================================================================
#
# The GLAD GLCLU2020 dataset (Potapov et al. 2022, Front. Remote Sens.) is
# NOT a discrete hard-class land cover map. It is a SUITE of independently
# derived thematic layers, each mapping one biophysical phenomenon:
#   - Forest height & extent (continuous + binary)
#   - Cropland extent (binary)
#   - Built-up land (binary + probability)
#   - Surface water dynamics (categorical + continuous)
#   - Perennial snow/ice (binary)
#
# This design makes it directly compatible with the LCML element/property
# framework in your modular training data template — each GLAD layer maps
# cleanly onto one or more elements in the CSV.
#
# The LCLUC_2015 composite image used in the original code encodes MULTIPLE
# of these layers as a single pixel value (0-255). The mapping is:
#
#   Value 1   = Stable forest 2000-2020 (tree presence + stable)
#   Value 2   = Forest loss 2001-2020   (tree presence + disturbed)
#   Value 3   = Forest gain 2001-2020   (tree presence + gain)
#   Value 4   = Forest disturbance      (tree presence + degraded)
#   Value 5   = Wetland forest          (tree presence + wet)
#   Value 6   = Cropland 2000-2020      (herb presence + cultivated)
#   Value 7   = Built-up 2000-2020      (builtup presence)
#   Value 8   = Surface water           (water presence)
#   Value 9   = Perennial snow/ice      (snow/ice presence)
#   Value 20  = Short vegetation (non-tree, non-crop) = shrub/herb/grass
#   Value 25  = Short veg in wetland context
#   Value 30  = Sparse vegetation / bare-ish
#   Value 35  = Open bare (sparse veg)
#   Value 40  = Bare soil / rock / sand
#   Value 48  = Closed-canopy tree cover (dense forest)
#   Value 200 = Ocean/water mask
#
# Because GLAD maps each theme INDEPENDENTLY, the dedicated single-theme
# layers (Forest_height_2020, Builtup_type etc.) are more reliable than
# trying to decode the combined LCLUC composite. We use BOTH below.
#
# COVERAGE AGAINST YOUR 40-ROW TEMPLATE:
# ──────────────────────────────────────────────────────────────────────
# sort_ID  element / property              GLAD source          Notes
# ──────────────────────────────────────────────────────────────────────
#  1  tree / elementPresenceType          Forest_extent        binary
#  2  tree / cover                        Forest_height proxy  no direct cover; use height>0 + GFCC
#  3  woodyGrowthForm / height            Forest_height_2020   continuous
#  4  vegetation / leafCharacterSizeType  MODIS LAI proxy      GLAD has no leaf size
#  5  tree / elementHorizontalSpreading   NOT in GLAD          spatial texture proxy only
#  6  tree / temporalType                 NOT in GLAD          —
#  7  tree / lengthOfTemporalRelationship NOT in GLAD          —
#  8  shrub / elementPresenceType         LCLUC_2015 val=20    binary (proxy; GLAD lacks shrub layer)
#  9  woodyGrowthForm / woodyLeafPhenology NOT direct in GLAD  MODIS MCD12Q1 needed
# 10  woodyGrowthForm / woodyLeafType     NOT direct in GLAD   MODIS MCD12Q1 needed
# 11-12 shrub horizontal/temporal         NOT in GLAD          —
# 13  herbaceousGrowthForm / presence     LCLUC_2015 val=20,30 binary proxy
# 14  herbaceous / temporalType           Cropland_dynamic     partial proxy
# 15  graminoid / presence                LCLUC_2015 val=30    binary proxy
# 16-17 graminoid phenology/spreading     NOT in GLAD          —
# 18  forb / presence                     NOT directly in GLAD —
# 20  builtUpSurface / presence           Builtup_type         binary
# 22  nonLinearSurface / presence         Builtup_type         same layer
# 23  building / presence                 Builtup_type         same layer
# 24  linearSurface / presence            NOT in GLAD          OSM needed
# 26  naturalSurface / presence           derived: not built/water/snow
# 27  bareSoil / presence                 LCLUC_2015 val=40    binary proxy
# 29  waterBody / presence                Surface_water        binary
# 30  waterBody / dynamics                Surface_water type   categorical (partial)
# 31  waterBody / position                NOT in GLAD          —
# 32  waterBody / periodVariationType     Surface_water seasonality
# 34  waterSalinity                       NOT in GLAD          —
# 35  element_artificiality               NOT in GLAD          —
# 39  vegetationArtificiality             Cropland = managed   binary proxy
# ──────────────────────────────────────────────────────────────────────

# =============================================================================
# REGION AND TRAINING SAMPLES
# =============================================================================

aoi = ee.FeatureCollection("projects/ee-agilfahrezy60/assets/AOI_Oganilir")
aoi_geom = aoi.geometry()

# Load your training point collections and merge them.
# Each collection should carry at minimum: 'class_id', 'class_name'.
# Replace asset paths with your actual GEE assets.
# Example structure (uncomment and fill in your paths):
#
# def label(fc, class_id, class_name):
#     return fc.map(lambda f: f.set({'class_id': class_id, 'class_name': class_name}))
#
# all_samples = (
#     label(ee.FeatureCollection("your_asset/undisturbedForest"),  1, "undisturbedForest")
#     .merge(label(ee.FeatureCollection("your_asset/loggedForest"), 2, "loggedForest"))
#     # ... add more classes
# ).filterBounds(aoi_geom)
#
# For now, use the AOI as a placeholder:
all_samples = aoi.filterBounds(aoi_geom)

# Reference year — change to match your training data vintage
YEAR = 2015

# =============================================================================
# GLAD DATASETS — primary element source
# =============================================================================

# Ocean mask (apply to all GLAD layers)
landmask = ee.Image("projects/glad/OceanMask").lte(1)

# Combined LCLUC composite (used as fallback for classes GLAD has no
# dedicated layer for, e.g. shrub, herbaceous, bare soil)
glad_lc = (ee.Image('projects/glad/GLCLU2020/v2/LCLUC_2015')
           .updateMask(landmask))

# =============================================================================
# ELEMENT 1: TREE BLOCK
# Template rows: sort_ID 1, 2, 3
# =============================================================================

# --- sort_ID 1: tree / elementPresenceType  (binary 0/1) ---
# GLAD dedicated forest extent layer: pixels with forest height >= 5 m
# More reliable than decoding the composite value
tree_presence = (ee.Image('projects/glad/GLCLU2020/Forest_extent_2020')
                 .updateMask(landmask)
                 .rename('treePresenceType'))
# Note: Forest_extent_2020 pixel value = 1 where forest present; already binary

# --- sort_ID 2: tree / cover  (continuous 0-100%) ---
# GLAD has NO dedicated canopy cover layer — only height.
# Best available 30 m cover proxy: GFCC30TC (NASA MEASURES)
# GLAD forest height is used as a complementary structural indicator.
tree_cover = (ee.ImageCollection('NASA/MEASURES/GFCC/TC/v3')
              .filter(ee.Filter.date(f'{YEAR}-01-01', f'{YEAR}-12-31'))
              .select('tree_canopy_cover')
              .first()
              .rename('treecover'))

# --- sort_ID 3: woodyGrowthForm / height  (continuous, metres) ---
# GLAD dedicated forest height layer (GEDI-calibrated, 2020)
tree_height = (ee.Image('projects/glad/GLCLU2020/Forest_height_2020')
               .updateMask(landmask)
               .unmask(0)
               .rename('treeheight'))

# --- sort_ID 4: vegetation / leafCharacterSizeType ---
# GLAD has NO leaf size information.
# Best available proxy: MODIS LAI (500 m, use with caution at 30 m)
lai_mean = (ee.ImageCollection('MODIS/061/MOD15A2H')
            .filterDate(f'{YEAR}-01-01', f'{YEAR}-12-31')
            .select('Lai_500m')
            .mean()
            .multiply(0.1)
            .rename('lai_mean'))

fpar_mean = (ee.ImageCollection('MODIS/061/MOD15A2H')
             .filterDate(f'{YEAR}-01-01', f'{YEAR}-12-31')
             .select('Fpar_500m')
             .mean()
             .multiply(0.01)
             .rename('fpar_mean'))

# --- sort_ID 5-7: tree horizontal spreading, temporal type, length ---
# NOT available in GLAD. No suitable global 30 m proxy exists.
# Omitted from extraction; leave blank in training CSV.

# --- sort_ID 9: woodyGrowthForm / woodyLeafPhenology (Evergreen/Deciduous) ---
# GLAD has NO phenology layer.
# Use MODIS MCD12Q1 LC_Type5:
#   1 = evergreen needleleaf, 2 = evergreen broadleaf,
#   3 = deciduous needleleaf, 4 = deciduous broadleaf
woody_leaf_phenology = (ee.ImageCollection('MODIS/061/MCD12Q1')
                        .filterDate(f'{YEAR}-01-01', f'{YEAR+1}-01-01')
                        .first()
                        .select('LC_Type5')
                        .rename('treewoodyLeafPhenology'))

# --- sort_ID 10: woodyGrowthForm / woodyLeafType (Broadleaf/Needleleaf) ---
# Same MODIS layer; values 1,3 = needleleaf; 2,4 = broadleaf
# Remap: 1,3 → 1 (needleleaf), 2,4 → 2 (broadleaf), others → 0
woody_leaf_type = (woody_leaf_phenology
                   .remap([1, 2, 3, 4], [1, 2, 1, 2], defaultValue=0)
                   .rename('woodyLeafType'))

# =============================================================================
# ELEMENT 2: SHRUB BLOCK
# Template rows: sort_ID 8, 11, 12
# =============================================================================

# --- sort_ID 8: shrub / elementPresenceType (binary 0/1) ---
# GLAD has NO dedicated shrub layer.
# Best proxy: LCLUC composite value 20 = short woody vegetation
# (includes shrub and low scrub in the GLAD legend)
shrub_presence = glad_lc.eq(20).rename('shrubPresenceType')
# Limitation: GLAD value 20 is a mixed class — shrub, low woody veg,
# and transitional vegetation are not separately distinguishable.

# --- sort_ID 11-12: shrub horizontal spreading, temporal type ---
# NOT available in GLAD. Omitted.

# =============================================================================
# ELEMENT 3: HERBACEOUS / GRAMINOID / FORB BLOCK
# Template rows: sort_ID 13, 14, 15, 16, 17, 18, 19
# =============================================================================

# --- sort_ID 13: herbaceousGrowthForm / elementPresenceType (binary) ---
# GLAD proxy: LCLUC value 20 (short veg) OR 30 (sparse veg/open herb)
# Combined = any short non-woody cover
herb_presence = (glad_lc.eq(20).Or(glad_lc.eq(30))
                 .rename('herbaceousPresenceType'))

# --- sort_ID 14: herbaceous / temporalType ---
# GLAD cropland dynamic layer partially captures seasonal herbaceous change.
# Not a direct proxy for herb temporal type. Omitted from extraction.

# --- sort_ID 15: graminoid / elementPresenceType (binary) ---
# GLAD proxy: LCLUC value 30 = sparse/open vegetation, mostly grass-like.
# Note: GLAD does not separately distinguish graminoids from other herbs.
graminoid_presence = glad_lc.eq(30).rename('graminoidPresenceType')

# --- sort_ID 16-17: graminoid leaf phenology, horizontal spreading ---
# NOT available in GLAD. Omitted.

# --- sort_ID 18-19: forb presence, forb status ---
# NOT available in GLAD or any current global 30 m product. Omitted.

# Vegetation NDVI statistics (complement to GLAD for herb/graminoid density)
ndvi_col   = (ee.ImageCollection('MODIS/061/MOD13Q1')
              .filterDate(f'{YEAR}-01-01', f'{YEAR}-12-31')
              .select('NDVI')
              .map(lambda img: img.multiply(0.0001)))
ndvi_mean  = ndvi_col.mean().rename('ndvi_mean')
ndvi_min   = ndvi_col.min().rename('ndvi_min')
ndvi_max   = ndvi_col.max().rename('ndvi_max')
ndvi_std   = ndvi_col.reduce(ee.Reducer.stdDev()).rename('ndvi_std')

# =============================================================================
# ELEMENT 4: BUILT-UP SURFACE BLOCK
# Template rows: sort_ID 20, 21, 22, 23, 24, 25
# =============================================================================

# --- sort_ID 20, 22, 23: builtup / nonLinear / building presence (binary) ---
# GLAD dedicated built-up layer:
#   Value 1 = stable built-up area 2000-2020
#   Value 2 = built-up expansion 2000-2020
# Both values indicate built-up presence → remap to binary
glad_builtup_raw = (ee.Image('projects/glad/GLCLU2020/Builtup_type')
                    .updateMask(landmask))
builtup_presence = (glad_builtup_raw.gte(1).unmask(0)
                    .rename('builtUpPresenceType'))
# Same binary layer used for nonLinearSurface and building presence
# since GLAD does not distinguish building vs road vs plaza
nonlinear_surface_presence = builtup_presence.rename('nonLinearSurface_presence')
building_presence          = builtup_presence.rename('buildingPresenceType')

# --- sort_ID 21: constructionMaterial ---
# NOT available in GLAD or any current global 30 m product. Omitted.

# --- sort_ID 24-25: linearSurface presence and type ---
# NOT available in GLAD. Roads/railways require OSM or similar. Omitted.

# Continuous built-up surface area (m²) — GHSL (complements GLAD)
ghsl             = ee.Image('JRC/GHSL/P2023A/GHS_BUILT_S_10m/2018')
built_surface_m2 = ghsl.select('built_surface').rename('built_surface_m2')
built_nres_m2    = ghsl.select('built_surface_nres').rename('built_nres_m2')

# =============================================================================
# ELEMENT 5: NATURAL SURFACE / BARE SOIL BLOCK
# Template rows: sort_ID 26, 27, 28
# =============================================================================

# --- sort_ID 26: naturalSurface / elementPresenceType (binary) ---
# Derived: any pixel that is NOT built-up, NOT water, NOT snow = natural surface
# Use GLAD layers to mask out artificial/water/snow
glad_snow   = (ee.Image('projects/glad/GLCLU2020/LCLUC_2015')
               .updateMask(landmask).eq(9))  # value 9 = snow/ice
natural_surface_presence = (builtup_presence.eq(0)
                             .And(glad_lc.eq(8).Not())   # not water
                             .And(glad_snow.Not())        # not snow
                             .rename('naturalSurface_presence'))

# --- sort_ID 27: bareSoil / elementPresenceType (binary) ---
# GLAD proxy: LCLUC value 40 = bare/sparse land (rock, sand, soil)
# Also value 35 = open bare-ish ground
bare_soil_presence = (glad_lc.eq(40).Or(glad_lc.eq(35))
                      .rename('bareSoilPresence'))

# --- sort_ID 28: bareSoil / lengthOfTemporalRelationship ---
# GLAD has no temporal bare soil metric.
# Partial proxy: GLC-FCS30D annual bare land class
bareland_col = ee.ImageCollection("projects/sat-io/open-datasets/GLC-FCS30D/annual")
bare_temporal = (bareland_col
                 .filterDate(f'{YEAR}-01-01', f'{YEAR}-12-31')
                 .first()
                 .eq(60)
                 .unmask(0)
                 .rename('bareSoil_GLC'))

# =============================================================================
# ELEMENT 6: WATER BODY BLOCK
# Template rows: sort_ID 29, 30, 31, 32, 33, 34, 35
# =============================================================================

# GLAD dedicated surface water dataset
# GEE path: projects/glad/GLCLU2020/SW_2015 (annual surface water)
# Note: GLAD SW uses different encoding than JRC GSW
# Values: 1 = permanent water, 2 = seasonal water, 3 = ephemeral water

glad_sw = (ee.Image('projects/glad/GLCLU2020/SW_2015')
           .updateMask(landmask)
           .unmask(0))

# --- sort_ID 29: waterBody / elementPresenceType (binary) ---
# Any GLAD water value (1,2,3) = water present
water_presence = glad_sw.gte(1).rename('waterBodyPresence')

# --- sort_ID 30: waterBody / dynamics (flowing/standing) ---
# GLAD does not distinguish flowing vs standing.
# JRC GSW transition layer provides the best available proxy.
gsw = ee.Image('JRC/GSW1_4/GlobalSurfaceWater')
# Use JRC occurrence as continuous water metric
water_occurrence  = gsw.select('occurrence').rename('waterOccurrence')
# Seasonality = number of months water present per year (0-12)
water_seasonality = gsw.select('seasonality').rename('waterSeasonality')

# --- sort_ID 32: waterBody / periodVariationType (categorical) ---
# GLAD directly encodes this:
#   Value 1 = permanent water   → maps to "permanent"
#   Value 2 = seasonal water    → maps to "seasonal"
#   Value 3 = ephemeral water   → maps to "atmospheric/ephemeral"
water_period_type = glad_sw.rename('waterPeriodVariationType')
# Integer meaning: 0=no water, 1=permanent, 2=seasonal, 3=ephemeral

# --- sort_ID 33: waterBody / horizontalSpreading ---
# Not extractable from GLAD as a point-level property. Omitted.

# --- sort_ID 31, 34, 35: position, salinity, artificiality ---
# GLAD has NO salinity or artificiality information.
# No suitable global 30 m source exists. Omitted with documentation.

# =============================================================================
# ELEMENT 7: VEGETATION ARTIFICIALITY
# Template row: sort_ID 39
# =============================================================================

# --- sort_ID 39: vegetationArtificiality (natural vs cultivated) ---
# GLAD dedicated cropland layer:
#   projects/glad/GLCLU2020/Cropland
#   Value 1 = cropland (cultivated/managed herbaceous)
glad_cropland = (ee.Image('projects/glad/GLCLU2020/Cropland')
                 .updateMask(landmask)
                 .unmask(0))
# Remap: cropland = 2 (cultivated), non-cropland non-built = 1 (natural/semi-natural)
# This is a binary proxy — does not capture managed forests or pastures
vegetation_artificiality = (glad_cropland.gte(1)
                             .rename('vegetationArtificiality'))
# 0 = naturalOrSeminatural, 1 = cultivatedAndManaged

# Crop intensity from GCI30 (complements GLAD cropland)
gci30          = ee.ImageCollection("projects/sat-io/open-datasets/GCI30").median()
crop_intensity = (gci30.select('b1')
                  .where(gci30.select('b1').eq(-1), 0)
                  .rename('crop_intensity'))
crop_cycles    = gci30.select('b2').rename('crop_cycles')

# =============================================================================
# ASSEMBLE FINAL STACK
# =============================================================================
# Only include layers that have actual data — omitted elements are documented
# above as NOT AVAILABLE in GLAD.

stack = ee.Image.cat([
    # ── TREE BLOCK ──────────────────────────────────────────────────
    tree_presence,          # sort_ID 1:  binary, GLAD Forest_extent
    tree_cover,             # sort_ID 2:  continuous %, GFCC30TC (GLAD has no cover)
    tree_height,            # sort_ID 3:  continuous m, GLAD Forest_height_2020
    lai_mean,               # sort_ID 4:  continuous, MODIS proxy for leaf size
    fpar_mean,              # sort_ID 4:  continuous, MODIS proxy (companion to LAI)
    woody_leaf_phenology,   # sort_ID 9:  categorical, MODIS MCD12Q1 LC_Type5
    woody_leaf_type,        # sort_ID 10: categorical, derived from above
    # ── SHRUB BLOCK ─────────────────────────────────────────────────
    shrub_presence,         # sort_ID 8:  binary, GLAD LCLUC val=20 (proxy)
    # ── HERBACEOUS / GRAMINOID BLOCK ────────────────────────────────
    herb_presence,          # sort_ID 13: binary, GLAD LCLUC val=20|30 (proxy)
    graminoid_presence,     # sort_ID 15: binary, GLAD LCLUC val=30 (proxy)
    ndvi_mean,              # vegetation density proxy
    ndvi_min,               # dry-season vegetation signal
    ndvi_max,               # peak-season vegetation signal
    ndvi_std,               # phenological variability
    # ── BUILT-UP BLOCK ──────────────────────────────────────────────
    builtup_presence,       # sort_ID 20: binary, GLAD Builtup_type
    nonlinear_surface_presence,  # sort_ID 22: same GLAD layer
    building_presence,      # sort_ID 23: same GLAD layer
    built_surface_m2,       # continuous, GHSL (complements GLAD)
    built_nres_m2,          # continuous, GHSL non-residential
    # ── NATURAL SURFACE / BARE SOIL BLOCK ───────────────────────────
    natural_surface_presence,    # sort_ID 26: derived from GLAD layers
    bare_soil_presence,          # sort_ID 27: GLAD LCLUC val=40|35 (proxy)
    bare_temporal,               # sort_ID 28: GLC-FCS30D temporal proxy
    # ── WATER BODY BLOCK ────────────────────────────────────────────
    water_presence,         # sort_ID 29: binary, GLAD SW
    water_period_type,      # sort_ID 32: categorical, GLAD SW (0/1/2/3)
    water_occurrence,       # continuous %, JRC GSW (complements GLAD)
    water_seasonality,      # months/year, JRC GSW (complements GLAD)
    # ── VEGETATION ARTIFICIALITY ─────────────────────────────────────
    vegetation_artificiality,    # sort_ID 39: GLAD Cropland proxy
    crop_intensity,         # GCI30 crop intensity
    crop_cycles,            # GCI30 crop cycles
])

print("Stack band names:")
print(stack.bandNames().getInfo())

# =============================================================================
# EXTRACT RS VALUES AT TRAINING POINT LOCATIONS
# =============================================================================

extracted = stack.reduceRegions(
    collection=all_samples,
    reducer=ee.Reducer.mean(),
    scale=30,
    tileScale=4,
)

# =============================================================================
# EXPORT
# =============================================================================

# Columns to export — mirrors the sort_ID order in the template CSV
selectors = [
    'system:index', 'class_id', 'class_name',
    # Tree block
    'treePresenceType',         # sort_ID 1  — GLAD Forest_extent (binary)
    'treecover',                # sort_ID 2  — GFCC30TC (%)
    'treeheight',               # sort_ID 3  — GLAD Forest_height_2020 (m)
    'lai_mean',                 # sort_ID 4  — MODIS LAI proxy
    'fpar_mean',                # sort_ID 4  — MODIS FPAR proxy
    'treewoodyLeafPhenology',   # sort_ID 9  — MODIS MCD12Q1 LC_Type5
    'woodyLeafType',            # sort_ID 10 — derived from LC_Type5
    # Shrub block
    'shrubPresenceType',        # sort_ID 8  — GLAD LCLUC val=20 (proxy)
    # Herbaceous / graminoid block
    'herbaceousPresenceType',   # sort_ID 13 — GLAD LCLUC val=20|30
    'graminoidPresenceType',    # sort_ID 15 — GLAD LCLUC val=30
    'ndvi_mean',                # vegetation density
    'ndvi_min',
    'ndvi_max',
    'ndvi_std',
    # Built-up block
    'builtUpPresenceType',      # sort_ID 20 — GLAD Builtup_type
    'nonLinearSurface_presence',# sort_ID 22 — GLAD Builtup_type (same)
    'buildingPresenceType',     # sort_ID 23 — GLAD Builtup_type (same)
    'built_surface_m2',         # GHSL continuous built-up area
    'built_nres_m2',            # GHSL non-residential
    # Natural surface / bare soil block
    'naturalSurface_presence',  # sort_ID 26 — derived from GLAD
    'bareSoilPresence',         # sort_ID 27 — GLAD LCLUC val=40|35
    'bareSoil_GLC',             # sort_ID 28 — GLC-FCS30D temporal proxy
    # Water body block
    'waterBodyPresence',        # sort_ID 29 — GLAD SW (binary)
    'waterPeriodVariationType', # sort_ID 32 — GLAD SW (0/1/2/3)
    'waterOccurrence',          # JRC GSW % occurrence
    'waterSeasonality',         # JRC GSW months/year
    # Vegetation artificiality
    'vegetationArtificiality',  # sort_ID 39 — GLAD Cropland proxy
    'crop_intensity',           # GCI30
    'crop_cycles',              # GCI30
]

task = ee.batch.Export.table.toDrive(
    collection=extracted,
    description=f'Feature_Extraction_GLAD_OganIlir_{YEAR}',
    folder='GEE_exports',
    fileNamePrefix=f'OganIlir_GLAD_elements_{YEAR}',
    fileFormat='SHP',
    selectors=selectors,
)

task.start()
print("Export task submitted:", task.status())

# =============================================================================
# COVERAGE SUMMARY — which template rows are covered vs not
# =============================================================================
#
# COVERED BY GLAD (primary source):
#   sort_ID 1  tree presence          → GLAD Forest_extent_2020
#   sort_ID 3  tree height            → GLAD Forest_height_2020
#   sort_ID 8  shrub presence         → GLAD LCLUC val=20 (proxy only)
#   sort_ID 13 herb presence          → GLAD LCLUC val=20|30 (proxy)
#   sort_ID 15 graminoid presence     → GLAD LCLUC val=30 (proxy)
#   sort_ID 20 builtup presence       → GLAD Builtup_type
#   sort_ID 22 nonlinear surface      → GLAD Builtup_type
#   sort_ID 23 building presence      → GLAD Builtup_type
#   sort_ID 27 bare soil presence     → GLAD LCLUC val=40|35 (proxy)
#   sort_ID 29 water body presence    → GLAD SW
#   sort_ID 32 water period type      → GLAD SW (0/1/2/3)
#   sort_ID 39 veg artificiality      → GLAD Cropland (binary proxy)
#
# COVERED BY SUPPLEMENTARY DATASETS (GLAD has no suitable layer):
#   sort_ID 2  tree cover             → GFCC30TC (NASA MEASURES)
#   sort_ID 4  leaf size              → MODIS MOD15A2H (LAI proxy)
#   sort_ID 9  woody leaf phenology   → MODIS MCD12Q1 LC_Type5
#   sort_ID 10 woody leaf type        → MODIS MCD12Q1 (derived)
#   sort_ID 26 natural surface        → derived from GLAD layers
#   sort_ID 28 bare soil temporal     → GLC-FCS30D annual
#   sort_ID 29+ water continuous      → JRC GSW (complements GLAD SW)
#
# NOT COVERED BY ANY CURRENT GLOBAL 30 m PRODUCT:
#   sort_ID 5  tree horizontal spreading
#   sort_ID 6  tree temporal type
#   sort_ID 7  tree temporal length
#   sort_ID 11 shrub horizontal spreading
#   sort_ID 12 shrub temporal type
#   sort_ID 14 herb temporal type
#   sort_ID 16 graminoid leaf phenology
#   sort_ID 17 graminoid horizontal spreading
#   sort_ID 18 forb presence
#   sort_ID 19 forb status
#   sort_ID 21 construction material
#   sort_ID 24 linear surface presence
#   sort_ID 25 linear surface type
#   sort_ID 31 water body position
#   sort_ID 33 water horizontal spreading
#   sort_ID 34 water salinity
#   sort_ID 35 water artificiality
#   sort_ID 36 construction use
#   sort_ID 37 species name
#   sort_ID 38 water stress
#   sort_ID 40 land use classes
# =============================================================================