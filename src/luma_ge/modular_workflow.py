"""
Modular land cover map generation from LCML-structured reference data.
Supports multiple classification schemes from a single training dataset.

Workflow
--------
    Step 1 — Load classification scheme
        rules = load_scheme("scheme1.csv")
        rules["scheme"] = "scheme1"

    Step 2 — Load reference data
        data = load_modular_training_data("training_points.shp", aoi=aoi)

    Step 3a — Direct labelling pathway 
        labeller   = TrainingDataLabeller(rules, scheme_name="scheme1")
        labeled_fc = labeller.label(data["ee_fc"])

    Step 3b — Primitive layer pathway 
        trainer         = PrimitiveLayerTrainer(image=img, roi=data["ee_fc"])
        prob_layers     = trainer.train_all_mc()
        primitive_stack = ee.Image.cat(list(prob_layers.values()))
        classifier      = RuleSetClassifier(primitive_stack, rules, aoi)
        mc_results      = classifier.classify_scheme_monte_carlo("scheme1")

    Step 4 — Validate and compare
        det_map = classifier.classify_scheme_deterministic("scheme1")
        det_arr = validate_deterministic(det_map, rules, "scheme1", aoi)
        validate_monte_carlo(mc_results, rules, "scheme1")
        compare_det_vs_mc(det_arr, mc_results, rules, "scheme1")
"""

import io
import logging
import warnings
import zipfile
from typing import Optional

import ee
import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import requests
from matplotlib.gridspec import GridSpec
from shapely.geometry import shape

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


# ===========================================================================
# Constants
# ===========================================================================

# Columns that are never treated as primitive elements
_NON_PRIMITIVE_COLS = {"LULC_Type", "ID", "geometry", "system:index", "class_id"}

# Operator string → Python symbol mapping (used in load_scheme)
_OP_MAP = {
    "more than":           ">",
    "greater than":        ">",
    "equal to":            "==",
    "less than":           "<",
    "more or equal":       ">=",
    "greater or equal":    ">=",
    "greater or equal to": ">=",
    "less or equal":       "<=",
    "less or equal to":    "<=",
}

# Shared colour palette for validation plots (index 0 = nodata)
_CLASS_PALETTE = [
    "#888780",  # 0  nodata  — gray
    "#1D9E75",  # 1          — teal
    "#378ADD",  # 2          — blue
    "#D85A30",  # 3          — coral
    "#BA7517",  # 4          — amber
    "#7F77DD",  # 5          — purple
    "#639922",  # 6          — green
    "#D4537E",  # 7          — pink
]


# ===========================================================================
# Step 1: Load classification scheme
# load_scheme() — converts a human-authored CSV into a rule DataFrame
# ===========================================================================

def load_scheme(csv_path: str) -> pd.DataFrame:
    """
    Load a classification scheme CSV and convert it into a rule DataFrame.

    The CSV defines one land cover class per row. Required columns:

        class_id      int     Unique numeric identifier
        class_name    str     Human-readable label
        rule_general  str     "none" | "and" | "or"
        priority      int     Evaluation order (1 = highest priority)

    Plus one pair of columns per primitive element:

        tree_pres          float   Threshold value (e.g. 0.6)
        rule_tree_pres     str     Plain-English operator (e.g. "more than")

    rule_general behaviour
    ----------------------
        "none"  — use only the single most discriminating condition.
                  Zero-valued ("equal to 0") conditions are exclusions
                  and are skipped; the first non-zero column is selected.
        "and"   — join all non-NaN conditions with AND.
        "or"    — join all non-NaN conditions with OR.

    Parameters
    ----------
    csv_path : str
        Path to the scheme CSV file.

    Returns
    -------
    pd.DataFrame
        Columns: class_id, class_name, rule, priority
        Each row contains a fully formed rule expression string, e.g.:
            'tree_pres > 0.6'
            'tree_pres > 0.5 AND buil_pres > 0.3'

    Example
    -------
    >>> rules = load_scheme("../data/scheme1.csv")
    >>> rules["scheme"] = "scheme1"
    >>> print(rules)
    """
    df = pd.read_csv(csv_path)

    required = ["class_id", "class_name", "rule_general", "priority"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in '{csv_path}': {missing}")

    # Primitive columns: end in _pres, not prefixed with rule_
    pres_cols = [
        c for c in df.columns
        if c.endswith("_pres") and not c.startswith("rule_")
    ]

    def _get_op(op_text, class_id, rule_col):
        key = op_text.lower().strip()
        if key not in _OP_MAP:
            raise ValueError(
                f"Unknown operator '{op_text}' in column '{rule_col}' "
                f"for class_id {class_id}. Supported: {list(_OP_MAP.keys())}"
            )
        return _OP_MAP[key]

    def _build_condition(pres_col, rule_col, op_text, thresh, class_id):
        return f"{pres_col} {_get_op(op_text, class_id, rule_col)} {thresh}"

    def _is_exclusion(op_text, thresh):
        """Zero-valued 'equal to 0' conditions are exclusions, not discriminators."""
        return op_text.lower().strip() == "equal to" and float(thresh) == 0

    def _build_none_rule(row):
        """Pick the single defining condition, skipping zero-exclusions."""
        for pres_col in pres_cols:
            thresh = row[pres_col]
            if pd.isna(thresh):
                continue
            primitive = pres_col.replace("_pres", "")
            rule_col  = f"rule_{primitive}_pres"
            if rule_col not in df.columns:
                raise ValueError(
                    f"No operator column '{rule_col}' for '{pres_col}' "
                    f"in class_id {row['class_id']}."
                )
            op_text = str(row[rule_col])
            if _is_exclusion(op_text, thresh):
                continue
            return _build_condition(pres_col, rule_col, op_text, thresh, row["class_id"])
        raise ValueError(
            f"'none' rule in class_id {row['class_id']} has no non-zero threshold."
        )

    def _build_combined_rule(row, combination):
        """Join all non-NaN conditions with AND or OR."""
        conds = []
        for pres_col in pres_cols:
            thresh = row[pres_col]
            if pd.isna(thresh):
                continue
            primitive = pres_col.replace("_pres", "")
            rule_col  = f"rule_{primitive}_pres"
            if rule_col not in df.columns:
                raise ValueError(
                    f"No operator column '{rule_col}' for '{pres_col}' "
                    f"in class_id {row['class_id']}."
                )
            conds.append(_build_condition(
                pres_col, rule_col, str(row[rule_col]), thresh, row["class_id"]
            ))
        if not conds:
            raise ValueError(f"No conditions for class_id {row['class_id']}.")
        joiner = " AND " if combination == "and" else " OR "
        return joiner.join(conds)

    rules = []
    for _, row in df.iterrows():
        combination = str(row["rule_general"]).lower().strip()
        if combination == "none":
            rules.append(_build_none_rule(row))
        elif combination in ("and", "or"):
            rules.append(_build_combined_rule(row, combination))
        else:
            raise ValueError(
                f"Invalid rule_general '{row['rule_general']}' in "
                f"class_id {row['class_id']}. Use 'none', 'and', or 'or'."
            )

    result = df[["class_id", "class_name", "priority"]].copy()
    result["rule"] = rules
    return result[["class_id", "class_name", "rule", "priority"]]


# ===========================================================================
# Step 2: Training Data
# load_modular_training_data() — loads shapefile reference data
# TrainingDataLabeller         — labels features by scheme rules (for Step 3b)
# ===========================================================================

def load_modular_training_data(shp_path: str, aoi=None) -> dict:
    """
    Load a shapefile containing LCML element attributes as reference data.

    Each feature should have binary (0/1) properties for each LCML element
    (e.g. tree_pres, buil_pres, water_pres) indicating element presence.

    Parameters
    ----------
    shp_path : str
        Path to the shapefile.
    aoi : ee.FeatureCollection or ee.Geometry, optional
        If provided, only features intersecting the AOI are retained.

    Returns
    -------
    dict with keys:
        "gdf"     — geopandas GeoDataFrame (filtered to AOI if provided)
        "ee_fc"   — ee.FeatureCollection of the same features
        "columns" — list of column names (excluding geometry)
        "size"    — number of features after filtering

    Example
    -------
    >>> data = load_modular_training_data("training_points.shp", aoi=aoi)
    >>> training_fc = data["ee_fc"]
    >>> print(data["columns"], data["size"])
    """
    gdf = gpd.read_file(shp_path)

    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    gdf = gdf.to_crs("EPSG:4326")

    # Filter to AOI using shapely geometry intersection
    if aoi is not None:
        aoi_geom  = aoi.geometry().getInfo()
        aoi_shape = shape(aoi_geom)
        gdf = gdf[gdf.intersects(aoi_shape)].reset_index(drop=True)

    features = [
        ee.Feature(
            ee.Geometry(row.geometry.__geo_interface__),
            row.drop("geometry").to_dict()
        )
        for _, row in gdf.iterrows()
    ]

    return {
        "gdf":     gdf,
        "ee_fc":   ee.FeatureCollection(features),
        "columns": list(gdf.columns),
        "size":    len(gdf),
    }


# ===========================================================================
# Get external sources of training data information 
# load_element_mapping()   — reads the element→primitive→GEE mapping CSV
# enrich_training_data()   — samples external GEE datasets at training points
#                            and fills missing primitive columns
# ===========================================================================

def load_element_mapping(mapping_csv_path: str) -> pd.DataFrame:
    """
    Load the element mapping CSV that defines how LCML elements translate
    into primitive layer columns and which external GEE dataset provides
    the values for each.

    This CSV is the bridge between the LCML element taxonomy and the
    primitive layer names used by load_scheme() and PrimitiveLayerTrainer.
    It lets you extend the training data with values from external sources
    (e.g. tree canopy cover from GFCC, water occurrence from JRC) without
    manually editing each training point.

    Expected CSV columns
    --------------------
    element_block   str   LCML block name (e.g. "tree", "waterBody")
    element_name    str   LCML element name (e.g. "elementPresenceType")
    primitive_col   str   Column name in training data (e.g. "tree_pres")
    value_type      str   "binary" | "continuous"
    gee_dataset     str   GEE image asset path (e.g. "NASA/MEASURES/GFCC/TC/v3")
    band_name       str   Band to sample from the GEE image
    scale           int   Sampling scale in metres
    transform       str   Post-sampling transform: "none" | "divide_100" |
                          "divide_max" | "log10"
    threshold_type  str   How the value is used: "binary" | "continuous"
    notes           str   Optional description (ignored by code)

    Parameters
    ----------
    mapping_csv_path : str
        Path to the element mapping CSV.

    Returns
    -------
    pd.DataFrame
        Validated mapping table ready for use in enrich_training_data().

    Example
    -------
    >>> mapping = load_element_mapping("element_mapping.csv")
    >>> print(mapping[["element_block", "primitive_col", "gee_dataset"]])
    """
    import logging
    import pandas as pd

    logger = logging.getLogger("load_element_mapping")

    df = pd.read_csv(mapping_csv_path)

    # --- minimal required structure ---
    required = ["element_block", "element_name"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in '{mapping_csv_path}': {missing}"
        )

    # Ensure primitive_col exists (can be empty)
    if "primitive_col" not in df.columns:
        df["primitive_col"] = ""

    # Normalize primitive_col (treat NaN as empty string)
    df["primitive_col"] = df["primitive_col"].fillna("").astype(str).str.strip()

    # Split active vs inactive rows
    active = df["primitive_col"] != ""
    df_active = df[active]

    # --- validation only for active rows ---
    required_active = [
        "value_type", "gee_dataset", "band_name", "scale", "transform"
    ]
    missing_active = [c for c in required_active if c not in df.columns]
    if missing_active:
        raise ValueError(
            f"Missing required columns for active mappings: {missing_active}"
        )

    valid_transforms = {"none", "divide_100", "divide_max", "log10"}

    invalid = df_active[
        ~df_active["transform"].isin(valid_transforms)
    ]["transform"].dropna().unique()

    if len(invalid) > 0:
        raise ValueError(
            f"Unknown transform values in active rows: {list(invalid)}. "
            f"Supported: {valid_transforms}"
        )

    # Optional: enforce value_type only for active rows
    valid_value_types = {"binary", "continuous"}
    invalid_vt = df_active[
        ~df_active["value_type"].isin(valid_value_types)
    ]["value_type"].dropna().unique()

    if len(invalid_vt) > 0:
        raise ValueError(
            f"Unknown value_type values in active rows: {list(invalid_vt)}. "
            f"Supported: {valid_value_types}"
        )

    logger.info(
        f"Loaded element mapping: {len(df)} rows "
        f"({active.sum()} active, {(~active).sum()} inactive), "
        f"{df_active['primitive_col'].nunique()} primitive columns."
    )

    return df


def enrich_training_data(
    gdf: gpd.GeoDataFrame,
    mapping: pd.DataFrame,
    binary_threshold: float = 0.1,
    overwrite_existing: bool = False,
    scale_override: Optional[int] = None,
) -> dict:
    """
    Enrich training data by sampling external GEE datasets at each training
    point and filling the corresponding primitive columns.

    For each row in the element mapping table, the function:
        1. Loads the specified GEE image and selects the target band.
        2. Applies the specified transform (e.g. divide_100 for percentage bands).
        3. Samples the value at each training point geometry.
        4. For binary primitives: converts to 0/1 using binary_threshold.
           For continuous primitives: writes the raw transformed value.
        5. Writes the result into the primitive_col column of the GeoDataFrame.

    Columns that already have values are skipped unless overwrite_existing=True,
    so you can safely call this on partially-filled training data.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Training data GeoDataFrame from load_modular_training_data().
        Must have a valid geometry column and CRS set to EPSG:4326.
    mapping : pd.DataFrame
        Output of load_element_mapping().
    binary_threshold : float
        Transformed values above this threshold are assigned 1 (present),
        at or below are assigned 0 (absent). Default 0.1 — equivalent to
        10% cover for datasets like tree canopy or water occurrence.
        Only applied to rows where value_type = "binary".
    overwrite_existing : bool
        If True, re-sample and overwrite columns that already have values.
        If False (default), skip columns that are already populated.
    scale_override : int or None
        If provided, overrides the scale in the mapping table for all
        datasets. Useful for quick testing at coarser resolution.

    Returns
    -------
    dict with keys:
        "gdf"          — enriched GeoDataFrame with new/updated columns
        "ee_fc"        — ee.FeatureCollection of the enriched features
        "columns"      — list of all column names
        "size"         — number of features
        "enriched"     — list of primitive_col names that were filled
        "skipped"      — list of primitive_col names that were skipped
        "failed"       — list of (primitive_col, error_message) tuples

    Example
    -------
    >>> mapping = load_element_mapping("element_mapping.csv")
    >>> data    = load_modular_training_data("training_points.shp", aoi=aoi)
    >>> result  = enrich_training_data(data["gdf"], mapping)
    >>> print("Enriched columns:", result["enriched"])
    >>> training_fc = result["ee_fc"]
    """
    logger = logging.getLogger("enrich_training_data")

    gdf = gdf.copy()
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    gdf = gdf.to_crs("EPSG:4326")

    enriched_cols = []
    skipped_cols  = []
    failed_cols   = []

    for _, map_row in mapping.iterrows():
        prim_col    = map_row["primitive_col"]
        gee_dataset = map_row["gee_dataset"]
        band_name   = map_row["band_name"]
        scale       = scale_override or int(map_row["scale"])
        transform   = str(map_row["transform"]).lower().strip()
        value_type  = str(map_row["value_type"]).lower().strip()
        block       = map_row["element_block"]
        element     = map_row["element_name"]

        # Skip if column already has values and overwrite is off
        if prim_col in gdf.columns and not overwrite_existing:
            existing_valid = gdf[prim_col].notna().sum()
            if existing_valid > 0:
                logger.info(
                    f"  Skipping '{prim_col}' — {existing_valid} values "
                    "already present (set overwrite_existing=True to refill)."
                )
                skipped_cols.append(prim_col)
                continue

        logger.info(
            f"  Enriching '{prim_col}'  ←  [{block}.{element}]  "
            f"from {gee_dataset} / {band_name}  (scale={scale}m)"
        )

        try:
            # Load image and select band
            image = ee.Image(gee_dataset).select(band_name)

            # Build ee.FeatureCollection of training point geometries
            ee_points = ee.FeatureCollection([
                ee.Feature(
                    ee.Geometry.Point([row.geometry.x, row.geometry.y]),
                    {"_row_idx": int(idx)}
                )
                for idx, row in gdf.iterrows()
            ])

            # Sample image at each point
            sampled = image.sampleRegions(
                collection=ee_points,
                properties=["_row_idx"],
                scale=scale,
                geometries=False,
            )

            # Pull results to Python
            features = sampled.getInfo()["features"]

            # Build row_idx → sampled_value lookup
            value_lookup = {}
            for feat in features:
                props = feat["properties"]
                idx   = int(props["_row_idx"])
                val   = props.get(band_name, None)
                if val is not None:
                    value_lookup[idx] = float(val)

            if len(value_lookup) == 0:
                raise ValueError(
                    f"No values returned from {gee_dataset}/{band_name}. "
                    "Check that the dataset covers your AOI and date range."
                )

            logger.info(
                f"    Sampled {len(value_lookup)} / {len(gdf)} points "
                f"({100*len(value_lookup)/len(gdf):.0f}% coverage)."
            )

            # Apply transform
            def _apply_transform(v: float) -> float:
                if transform == "divide_100":
                    return v / 100.0
                if transform == "divide_max":
                    max_val = max(value_lookup.values())
                    return v / max_val if max_val > 0 else 0.0
                if transform == "log10":
                    return float(np.log10(v + 1e-6))
                return v  # "none"

            transformed = {
                idx: _apply_transform(v)
                for idx, v in value_lookup.items()
            }

            # Write values into GeoDataFrame
            if prim_col not in gdf.columns:
                gdf[prim_col] = np.nan

            for idx, val in transformed.items():
                if value_type == "binary":
                    gdf.at[idx, prim_col] = 1 if val > binary_threshold else 0
                else:
                    gdf.at[idx, prim_col] = round(val, 4)

            # Fill any points where the dataset had no coverage with 0
            no_coverage = [i for i in gdf.index if i not in value_lookup]
            if no_coverage:
                logger.warning(
                    f"    {len(no_coverage)} points had no coverage in "
                    f"{gee_dataset} — filled with 0."
                )
                gdf.loc[no_coverage, prim_col] = 0

            enriched_cols.append(prim_col)

        except Exception as exc:
            logger.error(f"  Failed to enrich '{prim_col}': {exc}")
            failed_cols.append((prim_col, str(exc)))

    # Convert enriched GeoDataFrame back to ee.FeatureCollection
    features_ee = [
        ee.Feature(
            ee.Geometry(row.geometry.__geo_interface__),
            row.drop("geometry").to_dict()
        )
        for _, row in gdf.iterrows()
    ]
    ee_fc = ee.FeatureCollection(features_ee)

    logger.info(
        f"Enrichment complete — "
        f"{len(enriched_cols)} filled, "
        f"{len(skipped_cols)} skipped, "
        f"{len(failed_cols)} failed."
    )

    return {
        "gdf":      gdf,
        "ee_fc":    ee_fc,
        "columns":  list(gdf.columns),
        "size":     len(gdf),
        "enriched": enriched_cols,
        "skipped":  skipped_cols,
        "failed":   failed_cols,
    }


class TrainingDataLabeller:
    """
    Labels training features with class_ids using a classification scheme.

    This is the direct pathway alternative to the primitive layer approach.
    Instead of building per-element probability maps, this class evaluates
    the scheme rules directly against the binary element attributes on each
    training feature and assigns the matching class_id.

    The resulting labeled ee.FeatureCollection can then be used to train a
    single direct Random Forest classifier per scheme.

    Parameters
    ----------
    rules_df : pd.DataFrame
        Output of load_scheme() — columns: class_id, class_name, rule, priority.
    scheme_name : str
        Used in log output for traceability.
    nodata_value : int
        class_id written to features that satisfy no rule. Default 0.

    Example
    -------
    >>> labeller   = TrainingDataLabeller(rules_df=scheme1_rules, scheme_name="scheme1")
    >>> labeled_fc = labeller.label(training_fc)
    >>> clean_fc   = labeled_fc.filter(ee.Filter.neq("class_id", 0))
    """

    def __init__(self, rules_df: pd.DataFrame, scheme_name: str,
                 nodata_value: int = 0):
        if rules_df.empty:
            raise ValueError(f"Empty rules_df passed for scheme '{scheme_name}'.")

        self.rules_df     = rules_df.sort_values("priority").reset_index(drop=True)
        self.scheme_name  = scheme_name
        self.nodata_value = nodata_value

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(
            f"TrainingDataLabeller initialised for '{scheme_name}' — "
            f"{len(self.rules_df)} classes, evaluated in priority order."
        )

    # ---------------------------------
    # Evaluate one rule against a feature
    # ---------------------------------

    @staticmethod
    def _evaluate_rule_on_feature(rule_expr: str, feat: ee.Feature) -> ee.Number:
        """
        Evaluate a rule string against a single ee.Feature.

        Returns ee.Number(1) if satisfied, ee.Number(0) otherwise.
        Supports AND / OR combinations and operators: ==  !=  >  >=  <  <=

        Parameters
        ----------
        rule_expr : str
            Rule expression, e.g. 'tree_pres > 0.6 AND buil_pres > 0.3'.
        feat : ee.Feature
            Feature whose properties are evaluated.
        """
        expr = rule_expr.replace("&&", " AND ").replace("||", " OR ")

        def _single(cond: str) -> ee.Number:
            cond = cond.strip()
            for op in [">=", "<=", "!=", ">", "<", "=="]:
                if op in cond:
                    left, right = cond.split(op, 1)
                    val    = ee.Number(feat.get(left.strip()))
                    thresh = float(right.strip())
                    return {
                        "==": val.eq,   "!=": val.neq,
                        ">":  val.gt,   ">=": val.gte,
                        "<":  val.lt,   "<=": val.lte,
                    }[op](thresh)
            raise ValueError(f"Cannot parse condition: '{cond}'")

        if " OR " in expr.upper():
            parts  = expr.split(" OR ")
            result = _single(parts[0])
            for part in parts[1:]:
                result = result.max(_single(part))
            return result.min(ee.Number(1))

        if " AND " in expr.upper():
            parts  = expr.split(" AND ")
            result = _single(parts[0])
            for part in parts[1:]:
                result = result.multiply(_single(part))
            return result

        return _single(expr)

    # ---------------------------------
    # Build the GEE map() function
    # ---------------------------------

    def _build_labeller_fn(self):
        """
        Build a function suitable for ee.FeatureCollection.map().

        Rules are applied from lowest to highest priority (reversed list),
        building a nested ee.Algorithms.If chain so that priority=1 wins.
        """
        rules_reversed = self.rules_df.iloc[::-1].reset_index(drop=True)
        nodata_value   = self.nodata_value

        def _labeller(feat):
            feat     = ee.Feature(feat)
            class_id = ee.Number(nodata_value)
            for _, row in rules_reversed.iterrows():
                cid       = int(row["class_id"])
                rule_expr = str(row["rule"])
                condition = TrainingDataLabeller._evaluate_rule_on_feature(
                    rule_expr, feat
                )
                class_id = ee.Number(
                    ee.Algorithms.If(condition.eq(1), cid, class_id)
                )
            return feat.set({"class_id": class_id})

        return _labeller

    # ---------------------------------
    # Public: label the feature collection
    # ---------------------------------

    def label(self, training_fc: ee.FeatureCollection) -> ee.FeatureCollection:
        """
        Apply the scheme rules to label every feature with a class_id.

        Parameters
        ----------
        training_fc : ee.FeatureCollection
            Reference data with element attributes as properties.

        Returns
        -------
        ee.FeatureCollection
            Same features with 'class_id' property added.
            Filter out nodata before training:
                labeled_fc.filter(ee.Filter.neq("class_id", 0))

        Example
        -------
        >>> labeled_fc = labeller.label(training_fc)
        >>> print(labeled_fc.first().getInfo())
        """
        labeller_fn = self._build_labeller_fn()
        labeled_fc  = training_fc.map(labeller_fn)

        self.logger.info(f"Labelling complete. Class distribution for '{self.scheme_name}':")
        all_ids = [self.nodata_value] + sorted(self.rules_df["class_id"].tolist())
        for cid in all_ids:
            count = labeled_fc.filter(ee.Filter.eq("class_id", cid)).size().getInfo()
            name  = (
                "nodata" if cid == self.nodata_value
                else self.rules_df.loc[
                    self.rules_df["class_id"] == cid, "class_name"
                ].values[0]
            )
            self.logger.info(f"  [{cid:>2}] {name:<18}: {count:>4} features")

        return labeled_fc


# ===========================================================================
# Step 3b: Primitive Layer Trainer
# PrimitiveLayerTrainer — trains one RF per LCML element
# ===========================================================================

class PrimitiveLayerTrainer:
    """
    Trains one Random Forest model per LCML element and generates a
    primitive layer (ee.Image) for each.

    The primitive datacube produced here is the core reusable asset of
    the workflow. It encodes per-pixel element presence independently of
    any classification scheme, allowing multiple schemes to be applied
    without retraining.

    Two training modes
    ------------------
        train_all()    — binary output (0 or 1), deterministic pathway.
        train_all_mc() — probabilistic output ([0, 1]), MC pathway.
                         Uses .setOutputMode('PROBABILITY').

    Parameters
    ----------
    image : ee.Image
        Multi-band predictor image (e.g. stacked Landsat composite).
        All bands are used as RF input features.
    roi : ee.FeatureCollection
        Reference data. Each feature must have binary (0/1) properties
        for each element to be trained.
    n_trees : int
        Number of trees per Random Forest. Default 50.
    scale : int
        Pixel scale in metres for region sampling. Default 30.

    Example
    -------
    >>> trainer         = PrimitiveLayerTrainer(image=stacked_image, roi=training_fc)
    >>> prob_layers     = trainer.train_all_mc()
    >>> primitive_stack = ee.Image.cat(list(prob_layers.values()))
    """

    def __init__(self, image: ee.Image, roi: ee.FeatureCollection,
                 n_trees: int = 50, scale: int = 30):
        self.image   = image
        self.roi     = roi
        self.n_trees = n_trees
        self.scale   = scale

        self.logger = logging.getLogger(self.__class__.__name__)

        self.primitives = self._detect_primitives()
        self.logger.info(f"Detected primitives: {self.primitives}")

    # ---------------------------------
    # Detect which columns are elements
    # ---------------------------------

    def _detect_primitives(self) -> list:
        """
        Read property names from the first reference feature and return
        those not in the exclusion list.
        """
        props = self.roi.first().propertyNames().getInfo()
        return [p for p in props if p not in _NON_PRIMITIVE_COLS]

    # ---------------------------------
    # Clean roi before sampling
    # ---------------------------------

    def _clean_roi(self) -> ee.FeatureCollection:
        """Remove system:index to avoid conflicts during sampleRegions."""
        return self.roi.map(
            lambda f: f.select(f.propertyNames().remove("system:index"))
        )

    # ---------------------------------
    # Shared RF training logic
    # ---------------------------------

    def _train_one(self, primitive: str, output_mode: str) -> ee.Image:
        """
        Train a single RF for one primitive and classify the image.

        Parameters
        ----------
        primitive : str
            Element property name (e.g. 'tree_pres').
        output_mode : str
            'CLASSIFICATION' for binary output, 'PROBABILITY' for [0,1].

        Returns
        -------
        ee.Image
            Single-band image named after the primitive.
        """
        sample = self.image.sampleRegions(
            collection=self._clean_roi(),
            properties=[primitive],
            scale=self.scale,
            geometries=False,
        )

        classifier = (
            ee.Classifier.smileRandomForest(self.n_trees)
            .setOutputMode(output_mode)
        )

        trained = classifier.train(
            features=sample,
            classProperty=primitive,
            inputProperties=self.image.bandNames(),
        )

        return self.image.classify(trained).rename(primitive)

    # ---------------------------------
    # Deterministic pathway
    # ---------------------------------

    def train_one(self, primitive: str) -> ee.Image:
        """
        Train a single primitive with binary (0/1) output.

        Parameters
        ----------
        primitive : str
            Element property name.

        Returns
        -------
        ee.Image
            Single-band binary image.

        Example
        -------
        >>> tree_layer = trainer.train_one("tree_pres")
        """
        return self._train_one(primitive, output_mode="CLASSIFICATION")

    def train_all(self) -> dict:
        """
        Train all detected primitives with binary output.

        Returns
        -------
        dict[str, ee.Image]
            Keys are primitive names, values are single-band binary ee.Images.

        Example
        -------
        >>> binary_layers = trainer.train_all()
        """
        outputs = {}
        for p in self.primitives:
            self.logger.info(f"Training (deterministic): {p}")
            outputs[p] = self.train_one(p)
        return outputs

    # ---------------------------------
    # Monte Carlo pathway
    # ---------------------------------

    def train_one_mc(self, primitive: str) -> ee.Image:
        """
        Train a single primitive with probabilistic [0, 1] output.

        The output is the fraction of RF trees that voted for class 1
        (presence) at each pixel — the per-pixel confidence score used
        by the Monte Carlo simulation.

        Parameters
        ----------
        primitive : str
            Element property name.

        Returns
        -------
        ee.Image
            Single-band image with float values in [0, 1].

        Example
        -------
        >>> tree_prob = trainer.train_one_mc("tree_pres")
        """
        return self._train_one(primitive, output_mode="PROBABILITY")

    def train_all_mc(self) -> dict:
        """
        Train all detected primitives with probabilistic output.

        Returns
        -------
        dict[str, ee.Image]
            Keys are primitive names, values are single-band ee.Images
            with float values in [0, 1].

        Notes
        -----
        Stack into a single image before passing to RuleSetClassifier:
            primitive_stack = ee.Image.cat(list(outputs.values()))

        Example
        -------
        >>> prob_layers     = trainer.train_all_mc()
        >>> primitive_stack = ee.Image.cat(list(prob_layers.values()))
        """
        outputs = {}
        for p in self.primitives:
            self.logger.info(f"Training (probabilistic): {p}")
            outputs[p] = self.train_one_mc(p)
        return outputs


# ===========================================================================
# Step 4: Rule Set Classifier
# RuleSetClassifier — applies scheme rules to produce a land cover map
# ===========================================================================

class RuleSetClassifier:
    """
    Classifies a primitive datacube using CSV-defined rules.

    Supports two classification methods:

        Deterministic — applies rules as hard thresholds on the GEE server.
                        No uncertainty output. Fast.

        Monte Carlo   — downloads probabilistic primitive arrays locally,
                        samples binary realisations per iteration, passes
                        them through the same ruleset, and aggregates to
                        produce a mode map, per-pixel entropy, and class
                        probability surfaces.

    Both methods use the same rules_df so results are directly comparable.

    Parameters
    ----------
    primitive_image : ee.Image
        Stack of primitive layers. Deterministic: binary bands.
        Monte Carlo: probabilistic bands in [0, 1] from train_all_mc().
    rules_df : pd.DataFrame
        Output of load_scheme() with a 'scheme' column added by the caller:
            rules_df["scheme"] = "scheme1"
    aoi : ee.FeatureCollection or ee.Geometry
        Area of interest for clipping results.

    Example
    -------
    >>> classifier = RuleSetClassifier(primitive_stack, rules, aoi)
    >>> det_map    = classifier.classify_scheme_deterministic("scheme1")
    >>> mc_results = classifier.classify_scheme_monte_carlo("scheme1", n_iterations=300)
    """

    def __init__(self, primitive_image: ee.Image, rules_df: pd.DataFrame, aoi):
        self.primitive_image = primitive_image
        self.df  = rules_df
        self.aoi = aoi

        self.logger = logging.getLogger(self.__class__.__name__)

    # ---------------------------------
    # Deterministic classification
    # ---------------------------------

    def classify_scheme_deterministic(self, scheme_name: str) -> ee.Image:
        """
        Classify using hard threshold rules on the GEE server.

        Rules are applied in priority order. Pixels satisfying no rule
        receive class_id = 0 (nodata).

        Parameters
        ----------
        scheme_name : str
            Must match the 'scheme' column in rules_df.

        Returns
        -------
        ee.Image
            Single-band image named 'class_id', clipped to AOI.

        Example
        -------
        >>> det_map = classifier.classify_scheme_deterministic("scheme1")
        """
        subset   = self._get_subset(scheme_name)
        aoi_geom = self._resolve_geometry()
        result   = ee.Image(0).rename("class_id").toFloat()
        band_dict = {b: self.primitive_image.select(b)
                     for b in self.primitive_image.bandNames().getInfo()}

        for _, row in subset.iterrows():
            class_id  = float(row["class_id"])
            rule_expr = row["rule"].replace("AND", "&&").replace("OR", "||")
            try:
                mask   = ee.Image().expression(rule_expr, band_dict).eq(1)
                result = result.where(mask, class_id)
            except Exception as exc:
                self.logger.warning(f"Skipping class_id {class_id}: {exc}")

        return result.clip(aoi_geom)

    def classify_all_schemes(self) -> dict:
        """
        Run deterministic classification for every scheme in rules_df.

        Returns
        -------
        dict[str, ee.Image]

        Example
        -------
        >>> maps = classifier.classify_all_schemes()
        """
        results = {}
        for scheme in self.df["scheme"].unique():
            self.logger.info(f"Classifying (deterministic): {scheme}")
            results[scheme] = self.classify_scheme_deterministic(scheme)
        return results

    # ---------------------------------
    # Monte Carlo classification
    # ---------------------------------

    def classify_scheme_monte_carlo(
        self,
        scheme_name: str,
        n_iterations: int = 300,
        nodata_value: float = 0.0,
        seed: Optional[int] = 42,
        scale: int = 30,
    ) -> dict:
        """
        Classify using repeated Bernoulli sampling from probabilistic primitives.

        In each iteration:
            1. Each pixel's probability p is sampled as Bernoulli(p) → 0 or 1.
            2. The binary values are passed through the deterministic ruleset.
            3. The class assignment is recorded.

        After all iterations the pixel counts are aggregated to produce:
            - mode_map    : most frequently assigned class per pixel
            - entropy_map : Shannon entropy of the class distribution (nats)
            - class_probs : fraction of iterations each class won per pixel

        High entropy = the simulation disagreed often = genuinely uncertain pixel.

        Parameters
        ----------
        scheme_name : str
            Must match the 'scheme' column in rules_df.
        n_iterations : int
            Number of MC draws. 200–500 is typically sufficient.
        nodata_value : float
            Class ID for pixels that match no rule in a given iteration.
        seed : int or None
            Random seed for reproducibility.
        scale : int
            Pixel scale in metres for the primitive array download.

        Returns
        -------
        dict with keys:
            "mode_map"    — (H, W) int array
            "entropy_map" — (H, W) float array (nats)
            "class_probs" — dict[class_id -> (H, W) float array]
            "n_iterations"— int

        Example
        -------
        >>> results     = classifier.classify_scheme_monte_carlo("scheme1", n_iterations=300)
        >>> mode_map    = results["mode_map"]
        >>> entropy_map = results["entropy_map"]
        """
        subset = self._get_subset(scheme_name)

        # Download primitive arrays once — all MC iterations use the same arrays
        self.logger.info(f"Downloading primitive arrays for '{scheme_name}'...")
        band_arrays = self._download_band_arrays(scale)
        self._validate_probability_range(band_arrays)

        H, W = next(iter(band_arrays.values())).shape

        # Set up class tracking
        all_class_ids = sorted(subset["class_id"].unique().tolist())
        if nodata_value not in all_class_ids:
            all_class_ids = [nodata_value] + all_class_ids

        class_id_to_idx = {cid: i for i, cid in enumerate(all_class_ids)}
        counts = np.zeros((len(all_class_ids), H, W), dtype=np.int32)
        rng    = np.random.default_rng(seed)

        # Monte Carlo loop
        self.logger.info(f"Running {n_iterations} Monte Carlo iterations...")
        for _ in range(n_iterations):
            binary_bands     = self._sample_bernoulli(band_arrays, rng, H, W)
            iteration_result = self._apply_ruleset(subset, binary_bands, H, W, nodata_value)
            for cid, idx in class_id_to_idx.items():
                counts[idx] += (iteration_result == cid).astype(np.int32)

        return self._aggregate(counts, all_class_ids, class_id_to_idx, n_iterations)

    # ---------------------------------
    # Internal helpers
    # ---------------------------------

    def _get_subset(self, scheme_name: str) -> pd.DataFrame:
        """Return rules for one scheme sorted by priority then class_id."""
        subset = self.df[self.df["scheme"] == scheme_name].copy()
        if subset.empty:
            raise ValueError(f"No rules found for scheme '{scheme_name}'.")
        return subset.sort_values(["priority", "class_id"]).reset_index(drop=True)

    def _resolve_geometry(self):
        """Return ee.Geometry regardless of whether AOI is FC or Geometry."""
        return self.aoi.geometry() if hasattr(self.aoi, "geometry") else self.aoi

    def _download_band_arrays(self, scale: int) -> dict:
        """
        Download all primitive bands as numpy arrays via getDownloadURL.

        Handles both single multi-band GeoTIFF and ZIP of single-band
        GeoTIFFs (GEE returns different formats depending on band count).

        Returns dict[band_name -> (H, W) float32 array].
        """
        aoi_geom   = self._resolve_geometry()
        band_names = self.primitive_image.bandNames().getInfo()

        self.logger.info(f"  Fetching {len(band_names)} bands at {scale}m resolution...")

        url = self.primitive_image.getDownloadURL({
            "bands":  band_names,
            "region": aoi_geom,
            "scale":  scale,
            "format": "GEO_TIFF",
            "crs":    "EPSG:4326",
        })

        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()
        raw = response.content

        band_arrays = {}

        if raw[:4] == b"PK\x03\x04":
            # ZIP of individual single-band GeoTIFFs
            with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                tif_names = sorted(n for n in zf.namelist() if n.endswith(".tif"))
                for band_name, tif_name in zip(band_names, tif_names):
                    with zf.open(tif_name) as f:
                        with rasterio.open(io.BytesIO(f.read())) as src:
                            band_arrays[band_name] = src.read(1).astype(np.float32)
        else:
            # Single multi-band GeoTIFF
            with rasterio.open(io.BytesIO(raw)) as src:
                for i, band_name in enumerate(band_names, start=1):
                    band_arrays[band_name] = src.read(i).astype(np.float32)

        h, w = next(iter(band_arrays.values())).shape
        self.logger.info(f"  Downloaded: {h} x {w} px ({h * w:,} pixels per band)")
        return band_arrays

    def _validate_probability_range(self, band_arrays: dict):
        """Raise if any band has values outside [0, 1]."""
        for name, arr in band_arrays.items():
            if arr.min() < 0 or arr.max() > 1:
                raise ValueError(
                    f"Band '{name}' has values outside [0, 1] "
                    f"(min={arr.min():.4f}, max={arr.max():.4f}). "
                    "Ensure primitives were trained with .setOutputMode('PROBABILITY')."
                )

    @staticmethod
    def _sample_bernoulli(band_arrays: dict, rng: np.random.Generator,
                          H: int, W: int) -> dict:
        """
        Sample one binary realisation from each probabilistic primitive.

        A pixel with p=0.94 becomes 1 in ~94% of iterations.
        A pixel with p=0.53 fluctuates nearly equally between 0 and 1.
        """
        return {
            name: (rng.random((H, W)) < prob_arr).astype(np.uint8)
            for name, prob_arr in band_arrays.items()
        }

    @staticmethod
    def _evaluate_rule_numpy(rule_expr: str, band_arrays: dict) -> np.ndarray:
        """
        Evaluate a rule expression over 2-D numpy arrays.

        Translates GEE-style operators (AND, OR, &&, ||) to numpy equivalents
        and evaluates using eval() with band arrays as local variables.

        Returns a boolean (H, W) array.
        """
        expr = (
            rule_expr
            .replace("&&",  " & ")
            .replace("||",  " | ")
            .replace("AND", " & ")
            .replace("OR",  " | ")
        )
        local_vars = {name: arr.astype(float) for name, arr in band_arrays.items()}
        try:
            result = eval(expr, {"__builtins__": {}}, local_vars)  # noqa: S307
            return np.asarray(result, dtype=bool)
        except Exception as exc:
            raise ValueError(f"Cannot evaluate rule '{rule_expr}': {exc}") from exc

    def _apply_ruleset(self, subset: pd.DataFrame, binary_bands: dict,
                       H: int, W: int, nodata_value: float) -> np.ndarray:
        """
        Apply all rules in priority order to the binary band arrays.

        Rules are applied sequentially — each rule overwrites earlier ones
        for matching pixels. Since rules are sorted by priority ascending,
        the highest-priority class (priority=1) is applied last and wins.
        """
        result = np.full((H, W), nodata_value, dtype=float)
        for _, row in subset.iterrows():
            class_id  = float(row["class_id"])
            rule_expr = str(row["rule"])
            try:
                mask = self._evaluate_rule_numpy(rule_expr, binary_bands)
                result[mask] = class_id
            except ValueError as exc:
                warnings.warn(str(exc), stacklevel=2)
        return result

    @staticmethod
    def _aggregate(counts: np.ndarray, all_class_ids: list,
                   class_id_to_idx: dict, n_iterations: int) -> dict:
        """Convert raw iteration counts into mode map, entropy map, class probs."""
        idx_to_class = np.array(all_class_ids, dtype=float)
        mode_map     = idx_to_class[np.argmax(counts, axis=0)].astype(int)

        probs = counts.astype(float) / n_iterations

        # Shannon entropy: H = -sum(p * ln(p)),  0*ln(0) = 0 by convention
        with np.errstate(divide="ignore", invalid="ignore"):
            log_probs = np.where(probs > 0, np.log(probs), 0.0)
        entropy_map = -np.sum(probs * log_probs, axis=0)

        class_probs = {cid: probs[idx] for cid, idx in class_id_to_idx.items()}

        return {
            "mode_map":     mode_map,
            "entropy_map":  entropy_map,
            "class_probs":  class_probs,
            "n_iterations": n_iterations,
        }


# ===========================================================================
# Module 4b: Hierarchical Dichotomous Key Classifier
# load_hierarchical_scheme()       — loads the tree-structured scheme CSV
# HierarchicalRuleSetClassifier    — evaluates primitives in a fixed order:
#                                    vegetation presence → veg properties
#                                    (cover, height, phenology) →
#                                    non-veg (bare, built, water)
# ===========================================================================

def load_hierarchical_scheme(csv_path: str) -> pd.DataFrame:
    """
    Load a hierarchical (dichotomous key) classification scheme CSV.

    Unlike load_scheme() which defines flat per-class rules, this function
    loads a tree-structured scheme where each row is a decision node.
    Nodes are evaluated in a fixed primitive-first order:

        L1  vegetation check  — tree_pres / shrub_pres
        L2a veg sub-keys      — tree_cover, vegetation_height, phenology
        L2b non-veg sub-keys  — bare_pres, builtup_pres, water_pres
        leaf                  — assigns final class_id

    Expected CSV columns
    --------------------
    node_id     int    Unique identifier for this node.
    parent_id   int    ID of parent node. Leave blank / NaN for root nodes.
    primitive   str    Primitive band name to evaluate at this node
                       (e.g. 'tree_pres', 'tree_cover', 'water_pres').
    operator    str    Comparison operator: '>=' | '>' | '<=' | '<' | '=='
    threshold   float  Threshold value for the comparison.
    class_id    int    Assigned class ID if this is a leaf node. NaN for splits.
    class_name  str    Human-readable class label (leaf nodes only).
    node_type   str    'split' — evaluates primitive and branches.
                       'leaf'  — assigns class_id, no further branching.
    priority    int    Evaluation order within the same parent level.
                       Lower number = evaluated first.

    Parameters
    ----------
    csv_path : str
        Path to the hierarchical scheme CSV.

    Returns
    -------
    pd.DataFrame
        Validated node table ready for HierarchicalRuleSetClassifier.

    Example
    -------
    >>> tree = load_hierarchical_scheme("scheme_hierarchical.csv")
    >>> classifier = HierarchicalRuleSetClassifier(primitive_stack, tree, aoi)
    """
    logger = logging.getLogger("load_hierarchical_scheme")

    df = pd.read_csv(csv_path)

    required = ["node_id", "primitive", "operator", "threshold", "node_type", "priority"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in '{csv_path}': {missing}"
        )

    valid_ops   = {">", ">=", "<", "<=", "==", "!="}
    invalid_ops = df[~df["operator"].isin(valid_ops)]["operator"].unique()
    if len(invalid_ops) > 0:
        raise ValueError(
            f"Unknown operators: {list(invalid_ops)}. Supported: {valid_ops}"
        )

    # parent_id NaN → root node
    if "parent_id" not in df.columns:
        df["parent_id"] = np.nan

    leaf_rows = df[df["node_type"] == "leaf"]
    missing_class = leaf_rows[leaf_rows["class_id"].isna()]
    if len(missing_class) > 0:
        raise ValueError(
            f"Leaf nodes missing class_id: node_ids "
            f"{missing_class['node_id'].tolist()}"
        )

    logger.info(
        f"Loaded hierarchical scheme: {len(df)} nodes "
        f"({(df['node_type']=='split').sum()} splits, "
        f"{(df['node_type']=='leaf').sum()} leaves)"
    )
    return df.sort_values(["parent_id", "priority"]).reset_index(drop=True)


class HierarchicalRuleSetClassifier:
    """
    Classifies a primitive datacube using a hierarchical dichotomous key.

    The key difference from RuleSetClassifier is the evaluation logic:
    instead of applying a flat priority-ordered list of class rules,
    this classifier traverses a decision tree where each node tests ONE
    primitive. The order of primitive evaluation is fixed by the tree
    structure — vegetation is always checked before cover, cover before
    height, non-veg before built/water/bare.

    This directly mirrors the LCML dichotomous key structure and avoids
    the reachability problem (where compound AND classes were structurally
    unreachable in the flat ruleset).

    Supports both deterministic and Monte Carlo classification. The MC
    method propagates uncertainty through the tree by sampling Bernoulli(p)
    at each primitive node per iteration — so entropy accumulates from
    every decision in the path, not just the final class assignment.

    Parameters
    ----------
    primitive_image : ee.Image
        Stack of probabilistic primitive layers in [0, 1].
    scheme_df : pd.DataFrame
        Output of load_hierarchical_scheme().
    aoi : ee.FeatureCollection or ee.Geometry
        Area of interest for clipping results.

    Example
    -------
    >>> tree  = load_hierarchical_scheme("scheme_hierarchical.csv")
    >>> clf   = HierarchicalRuleSetClassifier(primitive_stack, tree, aoi)
    >>> det   = clf.classify_deterministic()
    >>> mc    = clf.classify_monte_carlo(n_iterations=300)
    """

    def __init__(self, primitive_image: ee.Image,
                 scheme_df: pd.DataFrame, aoi):
        self.primitive_image = primitive_image
        self.scheme_df       = scheme_df
        self.aoi             = aoi
        self.logger          = logging.getLogger(self.__class__.__name__)

    # ---------------------------------
    # Internal: tree traversal (numpy)
    # ---------------------------------

    def _traverse_tree_numpy(
        self,
        band_arrays: dict,
        H: int,
        W: int,
        nodata_value: float = 0.0,
    ) -> np.ndarray:
        """
        Traverse the decision tree pixel-by-pixel using numpy arrays.

        Each pixel starts unassigned. At each node, the node's primitive
        is evaluated for all currently-unassigned pixels. Pixels that
        satisfy the condition either receive a class (leaf) or continue
        to child nodes (split). Pixels that fail a condition fall through
        to sibling nodes at the same level, then to the next level.

        The traversal is breadth-first within each parent, respecting
        the priority column for sibling ordering.

        Parameters
        ----------
        band_arrays : dict[str, np.ndarray]
            Binary (0/1) sampled primitive arrays for one MC iteration,
            or threshold-applied continuous arrays for deterministic use.
        H, W : int
            Spatial dimensions.
        nodata_value : float
            Assigned to pixels that reach no leaf node.

        Returns
        -------
        np.ndarray
            (H, W) array of class_id values.
        """
        result    = np.full((H, W), nodata_value, dtype=float)
        # unresolved mask: True = pixel not yet assigned a class
        unresolved = np.ones((H, W), dtype=bool)

        def _eval_node(row, mask: np.ndarray) -> np.ndarray:
            """Return boolean mask of pixels satisfying this node's condition."""
            prim   = row["primitive"]
            op     = row["operator"]
            thresh = float(row["threshold"])

            if prim not in band_arrays:
                self.logger.warning(
                    f"Primitive '{prim}' not in band_arrays — "
                    "node treated as always-false."
                )
                return np.zeros_like(mask)

            arr = band_arrays[prim].astype(float)
            ops = {
                ">":  arr >  thresh,
                ">=": arr >= thresh,
                "<":  arr <  thresh,
                "<=": arr <= thresh,
                "==": arr == thresh,
                "!=": arr != thresh,
            }
            return mask & ops[op]

        def _process_level(parent_id, active_mask: np.ndarray):
            """
            Recursively process all nodes whose parent_id matches.
            Pixels are claimed by the first node (in priority order)
            whose condition fires. Unclaimed pixels move to the next
            sibling, then fall through as unresolved.
            """
            if not active_mask.any():
                return

            if pd.isna(parent_id):
                children = self.scheme_df[self.scheme_df["parent_id"].isna()]
            else:
                children = self.scheme_df[
                    self.scheme_df["parent_id"].apply(
                        lambda x: (not pd.isna(x)) and (int(x) == int(parent_id))
                    )
                ]

            children = children.sort_values("priority")
            remaining = active_mask.copy()

            for _, node in children.iterrows():
                if not remaining.any():
                    break

                fires = _eval_node(node, remaining)

                if node["node_type"] == "leaf":
                    # Assign class to all pixels where this leaf fires
                    result[fires] = float(node["class_id"])
                    remaining[fires] = False

                elif node["node_type"] == "split":
                    # Recurse into children for pixels where this split fires
                    _process_level(int(node["node_id"]), fires)
                    # Pixels processed by children are no longer remaining
                    # (they were resolved inside the recursion or left unresolved)
                    remaining[fires] = False

        _process_level(np.nan, unresolved)
        return result

    # ---------------------------------
    # Deterministic classification
    # ---------------------------------

    def classify_deterministic(self, scale: int = 30) -> dict:
        """
        Classify using the decision tree with hard thresholds.

        Downloads the primitive stack, applies each node's threshold
        deterministically (value >= threshold → 1, else → 0), and
        traverses the tree. Returns a numpy array and an ee.Image.

        Parameters
        ----------
        scale : int
            Pixel scale in metres for downloading primitive arrays.

        Returns
        -------
        dict with keys:
            'class_map'  — (H, W) int array of class IDs
            'ee_image'   — ee.Image of the result clipped to AOI

        Example
        -------
        >>> result = clf.classify_deterministic()
        >>> print(result["class_map"].shape)
        """
        self.logger.info("Hierarchical deterministic classification...")
        band_arrays = self._download_band_arrays(scale)
        H, W        = next(iter(band_arrays.values())).shape

        # For deterministic: binarise at the node threshold
        # (each node evaluates the raw probability against its own threshold)
        class_map = self._traverse_tree_numpy(band_arrays, H, W)

        ee_image = self._numpy_to_ee_image(class_map)
        self.logger.info("Deterministic classification complete.")

        return {
            "class_map": class_map.astype(int),
            "ee_image":  ee_image,
        }

    # ---------------------------------
    # Monte Carlo classification
    # ---------------------------------

    def classify_monte_carlo(
        self,
        n_iterations:  int   = 300,
        nodata_value:  float = 0.0,
        seed:          Optional[int] = 42,
        scale:         int   = 30,
    ) -> dict:
        """
        Classify using repeated Bernoulli sampling through the decision tree.

        In each iteration, each primitive probability p is sampled as
        Bernoulli(p) → 0 or 1 at every pixel. The sampled binary values
        are passed through the decision tree. Uncertainty accumulates at
        every branching node in the path, not just the final class —
        a pixel that is uncertain at the vegetation/non-vegetation split
        will show high entropy even if the sub-tree it falls into is
        completely decisive.

        Parameters
        ----------
        n_iterations : int
            Number of Monte Carlo draws.
        nodata_value : float
            Assigned to pixels reaching no leaf in a given iteration.
        seed : int or None
            Random seed for reproducibility.
        scale : int
            Pixel scale in metres.

        Returns
        -------
        dict with keys:
            'mode_map'    — (H, W) int array of most-frequent class
            'entropy_map' — (H, W) float array (nats)
            'class_probs' — dict[class_id -> (H, W) float array]
            'n_iterations'— int

        Example
        -------
        >>> mc = clf.classify_monte_carlo(n_iterations=300)
        >>> print(mc["entropy_map"].mean())
        """
        self.logger.info(
            f"Hierarchical Monte Carlo classification — {n_iterations} iterations..."
        )
        band_arrays = self._download_band_arrays(scale)
        self._validate_probability_range(band_arrays)
        H, W = next(iter(band_arrays.values())).shape

        leaf_nodes    = self.scheme_df[self.scheme_df["node_type"] == "leaf"]
        all_class_ids = sorted(leaf_nodes["class_id"].dropna().unique().tolist())
        if nodata_value not in all_class_ids:
            all_class_ids = [nodata_value] + all_class_ids

        class_id_to_idx = {cid: i for i, cid in enumerate(all_class_ids)}
        counts = np.zeros((len(all_class_ids), H, W), dtype=np.int32)
        rng    = np.random.default_rng(seed)

        for _ in range(n_iterations):
            # Sample binary realisations from probabilistic primitives
            binary_bands = {
                name: (rng.random((H, W)) < prob_arr).astype(np.uint8)
                for name, prob_arr in band_arrays.items()
            }
            # Traverse the decision tree with this iteration's binary values
            iter_result = self._traverse_tree_numpy(
                binary_bands, H, W, nodata_value
            )
            for cid, idx in class_id_to_idx.items():
                counts[idx] += (iter_result == cid).astype(np.int32)

        # Aggregate
        idx_to_class = np.array(all_class_ids, dtype=float)
        mode_map     = idx_to_class[np.argmax(counts, axis=0)].astype(int)
        probs        = counts.astype(float) / n_iterations

        with np.errstate(divide="ignore", invalid="ignore"):
            log_probs = np.where(probs > 0, np.log(probs), 0.0)
        entropy_map = -np.sum(probs * log_probs, axis=0)
        class_probs = {cid: probs[idx] for cid, idx in class_id_to_idx.items()}

        self.logger.info("Monte Carlo classification complete.")
        return {
            "mode_map":     mode_map,
            "entropy_map":  entropy_map,
            "class_probs":  class_probs,
            "n_iterations": n_iterations,
        }

    # ---------------------------------
    # Internal helpers
    # ---------------------------------

    def _resolve_geometry(self):
        return self.aoi.geometry() if hasattr(self.aoi, "geometry") else self.aoi

    def _download_band_arrays(self, scale: int) -> dict:
        """Download all primitive bands as numpy float32 arrays."""
        aoi_geom   = self._resolve_geometry()
        band_names = self.primitive_image.bandNames().getInfo()
        self.logger.info(f"  Fetching {len(band_names)} bands at {scale}m...")

        url = self.primitive_image.getDownloadURL({
            "bands":  band_names,
            "region": aoi_geom,
            "scale":  scale,
            "format": "GEO_TIFF",
            "crs":    "EPSG:4326",
        })
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()
        raw = response.content

        band_arrays = {}
        if raw[:4] == b"PK\x03\x04":
            with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                tif_names = sorted(n for n in zf.namelist() if n.endswith(".tif"))
                for band_name, tif_name in zip(band_names, tif_names):
                    with zf.open(tif_name) as f:
                        with rasterio.open(io.BytesIO(f.read())) as src:
                            band_arrays[band_name] = src.read(1).astype(np.float32)
        else:
            with rasterio.open(io.BytesIO(raw)) as src:
                for i, band_name in enumerate(band_names, start=1):
                    band_arrays[band_name] = src.read(i).astype(np.float32)

        h, w = next(iter(band_arrays.values())).shape
        self.logger.info(f"  Downloaded: {h} x {w} px ({h*w:,} pixels per band)")
        return band_arrays

    def _validate_probability_range(self, band_arrays: dict):
        for name, arr in band_arrays.items():
            if arr.min() < 0 or arr.max() > 1:
                raise ValueError(
                    f"Band '{name}' outside [0,1] — use .setOutputMode('PROBABILITY')."
                )

    def _numpy_to_ee_image(self, class_map: np.ndarray) -> ee.Image:
        """Upload a numpy class map back to GEE as an ee.Image."""
        aoi_geom = self._resolve_geometry()
        H, W     = class_map.shape
        flat     = class_map.flatten().tolist()
        return (
            ee.Image(ee.Array(flat).reshape([H, W]))
            .rename("class_id")
            .clip(aoi_geom)
        )

    def summary(self) -> None:
        """Print the decision tree structure for inspection."""
        print("\nDecision tree structure:")
        print(f"{'node_id':>8} {'parent':>8} {'priority':>8}  "
              f"{'type':<8} {'primitive':<20} {'op':>4} {'thresh':>8}  class")
        print("-" * 76)
        for _, row in self.scheme_df.iterrows():
            pid   = int(row["parent_id"]) if not pd.isna(row["parent_id"]) else "root"
            cname = row.get("class_name", "") if row["node_type"] == "leaf" else ""
            cid   = int(row["class_id"])  if not pd.isna(row.get("class_id")) else ""
            print(
                f"{int(row['node_id']):>8} {str(pid):>8} {int(row['priority']):>8}  "
                f"{row['node_type']:<8} {row['primitive']:<20} "
                f"{row['operator']:>4} {float(row['threshold']):>8.2f}  "
                f"{cid} {cname}"
            )


# ===========================================================================
# Module 5: Validation
# validate_deterministic() — download and summarise a deterministic result
# validate_monte_carlo()   — summarise and plot an MC result
# compare_det_vs_mc()      — side-by-side comparison
# ===========================================================================

def _ee_image_to_numpy(ee_image, aoi, scale: int = 30) -> np.ndarray:
    """Download a single-band ee.Image as a (H, W) float32 numpy array."""
    aoi_geom = aoi.geometry() if hasattr(aoi, "geometry") else aoi

    url = ee_image.getDownloadURL({
        "bands":  ee_image.bandNames().getInfo(),
        "region": aoi_geom,
        "scale":  scale,
        "format": "GEO_TIFF",
        "crs":    "EPSG:4326",
    })

    response = requests.get(url, stream=True, timeout=300)
    response.raise_for_status()
    raw = response.content

    if raw[:4] == b"PK\x03\x04":
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            tif_name = sorted(n for n in zf.namelist() if n.endswith(".tif"))[0]
            with zf.open(tif_name) as f:
                with rasterio.open(io.BytesIO(f.read())) as src:
                    return src.read(1).astype(np.float32)
    else:
        with rasterio.open(io.BytesIO(raw)) as src:
            return src.read(1).astype(np.float32)


def validate_deterministic(
    det_ee_image,
    rules_df: pd.DataFrame,
    scheme_name: str,
    aoi,
    scale: int = 30,
    scheme_label: str = "",
) -> np.ndarray:
    """
    Download and validate a deterministic classification result.

    Produces a console summary in the same format as validate_monte_carlo()
    so results are directly comparable. Also shows a class map and area
    bar chart.

    Parameters
    ----------
    det_ee_image : ee.Image
        Output of RuleSetClassifier.classify_scheme_deterministic().
    rules_df : pd.DataFrame
        Rules table (must contain 'scheme' column).
    scheme_name : str
        Scheme key in rules_df.
    aoi : ee.FeatureCollection or ee.Geometry
    scale : int
        Pixel scale in metres.
    scheme_label : str
        Display label for plot titles.

    Returns
    -------
    np.ndarray
        (H, W) int array of class IDs — same shape as MC mode_map.

    Example
    -------
    >>> det_arr = validate_deterministic(det_map, rules, "scheme1", aoi)
    """
    logger     = logging.getLogger("validate_deterministic")
    label      = scheme_label or scheme_name
    subset     = rules_df[rules_df["scheme"] == scheme_name].copy()
    id_to_name = dict(zip(subset["class_id"], subset["class_name"]))
    class_ids  = sorted(subset["class_id"].unique().tolist())

    logger.info(f"Downloading deterministic result for '{label}'...")
    class_map    = _ee_image_to_numpy(det_ee_image, aoi, scale).astype(int)
    H, W         = class_map.shape
    total_pixels = class_map.size

    # Console summary
    print(f"\n{'='*55}")
    print(f"  Validation — {label}  (deterministic)")
    print(f"{'='*55}")
    print(f"  Spatial extent : {H} x {W} px")
    print(f"  Entropy        : 0.0000 nats  (deterministic — no uncertainty)")
    print(f"\n  Per-class area share:")

    area_shares = {}
    for cid in [0] + class_ids:
        pct  = 100 * (class_map == cid).sum() / total_pixels
        name = id_to_name.get(cid, f"class {cid}")
        area_shares[cid] = pct
        print(f"    [{int(cid):>2}] {name:<20}  area={pct:5.1f}%")
    print(f"{'='*55}\n")

    # Figure: class map + area bar chart
    fig = plt.figure(figsize=(12, 5), constrained_layout=True)
    fig.suptitle(f"Deterministic — {label}", fontsize=13, fontweight="500")
    gs = GridSpec(1, 2, figure=fig)

    all_ids     = [0] + class_ids
    cmap        = mcolors.ListedColormap(
        [_CLASS_PALETTE[i % len(_CLASS_PALETTE)] for i in range(len(all_ids))]
    )
    bounds      = [i - 0.5 for i in range(len(all_ids) + 1)]
    norm        = mcolors.BoundaryNorm(bounds, cmap.N)
    display_map = np.zeros_like(class_map)
    for idx, cid in enumerate(all_ids):
        display_map[class_map == cid] = idx

    ax_map = fig.add_subplot(gs[0, 0])
    im     = ax_map.imshow(display_map, cmap=cmap, norm=norm, interpolation="nearest")
    cbar   = plt.colorbar(im, ax=ax_map, ticks=range(len(all_ids)),
                          fraction=0.046, pad=0.04)
    cbar.set_ticklabels(
        [f"[{int(c)}] {id_to_name.get(c, 'nodata')}" for c in all_ids]
    )
    ax_map.set_title("Class map")
    ax_map.axis("off")

    ax_bar      = fig.add_subplot(gs[0, 1])
    named_ids   = [c for c in class_ids if area_shares.get(c, 0) > 0]
    named_names = [id_to_name.get(c, f"class {c}") for c in named_ids]
    named_areas = [area_shares[c] for c in named_ids]
    bar_colours = [_CLASS_PALETTE[(i + 1) % len(_CLASS_PALETTE)]
                   for i in range(len(named_ids))]

    bars = ax_bar.barh(named_names, named_areas, color=bar_colours,
                       edgecolor="none", height=0.5)
    for bar, pct in zip(bars, named_areas):
        ax_bar.text(bar.get_width() + 1,
                    bar.get_y() + bar.get_height() / 2,
                    f"{pct:.1f}%", va="center", fontsize=9)
    ax_bar.set_xlabel("Area share (%)")
    ax_bar.set_title("Per-class area share")
    ax_bar.set_xlim(0, 100)

    plt.show()
    return class_map


def validate_monte_carlo(
    results: dict,
    rules_df: pd.DataFrame,
    scheme_name: str,
    entropy_threshold: float = 0.5,
    scheme_label: str = "",
):
    """
    Summarise and visualise a Monte Carlo classification result.

    Produces a console summary and four types of plot:
        - Entropy distribution histogram
        - Entropy spatial map (with high-uncertainty overlay)
        - Mode map
        - Per-class probability histograms

    Parameters
    ----------
    results : dict
        Output of RuleSetClassifier.classify_scheme_monte_carlo().
    rules_df : pd.DataFrame
        Rules table (must contain 'scheme' column).
    scheme_name : str
        Scheme key in rules_df.
    entropy_threshold : float
        Pixels above this value (nats) are flagged as high-uncertainty.
        A principled choice is ln(n_classes) / 2.
    scheme_label : str
        Display label for titles.

    Example
    -------
    >>> validate_monte_carlo(mc_results, rules, "scheme1", entropy_threshold=0.5)
    """
    label       = scheme_label or scheme_name
    mode_map    = results["mode_map"]
    entropy_map = results["entropy_map"]
    class_probs = results["class_probs"]
    n_iter      = results["n_iterations"]

    subset     = rules_df[rules_df["scheme"] == scheme_name].copy()
    id_to_name = dict(zip(subset["class_id"], subset["class_name"]))
    n_classes  = len(class_probs)

    total_pixels    = mode_map.size
    high_unc_pixels = (entropy_map > entropy_threshold).sum()
    high_unc_pct    = 100 * high_unc_pixels / total_pixels

    # Console summary
    print(f"\n{'='*55}")
    print(f"  Validation — {label}  ({n_iter} iterations)")
    print(f"{'='*55}")
    print(f"  Spatial extent   : {mode_map.shape[0]} x {mode_map.shape[1]} px")
    print(f"  Entropy range    : {entropy_map.min():.4f} – {entropy_map.max():.4f} nats")
    print(f"  Mean entropy     : {entropy_map.mean():.4f} nats")
    print(f"  High-uncertainty : {high_unc_pixels:,} px  "
          f"({high_unc_pct:.1f}%,  threshold={entropy_threshold})")
    print(f"\n  Per-class area share (mode map):")
    for cid, prob_map in class_probs.items():
        area_pct  = 100 * (mode_map == cid).sum() / total_pixels
        mean_conf = prob_map.mean()
        name      = id_to_name.get(cid, f"class {cid}")
        print(f"    [{int(cid):>2}] {name:<20}  "
              f"area={area_pct:5.1f}%   mean prob={mean_conf:.3f}")
    print(f"{'='*55}\n")

    # Figure layout: [entropy hist | entropy map | mode map] + per-class hists
    n_cols = max(3, min(n_classes, 4))
    n_rows = 2 + (n_classes - 1) // n_cols
    fig    = plt.figure(figsize=(5 * n_cols, 4 * n_rows), constrained_layout=True)
    fig.suptitle(f"Monte Carlo validation — {label}", fontsize=13, fontweight="500")
    gs = GridSpec(n_rows, n_cols, figure=fig)

    # Row 0a: entropy distribution histogram
    ax_hist = fig.add_subplot(gs[0, 0])
    ax_hist.hist(entropy_map.ravel(), bins=60, color="#5DCAA5",
                 edgecolor="none", alpha=0.85)
    ax_hist.axvline(entropy_threshold, color="#D85A30", linewidth=1.5,
                    linestyle="--", label=f"threshold={entropy_threshold}")
    ax_hist.axvline(entropy_map.mean(), color="#7F77DD", linewidth=1.5,
                    label=f"mean={entropy_map.mean():.3f}")
    ax_hist.set_title("Entropy distribution")
    ax_hist.set_xlabel("Shannon entropy (nats)")
    ax_hist.legend(fontsize=9)

    # Row 0b: entropy spatial map with high-uncertainty overlay
    ax_emap = fig.add_subplot(gs[0, 1])
    emap    = ax_emap.imshow(entropy_map, cmap="YlOrRd", vmin=0,
                             vmax=entropy_map.max())
    plt.colorbar(emap, ax=ax_emap, fraction=0.046, pad=0.04, label="nats")
    overlay = np.where(entropy_map > entropy_threshold, 1.0, np.nan)
    ax_emap.imshow(overlay, cmap="cool", alpha=0.4, vmin=0, vmax=1)
    ax_emap.set_title(f"Entropy map  ({high_unc_pct:.1f}% high-unc, cyan)")
    ax_emap.axis("off")

    # Row 0c: mode map
    class_ids  = sorted(id_to_name.keys())
    id_to_idx  = {cid: i for i, cid in enumerate(class_ids)}
    mode_idx   = np.vectorize(lambda x: id_to_idx.get(x, -1))(mode_map)
    colours    = plt.cm.tab20(np.linspace(0, 1, len(class_ids)))
    cmap_mode  = mcolors.ListedColormap(colours)
    norm_mode  = mcolors.BoundaryNorm(
        np.arange(len(class_ids) + 1) - 0.5, len(class_ids)
    )
    ax_mode = fig.add_subplot(gs[0, 2])
    ax_mode.imshow(mode_idx, cmap=cmap_mode, norm=norm_mode)
    ax_mode.set_title("Mode map")
    ax_mode.axis("off")
    handles = [
        plt.Line2D([0], [0], marker="s", color=colours[i],
                   linestyle="", markersize=8, label=id_to_name[cid])
        for i, cid in enumerate(class_ids)
    ]
    ax_mode.legend(handles=handles, bbox_to_anchor=(1.05, 1),
                   loc="upper left", fontsize=8)

    # Rows 1+: per-class probability histograms
    class_items = [(cid, p) for cid, p in class_probs.items() if cid != 0.0]
    for i, (cid, prob_map) in enumerate(class_items):
        row = 1 + i // n_cols
        col = i % n_cols
        ax  = fig.add_subplot(gs[row, col])
        ax.hist(prob_map.ravel(), bins=50, color="#378ADD",
                edgecolor="none", alpha=0.85)
        ax.axvline(0.5, color="#E24B4A", linestyle="--", label="p=0.5")
        ax.axvline(prob_map.mean(), color="#BA7517",
                   label=f"mean={prob_map.mean():.3f}")
        ax.set_title(f"[{int(cid)}] {id_to_name.get(cid, f'class {cid}')}")
        ax.set_xlabel("P(class assigned)")
        ax.legend(fontsize=8)

    plt.show()


def compare_det_vs_mc(
    det_class_map: np.ndarray,
    mc_results: dict,
    rules_df: pd.DataFrame,
    scheme_name: str,
    scheme_label: str = "",
):
    """
    Compare a deterministic class map against the MC mode map.

    Prints a side-by-side area share table and shows three panels:
        - Deterministic class map
        - MC mode map
        - Disagreement map coloured by MC entropy

    Parameters
    ----------
    det_class_map : np.ndarray
        Return value of validate_deterministic() — (H, W) int array.
    mc_results : dict
        Return value of RuleSetClassifier.classify_scheme_monte_carlo().
    rules_df : pd.DataFrame
        Rules table (must contain 'scheme' column).
    scheme_name : str
        Scheme key in rules_df.
    scheme_label : str
        Display label for titles.

    Example
    -------
    >>> compare_det_vs_mc(det_arr, mc_results, rules, "scheme1")
    """
    label      = scheme_label or scheme_name
    subset     = rules_df[rules_df["scheme"] == scheme_name].copy()
    id_to_name = dict(zip(subset["class_id"], subset["class_name"]))
    class_ids  = sorted(subset["class_id"].unique().tolist())

    mode_map    = mc_results["mode_map"]
    entropy_map = mc_results["entropy_map"]
    total       = det_class_map.size

    agreement     = (det_class_map == mode_map).sum()
    agreement_pct = 100 * agreement / total
    disagree_pct  = 100 - agreement_pct

    # Console summary
    print(f"\n{'='*55}")
    print(f"  Deterministic vs MC — {label}")
    print(f"{'='*55}")
    print(f"  Pixel agreement : {agreement:,} / {total:,}  ({agreement_pct:.1f}%)")
    print(f"  Mean MC entropy : {entropy_map.mean():.4f} nats")
    print(f"\n  {'Class':<22}  {'Det':>8}  {'MC':>8}  {'Δ':>7}")
    print(f"  {'-'*50}")
    for cid in class_ids:
        name    = id_to_name.get(cid, f"class {cid}")
        det_pct = 100 * (det_class_map == cid).sum() / total
        mc_pct  = 100 * (mode_map == cid).sum() / total
        delta   = mc_pct - det_pct
        sign    = "+" if delta >= 0 else ""
        print(f"  [{int(cid):>2}] {name:<18}  "
              f"{det_pct:>7.1f}%  {mc_pct:>7.1f}%  {sign}{delta:>5.1f}%")
    print(f"{'='*55}\n")

    # Three-panel figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    fig.suptitle(f"Deterministic vs MC — {label}", fontsize=13, fontweight="500")

    axes[0].imshow(det_class_map, cmap="tab10", interpolation="nearest")
    axes[0].set_title("Deterministic")
    axes[0].axis("off")

    axes[1].imshow(mode_map, cmap="tab10", interpolation="nearest")
    axes[1].set_title("MC mode map")
    axes[1].axis("off")

    # Disagreement map: only show entropy where the two methods disagree
    disagree_display = np.where(det_class_map != mode_map, entropy_map, 0)
    im = axes[2].imshow(disagree_display, cmap="YlOrRd", vmin=0,
                        vmax=entropy_map.max(), interpolation="nearest")
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04, label="MC entropy (nats)")
    axes[2].set_title(
        f"Disagreement ({disagree_pct:.1f}% of pixels)\ncoloured by MC entropy"
    )
    axes[2].axis("off")

    plt.show()