"""

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
# Step 1b: Setup classifiation ruleset
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
        """Join all non-NaN, non-exclusion conditions with AND or OR."""
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

            op_text = str(row[rule_col])

            # Skip zero-exclusions — "equal to 0" means "not present",
            # which is handled by priority ordering, not explicit AND conditions
            if _is_exclusion(op_text, thresh):
                continue

            conds.append(_build_condition(
                pres_col, rule_col, op_text, thresh, row["class_id"]
            ))

        if not conds:
            raise ValueError(
                f"No non-exclusion conditions for class_id {row['class_id']}. "
                "Check that at least one *_pres column has a non-zero threshold."
            )

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
# TrainingDataLabeller         — labels features by scheme rules (direct pathway)
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
# Post Step 4: Validation
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

def compare_rf_vs_mc(
    rf_class_map: np.ndarray,
    mc_results: dict,
    rules_df: pd.DataFrame,
    scheme_name: str,
    scheme_label: str = "",
):
    """
    Compare a Random Forest classification map against the MC mode map.

    Outputs ONLY statistics (no plots), focusing on disagreement behaviour.

    Parameters
    ----------
    rf_class_map : np.ndarray
        Classified map from Random Forest (e.g. classified_step3a).
    mc_results : dict
        Output from classify_scheme_monte_carlo().
    rules_df : pd.DataFrame
        Rules table with 'scheme', 'class_id', 'class_name'.
    scheme_name : str
        Scheme key.
    scheme_label : str
        Optional display name.
    """

    label      = scheme_label or scheme_name
    subset     = rules_df[rules_df["scheme"] == scheme_name].copy()
    id_to_name = dict(zip(subset["class_id"], subset["class_name"]))
    class_ids  = sorted(subset["class_id"].unique().tolist())

    mode_map    = mc_results["mode_map"]
    entropy_map = mc_results["entropy_map"]

    total = rf_class_map.size

    # --- Agreement stats ---
    agreement     = (rf_class_map == mode_map).sum()
    agreement_pct = 100 * agreement / total
    disagree_mask = rf_class_map != mode_map
    disagree_pct  = 100 - agreement_pct

    # --- Entropy stats ONLY where disagreement happens ---
    disagreement_entropy = entropy_map[disagree_mask]

    mean_entropy_disagree = disagreement_entropy.mean() if disagreement_entropy.size else 0
    p90_entropy_disagree  = np.percentile(disagreement_entropy, 90) if disagreement_entropy.size else 0

    # --- Console output ---
    print(f"\n{'='*60}")
    print(f"  RF vs MC Comparison — {label}")
    print(f"{'='*60}")
    print(f"  Pixel agreement       : {agreement:,} / {total:,} ({agreement_pct:.1f}%)")
    print(f"  Pixel disagreement    : {disagree_pct:.1f}%")

    print("\n  --- Entropy where disagreement occurs ---")
    print(f"  Mean entropy          : {mean_entropy_disagree:.4f} nats")
    print(f"  90th percentile       : {p90_entropy_disagree:.4f} nats")

    # --- Per-class disagreement ---
    print(f"\n  {'Class':<22} {'RF%':>7} {'MC%':>7} {'Disagree%':>10}")
    print(f"  {'-'*55}")

    for cid in class_ids:
        name = id_to_name.get(cid, f"class {cid}")

        rf_mask = rf_class_map == cid
        mc_mask = mode_map == cid

        rf_pct = 100 * rf_mask.sum() / total
        mc_pct = 100 * mc_mask.sum() / total

        # disagreement involving this class (either side)
        class_disagree = ((rf_class_map == cid) | (mode_map == cid)) & disagree_mask
        class_disagree_pct = 100 * class_disagree.sum() / total

        print(f"  [{int(cid):>2}] {name:<18} "
              f"{rf_pct:>6.1f}% {mc_pct:>6.1f}% {class_disagree_pct:>9.1f}%")

    # --- Confusion insight ---
    print(f"\n  --- Top Confusions (RF → MC) ---")
    print(f"  {'RF class':<20} {'MC class':<20} {'Pixels':>10}")

    from collections import Counter

    confusion_pairs = Counter(
        zip(rf_class_map[disagree_mask].ravel(),
            mode_map[disagree_mask].ravel())
    )

    for (rf_c, mc_c), count in confusion_pairs.most_common(10):
        rf_name = id_to_name.get(rf_c, str(rf_c))
        mc_name = id_to_name.get(mc_c, str(mc_c))
        print(f"  {rf_name:<20} → {mc_name:<20} {count:>10}")

    print(f"{'='*60}\n")