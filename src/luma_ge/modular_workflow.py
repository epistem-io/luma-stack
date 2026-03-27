import pandas as pd

def load_scheme(csv_path):
    """
    Load scheme that automatically turns them into ruleset with threshold + operator columns.
    """
    
    df = pd.read_csv(csv_path)

    
    required_cols = ["class_id", "class_name", "rule_general", "priority"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in {csv_path}")
    
    rules = []
    
    for _, row in df.iterrows():
        # Find *_pres columns (thresholds) - exclude rule columns
        pres_cols = [c for c in df.columns if c.endswith("_pres") and not c.startswith("rule_")]
        
        # For "none" rules, find the FIRST non-NaN *_pres column as the single condition
        if str(row["rule_general"]).lower().strip() == "none":
            single_cond_pres = None
            for pres_col in pres_cols:
                if pd.isna(row[pres_col]):
                    continue
                
                # Find the rule column for this primitive
                primitive_name = pres_col.replace("_pres", "")
                rule_col = f"rule_{primitive_name}_pres"
                if rule_col not in df.columns:
                    continue
                
                op_text = str(row[rule_col]).lower().strip()
                
                # Skip "equal to 0" conditions — these are exclusions, not the
                # defining condition. Pick the first non-zero threshold column.
                thresh = row[pres_col]
                if op_text == "equal to" and float(thresh) == 0:
                    continue
                single_cond_pres = pres_col
                break
            
            if single_cond_pres is None:
                raise ValueError(f"'none' rule in row {row['class_id']} has no threshold values in *_pres columns")
            
            # Build ONLY this one condition
            primitive_name = single_cond_pres.replace("_pres", "")
            rule_col = f"rule_{primitive_name}_pres"
            
            if rule_col not in df.columns:
                raise ValueError(f"No matching rule column '{rule_col}' for '{single_cond_pres}' in row {row['class_id']}")
            
            op_text = row[rule_col].lower().strip()
            thresh = row[single_cond_pres]
            
            op_map = {
                "more than": ">", "greater than": ">",
                "equal to": "==",
                "less than": "<",
                "more or equal": ">=", "greater or equal": ">=", "greater or equal to": ">",
                "less or equal": "<=", "less or equal to": "<="
            }
            
            if op_text not in op_map:
                raise ValueError(f"Unknown operator '{row[rule_col]}' in row {row['class_id']}")
            
            rule_expr = f"{single_cond_pres} {op_map[op_text]} {thresh}"
        
        else:  # "and" or "or" - use ALL non-NaN conditions
            conds = []
            for pres_col in pres_cols:
                thresh = row[pres_col]
                if pd.isna(thresh):
                    continue
                
                primitive_name = pres_col.replace("_pres", "")
                rule_col = f"rule_{primitive_name}_pres"
                
                if rule_col not in df.columns:
                    raise ValueError(f"No matching rule column '{rule_col}' for '{pres_col}' in row {row['class_id']}")
                
                op_text = row[rule_col].lower().strip()
                thresh = row[pres_col]
                
                op_map = {
                    "more than": ">", "greater than": ">",
                    "equal to": "==",
                    "less than": "<",
                    "more or equal": ">=", "greater or equal": ">=", "greater or equal to": ">",
                    "less or equal": "<=", "less or equal to": "<="
                }
                
                if op_text not in op_map:
                    raise ValueError(f"Unknown operator '{row[rule_col]}' in row {row['class_id']}")
                
                conds.append(f"{pres_col} {op_map[op_text]} {thresh}")
            
            if not conds:
                raise ValueError(f"No conditions found for row {row['class_id']}")
            
            combination_type = str(row["rule_general"]).lower().strip()
            if combination_type == "and":
                rule_expr = " AND ".join(conds)
            elif combination_type == "or":
                rule_expr = " OR ".join(conds)
            else:
                raise ValueError(f"Invalid rule_general '{row['rule_general']}'")
        
        rules.append(rule_expr)
    
    df["rule"] = rules
    return df[["class_id", "class_name", "rule", "priority"]]

import geopandas as gpd
import ee
from shapely.geometry import shape


def load_modular_training_data(
    shp_path,
    aoi=None,
):
    """
    Load shapefile with LCML attributes
    and filter by AOI (EE FeatureCollection).
    """

    # -------------------------
    # read shapefile
    # -------------------------

    gdf = gpd.read_file(shp_path)

    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")

    gdf = gdf.to_crs("EPSG:4326")

    # -------------------------
    # AOI filter (EE FeatureCollection)
    # -------------------------

    if aoi is not None:

        # convert EE FeatureCollection → geometry → shapely
        aoi_geom = aoi.geometry().getInfo()

        aoi_shape = shape(aoi_geom)

        gdf = gdf[gdf.intersects(aoi_shape)]

    # -------------------------
    # convert to EE FeatureCollection
    # -------------------------

    features = []

    for _, row in gdf.iterrows():

        geom = ee.Geometry(row.geometry.__geo_interface__)

        props = row.drop("geometry").to_dict()

        features.append(
            ee.Feature(geom, props)
        )

    ee_fc = ee.FeatureCollection(features)

    # -------------------------
    # output
    # -------------------------

    return {
        "gdf": gdf,
        "ee_fc": ee_fc,
        "columns": list(gdf.columns),
        "size": len(gdf),
    }

from .classification import FeatureExtraction
from .classification import Generate_LULC

class PrimitiveLayerTrainer:

    def __init__(self, image, roi):

        self.image = image
        self.roi = roi

        self.primitives = self._get_primitives_from_training()


    # ---------------------------------
    # detect primitives from attributes
    # ---------------------------------

    def _get_primitives_from_training(self):

        props = (
            self.roi.first()
            .propertyNames()
            .getInfo()
        )

        exclude = ["LULC_Type", "ID", "geometry", "system:index"]

        primitives = [
            p for p in props
            if p not in exclude
        ]

        print("Detected primitives:", primitives)

        return primitives


    # ---------------------------------
    # train one primitive
    # ---------------------------------

    def train_one(self, primitive):

        # remove system:index
        roi_clean = self.roi.map(
            lambda f: f.select(
                f.propertyNames().remove("system:index")
            )
        )

        sample = self.image.sampleRegions(
            collection=roi_clean,
            properties=[primitive],
            scale=30,
            geometries=False
        )

        classifier = ee.Classifier.smileRandomForest(50)

        trained = classifier.train(
            features=sample,
            classProperty=primitive,
            inputProperties=self.image.bandNames()
        )

        result = self.image.classify(trained)

        return result.rename(primitive)

    # ---------------------------------
    # train all primitives
    # ---------------------------------

    def train_all(self):

        outputs = {}

        for p in self.primitives:

            print("Training:", p)

            outputs[p] = self.train_one(p)

        return outputs
    
        # ---------------------------------
    # train one primitive using RF classifier with probability output
    # ---------------------------------

    def train_one_mc(self, primitive):

        # remove system:index
        roi_clean = self.roi.map(
            lambda f: f.select(
                f.propertyNames().remove("system:index") # additional prevention in case system:index is present in the training data, which can cause issues with sampling and classification
            )
        )

    # Feature Extraction of each elements

        sample = self.image.sampleRegions(
            collection=roi_clean,
            properties=[primitive],
            scale=30,
            geometries=False
        )

    # Classification 
        classifier = ee.Classifier.smileRandomForest(50)\
            .setOutputMode('PROBABILITY') #result will be in probability value for monte carlo testing

        trained = classifier.train(
            features=sample,
            classProperty=primitive,
            inputProperties=self.image.bandNames()
        )

        result = self.image.classify(trained)

        return result.rename(primitive)

    # ---------------------------------
    # train all primitives
    # ---------------------------------

    def train_all_mc(self):

        outputs = {}

        for p in self.primitives:

            print("Training:", p)

            outputs[p] = self.train_one_mc(p)

        return outputs
    
import ee
import io
import zipfile
import rasterio
import numpy as np
import pandas as pd
import requests
import warnings
from typing import Optional


class RuleSetClassifier:
    """
    Classify an EE primitive image using CSV-defined rules.
    Supports multiple schemes and rule-based methods.
    """

    def __init__(self, primitive_image: ee.Image, rules_df: pd.DataFrame, aoi):
        """
        Args:
            primitive_image (ee.Image): Stack of primitive layers.
            rules_df (pd.DataFrame): Rules table with columns:
                                     class_id, class_name, rule, scheme
            aoi (ee.FeatureCollection or ee.Geometry): Area of interest for clipping
        """
        self.primitive_image = primitive_image
        self.df = rules_df
        self.aoi = aoi

    # -------------------------------------------------------------------------
    # Deterministic classification
    # -------------------------------------------------------------------------

    def classify_scheme_deterministic(self, scheme_name: str) -> ee.Image:
        subset = self.df[self.df["scheme"] == scheme_name]
        if subset.empty:
            raise ValueError(f"No rules found for scheme '{scheme_name}'")

        aoi_geom = self.aoi.geometry() if hasattr(self.aoi, "geometry") else self.aoi
        result = ee.Image(0).rename("class_id").toFloat()

        band_names = self.primitive_image.bandNames().getInfo()
        band_dict = {b: self.primitive_image.select(b) for b in band_names}

        for _, row in subset.iterrows():
            class_id = float(row["class_id"])
            rule_expr = row["rule"].replace("AND", "&&").replace("OR", "||")
            try:
                mask = ee.Image().expression(rule_expr, band_dict).eq(1)
                result = result.where(mask, class_id)
            except Exception as e:
                print(f"Skipping class_id {class_id} due to EE expression error: {e}")

        return result.clip(aoi_geom)

    # -------------------------------------------------------------------------

    def classify_all_schemes(self) -> dict:
        """
        Run deterministic classification for all unique schemes in rules_df.
        Returns a dictionary of {scheme_name: ee.Image}.
        """
        results = {}
        for s in self.df["scheme"].unique():
            print(f"Classifying scheme: {s}")
            results[s] = self.classify_scheme_deterministic(s)
        return results

    # -------------------------------------------------------------------------
    # Monte Carlo classification (probabilistic primitives)
    # -------------------------------------------------------------------------

    @staticmethod
    def _evaluate_rule_numpy(
        rule_expr: str,
        band_arrays: dict,
    ) -> np.ndarray:
        """
        Evaluate a GEE-style boolean expression over named numpy arrays.
        Supports: &&  ||  AND  OR  ==  !=  >=  <=  >  <  numeric literals.
        Returns a boolean 2-D array.
        """
        expr = (
            rule_expr
            .replace("&&", " & ")
            .replace("||", " | ")
            .replace("AND", " & ")
            .replace("OR", " | ")
        )
        local_vars = {name: arr.astype(float) for name, arr in band_arrays.items()}
        try:
            result = eval(expr, {"__builtins__": {}}, local_vars)  # noqa: S307
            return np.asarray(result, dtype=bool)
        except Exception as exc:
            raise ValueError(f"Cannot evaluate rule '{rule_expr}': {exc}") from exc

    # -------------------------------------------------------------------------

    def _get_band_arrays(self, scale: int = 30) -> dict:
        """
        Pull all primitive bands from the EE image as numpy arrays using
        getDownloadURL. Works correctly with shapefile-derived AOIs loaded
        via geemap.shp_to_ee() (ee.FeatureCollection or ee.Geometry).
 
        The image is exported as a multi-band GeoTIFF zip, then read back
        with rasterio. Band order in the file matches bandNames() order.
 
        Returns dict[band_name -> (H, W) float32 array].
        """
        # Resolve geometry — works for both ee.FeatureCollection and ee.Geometry
        aoi_geom = self.aoi.geometry() if hasattr(self.aoi, "geometry") else self.aoi
        band_names = self.primitive_image.bandNames().getInfo()
 
        print(f"  Fetching {len(band_names)} bands via getDownloadURL "
              f"(scale={scale}m)...")
 
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
 
        # GEE returns either a raw GeoTIFF (single band) or a ZIP of per-band
        # GeoTIFFs depending on the number of bands requested.
        if raw[:4] == b"PK\x03\x04":
            # --- ZIP of individual single-band GeoTIFFs ----------------------
            with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                tif_names = sorted(n for n in zf.namelist() if n.endswith(".tif"))
                if len(tif_names) != len(band_names):
                    raise ValueError(
                        f"Expected {len(band_names)} GeoTIFFs in ZIP, "
                        f"got {len(tif_names)}: {tif_names}"
                    )
                for band_name, tif_name in zip(band_names, tif_names):
                    with zf.open(tif_name) as f:
                        with rasterio.open(io.BytesIO(f.read())) as src:
                            band_arrays[band_name] = src.read(1).astype(np.float32)
        else:
            # --- Single multi-band GeoTIFF -----------------------------------
            with rasterio.open(io.BytesIO(raw)) as src:
                if src.count != len(band_names):
                    raise ValueError(
                        f"Expected {len(band_names)} bands in GeoTIFF, "
                        f"got {src.count}."
                    )
                for i, band_name in enumerate(band_names, start=1):
                    band_arrays[band_name] = src.read(i).astype(np.float32)
 
        shapes = {n: a.shape for n, a in band_arrays.items()}
        unique_shapes = set(shapes.values())
        if len(unique_shapes) > 1:
            raise ValueError(f"Band arrays have inconsistent shapes: {shapes}")
 
        h, w = next(iter(shapes.values()))
        print(f"  Downloaded arrays: {h} x {w} px  "
              f"({h * w:,} pixels per band)")
 
        return band_arrays

    # -------------------------------------------------------------------------

    def classify_scheme_monte_carlo(
        self,
        scheme_name: str,
        n_iterations: int = 200,
        nodata_value: float = 0.0,
        seed: Optional[int] = 42,
        scale: int = 30,
    ) -> dict:
        """
        Monte Carlo LULC classification from probabilistic primitive layers.

        Each primitive band contains per-pixel probabilities in [0, 1] as
        produced by GEE's smileRandomForest with .setOutputMode('PROBABILITY').
        In each iteration, every pixel is binarised by sampling Bernoulli(p),
        and the resulting binary primitives are passed through the deterministic
        ruleset — propagating per-pixel uncertainty into the final class map.

        Parameters
        ----------
        scheme_name : str
            Which classification scheme to use (must exist in self.df).
        n_iterations : int
            Number of Monte Carlo draws (200–500 is usually sufficient).
        nodata_value : float
            Class ID written to pixels that match no rule in a given iteration.
        seed : int or None
            Random seed for reproducibility.
        scale : int
            Pixel scale in metres used when pulling arrays from GEE.

        Returns
        -------
        dict with keys:
            'mode_map'    – (H, W) int array  : most-frequent class_id per pixel.
            'entropy_map' – (H, W) float array: Shannon entropy (nats); high
                            values indicate low confidence / high disagreement.
            'class_probs' – dict[class_id -> (H, W) float array]: fraction of
                            iterations each class was assigned per pixel.
            'n_iterations'– int: number of iterations run.
        """
        subset = self.df[self.df["scheme"] == scheme_name].copy()
        if subset.empty:
            raise ValueError(f"No rules found for scheme '{scheme_name}'")

        subset = subset.sort_values(["priority", "class_id"]).reset_index(drop=True)

        # --- Pull probabilistic primitive arrays from GEE (once) --------------
        print(f"Pulling primitive arrays from GEE for scheme '{scheme_name}'...")
        band_arrays = self._get_band_arrays(scale=scale)

        # Validate that all bands are probability values in [0, 1]
        for band_name, arr in band_arrays.items():
            if arr.min() < 0 or arr.max() > 1:
                raise ValueError(
                    f"Band '{band_name}' contains values outside [0, 1]. "
                    "Ensure primitives are trained with .setOutputMode('PROBABILITY')."
                )

        sample_band = next(iter(band_arrays.values()))
        H, W = sample_band.shape

        all_class_ids = sorted(subset["class_id"].unique().tolist())
        if nodata_value not in all_class_ids:
            all_class_ids = [nodata_value] + all_class_ids

        class_id_to_idx = {cid: i for i, cid in enumerate(all_class_ids)}
        n_classes = len(all_class_ids)
        counts = np.zeros((n_classes, H, W), dtype=np.int32)

        rng = np.random.default_rng(seed)

        print(f"Running {n_iterations} Monte Carlo iterations...")
        for i in range(n_iterations):

            # --- 1. Sample binary realisations from each probabilistic primitive
            #
            #   p = 0.94  ->  almost always 1   (confident tree pixel)
            #   p = 0.53  ->  nearly coin flip  (uncertain pixel)
            #   p = 0.07  ->  almost always 0   (confident non-tree pixel)
            #
            binary_bands = {
                band_name: (rng.random((H, W)) < prob_arr).astype(np.uint8)
                for band_name, prob_arr in band_arrays.items()
            }

            # --- 2. Apply deterministic ruleset to sampled binary primitives ---
            iteration_result = np.full((H, W), nodata_value, dtype=float)

            for _, row in subset.iterrows():
                class_id = float(row["class_id"])
                try:
                    mask = self._evaluate_rule_numpy(row["rule"], binary_bands)
                    iteration_result[mask] = class_id
                except ValueError as exc:
                    warnings.warn(str(exc), stacklevel=2)

            # --- 3. Accumulate per-class pixel counts --------------------------
            for cid, idx in class_id_to_idx.items():
                counts[idx] += (iteration_result == cid).astype(np.int32)

        # -----------------------------------------------------------------------
        # Aggregate across iterations
        # -----------------------------------------------------------------------

        # Mode map: class with the highest iteration count per pixel
        best_idx = np.argmax(counts, axis=0)
        idx_to_class_id = np.array(all_class_ids, dtype=float)
        mode_map = idx_to_class_id[best_idx].astype(int)

        # Empirical class probabilities  p_c = count_c / n_iterations
        probs = counts.astype(float) / n_iterations  # (n_classes, H, W)

        # Shannon entropy  H = -sum(p * ln(p)),  convention: 0 * ln(0) = 0
        with np.errstate(divide="ignore", invalid="ignore"):
            log_probs = np.where(probs > 0, np.log(probs), 0.0)
        entropy_map = -np.sum(probs * log_probs, axis=0)  # (H, W)

        class_probs = {cid: probs[idx] for cid, idx in class_id_to_idx.items()}

        return {
            "mode_map": mode_map,
            "entropy_map": entropy_map,
            "class_probs": class_probs,
            "n_iterations": n_iterations,
        }
    
"""
Validation for deterministic LULC classification.
Pulls the ee.Image result as a numpy array and produces
the same console summary + plots as validate_monte_carlo(),
so both approaches can be compared side by side.
"""

import io
import zipfile
import requests
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# ---------------------------------------------------------------------------
# Helper: pull a single-band ee.Image to numpy
# ---------------------------------------------------------------------------

def _ee_image_to_numpy(ee_image, aoi, scale: int = 30) -> np.ndarray:
    """
    Download a single-band ee.Image clipped to aoi as a numpy array.
    Works with ee.FeatureCollection or ee.Geometry AOIs (e.g. from
    geemap.shp_to_ee).
    """
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


# ---------------------------------------------------------------------------
# Main validation function
# ---------------------------------------------------------------------------

def validate_deterministic(
    det_ee_image,
    rules_df: pd.DataFrame,
    scheme_name: str,
    aoi,
    scale: int = 30,
    scheme_label: str = "",
):
    """
    Validation for a deterministic classification result.
    Produces the same console summary and plots as validate_monte_carlo()
    so the two approaches can be directly compared.

    Parameters
    ----------
    det_ee_image : ee.Image
        Output of classify_scheme_deterministic() — single band 'class_id'.
    rules_df : pd.DataFrame
        Rules table used for classification (to get class names).
    scheme_name : str
        Scheme key in rules_df.
    aoi : ee.FeatureCollection or ee.Geometry
        Area of interest — same object passed to RuleSetClassifier.
    scale : int
        Pixel scale in metres for downloading the result.
    scheme_label : str
        Display label for plot titles.

    Returns
    -------
    class_map : np.ndarray
        (H, W) int array of class IDs, matching mode_map shape from MC.
    """
    label  = scheme_label or scheme_name
    subset = rules_df[rules_df["scheme"] == scheme_name].copy()
    id_to_name  = dict(zip(subset["class_id"], subset["class_name"]))
    class_ids   = sorted(subset["class_id"].unique().tolist())

    # --- Pull ee.Image to numpy ---------------------------------------------
    print(f"Downloading deterministic result for '{label}'...")
    class_map = _ee_image_to_numpy(det_ee_image, aoi, scale=scale).astype(int)
    H, W = class_map.shape
    total_pixels = class_map.size

    # --- Console summary ----------------------------------------------------
    print(f"\n{'='*55}")
    print(f"  Validation — {label}  (deterministic)")
    print(f"{'='*55}")
    print(f"  Spatial extent  : {H} x {W} px")
    print(f"  Entropy         : 0.0000 nats  (no uncertainty — deterministic)")
    print(f"\n  Per-class area share (class map):")

    area_shares = {}
    for cid in [0] + class_ids:
        area_pct = 100 * (class_map == cid).sum() / total_pixels
        area_shares[cid] = area_pct
        name = id_to_name.get(cid, f"class {cid}")
        print(f"    [{int(cid):>2}] {name:<20}  area={area_pct:5.1f}%")

    print(f"{'='*55}\n")

    # --- Figure: class map + per-class area bar chart -----------------------
    n_cols = 2
    fig = plt.figure(figsize=(12, 5), constrained_layout=True)
    fig.suptitle(f"Deterministic validation — {label}", fontsize=13, fontweight="500")
    gs = GridSpec(1, n_cols, figure=fig)

    # Panel 1: spatial class map
    base_palette = [
        "#888780",  # 0 nodata   — gray
        "#1D9E75",  # 1          — teal
        "#378ADD",  # 2          — blue
        "#D85A30",  # 3          — coral
        "#BA7517",  # 4          — amber
        "#7F77DD",  # 5          — purple
        "#639922",  # 6          — green
    ]
    all_ids  = [0] + class_ids
    cmap     = plt.cm.colors.ListedColormap(
        [base_palette[i % len(base_palette)] for i in range(len(all_ids))]
    )
    bounds   = [i - 0.5 for i in range(len(all_ids) + 1)]
    norm     = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

    # Remap class IDs to contiguous indices for imshow
    display_map = np.zeros_like(class_map)
    for idx, cid in enumerate(all_ids):
        display_map[class_map == cid] = idx

    ax_map = fig.add_subplot(gs[0, 0])
    im = ax_map.imshow(display_map, cmap=cmap, norm=norm, interpolation="nearest")
    cbar = plt.colorbar(im, ax=ax_map, ticks=range(len(all_ids)),
                        fraction=0.046, pad=0.04)
    cbar.set_ticklabels(
        [f"[{int(cid)}] {id_to_name.get(cid, 'nodata')}" for cid in all_ids]
    )
    ax_map.set_title("Class map", fontsize=11)
    ax_map.axis("off")

    # Panel 2: area bar chart (only named classes, skip nodata=0)
    ax_bar = fig.add_subplot(gs[0, 1])
    named_ids   = [cid for cid in class_ids if area_shares.get(cid, 0) > 0]
    named_names = [id_to_name.get(cid, f"class {cid}") for cid in named_ids]
    named_areas = [area_shares[cid] for cid in named_ids]
    bar_colors  = [base_palette[(i + 1) % len(base_palette)]
                   for i, cid in enumerate(class_ids)
                   if area_shares.get(cid, 0) > 0]

    bars = ax_bar.barh(named_names, named_areas, color=bar_colors,
                       edgecolor="none", height=0.5)
    ax_bar.set_xlabel("Area share (%)", fontsize=11)
    ax_bar.set_title("Per-class area share", fontsize=11)
    ax_bar.set_xlim(0, 100)

    for bar, pct in zip(bars, named_areas):
        ax_bar.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                    f"{pct:.1f}%", va="center", fontsize=9)

    plt.show()

    return class_map

"""
Implementation: Monte Carlo LULC classification
with validation checks and geemap visualization.
"""
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec

# ===========================================================================
# VALIDATION
# ===========================================================================

def validate_monte_carlo(
    results: dict,
    rules_df: pd.DataFrame,
    scheme_name: str,
    entropy_threshold: float = 0.5,
    scheme_label: str = "",
):

    mode_map    = results["mode_map"]
    entropy_map = results["entropy_map"]
    class_probs = results["class_probs"]
    n_iter      = results["n_iterations"]

    subset = rules_df[rules_df["scheme"] == scheme_name].copy()
    id_to_name = dict(zip(subset["class_id"], subset["class_name"]))

    label = scheme_label or scheme_name
    n_classes = len(class_probs)

    # -----------------------------------------------------------------------
    # Console summary
    # -----------------------------------------------------------------------
    total_pixels = mode_map.size
    high_unc_pixels = (entropy_map > entropy_threshold).sum()
    high_unc_pct = 100 * high_unc_pixels / total_pixels

    print(f"\n{'='*55}")
    print(f"  Validation — {label}  ({n_iter} iterations)")
    print(f"{'='*55}")
    print(f"  Spatial extent  : {mode_map.shape[0]} x {mode_map.shape[1]} px")
    print(f"  Entropy range   : {entropy_map.min():.4f} – {entropy_map.max():.4f} nats")
    print(f"  Mean entropy    : {entropy_map.mean():.4f} nats")
    print(f"  High-uncertainty pixels (entropy > {entropy_threshold}): "
          f"{high_unc_pixels:,}  ({high_unc_pct:.1f}%)")

    print(f"\n  Per-class area share (mode map):")
    for cid, prob_map in class_probs.items():
        area_pct = 100 * (mode_map == cid).sum() / total_pixels
        mean_conf = prob_map.mean()
        name = id_to_name.get(cid, f"class {cid}")
        print(f"    [{int(cid):>2}] {name:<20}  "
              f"area={area_pct:5.1f}%   mean prob={mean_conf:.3f}")
    print(f"{'='*55}\n")

    class_entropy = {}
    for cid, prob_map in class_probs.items():
        arr = prob_map.ravel()
        # Shannon entropy per pixel: -p*log(p) - (1-p)*log(1-p)
        # Avoid log(0) by adding epsilon
        eps = 1e-10
        pixel_entropy = -(arr * np.log(arr + eps) + (1 - arr) * np.log(1 - arr + eps))
        # Mean entropy for pixels where class probability > 0 (ignore empty areas)
        mask = arr > 0
        if np.any(mask):
            class_entropy[cid] = pixel_entropy[mask].mean()
        else:
            class_entropy[cid] = 0.0

    print(f"\n  Per-class area share and entropy (mode map):")
    for cid, prob_map in class_probs.items():
        area_pct = 100 * (mode_map == cid).sum() / total_pixels
        mean_conf = prob_map.mean()
        mean_ent = class_entropy[cid]
        name = id_to_name.get(cid, f"class {cid}")
        print(f"    [{int(cid):>2}] {name:<20}  "
            f"area={area_pct:5.1f}%   mean prob={mean_conf:.3f}   mean entropy={mean_ent:.3f}")

    # -----------------------------------------------------------------------
    # Figure layout
    # row 0: hist | entropy | mode
    # rows 1+: histograms
    # -----------------------------------------------------------------------

    n_cols = max(3, min(n_classes, 4))
    n_rows = 2 + (n_classes - 1) // n_cols

    fig = plt.figure(figsize=(5 * n_cols, 4 * n_rows), constrained_layout=True)
    fig.suptitle(f"Monte Carlo validation — {label}", fontsize=13, fontweight="500")
    gs = GridSpec(n_rows, n_cols, figure=fig)

    # -----------------------------------------------------------------------
    # Row 0 — entropy histogram
    # -----------------------------------------------------------------------

    ax_hist = fig.add_subplot(gs[0, 0])

    ax_hist.hist(
        entropy_map.ravel(),
        bins=60,
        color="#5DCAA5",
        edgecolor="none",
        alpha=0.85,
    )

    ax_hist.axvline(
        entropy_threshold,
        color="#D85A30",
        linewidth=1.5,
        linestyle="--",
        label=f"threshold = {entropy_threshold}",
    )

    ax_hist.axvline(
        entropy_map.mean(),
        color="#7F77DD",
        linewidth=1.5,
        linestyle="-",
        label=f"mean = {entropy_map.mean():.3f}",
    )

    ax_hist.set_title("Entropy distribution")
    ax_hist.legend(fontsize=9)

    # -----------------------------------------------------------------------
    # Row 0 — entropy map
    # -----------------------------------------------------------------------

    ax_map = fig.add_subplot(gs[0, 1])

    emap = ax_map.imshow(
        entropy_map,
        cmap="YlOrRd",
        vmin=0,
        vmax=entropy_map.max(),
    )

    plt.colorbar(
        emap,
        ax=ax_map,
        fraction=0.046,
        pad=0.04,
        label="entropy (nats)",
    )

    high_unc_mask = np.where(entropy_map > entropy_threshold, 1.0, np.nan)

    ax_map.imshow(
        high_unc_mask,
        cmap="cool",
        alpha=0.4,
        vmin=0,
        vmax=1,
    )

    ax_map.set_title(
        f"Entropy map ({high_unc_pct:.1f}% high-unc)"
    )

    ax_map.axis("off")

    # -----------------------------------------------------------------------
    # Row 0 — mode map
    # -----------------------------------------------------------------------

    ax_mode = fig.add_subplot(gs[0, 2])

    class_ids = sorted(id_to_name.keys())

    colors = plt.cm.tab20(np.linspace(0, 1, len(class_ids)))
    cmap = mcolors.ListedColormap(colors)

    norm = mcolors.BoundaryNorm(
        boundaries=np.arange(len(class_ids) + 1) - 0.5,
        ncolors=len(class_ids),
    )

    # remap class ids → index
    id_to_idx = {cid: i for i, cid in enumerate(class_ids)}

    mode_idx = np.vectorize(lambda x: id_to_idx.get(x, -1))(mode_map)

    im = ax_mode.imshow(
        mode_idx,
        cmap=cmap,
        norm=norm,
    )

    ax_mode.set_title("Mode map")
    ax_mode.axis("off")

    # legend
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="s",
            color=colors[i],
            linestyle="",
            markersize=8,
            label=id_to_name[cid],
        )
        for i, cid in enumerate(class_ids)
    ]

    ax_mode.legend(
        handles=handles,
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        fontsize=8,
    )

    # -----------------------------------------------------------------------
    # Per-class histograms
    # -----------------------------------------------------------------------

    class_items = [
        (cid, prob_map)
        for cid, prob_map in class_probs.items()
        if cid != 0.0
    ]

    for i, (cid, prob_map) in enumerate(class_items):

        row = 1 + i // n_cols
        col = i % n_cols

        ax = fig.add_subplot(gs[row, col])

        name = id_to_name.get(cid, f"class {cid}")

        ax.hist(
            prob_map.ravel(),
            bins=50,
            color="#378ADD",
            edgecolor="none",
            alpha=0.85,
        )

        ax.axvline(
            0.5,
            color="#E24B4A",
            linestyle="--",
        )

        ax.axvline(
            prob_map.mean(),
            color="#BA7517",
        )

        ax.set_title(f"[{int(cid)}] {name}")

    plt.show()


# ---------------------------------------------------------------------------
# Side-by-side comparison helper
# ---------------------------------------------------------------------------

def compare_det_vs_mc(
    det_class_map: np.ndarray,
    mc_results: dict,
    rules_df: pd.DataFrame,
    scheme_name: str,
    scheme_label: str = "",
):
    """
    Print a side-by-side area share comparison and compute pixel-level
    agreement between the deterministic map and the MC mode map.

    Parameters
    ----------
    det_class_map : np.ndarray
        Output of validate_deterministic() — (H, W) int array.
    mc_results : dict
        Output of classify_scheme_monte_carlo().
    rules_df : pd.DataFrame
        Rules table (for class names).
    scheme_name : str
        Scheme key in rules_df.
    scheme_label : str
        Display label.
    """
    label      = scheme_label or scheme_name
    subset     = rules_df[rules_df["scheme"] == scheme_name].copy()
    id_to_name = dict(zip(subset["class_id"], subset["class_name"]))
    class_ids  = sorted(subset["class_id"].unique().tolist())

    mode_map    = mc_results["mode_map"]
    entropy_map = mc_results["entropy_map"]
    total       = det_class_map.size

    # Overall pixel agreement
    agreement     = (det_class_map == mode_map).sum()
    agreement_pct = 100 * agreement / total

    print(f"\n{'='*55}")
    print(f"  Deterministic vs MC — {label}")
    print(f"{'='*55}")
    print(f"  Overall pixel agreement : {agreement:,} / {total:,}  "
          f"({agreement_pct:.1f}%)")
    print(f"  Mean MC entropy         : {entropy_map.mean():.4f} nats")
    print(f"\n  {'Class':<22}  {'Det area':>9}  {'MC area':>9}  {'Δ':>7}")
    print(f"  {'-'*52}")

    for cid in class_ids:
        name     = id_to_name.get(cid, f"class {cid}")
        det_pct  = 100 * (det_class_map == cid).sum() / total
        mc_pct   = 100 * (mode_map == cid).sum() / total
        delta    = mc_pct - det_pct
        sign     = "+" if delta >= 0 else ""
        print(f"  [{int(cid):>2}] {name:<18}  "
              f"{det_pct:>8.1f}%  {mc_pct:>8.1f}%  {sign}{delta:>5.1f}%")

    print(f"{'='*55}\n")

    # --- Disagreement map ---------------------------------------------------
    disagree_mask = (det_class_map != mode_map).astype(float)
    disagree_pct  = 100 * disagree_mask.mean()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    fig.suptitle(f"Deterministic vs MC comparison — {label}",
                 fontsize=13, fontweight="500")

    # Deterministic class map
    axes[0].imshow(det_class_map, cmap="tab10", interpolation="nearest")
    axes[0].set_title("Deterministic", fontsize=11)
    axes[0].axis("off")

    # MC mode map
    axes[1].imshow(mode_map, cmap="tab10", interpolation="nearest")
    axes[1].set_title("MC mode map", fontsize=11)
    axes[1].axis("off")

    # Disagreement map — white=agree, red=disagree
    # Overlay MC entropy as intensity so high-entropy disagreements stand out
    disagree_display = np.where(disagree_mask == 1, entropy_map, 0)
    im = axes[2].imshow(disagree_display, cmap="YlOrRd",
                        vmin=0, vmax=entropy_map.max(),
                        interpolation="nearest")
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04,
                 label="MC entropy (nats)")
    axes[2].set_title(
        f"Disagreement pixels ({disagree_pct:.1f}%)\ncoloured by MC entropy",
        fontsize=11
    )
    axes[2].axis("off")

    plt.show()

import ee
import pandas as pd

class TrainingDataLabeller:
    """
    Labels an ee.FeatureCollection with class_ids according to a
    pre-defined scheme of rules.
    """

    def __init__(self, rules_df: pd.DataFrame, scheme_name: str, nodata_value: int = 0):
        """
        Parameters
        ----------
        rules_df : pd.DataFrame
            DataFrame with columns: class_id, class_name, rule, priority
            Rules should be logical expressions already in the 'rule' column.
        scheme_name : str
            Name of the scheme for printing/logging purposes.
        nodata_value : int
            class_id assigned to features that satisfy no rule.
        """
        if rules_df.empty:
            raise ValueError(f"No rules found for scheme '{scheme_name}'")

        self.rules_df = rules_df.sort_values("priority").reset_index(drop=True)
        self.scheme_name = scheme_name
        self.nodata_value = nodata_value
        print(f"Initialized labeller for '{scheme_name}' "
              f"({len(self.rules_df)} classes, priority order)")

    @staticmethod
    def _build_ee_condition(rule_expr: str, feat: ee.Feature) -> ee.Number:
        """
        Evaluate a single rule expression against an ee.Feature.
        Returns ee.Number(1) if satisfied, ee.Number(0) otherwise.
        Supports AND / OR combinations.
        """
        expr = rule_expr.replace("&&", " AND ").replace("||", " OR ")

        def _eval_single(cond: str) -> ee.Number:
            cond = cond.strip()
            for op in [">=", "<=", "!=", ">", "<", "=="]:
                if op in cond:
                    left, right = cond.split(op, 1)
                    prop_val = ee.Number(feat.get(left.strip()))
                    thresh = float(right.strip())
                    return {
                        "==": prop_val.eq,
                        "!=": prop_val.neq,
                        ">": prop_val.gt,
                        ">=": prop_val.gte,
                        "<": prop_val.lt,
                        "<=": prop_val.lte
                    }[op](thresh)
            raise ValueError(f"Cannot parse condition: '{cond}'")

        if " OR " in expr.upper():
            parts_orig = expr.split(" OR ")
            result = _eval_single(parts_orig[0])
            for part in parts_orig[1:]:
                result = result.max(_eval_single(part))
            return result.min(ee.Number(1))

        if " AND " in expr.upper():
            parts_orig = expr.split(" AND ")
            result = _eval_single(parts_orig[0])
            for part in parts_orig[1:]:
                result = result.multiply(_eval_single(part))
            return result

        return _eval_single(expr)

    def _make_labeller(self):
        """
        Returns a function that labels a feature according to all rules.
        """
        rules_reversed = self.rules_df.iloc[::-1].reset_index(drop=True)
        nodata_value = self.nodata_value

        def labeller(feat):
            feat = ee.Feature(feat)
            class_id = ee.Number(nodata_value)
            for _, row in rules_reversed.iterrows():
                cid = int(row["class_id"])
                rule_expr = str(row["rule"])
                condition = self._build_ee_condition(rule_expr, feat)
                class_id = ee.Number(ee.Algorithms.If(condition.eq(1), cid, class_id))
            return feat.set({"class_id": class_id})

        return labeller

    def label(self, training_fc: ee.FeatureCollection) -> ee.FeatureCollection:
        """
        Map the labeller function over the feature collection.
        Prints class distribution after labeling.
        """
        labeller = self._make_labeller()
        labeled_fc = training_fc.map(labeller)

        print(f"Labelling complete for '{self.scheme_name}'. Checking distribution...")
        subset_ids = [self.nodata_value] + sorted(self.rules_df["class_id"].tolist())
        for cid in subset_ids:
            count = labeled_fc.filter(ee.Filter.eq("class_id", cid)).size().getInfo()
            name = "nodata" if cid == self.nodata_value else \
                   self.rules_df.loc[self.rules_df["class_id"] == cid, "class_name"].values[0]
            print(f"  class {cid:>2} ({name:<16}): {count:>4} features")

        return labeled_fc