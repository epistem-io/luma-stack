"""
threshold_optimiser.py
======================
Optimises the thresholds and priority order in a hierarchical scheme CSV.

The user provides semantic bounds per NODE (not per primitive), so the
same primitive used at different tree levels can have different allowed
ranges. For example, the treecover node that splits dense vs open forest
might allow [15, 60], while the treecover node that splits grass vs cleared
might allow [5, 20].

What the optimiser changes
--------------------------
- threshold at each split node  (within the user-supplied semantic bounds)
- priority order of root-level nodes

What the optimiser never changes
---------------------------------
- which primitive is tested at each node
- the tree structure (parent_id relationships)
- which class_id is assigned at each leaf
- the operator direction (> or <=)
- any leaf node

Scoring approach
----------------
For each split node the optimiser generates candidate thresholds at
percentiles of the primitive distribution that fall within the semantic
bounds, then scores each candidate on three independent criteria:

  Reachability  — does the threshold fall inside the actual data range?
                  A threshold outside the range can never split any pixel.
                  Hard gate: candidates outside the range are discarded.

  Balance       — does the threshold split pixels into two non-trivial
                  groups? Score = 1 at a 50/50 split, falls to 0 when
                  either side gets fewer than min_split_frac of pixels.
                  This prevents a class from being assigned to 0% of the map.

  Alignment     — given training points with known class labels, does the
                  split correctly separate the target class pixels from the
                  rest? Measured as F1 between predicted side and true label.
                  Requires training_gdf and aoi_bounds to be provided.

Final score = balance_weight × balance + alignment_weight × alignment
(reachability is a hard filter, not a weighted term)

Priority optimisation
---------------------
Root-level nodes are re-ordered by how selective their condition is at
the optimised threshold — a node that fires for only 5% of pixels (e.g.
water_pres > 0.7) is more decisive than one that fires for 40% of pixels,
and should be resolved first. This prevents a permissive catch-all node
from consuming most of the AOI before specific classes get evaluated.

Usage
-----
    from threshold_optimiser import ThresholdOptimiser, describe_primitives

    # Step 1 — inspect distributions to set informed semantic bounds
    describe_primitives(band_arrays)

    # Step 2 — define per-node semantic bounds
    # Keys are node_ids from your scheme CSV.
    # The optimiser will only search thresholds within [min_thresh, max_thresh].
    # Nodes not listed here use the full observed data range as bounds.
    node_bounds = {
        1:  {"min_thresh": 0.05, "max_thresh": 0.60,
             "note": "tree_pres RF prob — noise floor 0.05"},
        3:  {"min_thresh": 15.0, "max_thresh": 60.0,
             "note": "treecover — dense vs open forest split"},
        6:  {"min_thresh": 0.05, "max_thresh": 0.50,
             "note": "tree_pres — no-tree branch threshold"},
        8:  {"min_thresh": 5.0,  "max_thresh": 20.0,
             "note": "treeheight — very low structure split"},
        11: {"min_thresh": 5.0,  "max_thresh": 20.0,
             "note": "treecover — grass vs cleared split"},
    }

    opt = ThresholdOptimiser(
        scheme_csv    = "scheme_hierarchical.csv",
        band_arrays   = band_arrays,
        node_bounds   = node_bounds,
        training_gdf  = data["gdf"],
        class_col     = "LULC_Type",
        aoi_bounds    = (west, south, east, north),
    )

    optimised_df = opt.run()
    optimised_df.to_csv("scheme_optimised.csv", index=False)
"""

import logging
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


# ---------------------------------------------------------------------------
# Internal scoring functions
# ---------------------------------------------------------------------------

def _score_balance(
    threshold: float,
    arr: np.ndarray,
    operator: str,
    min_split_frac: float = 0.05,
) -> float:
    """
    How evenly does this threshold split the pixel distribution?

    Returns 1.0 for a 50/50 split, falling linearly to 0.0 as the split
    becomes more lopsided. Returns 0.0 if either side gets fewer than
    min_split_frac of all pixels — this hard floor prevents degenerate
    splits that assign a class to near-zero area on the final map.

    Parameters
    ----------
    threshold : float
    arr : np.ndarray
        Flattened finite pixel values for this primitive.
    operator : str
        The node operator ('>' or '<=' etc.) — determines which side is
        the "true" branch.
    min_split_frac : float
        Minimum acceptable fraction for either side. Default 0.05 (5%).
    """
    frac_pos = float((arr > threshold).mean()) if operator in (">", ">=") \
               else float((arr <= threshold).mean())
    frac_neg = 1.0 - frac_pos

    if frac_pos < min_split_frac or frac_neg < min_split_frac:
        return 0.0

    return 1.0 - abs(frac_pos - 0.5) * 2.0


def _score_alignment(
    threshold: float,
    operator: str,
    target_class: Optional[str],
    training_df: Optional[pd.DataFrame],
) -> float:
    """
    How well does this threshold separate training points of the target
    class from all other training points?

    The target class is the class that should fall on the POSITIVE side
    of the operator (i.e. tree_pres > 0.15 → the positive side should
    contain tree-class training points).

    Returns an F1-like score (harmonic mean of precision and recall).
    Returns 0.5 (neutral) when no training data or target class is given.

    Parameters
    ----------
    threshold : float
    operator : str
    target_class : str or None
        Class name that should be on the positive side of the split.
        Inferred from the scheme tree (first leaf on the positive branch).
    training_df : pd.DataFrame or None
        Rows: one per training point sampled at this primitive.
        Columns: 'value' (float), 'class_name' (str).
    """
    if training_df is None or training_df.empty or target_class is None:
        return 0.5

    target    = training_df["class_name"] == target_class
    predicted = training_df["value"] > threshold if operator in (">", ">=") \
                else training_df["value"] <= threshold

    tp = int((predicted & target).sum())
    fp = int((predicted & ~target).sum())
    fn = int((~predicted & target).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Training point value extraction
# ---------------------------------------------------------------------------

def _extract_training_values(
    band_arrays: dict,
    training_gdf: gpd.GeoDataFrame,
    primitive: str,
    aoi_bounds: tuple,
    class_col: str = "class_name",
) -> Optional[pd.DataFrame]:
    """
    Sample the primitive raster at each training point location and return
    a DataFrame with the sampled value and the class label.

    Uses coordinate-to-pixel mapping from the AOI geographic bounds.
    Points that fall outside the raster extent are silently dropped.

    Parameters
    ----------
    band_arrays : dict[str, np.ndarray]
        Full-AOI primitive arrays.
    training_gdf : gpd.GeoDataFrame
        Training points in EPSG:4326 with a class label column.
    primitive : str
        Which band to sample.
    aoi_bounds : tuple
        (west, south, east, north).
    class_col : str
        Column name for the class label in training_gdf.

    Returns
    -------
    pd.DataFrame with columns ['value', 'class_name'], or None.
    """
    if primitive not in band_arrays:
        return None
    if training_gdf is None or "geometry" not in training_gdf.columns:
        return None

    arr              = band_arrays[primitive]
    H, W             = arr.shape
    west, south, east, north = aoi_bounds
    lon_range        = east  - west
    lat_range        = north - south

    records = []
    for _, row in training_gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        x, y = geom.x, geom.y
        col  = int((x - west)  / lon_range * W)
        r    = int((north - y) / lat_range * H)
        if 0 <= r < H and 0 <= col < W:
            label = row.get(class_col, row.get("LULC_Type", "unknown"))
            records.append({"value": float(arr[r, col]), "class_name": str(label)})

    return pd.DataFrame(records) if records else None


# ---------------------------------------------------------------------------
# Class hint inference
# ---------------------------------------------------------------------------

def _infer_target_class(
    node_id: int,
    operator: str,
    scheme_df: pd.DataFrame,
) -> Optional[str]:
    """
    Walk the scheme tree to find the class name that sits on the positive
    branch of this node.

    The positive branch is the one where the operator condition is TRUE
    (e.g. for operator '>', the positive branch is the subtree where the
    value exceeds the threshold).

    Traversal: find children of node_id whose own operator matches the
    positive direction, then follow to the first leaf.

    Returns None if no leaf is reachable on the positive branch.
    """
    pos_ops = {">", ">="}
    neg_ops = {"<", "<="}
    branch_ops = pos_ops if operator in pos_ops else neg_ops

    children = scheme_df[
        scheme_df["parent_id"].apply(
            lambda x: not pd.isna(x) and int(x) == node_id
        )
    ]

    # Children whose operator is on the same side as the parent's positive branch
    # In the CSV convention: first child by priority on the positive branch
    children_sorted = children.sort_values("priority")

    for _, child in children_sorted.iterrows():
        if child["node_type"] == "leaf":
            name = child.get("class_name", "")
            return str(name) if name and not pd.isna(name) else None
        elif child["node_type"] == "split":
            # Recurse into split child to find its first leaf
            result = _infer_target_class(int(child["node_id"]), child["operator"], scheme_df)
            if result:
                return result

    return None


# ---------------------------------------------------------------------------
# Main optimiser class
# ---------------------------------------------------------------------------

class ThresholdOptimiser:
    """
    Optimises the threshold at each split node and the priority order of
    root-level nodes in a hierarchical scheme CSV.

    The optimiser takes per-node semantic bounds as its primary input.
    This means the same primitive (e.g. treecover) used at two different
    levels of the tree can have different allowed ranges at each level,
    preserving the ecological meaning of each split independently.

    Parameters
    ----------
    scheme_csv : str
        Path to the hierarchical scheme CSV.
    band_arrays : dict[str, np.ndarray]
        Primitive band arrays (H×W float32) from the classifier download.
    node_bounds : dict[int, dict], optional
        Per-node semantic bounds. Keys are node_id integers.
        Each value is a dict with:
            'min_thresh' : float  — minimum allowed threshold
            'max_thresh' : float  — maximum allowed threshold
            'note'       : str    — documentation only, not used in scoring
        Nodes not listed use the full observed data range as bounds.
        Example::

            node_bounds = {
                1: {"min_thresh": 0.05, "max_thresh": 0.60,
                    "note": "tree RF prob noise floor"},
                3: {"min_thresh": 15.0, "max_thresh": 60.0,
                    "note": "treecover dense/open split"},
            }

    training_gdf : gpd.GeoDataFrame, optional
        Training points with geometry (EPSG:4326) and a class label column.
        When provided, alignment scoring is enabled for all split nodes.
    class_col : str
        Column name in training_gdf for class labels. Default 'LULC_Type'.
    aoi_bounds : tuple of float, optional
        (west, south, east, north) in EPSG:4326 decimal degrees.
        Required for training point alignment. Obtain from::

            bounds = aoi.geometry().bounds().getInfo()["coordinates"][0]
            xs = [p[0] for p in bounds]; ys = [p[1] for p in bounds]
            aoi_bounds = (min(xs), min(ys), max(xs), max(ys))

    n_candidates : int
        Number of candidate threshold values to evaluate per node.
        Generated as evenly spaced percentiles within the semantic bounds.
        Default 19 (p5, p10, ..., p95).
    balance_weight : float
        Weight for the balance term in the combined score. Default 0.5.
    alignment_weight : float
        Weight for the alignment term. Default 0.5.
        The two weights must sum to 1.0.
    min_split_frac : float
        Minimum fraction of pixels required on each side of a split.
        Candidates that produce a more lopsided split are scored 0 for
        balance. Default 0.05 (5%).

    Example
    -------
    >>> from threshold_optimiser import ThresholdOptimiser, describe_primitives
    >>>
    >>> # Inspect distributions first
    >>> describe_primitives(band_arrays)
    >>>
    >>> node_bounds = {
    ...     1:  {"min_thresh": 0.05, "max_thresh": 0.50,
    ...          "note": "tree_pres — genuine tree signal"},
    ...     3:  {"min_thresh": 15.0, "max_thresh": 60.0,
    ...          "note": "treecover — dense vs open forest"},
    ...     5:  {"min_thresh":  8.0, "max_thresh": 20.0,
    ...          "note": "treeheight — tall vs short trees"},
    ... }
    >>>
    >>> opt = ThresholdOptimiser(
    ...     scheme_csv   = "scheme_hierarchical.csv",
    ...     band_arrays  = band_arrays,
    ...     node_bounds  = node_bounds,
    ...     training_gdf = data["gdf"],
    ...     class_col    = "LULC_Type",
    ...     aoi_bounds   = (103.7, -2.3, 104.0, -1.8),
    ... )
    >>> optimised_df = opt.run()
    >>> optimised_df.to_csv("scheme_optimised.csv", index=False)
    """

    def __init__(
        self,
        scheme_csv:       str,
        band_arrays:      dict,
        node_bounds:      Optional[dict] = None,
        training_gdf:     Optional[gpd.GeoDataFrame] = None,
        class_col:        str   = "LULC_Type",
        aoi_bounds:       Optional[tuple] = None,
        n_candidates:     int   = 19,
        balance_weight:   float = 0.5,
        alignment_weight: float = 0.5,
        min_split_frac:   float = 0.05,
    ):
        if abs(balance_weight + alignment_weight - 1.0) > 1e-6:
            raise ValueError(
                f"balance_weight ({balance_weight}) + alignment_weight "
                f"({alignment_weight}) must sum to 1.0."
            )

        self.scheme_df        = pd.read_csv(scheme_csv)
        self.band_arrays      = band_arrays
        self.node_bounds      = node_bounds or {}
        self.training_gdf     = training_gdf
        self.class_col        = class_col
        self.aoi_bounds       = aoi_bounds
        self.n_candidates     = n_candidates
        self.balance_weight   = balance_weight
        self.alignment_weight = alignment_weight
        self.min_split_frac   = min_split_frac

        self.logger = logging.getLogger(self.__class__.__name__)

        # Pre-extract training values per primitive to avoid repeated GDF scans
        self._training_cache: dict = {}
        if training_gdf is not None and aoi_bounds is not None:
            renamed = training_gdf.rename(columns={class_col: "class_name"}) \
                      if class_col != "class_name" else training_gdf
            for prim in band_arrays:
                tv = _extract_training_values(
                    band_arrays, renamed, prim, aoi_bounds
                )
                if tv is not None and not tv.empty:
                    self._training_cache[prim] = tv

        n_with_training = len(self._training_cache)
        self.logger.info(
            f"ThresholdOptimiser ready: "
            f"{len(self.scheme_df)} nodes, "
            f"{len(band_arrays)} primitives, "
            f"{len(self.node_bounds)} node-level bounds, "
            f"{n_with_training} primitives with training data"
        )

    # ---------------------------------
    # Candidate threshold generation
    # ---------------------------------

    def _get_candidates(
        self, node_id: int, primitive: str, operator: str
    ) -> np.ndarray:
        """
        Generate candidate threshold values for a specific node.

        Uses the per-node semantic bounds if provided, otherwise falls
        back to the full observed range of the primitive.
        """
        arr = self.band_arrays.get(primitive)
        if arr is None:
            return np.array([])

        flat = arr.ravel()
        flat = flat[np.isfinite(flat)]
        if len(flat) == 0:
            return np.array([])

        obs_min = float(flat.min())
        obs_max = float(flat.max())

        bounds  = self.node_bounds.get(node_id, {})
        lo      = float(bounds.get("min_thresh", obs_min))
        hi      = float(bounds.get("max_thresh", obs_max))

        # Clamp semantic bounds to observed range — a bound outside the
        # observed range is meaningless and would produce no candidates
        lo = max(lo, obs_min)
        hi = min(hi, obs_max)

        if lo >= hi:
            self.logger.warning(
                f"  Node {node_id} ({primitive}): semantic bounds "
                f"[{lo:.4f}, {hi:.4f}] collapse to a single point after "
                f"clamping to observed range [{obs_min:.4f}, {obs_max:.4f}]. "
                f"Using full observed range instead."
            )
            lo, hi = obs_min, obs_max

        # Percentile-based candidates within [lo, hi]
        pct_vals = np.percentile(flat, np.linspace(5, 95, self.n_candidates))
        candidates = pct_vals[(pct_vals >= lo) & (pct_vals <= hi)]

        if len(candidates) == 0:
            self.logger.warning(
                f"  Node {node_id}: no percentile candidates fall within "
                f"[{lo:.4f}, {hi:.4f}]. Expanding to full range."
            )
            candidates = pct_vals

        return np.unique(candidates)

    # ---------------------------------
    # Single-node optimisation
    # ---------------------------------

    def _optimise_node(self, node_row: pd.Series) -> tuple:
        """
        Find the best threshold for one split node.

        Returns (optimised_threshold, score_record_dict).
        Falls back to the original threshold if no candidates score > 0.
        """
        node_id    = int(node_row["node_id"])
        primitive  = str(node_row["primitive"])
        operator   = str(node_row["operator"])
        orig_thresh= float(node_row["threshold"])

        candidates = self._get_candidates(node_id, primitive, operator)
        if len(candidates) == 0:
            self.logger.warning(f"  Node {node_id}: no candidates, keeping original.")
            return orig_thresh, {}

        arr = self.band_arrays.get(primitive)
        if arr is None:
            return orig_thresh, {}

        flat = arr.ravel()
        flat = flat[np.isfinite(flat)]

        training_df  = self._training_cache.get(primitive)
        target_class = _infer_target_class(node_id, operator, self.scheme_df)

        best_thresh  = orig_thresh
        best_score   = -1.0
        score_records = []

        for thresh in candidates:
            # Hard gate: threshold must be reachable in the data
            if not (flat.min() <= thresh <= flat.max()):
                continue

            b = _score_balance(thresh, flat, operator, self.min_split_frac)
            a = _score_alignment(thresh, operator, target_class, training_df)

            score = self.balance_weight * b + self.alignment_weight * a
            score_records.append({
                "threshold": thresh,
                "balance":   round(b, 4),
                "alignment": round(a, 4),
                "score":     round(score, 4),
            })

            if score > best_score:
                best_score  = score
                best_thresh = thresh

        bounds_used = self.node_bounds.get(node_id, {})
        lo_str = f"{bounds_used.get('min_thresh', 'obs')}"
        hi_str = f"{bounds_used.get('max_thresh', 'obs')}"

        self.logger.info(
            f"  Node {node_id:>3} [{primitive:<22} {operator}]  "
            f"bounds=[{lo_str}, {hi_str}]  "
            f"target='{target_class or '?'}'  "
            f"{orig_thresh:.4f} → {best_thresh:.4f}  "
            f"(score={best_score:.3f}, "
            f"B={[r['balance'] for r in score_records if r['threshold']==best_thresh][0] if score_records else '?':.3f}, "
            f"A={[r['alignment'] for r in score_records if r['threshold']==best_thresh][0] if score_records else '?':.3f})"
        )

        return float(best_thresh), {
            "node_id":       node_id,
            "primitive":     primitive,
            "operator":      operator,
            "target_class":  target_class,
            "orig_threshold":orig_thresh,
            "opt_threshold": best_thresh,
            "score":         best_score,
            "candidates":    score_records,
        }

    # ---------------------------------
    # Priority optimisation
    # ---------------------------------

    def _optimise_priority(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Re-order root-level nodes (parent_id is NaN) by selectivity.

        A node is more selective if its condition fires for a small
        fraction of all pixels — it makes a strong, specific claim.
        Selective nodes should be evaluated first so they claim their
        pixels before permissive nodes get a chance.

        Selectivity = distance from 50% firing rate.
        A node firing for 5% is very selective (score = 0.45).
        A node firing for 45% is barely selective (score = 0.05).
        """
        root_mask  = df["parent_id"].isna()
        root_nodes = df[root_mask].copy()
        other_nodes= df[~root_mask].copy()

        selectivity = []
        for _, row in root_nodes.iterrows():
            prim = row["primitive"]
            op   = row["operator"]
            th   = float(row["threshold"])
            arr  = self.band_arrays.get(prim)

            if arr is None:
                selectivity.append(0.0)
                continue

            flat = arr.ravel()
            flat = flat[np.isfinite(flat)]

            frac = float((flat > th).mean()) if op in (">", ">=") \
                   else float((flat <= th).mean())

            # selectivity: closer to 0 or 1 = more decisive = higher score
            selectivity.append(abs(frac - 0.5))

        root_nodes["_sel"] = selectivity
        # Most selective → lowest priority number (fires first)
        root_nodes = root_nodes.sort_values("_sel", ascending=False)
        root_nodes["priority"] = range(1, len(root_nodes) + 1)
        root_nodes = root_nodes.drop(columns=["_sel"])

        self.logger.info("Root node priority after optimisation:")
        for _, row in root_nodes.iterrows():
            prim = row["primitive"]
            arr  = self.band_arrays.get(prim)
            frac_str = ""
            if arr is not None:
                flat = arr.ravel()
                flat = flat[np.isfinite(flat)]
                op   = row["operator"]
                th   = float(row["threshold"])
                frac = float((flat > th).mean()) if op in (">", ">=") \
                       else float((flat <= th).mean())
                frac_str = f"  fires for {frac*100:.1f}% of pixels"
            self.logger.info(
                f"  [{int(row['priority'])}] {prim} {row['operator']} "
                f"{float(row['threshold']):.4f}{frac_str}"
            )

        return pd.concat([root_nodes, other_nodes]).sort_index()

    # ---------------------------------
    # Main run
    # ---------------------------------

    def run(self) -> pd.DataFrame:
        """
        Run the full optimisation and return the updated scheme DataFrame.

        Step 1: optimise thresholds for all split nodes.
        Step 2: re-order root-level node priorities by selectivity.
        Leaf nodes are never modified.

        Returns
        -------
        pd.DataFrame
            Updated scheme table. Save to CSV and pass to
            load_hierarchical_scheme() to use in classification.

        Example
        -------
        >>> df = opt.run()
        >>> df.to_csv("scheme_optimised.csv", index=False)
        """
        df = self.scheme_df.copy()
        self._score_log = []

        # --- Step 1: threshold optimisation ----------------------------------
        self.logger.info("=" * 55)
        self.logger.info("Step 1: Threshold optimisation")
        self.logger.info("=" * 55)

        split_nodes = df[df["node_type"] == "split"]
        for idx, row in split_nodes.iterrows():
            new_thresh, record = self._optimise_node(row)
            df.at[idx, "threshold"] = round(new_thresh, 4)
            if record:
                self._score_log.append(record)

        # --- Step 2: priority optimisation -----------------------------------
        self.logger.info("=" * 55)
        self.logger.info("Step 2: Root node priority optimisation")
        self.logger.info("=" * 55)

        df = self._optimise_priority(df)

        self.logger.info("Optimisation complete.")
        self._print_summary(df)
        return df

    # ---------------------------------
    # Diagnostic output
    # ---------------------------------

    def _print_summary(self, df: pd.DataFrame) -> None:
        """Print a structured before/after summary to stdout."""
        orig = self.scheme_df
        W    = 72

        print("\n" + "=" * W)
        print("  THRESHOLD OPTIMISATION SUMMARY")
        print("=" * W)
        print(f"  {'node':>5}  {'primitive':<22} {'op':>3}  "
              f"{'original':>9}  {'optimised':>9}  {'Δ':>8}  "
              f"{'score':>6}  target class")
        print("-" * W)

        for record in self._score_log:
            nid    = record["node_id"]
            delta  = record["opt_threshold"] - record["orig_threshold"]
            sign   = "+" if delta >= 0 else ""
            tclass = record["target_class"] or "—"
            print(
                f"  {nid:>5}  {record['primitive']:<22} "
                f"{record['operator']:>3}  "
                f"{record['orig_threshold']:>9.4f}  "
                f"{record['opt_threshold']:>9.4f}  "
                f"{sign}{delta:>7.4f}  "
                f"{record['score']:>6.3f}  {tclass}"
            )

        print()
        print("  Root node priority order (after optimisation):")
        roots = df[df["parent_id"].isna()].sort_values("priority")
        for _, row in roots.iterrows():
            ntype = row["node_type"]
            label = (row.get("class_name", "") or "") if ntype == "leaf" else "(split)"
            arr   = self.band_arrays.get(row["primitive"])
            fire_str = ""
            if arr is not None:
                flat = arr.ravel(); flat = flat[np.isfinite(flat)]
                op = row["operator"]; th = float(row["threshold"])
                frac = float((flat > th).mean()) if op in (">", ">=") \
                       else float((flat <= th).mean())
                fire_str = f"  [{frac*100:.1f}% of pixels]"
            print(
                f"    [{int(row['priority'])}] {row['primitive']:<22} "
                f"{row['operator']} {float(row['threshold']):.4f}  "
                f"{ntype}  {label}{fire_str}"
            )

        print("=" * W + "\n")

    def score_detail(self) -> pd.DataFrame:
        """
        Return a DataFrame with per-candidate scores for every split node.
        Useful for inspecting why a particular threshold was chosen.

        Returns
        -------
        pd.DataFrame
            Columns: node_id, primitive, operator, target_class,
                     threshold, balance, alignment, score.

        Example
        -------
        >>> detail = opt.score_detail()
        >>> print(detail[detail.node_id == 3].to_string())
        """
        if not hasattr(self, "_score_log"):
            raise RuntimeError("Call .run() before .score_detail().")

        rows = []
        for record in self._score_log:
            for cand in record.get("candidates", []):
                rows.append({
                    "node_id":      record["node_id"],
                    "primitive":    record["primitive"],
                    "operator":     record["operator"],
                    "target_class": record["target_class"],
                    **cand,
                })
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Standalone diagnostic: inspect primitive distributions
# ---------------------------------------------------------------------------

def describe_primitives(
    band_arrays: dict,
    node_bounds: Optional[dict] = None,
    scheme_df:   Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Print a summary of each primitive's value distribution and flag
    whether any per-node semantic bounds fall outside the observed range.

    Run this BEFORE defining node_bounds to understand what threshold
    values are meaningful for each primitive in your AOI.

    **Note:** Masked/NaN pixels are automatically excluded from statistics,
    matching GEE's reduceRegion behavior (which also excludes invalid pixels).

    Parameters
    ----------
    band_arrays : dict[str, np.ndarray]
        Primitive band arrays from _download_band_arrays(). Any NaN values
        (representing masked/nodata pixels) are automatically filtered out.
    node_bounds : dict[int, dict], optional
        Per-node semantic bounds (same format as ThresholdOptimiser).
        If provided alongside scheme_df, each node's bounds are checked
        against the observed range of its primitive.
    scheme_df : pd.DataFrame, optional
        Loaded scheme (from load_hierarchical_scheme). Used to map
        node_ids back to primitive names when checking bounds.

    Returns
    -------
    pd.DataFrame
        One row per primitive with percentile statistics (computed on
        valid, non-masked pixels only).

    Example
    -------
    >>> describe_primitives(band_arrays)
    >>> describe_primitives(band_arrays, node_bounds=my_bounds, scheme_df=tree)
    """
    rows = []
    for name, arr in band_arrays.items():
        flat = arr.ravel()
        flat = flat[np.isfinite(flat)]
        if len(flat) == 0:
            continue
        pcts = np.percentile(flat, [5, 10, 25, 50, 75, 90, 95])
        rows.append({
            "primitive": name,
            "min":  round(float(flat.min()), 4),
            "p05":  round(float(pcts[0]), 4),
            "p10":  round(float(pcts[1]), 4),
            "p25":  round(float(pcts[2]), 4),
            "p50":  round(float(pcts[3]), 4),
            "p75":  round(float(pcts[4]), 4),
            "p90":  round(float(pcts[5]), 4),
            "p95":  round(float(pcts[6]), 4),
            "max":  round(float(flat.max()), 4),
        })

    df = pd.DataFrame(rows).set_index("primitive")
    print("\nPrimitive distributions:")
    print(df.to_string())

    # Cross-check node_bounds against observed ranges
    if node_bounds and scheme_df is not None:
        print("\nNode bounds vs observed range:")
        print(f"  {'node':>5}  {'primitive':<22}  "
              f"{'sem_min':>8}  {'sem_max':>8}  "
              f"{'obs_min':>8}  {'obs_max':>8}  status")
        print("  " + "-" * 68)

        for node_id, bounds in node_bounds.items():
            node_row = scheme_df[scheme_df["node_id"] == node_id]
            if node_row.empty:
                continue
            prim = node_row.iloc[0]["primitive"]
            if prim not in df.index:
                continue

            lo      = bounds.get("min_thresh", "—")
            hi      = bounds.get("max_thresh", "—")
            obs_min = df.loc[prim, "min"]
            obs_max = df.loc[prim, "max"]

            issues = []
            if isinstance(lo, (int, float)) and lo < obs_min:
                issues.append(f"sem_min < obs_min")
            if isinstance(hi, (int, float)) and hi > obs_max:
                issues.append(f"sem_max > obs_max")
            if isinstance(lo, (int, float)) and isinstance(hi, (int, float)) \
               and lo > obs_max:
                issues.append("UNREACHABLE: sem_min > obs_max")

            status = ", ".join(issues) if issues else "ok"
            print(
                f"  {node_id:>5}  {prim:<22}  "
                f"{str(lo):>8}  {str(hi):>8}  "
                f"{obs_min:>8.4f}  {obs_max:>8.4f}  {status}"
            )

    return df