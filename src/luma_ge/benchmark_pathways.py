"""
benchmark_pathways.py
=====================
Benchmarking framework for comparing the two LULC classification pathways:

    Pathway A — Primitive Layer + Monte Carlo (MC)
        GEE:    Train one RF per element (PROBABILITY mode)
        GEE:    Download primitive stack via getDownloadURL
        Python: Run N Monte Carlo iterations locally
        Output: mode_map + entropy_map + class_probs

    Pathway B — Direct Training Data Assembly
        GEE:    Label training features by scheme rules (TrainingDataLabeller)
        GEE:    Train one RF per scheme (CLASSIFICATION mode) on labeled features
        GEE:    Classify image directly on GEE server
        Output: class_id map (no uncertainty)

Benchmarked dimensions
----------------------
    Wall-clock time     — total elapsed seconds per stage and overall
    GEE EECU            — Earth Engine Compute Unit consumption (via getInfo
                          on ee.Number operations; approximated from task
                          profiler when available)
    Peak RAM (Python)   — maximum resident set size during Python-side stages
    Output quality      — pixel agreement between the two pathways, entropy
                          of the MC result, and per-class area share

Each stage is timed and measured independently so you can pinpoint exactly
where the computational cost lives across both pathways.

Usage
-----
    from benchmark_pathways import PathwayBenchmark

    bench = PathwayBenchmark(
        image       = stacked_landsat,
        training_fc = data["ee_fc"],
        rules_df    = scheme1_rules,
        scheme_name = "scheme1",
        aoi         = aoi,
        scale       = 30,
        n_iterations= 300,   # MC only
    )

    report = bench.run_all()
    bench.print_report(report)
    bench.plot_report(report)
"""

import gc
import logging
import os
import time
import tracemalloc
from dataclasses import dataclass, field
from typing import Optional

import ee
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import luma_ge

# Import both pathways from the main module
from luma_ge.modular_workflow import (
    PrimitiveLayerTrainer,
    RuleSetClassifier,
    TrainingDataLabeller,
    _ee_image_to_numpy,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


# ===========================================================================
# Data structures
# ===========================================================================

@dataclass
class StageResult:
    """
    Records the cost of a single pipeline stage.

    Attributes
    ----------
    name        : Human-readable stage label.
    pathway     : "A_mc" or "B_direct".
    where       : "GEE" if the work runs on GEE servers, "Python" if local.
    wall_sec    : Elapsed wall-clock seconds.
    ram_peak_mb : Peak Python RSS in MB (0.0 for pure GEE stages).
    eecu_approx : Approximate EECU consumed. GEE does not expose a live
                  EECU counter via the Python API — this is estimated from
                  task duration × a conservative 0.01 EECU/s coefficient
                  for interactive operations (not batch exports).
                  Treat as a relative comparator, not an absolute figure.
    notes       : Any extra diagnostic string.
    """
    name:         str
    pathway:      str
    where:        str
    wall_sec:     float = 0.0
    ram_peak_mb:  float = 0.0
    eecu_approx:  float = 0.0
    notes:        str   = ""


@dataclass
class PathwayReport:
    """
    Full benchmark report for one run across both pathways.
    """
    scheme_name:    str
    n_primitives:   int
    n_classes:      int
    n_iterations:   int          # MC only
    aoi_pixels:     int
    scale_m:        int
    stages:         list         = field(default_factory=list)

    # Final outputs stored for quality comparison
    mc_mode_map:    Optional[np.ndarray] = None
    mc_entropy_map: Optional[np.ndarray] = None
    direct_map:     Optional[np.ndarray] = None


# ===========================================================================
# Timing and memory context managers
# ===========================================================================

class _Timer:
    """Context manager: records wall-clock elapsed time in seconds."""
    def __init__(self):
        self.elapsed = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed = time.perf_counter() - self._start


class _MemTracker:
    """Context manager: tracks peak Python memory allocation in MB."""
    def __init__(self):
        self.peak_mb = 0.0

    def __enter__(self):
        tracemalloc.start()
        return self

    def __exit__(self, *_):
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.peak_mb = peak / (1024 ** 2)


# ===========================================================================
# EECU estimation helpers
# ===========================================================================

# Conservative coefficient: interactive GEE operations (getInfo, classify,
# sampleRegions) consume roughly 0.005–0.02 EECU per second of server time.
# We use 0.01 as the midpoint for relative comparison purposes.
_EECU_PER_SEC_INTERACTIVE = 0.01


def _estimate_eecu(wall_sec: float, where: str) -> float:
    """Return approximate EECU consumed by a GEE stage."""
    if where != "GEE":
        return 0.0
    return round(wall_sec * _EECU_PER_SEC_INTERACTIVE, 4)


# ===========================================================================
# Main benchmark class
# ===========================================================================

class PathwayBenchmark:
    """
    Runs and measures both classification pathways side by side.

    Parameters
    ----------
    image : ee.Image
        Multi-band predictor image (e.g. stacked Landsat composite).
    training_fc : ee.FeatureCollection
        Reference data with binary element attributes.
    rules_df : pd.DataFrame
        Output of load_scheme() with 'scheme' column added.
    scheme_name : str
        Scheme key in rules_df.
    aoi : ee.FeatureCollection or ee.Geometry
        Area of interest.
    scale : int
        Pixel scale in metres. Default 30.
    n_iterations : int
        Number of Monte Carlo iterations (Pathway A only). Default 300.
    n_trees : int
        Number of RF trees for both pathways. Default 50.

    Example
    -------
    >>> bench  = PathwayBenchmark(image, training_fc, rules, "scheme1", aoi)
    >>> report = bench.run_all()
    >>> bench.print_report(report)
    >>> bench.plot_report(report)
    """

    def __init__(
        self,
        image:         ee.Image,
        training_fc:   ee.FeatureCollection,
        rules_df:      pd.DataFrame,
        scheme_name:   str,
        aoi,
        scale:         int = 30,
        n_iterations:  int = 300,
        n_trees:       int = 50,
    ):
        self.image        = image
        self.training_fc  = training_fc
        self.rules_df     = rules_df
        self.scheme_name  = scheme_name
        self.aoi          = aoi
        self.scale        = scale
        self.n_iterations = n_iterations
        self.n_trees      = n_trees

        self.logger = logging.getLogger(self.__class__.__name__)

        subset              = rules_df[rules_df["scheme"] == scheme_name]
        self._n_classes     = len(subset)
        self._n_primitives  = len([
            c for c in training_fc.first().propertyNames().getInfo()
            if c not in {"LULC_Type", "ID", "geometry", "system:index", "class_id"}
        ])

    # =========================================================================
    # PATHWAY A — Primitive Layer + Monte Carlo
    # =========================================================================

    def _run_pathway_a(self, report: PathwayReport) -> None:
        """
        Benchmark all stages of the primitive layer + Monte Carlo pathway.

        Stages:
            A1  GEE    Train probabilistic RF per element
            A2  GEE    Download primitive stack (getDownloadURL)
            A3  Python Run Monte Carlo iterations
            A4  Python Aggregate mode map + entropy
        """
        self.logger.info("--- Pathway A: Primitive Layer + Monte Carlo ---")

        # -----------------------------------------------------------------
        # A1: Train probabilistic primitive layers (GEE)
        # -----------------------------------------------------------------
        self.logger.info("A1: Training probabilistic primitive layers on GEE...")
        with _Timer() as t:
            trainer = PrimitiveLayerTrainer(
                image=self.image,
                roi=self.training_fc,
                n_trees=self.n_trees,
                scale=self.scale,
            )
            prob_layers = trainer.train_all_mc()
            # Force GEE computation by stacking and calling getInfo on band names
            primitive_stack = ee.Image.cat(list(prob_layers.values()))
            _ = primitive_stack.bandNames().getInfo()  # triggers server computation

        report.stages.append(StageResult(
            name="A1: Train primitive RFs (probabilistic)",
            pathway="A_mc",
            where="GEE",
            wall_sec=round(t.elapsed, 2),
            eecu_approx=_estimate_eecu(t.elapsed, "GEE"),
            notes=f"{self._n_primitives} primitives × {self.n_trees} trees",
        ))

        # -----------------------------------------------------------------
        # A2: Download primitive stack from GEE to Python (getDownloadURL)
        # -----------------------------------------------------------------
        self.logger.info("A2: Downloading primitive stack from GEE...")
        classifier_a = RuleSetClassifier(primitive_stack, self.rules_df, self.aoi)

        with _Timer() as t, _MemTracker() as m:
            band_arrays = classifier_a._download_band_arrays(self.scale)
            classifier_a._validate_probability_range(band_arrays)

        h, w = next(iter(band_arrays.values())).shape
        report.aoi_pixels = h * w

        report.stages.append(StageResult(
            name="A2: Download primitive stack (getDownloadURL)",
            pathway="A_mc",
            where="GEE",
            wall_sec=round(t.elapsed, 2),
            ram_peak_mb=round(m.peak_mb, 2),
            eecu_approx=_estimate_eecu(t.elapsed, "GEE"),
            notes=f"{h}×{w} px, {self._n_primitives} bands",
        ))

        # -----------------------------------------------------------------
        # A3: Monte Carlo sampling iterations (Python)
        # -----------------------------------------------------------------
        self.logger.info(f"A3: Running {self.n_iterations} MC iterations locally...")
        subset = self.rules_df[self.rules_df["scheme"] == self.scheme_name].copy()
        subset = subset.sort_values(["priority", "class_id"]).reset_index(drop=True)

        all_class_ids = sorted(subset["class_id"].unique().tolist())
        if 0.0 not in all_class_ids:
            all_class_ids = [0.0] + all_class_ids
        class_id_to_idx = {cid: i for i, cid in enumerate(all_class_ids)}
        counts = np.zeros((len(all_class_ids), h, w), dtype=np.int32)
        rng    = np.random.default_rng(42)

        with _Timer() as t, _MemTracker() as m:
            for _ in range(self.n_iterations):
                binary_bands = classifier_a._sample_bernoulli(band_arrays, rng, h, w)
                iter_result  = classifier_a._apply_ruleset(
                    subset, binary_bands, h, w, nodata_value=0.0
                )
                for cid, idx in class_id_to_idx.items():
                    counts[idx] += (iter_result == cid).astype(np.int32)

        report.stages.append(StageResult(
            name="A3: Monte Carlo sampling loop",
            pathway="A_mc",
            where="Python",
            wall_sec=round(t.elapsed, 2),
            ram_peak_mb=round(m.peak_mb, 2),
            notes=(
                f"{self.n_iterations} iterations × "
                f"{self._n_primitives} primitives × "
                f"{h * w:,} pixels"
            ),
        ))

        # -----------------------------------------------------------------
        # A4: Aggregate — mode map + entropy (Python)
        # -----------------------------------------------------------------
        self.logger.info("A4: Aggregating MC results...")
        with _Timer() as t, _MemTracker() as m:
            mc_out = classifier_a._aggregate(
                counts, all_class_ids, class_id_to_idx, self.n_iterations
            )

        report.mc_mode_map    = mc_out["mode_map"]
        report.mc_entropy_map = mc_out["entropy_map"]

        report.stages.append(StageResult(
            name="A4: Aggregate mode map + entropy",
            pathway="A_mc",
            where="Python",
            wall_sec=round(t.elapsed, 2),
            ram_peak_mb=round(m.peak_mb, 2),
            notes="mode, Shannon entropy, class probs",
        ))

        gc.collect()

    # =========================================================================
    # PATHWAY B — Direct Training Data Assembly + single RF
    # =========================================================================

    def _run_pathway_b(self, report: PathwayReport) -> None:
        """
        Benchmark all stages of the direct labelling + single RF pathway.

        Stages:
            B1  GEE    Label training features by scheme rules
            B2  GEE    Train single direct RF on labeled features
            B3  GEE    Classify image on GEE server
            B4  GEE    Download result to Python for comparison
        """
        self.logger.info("--- Pathway B: Direct Training Data Assembly ---")

        # -----------------------------------------------------------------
        # B1: Label training features by scheme rules (GEE)
        # -----------------------------------------------------------------
        self.logger.info("B1: Labelling training features by scheme rules on GEE...")
        subset_df = self.rules_df[self.rules_df["scheme"] == self.scheme_name].copy()

        with _Timer() as t:
            labeller   = TrainingDataLabeller(
                rules_df=subset_df,
                scheme_name=self.scheme_name,
                nodata_value=0,
            )
            labeled_fc = labeller.label(self.training_fc)
            # Filter out nodata before training
            clean_fc   = labeled_fc.filter(ee.Filter.neq("class_id", 0))
            # Force GEE computation by checking size
            n_labeled  = clean_fc.size().getInfo()

        report.stages.append(StageResult(
            name="B1: Label training features by rules",
            pathway="B_direct",
            where="GEE",
            wall_sec=round(t.elapsed, 2),
            eecu_approx=_estimate_eecu(t.elapsed, "GEE"),
            notes=f"{n_labeled} features labeled (nodata excluded)",
        ))

        # -----------------------------------------------------------------
        # B2: Train single direct RF on labeled features (GEE)
        # -----------------------------------------------------------------
        self.logger.info("B2: Training direct RF classifier on GEE...")
        with _Timer() as t:
            # Sample image band values at each labeled training point
            # This is the step the benchmark was missing
            sample = self.image.sampleRegions(
                collection=clean_fc,
                properties=["class_id"],
                scale=self.scale,
                geometries=False,
            )

            classifier_b = (
                ee.Classifier.smileRandomForest(self.n_trees)
                .setOutputMode("CLASSIFICATION")
            )
            trained_b = classifier_b.train(
                features=sample,
                classProperty="class_id",
                inputProperties=self.image.bandNames(),
            )
            # Force training computation
            _ = trained_b.explain().getInfo()

        # -----------------------------------------------------------------
        # B3: Classify image on GEE server
        # -----------------------------------------------------------------
        self.logger.info("B3: Classifying image on GEE server...")
        aoi_geom = self.aoi.geometry() if hasattr(self.aoi, "geometry") else self.aoi

        with _Timer() as t:
            direct_map_ee = (
                self.image
                .classify(trained_b)
                .rename("class_id")
                .clip(aoi_geom)
            )
            # Force computation by sampling one pixel
            test_pt = aoi_geom.centroid(maxError=1)
            _ = direct_map_ee.sample(region=test_pt, scale=self.scale,
                                     numPixels=1).first().getInfo()

        report.stages.append(StageResult(
            name="B3: Classify image on GEE server",
            pathway="B_direct",
            where="GEE",
            wall_sec=round(t.elapsed, 2),
            eecu_approx=_estimate_eecu(t.elapsed, "GEE"),
            notes="server-side classification, no local compute",
        ))

        # -----------------------------------------------------------------
        # B4: Download result to Python (for quality comparison)
        # -----------------------------------------------------------------
        self.logger.info("B4: Downloading direct classification result...")
        with _Timer() as t, _MemTracker() as m:
            report.direct_map = _ee_image_to_numpy(
                direct_map_ee, self.aoi, scale=self.scale
            ).astype(int)

        h, w = report.direct_map.shape
        if report.aoi_pixels == 0:
            report.aoi_pixels = h * w

        report.stages.append(StageResult(
            name="B4: Download classification result",
            pathway="B_direct",
            where="GEE",
            wall_sec=round(t.elapsed, 2),
            ram_peak_mb=round(m.peak_mb, 2),
            eecu_approx=_estimate_eecu(t.elapsed, "GEE"),
            notes=f"{h}×{w} px downloaded as GeoTIFF",
        ))

        gc.collect()

    # =========================================================================
    # Quality comparison (runs after both pathways complete)
    # =========================================================================

    def _compute_quality(self, report: PathwayReport) -> dict:
        """
        Compare output quality between pathways.
        Only runs if both mode_map and direct_map are available.
        """
        if report.mc_mode_map is None or report.direct_map is None:
            return {}

        total     = report.mc_mode_map.size
        agreement = (report.mc_mode_map == report.direct_map).sum()
        agree_pct = 100 * agreement / total

        subset     = self.rules_df[self.rules_df["scheme"] == self.scheme_name]
        class_ids  = sorted(subset["class_id"].unique().tolist())
        id_to_name = dict(zip(subset["class_id"], subset["class_name"]))

        per_class = {}
        for cid in class_ids:
            name = id_to_name.get(cid, f"class {cid}")
            per_class[name] = {
                "mc_pct":     round(100 * (report.mc_mode_map == cid).sum() / total, 2),
                "direct_pct": round(100 * (report.direct_map == cid).sum() / total, 2),
            }

        return {
            "pixel_agreement_pct": round(agree_pct, 2),
            "mc_mean_entropy":     round(float(report.mc_entropy_map.mean()), 4),
            "mc_max_entropy":      round(float(report.mc_entropy_map.max()), 4),
            "mc_high_unc_pct":     round(
                100 * (report.mc_entropy_map > 0.5).mean(), 2
            ),
            "per_class_area":      per_class,
        }

    # =========================================================================
    # Run all
    # =========================================================================

    def run_all(self) -> PathwayReport:
        """
        Execute both pathways sequentially and return a full report.

        Returns
        -------
        PathwayReport
            All stage results, quality metrics, and output arrays.

        Example
        -------
        >>> report = bench.run_all()
        """
        report = PathwayReport(
            scheme_name   = self.scheme_name,
            n_primitives  = self._n_primitives,
            n_classes     = self._n_classes,
            n_iterations  = self.n_iterations,
            aoi_pixels    = 0,
            scale_m       = self.scale,
        )

        self._run_pathway_a(report)
        self._run_pathway_b(report)

        report.quality = self._compute_quality(report)
        return report

    # =========================================================================
    # Reporting
    # =========================================================================
 
    def print_report(self, report: PathwayReport) -> None:
        """
        Print a structured benchmark report to stdout.
 
        Parameters
        ----------
        report : PathwayReport
            Return value of run_all().
        """
        W = 68
        div = "=" * W
 
        print(f"\n{div}")
        print(f"  BENCHMARK REPORT — {report.scheme_name}")
        print(div)
        print(f"  Primitives : {report.n_primitives}")
        print(f"  Classes    : {report.n_classes}")
        print(f"  Pixels     : {report.aoi_pixels:,}  ({report.scale_m}m resolution)")
        print(f"  MC iters   : {report.n_iterations}")
        print()
 
        # ── Stage table ──────────────────────────────────────────────────────
        header = (
            f"  {'Stage':<42} {'Where':>6} {'Time(s)':>8} "
            f"{'RAM(MB)':>8} {'EECU':>7}"
        )
        print(header)
        print(f"  {'-'*64}")
 
        pathway_totals = {"A_mc": 0.0, "B_direct": 0.0}
        for s in report.stages:
            pathway_totals[s.pathway] += s.wall_sec
            label = f"[{s.pathway}] {s.name}"
            print(
                f"  {label:<42} {s.where:>6} {s.wall_sec:>8.1f} "
                f"{s.ram_peak_mb:>8.1f} {s.eecu_approx:>7.4f}"
            )
 
        print(f"  {'-'*64}")
 
        # Pathway totals
        total_a = pathway_totals["A_mc"]
        total_b = pathway_totals["B_direct"]
        eecu_a  = sum(s.eecu_approx for s in report.stages if s.pathway == "A_mc")
        eecu_b  = sum(s.eecu_approx for s in report.stages if s.pathway == "B_direct")
        ram_a   = max((s.ram_peak_mb for s in report.stages if s.pathway == "A_mc"),
                      default=0)
        ram_b   = max((s.ram_peak_mb for s in report.stages if s.pathway == "B_direct"),
                      default=0)
 
        print(
            f"  {'TOTAL — Pathway A (MC)':<42} {'':>6} {total_a:>8.1f} "
            f"{ram_a:>8.1f} {eecu_a:>7.4f}"
        )
        print(
            f"  {'TOTAL — Pathway B (Direct)':<42} {'':>6} {total_b:>8.1f} "
            f"{ram_b:>8.1f} {eecu_b:>7.4f}"
        )
 
        has_a = total_a > 0
        has_b = total_b > 0
        if has_a and has_b:
            ratio = total_a / total_b
            direction = "slower" if ratio > 1 else "faster"
            print(f"\n  Pathway A is {ratio:.1f}× {direction} than Pathway B "
                  f"in total wall-clock time.")
        elif has_a and not has_b:
            print(f"\n  Only Pathway A was run — no Pathway B total to compare.")
        elif has_b and not has_a:
            print(f"\n  Only Pathway B was run — no Pathway A total to compare.")
 
        # ── Quality comparison ───────────────────────────────────────────────
        q = getattr(report, "quality", {})
        if q:
            print(f"\n{'-'*W}")
            print("  OUTPUT QUALITY COMPARISON")
            print(f"{'-'*W}")
            print(f"  Pixel agreement (A vs B) : {q['pixel_agreement_pct']:.1f}%")
            print(f"  MC mean entropy          : {q['mc_mean_entropy']:.4f} nats")
            print(f"  MC high-uncertainty px   : {q['mc_high_unc_pct']:.1f}%  "
                  f"(entropy > 0.5)")
            print()
            print(f"  {'Class':<22} {'MC area':>10} {'Direct area':>12}")
            print(f"  {'-'*46}")
            for name, vals in q["per_class_area"].items():
                print(
                    f"  {name:<22} {vals['mc_pct']:>9.1f}%  "
                    f"{vals['direct_pct']:>10.1f}%"
                )
 
        print(f"\n{div}\n")
 
    def plot_report(self, report: PathwayReport) -> None:
        """
        Produce two figures:
            1. Grouped bar chart: wall-clock time per stage, coloured by pathway
            2. Side-by-side maps: MC mode map | Direct map | Entropy map
 
        Parameters
        ----------
        report : PathwayReport
            Return value of run_all().
        """
        stages_a = [s for s in report.stages if s.pathway == "A_mc"]
        stages_b = [s for s in report.stages if s.pathway == "B_direct"]
 
        # ── Figure 1: Stage timing comparison ───────────────────────────────
        fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
        fig.suptitle(
            f"Benchmark — {report.scheme_name}  "
            f"({report.aoi_pixels:,} px, {report.n_iterations} MC iters)",
            fontsize=12, fontweight="500"
        )
 
        # Panel 1a: Wall-clock time per stage
        ax = axes[0]
        labels_a = [s.name.split(":")[0] for s in stages_a]
        labels_b = [s.name.split(":")[0] for s in stages_b]
        times_a  = [s.wall_sec for s in stages_a]
        times_b  = [s.wall_sec for s in stages_b]
 
        x_a = np.arange(len(stages_a))
        x_b = np.arange(len(stages_b))
        ax.barh(labels_a, times_a, color="#1D9E75", alpha=0.85,
                label="Pathway A (MC)", height=0.4)
        ax.barh(
            [f"  {l}" for l in labels_b], times_b,
            color="#378ADD", alpha=0.85, label="Pathway B (Direct)", height=0.4
        )
        for i, (t, l) in enumerate(zip(times_a + times_b, labels_a + labels_b)):
            ax.text(t + 0.3, i, f"{t:.1f}s", va="center", fontsize=8)
        ax.set_xlabel("Wall-clock time (seconds)")
        ax.set_title("Time per stage")
        ax.legend(fontsize=8)
 
        # Panel 1b: Compute location breakdown (GEE vs Python)
        ax2 = axes[1]
        x_pos = 0
        x_ticks, x_labels = [], []
        for pathway, colour, label in [
            ("A_mc",     "#1D9E75", "Pathway A"),
            ("B_direct", "#378ADD", "Pathway B"),
        ]:
            s_list = [s for s in report.stages if s.pathway == pathway]
            if not s_list:
                x_pos += 2
                continue
            gee_t = sum(s.wall_sec for s in s_list if s.where == "GEE")
            py_t  = sum(s.wall_sec for s in s_list if s.where == "Python")
 
            # Two separate bar() calls — alpha must be a scalar
            ax2.bar(x_pos,     gee_t, color=colour, alpha=0.85, width=0.5)
            ax2.bar(x_pos + 1, py_t,  color=colour, alpha=0.45, width=0.5)
            ax2.text(x_pos,     gee_t + 0.1, f"{gee_t:.1f}s",
                     ha="center", fontsize=8)
            ax2.text(x_pos + 1, py_t  + 0.1, f"{py_t:.1f}s",
                     ha="center", fontsize=8)
 
            x_ticks += [x_pos, x_pos + 1]
            x_labels += [label + "\nGEE", label + "\nPython"]
            x_pos += 3   # gap between pathway pairs
 
        ax2.set_xticks(x_ticks)
        ax2.set_xticklabels(x_labels, fontsize=8)
        ax2.set_ylabel("Seconds")
        ax2.set_title("GEE vs Python time split")
 
        # Panel 1c: Peak RAM comparison
        ax3 = axes[2]
        ram_stages = [(s.name.split(":")[0], s.ram_peak_mb, s.pathway)
                      for s in report.stages if s.ram_peak_mb > 0]
        if ram_stages:
            names   = [r[0] for r in ram_stages]
            ram_mb  = [r[1] for r in ram_stages]
            colours = ["#1D9E75" if r[2] == "A_mc" else "#378ADD"
                       for r in ram_stages]
            ax3.barh(names, ram_mb, color=colours, alpha=0.85, height=0.5)
            for i, v in enumerate(ram_mb):
                ax3.text(v + 0.3, i, f"{v:.1f} MB", va="center", fontsize=8)
        ax3.set_xlabel("Peak RAM (MB)")
        ax3.set_title("Python-side peak RAM")
 
        plt.show()
 
        # ── Figure 2: Map comparison ─────────────────────────────────────────
        if report.mc_mode_map is None or report.direct_map is None:
            return
 
        fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
        fig2.suptitle("Output map comparison", fontsize=12, fontweight="500")
 
        axes2[0].imshow(report.mc_mode_map, cmap="tab10", interpolation="nearest")
        axes2[0].set_title("Pathway A — MC mode map")
        axes2[0].axis("off")
 
        axes2[1].imshow(report.direct_map, cmap="tab10", interpolation="nearest")
        axes2[1].set_title("Pathway B — Direct RF map")
        axes2[1].axis("off")
 
        entropy_display = report.mc_entropy_map if report.mc_entropy_map is not None \
            else np.zeros_like(report.mc_mode_map, dtype=float)
        im = axes2[2].imshow(entropy_display, cmap="YlOrRd",
                             vmin=0, vmax=entropy_display.max())
        plt.colorbar(im, ax=axes2[2], fraction=0.046, pad=0.04, label="nats")
        axes2[2].set_title("Pathway A — MC entropy\n(uncertainty, no equivalent in B)")
        axes2[2].axis("off")
 
        plt.show()
 
 
# ===========================================================================
# Convenience: run a single pathway only
# ===========================================================================
 
def benchmark_pathway_a_only(
    image, training_fc, rules_df, scheme_name, aoi,
    scale=30, n_iterations=300, n_trees=50
) -> PathwayReport:
    """
    Run and time only Pathway A (MC). Useful when comparing MC iteration
    counts or primitive counts without running Pathway B.
 
    Example
    -------
    >>> report = benchmark_pathway_a_only(image, fc, rules, "scheme1", aoi,
    ...                                   n_iterations=500)
    """
    bench = PathwayBenchmark(
        image, training_fc, rules_df, scheme_name, aoi,
        scale, n_iterations, n_trees
    )
    report = PathwayReport(
        scheme_name=scheme_name,
        n_primitives=bench._n_primitives,
        n_classes=bench._n_classes,
        n_iterations=n_iterations,
        aoi_pixels=0,
        scale_m=scale,
    )
    bench._run_pathway_a(report)
    report.quality = {}
    return report
 
 
def benchmark_pathway_b_only(
    image, training_fc, rules_df, scheme_name, aoi,
    scale=30, n_trees=50
) -> PathwayReport:
    """
    Run and time only Pathway B (Direct). Useful for isolated testing.
 
    Example
    -------
    >>> report = benchmark_pathway_b_only(image, fc, rules, "scheme1", aoi)
    """
    bench = PathwayBenchmark(
        image, training_fc, rules_df, scheme_name, aoi, scale, 0, n_trees
    )
    report = PathwayReport(
        scheme_name=scheme_name,
        n_primitives=bench._n_primitives,
        n_classes=bench._n_classes,
        n_iterations=0,
        aoi_pixels=0,
        scale_m=scale,
    )
    bench._run_pathway_b(report)
    report.quality = {}
    return report
 