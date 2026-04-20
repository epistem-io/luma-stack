# `luma_ge` — Modular Workflow Summary

## Project Context

`luma_ge` is the geospatial engine for the **Luma (Land Use Mapping for All)** platform by EpistemX. It processes satellite imagery via Google Earth Engine (GEE) to produce land cover maps structured around the **Land Cover Meta Language (LCML)** standard. The package is designed to support multiple classification schemes from a single training dataset, with built-in uncertainty quantification.

---

## Repository Structure

```
luma_ge/
├── ee_config.py              # GEE authentication (service account, manual, env vars)
├── helpers.py                # Shared utilities (AOI from GAUL, etc.)
├── input_utils.py            # Input file validation
├── data_acquisition.py       # Module 1 — Near-cloud-free satellite imagery
├── classification_scheme.py  # Module 2 — LULC scheme management
├── sample_data.py            # Module 3 — Sample data generation
├── sample_data_quality.py    # Module 4 — Sample data quality analysis
├── predictor.py              # Module 5 — Predictor/feature layers
├── classification.py         # Module 6 — Traditional RF classification
├── accuracy.py               # Module 7 — Thematic accuracy assessment
├── modular_workflow.py       # Extended pipeline (multi-scheme + uncertainty)
└── benchmark_pathways.py     # Benchmarking framework (Pathway A vs B)
```

`modular_workflow.py` is a **parallel, higher-level workflow** that extends the traditional pipeline without replacing it.

---

## The Core Problem It Solves

The traditional pipeline (`classification.py`) is **scheme-coupled** — changing the classification scheme requires retraining the entire Random Forest from scratch. `modular_workflow.py` decouples scheme definition from model training:

| Concern | Traditional Pipeline | Modular Workflow |
|---|---|---|
| Scheme change | Retrain everything | Swap the CSV |
| Uncertainty output | None | Shannon entropy via Monte Carlo |
| Rule transparency | Implicit in training labels | Explicit, auditable CSV |
| Multi-scheme support | One full run per scheme | One primitive datacube → N schemes |
| Classification structure | Flat RF | Flat rules or hierarchical decision tree |

---

## Workflow Steps

### Step 1 — Load Classification Scheme (`load_scheme`)

Reads a human-authored CSV that defines land cover classes and their rules. Each class specifies:
- `class_id`, `class_name`, `priority`
- `rule_general`: `"none"` | `"and"` | `"or"` — how conditions are combined
- Per-element threshold columns (e.g. `tree_pres`, `rule_tree_pres`)

Rule assembly logic:
- `"none"` — picks the single most discriminating non-zero condition
- `"and"` — joins all conditions with AND
- `"or"` — joins all conditions with OR

Output: a DataFrame with columns `class_id`, `class_name`, `rule`, `priority` where `rule` is a fully-formed expression string like `tree_pres > 0.6 AND buil_pres > 0.3`.

---

### Step 2 — Load Training Data

**`load_modular_training_data(shp_path, aoi)`**

Loads a shapefile of reference points. Each point carries binary (0/1) LCML element attributes (e.g. `tree_pres`, `water_pres`, `buil_pres`). Optionally filters to an AOI using Shapely geometry intersection. Returns both a GeoDataFrame and an `ee.FeatureCollection`.

**`load_element_mapping(csv_path)`** + **`enrich_training_data(gdf, mapping)`**

Optional enrichment step. Samples external GEE datasets (e.g. GFCC tree canopy, JRC water occurrence) at each training point to fill missing primitive columns automatically. Supports transforms: `none`, `divide_100`, `divide_max`, `log10`. Skips already-populated columns unless `overwrite_existing=True`.

---

### Step 3 — Classification (Two Pathways)

#### Pathway A — Direct Labelling (`TrainingDataLabeller`)

Evaluates scheme rules directly against training feature attributes on the GEE server. Assigns `class_id` to each feature via a nested `ee.Algorithms.If` chain (priority order). The labeled `ee.FeatureCollection` is then used to train a single direct RF classifier per scheme.

- Fast, no uncertainty output
- Scheme-coupled — changing the scheme requires re-labelling and retraining

#### Pathway B — Primitive Layer (`PrimitiveLayerTrainer` → `RuleSetClassifier`)

Trains one RF per LCML element independently, producing a **reusable primitive datacube**. The datacube encodes per-pixel element presence independently of any scheme, so multiple schemes can be applied without retraining.

Training modes:
- `train_all()` — binary output (0/1), deterministic pathway
- `train_all_mc()` — probabilistic output [0,1] via `.setOutputMode('PROBABILITY')`, Monte Carlo pathway

Classification modes on the primitive datacube:

| Classifier | Structure | Uncertainty |
|---|---|---|
| `RuleSetClassifier` | Flat priority-ordered rules | Deterministic or Monte Carlo |
| `HierarchicalRuleSetClassifier` | Decision tree (LCML dichotomous key) | Deterministic or Monte Carlo |

The hierarchical classifier mirrors the LCML dichotomous key (vegetation → cover/height/phenology → non-veg → bare/built/water). It avoids the **reachability problem** of flat rulesets, where compound AND classes can be structurally unreachable if a higher-priority class fires first.

---

### Monte Carlo Uncertainty Quantification

In each iteration:
1. Each pixel's probability `p` is sampled as `Bernoulli(p)` → 0 or 1
2. The binary values are passed through the deterministic ruleset
3. The class assignment is recorded

After N iterations, the pixel counts are aggregated into:
- `mode_map` — most frequently assigned class per pixel
- `entropy_map` — Shannon entropy `H = -Σ p·ln(p)` (nats)
- `class_probs` — fraction of iterations each class won per pixel

High entropy = the simulation disagreed often = genuinely uncertain pixel.

In the hierarchical classifier, uncertainty accumulates at **every branching node** in the path, not just the final class — a pixel uncertain at the vegetation/non-vegetation split shows high entropy even if its sub-tree is decisive.

---

### Step 4 — Validation

| Function | Output |
|---|---|
| `validate_deterministic()` | Area share table, class map, bar chart |
| `validate_monte_carlo()` | Entropy stats, entropy histogram, entropy spatial map, mode map, per-class probability histograms |
| `compare_det_vs_mc()` | Side-by-side area share table, disagreement map coloured by MC entropy |

All three functions share the same console summary format so deterministic and MC results are directly comparable.

---

## Benchmarking (`benchmark_pathways.py`)

Compares Pathway A (direct RF) vs Pathway B (primitive layer + MC) across:
- Wall-clock time per stage
- GEE EECU consumption
- Peak Python RAM
- Output quality (pixel agreement, entropy, per-class area share)

---

## Key Design Decisions

**Primitive datacube as the core reusable asset**
Train once per element, apply any number of schemes. The datacube is scheme-agnostic.

**Rules as auditable CSV**
Classification logic lives in a human-readable file, not buried in training labels. Non-technical domain experts can author and review rules directly.

**Two classifier structures for two LCML paradigms**
Flat rules (`RuleSetClassifier`) for simple compound conditions. Hierarchical tree (`HierarchicalRuleSetClassifier`) for the full LCML dichotomous key, which avoids reachability issues.

**Monte Carlo over GEE-native uncertainty**
Probabilistic primitives are downloaded once; all MC iterations run locally in NumPy. This avoids repeated GEE round-trips and gives full control over the sampling and aggregation logic.

**Shannon entropy as the uncertainty metric**
Directly interpretable: `ln(n_classes)` is the maximum possible entropy (uniform distribution). Pixels near this value are maximally uncertain; pixels near 0 are consistently classified across all iterations.

---

## Data Flow Diagram

```
Scheme CSV ──► load_scheme() ──────────────────────────────────────────┐
                                                                        │
Shapefile ───► load_modular_training_data()                             │
                    │                                                   │
                    ├──► (optional) enrich_training_data()              │
                    │         ▲                                         │
                    │    element_mapping.csv                            │
                    │                                                   ▼
                    │                              ┌─────────────────────────────┐
                    │   Pathway A                  │   Pathway B                 │
                    ├──► TrainingDataLabeller ──►  │   PrimitiveLayerTrainer     │
                    │    labeled ee.FC             │   primitive datacube        │
                    │    → train 1 RF/scheme       │   (1 RF/element, reusable)  │
                    │    → classify on GEE         │                             │
                    │                              │   RuleSetClassifier         │
                    │                              │   or                        │
                    │                              │   HierarchicalRuleSetClassifier
                    │                              │                             │
                    │                              │   Deterministic  Monte Carlo│
                    │                              └─────────────────────────────┘
                    │                                        │
                    └────────────────────────────────────────┤
                                                             ▼
                                              validate_deterministic()
                                              validate_monte_carlo()
                                              compare_det_vs_mc()
```
