import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import rasterio
from rasterio.mask import mask
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_bounds
import geopandas as gpd
import ee
import geemap
from shapely.geometry import mapping


# ----------------------------------------------------------------------
# Raster I/O
# ----------------------------------------------------------------------
def read_mosaic_rast(mosaic_rast, aoi_gdf=None, scale=None):
    """
    Load a raster (local or GEE asset) and optionally clip to AOI.

    Parameters
    ----------
    mosaic_rast : str
        Path to local file or GEE asset ID (starts with projects/, users/, assets/).
    aoi_gdf : GeoDataFrame, optional
        Clipping polygon.
    scale : float, optional
        Output resolution in metres (GEE only).

    Returns
    -------
    data : ndarray (bands, rows, cols)
    transform : Affine
    meta : dict
    band_names : list
    """
    if (mosaic_rast.startswith('projects/') or
        mosaic_rast.startswith('users/') or
        mosaic_rast.startswith('assets/')):
        return _read_gee_asset(mosaic_rast, aoi_gdf, scale)
    return _read_local_tif(mosaic_rast, aoi_gdf)


def _read_local_tif(mosaic_rast, aoi_gdf):
    """Read local GeoTIFF, optionally mask with AOI."""
    with rasterio.open(mosaic_rast) as src:
        desc = src.descriptions
        band_names = [d if d else f'band_{i+1}' for i, d in enumerate(desc)] if any(desc) else [f'band_{i+1}' for i in range(src.count)]
        if aoi_gdf is not None:
            if aoi_gdf.crs != src.crs:
                aoi_gdf_proj = aoi_gdf.to_crs(src.crs)
            else:
                aoi_gdf_proj = aoi_gdf
            shapes = aoi_gdf_proj.geometry.values
            data, transform = mask(src, shapes, crop=True)
            meta = src.meta.copy()
            meta.update(height=data.shape[1], width=data.shape[2], transform=transform)
        else:
            data = src.read()
            transform = src.transform
            meta = src.meta.copy()
    return data, transform, meta, band_names


def _read_gee_asset(asset_id, aoi_gdf, scale):
    """Read GEE Image or ImageCollection asset and return as numpy array."""
    try:
        ee.data.getAssetRoots()
    except ee.EEException as e:
        raise ee.EEException("Earth Engine not initialized. Call ee.Authenticate() and ee.Initialize().") from e

    info = ee.data.getAsset(asset_id)
    asset = ee.Image(asset_id) if info['type'] == 'IMAGE' else ee.ImageCollection(asset_id).mosaic() if info['type'] == 'IMAGE_COLLECTION' else None
    if asset is None:
        raise ValueError(f"Unsupported GEE asset type: {info['type']}")

    band_names = asset.bandNames().getInfo()
    if aoi_gdf is not None:
        if aoi_gdf.crs and aoi_gdf.crs.to_wkt() != 'GEOGCS["WGS 84",...]':
            aoi_gdf = aoi_gdf.to_crs('EPSG:4326')
        geom = ee.Geometry(mapping(aoi_gdf.geometry.unary_union))
        asset = asset.clip(geom)
        region = geom
    else:
        region = asset.geometry()

    if scale is None:
        proj = asset.select([band_names[0]]).projection()
        scale = proj.nominalScale().getInfo()
    scale = max(scale, 0.1)

    array = geemap.ee_to_numpy(asset, region=region, scale=scale)
    bounds = region.bounds().getInfo()
    coords = bounds['coordinates'][0] if bounds['type'] == 'Polygon' else [pt for poly in bounds['coordinates'] for ring in poly for pt in ring]
    xs, ys = [p[0] for p in coords], [p[1] for p in coords]
    xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)

    data = np.transpose(array, (2, 0, 1))
    h, w = data.shape[1], data.shape[2]
    transform = from_bounds(xmin, ymin, xmax, ymax, w, h)
    meta = {'driver': 'GTiff', 'dtype': data.dtype.name, 'nodata': None,
            'width': w, 'height': h, 'count': data.shape[0],
            'crs': 'EPSG:4326', 'transform': transform}
    return data, transform, meta, band_names


# ----------------------------------------------------------------------
# Classification scheme and mapping
# ----------------------------------------------------------------------
def read_classification_scheme(df):
    """
    Build hierarchical scheme tables from classification DataFrame.

    Parameters
    ----------
    df : DataFrame with columns: ID, Level3_Name, Level2_Name, Level1_Name, Terminal_Level, Color.

    Returns
    -------
    dict with keys: table, level2_table, level1_table, terminal_level,
                    level3_to_level2, level2_to_level1.
    """
    required = ["ID", "Level3_Name", "Level2_Name", "Level1_Name", "Terminal_Level"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df[df["Terminal_Level"] != 0].copy()
    if df.empty:
        raise ValueError("No classes after excluding Terminal_Level == 0.")
    if not df["Terminal_Level"].isin([1, 2, 3]).all():
        raise ValueError("Terminal_Level must be 1, 2 or 3.")
    if df["ID"].duplicated().any():
        raise ValueError("Duplicated ID.")

    l2 = df[["Level2_Name"]].drop_duplicates().reset_index(drop=True)
    l2["Level2_ID"] = l2.index + 1
    l1 = df[["Level1_Name"]].drop_duplicates().reset_index(drop=True)
    l1["Level1_ID"] = l1.index + 1

    df = df.merge(l2, on="Level2_Name").merge(l1, on="Level1_Name")
    l2 = df[["Level2_ID", "Level2_Name", "Level1_ID"]].drop_duplicates().sort_values("Level2_ID")
    l1 = df[["Level1_ID", "Level1_Name"]].drop_duplicates().sort_values("Level1_ID")

    return {
        "table": df,
        "level2_table": l2,
        "level1_table": l1,
        "terminal_level": df.set_index("ID")["Terminal_Level"].to_dict(),
        "level3_to_level2": df.set_index("ID")["Level2_ID"].to_dict(),
        "level2_to_level1": l2.set_index("Level2_ID")["Level1_ID"].to_dict()
    }


def build_mapping(scheme):
    """
    Build aggregation mapping from scheme: map each output class to member IDs.

    Parameters
    ----------
    scheme : dict from read_classification_scheme.

    Returns
    -------
    dict with keys: output_lookup (per output class info), output_table (DataFrame).
    """
    df = scheme["table"].copy()
    df["Band_Index"] = df["ID"] - 1
    lookup = {}
    out_id = 1

    for _, row in df[df.Terminal_Level == 3].iterrows():
        lookup[out_id] = {"output_id": out_id, "output_name": row.Level3_Name,
                          "output_level": 3, "color": row.Color,
                          "member_ids": [row.ID], "member_band_indices": [row.Band_Index]}
        out_id += 1

    for name, group in df[df.Terminal_Level == 2].groupby("Level2_Name"):
        lookup[out_id] = {"output_id": out_id, "output_name": name,
                          "output_level": 2, "color": group.Color.iloc[0],
                          "member_ids": group.ID.tolist(),
                          "member_band_indices": group.Band_Index.tolist()}
        out_id += 1

    for name, group in df[df.Terminal_Level == 1].groupby("Level1_Name"):
        lookup[out_id] = {"output_id": out_id, "output_name": name,
                          "output_level": 1, "color": group.Color.iloc[0],
                          "member_ids": group.ID.tolist(),
                          "member_band_indices": group.Band_Index.tolist()}
        out_id += 1

    table = pd.DataFrame([{"Output_ID": v["output_id"], "Output_Name": v["output_name"],
                           "Output_Level": v["output_level"], "Color": v["color"],
                           "Member_IDs": v["member_ids"]} for v in lookup.values()])
    return {"output_lookup": lookup, "output_table": table}


# ----------------------------------------------------------------------
# Aggregation
# ----------------------------------------------------------------------
def aggregate_probability(raster_data, mapping, band_names=None):
    """
    Sum probabilities of member classes to produce aggregated probability cube.

    Parameters
    ----------
    raster_data : ndarray (n_bands, rows, cols)
    mapping : dict from build_mapping.
    band_names : list, optional; used to map member IDs to band indices.

    Returns
    -------
    aggregated_cube : ndarray (n_aggregated, rows, cols)
    output_info : DataFrame with columns Class_ID, Class_Name, Class_Level, Color, Band_IDs, Band_Names.
    """
    lookup = mapping["output_lookup"]
    n_out = len(lookup)
    rows, cols = raster_data.shape[1], raster_data.shape[2]
    out = np.zeros((n_out, rows, cols), dtype=raster_data.dtype)
    summary = []

    if band_names is not None:
        if len(band_names) != raster_data.shape[0]:
            raise ValueError("band_names length mismatch")
        name_to_idx = {n: i for i, n in enumerate(band_names)}
        use_name = True
    else:
        use_name = False
        print("Warning: band_names not provided; using member_band_indices.")

    for i, info in enumerate(lookup.values()):
        mids = info["member_ids"]
        valid_idx, valid_names = [], []
        missing = []

        if use_name:
            for mid in mids:
                bn = f"prob_{mid}"
                if bn in name_to_idx:
                    idx = name_to_idx[bn]
                    valid_idx.append(idx)
                    valid_names.append(bn)
                else:
                    missing.append(mid)
        else:
            for idx in info["member_band_indices"]:
                if 0 <= idx < raster_data.shape[0]:
                    valid_idx.append(idx)
                    valid_names.append(str(idx))
                else:
                    missing.append(idx)

        if missing:
            print(f"Warning: '{info['output_name']}' missing bands for IDs {missing}")

        if valid_idx:
            out[i] = np.sum(raster_data[valid_idx], axis=0)
        else:
            out[i] = np.zeros((rows, cols), dtype=raster_data.dtype)

        summary.append({
            "Class_ID": info["output_id"],
            "Class_Name": info["output_name"],
            "Class_Level": info["output_level"],
            "Color": info["color"],
            "Band_IDs": valid_idx,
            "Band_Names": valid_names
        })

    return out, pd.DataFrame(summary)


# ----------------------------------------------------------------------
# Argmax classification 
# ----------------------------------------------------------------------
def predict_class(aggregated_cube, output_info, raster_meta):
    """
    Perform argmax classification on aggregated probability cube.

    Parameters
    ----------
    aggregated_cube : ndarray (n_classes, rows, cols)
    output_info : DataFrame from aggregate_probability.
    raster_meta : dict metadata.

    Returns
    -------
    classified_map : ndarray (rows, cols) int16
    classified_meta : dict
    max_prob_map : ndarray (rows, cols)
    argmax_indices : ndarray (rows, cols)
    selected_class_ids : list
    """
    if aggregated_cube.ndim != 3:
        raise ValueError("aggregated_cube must be 3D (bands, rows, cols)")

    class_ids = output_info["Class_ID"].tolist()
    if len(class_ids) != aggregated_cube.shape[0]:
        print("Warning: output_info mismatch; using sequential IDs.")
        class_ids = list(range(1, aggregated_cube.shape[0] + 1))

    argmax = np.argmax(aggregated_cube, axis=0)
    maxprob = np.max(aggregated_cube, axis=0)

    flat = np.array([class_ids[i] for i in argmax.flat], dtype=np.int16).reshape(argmax.shape)
    flat[maxprob == 0] = 0

    meta = raster_meta.copy()
    meta.update(count=1, dtype="int16")
    return flat, meta, maxprob, argmax, class_ids


# ----------------------------------------------------------------------
# Rule‐based refinement
# ----------------------------------------------------------------------
DEFAULT_PREDICTOR_MAP = {
    "aceh_dem_30m": "dem",
    "aceh_slope_30m_deg": "slope",
    "aceh_dist_to_coast_100m": "dist_coast",
    "aceh_worldpop_2020_30m_ne": "worldpop",
}

def resample_predictors_to_probability_grid(stack, src_meta, dst_meta, resampling=Resampling.average):
    """
    Reproject/resample predictor stack onto the probability raster's grid.

    Parameters
    ----------
    stack : ndarray (n_bands, src_rows, src_cols)
    src_meta : dict with transform, crs, nodata
    dst_meta : dict with transform, crs, height, width
    resampling : Resampling method

    Returns
    -------
    ndarray (n_bands, dst_height, dst_width)
    """
    n = stack.shape[0]
    out = np.full((n, dst_meta["height"], dst_meta["width"]), np.nan, dtype=np.float32)
    for i in range(n):
        reproject(source=stack[i], destination=out[i],
                  src_transform=src_meta["transform"], src_crs=src_meta["crs"],
                  dst_transform=dst_meta["transform"], dst_crs=dst_meta["crs"],
                  resampling=resampling,
                  src_nodata=src_meta.get("nodata"), dst_nodata=np.nan)
    return out


def build_validity_stack(predictor_grid, predictor_names, class_ids, ruleset_df, predictor_map=None):
    """
    Build boolean validity mask for each class based on predictor rules.

    Parameters
    ----------
    predictor_grid : ndarray (n_predictors, rows, cols) already on target grid
    predictor_names : list of band names
    class_ids : list of Class_ID in probability cube order
    ruleset_df : DataFrame indexed by Class_ID with min/max columns
    predictor_map : dict mapping predictor band name to rule prefix (default DEFAULT_PREDICTOR_MAP)

    Returns
    -------
    validity : ndarray (len(class_ids), rows, cols) dtype bool
    """
    if predictor_map is None:
        predictor_map = DEFAULT_PREDICTOR_MAP
    name_to_idx = {n: i for i, n in enumerate(predictor_names)}
    predictors = {name: predictor_grid[i].copy() for name, i in name_to_idx.items()}

    rows, cols = predictor_grid.shape[1], predictor_grid.shape[2]
    valid = np.ones((len(class_ids), rows, cols), dtype=bool)
    missing_classes = []

    for b, cid in enumerate(class_ids):
        if cid not in ruleset_df.index:
            missing_classes.append(cid)
            continue
        rule = ruleset_df.loc[cid]
        for band_name, prefix in predictor_map.items():
            if band_name not in predictors:
                continue
            min_col, max_col = f"{prefix}_min", f"{prefix}_max"
            if min_col not in rule or max_col not in rule:
                continue
            vals = predictors[band_name]
            lo, hi = rule[min_col], rule[max_col]
            ok = np.ones((rows, cols), dtype=bool)
            if pd.notna(lo):
                ok &= vals >= lo
            if pd.notna(hi):
                ok &= vals <= hi
            ok |= np.isnan(vals)  
            valid[b] &= ok

    if missing_classes:
        print(f"Warning: no ruleset row for IDs {sorted(set(missing_classes))}; unconstrained.")
    return valid


def predict_class_ruleset(valid, aggregated_cube, raw_argmax, raw_maxprob, raw_map, raw_meta, class_ids):
    """
    Argmax classification followed by rule-based post‑processing.

    Parameters
    ----------
    valid : ndarray (n_classes, rows, cols) boolean
    aggregated_cube : ndarray (n_classes, rows, cols)
    output_info : DataFrame
    raster_meta : dict
    predictor_stack : ndarray (n_pred, pred_rows, pred_cols)
    predictor_meta : dict for predictor stack
    predictor_names : list of predictor band names
    ruleset_df : DataFrame from load_ruleset
    predictor_map : optional mapping
    resampling : Resampling method

    Returns
    -------
    dict with keys: classified_map, classified_meta, max_prob_map, argmax_indices,
                    selected_class_ids, raw_classified_map, raw_argmax_indices,
                    validity_stack, corrected_mask, no_valid_class_mask.
    """
    
    # eliminate invalid classes, then re‑argmax
    adjusted_cube = np.where(valid, aggregated_cube, -1.0)
    adj_argmax = np.argmax(adjusted_cube, axis=0)
    adj_maxprob = np.max(adjusted_cube, axis=0)

    no_valid = ~valid.any(axis=0)
    if no_valid.any():
        print(f"Note: {int(no_valid.sum())} pixel(s) with no valid class; fallback to raw argmax.")

    final_argmax = np.where(no_valid, raw_argmax, adj_argmax)
    final_maxprob = np.where(no_valid, raw_maxprob, adj_maxprob)

    final_map = np.array([class_ids[i] for i in final_argmax.flat], dtype=np.int16).reshape(final_argmax.shape)
    final_map[raw_maxprob == 0] = 0

    corrected = (final_argmax != raw_argmax) & (raw_maxprob != 0)
    if corrected.any():
        pct = 100 * corrected.mean()
        print(f"Ruleset changed {int(corrected.sum())} pixels ({pct:.2f}% of tile).")

    return {
        "classified_map": final_map,
        "classified_meta": raw_meta,
        "max_prob_map": final_maxprob,
        "argmax_indices": final_argmax,
        "selected_class_ids": class_ids,
        "raw_classified_map": raw_map,
        "raw_argmax_indices": raw_argmax,
        "validity_stack": valid,
        "corrected_mask": corrected,
        "no_valid_class_mask": no_valid,
    }


# ----------------------------------------------------------------------
# Plotting helpers
# ----------------------------------------------------------------------
def _build_palette(class_ids, output_info):
    """Build colour palette DataFrame for given class IDs."""
    palette = pd.DataFrame({"Class_ID": np.unique(np.asarray(class_ids).astype(int))})
    palette = palette.merge(output_info[["Class_ID", "Class_Name", "Color"]], on="Class_ID", how="left")
    palette.loc[palette["Class_ID"] == 0, "Class_Name"] = "Background / NoData"
    palette.loc[palette["Class_ID"] == 0, "Color"] = "#e0e0e0"
    palette["Color"] = palette["Color"].fillna("#ffffff")
    palette["Class_Name"] = palette["Class_Name"].fillna("Unclassified")
    return palette.sort_values("Class_ID").reset_index(drop=True)


def plot_classification_summary(classified_map, max_prob_map, output_info,
                                save_path="classified_map_ruleset.png",
                                title="Rule-Refined LULC Map",
                                corrected_mask=None):
    """
    Plot 3‑panel summary: classified map, max probability, and class distribution.

    Parameters
    ----------
    classified_map : ndarray (rows, cols)
    max_prob_map : ndarray (rows, cols)
    output_info : DataFrame with Class_ID, Class_Name, Color
    save_path : str
    title : str
    corrected_mask : ndarray, optional; if provided, prints change stats.
    """
    unique_ids, counts = np.unique(classified_map, return_counts=True)
    summary = pd.DataFrame({"Class_ID": unique_ids.astype(int), "Pixel_Count": counts})
    summary = summary.merge(output_info[["Class_ID", "Class_Name", "Color"]], on="Class_ID", how="left")
    summary.loc[summary["Class_ID"] == 0, "Class_Name"] = "Background / NoData"
    summary.loc[summary["Class_ID"] == 0, "Color"] = "#e0e0e0"
    summary["Color"] = summary["Color"].fillna("#ffffff")
    summary["Class_Name"] = summary["Class_Name"].fillna("Unclassified")
    total = summary["Pixel_Count"].sum()
    summary["Percentage"] = 100 * summary["Pixel_Count"] / total
    summary = summary.sort_values("Class_ID").reset_index(drop=True)
    print("Class distribution:\n", summary[["Class_ID", "Class_Name", "Pixel_Count", "Percentage"]])

    if corrected_mask is not None:
        corrected_mask = np.asarray(corrected_mask)
        n = int(corrected_mask.sum())
        print(f"Ruleset changed {n} pixels ({100*n/corrected_mask.size:.2f}%)")

    fig, ax = plt.subplots(1, 3, figsize=(20, 6), gridspec_kw={"width_ratios": [1, 1, 1.2]})
    cmap = ListedColormap(summary["Color"].tolist())
    idx_map = np.searchsorted(summary["Class_ID"].values, classified_map)

    im0 = ax[0].imshow(idx_map, cmap=cmap, interpolation="nearest")
    ax[0].set_title(title, fontsize=12, fontweight="bold")
    ax[0].axis("off")
    cbar0 = fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04, ticks=range(len(summary)))
    cbar0.ax.set_yticklabels(summary["Class_ID"].astype(str))
    cbar0.set_label("Class ID", fontsize=10)

    im1 = ax[1].imshow(max_prob_map, cmap="viridis", interpolation="nearest")
    ax[1].set_title("Probability Map (selected class)", fontsize=12, fontweight="bold")
    ax[1].axis("off")
    fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04, label="Probability")

    plot_data = summary[summary["Class_ID"] != 0]
    ax[2].bar(plot_data["Class_ID"].astype(str), plot_data["Pixel_Count"],
              color=plot_data["Color"], edgecolor="black", linewidth=0.8)
    ax[2].set_title("Class Distribution (pixel count)", fontsize=12, fontweight="bold")
    ax[2].set_xlabel("Class ID")
    ax[2].set_ylabel("Pixel Count")
    ax[2].tick_params(axis="x", rotation=45)
    ax[2].grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()
    return summary


def plot_ruleset_comparison(result, output_info,
                            save_path="ruleset_comparison.png"):
    """
    Side‑by‑side comparison of raw argmax vs ruleset‑corrected classification.

    Parameters
    ----------
    result : dict from predict_class_ruleset
    output_info : DataFrame
    save_path : str

    Returns
    -------
    comparison_table : DataFrame with per‑class counts before/after.
    """
    raw = result["raw_classified_map"]
    final = result["classified_map"]
    corrected = np.asarray(result["corrected_mask"])
    no_valid = np.asarray(result["no_valid_class_mask"])

    all_ids = np.union1d(np.unique(raw), np.unique(final))
    palette = _build_palette(all_ids, output_info)
    cmap = ListedColormap(palette["Color"].tolist())

    def to_idx(m):
        return np.searchsorted(palette["Class_ID"].values, m)

    # comparison table
    raw_ids, raw_cnt = np.unique(raw, return_counts=True)
    final_ids, final_cnt = np.unique(final, return_counts=True)
    raw_d = dict(zip(raw_ids.astype(int), raw_cnt))
    final_d = dict(zip(final_ids.astype(int), final_cnt))
    comp = palette.copy()
    comp["Raw_Count"] = comp["Class_ID"].map(raw_d).fillna(0).astype(int)
    comp["Corrected_Count"] = comp["Class_ID"].map(final_d).fillna(0).astype(int)
    comp["Change"] = comp["Corrected_Count"] - comp["Raw_Count"]
    print("Raw vs corrected:\n", comp[["Class_ID", "Class_Name", "Raw_Count", "Corrected_Count", "Change"]])

    n_corr = int(corrected.sum())
    n_fb = int(no_valid.sum())
    total = corrected.size
    print(f"{n_corr} pixels changed ({100*n_corr/total:.2f}%)")
    print(f"{n_fb} pixels with no valid class (fallback) ({100*n_fb/total:.2f}%)")

    # status map: 0=background, 1=unchanged, 2=corrected, 3=fallback
    status = np.ones(raw.shape, dtype=np.int8)
    status[raw == 0] = 0
    status[corrected] = 2
    status[no_valid & (raw != 0)] = 3
    status_cmap = ListedColormap(["#e0e0e0", "#d9f0d3", "#e31a1c", "#6a3d9a"])
    status_labels = ["Background", "Unchanged", "Corrected", "Fallback"]

    fig, ax = plt.subplots(2, 2, figsize=(16, 14))

    im0 = ax[0, 0].imshow(to_idx(raw), cmap=cmap, interpolation="nearest")
    ax[0, 0].set_title("Raw Argmax", fontsize=12, fontweight="bold")
    ax[0, 0].axis("off")
    cbar0 = fig.colorbar(im0, ax=ax[0,0], fraction=0.046, pad=0.04, ticks=range(len(palette)))
    cbar0.ax.set_yticklabels(palette["Class_ID"].astype(str))
    cbar0.set_label("Class ID")

    im1 = ax[0, 1].imshow(to_idx(final), cmap=cmap, interpolation="nearest")
    ax[0, 1].set_title("Ruleset-Corrected", fontsize=12, fontweight="bold")
    ax[0, 1].axis("off")
    cbar1 = fig.colorbar(im1, ax=ax[0,1], fraction=0.046, pad=0.04, ticks=range(len(palette)))
    cbar1.ax.set_yticklabels(palette["Class_ID"].astype(str))
    cbar1.set_label("Class ID")

    im2 = ax[1, 0].imshow(status, cmap=status_cmap, interpolation="nearest", vmin=0, vmax=3)
    ax[1, 0].set_title("Change Status", fontsize=12, fontweight="bold")
    ax[1, 0].axis("off")
    cbar2 = fig.colorbar(im2, ax=ax[1,0], fraction=0.046, pad=0.04, ticks=[0,1,2,3])
    cbar2.ax.set_yticklabels(status_labels, fontsize=8)

    chart = comp[comp["Class_ID"] != 0]
    x = np.arange(len(chart))
    width = 0.38
    ax[1, 1].bar(x - width/2, chart["Raw_Count"], width, label="Raw argmax", color="#9e9e9e", edgecolor="black")
    ax[1, 1].bar(x + width/2, chart["Corrected_Count"], width, label="Ruleset-corrected", color="#e31a1c", edgecolor="black")
    ax[1, 1].set_xticks(x)
    ax[1, 1].set_xticklabels(chart["Class_ID"].astype(str), rotation=45)
    ax[1, 1].set_title("Class Distribution: Raw vs Corrected", fontsize=12, fontweight="bold")
    ax[1, 1].set_xlabel("Class ID")
    ax[1, 1].set_ylabel("Pixel Count")
    ax[1, 1].legend()
    ax[1, 1].grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()
    return comp