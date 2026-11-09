"""
Multitemporal Land Use / Land Cover (LULC) Change Detection Module

This module provides tools for multitemporal LULC comparison analysis,
transition matrices (crosstabulation), spatial change mapping, gain/loss accounting,
and automated reporting.
"""

import os
import math
import logging
from typing import Optional, Dict, Any, List, Tuple, Union

import ee
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.crs import CRS
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject, transform_bounds
from shapely.geometry import box, shape
from shapely.ops import unary_union
from shapely.validation import make_valid
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

from .ee_config import ensure_ee_initialized
from .classification_scheme import LULC_Scheme_Manager
from .input_utils import shapefile_validator

logger = logging.getLogger(__name__)

M_PER_DEG = 111_320.0
NODATA = -9999


def is_ee_source(src: Any) -> bool:
    """Check whether a source path points to a Google Earth Engine asset."""
    return str(src).startswith(("projects/", "users/"))


def src_info(src: str, aoi_ll: gpd.GeoDataFrame, m_per_deg: float = M_PER_DEG) -> Dict[str, Any]:
    """
    Retrieve native resolution (m) of a raster map and verify if its footprint touches the AOI.
    
    Parameters
    ----------
    src : str
        Local raster file path or Earth Engine Image asset id.
    aoi_ll : gpd.GeoDataFrame
        Area of Interest GeoDataFrame in geographic coordinates (EPSG:4326).
    m_per_deg : float, default 111320.0
        Meters per degree for geographic CRS resolution conversion.

    Returns
    -------
    dict
        Dictionary containing 'src', 'res_m' (float), and 'hits_aoi' (bool).
    """
    aoi_geom = aoi_ll.geometry.iloc[0]
    if is_ee_source(src):
        ensure_ee_initialized()
        img = ee.Image(src)
        res = img.select(0).projection().nominalScale().getInfo()
        hit = img.geometry().intersects(ee.Geometry(aoi_geom.__geo_interface__), ee.ErrorMargin(1)).getInfo()
    else:
        with rasterio.open(src) as ds:
            if ds.crs is None:
                raise ValueError(f"{src} has no CRS defined")
            hit = box(*transform_bounds(ds.crs, "EPSG:4326", *ds.bounds)).intersects(aoi_geom)
            if ds.crs.is_geographic:
                res = abs(ds.res[1]) * m_per_deg
            else:
                res = abs(ds.res[1]) * ds.crs.linear_units_factor[1]
    return dict(src=src, res_m=float(res), hits_aoi=bool(hit))


def build_grid(aoi_ll: gpd.GeoDataFrame, res_m: float, max_pixels: int = 150_000_000) -> Dict[str, Any]:
    """
    Build common analysis grid in the estimated UTM zone of the AOI centroid.

    Parameters
    ----------
    aoi_ll : gpd.GeoDataFrame
        AOI GeoDataFrame in EPSG:4326.
    res_m : float
        Target resolution in meters.
    max_pixels : int, default 150,000,000
        Safety limit for total in-memory grid cell count.

    Returns
    -------
    dict
        Grid specification dictionary (crs, transform, width, height, res_m, x0, y1, aoi_utm).
    """
    utm = aoi_ll.estimate_utm_crs()
    aoi_utm = aoi_ll.to_crs(utm)
    minx, miny, maxx, maxy = aoi_utm.total_bounds
    x0 = math.floor(minx / res_m) * res_m
    y1 = math.ceil(maxy / res_m) * res_m
    width = int(math.ceil((maxx - x0) / res_m))
    height = int(math.ceil((y1 - miny) / res_m))

    if width * height > max_pixels:
        raise ValueError(
            f"Grid is {width} x {height} = {width * height:,} pixels (limit {max_pixels:,}). "
            "Use a smaller AOI or a coarser target_res_m."
        )

    return dict(
        crs=utm,
        transform=from_origin(x0, y1, res_m, res_m),
        width=width,
        height=height,
        res_m=res_m,
        x0=x0,
        y1=y1,
        aoi_utm=aoi_utm,
    )


def to_ids(arr: np.ndarray, label: str = "map", nodata: int = NODATA) -> np.ndarray:
    """Validate that valid grid pixels are whole numbers and cast float32 grid to int32."""
    valid = arr != nodata
    if np.any(valid & (arr != np.round(arr))):
        raise ValueError(f"{label} contains non-integer class values")
    return arr.astype(np.int32)


def read_local(path: str, grid: Dict[str, Any], nodata: int = NODATA) -> np.ndarray:
    """Reproject local raster band onto common target grid using nearest neighbor resampling."""
    dst = np.full((grid["height"], grid["width"]), nodata, dtype=np.float32)
    with rasterio.open(path) as ds:
        reproject(
            source=rasterio.band(ds, 1),
            destination=dst,
            dst_transform=grid["transform"],
            dst_crs=CRS.from_user_input(grid["crs"]),
            src_nodata=ds.nodata,
            dst_nodata=nodata,
            resampling=Resampling.nearest,
        )
    return dst


def read_ee(asset: str, grid: Dict[str, Any], tile: int = 2048, nodata: int = NODATA) -> np.ndarray:
    """
    Pull an Earth Engine image onto the common UTM grid in tiles using computePixels.
    """
    ensure_ee_initialized()
    img = ee.Image(asset).select(0).toFloat().unmask(nodata, False)
    out = np.full((grid["height"], grid["width"]), nodata, dtype=np.float32)
    res = grid["res_m"]
    crs_code = f"EPSG:{grid['crs'].to_epsg()}"
    
    for r in range(0, grid["height"], tile):
        for c in range(0, grid["width"], tile):
            h = min(tile, grid["height"] - r)
            w = min(tile, grid["width"] - c)
            req = dict(
                expression=img,
                fileFormat="NUMPY_NDARRAY",
                grid=dict(
                    dimensions=dict(width=w, height=h),
                    affineTransform=dict(
                        scaleX=res, shearX=0, translateX=grid["x0"] + c * res,
                        shearY=0, scaleY=-res, translateY=grid["y1"] - r * res,
                    ),
                    crsCode=crs_code,
                ),
            )
            arr = ee.data.computePixels(req)
            out[r:r + h, c:c + w] = arr[arr.dtype.names[0]]
    return out


def read_map(
    src: str,
    grid: Dict[str, Any],
    label: str = "map",
    extra_nodata: tuple = (),
    nodata: int = NODATA
) -> np.ndarray:
    """Read either an Earth Engine or local raster onto the analysis grid and apply nodata masks."""
    arr = read_ee(src, grid, nodata=nodata) if is_ee_source(src) else read_local(src, grid, nodata=nodata)
    ids = to_ids(arr, label=label, nodata=nodata)
    for v in extra_nodata:
        ids[ids == v] = nodata
    return ids


def load_lookup_table(path: str) -> pd.DataFrame:
    """
    Load and validate a classification lookup CSV using LULC_Scheme_Manager.
    
    Returns DataFrame with columns ['id', 'name', 'color'].
    """
    df_raw = pd.read_csv(path)
    manager = LULC_Scheme_Manager()
    id_col, name_col, color_col = manager.auto_detect_csv_columns(df_raw)
    if not id_col or not name_col:
        raise ValueError(f"Lookup table needs recognizable ID and Name columns. Found: {list(df_raw.columns)}")
    
    ok, msg = manager.process_csv_upload(df_raw, id_col, name_col, color_col)
    if not ok:
        raise ValueError(f"Failed to process classification scheme from {path}: {msg}")
        
    manager.finalize_csv_upload()
    lk = manager.get_dataframe().rename(
        columns={"ID": "id", "Land Cover Class": "name", "Color Palette": "color"}
    )
    lk["id"] = lk["id"].astype(int)
    lk["name"] = lk["name"].astype(str).str.strip()
    if lk["id"].duplicated().any():
        raise ValueError(f"Duplicate IDs in lookup: {lk.loc[lk['id'].duplicated(), 'id'].tolist()}")
    if lk["name"].duplicated().any():
        raise ValueError(f"Duplicate names in lookup: {lk.loc[lk['name'].duplicated(), 'name'].tolist()}")
    return lk.sort_values("id").reset_index(drop=True)


def load_aoi_geometry(src: Union[str, gpd.GeoDataFrame]) -> gpd.GeoDataFrame:
    """
    Load and dissolve AOI into a single-row GeoDataFrame in EPSG:4326.
    Supports Earth Engine FeatureCollection assets, local shapefiles, GeoJSON, and GeoDataFrames.
    """
    if isinstance(src, gpd.GeoDataFrame):
        gdf = src
        if gdf.crs is None or gdf.crs != "EPSG:4326":
            gdf = gdf.to_crs("EPSG:4326")
    elif is_ee_source(src):
        ensure_ee_initialized()
        geom = shape(ee.FeatureCollection(src).geometry().getInfo())
        gdf = gpd.GeoDataFrame(geometry=[geom], crs="EPSG:4326")
    else:
        raw_gdf = gpd.read_file(src)
        validator = shapefile_validator(verbose=False)
        gdf = validator.validate_and_fix_geometry(raw_gdf, geometry="polygon")
        if gdf is None or len(gdf) == 0:
            raise ValueError(f"AOI geometry validation failed for {src}")
            
    geom = make_valid(unary_union(list(gdf.geometry)))
    return gpd.GeoDataFrame(geometry=[geom], crs="EPSG:4326")


def validate_and_align(
    t1_src: str,
    t2_src: str,
    aoi_src: Union[str, gpd.GeoDataFrame],
    lookup: pd.DataFrame,
    t1_year: int,
    t2_year: int,
    target_res_m: Optional[float] = None,
    max_res_ratio: float = 2.0,
    max_pixels: int = 150_000_000,
    extra_nodata: tuple = (),
    nodata: int = NODATA
) -> Dict[str, Any]:
    """
    Complete validation and alignment workflow for multitemporal LULC comparisons.
    """
    if not t1_year < t2_year:
        raise ValueError(f"t2 year ({t2_year}) must be later than t1 year ({t1_year})")

    aoi_ll = load_aoi_geometry(aoi_src)
    info_t1 = src_info(t1_src, aoi_ll)
    info_t2 = src_info(t2_src, aoi_ll)

    for label, inf in (("t1", info_t1), ("t2", info_t2)):
        if not inf["hits_aoi"]:
            raise ValueError(f"{label} map does not overlap the AOI")

    ratio = max(info_t1["res_m"], info_t2["res_m"]) / min(info_t1["res_m"], info_t2["res_m"])
    if ratio > max_res_ratio:
        raise ValueError(
            f"Resolutions differ too much ({info_t1['res_m']:.1f} m vs {info_t2['res_m']:.1f} m, "
            f"ratio {ratio:.1f} > {max_res_ratio})"
        )

    res = target_res_m or round(info_t1["res_m"], 1)
    grid = build_grid(aoi_ll, res, max_pixels=max_pixels)

    a1 = read_map(t1_src, grid, label="t1", extra_nodata=extra_nodata, nodata=nodata)
    a2 = read_map(t2_src, grid, label="t2", extra_nodata=extra_nodata, nodata=nodata)

    aoi_mask = rasterize(
        [(grid["aoi_utm"].geometry.iloc[0], 1)],
        out_shape=(grid["height"], grid["width"]),
        transform=grid["transform"],
        fill=0,
        dtype="uint8",
    ).astype(bool)

    if not aoi_mask.any():
        raise ValueError("AOI is smaller than one pixel on the grid")

    v1, v2 = a1 != nodata, a2 != nodata

    known_ids = set(lookup["id"])
    for label, arr, v in (("t1", a1, v1), ("t2", a2, v2)):
        missing = sorted(set(np.unique(arr[aoi_mask & v]).tolist()) - known_ids)
        if missing:
            raise ValueError(f"{label}: class IDs not in lookup table: {missing}")

    valid = aoi_mask & v1 & v2
    if not valid.any():
        raise ValueError("No pixels are valid in both maps inside the AOI")

    px_ha = res ** 2 / 10_000
    n_aoi = int(aoi_mask.sum())
    area_summary = pd.DataFrame(
        [
            ("AOI area, vector", grid["aoi_utm"].area.sum() / 10_000),
            ("AOI area, raster", n_aoi * px_ha),
            ("Included (valid in t1 and t2)", int(valid.sum()) * px_ha),
            ("Excluded: nodata in t1 only", int((aoi_mask & ~v1 & v2).sum()) * px_ha),
            ("Excluded: nodata in t2 only", int((aoi_mask & v1 & ~v2).sum()) * px_ha),
            ("Excluded: nodata in both", int((aoi_mask & ~v1 & ~v2).sum()) * px_ha),
            ("Excluded total", (n_aoi - int(valid.sum())) * px_ha),
        ],
        columns=["item", "area_ha"],
    )
    area_summary["pct_of_aoi_raster"] = np.where(
        area_summary["item"] == "AOI area, vector", np.nan, area_summary["area_ha"] / (n_aoi * px_ha) * 100
    )

    info = pd.DataFrame(
        [
            ("t1 source", t1_src), ("t1 year", t1_year),
            ("t2 source", t2_src), ("t2 year", t2_year),
            ("t1 native resolution (m)", round(info_t1["res_m"], 2)),
            ("t2 native resolution (m)", round(info_t2["res_m"], 2)),
            ("analysis CRS", grid["crs"].to_string()), ("analysis resolution (m)", res),
            ("grid size (cols x rows)", f"{grid['width']} x {grid['height']}"),
            ("resampling", "nearest neighbour"),
        ],
        columns=["item", "value"],
    )

    return dict(
        a1=a1,
        a2=a2,
        valid=valid,
        grid=grid,
        px_ha=px_ha,
        area_summary=area_summary,
        info=info,
        aoi_mask=aoi_mask
    )


def run_crosstab(
    aligned: Dict[str, Any],
    lookup: pd.DataFrame,
    t1_year: Optional[int] = None,
    t2_year: Optional[int] = None
) -> Dict[str, Any]:
    """
    Compute transition matrices and summary statistics between two classified time periods.
    """
    if t1_year is None or t2_year is None:
        info_dict = dict(zip(aligned["info"]["item"], aligned["info"]["value"]))
        y1 = t1_year if t1_year is not None else info_dict.get("t1 year", "t1")
        y2 = t2_year if t2_year is not None else info_dict.get("t2 year", "t2")
    else:
        y1, y2 = t1_year, t2_year

    ids = lookup["id"].to_numpy()
    names = lookup["name"].to_numpy()
    n = len(ids)
    a1, a2, valid = aligned["a1"], aligned["a2"], aligned["valid"]

    i1 = np.searchsorted(ids, a1[valid])
    i2 = np.searchsorted(ids, a2[valid])
    counts = np.bincount(i1 * n + i2, minlength=n * n).reshape(n, n)
    area = counts * aligned["px_ha"]
    total = area.sum()

    wide = pd.DataFrame(area, index=names, columns=names)
    wide["Total"] = wide.sum(axis=1)
    wide.loc["Total"] = wide.sum()
    wide.index.name = f"t1 {y1} (rows) / t2 {y2} (columns)"

    ii, jj = np.nonzero(counts)
    long = pd.DataFrame({
        "t1_id": ids[ii], "t1_class": names[ii], "t2_id": ids[jj], "t2_class": names[jj],
        "area_ha": area[ii, jj],
    })
    long["pct_of_included"] = long["area_ha"] / total * 100
    long["type"] = np.where(long["t1_id"] == long["t2_id"], "persistence", "change")
    long = long.sort_values("area_ha", ascending=False).reset_index(drop=True)

    rows, cols, diag = area.sum(axis=1), area.sum(axis=0), np.diag(area)
    area_t1 = pd.DataFrame({"id": ids, "class": names, "area_ha": rows, "pct": rows / total * 100})
    area_t2 = pd.DataFrame({"id": ids, "class": names, "area_ha": cols, "pct": cols / total * 100})
    gain_loss = pd.DataFrame({
        "id": ids, "class": names, "t1_ha": rows, "t2_ha": cols,
        "gain_ha": cols - diag, "loss_ha": rows - diag, "net_ha": cols - rows, "persistence_ha": diag,
    })

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(rows[:, None] > 0, area / rows[:, None], np.nan)
    wide_ratio = pd.DataFrame(ratio, index=names, columns=names)
    wide_ratio["Total"] = wide_ratio.sum(axis=1).where(rows > 0)
    wide_ratio.index.name = f"t1 {y1} (rows) / t2 {y2} (columns)"

    long_ratio = pd.DataFrame({
        "t1_id": ids[ii], "t1_class": names[ii], "t2_id": ids[jj], "t2_class": names[jj],
        "ratio": ratio[ii, jj],
    })
    long_ratio["type"] = np.where(long_ratio["t1_id"] == long_ratio["t2_id"], "persistence", "change")
    long_ratio = long_ratio.sort_values(["t1_id", "ratio"], ascending=[True, False]).reset_index(drop=True)

    top10 = long[long["type"] == "change"].head(10).reset_index(drop=True)
    return dict(
        wide=wide, long=long, wide_ratio=wide_ratio, long_ratio=long_ratio,
        area_t1=area_t1, area_t2=area_t2, gain_loss=gain_loss, top10=top10,
    )


def _map_axes(
    cat: np.ndarray,
    g: Dict[str, Any],
    colors: List[str],
    labels: List[str],
    title: str
) -> plt.Figure:
    """Internal plotting utility for categorical raster maps."""
    cmap = ListedColormap(colors)
    extent = (g["x0"], g["x0"] + g["width"] * g["res_m"], g["y1"] - g["height"] * g["res_m"], g["y1"])
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(cat, cmap=cmap, vmin=0, vmax=len(colors) - 1, extent=extent, interpolation="nearest")
    g["aoi_utm"].boundary.plot(ax=ax, color="black", lw=0.6)
    ax.legend(
        handles=[Patch(facecolor=c, edgecolor="grey", label=l) for c, l in zip(colors, labels)],
        loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8
    )
    ax.set_title(title)
    ax.set_xlabel(f"Easting (m), {g['crs'].to_string()}")
    ax.set_ylabel("Northing (m)")
    ax.set_aspect("equal")
    fig.tight_layout()
    return fig


def plot_gain_loss(
    gl: pd.DataFrame,
    t1_year: Optional[int] = None,
    t2_year: Optional[int] = None
) -> plt.Figure:
    """
    Plot horizontal bar chart of gain and loss per LULC class, including net change markers and area labels.
    """
    d = gl.iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(11, 0.5 * len(d) + 2.5))
    y_positions = np.arange(len(d))

    # Horizontal bars for Loss and Gain
    ax.barh(y_positions, -d["loss_ha"] / 1000, color="#d32f2f", label="Loss", height=0.6)
    ax.barh(y_positions, d["gain_ha"] / 1000, color="#2e7d32", label="Gain", height=0.6)

    # Net change marker dot with white border
    ax.scatter(d["net_ha"] / 1000, y_positions, color="black", s=45, zorder=5, label="Net (Gain - Loss)", edgecolors="white", linewidths=0.6)

    # Data value labels on bars
    for y, (gain, loss) in enumerate(zip(d["gain_ha"], d["loss_ha"])):
        if pd.notna(gain) and gain > 0:
            ax.text(gain / 1000, y, f" {gain:,.0f} ha", va="center", ha="left", fontsize=8, color="#1b5e20", fontweight="bold")
        if pd.notna(loss) and loss > 0:
            ax.text(-loss / 1000, y, f"{loss:,.0f} ha ", va="center", ha="right", fontsize=8, color="#b71c1c", fontweight="bold")

    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(d["class"])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{abs(v):,.0f}k"))
    ax.set_xlabel("Area ('000 ha; negative = loss, positive = gain)")

    # Expand x-limits to prevent labels from being cut off
    x_vals = np.concatenate([-d["loss_ha"] / 1000, d["gain_ha"] / 1000, d["net_ha"] / 1000])
    max_val = max(abs(x_vals.min()), abs(x_vals.max())) if len(x_vals) > 0 else 0.0
    x_span = max_val * 1.35 if max_val > 0 else 1.0
    ax.set_xlim(-x_span, x_span)

    t1_str = str(t1_year) if t1_year is not None else "t1"
    t2_str = str(t2_year) if t2_year is not None else "t2"
    ax.set_title(f"Gain and loss per class, {t1_str} to {t2_str}")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def plot_change_map(
    aligned: Dict[str, Any],
    t1_year: Optional[int] = None,
    t2_year: Optional[int] = None
) -> plt.Figure:
    """Plot binary map showing changed vs unchanged pixels inside AOI."""
    a1, a2, valid, g = aligned["a1"], aligned["a2"], aligned["valid"], aligned["grid"]
    cat = np.zeros(a1.shape, np.uint8)
    cat[valid] = 1
    cat[valid & (a1 != a2)] = 2
    if t1_year is None or t2_year is None:
        info_dict = dict(zip(aligned["info"]["item"], aligned["info"]["value"]))
        t1_year = info_dict.get("t1 year", "t1")
        t2_year = info_dict.get("t2 year", "t2")
    return _map_axes(
        cat, g, ["#ffffff", "#d9d9d9", "#e65100"], ["Excluded", "Unchanged", "Changed"],
        f"Changed vs unchanged, {t1_year} to {t2_year}",
    )


def plot_class_map(
    aligned: Dict[str, Any],
    class_id: int,
    class_name: str,
    t1_year: Optional[int] = None,
    t2_year: Optional[int] = None
) -> plt.Figure:
    """Plot persistence, gain, and loss map for a specific target class."""
    a1, a2, valid, g = aligned["a1"], aligned["a2"], aligned["valid"], aligned["grid"]
    in1, in2 = a1 == class_id, a2 == class_id
    cat = np.zeros(a1.shape, np.uint8)
    cat[valid] = 1
    cat[valid & in1 & in2] = 2
    cat[valid & ~in1 & in2] = 3
    cat[valid & in1 & ~in2] = 4
    if t1_year is None or t2_year is None:
        info_dict = dict(zip(aligned["info"]["item"], aligned["info"]["value"]))
        t1_year = info_dict.get("t1 year", "t1")
        t2_year = info_dict.get("t2 year", "t2")
    return _map_axes(
        cat, g, ["#ffffff", "#efefef", "#7a9cc6", "#2e7d32", "#c62828"],
        ["Excluded", "Other classes", "No change", "Gain", "Loss"],
        f"{class_name}: gain / loss / no change, {t1_year} to {t2_year}",
    )


def plot_top_changes(
    top: pd.DataFrame,
    lookup: pd.DataFrame,
    t1_year: Optional[int] = None,
    t2_year: Optional[int] = None
) -> plt.Figure:
    """Plot bar chart of top spatial transition pairs."""
    colors = dict(zip(lookup["id"], lookup["color"]))
    d = top.iloc[::-1].reset_index(drop=True)
    bar_colors = [
        c if isinstance(c, str) and c.startswith("#") else "#546e7a"
        for c in (colors.get(i) for i in d["t2_id"])
    ]
    labels = [f"{a}  →  {b}" for a, b in zip(d["t1_class"], d["t2_class"])]
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(d) + 2))
    ax.barh(labels, d["area_ha"], color=bar_colors)
    for y, v in enumerate(d["area_ha"]):
        ax.text(v, y, f" {v:,.0f} ha", va="center", fontsize=8)
    if len(d["area_ha"]) > 0 and max(d["area_ha"]) > 0:
        ax.set_xlim(0, max(d["area_ha"]) * 1.25)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_xlabel("Area (ha)")
    t1_str = str(t1_year) if t1_year is not None else "t1"
    t2_str = str(t2_year) if t2_year is not None else "t2"
    ax.set_title(f"Top 10 LULC changes, {t1_str} to {t2_str} (bar colour = new class)")
    fig.tight_layout()
    return fig


def save_outputs(
    out_dir: str,
    aligned: Dict[str, Any],
    tables: Dict[str, Any],
    figs: Dict[str, plt.Figure],
    t1_year: Optional[int] = None,
    t2_year: Optional[int] = None
) -> str:
    """
    Save all cross-tabulation tables to a multi-sheet Excel workbook and export figures as PNGs.
    """
    os.makedirs(out_dir, exist_ok=True)
    if t1_year is None or t2_year is None:
        info_dict = dict(zip(aligned["info"]["item"], aligned["info"]["value"]))
        t1_year = info_dict.get("t1 year", "t1")
        t2_year = info_dict.get("t2 year", "t2")
    xlsx = os.path.join(out_dir, f"lulc_crosstab_{t1_year}_{t2_year}.xlsx")
    sheets = {
        "info": (aligned["info"], False, 2),
        "area_summary": (aligned["area_summary"], False, 2),
        "crosstab_wide": (tables["wide"], True, 2),
        "crosstab_long": (tables["long"], False, 2),
        "crosstab_wide_ratio": (tables["wide_ratio"], True, 4),
        "crosstab_long_ratio": (tables["long_ratio"], False, 4),
        "area_t1": (tables["area_t1"], False, 2),
        "area_t2": (tables["area_t2"], False, 2),
        "gain_loss": (tables["gain_loss"], False, 2),
        "top10_changes": (tables["top10"], False, 2),
    }
    with pd.ExcelWriter(xlsx, engine="openpyxl") as xw:
        for name, (df, keep_index, decimals) in sheets.items():
            df.round(decimals).to_excel(xw, sheet_name=name, index=keep_index)
        for ws in xw.book.worksheets:
            for col in ws.columns:
                width = max(len(str(c.value)) if c.value is not None else 0 for c in col)
                ws.column_dimensions[col[0].column_letter].width = min(width + 2, 60)
    for name, fig in figs.items():
        fig.savefig(os.path.join(out_dir, f"{name}.png"), dpi=200, bbox_inches="tight")
    return xlsx


class LULCChangeDetector:
    """
    Multitemporal Land Use / Land Cover (LULC) Change Detection and Crosstabulation Engine.

    Encapsulates input validation, raster alignment, matrix computation,
    visualization, and reporting with production-ready defaults.
    """
    def __init__(
        self,
        t1_src: str,
        t2_src: str,
        aoi_src: Union[str, gpd.GeoDataFrame],
        lookup_csv: Union[str, pd.DataFrame],
        t1_year: int,
        t2_year: int,
        out_dir: Optional[str] = None,
        focus_class_id: Any = "auto",
        target_res_m: Optional[float] = None,
        max_res_ratio: float = 2.0,
        max_pixels: int = 150_000_000,
        extra_nodata: tuple = (),
        nodata: int = NODATA,
        m_per_deg: float = M_PER_DEG,
        ee_project: Optional[str] = None,
    ):
        self.t1_src = t1_src
        self.t2_src = t2_src
        self.aoi_src = aoi_src
        self.lookup_csv = lookup_csv
        self.t1_year = t1_year
        self.t2_year = t2_year
        self.out_dir = out_dir
        self.focus_class_id = focus_class_id
        self.target_res_m = target_res_m
        self.max_res_ratio = max_res_ratio
        self.max_pixels = max_pixels
        self.extra_nodata = extra_nodata
        self.nodata = nodata
        self.m_per_deg = m_per_deg
        self.ee_project = ee_project

        self.lookup: Optional[pd.DataFrame] = None
        self.aoi_ll: Optional[gpd.GeoDataFrame] = None
        self.info_t1: Optional[Dict[str, Any]] = None
        self.info_t2: Optional[Dict[str, Any]] = None
        self.aligned: Optional[Dict[str, Any]] = None
        self.tables: Optional[Dict[str, Any]] = None
        self.figures: Dict[str, plt.Figure] = {}

    def load_inputs(self) -> Dict[str, Any]:
        """
        Load classification scheme and dissolved AOI, and inspect source metadata.
        """
        if isinstance(self.lookup_csv, pd.DataFrame):
            self.lookup = self.lookup_csv
        else:
            self.lookup = load_lookup_table(self.lookup_csv)

        if isinstance(self.aoi_src, gpd.GeoDataFrame):
            self.aoi_ll = self.aoi_src
        else:
            self.aoi_ll = load_aoi_geometry(self.aoi_src)

        self.info_t1 = src_info(self.t1_src, self.aoi_ll, m_per_deg=self.m_per_deg)
        self.info_t2 = src_info(self.t2_src, self.aoi_ll, m_per_deg=self.m_per_deg)
        return dict(lookup=self.lookup, aoi=self.aoi_ll, info_t1=self.info_t1, info_t2=self.info_t2)

    def validate_multitemporal_inputs(self) -> Dict[str, Any]:
        """Validate input coverage, temporal order, and resolutions."""
        if self.aoi_ll is None:
            self.load_inputs()
        if not self.t1_year < self.t2_year:
            raise ValueError(f"t2 year ({self.t2_year}) must be later than t1 year ({self.t1_year})")
        for label, inf in (("t1", self.info_t1), ("t2", self.info_t2)):
            if not inf["hits_aoi"]:
                raise ValueError(f"{label} map does not overlap the AOI")
        ratio = max(self.info_t1["res_m"], self.info_t2["res_m"]) / min(self.info_t1["res_m"], self.info_t2["res_m"])
        if ratio > self.max_res_ratio:
            raise ValueError(
                f"Resolutions differ too much ({self.info_t1['res_m']:.1f} m vs {self.info_t2['res_m']:.1f} m, "
                f"ratio {ratio:.1f} > {self.max_res_ratio})"
            )
        return dict(info_t1=self.info_t1, info_t2=self.info_t2)

    def align(self) -> Dict[str, Any]:
        """Align both rasters to the common target UTM grid and validate class IDs."""
        if self.lookup is None or self.aoi_ll is None:
            self.load_inputs()
        self.aligned = validate_and_align(
            t1_src=self.t1_src,
            t2_src=self.t2_src,
            aoi_src=self.aoi_ll,
            lookup=self.lookup,
            t1_year=self.t1_year,
            t2_year=self.t2_year,
            target_res_m=self.target_res_m,
            max_res_ratio=self.max_res_ratio,
            max_pixels=self.max_pixels,
            extra_nodata=self.extra_nodata,
            nodata=self.nodata
        )
        return self.aligned

    def validate_and_align(self) -> Dict[str, Any]:
        """Alias for align()."""
        return self.align()

    def compute_spatial_transitions(self) -> Dict[str, Any]:
        """Compute transition matrix and change tables."""
        if self.aligned is None:
            self.align()
        self.tables = run_crosstab(self.aligned, self.lookup, self.t1_year, self.t2_year)
        return self.tables

    def plot_gain_loss(self) -> plt.Figure:
        """Plot horizontal gain and loss bar chart with Net markers."""
        if self.tables is None:
            self.compute_spatial_transitions()
        return plot_gain_loss(self.tables["gain_loss"], self.t1_year, self.t2_year)

    def plot_change_map(self) -> plt.Figure:
        """Plot binary changed vs unchanged map."""
        if self.aligned is None:
            self.align()
        return plot_change_map(self.aligned, self.t1_year, self.t2_year)

    def plot_class_map(self, class_id: int, class_name: Optional[str] = None) -> plt.Figure:
        """Plot persistence/gain/loss map for a specific class ID."""
        if self.aligned is None:
            self.align()
        if class_name is None:
            matching = self.lookup.loc[self.lookup["id"] == class_id, "name"]
            class_name = matching.iloc[0] if len(matching) > 0 else f"Class {class_id}"
        return plot_class_map(self.aligned, class_id, class_name, self.t1_year, self.t2_year)

    def plot_top_changes(self) -> plt.Figure:
        """Plot top 10 spatial transition pairs bar chart."""
        if self.tables is None:
            self.compute_spatial_transitions()
        return plot_top_changes(self.tables["top10"], self.lookup, self.t1_year, self.t2_year)

    def plot_all(self, focus_class_id: Any = None) -> Dict[str, plt.Figure]:
        """Generate all figures (gain/loss, change map, class maps, top 10 changes)."""
        if self.tables is None:
            self.compute_spatial_transitions()

        target_fc = focus_class_id if focus_class_id is not None else self.focus_class_id
        gl = self.tables["gain_loss"]

        figs = {
            "gain_loss_chart": self.plot_gain_loss(),
            "change_map": self.plot_change_map(),
        }

        if target_fc is None:
            focus_ids = []
        elif isinstance(target_fc, str) and target_fc.lower() == "all":
            focus_ids = self.lookup["id"].tolist()
        elif isinstance(target_fc, str) and target_fc.lower() == "auto":
            focus_ids = [int(gl.loc[(gl["gain_ha"] + gl["loss_ha"]).idxmax(), "id"])]
        elif isinstance(target_fc, (list, tuple, set)):
            focus_ids = [int(i) for i in target_fc]
        else:
            focus_ids = [int(target_fc)]

        for cid in focus_ids:
            figs[f"class_map_{cid}"] = self.plot_class_map(cid)

        figs["top10_changes"] = self.plot_top_changes()
        self.figures = figs
        return figs

    def save(self, out_dir: Optional[str] = None) -> str:
        """Export all tables and figures to disk."""
        target_dir = out_dir or self.out_dir
        if not target_dir:
            raise ValueError("No out_dir specified to save outputs")
        if not self.figures:
            self.plot_all()
        return save_outputs(target_dir, self.aligned, self.tables, self.figures, self.t1_year, self.t2_year)

    def run(self, focus_class_id: Any = None, out_dir: Optional[str] = None) -> Dict[str, Any]:
        """Execute the full multitemporal analysis end-to-end."""
        self.load_inputs()
        self.align()
        self.compute_spatial_transitions()
        self.plot_all(focus_class_id=focus_class_id)
        saved_path = None
        if out_dir or self.out_dir:
            saved_path = self.save(out_dir=out_dir)
        return dict(
            aligned=self.aligned,
            tables=self.tables,
            figures=self.figures,
            saved_path=saved_path
        )
