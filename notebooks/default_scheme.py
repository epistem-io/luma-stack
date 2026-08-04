import pandas as pd
import numpy as np
import rasterio
from rasterio.mask import mask
import geopandas as gpd


def read_mosaic_rast(mosaic_rast, aoi_gdf=None):
    """
    Load a multi-band raster, optionally clip to an AOI polygon.

    Parameters
    ----------
    mosaic_rast : str
        Path to the multi-band raster file (e.g., GeoTIFF).
    aoi_gdf : geopandas.GeoDataFrame, optional
        GeoDataFrame with polygon geometries for clipping. If None (default),
        the entire raster is returned.

    Returns
    -------
    clipped_data : numpy.ndarray
        Raster data array (bands, height, width) – clipped or full.
    out_transform : affine.Affine
        Affine transform for the returned raster.
    meta : dict
        Raster metadata (updated if clipped, original if full).
    band_names : list of str
        List of band names extracted from the source descriptions.
    """
    with rasterio.open(mosaic_rast) as src:
        descriptions = src.descriptions
        
        if descriptions and any(desc for desc in descriptions):
            band_names = [desc if desc else f'band_{i+1}' for i, desc in enumerate(descriptions)]
        else:
            band_names = [f'band_{i+1}' for i in range(src.count)]

        if aoi_gdf is not None:
            if aoi_gdf.crs != src.crs:
                aoi_gdf_proj = aoi_gdf.to_crs(src.crs)
            else:
                aoi_gdf_proj = aoi_gdf

            shapes = aoi_gdf_proj.geometry.values
            raster_data, out_transform = mask(src, shapes, crop=True)

            meta = src.meta.copy()
            meta.update({
                "height": raster_data.shape[1],
                "width": raster_data.shape[2],
                "transform": out_transform
            })
        else:
            raster_data = src.read()
            out_transform = src.transform
            meta = src.meta.copy() 

    return raster_data, out_transform, meta, band_names


def read_classification_scheme(classification_df):
    """
    Read a hierarchical classification scheme.

    Parameters
    ----------
    classification_df : DataFrame

    Required columns
    ----------------
    ID
    Level3_Name
    Level2_Name
    Level1_Name
    Terminal_Level   (0 = exclude this class, 1/2/3 = terminal levels)
    Color (optional)

    Returns
    -------
    scheme : dict
    """
    df = classification_df.copy()
    required = ["ID", "Level3_Name", "Level2_Name", "Level1_Name", "Terminal_Level"]
    
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[df["Terminal_Level"] != 0].copy()

    if df.empty:
        raise ValueError("No classes remain after excluding Terminal_Level == 0.")

    if not df["Terminal_Level"].isin([1, 2, 3]).all():
        raise ValueError("Terminal_Level must contain only 1, 2 or 3.")

    if df["ID"].duplicated().any():
        raise ValueError("Duplicated ID detected.")

    level2_table = df[["Level2_Name"]].drop_duplicates().reset_index(drop=True)
    level2_table["Level2_ID"] = level2_table.index + 1

    level1_table = df[["Level1_Name"]].drop_duplicates().reset_index(drop=True)
    level1_table["Level1_ID"] = level1_table.index + 1

    df = df.merge(level2_table, on="Level2_Name", how="left")
    df = df.merge(level1_table, on="Level1_Name", how="left")

    level2_table = df[["Level2_ID", "Level2_Name", "Level1_ID"]].drop_duplicates().sort_values("Level2_ID")
    level1_table = df[["Level1_ID", "Level1_Name"]].drop_duplicates().sort_values("Level1_ID")

    scheme = {
        "table": df,
        "level2_table": level2_table,
        "level1_table": level1_table,
        "terminal_level": df.set_index("ID")["Terminal_Level"].to_dict(),
        "level3_to_level2": df.set_index("ID")["Level2_ID"].to_dict(),
        "level2_to_level1": level2_table.set_index("Level2_ID")["Level1_ID"].to_dict()
    }

    return scheme


def build_mapping(scheme):
    """
    Build aggregation mapping from the classification scheme.

    Parameters
    ----------
    scheme : dict
        Output from read_classification_scheme().

    Returns
    -------
    mapping : dict
    """
    df = scheme["table"].copy()
    df["Band_Index"] = df["ID"] - 1

    output_lookup = {}
    output_id = 1

    for _, row in df[df.Terminal_Level == 3].iterrows():
        output_lookup[output_id] = {
            "output_id": output_id,
            "output_name": row.Level3_Name,
            "output_level": 3,
            "color": row.Color,
            "member_ids": [row.ID],
            "member_band_indices": [row.Band_Index]
        }
        output_id += 1

    level2 = df[df.Terminal_Level == 2].groupby("Level2_Name")
    for level2_name, group in level2:
        output_lookup[output_id] = {
            "output_id": output_id,
            "output_name": level2_name,
            "output_level": 2,
            "color": group.Color.iloc[0],
            "member_ids": group.ID.tolist(),
            "member_band_indices": group.Band_Index.tolist()
        }
        output_id += 1

    level1 = df[df.Terminal_Level == 1].groupby("Level1_Name")
    for level1_name, group in level1:
        output_lookup[output_id] = {
            "output_id": output_id,
            "output_name": level1_name,
            "output_level": 1,
            "color": group.Color.iloc[0],
            "member_ids": group.ID.tolist(),
            "member_band_indices": group.Band_Index.tolist()
        }
        output_id += 1

    output_table = pd.DataFrame([{
        "Output_ID": x["output_id"],
        "Output_Name": x["output_name"],
        "Output_Level": x["output_level"],
        "Color": x["color"],
        "Member_IDs": x["member_ids"]
    } for x in output_lookup.values()])

    mapping = {
        "output_lookup": output_lookup,
        "output_table": output_table
    }

    return mapping


def aggregate_probability(raster_data, mapping):
    """
    Aggregate a Level-3 probability cube into the selected hierarchy.

    Parameters
    ----------
    raster_data : np.ndarray
        Multi-probability raster with shape (n_bands, n_rows, n_cols).
    mapping : dict
        Output from build_mapping().

    Returns
    -------
    aggregated_cube : np.ndarray
        Aggregated probability cube.
    output_info : pandas.DataFrame
        Information for each output band.
    """
    output_lookup = mapping["output_lookup"]
    n_output = len(output_lookup)
    n_rows = raster_data.shape[1]
    n_cols = raster_data.shape[2]

    aggregated_cube = np.zeros((n_output, n_rows, n_cols), dtype=raster_data.dtype)
    summary = []

    for i, info in enumerate(output_lookup.values()):
        valid_band_indices = [
            idx for idx in info["member_band_indices"]
            if 0 <= idx < raster_data.shape[0]
        ]
        missing_band_indices = sorted(set(info["member_band_indices"]) - set(valid_band_indices))

        if missing_band_indices:
            print(f"Warning: '{info['output_name']}' ignored missing raster band(s): {missing_band_indices}")

        if len(valid_band_indices) > 0:
            aggregated_cube[i] = np.sum(raster_data[valid_band_indices], axis=0)
        else:
            aggregated_cube[i] = np.zeros((n_rows, n_cols), dtype=raster_data.dtype)

        summary.append({
            "Output_ID": info["output_id"],
            "Output_Name": info["output_name"],
            "Output_Level": info["output_level"],
            "Color": info["color"],
            "Member_IDs": info["member_ids"],
            "Member_Bands": valid_band_indices
        })

    output_info = pd.DataFrame(summary)

    return aggregated_cube, output_info


def predict_class(aggregated_cube, output_info, raster_meta):
    """
    Predict land-cover class using argmax on the aggregated probability cube.

    Parameters
    ----------
    aggregated_cube : np.ndarray
        Output from aggregate_probability(). Shape = (n_classes, rows, cols)
    output_info : pandas.DataFrame
        Output from aggregate_probability().
    raster_meta : dict
        Metadata copied from the input probability raster.

    Returns
    -------
    classified_map : np.ndarray
        Integer classified raster.
    classified_meta : dict
        Raster metadata for writing.
    max_prob_map : np.ndarray
        Maximum probability for each pixel.
    argmax_indices : np.ndarray
        Index of the winning aggregated class.
    selected_class_ids : list
        Class IDs corresponding to each probability band.
    """
    if aggregated_cube.ndim != 3:
        raise ValueError("aggregated_cube must have shape (bands, rows, cols).")

    selected_class_ids = output_info["Output_ID"].tolist()

    if len(selected_class_ids) != aggregated_cube.shape[0]:
        print("Warning: output_info does not match aggregated probability cube.")
        selected_class_ids = list(range(1, aggregated_cube.shape[0] + 1))

    argmax_indices = np.argmax(aggregated_cube, axis=0)
    max_prob_map = np.max(aggregated_cube, axis=0)

    classified_map = np.array(
        [selected_class_ids[idx] for idx in argmax_indices.flat],
        dtype=np.int16
    ).reshape(argmax_indices.shape)

    classified_map[max_prob_map == 0] = 0

    classified_meta = raster_meta.copy()
    classified_meta.update({
        "count": 1,
        "dtype": "int16"
    })

    print("Argmax classification completed.")
    print("Output shape:", classified_map.shape)

    return classified_map, classified_meta, max_prob_map, argmax_indices, selected_class_ids