import ee
import time
import geopandas as gpd
from shapely.geometry import shape
import geemap

# ee.Authenticate(force=True)
ee.Initialize(project='epistem2')

# ---------------- Config ----------------
VERSION = 'v6'
CLASS_PROPERTY = 'label'
N_TREES = 100
MIN_LEAF = 5
SEED = 42
TRAIN_RATIO = 0.4
PROBABILITY_SCALE = 100
EXPORT_SCALE = 100
ASSET_FOLDER = 'projects/epistem2/assets'

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_VECT_PATH = (
    BASE_DIR
    / "data"
    / "modular_mapping_approach"
    / "nusatenggara_test"
    / f"nusatenggara_td_DTL_result_{VERSION}.shp"
)

POLL_INTERVAL_SEC = 30


def wait_for_task(task, label=""):
    """Block until an EE batch task finishes. Raises if it fails/cancels."""
    while True:
        status = task.status()
        state = status.get('state')
        if state in ('COMPLETED', 'FAILED', 'CANCELLED'):
            if state != 'COMPLETED':
                raise RuntimeError(
                    f"Task '{label}' ended with state={state}: "
                    f"{status.get('error_message')}"
                )
            print(f"  ✓ {label} completed")
            return status
        print(f"  ... {label} state={state}, waiting {POLL_INTERVAL_SEC}s")
        time.sleep(POLL_INTERVAL_SEC)

def asset_exists(asset_id):
    """Check whether an EE asset already exists. Cheap metadata call,
    not a heavy interactive computation — safe to call every iteration."""
    try:
        ee.data.getAsset(asset_id)
        return True
    except ee.ee_exception.EEException:
        return False


def main():
    from luma_ge.classification_scheme import LULC_Scheme_Manager
    from luma_ge.classification import FeatureExtraction, Generate_LULC

    manager = LULC_Scheme_Manager()
    manager.load_default_scheme("Epistem")

    feature_extractor = FeatureExtraction()
    classifier = Generate_LULC()

    stacked_landsat = ee.Image(f'projects/epistem2/assets/stacked_landsat_2020_nusatenggara_{VERSION}')
    band_names = stacked_landsat.bandNames()

    provinces = ee.FeatureCollection('projects/epistem2/assets/AOI_NusaTenggara_Provinces')
    province_list = provinces.toList(provinces.size())
    n_provinces = 2 # jawabali number of provinces

    # Local shapefile load — cheap, not EE compute, safe to do once
    TrainData = gpd.read_file(TRAIN_VECT_PATH)
    if TrainData.crs is None:
        TrainData = TrainData.set_crs("EPSG:4326")

    for i in range(1, n_provinces):
        province = ee.Feature(province_list.get(i))
        province_name = province.get('AoI').getInfo()
        province_name_clean = (
            province_name.replace(' ', '_').replace('/', '_').replace('-', '_')
        )
        province_geom = province.geometry()

        print(f"\n=== Province {i + 1}/{n_provinces}: {province_name} ===")

        # --- Clip training points to this province (local geopandas op) ---
        province_shape = shape(province_geom.getInfo())
        province_gdf = gpd.GeoDataFrame(
            {"geometry": [province_shape]}, crs="EPSG:4326"
        ).to_crs(TrainData.crs)

        province_train_gdf = gpd.clip(TrainData, province_gdf)
        print(f"  Training points in province: {len(province_train_gdf)}")

        if len(province_train_gdf) == 0:
            print("  No training points here — skipping.")
            continue

        labeled_roi = geemap.gdf_to_ee(province_train_gdf)
        province_stack = stacked_landsat.clip(province_geom)

        # --- Feature extraction (expensive, spatial) ---
        stratified_train, _ = feature_extractor.stratified_split(
            labeled_roi, province_stack,
            class_prop=CLASS_PROPERTY, train_ratio=TRAIN_RATIO
        )
        input_props = band_names.add(CLASS_PROPERTY)
        stratified_train_clean = stratified_train.select(input_props)

        # clean band names
        stratified_train_clean = stratified_train_clean.map(
            lambda ft: ft.set(CLASS_PROPERTY, ee.Number(ft.get(CLASS_PROPERTY)).toInt())
        )
        raw_features = stratified_train_clean.getInfo()['features']

        raw_vals = [f['properties'][CLASS_PROPERTY] for f in raw_features]
        print(f"  {len(raw_vals)} rows, {len(set(raw_vals))} distinct label values: {sorted(set(raw_vals))}")

        clean_features = []
        for f in raw_features:
            props = dict(f['properties'])
            props[CLASS_PROPERTY] = int(props[CLASS_PROPERTY])  # force plain Python int
            clean_features.append(ee.Feature(None, props))

        stratified_train_clean = ee.FeatureCollection(clean_features)

        # # --- Export extracted table to its own province-scoped asset ---
        # train_asset_id = (
        #     f'{ASSET_FOLDER}/sumatra_stratified_train_{province_name_clean}_v5'
        # )
        # extraction_task = ee.batch.Export.table.toAsset(
        #     collection=stratified_train_clean,
        #     description=f'stratified_train_{province_name_clean}_v5',
        #     assetId=train_asset_id
        # )
        # extraction_task.start()
        # wait_for_task(extraction_task, label=f"extraction [{province_name}]")

        # --- Reload the now-materialized table — decoupled from the extraction graph ---
        # stratified_train_imported = ee.FeatureCollection(train_asset_id)

        # --- Train + classify (OVR soft classification) ---
        probability_stack = classifier.soft_classification(
            training_data=stratified_train_clean,
            class_property=CLASS_PROPERTY,
            image=province_stack,
            include_final_map=False,
            ntrees=N_TREES,
            v_split=None,
            min_leaf=MIN_LEAF,
            seed=SEED,
            probability_scale=PROBABILITY_SCALE
        )

        # --- Rename bands to prob_[class_name] ---
        class_ids_ordered_local = sorted({int(f['properties'][CLASS_PROPERTY]) for f in raw_features})
        n_classes = len(class_ids_ordered_local)
        print(f"  Classes for this province: {class_ids_ordered_local} ({n_classes})")

        class_ids_ordered = ee.List(class_ids_ordered_local)
        new_band_names = class_ids_ordered.map(
            lambda cid: ee.String('prob_').cat(ee.Number(cid).int().format())
        )
        probability_stack = probability_stack.rename(new_band_names)

        # --- Export probability stack; optionally wait before moving to the next province ---
        prob_asset_id = (
            f'{ASSET_FOLDER}/probability_stack_{province_name_clean}_2020_{VERSION}'
        )

        if asset_exists(prob_asset_id):
            print(f" Already exists at {prob_asset_id} — skipping")
            continue
    
        prob_task = ee.batch.Export.image.toAsset(
            image=probability_stack,
            description=f'probability_stack_{province_name_clean}_2020_{VERSION}',
            assetId=prob_asset_id,
            region=province_geom,
            scale=EXPORT_SCALE,
            maxPixels=1e13
        )
        prob_task.start()
        wait_for_task(prob_task, label=f"probability export [{province_name}]") # avoid computation time out error

        print(f" {province_name} done")

    print("\nAll provinces processed.")


if __name__ == "__main__":
    main()