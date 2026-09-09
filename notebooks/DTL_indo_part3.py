import ee
import time
import geopandas as gpd
from shapely.geometry import shape
import geemap

# ee.Authenticate(force=True)
ee.Initialize(project='epistem2')

# ---------------- Config ----------------
VERSION = 'v7'
CLASS_PROPERTY = 'label'
N_TREES = 100
MIN_LEAF = 5
SEED = 42
TRAIN_RATIO = 0.8
PROBABILITY_SCALE = 100
EXPORT_SCALE = 100
ASSET_FOLDER = 'projects/epistem2/assets'
REGION = 'Sumatra'

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_VECT_PATH = (
    BASE_DIR
    / "data"
    / "modular_mapping_approach"
    / f"{REGION.lower()}_test"
    / f"{REGION.lower()}_td_DTL_result_{VERSION}.shp"
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
    from luma_ge.classification import FeatureExtraction
    import pandas as pd
    from imblearn.under_sampling import RandomUnderSampler

    manager = LULC_Scheme_Manager()
    manager.load_default_scheme("Epistem")

    feature_extractor = FeatureExtraction()

    stacked_landsat = ee.Image(f'projects/epistem2/assets/stacked_landsat_2021_{REGION.lower()}_{VERSION}')
    band_names = stacked_landsat.bandNames()
    feature_names = band_names.getInfo()

    provinces = ee.FeatureCollection(f'projects/epistem2/assets/AOI_{REGION}_Provinces')
    province_list = provinces.toList(provinces.size())
    n_provinces = 10

    # Local shapefile load — cheap, not EE compute, safe to do once
    TrainData = gpd.read_file(TRAIN_VECT_PATH)
    if TrainData.crs is None:
        TrainData = TrainData.set_crs("EPSG:4326")

    for i in range(5, n_provinces):
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

        province_train_gdf = province_train_gdf[province_train_gdf["label"] != 0].copy()

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
        #     f'{ASSET_FOLDER}/sumatra_stratified_train_{province_name_clean}_{VERSION}'
        # )
        # extraction_task = ee.batch.Export.table.toAsset(
        #     collection=stratified_train_clean,
        #     description=f'stratified_train_{province_name_clean}_{VERSION}',
        #     assetId=train_asset_id
        # )
        # extraction_task.start()
        # wait_for_task(extraction_task, label=f"extraction [{province_name}]")

        # --- Reload the now-materialized table — decoupled from the extraction graph ---
        # stratified_train_imported = ee.FeatureCollection(train_asset_id)

        # --- Train + classify (balanced OVR) ---
        train_data = [f['properties'] for f in raw_features]
        train_df = pd.DataFrame(train_data)
        train_df = train_df.dropna(subset=feature_names + [CLASS_PROPERTY])

        prob_images = []
        class_list = sorted(train_df[CLASS_PROPERTY].unique().tolist())

        for class_id in class_list:

            # Create a one-vs-rest target for the current class.
            binary_target = (train_df[CLASS_PROPERTY] == class_id).astype(int)

            # Balance the positive and negative samples before sending them to EE.
            rus = RandomUnderSampler(random_state=42)
            X_resampled, y_resampled = rus.fit_resample(
                train_df[feature_names],
                binary_target
            )

            print(f"Class {class_id}: training points after undersampling = {len(X_resampled)}")
            print(f"Class non-{class_id}: training points after undersampling = {len(y_resampled)}")

            balanced_df = X_resampled.copy()
            balanced_df["binary"] = y_resampled.to_numpy()
            balanced_records = balanced_df.to_dict(orient="records")

            # Convert numpy scalar values to native Python values for EE serialization.
            balanced_features = [
                ee.Feature(
                    None,
                    {
                        name: value.item() if hasattr(value, "item") else value
                        for name, value in record.items()
                    }
                )
                for record in balanced_records
            ]
            balanced_training = ee.FeatureCollection(balanced_features)

            # Train a native Earth Engine classifier in probability output mode.
            ee_classifier = ee.Classifier.smileRandomForest(
                numberOfTrees=N_TREES,
                variablesPerSplit=1,
                minLeafPopulation=MIN_LEAF,
                seed=42
            ).setOutputMode("PROBABILITY").train(
                features=balanced_training,
                classProperty="binary",
                inputProperties=feature_names
            )

            probability_image = (
                province_stack
                .select(feature_names)
                .classify(ee_classifier)
                .multiply(100)  # Scale probabilities to 0-100 range
                .rename(f"prob_{class_id}")
            )
            prob_images.append(probability_image)

        # Stack one probability band for every class.
        probability_stack = prob_images[0]
        for probability_image in prob_images[1:]:
            probability_stack = probability_stack.addBands(probability_image)

        #  BAND RENAMING FIX
        all_band_names = probability_stack.bandNames().getInfo()
        class_ids_ordered_local = sorted({int(f['properties'][CLASS_PROPERTY]) for f in raw_features})
        expected_band_names = {f'prob_{c}' for c in class_ids_ordered_local}

        # Keep only bands that actually match an expected class band name

        stray_bands = [b for b in all_band_names if b not in expected_band_names]
        if stray_bands:
            print(f"  Dropping stray non-class band(s): {stray_bands}")
            probability_stack = probability_stack.select(
                probability_stack.bandNames().filter(
                    ee.Filter.inList('item', list(expected_band_names))
                )
            )

        kept_band_names = probability_stack.bandNames().getInfo()
        print(f"  probability_stack bands after cleanup: {kept_band_names}")
        print(f"  {len(class_ids_ordered_local)} distinct classes: {class_ids_ordered_local}")

        if set(kept_band_names) != expected_band_names:
            missing = expected_band_names - set(kept_band_names)
            unexpected = set(kept_band_names) - expected_band_names
            raise RuntimeError(
                f"Band/class mismatch for {region_name}/{province_name}: "
                f"missing={missing}, unexpected={unexpected}. "
                f"Aborting rather than exporting a potentially mislabeled stack."
            )

        # --- Export probability stack; optionally wait before moving to the next province ---
        prob_asset_id = (
            f'{ASSET_FOLDER}/probability_stack_{province_name_clean}_2021_{VERSION}'
        )

        if asset_exists(prob_asset_id):
            print(f" Already exists at {prob_asset_id} — skipping")
            continue

        prob_task = ee.batch.Export.image.toAsset(
            image=probability_stack,
            description=f'probability_stack_{province_name_clean}_2021_{VERSION}',
            assetId=prob_asset_id,
            region=province_geom,
            scale=EXPORT_SCALE,
            maxPixels=1e13
        )
        prob_task.start()
        # wait_for_task(prob_task, label=f"probability export [{province_name}]") # avoid computation time out error

        print(f" {province_name} done")

    print("\nAll provinces processed.")


if __name__ == "__main__":
    main()