import ee
import time
import geopandas as gpd
from shapely.geometry import shape
import geemap

# ee.Authenticate(force=True)
ee.Initialize(project='epistem2',
              opt_url='https://earthengine-highvolume.googleapis.com')

# ---------------- Config ----------------
VERSION = 'v8'
CLASS_PROPERTY = 'label'
N_TREES = 100
MIN_LEAF = 5
SEED = 42
TRAIN_RATIO = 0.6
EXPORT_SCALE = 100
ASSET_FOLDER = 'projects/epistem2/assets'
REGION = 'Sumatra'
YEAR = 2025

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

def print_separability_summary(sample_quality, spectral_df):
    """Concise terminal summary of class separability (TD)."""
    summary = sample_quality.sum_separability(spectral_df)
    if summary.empty:
        print("  TD: unavailable")
        return

    row = summary.iloc[0]
    text = (
        f"  TD: {int(row['Good Separability Pairs'])} good, "
        f"{int(row['Weak Separability Pairs'])} weak, "
        f"{int(row['Poor Separability Pairs'])} poor / "
        f"{int(row['Total Pairs'])} pairs"
    )

    lowest = sample_quality.lowest_separability(spectral_df, top_n=1)
    if not lowest.empty:
        w = lowest.iloc[0]
        text += f" | Worst: {w['Class1_Name']}–{w['Class2_Name']}={w['TD_Distance']:.3f}"

    print(text)


def print_model_evaluation(metrics):
    """Concise terminal output of model evaluation metrics."""
    f1 = metrics["f1_scores"]
    macro_f1 = sum(f1) / len(f1) if f1 else 0.0

    print(
        f"  Model: n={metrics.get('n_test', '?')} | "
        f"OA={metrics['overall_accuracy']:.3f} | "
        f"Kappa={metrics['kappa']:.3f} | "
        f"Macro-F1={macro_f1:.3f} | "
        f"GMean={metrics['overall_gmean']:.3f}"
    )


def main():
    from luma_ge.classification_scheme import LULC_Scheme_Manager
    from luma_ge.classification import FeatureExtraction, Generate_LULC
    import numpy as np
    from luma_ge.sample_data_quality import sample_quality
    import pandas as pd
    from imblearn.under_sampling import RandomUnderSampler

    manager = LULC_Scheme_Manager()
    manager.load_default_scheme("Epistem")

    feature_extractor = FeatureExtraction()

    stacked_landsat = ee.Image(f'projects/epistem2/assets/stacked_landsat_pred_{YEAR}_{REGION.lower()}_{VERSION}')
    band_names = stacked_landsat.bandNames()
    feature_names = band_names.getInfo()

    provinces = ee.FeatureCollection(f'projects/epistem2/assets/AOI_{REGION}_Provinces')
    province_list = provinces.toList(provinces.size())
    n_provinces = 10

    # Local shapefile load — cheap, not EE compute, safe to do once
    TrainData = gpd.read_file(TRAIN_VECT_PATH)
    if TrainData.crs is None:
        TrainData = TrainData.set_crs("EPSG:4326")

    evaluation_results = []

    for i in range(0, n_provinces):
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

        province_train_gdf = province_train_gdf[province_train_gdf["label"] != 0].copy() # remove unclassified points

        print(f"  Training points in province: {len(province_train_gdf)}")

        if len(province_train_gdf) == 0:
            print("  No training points here — skipping.")
            continue

        labeled_roi = geemap.gdf_to_ee(province_train_gdf)
        province_stack = stacked_landsat.clip(province_geom)

        # --- Feature extraction (expensive, spatial) ---
        print("Doing feature extraction and stratified train/test split...")

        stratified_train, stratified_test = feature_extractor.stratified_split(
            labeled_roi, province_stack,
            class_prop=CLASS_PROPERTY, train_ratio=TRAIN_RATIO
        )
        input_props = band_names.add(CLASS_PROPERTY)
        stratified_train_clean = stratified_train.select(input_props)
        stratified_test_clean = stratified_test.select(input_props)

        # clean band names
        stratified_train_clean = stratified_train_clean.map(
            lambda ft: ft.set(CLASS_PROPERTY, ee.Number(ft.get(CLASS_PROPERTY)).toInt())
        )
        stratified_test_clean = stratified_test_clean.map(
            lambda ft: ft.set(CLASS_PROPERTY, ee.Number(ft.get(CLASS_PROPERTY)).toInt())
        )

        raw_train_features = (stratified_train_clean
            .getInfo()['features'])
        raw_test_features = (stratified_test_clean
            .getInfo()['features'])

        # spectral_records = []

        # # --- Build spectral dataframe for separability analysis ---
        # for f in raw_train_features:
        #     props = dict(f['properties'])
        #     props[CLASS_PROPERTY] = int(props[CLASS_PROPERTY])
        #     spectral_records.append(props)

        # for f in raw_test_features:
        #     props = dict(f['properties'])
        #     props[CLASS_PROPERTY] = int(
        #         props[CLASS_PROPERTY])
        #     spectral_records.append(props)

        # spectral_df = pd.DataFrame(spectral_records)

        # spectral_df = spectral_df.dropna(
        #     subset=feature_names + [CLASS_PROPERTY])

        # if spectral_df.empty:
        #     print("  No spectral data — skipped.")
        #     continue

        # n_samples = len(spectral_df)

        # n_classes = spectral_df[CLASS_PROPERTY].nunique()

        # print(
        #     f"  Samples: {n_samples} | "
        #     f"Classes: {n_classes}"
        # )

        # # --- Separability training data quality analysis  ---
        # sq = sample_quality(
        #     training_data=labeled_roi,
        #     image=province_stack,
        #     class_property=CLASS_PROPERTY,
        #     region=province_geom
        # )

        # print_separability_summary(sq, spectral_df)

        # clean_features = []

        # for f in raw_train_features:
        #     props = dict(f['properties'])

        #     props[CLASS_PROPERTY] = int(props[CLASS_PROPERTY])

        #     clean_features.append(
        #         ee.Feature(
        #             None,
        #             props))

        # stratified_train_clean = (ee.FeatureCollection(clean_features))

        # --- Prep local training table ---
        train_data = [f['properties'] for f in raw_train_features]
        train_df = pd.DataFrame(train_data)
        train_df = train_df.dropna(subset=feature_names + [CLASS_PROPERTY])

        class_list = sorted(train_df[CLASS_PROPERTY].unique().tolist())
        print(f"  {len(class_list)} distinct classes: {class_list}")

        # --- Balance all classes together (multiclass undersampling to the
        #     minority class count), then train a single multiclass RF ---
        print("Balancing classes via undersampling...")

        counts = train_df[CLASS_PROPERTY].value_counts()
        cap = 100
        strategy = {c: min(n, cap) for c, n in counts.items()}
        rus = RandomUnderSampler(sampling_strategy=strategy, random_state=SEED)
        X_resampled, y_resampled = rus.fit_resample(
            train_df[feature_names],
            train_df[CLASS_PROPERTY]
        )

        print(f"  Training points after undersampling: {len(X_resampled)}")
        for class_id, count in y_resampled.value_counts().sort_index().items():
            print(f"    class {class_id}: {count}")

        balanced_df = X_resampled.copy()
        balanced_df[CLASS_PROPERTY] = y_resampled.to_numpy()
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

        # Train a native Earth Engine multiclass classifier in classification
        # (predicted-label) output mode — no more one-vs-rest.
        ee_classifier = ee.Classifier.smileRandomForest(
            numberOfTrees=N_TREES,
            variablesPerSplit=None,
            minLeafPopulation=MIN_LEAF,
            seed=SEED
        ).setOutputMode("CLASSIFICATION").train(
            features=balanced_training,
            classProperty=CLASS_PROPERTY,
            inputProperties=feature_names
        )

        # --- Evaluate model on the test set ---
        print("Evaluating model on the test set...")
        classifier = Generate_LULC()

        metrics = classifier.evaluate_model(
            trained_model=ee_classifier,
            test_data=stratified_test_clean,
            class_property=CLASS_PROPERTY
        )

        # metrics["n_test"] = stratified_test_clean.size().getInfo()

        print_model_evaluation(metrics)

        evaluation_results.append({
            "province": province_name,
            "n_training": len(
                balanced_df
            ),
            # "n_test": metrics["n_test"],
            "accuracy": metrics[
                "overall_accuracy"
            ],
            "kappa": metrics[
                "kappa"
            ],
            "macro_f1": (
                sum(metrics["f1_scores"])
                / len(metrics["f1_scores"])
                if metrics["f1_scores"]
                else 0
            ),
            "gmean": metrics[
                "overall_gmean"
            ]
        })

        classified_image = (
            province_stack
            .select(feature_names)
            .classify(ee_classifier)
            .rename('classification')
        )

        # --- Export classified map; optionally wait before moving to the next province ---
        classification_asset_id = (
            f'{ASSET_FOLDER}/final_lulc_{province_name_clean}_{YEAR}_{VERSION}_hardclassification'
        )

        if asset_exists(classification_asset_id):
            print(f" Already exists at {classification_asset_id} — skipping")
            continue

        classify_task = ee.batch.Export.image.toAsset(
            image=classified_image,
            description=f'final_lulc_{province_name_clean}_{YEAR}_{VERSION}_hardclassification',
            assetId=classification_asset_id,
            region=province_geom,
            scale=EXPORT_SCALE,
            maxPixels=1e13
        )
        classify_task.start()
        # wait_for_task(classify_task, label=f"classification export [{province_name}]") # avoid computation time out error
        
        pd.DataFrame(evaluation_results).to_csv(BASE_DIR / "data" / f"{REGION.lower()}_eval_{VERSION}.csv", index=False)

        print(f" {province_name} done")


    print("\nAll provinces processed.")


if __name__ == "__main__":
    main()