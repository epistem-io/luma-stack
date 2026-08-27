# ============================================================
# DTL: Thematic validation
# Batch validation of final LULC maps by province
# ============================================================


# ============================================================
# A. Imports
# ============================================================

import ee
import time
import pandas as pd

from luma_ge.accuracy import thematic_accuracy
from luma_ge.ee_config import initialize_earth_engine


# ============================================================
# B. Configuration
# ============================================================

VERSION = 'v6'

POLL_INTERVAL_SEC = 30

EXPORT_FOLDER = 'GEE_Validation'
EXPORT_SCALE = 100

ASSET_FOLDER = 'projects/epistem2/assets'

PROVINCE_ASSET = (
    'projects/epistem2/assets/AOI_Sumatra_Provinces'
)

VALIDATION_ASSET = (
    'projects/epistem2/assets/Sumatra_Validation_Points'
)

# Reference class ID field
VALIDATION_CLASS_PROPERTY = 'IDe'

# Reference class name field
VALIDATION_NAME_PROPERTY = 'LULC20e'

# Existing GEE validation assets will not be overwritten.
SKIP_EXISTING_EXPORTS = True


# ============================================================
# C. Earth Engine initialization
# ============================================================

initialize_earth_engine(
    project='epistem-lumastack',
    force_reinit=True
)

# Explicitly use the intended Cloud API user project.
ee.data.setCloudApiUserProject(
    'epistem-lumastack'
)


# ============================================================
# D. LULC class definitions
# ============================================================

CLASS_NAMES = {
    1:  'Primary Dryland Forest',
    2:  'Secondary Dryland Forest',
    3:  'Primary Mangrove Forest',
    4:  'Secondary Mangrove Forest',
    5:  'Primary Swamp Forest',
    6:  'Secondary Swamp Forest',
    7:  'Plantation Forest',
    8:  'Rubber Monoculture',
    9:  'Oil palm Monoculture',
    10: 'Cacao Monoculture',
    11: 'Coconut monoculture',
    12: 'Other Monoculture',
    13: 'Other Cropland',
    14: 'Coffee agroforestry',
    15: 'Rubber agroforestry',
    16: 'Mixed/home garden',
    17: 'Paddy field',
    18: 'Grass or Savanna',
    19: 'Shrub',
    20: 'Settlement',
    21: 'Cleared Land',
    22: 'Mining area',
    23: 'Waterbody',
    24: 'Fish pond',
}

CLASS_IDS = list(CLASS_NAMES.keys())


# ============================================================
# E. Helper functions
# ============================================================

def wait_for_task(task, label=''):
    """
    Wait until an Earth Engine batch task finishes.
    """

    while True:

        status = task.status()
        state = status.get('state')

        if state in (
            'COMPLETED',
            'FAILED',
            'CANCELLED'
        ):

            if state != 'COMPLETED':

                raise RuntimeError(
                    f"Task '{label}' ended with "
                    f"state={state}: "
                    f"{status.get('error_message')}"
                )

            print(
                f"  [OK] {label} completed"
            )

            return status

        print(
            f"  ... {label} "
            f"state={state}, "
            f"waiting {POLL_INTERVAL_SEC}s"
        )

        time.sleep(
            POLL_INTERVAL_SEC
        )


def asset_exists(asset_id):
    """
    Check whether an Earth Engine asset exists.
    """

    try:

        ee.data.getAsset(asset_id)
        return True

    except ee.ee_exception.EEException:

        return False


def clean_province_name(province_name):
    """
    Convert province name to an asset-safe name.
    """

    return (
        province_name
        .replace(' ', '_')
        .replace('/', '_')
        .replace('-', '_')
    )


def add_correctness(feature):
    """
    Add numeric correctness flag.

    correct = 1 -> correct prediction
    correct = 0 -> incorrect prediction
    """

    reference = ee.Number(
        feature.get(
            VALIDATION_CLASS_PROPERTY
        )
    )

    predicted = ee.Number(
        feature.get('classification')
    )

    correct = (
        predicted
        .eq(reference)
        .int()
    )

    return feature.set(
        'correct',
        correct
    )


# ============================================================
# F. Build class-level summary
# ============================================================

def build_class_summary(
    validation_fc,
    province_name,
):
    """
    Calculate class-level accuracy metrics directly
    from sampled validation points.

    Metrics:
        reference_n
        predicted_n
        correct_n
        producer_accuracy
        user_accuracy
        f1_score
        omission_error
        commission_error
    """

    rows = []

    for class_id in CLASS_IDS:

        # ----------------------------------------------------
        # Reference sample count
        # ----------------------------------------------------

        reference_n = (
            validation_fc
            .filter(
                ee.Filter.eq(
                    VALIDATION_CLASS_PROPERTY,
                    class_id
                )
            )
            .size()
            .getInfo()
        )


        # ----------------------------------------------------
        # Prediction sample count
        # ----------------------------------------------------

        predicted_n = (
            validation_fc
            .filter(
                ee.Filter.eq(
                    'classification',
                    class_id
                )
            )
            .size()
            .getInfo()
        )


        # ----------------------------------------------------
        # Correct sample count
        # ----------------------------------------------------

        correct_n = (
            validation_fc
            .filter(
                ee.Filter.And(
                    ee.Filter.eq(
                        VALIDATION_CLASS_PROPERTY,
                        class_id
                    ),
                    ee.Filter.eq(
                        'classification',
                        class_id
                    )
                )
            )
            .size()
            .getInfo()
        )


        # ----------------------------------------------------
        # Producer accuracy
        # ----------------------------------------------------

        if reference_n > 0:

            producer_accuracy = (
                correct_n
                / reference_n
            )

        else:

            producer_accuracy = None


        # ----------------------------------------------------
        # User accuracy
        # ----------------------------------------------------

        if predicted_n > 0:

            user_accuracy = (
                correct_n
                / predicted_n
            )

        else:

            user_accuracy = None


        # ----------------------------------------------------
        # F1 score
        # ----------------------------------------------------

        if (
            producer_accuracy is not None
            and user_accuracy is not None
            and (
                producer_accuracy
                + user_accuracy
            ) > 0
        ):

            f1_score = (
                2
                * producer_accuracy
                * user_accuracy
                / (
                    producer_accuracy
                    + user_accuracy
                )
            )

        else:

            f1_score = None


        # ----------------------------------------------------
        # Omission error
        # ----------------------------------------------------

        if producer_accuracy is not None:

            omission_error = (
                1
                - producer_accuracy
            )

        else:

            omission_error = None


        # ----------------------------------------------------
        # Commission error
        # ----------------------------------------------------

        if user_accuracy is not None:

            commission_error = (
                1
                - user_accuracy
            )

        else:

            commission_error = None


        rows.append({

            'province': (
                province_name
            ),

            'class_id': (
                class_id
            ),

            'class_name': (
                CLASS_NAMES[class_id]
            ),

            'reference_n': (
                reference_n
            ),

            'predicted_n': (
                predicted_n
            ),

            'correct_n': (
                correct_n
            ),

            'producer_accuracy': (
                producer_accuracy
            ),

            'user_accuracy': (
                user_accuracy
            ),

            'f1_score': (
                f1_score
            ),

            'omission_error': (
                omission_error
            ),

            'commission_error': (
                commission_error
            ),
        })


    return pd.DataFrame(
        rows
    )


# ============================================================
# G. Build prediction inventory
# ============================================================

def build_prediction_inventory(
    validation_fc,
    province_name,
):
    """
    Count predicted classes in sampled validation points.

    This is used to verify whether all predictions fall
    within the expected 1-24 class domain.
    """

    histogram = (
        validation_fc
        .aggregate_histogram(
            'classification'
        )
        .getInfo()
    )

    rows = []

    for class_value, count in sorted(
        histogram.items(),
        key=lambda x: int(x[0])
    ):

        class_id = int(
            class_value
        )

        if class_id in CLASS_NAMES:

            class_name = (
                CLASS_NAMES[class_id]
            )

        else:

            class_name = (
                f'Outside expected '
                f'class domain'
            )

        rows.append({

            'province': (
                province_name
            ),

            'classification': (
                class_id
            ),

            'class_name': (
                class_name
            ),

            'prediction_n': (
                count
            ),

            'expected_class': (
                class_id in CLASS_NAMES
            ),
        })


    return pd.DataFrame(
        rows
    )


# ============================================================
# H. Build confusion matrix
# ============================================================

def build_confusion_matrix(
    validation_fc,
    province_name,
):
    """
    Build a 24 x 24 confusion matrix using
    reference classes 1-24 and predicted classes 1-24.
    """

    matrix = (
        validation_fc
        .errorMatrix(
            VALIDATION_CLASS_PROPERTY,
            'classification',
            CLASS_IDS
        )
        .getInfo()
    )


    matrix_df = pd.DataFrame(
        matrix,
        index=[
            f'Ref_{class_id}'
            for class_id in CLASS_IDS
        ],
        columns=[
            f'Pred_{class_id}'
            for class_id in CLASS_IDS
        ],
    )


    matrix_df.index.name = (
        'reference_class'
    )

    matrix_df.columns.name = (
        'predicted_class'
    )


    return matrix_df


# ============================================================
# I. Province validation
# ============================================================

def validate_province(
    province,
    province_name,
    province_geom,
):
    """
    Run complete thematic validation for one province.

    Returns
    -------
    tuple
        province_summary
        class_summary
        validation_results
        confusion_matrix
        prediction_inventory
    """

    print(
        "\n============================================================"
    )

    print(
        f"=== Validation: {province_name} ==="
    )

    print(
        "============================================================"
    )


    province_name_clean = (
        clean_province_name(
            province_name
        )
    )


    # --------------------------------------------------------
    # I1. Load final LULC map
    # --------------------------------------------------------

    final_map_asset = (

        f'{ASSET_FOLDER}/'

        f'final_lulc_stack_'

        f'{province_name_clean}_'

        f'2020_{VERSION}'
    )


    if not asset_exists(
        final_map_asset
    ):

        print(
            f"  [SKIP] Final LULC asset not found: "
            f"{final_map_asset}"
        )

        province_summary = {

            'province': (
                province_name
            ),

            'status': (
                'missing_final_map'
            ),

            'validation_points': 0,

            'usable_samples': 0,

            'nodata_points': 0,

            'coverage_percent': None,

            'correct_predictions': None,

            'incorrect_predictions': None,

            'observed_accuracy': None,

            'overall_accuracy': None,

            'kappa': None,

            'confidence_interval': None,
        }

        return (
            province_summary,
            pd.DataFrame(),
            None,
            None,
            pd.DataFrame(),
        )


    final_map = ee.Image(
        final_map_asset
    )


    # --------------------------------------------------------
    # I2. Check required final map bands
    # --------------------------------------------------------

    final_bands = (
        final_map
        .bandNames()
        .getInfo()
    )


    required_bands = [
        'classification',
        'confidence'
    ]


    missing_bands = [
        band
        for band in required_bands
        if band not in final_bands
    ]


    if missing_bands:

        raise RuntimeError(

            f"Final map for "
            f"{province_name} "
            f"is missing required bands: "
            f"{missing_bands}. "
            f"Available bands: "
            f"{final_bands}"
        )


    # --------------------------------------------------------
    # I3. Load validation points
    # --------------------------------------------------------

    validation_fc = (
        ee.FeatureCollection(
            VALIDATION_ASSET
        )
        .filterBounds(
            province_geom
        )
    )


    validation_count = (
        validation_fc
        .size()
        .getInfo()
    )


    print(
        f"  Validation points: "
        f"{validation_count}"
    )


    if validation_count == 0:

        print(
            "  [SKIP] No validation points"
        )

        province_summary = {

            'province': (
                province_name
            ),

            'status': (
                'no_validation_points'
            ),

            'validation_points': 0,

            'usable_samples': 0,

            'nodata_points': 0,

            'coverage_percent': None,

            'correct_predictions': None,

            'incorrect_predictions': None,

            'observed_accuracy': None,

            'overall_accuracy': None,

            'kappa': None,

            'confidence_interval': None,
        }

        return (
            province_summary,
            pd.DataFrame(),
            None,
            None,
            pd.DataFrame(),
        )


    # --------------------------------------------------------
    # I4. Sample final map at validation points
    # --------------------------------------------------------

    sampled_validation = (
        final_map
        .sampleRegions(
            collection=validation_fc,
            properties=[
                VALIDATION_CLASS_PROPERTY,
                VALIDATION_NAME_PROPERTY,
            ],
            scale=EXPORT_SCALE,
            geometries=True
        )
    )


    # --------------------------------------------------------
    # I5. Prediction class inventory
    # --------------------------------------------------------

    prediction_df = (
        build_prediction_inventory(
            sampled_validation,
            province_name,
        )
    )


    print(
        "  Prediction class distribution:"
    )


    for _, row in prediction_df.iterrows():

        print(
            f"    classification="
            f"{int(row['classification'])}: "
            f"{int(row['prediction_n'])} "
            f"({row['class_name']})"
        )


    # --------------------------------------------------------
    # I6. Sampling diagnostics
    # --------------------------------------------------------

    usable_samples = (
        sampled_validation
        .size()
        .getInfo()
    )


    nodata_points = (
        validation_count
        - usable_samples
    )


    coverage_percent = (
        usable_samples
        / validation_count
        * 100
    )


    print(
        f"  Usable samples: "
        f"{usable_samples}"
    )


    print(
        f"  NoData / excluded: "
        f"{nodata_points}"
    )


    print(
        f"  Validation coverage: "
        f"{coverage_percent:.2f}%"
    )


    # --------------------------------------------------------
    # I7. Check prediction inventory completeness
    # --------------------------------------------------------

    predicted_inventory_total = int(
        prediction_df[
            'prediction_n'
        ].sum()
    )


    unexplained_predictions = (
        usable_samples
        - predicted_inventory_total
    )


    print(
        f"  Predictions accounted for: "
        f"{predicted_inventory_total}"
    )


    print(
        f"  Predictions outside inventory: "
        f"{unexplained_predictions}"
    )


    # --------------------------------------------------------
    # I8. Add correctness
    # --------------------------------------------------------

    validation_with_accuracy = (
        sampled_validation
        .map(
            add_correctness
        )
    )


    # --------------------------------------------------------
    # I9. Point-level correctness
    # --------------------------------------------------------

    correct_predictions = (
        validation_with_accuracy
        .filter(
            ee.Filter.eq(
                'correct',
                1
            )
        )
        .size()
        .getInfo()
    )


    incorrect_predictions = (
        validation_with_accuracy
        .filter(
            ee.Filter.eq(
                'correct',
                0
            )
        )
        .size()
        .getInfo()
    )


    if usable_samples > 0:

        observed_accuracy = (
            correct_predictions
            / usable_samples
        )

    else:

        observed_accuracy = None


    print(
        f"  Correct predictions: "
        f"{correct_predictions}"
    )


    print(
        f"  Incorrect predictions: "
        f"{incorrect_predictions}"
    )


    if observed_accuracy is not None:

        print(
            f"  Observed accuracy: "
            f"{observed_accuracy * 100:.2f}%"
        )

    else:

        print(
            "  Observed accuracy: N/A"
        )


    # --------------------------------------------------------
    # I10. Thematic accuracy using luma_ge
    # --------------------------------------------------------

    assessor = thematic_accuracy()


    success, results = (
        assessor
        .run_accuracy_assessment(
            lcmap=final_map,
            validation_data=validation_fc,
            class_property=(
                VALIDATION_CLASS_PROPERTY
            ),
            scale=EXPORT_SCALE,
            confidence=0.95,
        )
    )


    if not success:

        raise RuntimeError(

            f"Assessment failed for "
            f"{province_name}: "
            f"{results.get('error', 'unknown error')}"
        )


    summary = (
        assessor
        .format_accuracy_summary(
            results
        )
    )


    # --------------------------------------------------------
    # I11. Class-level metrics
    # --------------------------------------------------------

    class_df = (
        build_class_summary(
            validation_with_accuracy,
            province_name,
        )
    )


    # --------------------------------------------------------
    # I12. Confusion matrix
    # --------------------------------------------------------

    confusion_df = (
        build_confusion_matrix(
            validation_with_accuracy,
            province_name,
        )
    )


    # --------------------------------------------------------
    # I13. Province summary
    # --------------------------------------------------------

    province_summary = {

        'province': (
            province_name
        ),

        'status': (
            'completed'
        ),

        'validation_points': (
            validation_count
        ),

        'usable_samples': (
            usable_samples
        ),

        'nodata_points': (
            nodata_points
        ),

        'coverage_percent': (
            coverage_percent
        ),

        'correct_predictions': (
            correct_predictions
        ),

        'incorrect_predictions': (
            incorrect_predictions
        ),

        'observed_accuracy': (
            observed_accuracy
        ),

        'overall_accuracy': (
            summary.get(
                'overall_accuracy'
            )
        ),

        'kappa': (
            summary.get(
                'kappa'
            )
        ),

        'confidence_interval': (
            summary.get(
                'confidence_interval'
            )
        ),
    }


    print(
        f"  Overall accuracy: "
        f"{province_summary['overall_accuracy']}"
    )


    print(
        f"  Kappa: "
        f"{province_summary['kappa']}"
    )


    print(
        f"  Sample size: "
        f"{summary.get('sample_size')}"
    )


    # --------------------------------------------------------
    # I14. Accuracy consistency check
    # --------------------------------------------------------

    library_accuracy = (
        summary
        .get('overall_accuracy')
    )


    if (
        observed_accuracy is not None
        and library_accuracy is not None
    ):

        library_accuracy_numeric = float(
            str(
                library_accuracy
            ).replace(
                '%',
                ''
            )
        ) / 100


        difference = abs(
            observed_accuracy
            - library_accuracy_numeric
        )


        if difference > 1e-9:

            print(
                f"  [WARNING] Observed accuracy "
                f"and luma_ge accuracy differ by "
                f"{difference:.6f}"
            )

        else:

            print(
                "  [OK] Observed accuracy and "
                "luma_ge accuracy are consistent"
            )


    return (
        province_summary,
        class_df,
        validation_with_accuracy,
        confusion_df,
        prediction_df,
    )


# ============================================================
# J. Export point-level validation results
# ============================================================

def export_validation_points(
    validation_fc,
    province_name,
):
    """
    Export point-level validation results
    to GEE Asset and Google Drive.
    """

    if validation_fc is None:
        return


    province_name_clean = (
        clean_province_name(
            province_name
        )
    )


    # --------------------------------------------------------
    # J1. Earth Engine Asset
    # --------------------------------------------------------

    asset_id = (

        f'{ASSET_FOLDER}/'

        f'validation_'

        f'{province_name_clean}_'

        f'2020_{VERSION}'
    )


    asset_description = (

        f'validation_'

        f'{province_name_clean}_'

        f'2020_{VERSION}'
    )


    if (
        SKIP_EXISTING_EXPORTS
        and asset_exists(
            asset_id
        )
    ):

        print(
            f"  [SKIP] GEE Asset already exists: "
            f"{asset_id}"
        )

    else:

        task = (
            ee.batch.Export.table.toAsset(
                collection=validation_fc,
                description=asset_description,
                assetId=asset_id,
            )
        )

        task.start()

        print(
            f"  [OK] Started GEE Asset export"
        )


    # --------------------------------------------------------
    # J2. Google Drive CSV
    # --------------------------------------------------------

    drive_task = (
        ee.batch.Export.table.toDrive(
            collection=validation_fc,
            description=asset_description,
            folder=EXPORT_FOLDER,
            fileNamePrefix=asset_description,
            fileFormat='CSV',
        )
    )


    drive_task.start()


    print(
        f"  [OK] Started Google Drive export"
    )


# ============================================================
# K. Export DataFrame to Google Drive
# ============================================================

def export_dataframe_to_drive(
    df,
    description,
    file_name,
):
    """
    Export a pandas DataFrame to Google Drive
    via an Earth Engine FeatureCollection.
    """

    if df is None or df.empty:

        print(
            f"  [SKIP] Empty DataFrame: "
            f"{description}"
        )

        return


    records = (
        df.to_dict(
            orient='records'
        )
    )


    features = []


    for record in records:

        properties = {}


        for key, value in record.items():

            if pd.isna(value):

                properties[key] = None

            else:

                if hasattr(
                    value,
                    'item'
                ):

                    value = (
                        value.item()
                    )

                properties[key] = value


        features.append(
            ee.Feature(
                None,
                properties
            )
        )


    fc = ee.FeatureCollection(
        features
    )


    task = (
        ee.batch.Export.table.toDrive(
            collection=fc,
            description=description,
            folder=EXPORT_FOLDER,
            fileNamePrefix=file_name,
            fileFormat='CSV',
        )
    )


    task.start()


    print(
        f"  [OK] Started summary export: "
        f"{file_name}.csv"
    )


# ============================================================
# L. Main batch processing
# ============================================================

def main():

    # --------------------------------------------------------
    # L1. Load provinces
    # --------------------------------------------------------

    provinces = (
        ee.FeatureCollection(
            PROVINCE_ASSET
        )
    )


    province_list = (
        provinces
        .toList(
            provinces.size()
        )
    )


    n_provinces = (
        provinces
        .size()
        .getInfo()
    )


    print(
        f"\nFound {n_provinces} provinces."
    )


    # --------------------------------------------------------
    # L2. Containers
    # --------------------------------------------------------

    province_results = []

    class_results = []

    confusion_results = []

    prediction_results = []

    all_validation_results = []


    # --------------------------------------------------------
    # L3. Loop over all provinces
    # --------------------------------------------------------

    for i in range(
        n_provinces
    ):

        province = ee.Feature(
            province_list.get(i)
        )


        province_name = (
            province
            .get('AoI')
            .getInfo()
        )


        province_geom = (
            province
            .geometry()
        )


        print(
            f"\n[{i + 1}/{n_provinces}] "
            f"Processing {province_name}"
        )


        (
            province_summary,
            class_df,
            validation_results,
            confusion_df,
            prediction_df,
        ) = validate_province(

            province=province,

            province_name=province_name,

            province_geom=province_geom,
        )


        # ----------------------------------------------------
        # Store province summary
        # ----------------------------------------------------

        province_results.append(
            province_summary
        )


        # ----------------------------------------------------
        # Store class summary
        # ----------------------------------------------------

        if (
            class_df is not None
            and not class_df.empty
        ):

            class_results.append(
                class_df
            )


        # ----------------------------------------------------
        # Store confusion matrix
        # ----------------------------------------------------

        if (
            confusion_df is not None
            and not confusion_df.empty
        ):

            confusion_long = (
                confusion_df
                .reset_index()
                .melt(
                    id_vars=[
                        'reference_class'
                    ],
                    var_name=(
                        'predicted_class'
                    ),
                    value_name='count'
                )
            )


            confusion_long.insert(
                0,
                'province',
                province_name
            )


            confusion_results.append(
                confusion_long
            )


        # ----------------------------------------------------
        # Store prediction inventory
        # ----------------------------------------------------

        if (
            prediction_df is not None
            and not prediction_df.empty
        ):

            prediction_results.append(
                prediction_df
            )


        # ----------------------------------------------------
        # Store all validation points
        # ----------------------------------------------------

        if validation_results is not None:

            all_validation_results.append(
                validation_results
            )


            # Export point-level results

            export_validation_points(
                validation_fc=(
                    validation_results
                ),
                province_name=(
                    province_name
                ),
            )


    # ========================================================
    # L4. Province summary DataFrame
    # ========================================================

    province_summary_df = (
        pd.DataFrame(
            province_results
        )
    )


    # ========================================================
    # L5. Province × class summary
    # ========================================================

    if class_results:

        class_summary_df = (
            pd.concat(
                class_results,
                ignore_index=True
            )
        )

    else:

        class_summary_df = (
            pd.DataFrame()
        )


    # ========================================================
    # L6. Province confusion summary
    # ========================================================

    if confusion_results:

        confusion_summary_df = (
            pd.concat(
                confusion_results,
                ignore_index=True
            )
        )

    else:

        confusion_summary_df = (
            pd.DataFrame()
        )


    # ========================================================
    # L7. Province prediction inventory
    # ========================================================

    if prediction_results:

        prediction_summary_df = (
            pd.concat(
                prediction_results,
                ignore_index=True
            )
        )

    else:

        prediction_summary_df = (
            pd.DataFrame()
        )


    # ========================================================
    # L8. Print province summary
    # ========================================================

    print(
        "\n============================================================"
    )

    print(
        "PROVINCE VALIDATION SUMMARY"
    )

    print(
        "============================================================"
    )


    print(
        province_summary_df
        .to_string(index=False)
    )


    # ========================================================
    # L9. Print prediction inventory
    # ========================================================

    if not prediction_summary_df.empty:

        print(
            "\n============================================================"
        )

        print(
            "PROVINCE PREDICTION INVENTORY"
        )

        print(
            "============================================================"
        )


        print(
            prediction_summary_df
            .to_string(index=False)
        )


    # ========================================================
    # L10. Print class summary
    # ========================================================

    if not class_summary_df.empty:

        print(
            "\n============================================================"
        )

        print(
            "CLASS-LEVEL VALIDATION SUMMARY"
        )

        print(
            "============================================================"
        )


        print(
            class_summary_df
            .to_string(index=False)
        )


    # ========================================================
    # L11. Pooled Sumatra validation
    # ========================================================

    if all_validation_results:

        pooled_validation = (
            all_validation_results[0]
        )


        for fc in all_validation_results[1:]:

            pooled_validation = (
                pooled_validation
                .merge(fc)
            )


        # ----------------------------------------------------
        # Total pooled samples
        # ----------------------------------------------------

        sumatra_usable_samples = (
            pooled_validation
            .size()
            .getInfo()
        )


        # ----------------------------------------------------
        # Correct / incorrect
        # ----------------------------------------------------

        sumatra_correct = (
            pooled_validation
            .filter(
                ee.Filter.eq(
                    'correct',
                    1
                )
            )
            .size()
            .getInfo()
        )


        sumatra_incorrect = (
            pooled_validation
            .filter(
                ee.Filter.eq(
                    'correct',
                    0
                )
            )
            .size()
            .getInfo()
        )


        # ----------------------------------------------------
        # Pooled observed accuracy
        # ----------------------------------------------------

        if sumatra_usable_samples > 0:

            sumatra_accuracy = (
                sumatra_correct
                / sumatra_usable_samples
            )

        else:

            sumatra_accuracy = None


        # ----------------------------------------------------
        # Pooled confusion matrix
        # ----------------------------------------------------

        sumatra_error_matrix = (
            pooled_validation
            .errorMatrix(
                VALIDATION_CLASS_PROPERTY,
                'classification',
                CLASS_IDS
            )
        )


        sumatra_kappa = (
            sumatra_error_matrix
            .kappa()
            .getInfo()
        )


        # ----------------------------------------------------
        # Reference points total
        # ----------------------------------------------------

        sumatra_reference_points = int(
            province_summary_df[
                province_summary_df[
                    'status'
                ] == 'completed'
            ]['validation_points']
            .sum()
        )


        sumatra_nodata_points = (
            sumatra_reference_points
            - sumatra_usable_samples
        )


        sumatra_coverage = (
            sumatra_usable_samples
            / sumatra_reference_points
            * 100
            if sumatra_reference_points > 0
            else None
        )


        # ----------------------------------------------------
        # Sumatra summary
        # ----------------------------------------------------

        sumatra_summary_df = pd.DataFrame([{

            'region': (
                'Sumatra'
            ),

            'status': (
                'completed'
            ),

            'reference_points': (
                sumatra_reference_points
            ),

            'usable_samples': (
                sumatra_usable_samples
            ),

            'nodata_points': (
                sumatra_nodata_points
            ),

            'coverage_percent': (
                sumatra_coverage
            ),

            'correct_predictions': (
                sumatra_correct
            ),

            'incorrect_predictions': (
                sumatra_incorrect
            ),

            'overall_accuracy': (
                sumatra_accuracy
            ),

            'kappa': (
                sumatra_kappa
            ),

            'accuracy_type': (
                'pooled sample-based'
            ),
        }])


        # ----------------------------------------------------
        # Sumatra class summary
        # ----------------------------------------------------

        sumatra_class_df = (
            build_class_summary(
                pooled_validation,
                'Sumatra',
            )
        )


        # ----------------------------------------------------
        # Sumatra prediction inventory
        # ----------------------------------------------------

        sumatra_prediction_df = (
            build_prediction_inventory(
                pooled_validation,
                'Sumatra',
            )
        )


        # ----------------------------------------------------
        # Sumatra confusion matrix
        # ----------------------------------------------------

        sumatra_confusion_df = (
            pd.DataFrame(
                sumatra_error_matrix.getInfo(),
                index=[
                    f'Ref_{class_id}'
                    for class_id in CLASS_IDS
                ],
                columns=[
                    f'Pred_{class_id}'
                    for class_id in CLASS_IDS
                ],
            )
        )


        sumatra_confusion_df.index.name = (
            'reference_class'
        )


        sumatra_confusion_df.columns.name = (
            'predicted_class'
        )


        # ----------------------------------------------------
        # Print Sumatra result
        # ----------------------------------------------------

        print(
            "\n============================================================"
        )

        print(
            "SUMATRA POOLED VALIDATION"
        )

        print(
            "============================================================"
        )


        print(
            f"Reference points   : "
            f"{sumatra_reference_points}"
        )


        print(
            f"Usable samples     : "
            f"{sumatra_usable_samples}"
        )


        print(
            f"NoData / excluded  : "
            f"{sumatra_nodata_points}"
        )


        print(
            f"Coverage           : "
            f"{sumatra_coverage:.2f}%"
            if sumatra_coverage is not None
            else "Coverage           : N/A"
        )


        print(
            f"Correct predictions: "
            f"{sumatra_correct}"
        )


        print(
            f"Incorrect predictions: "
            f"{sumatra_incorrect}"
        )


        print(
            f"Overall accuracy   : "
            f"{sumatra_accuracy * 100:.2f}%"
            if sumatra_accuracy is not None
            else "Overall accuracy   : N/A"
        )


        print(
            f"Kappa              : "
            f"{sumatra_kappa:.3f}"
        )


        # ----------------------------------------------------
        # Pooled prediction accounting
        # ----------------------------------------------------

        pooled_prediction_total = int(
            sumatra_prediction_df[
                'prediction_n'
            ].sum()
        )


        unexplained_pooled_predictions = (
            sumatra_usable_samples
            - pooled_prediction_total
        )


        print(
            f"Predictions accounted for: "
            f"{pooled_prediction_total}"
        )


        print(
            f"Predictions outside inventory: "
            f"{unexplained_pooled_predictions}"
        )


        # ----------------------------------------------------
        # Print Sumatra class summary
        # ----------------------------------------------------

        print(
            "\n============================================================"
        )

        print(
            "SUMATRA POOLED CLASS SUMMARY"
        )

        print(
            "============================================================"
        )


        print(
            sumatra_class_df
            .to_string(index=False)
        )


        # ----------------------------------------------------
        # Print Sumatra prediction inventory
        # ----------------------------------------------------

        print(
            "\n============================================================"
        )

        print(
            "SUMATRA PREDICTION INVENTORY"
        )

        print(
            "============================================================"
        )


        print(
            sumatra_prediction_df
            .to_string(index=False)
        )


    else:

        pooled_validation = None

        sumatra_summary_df = (
            pd.DataFrame()
        )

        sumatra_class_df = (
            pd.DataFrame()
        )

        sumatra_prediction_df = (
            pd.DataFrame()
        )

        sumatra_confusion_df = (
            pd.DataFrame()
        )


    # ========================================================
    # L12. Save local CSV outputs
    # ========================================================

    province_summary_file = (
        f'validation_summary_'
        f'Sumatra_2020_{VERSION}.csv'
    )


    province_summary_df.to_csv(
        province_summary_file,
        index=False
    )


    print(
        f"\n[OK] Saved province summary: "
        f"{province_summary_file}"
    )


    # --------------------------------------------------------
    # Province × class
    # --------------------------------------------------------

    if not class_summary_df.empty:

        class_summary_file = (
            f'validation_class_summary_'
            f'Sumatra_2020_{VERSION}.csv'
        )


        class_summary_df.to_csv(
            class_summary_file,
            index=False
        )


        print(
            f"[OK] Saved class summary: "
            f"{class_summary_file}"
        )


    # --------------------------------------------------------
    # Province prediction inventory
    # --------------------------------------------------------

    if not prediction_summary_df.empty:

        prediction_summary_file = (
            f'validation_prediction_summary_'
            f'Sumatra_2020_{VERSION}.csv'
        )


        prediction_summary_df.to_csv(
            prediction_summary_file,
            index=False
        )


        print(
            f"[OK] Saved prediction summary: "
            f"{prediction_summary_file}"
        )


    # --------------------------------------------------------
    # Province confusion summary
    # --------------------------------------------------------

    if not confusion_summary_df.empty:

        confusion_summary_file = (
            f'validation_confusion_summary_'
            f'Sumatra_2020_{VERSION}.csv'
        )


        confusion_summary_df.to_csv(
            confusion_summary_file,
            index=False
        )


        print(
            f"[OK] Saved province confusion summary: "
            f"{confusion_summary_file}"
        )


    # --------------------------------------------------------
    # Sumatra overall summary
    # --------------------------------------------------------

    if not sumatra_summary_df.empty:

        sumatra_summary_file = (
            f'validation_Sumatra_summary_'
            f'2020_{VERSION}.csv'
        )


        sumatra_summary_df.to_csv(
            sumatra_summary_file,
            index=False
        )


        print(
            f"[OK] Saved Sumatra summary: "
            f"{sumatra_summary_file}"
        )


    # --------------------------------------------------------
    # Sumatra class summary
    # --------------------------------------------------------

    if not sumatra_class_df.empty:

        sumatra_class_file = (
            f'validation_Sumatra_pooled_'
            f'class_summary_'
            f'2020_{VERSION}.csv'
        )


        sumatra_class_df.to_csv(
            sumatra_class_file,
            index=False
        )


        print(
            f"[OK] Saved Sumatra class summary: "
            f"{sumatra_class_file}"
        )


    # --------------------------------------------------------
    # Sumatra prediction inventory
    # --------------------------------------------------------

    if not sumatra_prediction_df.empty:

        sumatra_prediction_file = (
            f'validation_Sumatra_prediction_'
            f'summary_'
            f'2020_{VERSION}.csv'
        )


        sumatra_prediction_df.to_csv(
            sumatra_prediction_file,
            index=False
        )


        print(
            f"[OK] Saved Sumatra prediction summary: "
            f"{sumatra_prediction_file}"
        )


    # --------------------------------------------------------
    # Sumatra confusion matrix
    # --------------------------------------------------------

    if not sumatra_confusion_df.empty:

        sumatra_confusion_file = (
            f'validation_Sumatra_confusion_'
            f'matrix_'
            f'2020_{VERSION}.csv'
        )


        sumatra_confusion_df.to_csv(
            sumatra_confusion_file
        )


        print(
            f"[OK] Saved Sumatra confusion matrix: "
            f"{sumatra_confusion_file}"
        )


    # ========================================================
    # L13. Export summary tables to Google Drive
    # ========================================================

    export_dataframe_to_drive(

        province_summary_df,

        description=(
            f'Sumatra_province_summary_'
            f'2020_{VERSION}'
        ),

        file_name=(
            f'validation_Sumatra_province_summary_'
            f'2020_{VERSION}'
        ),
    )


    if not class_summary_df.empty:

        export_dataframe_to_drive(

            class_summary_df,

            description=(
                f'Sumatra_class_summary_'
                f'2020_{VERSION}'
            ),

            file_name=(
                f'validation_Sumatra_class_summary_'
                f'2020_{VERSION}'
            ),
        )


    if not prediction_summary_df.empty:

        export_dataframe_to_drive(

            prediction_summary_df,

            description=(
                f'Sumatra_prediction_summary_'
                f'2020_{VERSION}'
            ),

            file_name=(
                f'validation_Sumatra_prediction_summary_'
                f'2020_{VERSION}'
            ),
        )


    if not confusion_summary_df.empty:

        export_dataframe_to_drive(

            confusion_summary_df,

            description=(
                f'Sumatra_confusion_summary_'
                f'2020_{VERSION}'
            ),

            file_name=(
                f'validation_Sumatra_confusion_summary_'
                f'2020_{VERSION}'
            ),
        )


    if not sumatra_summary_df.empty:

        export_dataframe_to_drive(

            sumatra_summary_df,

            description=(
                f'Sumatra_summary_'
                f'2020_{VERSION}'
            ),

            file_name=(
                f'validation_Sumatra_summary_'
                f'2020_{VERSION}'
            ),
        )


    if not sumatra_class_df.empty:

        export_dataframe_to_drive(

            sumatra_class_df,

            description=(
                f'Sumatra_pooled_class_summary_'
                f'2020_{VERSION}'
            ),

            file_name=(
                f'validation_Sumatra_pooled_class_summary_'
                f'2020_{VERSION}'
            ),
        )


    if not sumatra_prediction_df.empty:

        export_dataframe_to_drive(

            sumatra_prediction_df,

            description=(
                f'Sumatra_prediction_inventory_'
                f'2020_{VERSION}'
            ),

            file_name=(
                f'validation_Sumatra_prediction_inventory_'
                f'2020_{VERSION}'
            ),
        )


    if not sumatra_confusion_df.empty:

        sumatra_confusion_export = (
            sumatra_confusion_df
            .reset_index()
        )


        export_dataframe_to_drive(

            sumatra_confusion_export,

            description=(
                f'Sumatra_confusion_matrix_'
                f'2020_{VERSION}'
            ),

            file_name=(
                f'validation_Sumatra_confusion_matrix_'
                f'2020_{VERSION}'
            ),
        )

    # ========================================================
    # L14. Final message
    # ========================================================

    missing_final_maps = sum(
        1
        for x in province_results
        if x.get('status') == 'missing_final_map'
    )

    completed_provinces = sum(
        1
        for x in province_results
        if x.get('status') == 'completed'
    )

    print(
        "\n============================================================"
    )

    print(
        "BATCH VALIDATION COMPLETED"
    )

    print(
        "============================================================"
    )

    print(
        f"Processed provinces : {n_provinces}"
    )

    print(
        f"Missing final maps  : {missing_final_maps}"
    )

    print(
        f"Completed provinces : {completed_provinces}"
    )

    print(
        "============================================================"
    )

# ============================================================
# M. Run
# ============================================================

if __name__ == '__main__':
    main()
    