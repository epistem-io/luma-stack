import ee
import time
import geopandas as gpd
from shapely.geometry import shape
import geemap

ee.Initialize(project='epistem2')

VERSION = 'v6'
POLL_INTERVAL_SEC = 30
EXPORT_FOLDER = 'GEE_exports'
EXPORT_SCALE = 100
ASSET_FOLDER = 'projects/epistem2/assets'

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
            print(f"  [OK] {label} completed")
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

    provinces = ee.FeatureCollection('projects/epistem2/assets/AOI_JawaBali_Provinces')
    province_list = provinces.toList(provinces.size())
    n_provinces = 7 # jawabali number of provinces

    # for i in range(n_provinces):
    for i in range(0, n_provinces):
        province = ee.Feature(province_list.get(i))
        province_name = province.get('AoI').getInfo()
        province_name_clean = (
            province_name.replace(' ', '_').replace('/', '_').replace('-', '_')
        )
        province_geom = province.geometry()

        print(f"\n=== Province {i + 1}/{n_provinces}: {province_name} ===")

        # Load probability stack from asset

        prob_stack = ee.Image(f'projects/epistem2/assets/probability_stack_{province_name_clean}_2020_{VERSION}')

        # Derive class_ids directly from prob_stack's band names — guaranteed to match
        prob_band_names = prob_stack.bandNames().getInfo()  # e.g. ['prob_18', 'prob_2', ...]

        # Extract the numeric class id from each band name, keep prob_stack's own order
        class_ids_list = [int(b.split('_')[1]) for b in prob_band_names]
        class_ids = ee.List(class_ids_list)

        print("class_ids (from prob_stack):", class_ids_list)

        # prob_stack is already in this exact order — no need to reorder/select
        max_prob_index = prob_stack.toArray().arrayArgmax().arrayGet(0)

        final_lc = max_prob_index.remap(
            ee.List.sequence(0, class_ids.size().subtract(1)),
            class_ids
        ).rename('classification')

        max_confidence = prob_stack.toArray().arrayReduce(ee.Reducer.max(), [0]).arrayGet([0]).rename('confidence')

        final_lulc_stack = final_lc.addBands(max_confidence)
        
        # --- Export probability stack; wait before moving to the next province ---

        lulc_asset_id = (
            f'{ASSET_FOLDER}/final_lulc_stack_{province_name_clean}_2020_{VERSION}'
        )

        if asset_exists(lulc_asset_id):
            print(f"  ✓ Already exists at {lulc_asset_id} — skipping")
            continue
    
        prob_task = ee.batch.Export.image.toAsset(
            image=final_lulc_stack,
            description=f'final_lulc_stack_{province_name_clean}_2020_{VERSION}',
            # folder=EXPORT_FOLDER,
            assetId=lulc_asset_id,
            region=province_geom,
            scale=EXPORT_SCALE,
            maxPixels=1e13
        )
        prob_task.start()
        wait_for_task(prob_task, label=f"final lulc export [{province_name}]")

        print(f"  [OK] {province_name} done")

    print("\nAll provinces processed.")


if __name__ == "__main__":
    main()