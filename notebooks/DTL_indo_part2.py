import ee

# !python -m pip install .. --quiet

# ee.Authenticate()
# ee.Initialize(project='epistem2')

VERSION = 'v8'

# ---------------- Regions ----------------
# Every region the pipeline should run for. The value is the region name
# exactly as it's cased where it appears capitalized in asset/collection
# names (e.g. 'AOI_Sumatra'). Wherever the region is needed in a
# lowercase-only name (e.g. 'stacked_landsat_2020_sumatra'), '.lower()' is
# applied to this same value — so the casing stays correct everywhere
# without having to store it twice.
REGIONS = {
    "sumatra": "Sumatra",
    # "kalimantan": "Kalimantan",
    # "sulawesi": "Sulawesi",
    # "maluku": "Maluku",
    # "papua": "Papua",
    # "jawabali": "JawaBali",
    # "nusatenggara": "NusaTenggara"

}


def asset_exists(asset_id):
    """Check whether an EE asset already exists. Cheap metadata call,
    not a heavy interactive computation — safe to call every iteration."""
    try:
        ee.data.getAsset(asset_id)
        return True
    except ee.ee_exception.EEException:
        return False


# 1. Satellite imagery
# AOI definition

# import region from Hadi's assets
# region_name = "Sumatera"
# regions_fc = ee.FeatureCollection("users/hadicu06/IIASA/RESTORE/vector_datasets/classification_regions")
# aoi = regions_fc.filter(ee.Filter.eq('region_name', region_name)).geometry()

from luma_ge.data_acquisition import Reflectance_Data, final_Image

optical_reflectance = Reflectance_Data()

composite = final_Image() #NEW FEATURE ADDED HERE
#define the start and end date for imagery collection
start = '2025-01-01'
end = '2025-12-31'

for region_key, region_name in REGIONS.items():
    region_lower = region_name.lower()

    print(f"\n#################### Region: {region_name} ####################")

    aoi = ee.FeatureCollection(f"projects/epistem2/assets/AOI_{region_name}")

    landsat_data, meta = optical_reflectance.get_optical_data(
        aoi, start, end, optical_data='L8_SR', 
        cloud_cover=50,
        compute_detailed_stats=False,
        scene_limit=500
    )

    stacked_landsat = landsat_data.mosaic().clip(aoi)

    # add predictor bands to the stacked landsat image

    from luma_ge.predictor import SpectralCalculator

    spectral_calc = SpectralCalculator()

    indices_to_compute = ['NDVI', 'MNDWI', 'NDBI']
    spectral_indices = spectral_calc.calculate_indices_with_collection(
        collection=landsat_data,
        aoi=aoi, # type: ignore
        index_list=indices_to_compute,
        reducer_method='mean'
    )

    final_stack = stacked_landsat.addBands(spectral_indices).toFloat()

    landsat_asset_id = f'projects/epistem2/assets/stacked_landsat_pred_2025_{region_lower}_{VERSION}'

    if asset_exists(landsat_asset_id):
        print(f" Already exists at {landsat_asset_id} — skipping")
        continue

    task = ee.batch.Export.image.toAsset(
        image=final_stack,
        description=f'stacked_landsat_pred_2025_{region_lower}_{VERSION}',
        assetId=landsat_asset_id,
        region=aoi.geometry(),
        scale=100,
        maxPixels=1e13
    )

    task.start()

    print(f"  [OK] {region_name} export started")

print("\nAll regions processed.")