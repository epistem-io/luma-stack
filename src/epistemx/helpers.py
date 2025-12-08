import ee
from .ee_config import ensure_ee_initialized

# Do not initialize Earth Engine at import time. Initialize when functions are called.

#############################  Area of Interest  ###########################
def get_aoi_from_gaul(country="Indonesia", province="Sumatera Selatan"):
    """
    Get Area of Interest geometry from GAUL administrative boundaries.
    
    Parameters:
    -----------
    country : str
        Country name (default: "Indonesia")
    province : str
        Province/state name (default: "Sumatera Selatan")
        
    Returns:
    --------
    ee.Geometry : Area of interest geometry
    """
    ensure_ee_initialized()
    admin = ee.FeatureCollection("FAO/GAUL/2015/level1")
    aoi_fc = admin.filter(ee.Filter.eq('ADM0_NAME', country)).filter(
        ee.Filter.eq('ADM1_NAME', province)
    )
    return aoi_fc.geometry()

