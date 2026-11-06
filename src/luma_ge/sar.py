"""
SAR predictor retrieval for LULC mapping (Module 5 auxiliary predictors).

Phase 1: Sentinel-1 C-band GRD backscatter (VV, VH) and polarization ratio.
Phase 2: ALOS PALSAR / PALSAR-2 L-band (HH, HV, HH/HV).

Outputs are backscatter polarizations and polarization ratios only
(no radar vegetation indices).
"""

import logging
from typing import List, Optional, Union
import ee
from .data_acquisition import parse_date_input
from .ee_config import ensure_ee_initialized

logger = logging.getLogger(__name__)


class get_sar_data:
    """
    Retrieve and preprocess SAR backscatter polarizations and polarization
    ratios as predictors for LULC mapping.

    Sentinel-1 preprocessing follows the main ideas of Mullissa et al. (2021)
    (https://github.com/adugnag/gee_s1_ard): IW mode filtering, incidence-angle
    masking, low-backscatter swath-edge masking and optional speckle filtering.
    The GEE `COPERNICUS/S1_GRD` collection already provides thermal noise removal,
    orbit correction, radiometric calibration and terrain correction (dB).

    PALSAR / PALSAR-2 yearly mosaics (JAXA) are 25m orthorectified products.
    Preprocessing includes QA masking (layover, shadow, nodata), filtering DN > 0,
    and radiometric calibration from 16-bit DN to gamma-naught (dB):
        gamma0_dB = 10 * log10(DN^2) - 83.0
    Multi-collection routing is supported across JAXA/ALOS/PALSAR/YEARLY/SAR (v1: 2007-2010, 2015-2020)
    and JAXA/ALOS/PALSAR/YEARLY/SAR_EPOCH (v2: 2007-2010, 2015-2024), with nearest-year fallback
    for the 2011-2014 mission gap or years beyond coverage.
    """

    PALSAR_V1_YEARS = [2007, 2008, 2009, 2010, 2015, 2016, 2017, 2018, 2019, 2020]
    PALSAR_EPOCH_YEARS = [2007, 2008, 2009, 2010, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
    ALL_AVAILABLE_PALSAR_YEARS = sorted(list(set(PALSAR_V1_YEARS + PALSAR_EPOCH_YEARS)))

    SAR_DATASETS = {
        "S1_GRD": {
            "collection": "COPERNICUS/S1_GRD",
            "polarizations": ["VV", "VH"],
            "output_bands": ["S1_VV", "S1_VH", "S1_VV_VH"],
            "instrument_mode": "IW",
            "min_date": "2014-10-03",
            "scale": 10,
            "description": "Sentinel-1 SAR GRD C-band (10m, dB)",
        },
        "PALSAR_EPOCH": {
            "collection": "JAXA/ALOS/PALSAR/YEARLY/SAR_EPOCH",
            "polarizations": ["HH", "HV"],
            "output_bands": ["PALSAR_HH", "PALSAR_HV", "PALSAR_HH_HV"],
            "qa_band": "qa",
            "scale": 25,
            "description": "ALOS/ALOS-2 PALSAR Yearly Mosaic v2 (2007-2010, 2015-2024, 25m, dB)",
        },
        "PALSAR_V1": {
            "collection": "JAXA/ALOS/PALSAR/YEARLY/SAR",
            "polarizations": ["HH", "HV"],
            "output_bands": ["PALSAR_HH", "PALSAR_HV", "PALSAR_HH_HV"],
            "qa_band": "qa",
            "scale": 25,
            "description": "ALOS PALSAR Yearly Mosaic v1 (2007-2010, 2015-2020, 25m, dB)",
        },
    }

    VALID_REDUCERS = ("median", "mean")
    VALID_SPECKLE_FILTERS = ("none", "focal_median", "focal_mean", "lee")
    VALID_ORBIT_PASSES = ("BOTH", "ASCENDING", "DESCENDING")

    # Lee filter: equivalent number of looks of S1 IW GRDH
    S1_ENL = 4.4
    # PALSAR 25m mosaic calibration constant (CF)
    PALSAR_CALIBRATION_FACTOR = -83.0

    def __init__(self):
        """Initialize the SAR data retriever"""
        ensure_ee_initialized()
        logger.info("get_sar_data initialized")

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _db_to_power(image: ee.Image) -> ee.Image:
        """Convert dB to linear power."""
        return ee.Image(10.0).pow(image.divide(10.0))

    @staticmethod
    def _power_to_db(image: ee.Image) -> ee.Image:
        """Convert linear power to dB."""
        return image.log10().multiply(10.0)

    def _reduce(self, collection: ee.ImageCollection, reducer: str) -> ee.Image:
        """Reduce a (dB) collection temporally with a median or mean reducer."""
        if reducer == "mean":
            return collection.mean()
        return collection.median()

    def apply_speckle_filter(self, image: ee.Image, method: str = "focal_median",
                             radius_m: float = 30.0) -> ee.Image:
        """
        Apply spatial speckle filtering to a dB image.

        Filtering is done in the linear power domain and converted back to dB.

        Parameters:
        -----------
        image : ee.Image
            Image in dB.
        method : str
            'none', 'focal_median', 'focal_mean' or 'lee'.
        radius_m : float
            Kernel radius in meters (circular kernel).
        """
        if method not in self.VALID_SPECKLE_FILTERS:
            raise ValueError(
                f"Unknown speckle filter '{method}'. Valid: {self.VALID_SPECKLE_FILTERS}"
            )
        if method == "none":
            return image

        band_names = image.bandNames()
        power = self._db_to_power(image)

        if method == "focal_median":
            filtered = power.focalMedian(radius_m, "circle", "meters")
        elif method == "focal_mean":
            filtered = power.focalMean(radius_m, "circle", "meters")
        else:  # lee
            kernel = ee.Kernel.circle(radius_m, "meters")
            local_mean = power.reduceNeighborhood(ee.Reducer.mean(), kernel).rename(band_names)
            local_var = power.reduceNeighborhood(ee.Reducer.variance(), kernel).rename(band_names)
            cu2 = 1.0 / self.S1_ENL
            ci2 = local_var.divide(local_mean.pow(2))
            weight = ee.Image(1.0).subtract(ee.Image(cu2).divide(ci2)).clamp(0, 1)
            filtered = local_mean.add(weight.multiply(power.subtract(local_mean)))

        return self._power_to_db(filtered).rename(band_names)

    # ------------------------------------------------------------------
    # Sentinel-1
    # ------------------------------------------------------------------
    def mask_s1_border_and_angle(self, image: ee.Image, min_angle: float = 30.63993,
                                 max_angle: float = 45.23728,
                                 edge_db_floor: float = -35.0) -> ee.Image:
        """
        Mask Sentinel-1 pixels with incidence angle outside [min_angle, max_angle]
        and low-backscatter swath-edge noise (VV or VH <= edge_db_floor).
        """
        angle = image.select("angle")
        vv = image.select("VV")
        vh = image.select("VH")
        valid = (angle.gte(min_angle)
                 .And(angle.lte(max_angle))
                 .And(vv.gt(edge_db_floor))
                 .And(vh.gt(edge_db_floor)))
        return image.select(["VV", "VH"]).updateMask(valid).copyProperties(
            image, ["system:time_start"]
        )

    def get_sentinel1(self, aoi: ee.Geometry, start_date: Union[int, str],
                      end_date: Union[int, str], orbit_pass: str = "BOTH",
                      reducer: str = "median", speckle_filter: str = "focal_median",
                      filter_radius_m: float = 30.0, include_ratio: bool = True,
                      max_scenes: Optional[int] = None) -> Optional[ee.Image]:
        """
        Build a Sentinel-1 backscatter predictor image for [start_date, end_date].

        Returns an image with bands `S1_VV`, `S1_VH` (dB) and, if `include_ratio`,
        `S1_VV_VH` (VV/VH polarization ratio expressed in dB, i.e. VV_dB - VH_dB).
        Returns None if the date range predates Sentinel-1 or no scenes are found.

        Parameters:
        -----------
        aoi : ee.Geometry
        start_date, end_date : int or str
            Year (e.g. 2022) or 'YYYY-MM-DD' string (same formats as Module 1).
        orbit_pass : str
            'BOTH', 'ASCENDING' or 'DESCENDING'.
        reducer : str
            Temporal reducer: 'median' or 'mean'.
        speckle_filter : str
            'none', 'focal_median', 'focal_mean' or 'lee'.
        filter_radius_m : float
            Speckle filter radius in meters.
        include_ratio : bool
            Whether to add the polarization ratio band.
        max_scenes : int, optional
            Optional cap on the number of scenes used.
        """
        orbit_pass = str(orbit_pass).upper()
        if orbit_pass not in self.VALID_ORBIT_PASSES:
            raise ValueError(f"Invalid orbit_pass '{orbit_pass}'. Valid: {self.VALID_ORBIT_PASSES}")
        if reducer not in self.VALID_REDUCERS:
            raise ValueError(f"Invalid reducer '{reducer}'. Valid: {self.VALID_REDUCERS}")

        cfg = self.SAR_DATASETS["S1_GRD"]
        start = parse_date_input(start_date, True)
        end = parse_date_input(end_date, False)

        if end < cfg["min_date"]:
            logger.warning(
                f"Sentinel-1 data starts {cfg['min_date']}; requested range "
                f"{start} to {end} has no coverage"
            )
            return None

        collection = (ee.ImageCollection(cfg["collection"])
                      .filterBounds(aoi)
                      .filterDate(start, end)
                      .filter(ee.Filter.eq("instrumentMode", cfg["instrument_mode"]))
                      .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
                      .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH")))
        if orbit_pass != "BOTH":
            collection = collection.filter(ee.Filter.eq("orbitProperties_pass", orbit_pass))
        if max_scenes:
            collection = collection.limit(int(max_scenes))

        try:
            scene_count = collection.size().getInfo()
        except Exception as e:
            logger.warning(f"Could not query Sentinel-1 scene count: {e}")
            return None
        if scene_count == 0:
            logger.warning(f"No Sentinel-1 IW VV+VH scenes found for {start} to {end}")
            return None
        logger.info(f"Sentinel-1: {scene_count} scenes for {start} to {end} ({orbit_pass})")

        masked = collection.map(self.mask_s1_border_and_angle)
        reduced = self._reduce(masked, reducer)
        filtered = self.apply_speckle_filter(reduced, speckle_filter, filter_radius_m)

        image = filtered.select(["VV", "VH"]).rename(["S1_VV", "S1_VH"])
        if include_ratio:
            ratio = image.select("S1_VV").subtract(image.select("S1_VH")).rename("S1_VV_VH")
            image = image.addBands(ratio)

        return image.clip(aoi).toFloat()

    # ------------------------------------------------------------------
    # ALOS PALSAR / PALSAR-2 (Phase 2)
    # ------------------------------------------------------------------
    def mask_and_calibrate_palsar(self, image: ee.Image,
                                  mask_layover_shadow: bool = True) -> ee.Image:
        """
        Mask invalid pixels using the 'qa' band and calibrate 16-bit DN to gamma-naught (dB).

        JAXA calibration formula:
            gamma0_dB = 10 * log10(DN^2) - 83.0 = 20 * log10(DN) - 83.0

        QA band values:
            0: No data
            50: Ocean / water
            100: Layover
            150: Shadow
            255: Land / valid observation

        Parameters:
        -----------
        image : ee.Image
            PALSAR yearly mosaic image containing 'HH', 'HV', and 'qa' bands.
        mask_layover_shadow : bool
            Whether to mask pixels flagged as layover (100) or shadow (150).
        """
        qa = image.select("qa")
        hh = image.select("HH")
        hv = image.select("HV")

        valid_mask = qa.neq(0).And(hh.gt(0)).And(hv.gt(0))
        if mask_layover_shadow:
            valid_mask = valid_mask.And(qa.neq(100)).And(qa.neq(150))

        # Calibrate DN to gamma-naught in dB: 10 * log10(DN^2) - 83.0
        # DN is uint16, cast to float before pow/log
        hh_cal = (hh.toFloat().pow(2).log10().multiply(10.0)
                  .add(self.PALSAR_CALIBRATION_FACTOR)
                  .rename("HH"))
        hv_cal = (hv.toFloat().pow(2).log10().multiply(10.0)
                  .add(self.PALSAR_CALIBRATION_FACTOR)
                  .rename("HV"))

        calibrated = ee.Image.cat([hh_cal, hv_cal]).updateMask(valid_mask)
        return calibrated.copyProperties(image, ["system:time_start", "system:index"])

    def _resolve_palsar_years(self, start_year: int, end_year: int,
                              allow_nearest_year_fallback: bool = True) -> List[int]:
        """
        Resolve the available PALSAR mosaic years intersecting [start_year, end_year].
        If no exact years intersect (e.g. 2011-2014 gap, or > 2024 / < 2007) and
        allow_nearest_year_fallback is True, finds the nearest available year.
        """
        direct_years = [y for y in self.ALL_AVAILABLE_PALSAR_YEARS if start_year <= y <= end_year]
        if direct_years:
            return direct_years

        if not allow_nearest_year_fallback:
            return []

        # Find nearest year across ALL_AVAILABLE_PALSAR_YEARS to the target midpoint
        target_mid = (start_year + end_year) / 2.0
        nearest = min(self.ALL_AVAILABLE_PALSAR_YEARS, key=lambda y: abs(y - target_mid))
        logger.warning(
            f"No PALSAR yearly mosaic directly available for {start_year}-{end_year} "
            f"(e.g. 2011-2014 mission gap or out-of-range). Falling back to nearest year: {nearest}"
        )
        return [nearest]

    def get_palsar(self, aoi: ee.Geometry, start_date: Union[int, str],
                   end_date: Union[int, str], reducer: str = "median",
                   speckle_filter: str = "focal_median", filter_radius_m: float = 30.0,
                   include_ratio: bool = True, mask_layover_shadow: bool = True,
                   allow_nearest_year_fallback: bool = True) -> Optional[ee.Image]:
        """
        Build an ALOS PALSAR / PALSAR-2 L-band backscatter predictor image for [start_date, end_date].

        Automatically handles collection routing across:
        - `JAXA/ALOS/PALSAR/YEARLY/SAR_EPOCH` (v2: covers 2007-2010, 2015-2024)
        - `JAXA/ALOS/PALSAR/YEARLY/SAR` (v1: fallback for 2007-2010, 2015-2020)
        - Nearest-year fallback for mission gap (2011-2014) or years after 2024.

        Returns an image with bands `PALSAR_HH`, `PALSAR_HV` (gamma0 in dB) and,
        if `include_ratio`, `PALSAR_HH_HV` (HH/HV polarization ratio in dB, i.e. HH_dB - HV_dB).
        Returns None if no scenes are found or date cannot be resolved.

        Parameters:
        -----------
        aoi : ee.Geometry
        start_date, end_date : int or str
            Year (e.g. 2020) or 'YYYY-MM-DD' string.
        reducer : str
            Temporal reducer ('median' or 'mean') when multi-year range is selected.
        speckle_filter : str
            'none', 'focal_median', 'focal_mean' or 'lee'.
        filter_radius_m : float
            Speckle filter radius in meters.
        include_ratio : bool
            Whether to add the polarization ratio band (PALSAR_HH_HV).
        mask_layover_shadow : bool
            Whether to mask layover/shadow from QA band.
        allow_nearest_year_fallback : bool
            Whether to fallback to nearest year if requested period falls in 2011-2014 or > 2024.
        """
        if reducer not in self.VALID_REDUCERS:
            raise ValueError(f"Invalid reducer '{reducer}'. Valid: {self.VALID_REDUCERS}")

        start_str = parse_date_input(start_date, True)
        end_str = parse_date_input(end_date, False)
        start_year = int(start_str[:4])
        end_year = int(end_str[:4])

        target_years = self._resolve_palsar_years(
            start_year, end_year, allow_nearest_year_fallback=allow_nearest_year_fallback
        )
        if not target_years:
            logger.warning(f"No PALSAR mosaic years available for {start_year}-{end_year}")
            return None

        min_y = min(target_years)
        max_y = max(target_years)

        # Route query: Prefer SAR_EPOCH (v2) which has 2007-2010 and 2015-2024
        epoch_cfg = self.SAR_DATASETS["PALSAR_EPOCH"]
        col = (ee.ImageCollection(epoch_cfg["collection"])
               .filterBounds(aoi)
               .filter(ee.Filter.calendarRange(min_y, max_y, "year")))

        # If target years contain pre-2015 and SAR_EPOCH has missing coverage, check v1
        try:
            col_size = col.size().getInfo()
        except Exception as e:
            logger.warning(f"Could not query PALSAR_EPOCH collection size: {e}")
            col_size = 0

        if col_size == 0:
            v1_cfg = self.SAR_DATASETS["PALSAR_V1"]
            logger.info(f"Querying fallback PALSAR v1 collection for years {min_y}-{max_y}...")
            col = (ee.ImageCollection(v1_cfg["collection"])
                   .filterBounds(aoi)
                   .filter(ee.Filter.calendarRange(min_y, max_y, "year")))
            try:
                col_size = col.size().getInfo()
            except Exception as e:
                logger.warning(f"Could not query PALSAR_V1 collection size: {e}")
                col_size = 0

        if col_size == 0:
            logger.warning(f"No PALSAR scenes found for years {target_years} over AOI")
            return None

        logger.info(f"PALSAR: {col_size} yearly mosaic scenes retrieved for years {target_years}")

        calibrated = col.map(lambda img: self.mask_and_calibrate_palsar(img, mask_layover_shadow=mask_layover_shadow))
        reduced = self._reduce(calibrated, reducer)
        filtered = self.apply_speckle_filter(reduced, speckle_filter, filter_radius_m)

        image = filtered.select(["HH", "HV"]).rename(["PALSAR_HH", "PALSAR_HV"])
        if include_ratio:
            ratio = image.select("PALSAR_HH").subtract(image.select("PALSAR_HV")).rename("PALSAR_HH_HV")
            image = image.addBands(ratio)

        return image.clip(aoi).toFloat()
