"""
Temporal Segmentation Compositing Module (Module 1 Companion)

Provides opt-in native Google Earth Engine temporal segmentation compositing
(``LandTrendr`` Fitted-To-Vertex / FTV) for multitemporal LULC mapping.

Architecture
------------
Follows the same composition boundary as ``imagery_correction.py``:
- ``data_acquisition.py`` remains the single source of truth for satellite
  collection IDs, cloud masking, band renaming, scaling, Roy et al. (2016)
  harmonization, BRDF/SCSc corrections, and median compositing.
- This module delegates per-year frame construction to ``Reflectance_Data``
  and ``final_Image``, and owns temporal window validation, auto-padding,
  segmentation driver-index preparation, native GEE ``LandTrendr`` execution,
  FTV ``<band>_fit`` extraction, and gap-fallback unmasking.
"""

from __future__ import annotations
import datetime
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple
import ee
from .ee_config import ensure_ee_initialized

logger = logging.getLogger(__name__)

#Supported 30m harmonized sensors for LandTrendr FTV temporal segmentation
#define all supported sensors as a frozenset for fast membership testing
SUPPORTED_LT_SENSORS = frozenset({'L4_SR', 'L5_SR', 'L7_SR', 'L8_SR', 'L9_SR','HLS', 'HLS_L30', 'HLS_S30'})
HLS_SENSOR_KEYS = frozenset({'HLS', 'HLS_L30', 'HLS_S30'})
#optical bands (30m) present across Landsat 4-9 SR and HLS v2.0
LT_OPTICAL_BANDS: List[str] = ['BLUE', 'GREEN', 'RED', 'NIR', 'SWIR1', 'SWIR2']
#Driver indices supported for LandTrendr breakpoint detection (inverted so disturbance > 0)
DRIVER_INDEX_CHOICES = frozenset({'NBR', 'NDVI', 'NDMI'})
#default parameter
#ref: https://emapr.github.io/LT-GEE/lt-gee-requirements.html#importantsteps
DEFAULT_LANDTRENDR_PARAMS: Dict[str, Any] = {
    'maxSegments': 6,
    'spikeThreshold': 0.9,
    'vertexCountOvershoot': 3,
    'preventOneYearRecovery': True,
    'recoveryThreshold': 0.25,
    'pvalThreshold': 0.05,
    'bestModelProportion': 0.75,
    'minObservationsNeeded': 6,
}

# Sensor launch year floors for auto-padding continuous annual stacks
_LANDSAT_SR_MIN_YEAR = 1984
_HLS_MIN_YEAR = 2013

#date parsing
def parse_year(date_val: Any, is_start: bool) -> Tuple[int, str]:
    """Parse an int year (`2015`) or ISO date (`'2015-05-01'`) into `(year, 'MM-DD')`."""
    if isinstance(date_val, int):
        return int(date_val), ('01-01' if is_start else '12-31')
    s = str(date_val).strip()
    if len(s) == 4 and s.isdigit():
        return int(s), ('01-01' if is_start else '12-31')
    if len(s) >= 10 and s[4] == '-' and s[7] == '-':
        return int(s[:4]), s[5:10]
    raise ValueError(f"Unsupported date format for temporal segmentation: {date_val!r}")

#configure sensor selection (landsat collection)
def _sensors_for_landsat_year(year: int) -> List[str]:
    """
    Return the optimal Landsat Collection 2 SR sensor(s) active in ``year``.

    Avoids Landsat-7 SLC-off striping once Landsat-8 OLI is operational (2013+),
    while using Landsat-7 in 2012 to bridge the Landsat-5 -> Landsat-8 gap.
    """
    if year < 1982:
        raise ValueError(f"Landsat Collection 2 SR is unavailable before 1982 (got {year}).")
    if year <= 1983:
        return ['L4_SR']
    if year <= 1993:
        return ['L4_SR', 'L5_SR']
    if year <= 1998:
        return ['L5_SR']
    if year <= 2011:
        return ['L5_SR', 'L7_SR']
    if year == 2012:
        return ['L7_SR']
    if year <= 2020:
        return ['L8_SR']
    return ['L8_SR', 'L9_SR']

#configure sensor selection (HLS collection)
def _sensor_for_hls_year(year: int) -> str:
    """Return `'HLS_L30'` for 2013-2014 (pre-Sentinel-2) and merged `'HLS'` for 2015+."""
    if year < _HLS_MIN_YEAR:
        raise ValueError(f"NASA HLS v2.0 is unavailable before {_HLS_MIN_YEAR} (got {year}).")
    return 'HLS_L30' if year < 2015 else 'HLS'


def _compute_inverted_driver_band(image: ee.Image, driver_index: str = 'NBR') -> ee.Image:
    """
    Compute the LandTrendr Band 0 segmentation driver.

    GEE's ``LandTrendr`` assumes that **increasing** values represent disturbance
    and **decreasing** values represent recovery. Vegetation/burn indices (`NBR`,
    `NDVI`, `NDMI`) decrease upon clearing/fire, so we multiply them by ``-1.0``.
    """
    idx_key = str(driver_index).strip().upper()
    if idx_key not in DRIVER_INDEX_CHOICES:
        raise ValueError(
            f"Unsupported driver_index '{driver_index}'. Supported: {sorted(DRIVER_INDEX_CHOICES)}"
        )

    if idx_key == 'NBR':
        idx_img = image.normalizedDifference(['NIR', 'SWIR2'])
    elif idx_key == 'NDVI':
        idx_img = image.normalizedDifference(['NIR', 'RED'])
    else:  # NDMI
        idx_img = image.normalizedDifference(['NIR', 'SWIR1'])

    return idx_img.multiply(-1.0).toFloat().rename('lt_driver')

#core class for LandtrendR
class LandTrendr_fit:
    """
    Opt-in Module 1 temporal segmentation compositor powered by Google Earth
    Engine's native ``ee.Algorithms.TemporalSegmentation.LandTrendr`` in
    Fitted-To-Vertex (``FTV``) mode.

    Builds a continuous annual harmonized 30m median composite stack across
    ``[stack_start_year .. stack_end_year]`` (auto-padded to ``>= min_years``),
    fits shared piecewise-linear temporal breakpoints across all 6 optical bands
    (``BLUE, GREEN, RED, NIR, SWIR1, SWIR2``), and extracts despiked, gap-filled
    surface reflectance composites for each requested mapping period.
    Main Reference: https://emapr.github.io/LT-GEE/introduction.html
    """

    SUPPORTED_SENSORS = SUPPORTED_LT_SENSORS
    OPTICAL_BANDS = LT_OPTICAL_BANDS
    DEFAULT_PARAMS = DEFAULT_LANDTRENDR_PARAMS

    def __init__(self, log_level: int = logging.INFO) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)

    # ------------------------------------------------------------------
    # 1. Validation & Continuous Annual Window Resolution
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_period_list(periods: Any) -> List[Dict[str, Any]]:
        """Normalize dict or sequence of PeriodConfig/dicts into a list of period dicts."""
        norm: List[Dict[str, Any]] = []
        if isinstance(periods, dict):
            for lbl, cfg in periods.items():
                d = cfg.to_dict() if hasattr(cfg, 'to_dict') else dict(cfg)
                d.setdefault('label', str(lbl))
                norm.append(d)
        elif isinstance(periods, (list, tuple)):
            for idx, cfg in enumerate(periods):
                d = cfg.to_dict() if hasattr(cfg, 'to_dict') else dict(cfg)
                d.setdefault('label', f'period_{idx}')
                norm.append(d)
        else:
            raise ValueError("periods must be a dict or sequence of PeriodConfig/dict objects.")
        return norm

    def validate_and_resolve_window(
        self,
        periods: Any,
        reference_period: str = 'baseline',
        min_years: int = 6,
        seasonal_window: Optional[Tuple[str, str]] = None,
        max_allowed_year: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Validate sensor compatibility and resolve the continuous annual stack
        ``[stack_start_year .. stack_end_year]`` for LandTrendr segmentation.
        """
        if min_years < 3:
            raise ValueError(f"min_years must be >= 3, got {min_years}.")

        norm_periods = self._normalize_period_list(periods)
        if len(norm_periods) < 2:
            raise ValueError(
                f"Multitemporal LandTrendr segmentation requires >= 2 periods, got {len(norm_periods)}."
            )

        period_order = [str(p['label']) for p in norm_periods]
        sensors_by_period: Dict[str, str] = {}
        unsupported: Dict[str, str] = {}
        target_year_by_period: Dict[str, int] = {}
        mmdd_by_period: Dict[str, Tuple[str, str]] = {}

        for p in norm_periods:
            lbl = str(p['label'])
            sensor = str(p.get('sensor') or p.get('optical_data') or 'L8_SR').strip()
            sensors_by_period[lbl] = sensor
            if sensor not in SUPPORTED_LT_SENSORS:
                unsupported[lbl] = sensor

            s_year, s_mmdd = parse_year(p['start_date'], is_start=True)
            e_year, e_mmdd = parse_year(p['end_date'], is_start=False)
            if e_year < s_year:
                raise ValueError(f"Period '{lbl}' has end_year {e_year} < start_year {s_year}.")

            target_year_by_period[lbl] = (s_year + e_year) // 2
            mmdd_by_period[lbl] = (s_mmdd, e_mmdd)

        if unsupported:
            return {
                'compatible': False,
                'reason': (
                    f"LandTrendr FTV temporal segmentation supports 30m Landsat SR (L4-L9) and HLS only; "
                    f"unsupported sensor(s): {unsupported}."
                ),
                'unsupported_sensors': unsupported,
                'sensor_mode': None,
                'years': [],
                'target_year_by_period': target_year_by_period,
                'period_order': period_order,
                'seasonal_window': seasonal_window or ('01-01', '12-31'),
                'padded': False,
            }

        if seasonal_window is not None:
            resolved_window = (str(seasonal_window[0]), str(seasonal_window[1]))
        else:
            ref_lbl = reference_period if reference_period in mmdd_by_period else period_order[-1]
            resolved_window = mmdd_by_period[ref_lbl]

        all_hls = all(s in HLS_SENSOR_KEYS for s in sensors_by_period.values())
        min_target_year = min(target_year_by_period.values())
        max_target_year = max(target_year_by_period.values())

        sensor_mode = 'hls' if (all_hls and min_target_year >= _HLS_MIN_YEAR) else 'landsat_harmonized'
        launch_floor = _HLS_MIN_YEAR if sensor_mode == 'hls' else _LANDSAT_SR_MIN_YEAR
        current_year = max_allowed_year if max_allowed_year is not None else datetime.date.today().year
        year_ceiling = max(current_year, max_target_year)

        start_y = min_target_year
        end_y = max_target_year
        padded = False

        if (end_y - start_y + 1) < min_years:
            needed = min_years - (end_y - start_y + 1)
            pad_back = min(needed, max(0, start_y - launch_floor))
            start_y -= pad_back
            remaining = needed - pad_back
            if remaining > 0:
                pad_fwd = min(remaining, max(0, year_ceiling - end_y))
                end_y += pad_fwd
            padded = True

        if (end_y - start_y + 1) < min_years and sensor_mode == 'hls':
            sensor_mode = 'landsat_harmonized'
            launch_floor = _LANDSAT_SR_MIN_YEAR
            needed = min_years - (end_y - start_y + 1)
            pad_back = min(needed, max(0, start_y - launch_floor))
            start_y -= pad_back
            padded = True

        years = list(range(start_y, end_y + 1))

        return {
            'compatible': True,
            'reason': '',
            'unsupported_sensors': {},
            'sensor_mode': sensor_mode,
            'years': years,
            'target_year_by_period': target_year_by_period,
            'period_order': period_order,
            'seasonal_window': resolved_window,
            'padded': padded,
        }

    # ------------------------------------------------------------------
    # 2. Harmonized Annual Median Stack Builder (delegates to data_acquisition)
    # ------------------------------------------------------------------
    def _build_single_year_composite(
        self,
        rd: Any,
        fi: Any,
        aoi: Any,
        geometry: ee.Geometry,
        year: int,
        sensor_mode: str,
        seasonal_window: Tuple[str, str],
        cloud_cover: float,
        reducer: str,
        apply_brdf: bool,
        apply_topo_correction: bool,
        dem: Optional[ee.Image],
        brdf_corrector: Optional[Any],
        topo_corrector: Optional[Any],
    ) -> ee.Image:
        """Build one harmonized 6-band annual composite delegating to `Reflectance_Data`."""
        s_mmdd, e_mmdd = seasonal_window
        start_d = f"{year}-{s_mmdd}"
        end_d = f"{year}-{e_mmdd}"

        if sensor_mode == 'hls':
            hls_key = _sensor_for_hls_year(year)
            col, _ = rd.get_hls_optical_data(
                aoi=aoi,
                start_date=start_d,
                end_date=end_d,
                dataset=hls_key,
                cloud_cover=cloud_cover,
                verbose=False,
                compute_detailed_stats=False,
            )
            if apply_topo_correction and topo_corrector is not None:
                col = topo_corrector.correct_collection(col, dem=dem, scale=30, verbose=False)
            merged_col = col.map(lambda img: img.select(LT_OPTICAL_BANDS).toFloat())
        else:
            sensors = _sensors_for_landsat_year(year)
            sub_cols: List[ee.ImageCollection] = []
            for sensor_key in sensors:
                c, _ = rd.get_optical_data(
                    aoi=aoi,
                    start_date=start_d,
                    end_date=end_d,
                    optical_data=sensor_key,
                    cloud_cover=cloud_cover,
                    verbose=False,
                    compute_detailed_stats=False,
                )
                sensor_code = rd.OPTICAL_DATASETS[sensor_key]['sensor']
                if sensor_code in ('L4', 'L5', 'L7'):
                    def _make_harm_fn(sc: str):
                        return lambda img: rd.apply_brdf_harmonization(img, sc)
                    c = c.map(_make_harm_fn(sensor_code))
                if apply_brdf and brdf_corrector is not None:
                    c = brdf_corrector.correct_collection(c, verbose=False)
                if apply_topo_correction and topo_corrector is not None:
                    c = topo_corrector.correct_collection(c, dem=dem, scale=30, verbose=False)
                sub_cols.append(c.map(lambda img: img.select(LT_OPTICAL_BANDS).toFloat()))

            merged_col = sub_cols[0]
            for extra in sub_cols[1:]:
                merged_col = merged_col.merge(extra)

        empty_img = (
            ee.Image.constant([0.0] * len(LT_OPTICAL_BANDS))
            .rename(LT_OPTICAL_BANDS)
            .toFloat()
            .updateMask(ee.Image.constant(0))
            .clip(geometry)
        )
        comp_img = fi.get_temporal_composite(
            collection=merged_col,
            aoi=aoi,
            reducer=reducer,
            calculate_coverage=False,
            verbose=False,
            return_image_only=True,
        )
        safe_comp = ee.Image(
            ee.Algorithms.If(
                merged_col.size().gt(0),
                ee.Image(comp_img).select(LT_OPTICAL_BANDS).toFloat().clip(geometry),
                empty_img,
            )
        )
        return ee.Image(safe_comp.set({
            'system:time_start': ee.Date.fromYMD(int(year), 7, 1).millis(),
            'year': int(year),
        }))

    def build_annual_collection(
        self,
        aoi: Any,
        years: Sequence[int],
        sensor_mode: str = 'landsat_harmonized',
        seasonal_window: Tuple[str, str] = ('01-01', '12-31'),
        cloud_cover: float = 40.0,
        reducer: str = 'median',
        apply_brdf: bool = False,
        apply_topo_correction: bool = False,
        dem: Optional[ee.Image] = None,
        driver_index: str = 'NBR',
    ) -> Tuple[ee.ImageCollection, Dict[int, ee.Image]]:
        """
        Build a continuous annual ``ee.ImageCollection`` ready for GEE ``LandTrendr``.

        Each image has Band 0 = inverted ``driver_index`` (``'lt_driver'``) followed
        by the 6 canonical optical bands ``['BLUE', 'GREEN', 'RED', 'NIR', 'SWIR1', 'SWIR2']``.
        """
        ensure_ee_initialized()
        from .data_acquisition import Reflectance_Data, final_Image

        rd = Reflectance_Data()
        fi = final_Image()
        geometry = aoi.geometry() if isinstance(aoi, ee.FeatureCollection) else aoi

        brdf_corrector = None
        topo_corrector = None
        if apply_brdf or apply_topo_correction:
            from .imagery_correction import BRDF_Correction, Topographic_Correction
            if apply_brdf:
                brdf_corrector = BRDF_Correction(log_level=self.logger.level)
            if apply_topo_correction:
                topo_corrector = Topographic_Correction(log_level=self.logger.level)

        lt_frames: List[ee.Image] = []
        raw_by_year: Dict[int, ee.Image] = {}

        for y in sorted(int(yr) for yr in years):
            raw_comp = self._build_single_year_composite(
                rd=rd,
                fi=fi,
                aoi=aoi,
                geometry=geometry,
                year=y,
                sensor_mode=sensor_mode,
                seasonal_window=seasonal_window,
                cloud_cover=cloud_cover,
                reducer=reducer,
                apply_brdf=apply_brdf,
                apply_topo_correction=apply_topo_correction,
                dem=dem,
                brdf_corrector=brdf_corrector,
                topo_corrector=topo_corrector,
            )
            raw_by_year[y] = raw_comp

            driver_band = _compute_inverted_driver_band(raw_comp, driver_index=driver_index)
            lt_frame = (
                ee.Image.cat([driver_band, raw_comp.select(LT_OPTICAL_BANDS)])
                .toFloat()
                .set({
                    'system:time_start': ee.Date.fromYMD(y, 7, 1).millis(),
                    'year': y,
                })
            )
            lt_frames.append(lt_frame) # type: ignore

        return ee.ImageCollection.fromImages(lt_frames), raw_by_year

    # ------------------------------------------------------------------
    # 3. Native GEE LandTrendr Execution & FTV Extraction
    # ------------------------------------------------------------------
    def run_landtrendr(
        self,
        annual_collection: ee.ImageCollection,
        n_years: Optional[int] = None,
        lt_params: Optional[Dict[str, Any]] = None,
    ) -> Tuple[ee.Image, Dict[str, Any]]:
        """Execute GEE's native server-side ``ee.Algorithms.TemporalSegmentation.LandTrendr``."""
        ensure_ee_initialized()
        merged = dict(DEFAULT_LANDTRENDR_PARAMS)
        if lt_params:
            merged.update(lt_params)

        if n_years is not None and merged.get('minObservationsNeeded', 6) > n_years:
            merged['minObservationsNeeded'] = max(3, int(n_years))

        lt_result = ee.Algorithms.TemporalSegmentation.LandTrendr(
            timeSeries=annual_collection,
            **merged,
        )
        return lt_result, merged

    @staticmethod
    def extract_fitted_year(
        lt_result: ee.Image,
        target_year: int,
        years: Sequence[int],
        bands: Sequence[str] = LT_OPTICAL_BANDS,
        fallback_image: Optional[ee.Image] = None,
        geometry: Optional[ee.Geometry] = None,
    ) -> ee.Image:
        """
        Extract a gap-filled, despiked multi-band FTV composite for ``target_year``
        from GEE ``LandTrendr``'s ``<band>_fit`` arrays.

        In GEE ``LandTrendr``, each secondary ``<band>_fit`` array already has fixed
        length ``len(years)`` with missing/cloudy years interpolated along the fitted
        segments. Pixels whose ``<band>_fit`` array is masked or shorter than
        ``len(years)`` (e.g. ``< minObservationsNeeded``) are safely masked before
        indexing and unmasked back to ``fallback_image``.
        """
        sorted_years = [int(y) for y in years]
        if int(target_year) not in sorted_years:
            raise ValueError(f"target_year {target_year} not found in LandTrendr years {sorted_years}.")
        year_idx = sorted_years.index(int(target_year))
        n_years = len(sorted_years)

        fitted_bands: List[ee.Image] = []
        for b in bands:
            fit_arr = lt_result.select(f"{b}_fit")
            valid_len = fit_arr.arrayLength(0).eq(n_years)
            safe_val = fit_arr.updateMask(valid_len).arrayGet([year_idx]).toFloat().rename(b)
            fitted_bands.append(safe_val)

        fitted_img = ee.Image.cat(fitted_bands).toFloat()

        if fallback_image is not None:
            fitted_img = fitted_img.unmask(fallback_image.select(list(bands)).toFloat())

        if geometry is not None:
            fitted_img = fitted_img.clip(geometry)

        return ee.Image(fitted_img.set({
            'temporal_segmentation': 'landtrendr_ftv',
            'target_year': int(target_year),
        }))

    def extract_fitted_composites(
        self,
        lt_result: ee.Image,
        years: Sequence[int],
        target_year_by_period: Dict[str, int],
        raw_composites_by_year: Optional[Dict[int, ee.Image]] = None,
        bands: Sequence[str] = LT_OPTICAL_BANDS,
        aoi: Optional[Any] = None,
    ) -> Dict[str, ee.Image]:
        """Extract fitted multi-band composites for all target periods."""
        geometry = aoi.geometry() if isinstance(aoi, ee.FeatureCollection) else aoi
        out: Dict[str, ee.Image] = {}
        for label, t_year in target_year_by_period.items():
            fallback = (raw_composites_by_year or {}).get(t_year)
            out[label] = self.extract_fitted_year(
                lt_result=lt_result,
                target_year=t_year,
                years=years,
                bands=bands,
                fallback_image=fallback,
                geometry=geometry,
            )
        return out

    # ------------------------------------------------------------------
    # 4. End-to-End Module 1 Entry Point
    # ------------------------------------------------------------------
    def generate_multitemporal_composites(self, aoi: Any, periods: Any,
        reference_period: str = 'baseline',
        driver_index: str = 'NBR',
        min_years: int = 6,
        seasonal_window: Optional[Tuple[str, str]] = None,
        cloud_cover: float = 40.0,
        reducer: str = 'median',
        apply_brdf: bool = False,
        apply_topo_correction: bool = False,
        dem: Optional[ee.Image] = None,
        lt_params: Optional[Dict[str, Any]] = None,
        lt_image: Optional[ee.Image] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate despiked, gap-filled multitemporal surface reflectance composites
        via native GEE ``LandTrendr`` FTV (or slice from a pre-exported ``lt_image``
        GEE Asset).

        Note
        ----
        CDF histogram matching is intentionally NOT applied after ``LandTrendr`` FTV
        because the annual stack is already radiometrically harmonized and fitted
        along a single continuous temporal trajectory.
        """
        ensure_ee_initialized()

        win = self.validate_and_resolve_window(
            periods=periods,
            reference_period=reference_period,
            min_years=min_years,
            seasonal_window=seasonal_window,
        )
        if not win['compatible']:
            raise ValueError(win['reason'])

        period_order = win['period_order']
        years = win['years']
        target_year_by_period = win['target_year_by_period']
        sensor_mode = win['sensor_mode']
        resolved_window = win['seasonal_window']

        if reference_period not in period_order:
            reference_period = period_order[-1]

        if verbose:
            self.logger.info(
                "LandTrendr FTV compositing: mode=%s, years=%d..%d (%d yrs, padded=%s), "
                "seasonal_window=%s, driver=-%s, precomputed_asset=%s",
                sensor_mode,
                years[0],
                years[-1],
                len(years),
                win['padded'],
                resolved_window,
                driver_index,
                lt_image is not None,
            )

        raw_by_year: Dict[int, ee.Image] = {}
        if lt_image is not None:
            lt_result = lt_image
            merged_params = dict(DEFAULT_LANDTRENDR_PARAMS)
            if lt_params:
                merged_params.update(lt_params)
        else:
            annual_col, raw_by_year = self.build_annual_collection(
                aoi=aoi,
                years=years,
                sensor_mode=sensor_mode,
                seasonal_window=resolved_window,
                cloud_cover=cloud_cover,
                reducer=reducer,
                apply_brdf=apply_brdf,
                apply_topo_correction=apply_topo_correction,
                dem=dem,
                driver_index=driver_index,
            )
            lt_result, merged_params = self.run_landtrendr(
                annual_collection=annual_col,
                n_years=len(years),
                lt_params=lt_params,
            )

        fitted_by_period = self.extract_fitted_composites(
            lt_result=lt_result,
            years=years,
            target_year_by_period=target_year_by_period,
            raw_composites_by_year=raw_by_year,
            bands=LT_OPTICAL_BANDS,
            aoi=aoi,
        )
        raw_by_period = {
            lbl: raw_by_year[target_year_by_period[lbl]]
            for lbl in period_order
            if target_year_by_period[lbl] in raw_by_year
        }

        return {
            'composites': fitted_by_period,
            'raw_composites': raw_by_period,
            'metadata': {
                lbl: {
                    'method': 'landtrendr_ftv',
                    'target_year': target_year_by_period[lbl],
                    'sensor_mode': sensor_mode,
                    'seasonal_window': resolved_window,
                }
                for lbl in period_order
            },
            'coverage': {lbl: None for lbl in period_order},
            'common_bands': list(LT_OPTICAL_BANDS),
            'reference_period': reference_period,
            'period_order': period_order,
            'histogram_matched': False,
            'temporal_segmentation': {
                'method': 'landtrendr_ftv',
                'driver_index': str(driver_index).strip().upper(),
                'sensor_mode': sensor_mode,
                'years': years,
                'padded': win['padded'],
                'seasonal_window': resolved_window,
                'target_year_by_period': target_year_by_period,
                'lt_params': merged_params,
                'lt_result': lt_result,
                'used_precomputed_asset': lt_image is not None,
            },
        }
