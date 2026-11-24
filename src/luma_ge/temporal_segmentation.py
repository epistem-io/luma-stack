"""
Temporal Segmentation Compositing Module

Provides opt-in native Google Earth Engine temporal segmentation compositing for multitemporal LULC mapping.
currently supported via LandTrendr (Landsat-based detection of Trends in Disturbance and Recovery) Fitted-To-Vertex / FTV.
More algorithms may be added in the future.
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
#recheck is requried
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

#core class for LandTrendr
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
        annual_col: Optional[ee.ImageCollection] = None
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
                'annual_collection': annual_col,
                'used_precomputed_asset': lt_image is not None,
            },
        }

    # ------------------------------------------------------------------
    # 5. Pixel Trajectory Diagnostics
    # ------------------------------------------------------------------
    @staticmethod
    def _to_point(point: Any) -> Any:
        """Accept a ``(lon, lat)`` tuple/list or an existing ``ee.Geometry``."""
        if isinstance(point, (list, tuple)):
            if len(point) != 2:
                raise ValueError(f"point must be (lon, lat), got {point!r}.")
            return ee.Geometry.Point([float(point[0]), float(point[1])])
        return point

    def sample_pixel_trajectory(
        self,
        lt_result: ee.Image,
        point: Any,
        years: Sequence[int],
        bands: Sequence[str] = ('SWIR1',),
        annual_collection: Optional[ee.ImageCollection] = None,
        driver_index: str = 'NBR',
        scale: int = 30,
    ) -> Any:
        """
        Sample the LandTrendr trajectory at a single pixel into a ``pandas.DataFrame``.

        Columns (indexed by ``year``):
        - ``<DRIVER>_observed`` / ``<DRIVER>_fitted``: segmentation index from the
          ``LandTrendr`` array (rows 1 and 2), shown in its natural orientation
          (the internal ``-NBR`` / ``-NDVI`` / ``-NDMI`` sign is undone).
        - ``is_vertex``: ``True`` for years LandTrendr picked as segment vertices (row 3).
        - ``<band>_fitted``: FTV values from ``<band>_fit``.
        - ``<band>_observed``: annual median composite values from ``annual_collection``
          (``NaN`` when not supplied, e.g. pre-exported ``lt_image`` assets).

        Raises ``ValueError`` if the pixel has no LandTrendr fit (masked, or fewer
        clear years than ``minObservationsNeeded``).
        """
        import math
        import pandas as pd

        ensure_ee_initialized()
        band_list = [str(b) for b in bands]
        unknown = [b for b in band_list if b not in LT_OPTICAL_BANDS]
        if unknown:
            raise ValueError(f"Unsupported band(s) {unknown}. Supported: {LT_OPTICAL_BANDS}")

        geom = self._to_point(point)
        drv = str(driver_index).strip().upper()
        year_index = [int(y) for y in years]

        lt_vals = (
            lt_result.select(['LandTrendr'] + [f"{b}_fit" for b in band_list])
            .reduceRegion(reducer=ee.Reducer.first(), geometry=geom, scale=scale)
            .getInfo()
        ) or {}
        lt_arr = lt_vals.get('LandTrendr')
        if not lt_arr or len(lt_arr) < 4 or not lt_arr[0]:
            raise ValueError(
                "No LandTrendr fit at this point (masked pixel or fewer clear years than "
                "minObservationsNeeded). Try another location inside the AOI."
            )

        yrs_row, obs_row, fit_row, vtx_row = lt_arr[0], lt_arr[1], lt_arr[2], lt_arr[3]
        drv_obs = {int(y): -float(v) for y, v in zip(yrs_row, obs_row) if v is not None}
        drv_fit = {int(y): -float(v) for y, v in zip(yrs_row, fit_row) if v is not None}
        vtx = {int(y): bool(v) for y, v in zip(yrs_row, vtx_row) if v is not None}

        df = pd.DataFrame(index=pd.Index(year_index, name='year'))
        df[f"{drv}_observed"] = [drv_obs.get(y, math.nan) for y in year_index]
        df[f"{drv}_fitted"] = [drv_fit.get(y, math.nan) for y in year_index]
        df['is_vertex'] = [vtx.get(y, False) for y in year_index]

        for b in band_list:
            fit_vals = lt_vals.get(f"{b}_fit") or []
            if len(fit_vals) == len(year_index):
                df[f"{b}_fitted"] = [math.nan if v is None else float(v) for v in fit_vals]
            else:
                df[f"{b}_fitted"] = math.nan

        observed: Dict[str, Dict[int, float]] = {b: {} for b in band_list}
        if annual_collection is not None:
            region = annual_collection.select(band_list).getRegion(geom, scale).getInfo() or []
            if len(region) > 1:
                header = region[0]
                t_idx = header.index('time')
                b_idx = {b: header.index(b) for b in band_list if b in header}
                for row in region[1:]:
                    if row[t_idx] is None:
                        continue
                    yr = datetime.datetime.fromtimestamp(
                        row[t_idx] / 1000.0, tz=datetime.timezone.utc
                    ).year
                    for b, i in b_idx.items():
                        if row[i] is not None:
                            observed[b][yr] = float(row[i])
        for b in band_list:
            df[f"{b}_observed"] = [observed[b].get(y, math.nan) for y in year_index]

        df.attrs['driver_index'] = drv
        df.attrs['bands'] = band_list
        return df

    @staticmethod
    def plot_pixel_trajectory(
        df: Any,
        bands: Optional[Sequence[str]] = None,
        show_driver: bool = True,
        highlight_years: Optional[Dict[str, int]] = None,
        title: Optional[str] = None,
        figsize: Optional[Tuple[float, float]] = None,
    ) -> Any:
        """
        Plot an LT-GEE style sanity-check figure:
        1. Top header strip showing ``<DRIVER>-identified vertex years`` as red dots.
        2. For each requested series, an **Observed** panel (grey annual dots with
           downward dashed arrows marking vertex years) above a **Fitted** panel
           (black piecewise-linear FTV trajectory with grey non-vertex dots and red
           vertex dots).

        Parameters
        ----------
        df : pandas.DataFrame
            Output of ``sample_pixel_trajectory``.
        bands : sequence of str, optional
            Bands to plot (defaults to ``df.attrs['bands']``).
        show_driver : bool, default=True
            Also plot the segmentation driver index (e.g. ``NBR``).
        highlight_years : dict, optional
            ``{period_label: year}`` mapping target periods to shade lightly.

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt

        drv = df.attrs.get('driver_index', 'NBR')
        band_list = list(bands) if bands is not None else list(df.attrs.get('bands', []))
        series = ([drv] if show_driver else []) + band_list
        if not series:
            raise ValueError("Nothing to plot: pass bands or set show_driver=True.")

        n_data_rows = 2 * len(series)
        height_ratios = [0.42] + [1.0] * n_data_rows
        fig, axes = plt.subplots(
            1 + n_data_rows,
            1,
            sharex=True,
            figsize=figsize or (10, 0.9 + 2.3 * n_data_rows),
            gridspec_kw={'height_ratios': height_ratios},
        )
        vertex_years = list(df.index[df['is_vertex'].astype(bool)])
        non_vertex_years = [y for y in df.index if y not in vertex_years]
        years = list(df.index)

        # Row 0: LT-GEE style "<DRIVER>-identified vertex years" strip
        ax_hdr = axes[0]
        ax_hdr.axhline(0.88, color='0.45', linewidth=1.0)
        ax_hdr.axhline(0.12, color='0.45', linewidth=1.0)
        if vertex_years:
            ax_hdr.scatter(
                vertex_years,
                [0.5] * len(vertex_years),
                color='#ff2a1a',
                edgecolor='0.2',
                s=85,
                zorder=4,
            )
            label_x = vertex_years[0] + 0.35
        else:
            label_x = years[0]
        ax_hdr.text(
            label_x,
            0.5,
            f"  {drv}-identified vertex years",
            va='center',
            ha='left',
            fontsize=9.5,
            color='0.15',
            bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 1.5},
            zorder=3,
        )
        ax_hdr.set_ylim(0.0, 1.0)
        ax_hdr.set_yticks([])
        for sp in ('top', 'right', 'left', 'bottom'):
            ax_hdr.spines[sp].set_visible(False)
        ax_hdr.tick_params(axis='x', bottom=False)

        for i, name in enumerate(series):
            ax_obs = axes[1 + 2 * i]
            ax_fit = axes[1 + 2 * i + 1]
            obs = df[f"{name}_observed"]
            fit = df[f"{name}_fitted"]

            valid_vals = list(obs.dropna().values) + list(fit.dropna().values)
            if valid_vals:
                ymin, ymax = min(valid_vals), max(valid_vals)
                pad = max((ymax - ymin) * 0.22, 0.015)
            else:
                ymin, ymax, pad = 0.0, 1.0, 0.1

            # --- Observed panel ---
            obs_valid = obs.dropna()
            ax_obs.scatter(
                obs_valid.index,
                obs_valid.values,
                color='0.52',
                edgecolor='0.25',
                s=58,
                zorder=3,
            )
            y_top = ymax + pad
            ax_obs.set_ylim(ymin - pad * 0.6, y_top)
            for vy in vertex_years:
                target_y = obs.loc[vy] if (vy in obs.index and obs.notna().loc[vy]) else fit.loc[vy]
                if target_y == target_y:  # not NaN
                    ax_obs.annotate(
                        '',
                        xy=(vy, float(target_y)),
                        xytext=(vy, y_top),
                        arrowprops={
                            'arrowstyle': '-|>',
                            'linestyle': ':',
                            'linewidth': 1.1,
                            'color': '0.15',
                            'shrinkA': 0,
                            'shrinkB': 5,
                        },
                        zorder=2,
                    )
            ax_obs.set_ylabel(f"{name}\nobserved")

            # --- Fitted panel ---
            fit_line = fit.dropna()
            ax_fit.plot(fit_line.index, fit_line.values, color='black', linewidth=1.8, zorder=2)
            fit_nv = fit.reindex(non_vertex_years).dropna()
            if not fit_nv.empty:
                ax_fit.scatter(
                    fit_nv.index,
                    fit_nv.values,
                    color='0.52',
                    edgecolor='0.25',
                    s=48,
                    zorder=3,
                    label='annual fitted' if i == 0 else None,
                )
            fit_v = fit.reindex(vertex_years).dropna()
            if not fit_v.empty:
                ax_fit.scatter(
                    fit_v.index,
                    fit_v.values,
                    color='#ff2a1a',
                    edgecolor='0.15',
                    s=72,
                    zorder=4,
                    label='segment vertex' if i == 0 else None,
                )
            ax_fit.set_ylim(ymin - pad * 0.6, ymax + pad)
            ax_fit.set_ylabel(f"{name}\nLandTrendr fit")

            for ax in (ax_obs, ax_fit):
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.grid(axis='y', alpha=0.22)
                for lbl, yr in (highlight_years or {}).items():
                    ax.axvspan(yr - 0.35, yr + 0.35, color='tab:blue', alpha=0.10, zorder=0)

        if highlight_years:
            for lbl, yr in highlight_years.items():
                axes[1].annotate(
                    lbl,
                    xy=(yr, 1.0),
                    xycoords=('data', 'axes fraction'),
                    ha='center',
                    va='bottom',
                    fontsize=8,
                    color='tab:blue',
                )

        axes[-1].set_xlabel('Years')
        axes[-1].set_xticks(years)
        axes[-1].tick_params(axis='x', labelrotation=45 if len(years) > 12 else 0)
        axes[2].legend(loc='best', fontsize=8, frameon=False)
        if title:
            fig.suptitle(title, y=1.01)
        fig.tight_layout()
        return fig

    def plot_trajectory_from_result(
        self,
        result: Dict[str, Any],
        point: Any,
        bands: Sequence[str] = ('SWIR1',),
        scale: int = 30,
        show_target_years: bool = True,
        **plot_kwargs: Any,
    ) -> Tuple[Any, Any]:
        """
        Convenience wrapper: sample and plot a pixel trajectory directly from the
        dict returned by ``generate_multitemporal_composites`` /
        ``final_Image.get_landtrendr_composites`` (or ``MultitemporalAnalysis.acquire_imagery()``).

        Returns
        -------
        tuple of (matplotlib.figure.Figure, pandas.DataFrame)
        """
        ts_meta = result.get('temporal_segmentation')
        if ts_meta is None and isinstance(result.get('composites'), dict):
            ts_meta = result['composites'].get('temporal_segmentation')
        if not ts_meta:
            raise ValueError(
                "result has no 'temporal_segmentation' metadata. Run with "
                "temporal_segmentation=True (LandTrendr FTV) first."
            )

        df = self.sample_pixel_trajectory(
            lt_result=ts_meta['lt_result'],
            point=point,
            years=ts_meta['years'],
            bands=bands,
            annual_collection=ts_meta.get('annual_collection'),
            driver_index=ts_meta.get('driver_index', 'NBR'),
            scale=scale,
        )
        if show_target_years and 'highlight_years' not in plot_kwargs:
            plot_kwargs['highlight_years'] = ts_meta.get('target_year_by_period')
        fig = self.plot_pixel_trajectory(df, bands=bands, **plot_kwargs)
        return fig, df
