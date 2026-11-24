"""
Multitemporal Analysis Modules

Provides:
- Conflict-safe multitemporal predictor computation & cross-sensor index filtering
  (without modifying `src/luma_ge/predictor.py` while upstream predictor branches
  are pending merge).
- Lightweight orchestration (`PeriodConfig`, `MultitemporalAnalysis`) across N >= 2
  mapping periods for Batch and Model Transfer workflows.
"""

from dataclasses import dataclass
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import ee
from .ee_config import ensure_ee_initialized
from .predictor import (
    RFE_CV,
    SpectralCalculator,
    distance_calculator,
    predictor_catalog,
    terrain_calculator,
)
from .sar import get_sar_data

logger = logging.getLogger(__name__)

# Mapping from spyndex short band codes to luma_ge standardized band names
_SPYNDEX_TO_LUMA_BAND = {
    'N': 'NIR',
    'R': 'RED',
    'G': 'GREEN',
    'B': 'BLUE',
    'S1': 'SWIR1',
    'S2': 'SWIR2',
    'RE1': 'RED_EDGE1',
    'RE2': 'RED_EDGE2',
    'RE3': 'RED_EDGE3',
    'RE4': 'RED_EDGE4',
    'T': 'THERMAL',
}

_LUMA_TO_SPYNDEX_BAND = {v: k for k, v in _SPYNDEX_TO_LUMA_BAND.items()}


def filter_compatible_indices(
    index_list: Sequence[str],
    available_bands: Sequence[str],
    spectral_calc: Optional[SpectralCalculator] = None,
) -> Tuple[List[str], List[str]]:
    """
    Partition ``index_list`` into ``(compatible_indices, skipped_indices)`` based
    on whether all optical bands required by each spyndex index exist in
    ``available_bands``.

    Ensures Model Transfer across restricted band intersections (e.g., Landsat MSS
    `['GREEN', 'RED', 'NIR']` paired with Landsat 5/8) cleanly drops indices that
    require missing bands (`BLUE`, `SWIR1`, `SWIR2`).
    """
    if not index_list:
        return [], []

    calc = spectral_calc or SpectralCalculator()
    avail_set = set(available_bands)
    compatible: List[str] = []
    skipped: List[str] = []

    for idx_name in index_list:
        if idx_name not in calc.spyndex_indices:
            logger.warning(f"Spectral index '{idx_name}' not found in spyndex; skipping.")
            skipped.append(idx_name)
            continue

        idx_info = calc.spyndex_indices[idx_name]
        req_codes = getattr(idx_info, 'bands', []) or []
        req_luma_bands = [_SPYNDEX_TO_LUMA_BAND.get(code, code) for code in req_codes]

        # Only check optical/thermal band codes present in our mapping; constants are in base_params
        missing = [
            b for b in req_luma_bands
            if b in _LUMA_TO_SPYNDEX_BAND and b not in avail_set
        ]
        if missing:
            logger.warning(
                f"Skipping spectral index '{idx_name}': requires {missing} not in {sorted(avail_set)}"
            )
            skipped.append(idx_name)
        else:
            compatible.append(idx_name)

    return compatible, skipped


def compute_single_image_indices(
    image: ee.Image,
    index_list: Sequence[str],
    available_bands: Optional[Sequence[str]] = None,
    coefficients: Optional[Dict[str, Dict[str, float]]] = None,
    spectral_calc: Optional[SpectralCalculator] = None,
) -> Optional[ee.Image]:
    """
    Compute spectral indices directly on a single composite ``ee.Image``.

    Delegates to ``spectral_calc.calculate_indices()`` if available (e.g., when
    `predictor_improvements` is merged), or falls back to a dynamic-band
    ``spyndex.computeIndex`` call selecting only bands present in
    ``available_bands``.
    """
    if not index_list:
        return None

    calc = spectral_calc or SpectralCalculator()

    # Delegate if SpectralCalculator exposes calculate_indices() and available_bands wasn't pre-resolved
    if hasattr(calc, 'calculate_indices') and available_bands is None:
        return calc.calculate_indices(
            image=image,
            index_list=list(index_list),
            coefficients=coefficients,
        )

    if available_bands is None:
        available_bands = list(image.bandNames().getInfo())

    compatible, _ = filter_compatible_indices(index_list, available_bands, spectral_calc=calc)
    if not compatible:
        return None

    params: Dict[str, Any] = {}
    avail_set = set(available_bands)
    for luma_band, spy_code in _LUMA_TO_SPYNDEX_BAND.items():
        if luma_band in avail_set:
            params[spy_code] = image.select(luma_band)

    # Resolve coefficients (custom overrides default)
    for idx_name in compatible:
        if coefficients and idx_name in coefficients:
            validation = calc.validate_coefficients(idx_name, coefficients[idx_name])
            if not validation['valid']:
                raise ValueError(f"Invalid coefficients for {idx_name}: {validation['warnings']}")
            params.update(coefficients[idx_name])
        else:
            defaults = calc.get_index_coefficients(idx_name)
            if defaults:
                params.update(defaults)

    return calc.spyndex.computeIndex(index=compatible, params=params)


def _extract_predictor_selections(predictor_config: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize both `individual_predictors` and flat `predictor_config` schemas."""
    if not predictor_config:
        return {
            'elevation': False,
            'slope': False,
            'aspect': False,
            'landform': False,
            'dist_roads': False,
            'dist_coast': False,
            'dist_settlement': False,
            's1_bands': [],
            'palsar_bands': [],
            'spectral_indices': [],
            'index_coefficients': None,
            'distance_options': {},
            'sentinel1_options': {},
            'palsar_options': {},
        }

    ind = (predictor_config.get('individual_predictors') or {}) if 'individual_predictors' in predictor_config else {}
    terrain_all = bool(predictor_config.get('terrain', False))
    dist_all = bool(predictor_config.get('distance', False) or ind.get('distance', False))

    raw_indices = list(
        (ind.get('spectral_indices') if 'individual_predictors' in predictor_config else predictor_config.get('spectral_indices'))
        or []
    )

    all_s1 = ['S1_VV', 'S1_VH', 'S1_VV_VH']
    all_palsar = ['PALSAR_HH', 'PALSAR_HV', 'PALSAR_HH_HV']
    s1_bands: List[str] = []
    palsar_bands: List[str] = []

    def _collect_sar(val, target, defaults):
        if val is True:
            for b in defaults:
                if b not in target:
                    target.append(b)
        elif isinstance(val, (list, tuple, set)):
            for v in val:
                vu = str(v).upper()
                if vu in defaults and vu not in target:
                    target.append(vu)

    _collect_sar(predictor_config.get('sentinel1'), s1_bands, all_s1)
    _collect_sar(ind.get('sentinel1'), s1_bands, all_s1)
    _collect_sar(predictor_config.get('palsar'), palsar_bands, all_palsar)
    _collect_sar(ind.get('palsar'), palsar_bands, all_palsar)

    spectral_indices: List[str] = []
    for item in raw_indices:
        iu = str(item).upper()
        if iu in all_s1:
            if iu not in s1_bands:
                s1_bands.append(iu)
        elif iu in all_palsar:
            if iu not in palsar_bands:
                palsar_bands.append(iu)
        else:
            spectral_indices.append(str(item))

    return {
        'elevation': bool(ind.get('elevation', terrain_all)),
        'slope': bool(ind.get('slope', terrain_all)),
        'aspect': bool(ind.get('aspect', terrain_all)),
        'landform': bool(ind.get('landform', predictor_config.get('landform', False))),
        'dist_roads': bool(ind.get('dist_roads', dist_all)),
        'dist_coast': bool(ind.get('dist_coast', dist_all)),
        'dist_settlement': bool(ind.get('dist_settlement', dist_all)),
        's1_bands': s1_bands,
        'palsar_bands': palsar_bands,
        'spectral_indices': spectral_indices,
        'index_coefficients': ind.get('index_coefficients', predictor_config.get('index_coefficients')),
        'distance_options': predictor_config.get('distance_options') or {},
        'sentinel1_options': predictor_config.get('sentinel1_options') or {},
        'palsar_options': predictor_config.get('palsar_options') or {},
    }


def compute_multitemporal_predictors(
    composites_by_period: Dict[str, Any],
    aoi,
    predictor_config: Dict[str, Any],
    collections_by_period: Optional[Dict[str, Any]] = None,
    common_bands: Optional[Sequence[str]] = None,
    histogram_matched: bool = False,
    mode: str = 'model_transfer',
    dem_source: str = 'NASADEM',
    period_order: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """
    Compute predictor stacks across **all** mapping periods for both Batch and
    Model Transfer workflows.

    Key guarantees:
    1. Predictors are always computed for every period (never skipped for target
       periods in Model Transfer mode).
    2. Static terrain layers (`elevation`, `slope`, `aspect`) are computed once
       for ``aoi`` and shared across all periods.
    3. When ``histogram_matched=True`` (or ``collections_by_period is None``),
       spectral indices are computed directly from each period's composite
       ``ee.Image`` so reference and target feature spaces match identically.
    4. In ``'model_transfer'`` mode (or when ``common_bands`` is supplied),
       optical bands and spectral indices are aligned to ``common_bands`` so
       ``inputProperties`` match 100% across periods.
    """
    ensure_ee_initialized()

    # Unwrap output of final_Image.get_multitemporal_composites() if passed directly
    if isinstance(composites_by_period, dict) and isinstance(
        composites_by_period.get('composites'), dict
    ):
        if period_order is None:
            period_order = composites_by_period.get('period_order')
        if common_bands is None:
            common_bands = composites_by_period.get('common_bands')
        if not histogram_matched:
            histogram_matched = bool(composites_by_period.get('histogram_matched', False))
        composites = composites_by_period['composites']
    else:
        composites = composites_by_period

    if period_order is None:
        period_order = list(composites.keys())
    else:
        period_order = list(period_order)

    if len(period_order) < 2:
        raise ValueError(
            f"compute_multitemporal_predictors requires at least 2 periods, got {len(period_order)}."
        )

    # Unwrap output of Reflectance_Data.get_multitemporal_optical_data() if provided
    collections: Dict[str, Optional[ee.ImageCollection]] = {}
    if collections_by_period is not None:
        col_entries = (
            collections_by_period['periods']
            if isinstance(collections_by_period.get('periods'), dict)
            else collections_by_period
        )
        for label in period_order:
            entry = col_entries.get(label)
            collections[label] = entry['collection'] if isinstance(entry, dict) and 'collection' in entry else entry
    else:
        for label in period_order:
            collections[label] = None

    selections = _extract_predictor_selections(predictor_config)
    spectral_calc = SpectralCalculator()

    # Compute static terrain + landform + distance layers once for AOI
    terrain_layers: List[ee.Image] = []
    terrain_bands: List[str] = []
    if any([selections['elevation'], selections['slope'], selections['aspect'], selections['landform']]):
        tc = terrain_calculator()
        if selections['elevation']:
            elev = tc.calculate_elevation(aoi, dem_source=dem_source)
            if elev is not None:
                terrain_layers.append(elev)
                terrain_bands.append('elevation')
        if selections['slope']:
            slp = tc.calculate_slope(aoi, dem_source=dem_source)
            if slp is not None:
                terrain_layers.append(slp)
                terrain_bands.append('slope')
        if selections['aspect']:
            asp = tc.calculate_aspect(aoi, dem_source=dem_source)
            if asp is not None:
                terrain_layers.append(asp)
                terrain_bands.append('aspect')
        if selections['landform']:
            lf = tc.get_landform(aoi)
            if lf is not None:
                terrain_layers.append(lf)
                terrain_bands.append('landform')

    distance_bands: List[str] = []
    if any([selections['dist_roads'], selections['dist_coast'], selections['dist_settlement']]):
        dc = distance_calculator()
        d_opts = selections['distance_options']
        dist_img = dc.calculate_distance_metrics(
            aoi=aoi,
            max_dist=d_opts.get('max_dist', 50000),
            in_meters=d_opts.get('in_meters', True),
            include_roads=selections['dist_roads'],
            include_coast=selections['dist_coast'],
            include_settlement=selections['dist_settlement'],
            road_source=d_opts.get('road_source', 'grip4'),
            main_roads_only=d_opts.get('main_roads_only', True),
            precomputed_road_asset=d_opts.get('precomputed_road_asset'),
        )
        if dist_img is not None:
            terrain_layers.append(dist_img)
            if selections['dist_roads']:
                distance_bands.append('dist_roads')
            if selections['dist_coast']:
                distance_bands.append('dist_coast')
            if selections['dist_settlement']:
                distance_bands.append('dist_settlement')

    terrain_stack = None
    if terrain_layers:
        terrain_stack = terrain_layers[0]
        for layer in terrain_layers[1:]:
            terrain_stack = terrain_stack.addBands(layer)

    requested_indices = selections['spectral_indices']
    index_coefficients = selections['index_coefficients']

    # Always determine common optical bands across all periods so both Batch and
    # Model Transfer modes share the exact same predictor set
    if common_bands is None:
        band_sets = []
        for label in period_order:
            band_sets.append(list(composites[label].bandNames().getInfo()))
        intersect = set(band_sets[0])
        for bset in band_sets[1:]:
            intersect &= set(bset)
        common_bands = [b for b in band_sets[0] if b in intersect]

    compatible_indices: List[str] = []
    skipped_indices: List[str] = []
    if requested_indices:
        if common_bands is not None:
            compatible_indices, skipped_indices = filter_compatible_indices(
                requested_indices, common_bands, spectral_calc=spectral_calc
            )
        else:
            compatible_indices = list(requested_indices)

    predictor_stacks: Dict[str, ee.Image] = {}
    feature_bands: Dict[str, List[str]] = {}

    full_collection_required_bands = {'BLUE', 'GREEN', 'RED', 'NIR', 'SWIR1', 'SWIR2'}

    for label in period_order:
        comp_img = composites[label]
        if common_bands is not None:
            base_img = comp_img.select(list(common_bands))
            period_optical_bands = list(common_bands)
        else:
            base_img = comp_img
            period_optical_bands = (
                list(common_bands)
                if common_bands is not None
                else list(comp_img.bandNames().getInfo())
            )

        stacked = base_img
        period_bands = list(period_optical_bands)

        if terrain_stack is not None:
            stacked = stacked.addBands(terrain_stack)
            period_bands.extend(terrain_bands)
            period_bands.extend(distance_bands)

        if requested_indices:
            if common_bands is not None:
                period_indices = compatible_indices
            else:
                period_indices, period_skipped = filter_compatible_indices(
                    requested_indices, period_optical_bands, spectral_calc=spectral_calc
                )
                for s in period_skipped:
                    if s not in skipped_indices:
                        skipped_indices.append(s)

            if period_indices:
                col = collections.get(label)
                can_use_collection = (
                    not histogram_matched
                    and col is not None
                    and full_collection_required_bands.issubset(set(period_optical_bands))
                )
                if can_use_collection:
                    spectral_img = spectral_calc.calculate_indices_with_collection(
                        collection=col,
                        aoi=aoi,
                        index_list=period_indices,
                        coefficients=index_coefficients,
                        reducer_method='mean',
                    )
                else:
                    spectral_img = compute_single_image_indices(
                        image=base_img,
                        index_list=period_indices,
                        available_bands=period_optical_bands,
                        coefficients=index_coefficients,
                        spectral_calc=spectral_calc,
                    )
                if spectral_img is not None:
                    stacked = stacked.addBands(spectral_img)
                    period_bands.extend(period_indices)

        predictor_stacks[label] = stacked
        feature_bands[label] = period_bands

    # Period-specific SAR predictors (Sentinel-1 & ALOS PALSAR): only stacked when available across ALL periods
    sar_bands_added: List[str] = []
    if selections['s1_bands'] or selections['palsar_bands']:
        sar_calc = get_sar_data()
        period_dates: Dict[str, Tuple[Optional[str], Optional[str]]] = {}
        col_entries = (
            collections_by_period['periods']
            if isinstance(collections_by_period, dict) and isinstance(collections_by_period.get('periods'), dict)
            else (collections_by_period or {})
        )
        for label in period_order:
            entry = col_entries.get(label) if isinstance(col_entries, dict) else None
            if isinstance(entry, dict) and entry.get('start_date') and entry.get('end_date'):
                period_dates[label] = (str(entry['start_date']), str(entry['end_date']))
            else:
                period_dates[label] = (None, None)

        if selections['s1_bands']:
            s1_opts = selections['sentinel1_options']
            s1_by_period: Dict[str, ee.Image] = {}
            for label in period_order:
                sd, ed = period_dates[label]
                if sd and ed:
                    s1_img = sar_calc.get_sentinel1(
                        aoi=aoi,
                        start_date=sd,
                        end_date=ed,
                        orbit_pass=s1_opts.get('orbit_pass', 'BOTH'),
                        reducer=s1_opts.get('reducer', 'median'),
                        speckle_filter=s1_opts.get('speckle_filter', 'focal_median'),
                        filter_radius_m=s1_opts.get('filter_radius_m', 30.0),
                        include_ratio=True,
                    )
                    if s1_img is not None:
                        s1_by_period[label] = s1_img.select(selections['s1_bands'])
            if len(s1_by_period) == len(period_order):
                for label in period_order:
                    predictor_stacks[label] = predictor_stacks[label].addBands(s1_by_period[label])
                    feature_bands[label].extend(selections['s1_bands'])
                sar_bands_added.extend(selections['s1_bands'])
            else:
                logger.warning(
                    "Skipping Sentinel-1 bands %s: not available across all periods %s.",
                    selections['s1_bands'],
                    period_order,
                )
                skipped_indices.extend(b for b in selections['s1_bands'] if b not in skipped_indices)

        if selections['palsar_bands']:
            p_opts = selections['palsar_options']
            palsar_by_period: Dict[str, ee.Image] = {}
            for label in period_order:
                sd, ed = period_dates[label]
                if sd and ed:
                    p_img = sar_calc.get_palsar(
                        aoi=aoi,
                        start_date=sd,
                        end_date=ed,
                        reducer=p_opts.get('reducer', 'median'),
                        speckle_filter=p_opts.get('speckle_filter', 'focal_median'),
                        filter_radius_m=p_opts.get('filter_radius_m', 30.0),
                        include_ratio=True,
                        mask_layover_shadow=p_opts.get('mask_layover_shadow', True),
                        allow_nearest_year_fallback=p_opts.get('allow_nearest_year_fallback', True),
                    )
                    if p_img is not None:
                        palsar_by_period[label] = p_img.select(selections['palsar_bands'])
            if len(palsar_by_period) == len(period_order):
                for label in period_order:
                    predictor_stacks[label] = predictor_stacks[label].addBands(palsar_by_period[label])
                    feature_bands[label].extend(selections['palsar_bands'])
                sar_bands_added.extend(selections['palsar_bands'])
            else:
                logger.warning(
                    "Skipping PALSAR bands %s: not available across all periods %s.",
                    selections['palsar_bands'],
                    period_order,
                )
                skipped_indices.extend(b for b in selections['palsar_bands'] if b not in skipped_indices)

    # Compute common feature intersection across all periods
    common_feature_set = set(feature_bands[period_order[0]])
    for label in period_order[1:]:
        common_feature_set &= set(feature_bands[label])
    common_feature_bands = [b for b in feature_bands[period_order[0]] if b in common_feature_set]

    return {
        'predictor_stacks': predictor_stacks,
        'feature_bands': feature_bands,
        'common_feature_bands': common_feature_bands,
        'compatible_indices': compatible_indices,
        'skipped_indices': skipped_indices,
        'terrain_bands': terrain_bands,
        'distance_bands': distance_bands,
        'sar_bands': sar_bands_added,
        'period_order': period_order,
    }


@dataclass
class PeriodConfig:
    """Configuration for a single mapping period in multitemporal analysis."""

    label: str
    start_date: Union[int, str]
    end_date: Union[int, str]
    sensor: str
    cloud_cover: int = 30
    order_index: Optional[int] = None
    scale: Optional[int] = None
    cs_threshold: float = 0.7

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to a serializable dictionary."""
        return {
            'label': self.label,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'sensor': self.sensor,
            'cloud_cover': self.cloud_cover,
            'order_index': self.order_index,
            'scale': self.scale,
            'cs_threshold': self.cs_threshold,
        }


def normalize_period_configs(
    periods: Union[Dict[str, Union[PeriodConfig, Dict[str, Any]]], Sequence[Union[PeriodConfig, Dict[str, Any]]]]
) -> Tuple[List[PeriodConfig], List[str]]:
    """
    Validate and chronologically order ``N >= 2`` period definitions.

    Returns ``(ordered_configs, period_order)``.
    """
    from .data_acquisition import parse_date_input

    configs: List[PeriodConfig] = []
    if isinstance(periods, dict):
        for idx, (label, cfg) in enumerate(periods.items()):
            raw = cfg.to_dict() if isinstance(cfg, PeriodConfig) else dict(cfg)
            raw['label'] = label
            if raw.get('order_index') is None:
                raw['order_index'] = idx
            configs.append(PeriodConfig(**{k: v for k, v in raw.items() if k in PeriodConfig.__dataclass_fields__}))
    elif isinstance(periods, (list, tuple)):
        for idx, cfg in enumerate(periods):
            if isinstance(cfg, PeriodConfig):
                c_dict = cfg.to_dict()
            elif isinstance(cfg, dict):
                c_dict = dict(cfg)
            else:
                c_dict = vars(cfg).copy()
            if not c_dict.get('label'):
                raise ValueError(f"Period at index {idx} is missing a 'label'.")
            if c_dict.get('order_index') is None:
                c_dict['order_index'] = idx
            configs.append(
                PeriodConfig(**{k: v for k, v in c_dict.items() if k in PeriodConfig.__dataclass_fields__})
            )
    else:
        raise ValueError("periods must be a dict or a list/tuple of PeriodConfig/dict items.")

    if len(configs) < 2:
        raise ValueError(f"Multitemporal analysis requires at least 2 periods, got {len(configs)}.")

    labels = [c.label for c in configs]
    if len(set(labels)) != len(labels):
        raise ValueError(f"Period labels must be unique, got {labels}.")

    ordered = sorted(
        configs,
        key=lambda c: (parse_date_input(c.start_date, is_start=True), c.order_index or 0),
    )
    return ordered, [c.label for c in ordered]


class MultitemporalAnalysis:
    """
    Lightweight orchestrator for N >= 2 period LULC mapping across Modules 1 and 3-7.

    Supports both **Batch Mode** (independent training data and models per period)
    and **Model Transfer Mode** (single training set on ``reference_period`` transferred
    across harmonized historical/target periods).
    """

    def __init__(
        self,
        aoi,
        periods: Union[Dict[str, Union[PeriodConfig, Dict[str, Any]]], Sequence[Union[PeriodConfig, Dict[str, Any]]]],
        reference_period: str = 'baseline',
        harmonize_landsat: bool = True,
        apply_brdf: bool = False,
        apply_topo_correction: bool = False,
        dem: Optional[ee.Image] = None,
        drop_aerosol: bool = True,
        harmonize_sensors: bool = True,
        histogram_match: bool = True,
        radiometric_matching: Optional[str] = None,
        pif_metric: str = 'sid',
        pif_percentile: float = 10.0,
        reducer: str = 'median',
        scale: int = 30,
        max_buckets: int = 256,
        temporal_segmentation: bool = False,
        segmentation_method: str = 'landtrendr',
        segmentation_driver_index: str = 'NBR',
        segmentation_params: Optional[Dict[str, Any]] = None,
        min_segmentation_years: int = 6,
    ):
        if aoi is None:
            raise ValueError("aoi must not be None.")

        self.aoi = aoi
        self.period_configs, self.period_order = normalize_period_configs(periods)
        self.periods_by_label: Dict[str, PeriodConfig] = {c.label: c for c in self.period_configs}

        if reference_period not in self.periods_by_label:
            raise ValueError(
                f"reference_period '{reference_period}' not found in periods {self.period_order}."
            )
        self.reference_period = reference_period

        self.harmonize_landsat = harmonize_landsat
        self.apply_brdf = apply_brdf
        self.apply_topo_correction = apply_topo_correction
        self.dem = dem
        self.drop_aerosol = drop_aerosol
        self.harmonize_sensors = harmonize_sensors
        self.histogram_match = histogram_match
        self.radiometric_matching = radiometric_matching
        self.pif_metric = pif_metric
        self.pif_percentile = pif_percentile
        self.reducer = reducer
        self.scale = scale
        self.max_buckets = max_buckets
        self.temporal_segmentation = bool(temporal_segmentation)
        self.segmentation_method = str(segmentation_method).strip().lower()
        self.segmentation_driver_index = str(segmentation_driver_index).strip().upper()
        self.segmentation_params = dict(segmentation_params) if segmentation_params else None
        self.min_segmentation_years = int(min_segmentation_years)

        # Pipeline stage state
        self.optical_data: Optional[Dict[str, Any]] = None
        self.composites_result: Optional[Dict[str, Any]] = None
        self.training_result: Optional[Dict[str, Any]] = None
        self.landcover_df: Optional[Any] = None
        self.separability_result: Optional[Dict[str, Any]] = None
        self.predictors_result: Optional[Dict[str, Any]] = None
        self.feature_extraction_result: Optional[Dict[str, Any]] = None
        self.rfe_result: Optional[Dict[str, Any]] = None
        self.classification_result: Optional[Dict[str, Any]] = None
        self.accuracy_result: Optional[Dict[str, Any]] = None
        self.mode: Optional[str] = None

    def acquire_imagery(
        self,
        include_thermal: bool = False,
        verbose: bool = True,
        compute_detailed_stats: bool = True,
        calculate_coverage: bool = False,
        sharpen_s2: bool = False,
        ccdc_images: Optional[Dict[str, ee.Image]] = None,
        lt_image: Optional[ee.Image] = None,
    ) -> Dict[str, Any]:
        """
        Acquire optical collections and build harmonized composites for all periods
        (`Module 1` + `imagery_correction.py` + optional `temporal_segmentation.py`).

        - When ``temporal_segmentation=False`` (default): builds harmonized median
          composites with optional CDF ``histogram_match``.
        - When ``temporal_segmentation=True``: builds despiked, gap-filled composites
          via native GEE ``LandTrendr`` FTV (or pre-exported ``lt_image`` asset) and
          skips post-FTV CDF histogram matching. Unsupported sensors (Sentinel-2 10m,
          MSS) fall back to standard median compositing with a warning.
        """
        ensure_ee_initialized()
        from .data_acquisition import Reflectance_Data, final_Image, parse_date_input

        period_dicts = [cfg.to_dict() for cfg in self.period_configs]
        use_lt = False
        if self.temporal_segmentation:
            if self.segmentation_method != 'landtrendr':
                raise ValueError(
                    f"Unsupported segmentation_method '{self.segmentation_method}'. Supported: ['landtrendr']"
                )
            from .temporal_segmentation import LandTrendr_fit
            lt_comp = LandTrendr_fit()
            win_check = lt_comp.validate_and_resolve_window(
                periods=period_dicts,
                reference_period=self.reference_period,
                min_years=self.min_segmentation_years,
            )
            if win_check['compatible']:
                use_lt = True
            else:
                logger.warning(
                    "Falling back to standard median compositing: %s", win_check['reason']
                )

        fi = final_Image()
        if use_lt:
            max_cc = max(float(p.get('cloud_cover', 30)) for p in period_dicts)
            self.composites_result = fi.get_landtrendr_composites(
                aoi=self.aoi,
                periods=period_dicts,
                reference_period=self.reference_period,
                driver_index=self.segmentation_driver_index,
                min_years=self.min_segmentation_years,
                cloud_cover=max_cc,
                reducer=self.reducer,
                apply_brdf=self.apply_brdf,
                apply_topo_correction=self.apply_topo_correction,
                dem=self.dem,
                lt_params=self.segmentation_params,
                lt_image=lt_image,
                verbose=verbose,
            )
            # Populate period metadata so downstream SAR predictors inherit dates cleanly
            self.optical_data = {
                'periods': {
                    cfg.label: {
                        'collection': None,
                        'thermal_collection': None,
                        'stats': self.composites_result['metadata'].get(cfg.label, {}),
                        'sensor': cfg.sensor,
                        'start_date': parse_date_input(cfg.start_date, True),
                        'end_date': parse_date_input(cfg.end_date, False),
                    }
                    for cfg in self.period_configs
                },
                'period_order': list(self.period_order),
            }
        else:
            rd = Reflectance_Data()
            self.optical_data = rd.get_multitemporal_optical_data(
                aoi=self.aoi,
                periods=period_dicts,
                harmonize_landsat=self.harmonize_landsat,
                apply_brdf=self.apply_brdf,
                apply_topo_correction=self.apply_topo_correction,
                dem=self.dem,
                include_thermal=include_thermal,
                verbose=verbose,
                compute_detailed_stats=compute_detailed_stats,
            )

            self.composites_result = fi.get_multitemporal_composites(
                collections_by_period=self.optical_data,
                aoi=self.aoi,
                reference_period=self.reference_period,
                reducer=self.reducer,
                drop_aerosol=self.drop_aerosol,
                harmonize_sensors=self.harmonize_sensors,
                histogram_match=self.histogram_match,
                radiometric_matching=self.radiometric_matching,
                pif_metric=self.pif_metric,
                pif_percentile=self.pif_percentile,
                scale=self.scale,
                max_buckets=self.max_buckets,
                sharpen_s2=sharpen_s2,
                calculate_coverage=calculate_coverage,
                verbose=verbose,
            )

        if ccdc_images:
            # Future hook for synthetic CCDC composite substitution per period
            for label, syn_img in ccdc_images.items():
                if label in self.composites_result.get('composites', {}) and syn_img is not None:
                    self.composites_result['composites'][label] = syn_img

        return {
            'optical_data': self.optical_data,
            'composites': self.composites_result,
            'period_order': self.period_order,
            'common_bands': self.composites_result.get('common_bands'),
        }

    def load_training_data(
        self,
        training_sources: Dict[str, Any],
        landcover_df: Optional[Any] = None,
        class_field: Optional[str] = None,
        use_class_ids: bool = True,
        min_samples: int = 20,
    ) -> Dict[str, Any]:
        """
        Load and validate 1-or-N training datasets (`Module 3`), automatically
        detecting ``'model_transfer'`` vs ``'batch'`` mode.

        Train/test stratified splitting is performed downstream inside
        ``extract_features()`` via ``FeatureExtraction.stratified_split()``.
        """
        from .sample_data import SyncTrainData

        self.landcover_df = landcover_df
        self.training_result = SyncTrainData.LoadMultitemporalTrainData(
            training_sources=training_sources,
            landcover_df=landcover_df,
            aoi_geometry=self.aoi,
            period_order=self.period_order,
            reference_period=self.reference_period,
            class_field=class_field,
            use_class_ids=use_class_ids,
            min_samples=min_samples,
        )
        self.mode = self.training_result['mode']
        self.feature_extraction_result = None
        return self.training_result

    def check_predictor_availability(self) -> Dict[str, Dict[str, Any]]:
        """Check which predictors are available across all configured periods."""
        return predictor_catalog.check_multitemporal_availability(self.period_configs)

    def check_separability(
        self,
        class_property: str = 'class_id',
        class_name_property: Optional[str] = None,
        region: Optional[Any] = None,
        scale: Optional[int] = None,
        max_pixels_per_class: int = 5000,
        method: str = 'TD',
        bands: Optional[List[str]] = None,
        include_silhouette: bool = False,
        silhouette_metric: str = 'euclidean',
        use_predictors: bool = False,
    ) -> Dict[str, Any]:
        """Run spectral/predictor separability (`Module 4`) on periods that have training data."""
        if self.composites_result is None and self.predictors_result is None:
            raise RuntimeError("Call acquire_imagery() before check_separability().")
        if self.training_result is None:
            raise RuntimeError("Call load_training_data() before check_separability().")

        from .sample_data_quality import sample_quality

        geom = region
        if geom is None:
            geom = self.aoi.geometry() if isinstance(self.aoi, ee.FeatureCollection) else self.aoi

        img_source = (
            self.predictors_result
            if (use_predictors and self.predictors_result is not None)
            else self.composites_result
        )
        self.separability_result = sample_quality.run_multitemporal_separability(
            composites_by_period=img_source,
            training_by_period=self.training_result['training_by_period'],
            class_property=class_property,
            region=geom,
            class_name_property=class_name_property,
            scale=scale or self.scale,
            max_pixels_per_class=max_pixels_per_class,
            method=method,
            period_order=self.period_order,
            bands=bands,
            include_silhouette=include_silhouette,
            silhouette_metric=silhouette_metric,
        )
        return self.separability_result

    def compute_predictors(
        self,
        predictor_config: Dict[str, Any],
        dem_source: str = 'NASADEM',
        mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Compute predictor stacks (`Module 5`) across all mapping periods."""
        if self.composites_result is None:
            raise RuntimeError("Call acquire_imagery() before compute_predictors().")

        effective_mode = mode or self.mode or 'model_transfer'
        self.predictors_result = compute_multitemporal_predictors(
            composites_by_period=self.composites_result,
            aoi=self.aoi,
            predictor_config=predictor_config,
            collections_by_period=self.optical_data,
            common_bands=self.composites_result.get('common_bands'),
            histogram_matched=bool(self.composites_result.get('histogram_matched', self.histogram_match)),
            mode=effective_mode,
            dem_source=dem_source,
            period_order=self.period_order,
        )
        self.feature_extraction_result = None
        return self.predictors_result

    def extract_features(
        self,
        class_property: str = 'class_id',
        mode: str = 'auto',
        scale: Optional[int] = None,
        tile_scale: int = 4,
        split_method: str = 'stratified',
        train_ratio: float = 0.7,
        seed: int = 0,
    ) -> Dict[str, Any]:
        """
        Execute single feature extraction (1 training set on ``reference_period``)
        or batch feature extraction (N training sets across all periods) via
        ``FeatureExtraction.extract_multitemporal_samples()``.
        """
        if self.predictors_result is None:
            raise RuntimeError("Call compute_predictors() before extract_features().")
        if self.training_result is None:
            raise RuntimeError("Call load_training_data() before extract_features().")

        from .classification import FeatureExtraction

        effective_mode = mode if mode != 'auto' else (self.mode or 'auto')
        common_bands = (
            self.predictors_result.get('common_feature_bands')
            if effective_mode in ('model_transfer', 'auto')
            else None
        )

        fe = FeatureExtraction()
        self.feature_extraction_result = fe.extract_multitemporal_samples(
            predictor_stacks=self.predictors_result['predictor_stacks'],
            training_splits=self.training_result['training_by_period'],
            class_property=class_property,
            reference_period=self.reference_period,
            common_bands=common_bands,
            mode=effective_mode,
            scale=scale or self.scale,
            tile_scale=tile_scale,
            split_method=split_method,
            train_ratio=train_ratio,
            seed=seed,
            period_order=self.period_order,
        )
        return self.feature_extraction_result

    def run_feature_selection(
        self,
        class_property: str = 'class_id',
        n_features_to_select: Optional[int] = None,
        step: Union[int, float] = 1,
        cv: int = 3,
        n_trees: int = 50,
        max_depth: Optional[int] = 12,
        min_samples_leaf: int = 2,
        seed: int = 42,
        max_samples: Optional[int] = 5000,
        per_period_rfe: bool = False,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Run optional ``RFE_CV`` recursive feature elimination.

        By default (``per_period_rfe=False``), selects a single unified predictor
        subset on ``reference_period`` and applies that band subset across **all**
        periods so Model Transfer and Batch workflows share the exact same features.
        """
        if self.predictors_result is None:
            raise RuntimeError("Call compute_predictors() before run_feature_selection().")
        if self.feature_extraction_result is None:
            self.extract_features(class_property=class_property, seed=seed)

        common_features = list(self.predictors_result['common_feature_bands'])
        sampled_splits = self.feature_extraction_result['sampled_splits']

        if not per_period_rfe or self.feature_extraction_result['mode_used'] == 'model_transfer':
            ref_train_fc, _ = sampled_splits[self.reference_period]
            rfe = RFE_CV(
                training_data=ref_train_fc,
                class_property=class_property,
                features=common_features,
                n_features_to_select=n_features_to_select,
                step=step,
                cv=cv,
                n_trees=n_trees,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                seed=seed,
                max_samples=max_samples,
                verbose=verbose,
            ).fit()
            selected = rfe.get_selected_features()
            for lbl in self.period_order:
                self.predictors_result['predictor_stacks'][lbl] = (
                    self.predictors_result['predictor_stacks'][lbl].select(selected)
                )
                self.predictors_result['feature_bands'][lbl] = list(selected)
            self.predictors_result['common_feature_bands'] = list(selected)
            self.rfe_result = {
                'mode': 'unified',
                'selected_features': selected,
                'ranking': rfe.get_ranking(),
                'rfe_objects': {self.reference_period: rfe},
            }
            return self.rfe_result

        # Per-period RFE in Batch mode
        selected_by_period: Dict[str, List[str]] = {}
        rankings_by_period: Dict[str, Dict[str, int]] = {}
        rfe_objs: Dict[str, RFE_CV] = {}
        for lbl in self.period_order:
            train_fc, _ = sampled_splits[lbl]
            rfe = RFE_CV(
                training_data=train_fc,
                class_property=class_property,
                features=list(self.predictors_result['feature_bands'][lbl]),
                n_features_to_select=n_features_to_select,
                step=step,
                cv=cv,
                n_trees=n_trees,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                seed=seed,
                max_samples=max_samples,
                verbose=verbose,
            ).fit()
            sel = rfe.get_selected_features()
            self.predictors_result['predictor_stacks'][lbl] = (
                self.predictors_result['predictor_stacks'][lbl].select(sel)
            )
            self.predictors_result['feature_bands'][lbl] = list(sel)
            selected_by_period[lbl] = sel
            rankings_by_period[lbl] = rfe.get_ranking()
            rfe_objs[lbl] = rfe

        self.rfe_result = {
            'mode': 'per_period',
            'selected_features_by_period': selected_by_period,
            'ranking_by_period': rankings_by_period,
            'rfe_objects': rfe_objs,
        }
        return self.rfe_result

    def run_classification(
        self,
        class_property: str = 'class_id',
        mode: str = 'auto',
        classification_type: str = 'soft',
        soft_mode: str = 'multiprobability',
        ntrees: int = 100,
        v_split: Optional[int] = None,
        min_leaf: int = 1,
        seed: int = 0,
        probability_scale: int = 100,
        scale: Optional[int] = None,
        tile_scale: int = 4,
        train_ratio: float = 0.7,
        split_method: str = 'stratified',
        class_names: Optional[Any] = None,
        include_log_loss: bool = False,
        spatial_smoothing: bool = False,
        smoothing_radius: int = 1,
        smoothing_sigma: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Extract pixel features via ``FeatureExtraction`` (if not already cached) and
        run ``Generate_LULC.classify_multitemporal()`` (`Module 6`).
        """
        if self.predictors_result is None:
            raise RuntimeError("Call compute_predictors() before run_classification().")

        if self.feature_extraction_result is None:
            self.extract_features(
                class_property=class_property,
                mode=mode,
                scale=scale,
                tile_scale=tile_scale,
                split_method=split_method,
                train_ratio=train_ratio,
                seed=seed,
            )

        from .classification import Generate_LULC

        lulc = Generate_LULC()
        common_bands = (
            self.predictors_result.get('common_feature_bands')
            if self.feature_extraction_result['mode_used'] == 'model_transfer'
            else None
        )
        resolved_names = class_names if class_names is not None else self.landcover_df
        self.classification_result = lulc.classify_multitemporal(
            predictor_stacks=self.predictors_result['predictor_stacks'],
            training_splits=self.feature_extraction_result['sampled_splits'],
            class_property=class_property,
            mode=self.feature_extraction_result['mode_used'],
            reference_period=self.reference_period,
            classification_type=classification_type,
            soft_mode=soft_mode,
            ntrees=ntrees,
            v_split=v_split,
            min_leaf=min_leaf,
            seed=seed,
            probability_scale=probability_scale,
            common_bands=common_bands,
            period_order=self.period_order,
            class_names=resolved_names,
            include_log_loss=include_log_loss,
            spatial_smoothing=spatial_smoothing,
            smoothing_radius=smoothing_radius,
            smoothing_sigma=smoothing_sigma,
        )
        return self.classification_result

    def assess_accuracy(
        self,
        validation_by_period: Dict[str, Optional[ee.FeatureCollection]],
        class_property: str = 'class_id',
        scale: Optional[int] = None,
        confidence: float = 0.95,
        required_periods: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """Run multitemporal thematic accuracy assessment (`Module 7`)."""
        if self.classification_result is None:
            raise RuntimeError("Call run_classification() before assess_accuracy().")

        from .accuracy import thematic_accuracy

        req = [self.reference_period] if required_periods is None else list(required_periods)
        taa = thematic_accuracy()
        self.accuracy_result = taa.run_multitemporal_assessment(
            maps_by_period=self.classification_result['maps'],
            validation_by_period=validation_by_period,
            class_property=class_property,
            scale=scale or self.scale,
            confidence=confidence,
            required_periods=req,
            period_order=self.period_order,
        )
        return self.accuracy_result

    def run_pipeline(
        self,
        training_sources: Dict[str, Any],
        landcover_df: Any,
        predictor_config: Dict[str, Any],
        validation_by_period: Optional[Dict[str, Optional[ee.FeatureCollection]]] = None,
        class_field: str = 'class_id',
        class_name_property: Optional[str] = None,
        use_class_ids: bool = True,
        min_samples: int = 20,
        train_split_pct: float = 0.7,
        random_state: int = 42,
        run_separability: bool = True,
        classification_type: str = 'soft',
        soft_mode: str = 'multiprobability',
        ntrees: int = 100,
        v_split: Optional[int] = None,
        min_leaf: int = 1,
        seed: int = 0,
        probability_scale: int = 100,
        tile_scale: int = 4,
        required_periods: Optional[Sequence[str]] = None,
        include_log_loss: bool = False,
        include_silhouette: bool = False,
        rfe_config: Optional[Dict[str, Any]] = None,
        spatial_smoothing: bool = False,
        smoothing_radius: int = 1,
        smoothing_sigma: float = 1.0,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Execute the full multitemporal LULC mapping pipeline end-to-end."""
        self.acquire_imagery(verbose=verbose)
        self.load_training_data(
            training_sources=training_sources,
            landcover_df=landcover_df,
            class_field=class_field,
            use_class_ids=use_class_ids,
            min_samples=min_samples,
        )
        self.compute_predictors(predictor_config=predictor_config)
        self.extract_features(
            class_property=class_field,
            tile_scale=tile_scale,
            train_ratio=train_split_pct,
            seed=random_state if seed == 0 else seed,
        )
        if rfe_config:
            rfe_kwargs = dict(rfe_config)
            rfe_kwargs.setdefault('class_property', class_field)
            rfe_kwargs.setdefault('verbose', verbose)
            self.run_feature_selection(**rfe_kwargs)
        if run_separability:
            self.check_separability(
                class_property=class_field,
                class_name_property=class_name_property,
                include_silhouette=include_silhouette,
                use_predictors=include_silhouette,
            )
        self.run_classification(
            class_property=class_field,
            classification_type=classification_type,
            soft_mode=soft_mode,
            ntrees=ntrees,
            v_split=v_split,
            min_leaf=min_leaf,
            seed=seed,
            probability_scale=probability_scale,
            tile_scale=tile_scale,
            class_names=landcover_df,
            include_log_loss=include_log_loss,
            spatial_smoothing=spatial_smoothing,
            smoothing_radius=smoothing_radius,
            smoothing_sigma=smoothing_sigma,
        )
        if validation_by_period is not None:
            self.assess_accuracy(
                validation_by_period=validation_by_period,
                class_property=class_field,
                required_periods=required_periods,
            )

        return {
            'period_order': self.period_order,
            'reference_period': self.reference_period,
            'mode': self.mode,
            'optical_data': self.optical_data,
            'composites': self.composites_result,
            'training': self.training_result,
            'separability': self.separability_result,
            'predictors': self.predictors_result,
            'feature_extraction': self.feature_extraction_result,
            'rfe': self.rfe_result,
            'classification': self.classification_result,
            'accuracy': self.accuracy_result,
        }
