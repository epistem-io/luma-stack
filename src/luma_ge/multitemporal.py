"""
Multitemporal Analysis & Predictor Orchestration Module

Provides:
- Conflict-safe multitemporal predictor computation & cross-sensor index filtering
  (without modifying `src/luma_ge/predictor.py` while upstream predictor branches
  are pending merge).
- Lightweight orchestration (`PeriodConfig`, `MultitemporalAnalysis`) across N >= 2
  mapping periods for Batch and Model Transfer workflows.
"""

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple
import ee

from .ee_config import ensure_ee_initialized
from .predictor import SpectralCalculator, terrain_calculator

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

    # Delegate if SpectralCalculator already exposes calculate_indices()
    if hasattr(calc, 'calculate_indices'):
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
            'spectral_indices': [],
            'index_coefficients': None,
        }

    if 'individual_predictors' in predictor_config:
        ind = predictor_config['individual_predictors'] or {}
        return {
            'elevation': bool(ind.get('elevation', False)),
            'slope': bool(ind.get('slope', False)),
            'aspect': bool(ind.get('aspect', False)),
            'spectral_indices': list(ind.get('spectral_indices', []) or []),
            'index_coefficients': ind.get('index_coefficients', predictor_config.get('index_coefficients')),
        }

    terrain_all = bool(predictor_config.get('terrain', False))
    return {
        'elevation': terrain_all,
        'slope': terrain_all,
        'aspect': terrain_all,
        'spectral_indices': list(predictor_config.get('spectral_indices', []) or []),
        'index_coefficients': predictor_config.get('index_coefficients'),
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

    # Compute static terrain layers once for AOI
    terrain_layers: List[ee.Image] = []
    terrain_bands: List[str] = []
    if any([selections['elevation'], selections['slope'], selections['aspect']]):
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

    terrain_stack = None
    if terrain_layers:
        terrain_stack = terrain_layers[0]
        for layer in terrain_layers[1:]:
            terrain_stack = terrain_stack.addBands(layer)

    requested_indices = selections['spectral_indices']
    index_coefficients = selections['index_coefficients']

    # Determine common optical bands if model_transfer and not provided explicitly
    if mode == 'model_transfer' and common_bands is None:
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
        if common_bands is not None and mode == 'model_transfer':
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

        if requested_indices:
            if common_bands is not None and mode == 'model_transfer':
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
        'period_order': period_order,
    }
