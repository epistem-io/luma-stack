"""
Multitemporal Analysis & Predictor Orchestration Module

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
        reducer: str = 'median',
        scale: int = 30,
        max_buckets: int = 256,
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
        self.reducer = reducer
        self.scale = scale
        self.max_buckets = max_buckets

        # Pipeline stage state
        self.optical_data: Optional[Dict[str, Any]] = None
        self.composites_result: Optional[Dict[str, Any]] = None
        self.training_result: Optional[Dict[str, Any]] = None
        self.landcover_df: Optional[Any] = None
        self.separability_result: Optional[Dict[str, Any]] = None
        self.predictors_result: Optional[Dict[str, Any]] = None
        self.feature_extraction_result: Optional[Dict[str, Any]] = None
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
    ) -> Dict[str, Any]:
        """
        Acquire optical collections and build harmonized/histogram-matched composites
        for all periods (`Module 1` + `imagery_correction.py`).

        Future hook: if ``ccdc_images`` is supplied (e.g. `{period_label: ee.Image}`
        from a CCDC synthetic generator), matching periods may bypass raw mosaic creation.
        """
        ensure_ee_initialized()
        from .data_acquisition import Reflectance_Data, final_Image

        rd = Reflectance_Data()
        period_dicts = [cfg.to_dict() for cfg in self.period_configs]

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

        fi = final_Image()
        self.composites_result = fi.get_multitemporal_composites(
            collections_by_period=self.optical_data,
            aoi=self.aoi,
            reference_period=self.reference_period,
            reducer=self.reducer,
            drop_aerosol=self.drop_aerosol,
            harmonize_sensors=self.harmonize_sensors,
            histogram_match=self.histogram_match,
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

    def check_separability(
        self,
        class_property: str = 'class_id',
        class_name_property: Optional[str] = None,
        region: Optional[Any] = None,
        scale: Optional[int] = None,
        max_pixels_per_class: int = 5000,
        method: str = 'TD',
    ) -> Dict[str, Any]:
        """Run spectral separability (`Module 4`) on periods that have training data."""
        if self.composites_result is None:
            raise RuntimeError("Call acquire_imagery() before check_separability().")
        if self.training_result is None:
            raise RuntimeError("Call load_training_data() before check_separability().")

        from .sample_data_quality import sample_quality

        geom = region
        if geom is None:
            geom = self.aoi.geometry() if isinstance(self.aoi, ee.FeatureCollection) else self.aoi

        self.separability_result = sample_quality.run_multitemporal_separability(
            composites_by_period=self.composites_result,
            training_by_period=self.training_result['training_by_period'],
            class_property=class_property,
            region=geom,
            class_name_property=class_name_property,
            scale=scale or self.scale,
            max_pixels_per_class=max_pixels_per_class,
            method=method,
            period_order=self.period_order,
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
        if run_separability:
            self.check_separability(
                class_property=class_field,
                class_name_property=class_name_property,
            )
        self.compute_predictors(predictor_config=predictor_config)
        self.extract_features(
            class_property=class_field,
            tile_scale=tile_scale,
            train_ratio=train_split_pct,
            seed=random_state if seed == 0 else seed,
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
            'classification': self.classification_result,
            'accuracy': self.accuracy_result,
        }
