"""
Post-Classification Consistency Analysis Module (Standalone)

Provides transition matrix validation, Earth Engine cross-tabulation and spatial
inconsistency mapping, and multitemporal consistency correction via:
1. **Primary (HMM / Viterbi Constrained MAP)** over ``MULTIPROBABILITY`` soft stacks
   (both ``anchor_reference=True`` keeping the latest reference map fixed and
   ``anchor_reference=False`` for joint N-period Viterbi decoding, with optional
   3x3 spatial neighborhood smoothing).
2. **Secondary (Rule-Based Reclassification)** supporting explicit
   ``(from_id, to_id) -> target_id`` rules (or snapping to ``to_id``) as either
   manual overrides on top of HMM output or a standalone fallback for hard maps.
"""

from __future__ import annotations
import io
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import ee
import pandas as pd
from .classification import build_prob_band_names, extract_class_name_map
from .ee_config import ensure_ee_initialized

logger = logging.getLogger(__name__)

_FORBIDDEN_LOG_PENALTY = -1e6
_PROB_EPSILON = 1e-6


def _parse_int_id(val: Any) -> int:
    """Parse an integer class ID from int, float, or numeric string."""
    if isinstance(val, bool):
        raise ValueError(f"Invalid class ID (boolean): {val!r}")
    return int(float(str(val).strip()))


def _normalize_cell_weight(val: Any) -> float:
    """Normalize a transition matrix cell value into a probability/weight in [0.0, 1.0]."""
    if isinstance(val, bool):
        return 1.0 if val else 0.0
    if isinstance(val, str):
        low = val.strip().lower()
        if low in ('true', 'yes', 'y', 'allowed'):
            return 1.0
        if low in ('false', 'no', 'n', 'forbidden'):
            return 0.0
    w = float(val)
    if math.isnan(w) or w < 0.0 or w > 1.0:
        raise ValueError(f"Transition matrix weight must be in [0.0, 1.0], got {val!r}")
    return w


def _normalize_rules(
    rules: Optional[Dict[Any, int]]
) -> Dict[Tuple[int, int], int]:
    """Normalize rule keys like `(1, 2)` or `'1->2'` into `{(from_id, to_id): target_id}`."""
    if not rules:
        return {}
    normalized: Dict[Tuple[int, int], int] = {}
    for key, target in rules.items():
        if isinstance(key, (tuple, list)) and len(key) == 2:
            pair = (int(key[0]), int(key[1]))
        elif isinstance(key, str) and '->' in key:
            left, right = key.split('->', 1)
            pair = (_parse_int_id(left), _parse_int_id(right))
        elif isinstance(key, str) and ',' in key:
            left, right = key.split(',', 1)
            pair = (_parse_int_id(left), _parse_int_id(right))
        else:
            raise ValueError(
                f"Invalid transition rule key {key!r}; expected (from_id, to_id) tuple or 'from->to' string."
            )
        normalized[pair] = int(target)
    return normalized


class ConsistencyAnalyzer:
    """
    Multitemporal post-classification consistency validation, cross-tabulation,
    and constrained correction (HMM/Viterbi + rule-based reclassification).
    """

    def __init__(self, log_level: int = logging.INFO):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)

    # ------------------------------------------------------------------
    # 1. Transition Matrix Parsing & Validation
    # ------------------------------------------------------------------
    @staticmethod
    def _load_matrix_source(matrix: Any) -> Union[pd.DataFrame, Dict[int, Dict[int, float]]]:
        """Parse CSV path/bytes, DataFrame, or nested dict into a raw table/dict."""
        if isinstance(matrix, pd.DataFrame):
            df = matrix.copy()
        elif isinstance(matrix, bytes):
            df = pd.read_csv(io.BytesIO(matrix))
        elif isinstance(matrix, (str, Path)):
            p = Path(matrix)
            if p.exists():
                df = pd.read_csv(p)
            elif isinstance(matrix, str) and ('\n' in matrix or ',' in matrix):
                df = pd.read_csv(io.StringIO(matrix))
            else:
                raise FileNotFoundError(f"Transition matrix file not found: {matrix}")
        elif isinstance(matrix, dict):
            # Already validated payload or raw nested dict
            if 'weight_matrix' in matrix and isinstance(matrix['weight_matrix'], dict):
                return matrix['weight_matrix']
            out: Dict[int, Dict[int, float]] = {}
            for rk, row_val in matrix.items():
                r_id = _parse_int_id(rk)
                out[r_id] = {}
                if isinstance(row_val, dict):
                    for ck, cell in row_val.items():
                        out[r_id][_parse_int_id(ck)] = _normalize_cell_weight(cell)
            return out
        else:
            raise ValueError(
                f"matrix must be a DataFrame, CSV path/bytes/string, or nested dict; got {type(matrix).__name__}"
            )

        # Normalize DataFrame index/columns when loaded from CSV or raw DataFrame
        id_index_cols = [c for c in ('from_id', 'from', 'From', 'FROM', 'ID', 'id') if c in df.columns]
        if id_index_cols and not isinstance(df.index, pd.MultiIndex):
            df = df.set_index(id_index_cols[0])
        return df

    def validate_transition_matrix(
        self,
        matrix: Any,
        class_ids: Sequence[int],
    ) -> Dict[str, Any]:
        """
        Validate a transition consistency matrix (`from_id x to_id`) across ``class_ids``.

        Supports binary values (`0/1`, `bool`) and continuous weights (`0.0-1.0`).
        Values `> 0.0` mark allowed transitions; `0.0` (or `False`) marks impossible transitions.

        Returns
        -------
        dict
            ``{'valid', 'missing_pairs', 'class_ids', 'matrix_df', 'binary_matrix', 'weight_matrix'}``.
        """
        sorted_ids = sorted(int(c) for c in class_ids)
        if not sorted_ids:
            raise ValueError("class_ids must contain at least one integer class ID.")

        # Fast-path if already a validated result matching all class_ids
        if (
            isinstance(matrix, dict)
            and 'weight_matrix' in matrix
            and 'binary_matrix' in matrix
            and 'missing_pairs' in matrix
            and all(f in matrix['weight_matrix'] and all(t in matrix['weight_matrix'][f] for t in sorted_ids) for f in sorted_ids)
        ):
            return matrix

        parsed = self._load_matrix_source(matrix)
        lookup: Dict[int, Dict[int, Any]] = {}

        if isinstance(parsed, pd.DataFrame):
            # Map DataFrame row index labels -> int class IDs
            row_map: Dict[int, Any] = {}
            for r_lbl in parsed.index:
                try:
                    row_map[_parse_int_id(r_lbl)] = r_lbl
                except ValueError:
                    continue
            # Map DataFrame column labels -> int class IDs
            col_map: Dict[int, Any] = {}
            for c_lbl in parsed.columns:
                try:
                    col_map[_parse_int_id(c_lbl)] = c_lbl
                except ValueError:
                    continue

            for fid in sorted_ids:
                if fid not in row_map:
                    continue
                lookup[fid] = {}
                r_lbl = row_map[fid]
                for tid in sorted_ids:
                    if tid in col_map:
                        lookup[fid][tid] = parsed.loc[r_lbl, col_map[tid]]
        else:
            lookup = parsed

        missing_pairs: List[Tuple[int, int]] = []
        binary_matrix: Dict[int, Dict[int, bool]] = {fid: {} for fid in sorted_ids}
        weight_matrix: Dict[int, Dict[int, float]] = {fid: {} for fid in sorted_ids}
        df_rows: Dict[int, Dict[int, float]] = {fid: {} for fid in sorted_ids}

        for fid in sorted_ids:
            row_dict = lookup.get(fid, {})
            for tid in sorted_ids:
                if tid not in row_dict or pd.isna(row_dict[tid]):
                    missing_pairs.append((fid, tid))
                    df_rows[fid][tid] = float('nan')
                    continue
                w = _normalize_cell_weight(row_dict[tid])
                weight_matrix[fid][tid] = w
                binary_matrix[fid][tid] = w > 0.0
                df_rows[fid][tid] = w

        matrix_df = pd.DataFrame.from_dict(df_rows, orient='index').reindex(
            index=sorted_ids, columns=sorted_ids
        )
        return {
            'valid': len(missing_pairs) == 0,
            'missing_pairs': missing_pairs,
            'class_ids': sorted_ids,
            'matrix_df': matrix_df,
            'binary_matrix': binary_matrix,
            'weight_matrix': weight_matrix,
        }

    def _require_valid_matrix(
        self,
        consistency_matrix: Any,
        class_ids: Optional[Sequence[int]] = None,
    ) -> Dict[str, Any]:
        """Validate transition matrix and infer `class_ids` from it when omitted."""
        if class_ids is None:
            if isinstance(consistency_matrix, dict) and 'class_ids' in consistency_matrix:
                class_ids = consistency_matrix['class_ids']
            else:
                parsed = self._load_matrix_source(consistency_matrix)
                if isinstance(parsed, pd.DataFrame):
                    inferred = []
                    for lbl in parsed.index:
                        try:
                            inferred.append(_parse_int_id(lbl))
                        except ValueError:
                            pass
                    class_ids = sorted(set(inferred))
                else:
                    class_ids = sorted(int(k) for k in parsed.keys())

        validated = self.validate_transition_matrix(consistency_matrix, class_ids)
        if not validated['valid']:
            raise ValueError(
                f"Transition consistency matrix is missing pairs: {validated['missing_pairs'][:10]}"
            )
        return validated

    @staticmethod
    def _compute_multiplier(class_ids: Sequence[int]) -> int:
        max_id = max((int(c) for c in class_ids), default=0)
        digits = max(3, len(str(abs(max_id) + 1)))
        return int(10 ** digits)

    @staticmethod
    def _resolve_prob_bands(
        image: ee.Image,
        class_ids: Sequence[int],
        prob_band_map: Optional[Dict[int, str]] = None,
        class_names: Optional[Any] = None,
    ) -> Optional[List[str]]:
        """
        Resolve ordered probability band names ``[band(cid) for cid in class_ids]``
        on ``image``, supporting both scheme class names (``prob_<Class_Name>``) and
        default numeric names (``prob_<id>``). Returns ``None`` if bands are absent.
        """
        sorted_ids = [int(c) for c in class_ids]
        img_bands = list(image.bandNames().getInfo() or [])
        img_band_set = set(img_bands)

        # 1. Explicit prob_band_map for this period
        if prob_band_map and all(cid in prob_band_map for cid in sorted_ids):
            candidate = [prob_band_map[cid] for cid in sorted_ids]
            if all(b in img_band_set for b in candidate):
                return candidate

        # 2. Scheme-derived prob_<Sanitized_Class_Name>
        if class_names is not None:
            named_bands, _ = build_prob_band_names(sorted_ids, class_names=class_names)
            if all(b in img_band_set for b in named_bands):
                return named_bands

        # 3. Default numeric prob_<id>
        numeric_bands = [f"prob_{cid}" for cid in sorted_ids]
        if all(b in img_band_set for b in numeric_bands):
            return numeric_bands

        return None

    @staticmethod
    def _extract_aoi_geometry(aoi: Any) -> Optional[ee.Geometry]:
        if aoi is None:
            return None
        return aoi.geometry() if isinstance(aoi, ee.FeatureCollection) else aoi

    # ------------------------------------------------------------------
    # 2. Server-Side Cross-Tabulation & Spatial Inconsistency Mapping
    # ------------------------------------------------------------------
    def cross_tabulate(
        self,
        past_map: ee.Image,
        future_map: ee.Image,
        consistency_matrix: Any,
        aoi: Any,
        scale: int = 30,
        class_names: Optional[Any] = None,
        class_ids: Optional[Sequence[int]] = None,
    ) -> Dict[str, Any]:
        """
        Compute transition cross-tabulation between ``past_map`` and ``future_map``
        in Google Earth Engine and build spatial inconsistency masks.
        """
        ensure_ee_initialized()
        validated = self._require_valid_matrix(consistency_matrix, class_ids=class_ids)
        sorted_ids = validated['class_ids']
        binary_matrix = validated['binary_matrix']
        name_map = extract_class_name_map(class_names)
        multiplier = self._compute_multiplier(sorted_ids)

        past_cls = past_map.select('classification').toInt()
        future_cls = future_map.select('classification').toInt()
        code_img = past_cls.multiply(multiplier).add(future_cls).rename('transition_code')

        geom = self._extract_aoi_geometry(aoi)
        if geom is None:
            raise ValueError("aoi must not be None for cross_tabulate().")

        raw_hist_dict = code_img.reduceRegion(
            reducer=ee.Reducer.frequencyHistogram(),
            geometry=geom,
            scale=scale,
            maxPixels=1e13,
            tileScale=4,
        ).getInfo() or {}
        hist = raw_hist_dict.get('transition_code') or {}

        rows: List[Dict[str, Any]] = []
        pixel_area_m2 = float(scale) * float(scale)
        for raw_code, raw_count in hist.items():
            code_int = int(float(raw_code))
            cnt = int(round(float(raw_count)))
            if cnt <= 0:
                continue
            from_id = code_int // multiplier
            to_id = code_int % multiplier
            is_ok = bool(binary_matrix.get(from_id, {}).get(to_id, True))
            rows.append({
                'from_id': from_id,
                'to_id': to_id,
                'from_name': name_map.get(from_id, str(from_id)),
                'to_name': name_map.get(to_id, str(to_id)),
                'transition_code': code_int,
                'pixel_count': cnt,
                'area_m2': cnt * pixel_area_m2,
                'is_consistent': is_ok,
            })

        cols = [
            'from_id', 'to_id', 'from_name', 'to_name',
            'transition_code', 'pixel_count', 'area_m2', 'is_consistent',
        ]
        if rows:
            transitions_df = pd.DataFrame(rows).sort_values(
                'pixel_count', ascending=False
            ).reset_index(drop=True)
            inconsistent_df = transitions_df[~transitions_df['is_consistent']].reset_index(drop=True)
        else:
            transitions_df = pd.DataFrame(columns=cols)
            inconsistent_df = pd.DataFrame(columns=cols)

        forbidden_codes = [
            fid * multiplier + tid
            for fid in sorted_ids
            for tid in sorted_ids
            if not binary_matrix[fid][tid]
        ]
        if forbidden_codes:
            inconsistent_mask = (
                code_img
                .remap(forbidden_codes, [1] * len(forbidden_codes), 0)
                .selfMask()
                .rename('inconsistent_mask')
            )
        else:
            inconsistent_mask = ee.Image.constant(0).selfMask().rename('inconsistent_mask')

        inconsistent_code_image = (
            code_img.updateMask(inconsistent_mask).rename('inconsistent_code')
        )

        return {
            'transitions_df': transitions_df,
            'inconsistent_transitions': inconsistent_df,
            'inconsistent_mask': inconsistent_mask,
            'inconsistent_code_image': inconsistent_code_image,
            'multiplier': multiplier,
            'total_pixels': int(transitions_df['pixel_count'].sum()) if not transitions_df.empty else 0,
            'inconsistent_pixels': int(inconsistent_df['pixel_count'].sum()) if not inconsistent_df.empty else 0,
        }

    # ------------------------------------------------------------------
    # 3. HMM / Viterbi & Rule-Based Consistency Correction
    # ------------------------------------------------------------------
    @staticmethod
    def _emission_log_probs(
        soft_image: ee.Image,
        prob_bands: Sequence[str],
        probability_scale: float = 100.0,
        spatial_smoothing: bool = False,
        spatial_weight: float = 0.25,
        hard_band_name: str = 'classification',
        class_ids: Optional[Sequence[int]] = None,
    ) -> ee.Image:
        """Convert scaled probability bands `[0..probability_scale]` into log-likelihood bands."""
        scaled = (
            soft_image
            .select(list(prob_bands))
            .toFloat()
            .divide(float(probability_scale))
            .max(_PROB_EPSILON)
        )
        log_img = scaled.log()

        if spatial_smoothing and spatial_weight > 0.0 and class_ids is not None:
            cls_img = soft_image.select(hard_band_name).toInt()
            one_hot = ee.Image.cat([cls_img.eq(int(cid)).toFloat() for cid in class_ids])
            neigh_frac = one_hot.reduceNeighborhood(
                reducer=ee.Reducer.mean(),
                kernel=ee.Kernel.square(1),
            )
            log_img = log_img.add(neigh_frac.multiply(float(spatial_weight)))

        return log_img

    @staticmethod
    def _log_transition_matrix(
        weight_matrix: Dict[int, Dict[int, float]],
        sorted_ids: Sequence[int],
    ) -> List[List[float]]:
        """Build KxK log transition scores with heavy negative penalty on forbidden transitions."""
        log_T: List[List[float]] = []
        for fid in sorted_ids:
            row: List[float] = []
            for tid in sorted_ids:
                w = float(weight_matrix[fid][tid])
                row.append(math.log(w) if w > 0.0 else _FORBIDDEN_LOG_PENALTY)
            log_T.append(row)
        return log_T

    @staticmethod
    def _apply_rules_override(
        base_past_cls: ee.Image,
        orig_past_cls: ee.Image,
        future_cls: ee.Image,
        rules: Dict[Tuple[int, int], int],
        multiplier: int,
    ) -> ee.Image:
        """Override pixels matching explicit `(from_id, to_id) -> target_id` rules."""
        if not rules:
            return base_past_cls
        orig_code = orig_past_cls.multiply(multiplier).add(future_cls)
        from_codes = [fid * multiplier + tid for (fid, tid) in rules.keys()]
        targets = [int(t) for t in rules.values()]
        rule_hit = orig_code.remap(from_codes, targets, -999999)
        return base_past_cls.where(rule_hit.neq(-999999), rule_hit).toInt().rename('classification')

    def _run_hmm_anchored_pair(
        self,
        past_map: ee.Image,
        future_map: ee.Image,
        prob_bands: Sequence[str],
        validated_matrix: Dict[str, Any],
        probability_scale: float,
        spatial_smoothing: bool,
        spatial_weight: float,
    ) -> ee.Image:
        """
        Transition-constrained MAP inference holding ``future_map`` fixed.

        For each pixel with future class $c_j$, finds $c_i$ maximizing
        $\\log P_0(c_i) [+ \\alpha \\cdot \\text{neigh}_0(c_i)] + \\log W_{i, j}$,
        with forbidden transitions penalized by ``_FORBIDDEN_LOG_PENALTY``.
        """
        sorted_ids = validated_matrix['class_ids']
        weight_matrix = validated_matrix['weight_matrix']
        binary_matrix = validated_matrix['binary_matrix']
        k = len(sorted_ids)
        log_T = self._log_transition_matrix(weight_matrix, sorted_ids)

        future_cls = future_map.select('classification').toInt()
        emit_log = self._emission_log_probs(
            soft_image=past_map,
            prob_bands=prob_bands,
            probability_scale=probability_scale,
            spatial_smoothing=spatial_smoothing,
            spatial_weight=spatial_weight,
            class_ids=sorted_ids,
        )

        # For each candidate past state i, remap future_cls -> T[i][future_state]
        trans_bands = [
            future_cls.remap(sorted_ids, log_T[i], _FORBIDDEN_LOG_PENALTY)
            for i in range(k)
        ]
        trans_img = ee.Image.cat(trans_bands)

        total_scores = emit_log.add(trans_img)
        best_idx = total_scores.toArray().arrayArgmax().arrayGet(0)
        decoded_cls = best_idx.remap(list(range(k)), sorted_ids).toInt()

        # Guard: if a future class has zero allowed incoming transitions, keep future_cls
        col_has_allowed = [
            1 if any(binary_matrix[fid][tid] for fid in sorted_ids) else 0
            for tid in sorted_ids
        ]
        has_allowed_img = future_cls.remap(sorted_ids, col_has_allowed, 0)
        return decoded_cls.where(has_allowed_img.eq(0), future_cls).toInt().rename('classification')

    def _run_viterbi_sequence(
        self,
        soft_maps: Sequence[ee.Image],
        prob_bands_list: Sequence[Sequence[str]],
        validated_matrix: Dict[str, Any],
        probability_scale: float,
        spatial_smoothing: bool,
        spatial_weight: float,
    ) -> List[ee.Image]:
        """
        Full N-step joint Viterbi MAP decoding across ordered periods $t = 0 \\dots T-1$
        without holding the final period anchored.
        """
        sorted_ids = validated_matrix['class_ids']
        weight_matrix = validated_matrix['weight_matrix']
        k = len(sorted_ids)
        n_steps = len(soft_maps)
        log_T = self._log_transition_matrix(weight_matrix, sorted_ids)

        emissions = [
            self._emission_log_probs(
                soft_image=soft_maps[t],
                prob_bands=prob_bands_list[t],
                probability_scale=probability_scale,
                spatial_smoothing=spatial_smoothing,
                spatial_weight=spatial_weight,
                class_ids=sorted_ids,
            )
            for t in range(n_steps)
        ]

        # Forward pass: v_img holds K bands [V_t(0), ..., V_t(K-1)]
        v_img = emissions[0]
        backpointers: List[ee.Image] = []

        for t in range(1, n_steps):
            emit_t = emissions[t]
            next_v_bands: List[ee.Image] = []
            bp_bands: List[ee.Image] = []

            for j in range(k):
                col_scores = ee.Image.constant([log_T[i][j] for i in range(k)])
                cand = v_img.add(col_scores)
                cand_arr = cand.toArray()
                best_prev_score = cand_arr.arrayReduce(ee.Reducer.max(), [0]).arrayGet([0])
                best_prev_idx = cand_arr.arrayArgmax().arrayGet(0)

                v_next_j = best_prev_score.add(emit_t.select(prob_bands_list[t][j]))
                next_v_bands.append(v_next_j)
                bp_bands.append(best_prev_idx)

            v_img = ee.Image.cat(next_v_bands)
            backpointers.append(ee.Image.cat(bp_bands))

        # Backtrace from t = T - 1 down to 0
        decoded_indices: List[Optional[ee.Image]] = [None] * n_steps
        cur_idx = v_img.toArray().arrayArgmax().arrayGet(0)
        decoded_indices[-1] = cur_idx

        for t in range(n_steps - 2, -1, -1):
            bp_img = backpointers[t]
            prev_idx = ee.Image.constant(0)
            for j in range(k):
                prev_idx = prev_idx.where(cur_idx.eq(j), bp_img.select(j))
            decoded_indices[t] = prev_idx
            cur_idx = prev_idx

        return [
            idx_img.remap(list(range(k)), sorted_ids).toInt().rename('classification')
            for idx_img in decoded_indices
        ]

    def _run_rule_reclassify_pair(
        self,
        past_map: ee.Image,
        future_map: ee.Image,
        validated_matrix: Dict[str, Any],
        multiplier: int,
        rules: Dict[Tuple[int, int], int],
        selected_transitions: Optional[Sequence[Tuple[int, int]]] = None,
        unmatched_policy: str = 'to_future',
    ) -> ee.Image:
        """Secondary rule-based reclassification on hard `'classification'` bands."""
        if unmatched_policy not in ('to_future', 'keep'):
            raise ValueError(
                f"unmatched_policy must be 'to_future' or 'keep', got '{unmatched_policy}'"
            )

        sorted_ids = validated_matrix['class_ids']
        binary_matrix = validated_matrix['binary_matrix']
        selected_set = set(tuple(p) for p in selected_transitions) if selected_transitions else None

        past_cls = past_map.select('classification').toInt()
        future_cls = future_map.select('classification').toInt()
        code_img = past_cls.multiply(multiplier).add(future_cls)

        from_codes: List[int] = []
        target_ids: List[int] = []
        for fid in sorted_ids:
            for tid in sorted_ids:
                if binary_matrix[fid][tid] and (fid, tid) not in rules:
                    continue
                if selected_set is not None and (fid, tid) not in selected_set and (fid, tid) not in rules:
                    continue
                if (fid, tid) in rules:
                    from_codes.append(fid * multiplier + tid)
                    target_ids.append(int(rules[(fid, tid)]))
                elif not binary_matrix[fid][tid] and unmatched_policy == 'to_future':
                    from_codes.append(fid * multiplier + tid)
                    target_ids.append(int(tid))

        if not from_codes:
            return past_cls.rename('classification')

        remapped = code_img.remap(from_codes, target_ids, -999999)
        return past_cls.where(remapped.neq(-999999), remapped).toInt().rename('classification')

    def _summarize_correction(
        self,
        orig_past: ee.Image,
        corr_past: ee.Image,
        future_cls: ee.Image,
        binary_matrix: Dict[int, Dict[int, bool]],
        sorted_ids: Sequence[int],
        multiplier: int,
        geom: ee.Geometry,
        scale: int,
    ) -> Tuple[int, int]:
        """Compute `(corrected_pixel_count, remaining_inconsistent_count)` via GEE reduceRegion."""
        changed = corr_past.select('classification').neq(orig_past.select('classification'))
        corr_code = corr_past.select('classification').toInt().multiply(multiplier).add(
            future_cls.select('classification').toInt()
        )
        forbidden_codes = [
            fid * multiplier + tid
            for fid in sorted_ids
            for tid in sorted_ids
            if not binary_matrix[fid][tid]
        ]
        if forbidden_codes:
            rem_inconsistent = corr_code.remap(forbidden_codes, [1] * len(forbidden_codes), 0)
        else:
            rem_inconsistent = ee.Image.constant(0)

        combined = ee.Image.cat([
            changed.rename('changed').toInt(),
            rem_inconsistent.rename('remaining').toInt(),
        ])
        sums = combined.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=geom,
            scale=scale,
            maxPixels=1e13,
            tileScale=4,
        ).getInfo() or {}

        return int(round(float(sums.get('changed', 0) or 0))), int(
            round(float(sums.get('remaining', 0) or 0))
        )

    def correct_inconsistencies(
        self,
        past_map: ee.Image,
        future_map: ee.Image,
        consistency_matrix: Any,
        aoi: Optional[Any] = None,
        method: str = 'auto',
        anchor_reference: bool = True,
        rules: Optional[Dict[Any, int]] = None,
        selected_transitions: Optional[Sequence[Tuple[int, int]]] = None,
        unmatched_policy: str = 'to_future',
        spatial_smoothing: bool = False,
        spatial_weight: float = 0.25,
        probability_scale: float = 100.0,
        scale: int = 30,
        class_ids: Optional[Sequence[int]] = None,
        class_names: Optional[Any] = None,
        prob_band_map: Optional[Dict[int, str]] = None,
        compute_stats: bool = True,
    ) -> Dict[str, Any]:
        """
        Eliminate impossible transitions between ``past_map`` and ``future_map``.

        Parameters
        ----------
        method : {'auto', 'hmm', 'reclassify'}, default='auto'
            ``'hmm'`` uses transition-constrained MAP/Viterbi over soft probability bands;
            ``'reclassify'`` applies explicit transition rules (or snaps to ``to_id``).
            ``'auto'`` selects ``'hmm'`` when probability bands are present on ``past_map``,
            and falls back to ``'reclassify'`` otherwise.
        anchor_reference : bool, default=True
            If True, holds ``future_map`` fixed and corrects ``past_map``.
            If False (with ``method='hmm'``), runs joint 2-step Viterbi updating both maps.
        rules : dict, optional
            Explicit ``{(from_id, to_id): target_id}`` rules. Applied directly in
            ``'reclassify'`` mode or layered as manual overrides after HMM decoding.
        """
        ensure_ee_initialized()
        if method not in ('auto', 'hmm', 'reclassify'):
            raise ValueError(f"method must be 'auto', 'hmm', or 'reclassify', got '{method}'")

        validated = self._require_valid_matrix(consistency_matrix, class_ids=class_ids)
        sorted_ids = validated['class_ids']
        binary_matrix = validated['binary_matrix']
        multiplier = self._compute_multiplier(sorted_ids)
        norm_rules = _normalize_rules(rules)

        past_prob_bands = self._resolve_prob_bands(
            past_map, sorted_ids, prob_band_map=prob_band_map, class_names=class_names
        )

        if method == 'auto':
            method_used = 'hmm' if past_prob_bands is not None else 'reclassify'
        else:
            method_used = method

        if method_used == 'hmm' and past_prob_bands is None:
            raise ValueError(
                "HMM consistency correction requires soft probability bands on past_map "
                "(pass method='reclassify' or 'auto' for hard classification maps)."
            )

        corrected_future_map: Optional[ee.Image] = None
        if method_used == 'hmm':
            if anchor_reference:
                corrected_past = self._run_hmm_anchored_pair(
                    past_map=past_map,
                    future_map=future_map,
                    prob_bands=past_prob_bands,
                    validated_matrix=validated,
                    probability_scale=probability_scale,
                    spatial_smoothing=spatial_smoothing,
                    spatial_weight=spatial_weight,
                )
            else:
                future_prob_bands = self._resolve_prob_bands(
                    future_map, sorted_ids, prob_band_map=prob_band_map, class_names=class_names
                )
                if future_prob_bands is None:
                    raise ValueError(
                        "Joint Viterbi (anchor_reference=False) requires soft probability bands on both maps."
                    )
                decoded = self._run_viterbi_sequence(
                    soft_maps=[past_map, future_map],
                    prob_bands_list=[past_prob_bands, future_prob_bands],
                    validated_matrix=validated,
                    probability_scale=probability_scale,
                    spatial_smoothing=spatial_smoothing,
                    spatial_weight=spatial_weight,
                )
                corrected_past, corrected_future_map = decoded[0], decoded[1]

            if norm_rules:
                ref_for_rule = corrected_future_map if corrected_future_map is not None else future_map
                corrected_past = self._apply_rules_override(
                    base_past_cls=corrected_past,
                    orig_past_cls=past_map.select('classification').toInt(),
                    future_cls=ref_for_rule.select('classification').toInt(),
                    rules=norm_rules,
                    multiplier=multiplier,
                )
        else:
            corrected_past = self._run_rule_reclassify_pair(
                past_map=past_map,
                future_map=future_map,
                validated_matrix=validated,
                multiplier=multiplier,
                rules=norm_rules,
                selected_transitions=selected_transitions,
                unmatched_policy=unmatched_policy,
            )

        corrected_pixel_count: Optional[int] = None
        remaining_inconsistent_count: Optional[int] = None
        geom = self._extract_aoi_geometry(aoi)
        if compute_stats and geom is not None:
            ref_future = corrected_future_map if corrected_future_map is not None else future_map
            corrected_pixel_count, remaining_inconsistent_count = self._summarize_correction(
                orig_past=past_map,
                corr_past=corrected_past,
                future_cls=ref_future,
                binary_matrix=binary_matrix,
                sorted_ids=sorted_ids,
                multiplier=multiplier,
                geom=geom,
                scale=scale,
            )

        out: Dict[str, Any] = {
            'corrected_map': corrected_past,
            'method_used': method_used,
            'anchor_reference': anchor_reference,
            'corrected_pixel_count': corrected_pixel_count,
            'remaining_inconsistent_count': remaining_inconsistent_count,
            'class_ids': sorted_ids,
        }
        if corrected_future_map is not None:
            out['corrected_future_map'] = corrected_future_map
        return out

    def correct_multitemporal_sequence(
        self,
        maps_by_period: Dict[str, ee.Image],
        period_order: Sequence[str],
        consistency_matrix: Any,
        aoi: Optional[Any] = None,
        method: str = 'auto',
        anchor_reference: bool = True,
        rules: Optional[Dict[Any, int]] = None,
        selected_transitions: Optional[Sequence[Tuple[int, int]]] = None,
        unmatched_policy: str = 'to_future',
        spatial_smoothing: bool = False,
        spatial_weight: float = 0.25,
        probability_scale: float = 100.0,
        scale: int = 30,
        class_ids: Optional[Sequence[int]] = None,
        class_names: Optional[Any] = None,
        prob_bands_by_period: Optional[Dict[str, Dict[int, str]]] = None,
        compute_stats: bool = True,
    ) -> Dict[str, Any]:
        """
        Propagate consistency correction across $N >= 2$ chronologically ordered periods.

        - When ``anchor_reference=True``: holds the final period ``period_order[-1]`` fixed
          and corrects backward pairwise (``t_{N-2} -> ... -> t_0``).
        - When ``anchor_reference=False``: runs full joint N-period Viterbi decoding over
          all periods simultaneously (requires soft probability bands across all periods).
        """
        ensure_ee_initialized()
        order = list(period_order)
        if len(order) < 2:
            raise ValueError(f"correct_multitemporal_sequence requires >= 2 periods, got {len(order)}.")
        for lbl in order:
            if lbl not in maps_by_period:
                raise ValueError(f"Period '{lbl}' missing from maps_by_period.")

        validated = self._require_valid_matrix(consistency_matrix, class_ids=class_ids)
        sorted_ids = validated['class_ids']
        binary_matrix = validated['binary_matrix']
        multiplier = self._compute_multiplier(sorted_ids)
        norm_rules = _normalize_rules(rules)

        if not anchor_reference and method in ('auto', 'hmm'):
            soft_list = [maps_by_period[p] for p in order]
            pb_lists: List[List[str]] = []
            for p in order:
                pb_map = (prob_bands_by_period or {}).get(p)
                resolved = self._resolve_prob_bands(
                    maps_by_period[p], sorted_ids, prob_band_map=pb_map, class_names=class_names
                )
                if resolved is None:
                    if method == 'hmm':
                        raise ValueError(
                            f"Joint Viterbi (anchor_reference=False) requires probability bands on period '{p}'."
                        )
                    pb_lists = []
                    break
                pb_lists.append(resolved)

            if pb_lists:
                decoded = self._run_viterbi_sequence(
                    soft_maps=soft_list,
                    prob_bands_list=pb_lists,
                    validated_matrix=validated,
                    probability_scale=probability_scale,
                    spatial_smoothing=spatial_smoothing,
                    spatial_weight=spatial_weight,
                )
                corrected_maps: Dict[str, ee.Image] = {p: img for p, img in zip(order, decoded)}
                if norm_rules:
                    for idx in range(len(order) - 2, -1, -1):
                        p_cur = order[idx]
                        p_nxt = order[idx + 1]
                        corrected_maps[p_cur] = self._apply_rules_override(
                            base_past_cls=corrected_maps[p_cur],
                            orig_past_cls=maps_by_period[p_cur].select('classification').toInt(),
                            future_cls=corrected_maps[p_nxt].select('classification').toInt(),
                            rules=norm_rules,
                            multiplier=multiplier,
                        )
                pair_summaries: Dict[str, Dict[str, Any]] = {}
                geom = self._extract_aoi_geometry(aoi)
                for idx in range(len(order) - 1):
                    p_cur, p_nxt = order[idx], order[idx + 1]
                    c_cnt = None
                    r_cnt = None
                    if compute_stats and geom is not None:
                        c_cnt, r_cnt = self._summarize_correction(
                            orig_past=maps_by_period[p_cur],
                            corr_past=corrected_maps[p_cur],
                            future_cls=corrected_maps[p_nxt],
                            binary_matrix=binary_matrix,
                            sorted_ids=sorted_ids,
                            multiplier=multiplier,
                            geom=geom,
                            scale=scale,
                        )
                    pair_summaries[f"{p_cur}->{p_nxt}"] = {
                        'corrected_pixel_count': c_cnt,
                        'remaining_inconsistent_count': r_cnt,
                    }
                return {
                    'corrected_maps': corrected_maps,
                    'period_order': order,
                    'method_used': 'hmm_joint_viterbi',
                    'anchor_reference': False,
                    'pair_summaries': pair_summaries,
                    'class_ids': sorted_ids,
                }

        # Backward pairwise correction: hold final period fixed and step backward t_{N-2} -> t_0
        corrected_maps = {order[-1]: maps_by_period[order[-1]].select('classification').toInt()}
        pair_summaries = {}
        methods_used = set()

        for idx in range(len(order) - 2, -1, -1):
            p_cur = order[idx]
            p_nxt = order[idx + 1]
            pb_map = (prob_bands_by_period or {}).get(p_cur)
            res = self.correct_inconsistencies(
                past_map=maps_by_period[p_cur],
                future_map=corrected_maps[p_nxt],
                consistency_matrix=validated,
                aoi=aoi,
                method=method,
                anchor_reference=True,
                rules=norm_rules,
                selected_transitions=selected_transitions,
                unmatched_policy=unmatched_policy,
                spatial_smoothing=spatial_smoothing,
                spatial_weight=spatial_weight,
                probability_scale=probability_scale,
                scale=scale,
                class_ids=sorted_ids,
                class_names=class_names,
                prob_band_map=pb_map,
                compute_stats=compute_stats,
            )
            corrected_maps[p_cur] = res['corrected_map']
            methods_used.add(res['method_used'])
            pair_summaries[f"{p_cur}->{p_nxt}"] = {
                'method_used': res['method_used'],
                'corrected_pixel_count': res['corrected_pixel_count'],
                'remaining_inconsistent_count': res['remaining_inconsistent_count'],
            }

        ordered_corrected = {p: corrected_maps[p] for p in order}
        return {
            'corrected_maps': ordered_corrected,
            'period_order': order,
            'method_used': next(iter(methods_used)) if len(methods_used) == 1 else 'mixed',
            'anchor_reference': True,
            'pair_summaries': pair_summaries,
            'class_ids': sorted_ids,
        }
