"""Unit & numerical verification tests for Phase 6: Post-Classification Consistency Analysis."""

from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
import pytest

from luma_ge.post_classification import ConsistencyAnalyzer


class ArrayImage:
    """Minimal NumPy-backed ee.Image simulator for numerically verifying GEE pixel algebra."""

    def __init__(self, bands_dict):
        # Store ordered {band_name: 2D np.ndarray (float64)}
        self.bands = {k: np.asarray(v, dtype=np.float64) for k, v in bands_dict.items()}
        self._names = list(self.bands.keys())

    @classmethod
    def constant(cls, val):
        if isinstance(val, (list, tuple)):
            return cls({f"const_{i}": np.full((1, 1), v, dtype=np.float64) for i, v in enumerate(val)})
        return cls({"constant": np.full((1, 1), val, dtype=np.float64)})

    @classmethod
    def cat(cls, images):
        out = {}
        seen = {}
        for img in images:
            for name, arr in img.bands.items():
                key = name
                if key in out:
                    seen[name] = seen.get(name, 0) + 1
                    key = f"{name}_{seen[name]}"
                out[key] = arr.copy()
        return cls(out)

    def _broadcast_pair(self, other):
        if isinstance(other, ArrayImage):
            other_arrs = list(other.bands.values())
        else:
            other_arrs = [np.full_like(next(iter(self.bands.values())), float(other))]
        if len(other_arrs) == 1 and len(self._names) > 1:
            other_arrs = other_arrs * len(self._names)
        return other_arrs

    def _binary_op(self, other, fn):
        other_arrs = self._broadcast_pair(other)
        out = {}
        for name, o_arr in zip(self._names, other_arrs):
            shape = np.broadcast_shapes(self.bands[name].shape, o_arr.shape)
            a = np.broadcast_to(self.bands[name], shape)
            b = np.broadcast_to(o_arr, shape)
            out[name] = fn(a, b)
        return ArrayImage(out)

    def bandNames(self):
        return MagicMock(getInfo=MagicMock(return_value=list(self._names)))

    def select(self, selector):
        if isinstance(selector, int):
            k = self._names[selector]
            return ArrayImage({k: self.bands[k]})
        if isinstance(selector, str):
            return ArrayImage({selector: self.bands[selector]})
        return ArrayImage({k: self.bands[k] for k in selector})

    def rename(self, new_name):
        if isinstance(new_name, str):
            assert len(self._names) == 1
            return ArrayImage({new_name: self.bands[self._names[0]]})
        return ArrayImage({nn: self.bands[old] for old, nn in zip(self._names, new_name)})

    def toInt(self):
        return ArrayImage({k: v.astype(np.int64).astype(np.float64) for k, v in self.bands.items()})

    def toFloat(self):
        return ArrayImage({k: v.astype(np.float64) for k, v in self.bands.items()})

    def add(self, other):
        return self._binary_op(other, lambda a, b: a + b)

    def multiply(self, other):
        return self._binary_op(other, lambda a, b: a * b)

    def divide(self, other):
        return self._binary_op(other, lambda a, b: a / b)

    def max(self, other):
        return self._binary_op(other, np.maximum)

    def log(self):
        return ArrayImage({k: np.log(v) for k, v in self.bands.items()})

    def eq(self, other):
        return self._binary_op(other, lambda a, b: (a == b).astype(np.float64))

    def neq(self, other):
        return self._binary_op(other, lambda a, b: (a != b).astype(np.float64))

    def where(self, cond, val):
        cond_arr = cond.bands[cond._names[0]].astype(bool)
        if isinstance(val, ArrayImage):
            val_arr = val.bands[val._names[0]]
        else:
            val_arr = float(val)
        a = self.bands[self._names[0]]
        shape = np.broadcast_shapes(a.shape, cond_arr.shape, np.asarray(val_arr).shape)
        res = np.where(np.broadcast_to(cond_arr, shape), np.broadcast_to(val_arr, shape), np.broadcast_to(a, shape))
        return ArrayImage({self._names[0]: res})

    def remap(self, from_vals, to_vals, default_val=0):
        arr = self.bands[self._names[0]]
        out = np.full(arr.shape, float(default_val), dtype=np.float64)
        for fv, tv in zip(from_vals, to_vals):
            out = np.where(arr == float(fv), float(tv), out)
        return ArrayImage({"remapped": out})

    def selfMask(self):
        return self

    def updateMask(self, mask):
        return self

    def toArray(self):
        stack = np.stack([self.bands[k] for k in self._names], axis=0)
        return _ArrayOp(stack)

    def reduceNeighborhood(self, reducer=None, kernel=None):
        # Compute 3x3 edge-padded mean matching GEE Kernel.square(1)
        out = {}
        for k, arr in self.bands.items():
            padded = np.pad(arr, 1, mode="edge")
            acc = np.zeros_like(arr, dtype=np.float64)
            for dr in range(3):
                for dc in range(3):
                    acc += padded[dr : dr + arr.shape[0], dc : dc + arr.shape[1]]
            out[k] = acc / 9.0
        return ArrayImage(out)

    @property
    def values(self):
        assert len(self._names) == 1
        return self.bands[self._names[0]]


class _ArrayOp:
    def __init__(self, stack):
        self.stack = stack  # (K, H, W)

    def arrayArgmax(self):
        idx = np.argmax(self.stack, axis=0)
        return _ArrayGetOp(idx)

    def arrayReduce(self, reducer=None, axes=None):
        mx = np.max(self.stack, axis=0)
        return _ArrayGetOp(mx)


class _ArrayGetOp:
    def __init__(self, arr):
        self.arr = arr

    def arrayGet(self, pos):
        return ArrayImage({"array": self.arr.astype(np.float64)})


def _patch_ee_with_array_image(mock_ee):
    mock_ee.Image = ArrayImage
    mock_ee.FeatureCollection = tuple  # dummy non-matching type
    mock_ee.Reducer.mean.return_value = "mean"
    mock_ee.Reducer.max.return_value = "max"
    mock_ee.Kernel.square.return_value = "square1"


class TestTransitionMatrixValidation:
    def test_validates_dataframe_csv_and_nested_dict(self):
        ca = ConsistencyAnalyzer()
        df = pd.DataFrame(
            [
                {"from_id": 1, "1": 1, "2": 0.5, "3": 0},
                {"from_id": 2, "1": 0, "2": 1, "3": 1},
                {"from_id": 3, "1": 0, "2": 0, "3": 1},
            ]
        )
        res = ca.validate_transition_matrix(df, class_ids=[1, 2, 3])
        assert res["valid"] is True
        assert res["missing_pairs"] == []
        assert res["binary_matrix"][3][1] is False
        assert res["weight_matrix"][1][2] == pytest.approx(0.5)

        csv_bytes = df.to_csv(index=False).encode("utf-8")
        res_csv = ca.validate_transition_matrix(csv_bytes, class_ids=[1, 2, 3])
        assert res_csv["valid"] is True
        assert res_csv["binary_matrix"] == res["binary_matrix"]

    def test_reports_missing_pairs_and_rejects_invalid_weights(self):
        ca = ConsistencyAnalyzer()
        incomplete = {1: {1: 1, 2: 0}, 2: {1: 0, 2: 1}}
        res = ca.validate_transition_matrix(incomplete, class_ids=[1, 2, 3])
        assert res["valid"] is False
        assert set(res["missing_pairs"]) == {(1, 3), (2, 3), (3, 1), (3, 2), (3, 3)}

        with pytest.raises(ValueError, match="\\[0.0, 1.0\\]"):
            ca.validate_transition_matrix({1: {1: 1.5, 2: 0}, 2: {1: 0, 2: 1}}, class_ids=[1, 2])


class TestCrossTabulation:
    @patch("luma_ge.post_classification.ensure_ee_initialized")
    def test_cross_tabulate_decodes_counts_and_masks_inconsistent(self, mock_init):
        ca = ConsistencyAnalyzer()
        past_map = MagicMock()
        future_map = MagicMock()
        code_img = MagicMock()
        past_cls = MagicMock()
        future_cls = MagicMock()

        past_map.select.return_value.toInt.return_value = past_cls
        future_map.select.return_value.toInt.return_value = future_cls
        past_cls.multiply.return_value.add.return_value.rename.return_value = code_img

        # 1->1 consistent (400 px), 3->1 inconsistent (100 px), 2->2 consistent (50 px)
        code_img.reduceRegion.return_value.getInfo.return_value = {
            "transition_code": {"1001": 400, "3001.0": 100, "2002": 50}
        }
        code_img.remap.return_value.selfMask.return_value.rename.return_value = "mask_img"
        code_img.updateMask.return_value.rename.return_value = "code_img"

        matrix = {
            1: {1: 1, 2: 1, 3: 1},
            2: {1: 0, 2: 1, 3: 1},
            3: {1: 0, 2: 0, 3: 1},
        }
        scheme_df = pd.DataFrame([
            {"ID": 1, "Land Cover Class": "Forest"},
            {"ID": 2, "Land Cover Class": "Cropland"},
            {"ID": 3, "Land Cover Class": "Urban"},
        ])

        res = ca.cross_tabulate(
            past_map=past_map,
            future_map=future_map,
            consistency_matrix=matrix,
            aoi=MagicMock(),
            scale=30,
            class_names=scheme_df,
            class_ids=[1, 2, 3],
        )
        assert res["multiplier"] == 1000
        assert res["total_pixels"] == 550
        assert res["inconsistent_pixels"] == 100
        inc = res["inconsistent_transitions"]
        assert len(inc) == 1
        assert inc.iloc[0]["from_id"] == 3
        assert inc.iloc[0]["to_id"] == 1
        assert inc.iloc[0]["from_name"] == "Urban"
        assert inc.iloc[0]["to_name"] == "Forest"
        assert inc.iloc[0]["area_m2"] == pytest.approx(100 * 900.0)


class TestHMMViterbiAndRuleReclassify:
    @patch("luma_ge.post_classification.ensure_ee_initialized")
    @patch("luma_ge.post_classification.ee")
    def test_hmm_anchored_map_math_with_scheme_band_names_and_rule_override(self, mock_ee, mock_init):
        _patch_ee_with_array_image(mock_ee)
        ca = ConsistencyAnalyzer()

        # 3 classes: 1=Forest, 2=Cropland, 3=Urban
        # Matrix: 3->1 forbidden (0.0); 1->1 weighted 0.9, 2->1 weighted 0.2
        matrix = {
            1: {1: 0.9, 2: 1.0, 3: 1.0},
            2: {1: 0.2, 2: 1.0, 3: 1.0},
            3: {1: 0.0, 2: 0.0, 3: 1.0},
        }
        scheme_df = pd.DataFrame([
            {"ID": 1, "Land Cover Class": "Forest"},
            {"ID": 2, "Land Cover Class": "Cropland"},
            {"ID": 3, "Land Cover Class": "Urban"},
        ])

        # Pixel (0,0): argmax=3 (55%) -> future=1; consistent alternatives P(1)=20%, P(2)=25%.
        #   score(1) = log(0.20) + log(0.9) = log(0.18) > score(2) = log(0.25) + log(0.2) = log(0.05) -> selects 1!
        # Pixel (0,1): argmax=3 (70%) -> future=3 (consistent) -> stays 3!
        # Pixel (0,2): argmax=2 (50%) -> future=1, and P(1)=10%, P(2)=40% (1->1:0.09 vs 2->1:0.08 -> picks 1;
        #   unless rule (2, 1) -> 2 overrides it!)
        past_soft = ArrayImage({
            "prob_Forest": [[20, 15, 10]],
            "prob_Cropland": [[25, 15, 40]],
            "prob_Urban": [[55, 70, 50]],
            "classification": [[3, 3, 2]],
        })
        future_map = ArrayImage({"classification": [[1, 3, 1]]})

        res_hmm = ca.correct_inconsistencies(
            past_map=past_soft,
            future_map=future_map,
            consistency_matrix=matrix,
            method="auto",
            anchor_reference=True,
            class_ids=[1, 2, 3],
            class_names=scheme_df,
            compute_stats=False,
        )
        assert res_hmm["method_used"] == "hmm"
        assert res_hmm["corrected_map"].values.tolist() == [[1.0, 3.0, 1.0]]

        # Manual rule override: force (3, 1) -> 2 instead of HMM's choice 1
        res_override = ca.correct_inconsistencies(
            past_map=past_soft,
            future_map=future_map,
            consistency_matrix=matrix,
            method="hmm",
            anchor_reference=True,
            rules={(3, 1): 2},
            class_ids=[1, 2, 3],
            class_names=scheme_df,
            compute_stats=False,
        )
        assert res_override["corrected_map"].values.tolist() == [[2.0, 3.0, 1.0]]

    @patch("luma_ge.post_classification.ensure_ee_initialized")
    @patch("luma_ge.post_classification.ee")
    def test_joint_viterbi_3_period_finds_global_optimum(self, mock_ee, mock_init):
        _patch_ee_with_array_image(mock_ee)
        ca = ConsistencyAnalyzer()

        # Allowed chain: 1->1, 1->2, 2->2, 2->3, 3->3 (cannot revert 2->1 or 3->1/2)
        matrix = {
            1: {1: 1.0, 2: 1.0, 3: 0.0},
            2: {1: 0.0, 2: 1.0, 3: 1.0},
            3: {1: 0.0, 2: 0.0, 3: 1.0},
        }
        # Raw greedy argmax trajectory: t0=3 (0.60), t1=1 (0.70), t2=1 (0.90) -> impossible!
        # Optimal global consistent trajectory with highest product:
        #   t0=1 (0.35), t1=1 (0.70), t2=1 (0.90) -> 0.35 * 0.70 * 0.90 = 0.2205
        # vs (3,3,3): 0.60 * 0.05 * 0.02 = 0.0006
        t0 = ArrayImage({"prob_1": [[35]], "prob_2": [[5]], "prob_3": [[60]], "classification": [[3]]})
        t1 = ArrayImage({"prob_1": [[70]], "prob_2": [[25]], "prob_3": [[5]], "classification": [[1]]})
        t2 = ArrayImage({"prob_1": [[90]], "prob_2": [[8]], "prob_3": [[2]], "classification": [[1]]})

        seq = ca.correct_multitemporal_sequence(
            maps_by_period={"t2010": t0, "t2018": t1, "t2025": t2},
            period_order=["t2010", "t2018", "t2025"],
            consistency_matrix=matrix,
            method="hmm",
            anchor_reference=False,
            class_ids=[1, 2, 3],
            compute_stats=False,
        )
        assert seq["method_used"] == "hmm_joint_viterbi"
        assert seq["corrected_maps"]["t2010"].values.tolist() == [[1.0]]
        assert seq["corrected_maps"]["t2018"].values.tolist() == [[1.0]]
        assert seq["corrected_maps"]["t2025"].values.tolist() == [[1.0]]

    @patch("luma_ge.post_classification.ensure_ee_initialized")
    @patch("luma_ge.post_classification.ee")
    def test_rule_based_fallback_and_backward_multitemporal_propagation(self, mock_ee, mock_init):
        _patch_ee_with_array_image(mock_ee)
        ca = ConsistencyAnalyzer()

        matrix = {
            1: {1: 1, 2: 1, 3: 1},
            2: {1: 0, 2: 1, 3: 1},
            3: {1: 0, 2: 0, 3: 1},
        }
        # Hard maps (no probability bands) -> auto falls back to 'reclassify'
        # Pixel 0: 3 -> 2 -> 1 (inconsistent at t1->t2, then propagated to t0->t1)
        #   Step t1 vs t2 (anchored): t1=2 -> t2=1 forbidden -> snaps to t2=1
        #   Step t0 vs corrected_t1 (=1): t0=3 -> 1 forbidden -> rule (3,1)->2 maps to 2!
        #     Wait: if t0 becomes 2, check backward propagation order: explicit rule (3, 1)->2 applies cleanly!
        # Pixel 1: 3 -> 1 -> 1 (already t1 consistent at 1, then t0=3->1 triggers rule -> 2)
        t0 = ArrayImage({"classification": [[3, 3]]})
        t1 = ArrayImage({"classification": [[2, 1]]})
        t2 = ArrayImage({"classification": [[1, 1]]})

        res = ca.correct_multitemporal_sequence(
            maps_by_period={"t2010": t0, "t2018": t1, "t2025": t2},
            period_order=["t2010", "t2018", "t2025"],
            consistency_matrix=matrix,
            method="auto",
            anchor_reference=True,
            rules={(3, 1): 2},
            class_ids=[1, 2, 3],
            compute_stats=False,
        )
        assert res["method_used"] == "reclassify"
        assert res["corrected_maps"]["t2025"].values.tolist() == [[1.0, 1.0]]
        assert res["corrected_maps"]["t2018"].values.tolist() == [[1.0, 1.0]]
        assert res["corrected_maps"]["t2010"].values.tolist() == [[2.0, 2.0]]

        # Keep unmatched inconsistent pairs when unmatched_policy='keep'
        keep_res = ca.correct_inconsistencies(
            past_map=ArrayImage({"classification": [[3, 2]]}),
            future_map=ArrayImage({"classification": [[1, 1]]}),
            consistency_matrix=matrix,
            method="reclassify",
            rules={(3, 1): 1},
            unmatched_policy="keep",
            class_ids=[1, 2, 3],
            compute_stats=False,
        )
        # (3,1)->1 fixed via explicit rule; (2,1) kept as 2 under 'keep' policy
        assert keep_res["corrected_map"].values.tolist() == [[1.0, 2.0]]
