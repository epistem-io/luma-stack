"""Unit tests for Phase 3 (Predictor Adapter) and Phase 4 (Classification & Accuracy)."""

from unittest.mock import MagicMock, patch
import pandas as pd
import pytest

from luma_ge.multitemporal import (
    filter_compatible_indices,
    compute_single_image_indices,
    compute_multitemporal_predictors,
)
from luma_ge.classification import (
    FeatureExtraction,
    Generate_LULC,
    build_prob_band_names,
    compute_log_loss_from_arrays,
)
from luma_ge.accuracy import thematic_accuracy


def _make_chainable_ee_image(bands=None):
    img = MagicMock()
    for method in (
        "select", "rename", "addBands", "multiply", "round", "byte",
        "arrayFlatten", "toArray", "arrayArgmax", "arrayGet", "arrayReduce",
        "remap", "classify", "clip",
    ):
        getattr(img, method).return_value = img
    img.bandNames.return_value.getInfo.return_value = bands or [
        "BLUE", "GREEN", "RED", "NIR", "SWIR1", "SWIR2"
    ]
    return img


class TestPhase3Predictors:
    def test_filter_compatible_indices_for_mss_3band_intersection(self):
        compatible, skipped = filter_compatible_indices(
            index_list=["NDVI", "EVI", "MNDWI", "GNDVI"],
            available_bands=["GREEN", "RED", "NIR"],
        )
        assert compatible == ["NDVI", "GNDVI"]
        assert skipped == ["EVI", "MNDWI"]

    def test_compute_single_image_indices_uses_only_available_bands(self):
        img = _make_chainable_ee_image(["GREEN", "RED", "NIR"])
        mock_calc = MagicMock(spec=["spyndex", "spyndex_indices", "validate_coefficients", "get_index_coefficients"])
        mock_calc.spyndex_indices = {"NDVI": MagicMock(bands=["N", "R"]), "EVI": MagicMock(bands=["N", "R", "B"])}
        mock_calc.get_index_coefficients.return_value = {}
        mock_calc.spyndex.computeIndex.return_value = _make_chainable_ee_image(["NDVI"])

        out = compute_single_image_indices(
            image=img,
            index_list=["NDVI", "EVI"],
            available_bands=["GREEN", "RED", "NIR"],
            spectral_calc=mock_calc,
        )
        assert out is not None
        call_kwargs = mock_calc.spyndex.computeIndex.call_args.kwargs
        assert call_kwargs["index"] == ["NDVI"]
        # Must NOT attempt to select BLUE or SWIR1
        selected_bands = [c.args[0] for c in img.select.call_args_list]
        assert "BLUE" not in selected_bands
        assert "SWIR1" not in selected_bands

    @patch("luma_ge.multitemporal.terrain_calculator")
    @patch("luma_ge.multitemporal.SpectralCalculator")
    @patch("luma_ge.multitemporal.ensure_ee_initialized")
    def test_compute_multitemporal_predictors_aligns_all_periods_in_model_transfer(
        self, mock_init, mock_sc_cls, mock_tc_cls
    ):
        mock_tc = MagicMock()
        mock_tc.calculate_elevation.return_value = _make_chainable_ee_image(["elevation"])
        mock_tc.calculate_slope.return_value = _make_chainable_ee_image(["slope"])
        mock_tc_cls.return_value = mock_tc

        mock_sc = MagicMock(spec=["spyndex", "spyndex_indices", "validate_coefficients", "get_index_coefficients", "calculate_indices_with_collection"])
        mock_sc.spyndex_indices = {
            "NDVI": MagicMock(bands=["N", "R"]),
            "MNDWI": MagicMock(bands=["G", "S1"]),
        }
        mock_sc.get_index_coefficients.return_value = {}
        mock_sc.spyndex.computeIndex.return_value = _make_chainable_ee_image(["NDVI"])
        mock_sc_cls.return_value = mock_sc

        composites_payload = {
            "composites": {
                "historical": _make_chainable_ee_image(["GREEN", "RED", "NIR"]),
                "baseline": _make_chainable_ee_image(["GREEN", "RED", "NIR"]),
            },
            "common_bands": ["GREEN", "RED", "NIR"],
            "period_order": ["historical", "baseline"],
            "histogram_matched": True,
        }
        predictor_config = {
            "individual_predictors": {
                "elevation": True,
                "slope": True,
                "aspect": False,
                "spectral_indices": ["NDVI", "MNDWI"],
            }
        }
        res = compute_multitemporal_predictors(
            composites_by_period=composites_payload,
            aoi=MagicMock(),
            predictor_config=predictor_config,
            mode="model_transfer",
        )
        # Predictors computed for BOTH periods, terrain computed ONCE
        assert set(res["predictor_stacks"].keys()) == {"historical", "baseline"}
        assert mock_tc.calculate_elevation.call_count == 1
        assert mock_tc.calculate_slope.call_count == 1
        # Incompatible MNDWI dropped; common feature bands match identically
        assert res["compatible_indices"] == ["NDVI"]
        assert res["skipped_indices"] == ["MNDWI"]
        assert res["feature_bands"]["historical"] == ["GREEN", "RED", "NIR", "elevation", "slope", "NDVI"]
        assert res["feature_bands"]["baseline"] == ["GREEN", "RED", "NIR", "elevation", "slope", "NDVI"]
        assert res["common_feature_bands"] == ["GREEN", "RED", "NIR", "elevation", "slope", "NDVI"]
        # Since histogram_matched=True, collection method must NOT be used
        mock_sc.calculate_indices_with_collection.assert_not_called()


class TestPhase4ClassificationAndAccuracy:
    @patch("luma_ge.classification.ee")
    def test_soft_classification_multiprobability_returns_model_and_bands(self, mock_ee):
        class FakeFC:
            pass

        class FakeImg:
            pass

        mock_ee.FeatureCollection = FakeFC
        mock_ee.Image = FakeImg

        train_fc = FakeFC()
        train_fc.aggregate_array = MagicMock(
            return_value=MagicMock(distinct=MagicMock(return_value=MagicMock(sort=MagicMock(return_value=MagicMock(getInfo=MagicMock(return_value=[3, 1, 2]))))))
        )

        img = FakeImg()
        img.bandNames = MagicMock(return_value=MagicMock(size=MagicMock(return_value=MagicMock(sqrt=MagicMock(return_value=MagicMock(ceil=MagicMock(return_value=2)))))))
        classified_img = _make_chainable_ee_image()
        img.classify = MagicMock(return_value=classified_img)

        mock_rf = MagicMock()
        mock_trained = MagicMock()
        mock_rf.setOutputMode.return_value = mock_rf
        mock_rf.train.return_value = mock_trained
        mock_ee.Classifier.smileRandomForest.return_value = mock_rf

        lulc = Generate_LULC()
        out_img, model_info = lulc.soft_classification(
            training_data=train_fc,
            class_property="class_id",
            image=img,
            mode="multiprobability",
            return_model=True,
        )
        mock_rf.setOutputMode.assert_called_once_with("MULTIPROBABILITY")
        assert model_info["mode"] == "multiprobability"
        assert model_info["class_ids"] == [1, 2, 3]
        assert model_info["model"] is mock_trained
        classified_img.arrayFlatten.assert_called_once_with([["prob_1", "prob_2", "prob_3"]])
        assert out_img is not None

    def test_classify_multitemporal_model_transfer_soft(self):
        lulc = Generate_LULC()
        ref_map = _make_chainable_ee_image(["prob_1", "prob_2", "classification", "confidence"])
        target_map = _make_chainable_ee_image(["prob_1", "prob_2", "classification", "confidence"])
        ref_model = MagicMock(name="rf_multiprob")

        lulc.soft_classification = MagicMock(
            return_value=(ref_map, {"model": ref_model, "class_ids": [1, 2], "mode": "multiprobability"})
        )
        lulc.classify_with_model = MagicMock(return_value=target_map)
        lulc.get_feature_importance = MagicMock(return_value=pd.DataFrame([{"Band": "NIR", "Importance": 1.0}]))
        lulc.evaluate_model = MagicMock(return_value={"overall_accuracy": 0.85, "kappa": 0.8})

        stacks = {
            "historical": _make_chainable_ee_image(["GREEN", "RED", "NIR"]),
            "baseline": _make_chainable_ee_image(["GREEN", "RED", "NIR"]),
        }
        splits = {"baseline": (MagicMock(name="train"), MagicMock(name="test"))}

        res = lulc.classify_multitemporal(
            predictor_stacks=stacks,
            training_splits=splits,
            mode="auto",
            reference_period="baseline",
            classification_type="soft",
            soft_mode="multiprobability",
            common_bands=["GREEN", "RED", "NIR"],
            period_order=["historical", "baseline"],
        )
        assert res["mode_used"] == "model_transfer"
        assert res["classification_type"] == "soft"
        assert res["maps"]["baseline"] is ref_map
        assert res["maps"]["historical"] is target_map
        assert res["class_ids"]["historical"] == [1, 2]
        assert "baseline" in res["model_evaluation"]
        assert "historical" not in res["model_evaluation"]
        lulc.classify_with_model.assert_called_once()

    @patch("luma_ge.accuracy.ensure_ee_initialized")
    def test_run_multitemporal_assessment_enforces_required_and_compares_accuracy(self, mock_init):
        ta = thematic_accuracy()
        ta.run_accuracy_assessment = MagicMock(
            return_value=(
                True,
                {
                    "overall_accuracy": 0.82,
                    "kappa": 0.76,
                    "f1_scores": [0.8, 0.84],
                    "n_total": 100,
                    "n_correct": 82,
                },
            )
        )
        maps = {"historical": MagicMock(), "baseline": MagicMock()}

        # Missing required baseline ground truth raises
        with pytest.raises(ValueError, match="baseline"):
            ta.run_multitemporal_assessment(
                maps_by_period=maps,
                validation_by_period={"historical": MagicMock(), "baseline": None},
                required_periods=["baseline"],
            )

        # Baseline only -> historical skipped cleanly
        acc_res = ta.run_multitemporal_assessment(
            maps_by_period=maps,
            validation_by_period={"baseline": MagicMock(), "historical": None},
            required_periods=["baseline"],
            period_order=["historical", "baseline"],
        )
        assert acc_res["assessed_periods"] == ["baseline"]
        assert acc_res["skipped_periods"] == ["historical"]

        # Both historical and baseline provided -> batch assessment runs on both
        acc_both = ta.run_multitemporal_assessment(
            maps_by_period=maps,
            validation_by_period={"historical": MagicMock(), "baseline": MagicMock()},
            required_periods=["baseline"],
            period_order=["historical", "baseline"],
        )
        assert acc_both["assessed_periods"] == ["historical", "baseline"]
        assert acc_both["skipped_periods"] == []
        assert set(acc_both["per_period"].keys()) == {"historical", "baseline"}

        delta = ta.compare_accuracy(
            raw_results={
                "overall_accuracy": 0.72,
                "kappa": 0.65,
                "f1_scores": [0.70, 0.74],
                "n_total": 100,
                "n_correct": 72,
            },
            corrected_results={
                "overall_accuracy": 0.84,
                "kappa": 0.79,
                "f1_scores": [0.82, 0.86],
                "n_total": 100,
                "n_correct": 84,
            },
        )
        assert delta["oa_delta"] == pytest.approx(0.12)
        assert delta["kappa_delta"] == pytest.approx(0.14)
        assert delta["per_class_f1_delta"] == pytest.approx([0.12, 0.12])

    @patch("luma_ge.classification.ensure_ee_initialized")
    def test_feature_extraction_single_vs_batch_routing(self, mock_init):
        fe = FeatureExtraction()
        hist_img = _make_chainable_ee_image(["GREEN", "RED", "NIR", "NDVI"])
        base_img = _make_chainable_ee_image(["GREEN", "RED", "NIR", "NDVI"])
        hist_img.sampleRegions.return_value = "hist_pixels"
        base_img.sampleRegions.return_value = "base_pixels"

        # 1 training dataset (reference_period only) -> single feature extraction
        res_single = fe.extract_multitemporal_samples(
            predictor_stacks={"historical": hist_img, "baseline": base_img},
            training_splits={"historical": None, "baseline": (MagicMock(), MagicMock())},
            class_property="ID_Kelas",
            reference_period="baseline",
            common_bands=["GREEN", "RED", "NIR"],
            period_order=["historical", "baseline"],
        )
        assert res_single["mode_used"] == "model_transfer"
        assert res_single["extraction_type"] == "single"
        assert res_single["extracted_periods"] == ["baseline"]
        assert res_single["skipped_periods"] == ["historical"]
        base_img.select.assert_called_with(["GREEN", "RED", "NIR"])
        assert base_img.sampleRegions.call_count == 2
        assert hist_img.sampleRegions.call_count == 0

        # 2 training datasets (all periods) -> batch feature extraction
        base_img.reset_mock()
        hist_img.reset_mock()
        res_batch = fe.extract_multitemporal_samples(
            predictor_stacks={"historical": hist_img, "baseline": base_img},
            training_splits={
                "historical": (MagicMock(), MagicMock()),
                "baseline": (MagicMock(), MagicMock()),
            },
            class_property="ID_Kelas",
            reference_period="baseline",
            period_order=["historical", "baseline"],
        )
        assert res_batch["mode_used"] == "batch"
        assert res_batch["extraction_type"] == "batch"
        assert res_batch["extracted_periods"] == ["historical", "baseline"]
        assert res_batch["skipped_periods"] == []
        assert hist_img.sampleRegions.call_count == 2
        assert base_img.sampleRegions.call_count == 2

    @patch("luma_ge.classification.ee")
    def test_soft_classification_multiprobability_with_scheme_class_names(self, mock_ee):
        class FakeFC:
            pass

        class FakeImg:
            pass

        mock_ee.FeatureCollection = FakeFC
        mock_ee.Image = FakeImg

        train_fc = FakeFC()
        train_fc.aggregate_array = MagicMock(
            return_value=MagicMock(distinct=MagicMock(return_value=MagicMock(sort=MagicMock(return_value=MagicMock(getInfo=MagicMock(return_value=[1, 2, 3]))))))
        )
        img = FakeImg()
        img.bandNames = MagicMock(return_value=MagicMock(size=MagicMock(return_value=MagicMock(sqrt=MagicMock(return_value=MagicMock(ceil=MagicMock(return_value=2)))))))
        classified_img = _make_chainable_ee_image()
        img.classify = MagicMock(return_value=classified_img)

        mock_rf = MagicMock()
        mock_rf.setOutputMode.return_value = mock_rf
        mock_rf.train.return_value = MagicMock()
        mock_ee.Classifier.smileRandomForest.return_value = mock_rf

        scheme_df = pd.DataFrame([
            {"ID": 1, "Land Cover Class": "Hutan Lahan Kering"},
            {"ID": 2, "Land Cover Class": "Hutan Mangrove"},
            {"ID": 3, "Land Cover Class": "Badan Air"},
        ])
        names, band_map = build_prob_band_names([1, 2, 3], class_names=scheme_df)
        assert names == ["prob_Hutan_Lahan_Kering", "prob_Hutan_Mangrove", "prob_Badan_Air"]
        assert band_map == {1: "prob_Hutan_Lahan_Kering", 2: "prob_Hutan_Mangrove", 3: "prob_Badan_Air"}

        lulc = Generate_LULC()
        _, model_info = lulc.soft_classification(
            training_data=train_fc,
            class_property="ID_Kelas",
            image=img,
            mode="multiprobability",
            return_model=True,
            class_names=scheme_df,
        )
        classified_img.arrayFlatten.assert_called_once_with(
            [["prob_Hutan_Lahan_Kering", "prob_Hutan_Mangrove", "prob_Badan_Air"]]
        )
        assert model_info["prob_band_names"] == ["prob_Hutan_Lahan_Kering", "prob_Hutan_Mangrove", "prob_Badan_Air"]
        assert model_info["prob_band_map"][1] == "prob_Hutan_Lahan_Kering"

    @patch("luma_ge.classification.ensure_ee_initialized")
    def test_feature_extraction_unsplit_and_partial_error(self, mock_init):
        fe = FeatureExtraction()
        fe.stratified_split = MagicMock(return_value=("train_pix", "test_pix"))
        p1 = _make_chainable_ee_image()
        p2 = _make_chainable_ee_image()
        p3 = _make_chainable_ee_image()

        # Unsplit fallback uses stratified_split on reference_period only
        res = fe.extract_multitemporal_samples(
            predictor_stacks={"t2010": p1, "t2015": p2, "t2025": p3},
            training_splits={"t2025": MagicMock()},
            class_property="class_id",
            reference_period="t2025",
            period_order=["t2010", "t2015", "t2025"],
        )
        assert res["extraction_type"] == "single"
        assert res["sampled_splits"]["t2025"] == ("train_pix", "test_pix")
        fe.stratified_split.assert_called_once()
        split_kw = fe.stratified_split.call_args.kwargs
        assert split_kw["image"] is p3
        assert split_kw["class_prop"] == "class_id"
        assert split_kw["tile_scale"] == 4
        # Unsplit batch across 2 periods invokes stratified_split twice
        fe.stratified_split.reset_mock()
        res_batch = fe.extract_multitemporal_samples(
            predictor_stacks={"t2010": p1, "t2025": p3},
            training_splits={"t2010": MagicMock(), "t2025": MagicMock()},
            class_property="class_id",
            reference_period="t2025",
            period_order=["t2010", "t2025"],
        )
        assert res_batch["extraction_type"] == "batch"
        assert fe.stratified_split.call_count == 2

        # Partial non-reference (2 of 3) raises clear ValueError
        with pytest.raises(ValueError, match="Cannot infer feature extraction mode"):
            fe.extract_multitemporal_samples(
                predictor_stacks={"t2010": p1, "t2015": p2, "t2025": p3},
                training_splits={"t2015": (MagicMock(), MagicMock()), "t2025": (MagicMock(), MagicMock())},
                reference_period="t2025",
                period_order=["t2010", "t2015", "t2025"],
            )

    def test_optional_log_loss_arrays_and_evaluate_model(self):
        import math

        ll = compute_log_loss_from_arrays(
            y_true=[1, 3],
            y_prob=[[0.8, 0.1, 0.1], [0.1, 0.2, 0.7]],
            class_ids=[1, 2, 3],
        )
        assert ll == pytest.approx(-(math.log(0.8) + math.log(0.7)) / 2.0)

        # Auto-normalizes percentage-scaled rows [0..100]
        ll_pct = compute_log_loss_from_arrays(
            y_true=[1, 3],
            y_prob=[[80, 10, 10], [10, 20, 70]],
            class_ids=[1, 2, 3],
        )
        assert ll_pct == pytest.approx(ll)

        lulc = Generate_LULC()
        test_fc = MagicMock()
        cm = MagicMock()
        cm.accuracy.return_value.getInfo.return_value = 0.85
        cm.kappa.return_value.getInfo.return_value = 0.80
        cm.producersAccuracy.return_value.getInfo.return_value = [[0.8], [0.9]]
        cm.consumersAccuracy.return_value.getInfo.return_value = [[0.85, 0.85]]
        cm.getInfo.return_value = [[8, 2], [1, 9]]
        test_fc.classify.return_value.errorMatrix.return_value = cm
        test_fc.classify.return_value.map.return_value = test_fc.classify.return_value
        test_fc.aggregate_array.return_value.distinct.return_value.sort.return_value.getInfo.return_value = [1, 2]

        with patch("luma_ge.classification.ee"):
            metrics_default = lulc.evaluate_model(
                trained_model=MagicMock(),
                test_data=test_fc,
                class_property="ID_Kelas",
                is_multiprobability=True,
                class_ids=[1, 2],
            )
            assert "log_loss" not in metrics_default

            lulc.compute_log_loss = MagicMock(return_value={"log_loss": 0.42, "n_samples": 20})
            metrics_with_ll = lulc.evaluate_model(
                trained_model=MagicMock(),
                test_data=test_fc,
                class_property="ID_Kelas",
                is_multiprobability=True,
                class_ids=[1, 2],
                include_log_loss=True,
            )
            assert metrics_with_ll["log_loss"] == pytest.approx(0.42)
            assert metrics_with_ll["overall_accuracy"] == pytest.approx(0.85)
