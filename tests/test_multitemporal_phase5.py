"""Unit & integration tests for Phase 5: PeriodConfig & MultitemporalAnalysis orchestrator."""

from unittest.mock import MagicMock, patch
import pandas as pd
import pytest

from luma_ge.multitemporal import (
    MultitemporalAnalysis,
    PeriodConfig,
    normalize_period_configs,
)


def _scheme_df():
    return pd.DataFrame(
        [
            {"ID": 1, "Land Cover Class": "Forest", "Color Palette": "#006400"},
            {"ID": 2, "Land Cover Class": "Cropland", "Color Palette": "#FFFF00"},
        ]
    )


def _mock_ee_image(bands=None):
    img = MagicMock()
    img.select.return_value = img
    img.addBands.return_value = img
    img.bandNames.return_value.getInfo.return_value = bands or [
        "BLUE", "GREEN", "RED", "NIR", "SWIR1", "SWIR2"
    ]
    return img


class TestPeriodConfigAndValidation:
    def test_chronological_normalization_sorts_unsorted_periods(self):
        periods = [
            PeriodConfig(label="baseline", start_date=2025, end_date=2025, sensor="HLS"),
            PeriodConfig(label="mid", start_date=2018, end_date=2018, sensor="L8_SR"),
            PeriodConfig(label="historical", start_date=2010, end_date=2010, sensor="L5_SR"),
        ]
        ordered, order = normalize_period_configs(periods)
        assert order == ["historical", "mid", "baseline"]
        assert [c.sensor for c in ordered] == ["L5_SR", "L8_SR", "HLS"]

    def test_requires_at_least_two_periods_and_valid_reference(self):
        with pytest.raises(ValueError, match="at least 2 periods"):
            normalize_period_configs([PeriodConfig("only", 2024, 2024, "L8_SR")])

        with pytest.raises(ValueError, match="not found in periods"):
            MultitemporalAnalysis(
                aoi=MagicMock(),
                periods={
                    "t2015": {"start_date": 2015, "end_date": 2015, "sensor": "L8_SR"},
                    "t2025": {"start_date": 2025, "end_date": 2025, "sensor": "HLS"},
                },
                reference_period="missing_ref",
            )


class TestMultitemporalAnalysisIntegration:
    def _run_mocked_pipeline(
        self,
        periods_cfg,
        reference_period,
        training_sources,
        validation_by_period,
    ):
        aoi = MagicMock()
        labels = [c.label for c in normalize_period_configs(periods_cfg)[0]]
        common_bands = ["BLUE", "GREEN", "RED", "NIR", "SWIR1", "SWIR2"]

        optical_mock = {
            "periods": {lbl: {"collection": MagicMock()} for lbl in labels},
            "period_order": labels,
        }
        composites_mock = {
            "composites": {lbl: _mock_ee_image(common_bands) for lbl in labels},
            "common_bands": common_bands,
            "period_order": labels,
            "histogram_matched": True,
        }

        provided = [lbl for lbl in labels if training_sources.get(lbl) is not None]
        expected_mode = "model_transfer" if len(provided) == 1 else "batch"
        expected_extraction_type = "single" if expected_mode == "model_transfer" else "batch"

        load_res = {
            "mode": expected_mode,
            "reference_period": reference_period,
            "period_order": labels,
            "provided_periods": provided,
            "training_by_period": {
                lbl: {"training_data": MagicMock(), "class_field": "ID_Kelas"}
                for lbl in provided
            },
            "validation_summary": {lbl: {"total_points": 50} for lbl in provided},
        }
        predictors_res = {
            "predictor_stacks": {lbl: _mock_ee_image(common_bands + ["elevation", "NDVI"]) for lbl in labels},
            "feature_bands": {lbl: common_bands + ["elevation", "NDVI"] for lbl in labels},
            "common_feature_bands": common_bands + ["elevation", "NDVI"],
            "compatible_indices": ["NDVI"],
            "skipped_indices": [],
            "terrain_bands": ["elevation"],
            "period_order": labels,
        }

        sampled_splits = {lbl: (f"{lbl}_train_px", f"{lbl}_test_px") for lbl in provided}
        fe_res = {
            "mode_used": expected_mode,
            "extraction_type": expected_extraction_type,
            "sampled_splits": sampled_splits,
            "extracted_periods": provided,
            "skipped_periods": [lbl for lbl in labels if lbl not in provided],
        }

        clf_res = {
            "mode_used": expected_mode,
            "classification_type": "soft",
            "soft_mode": "multiprobability",
            "reference_period": reference_period,
            "period_order": labels,
            "maps": {lbl: _mock_ee_image(["prob_1", "prob_2", "classification", "confidence"]) for lbl in labels},
            "models": {lbl: MagicMock() for lbl in labels},
            "feature_importance": {lbl: pd.DataFrame() for lbl in provided},
            "model_evaluation": {lbl: {"overall_accuracy": 0.85, "kappa": 0.80} for lbl in provided},
            "class_ids": {lbl: [1, 2] for lbl in labels},
        }

        acc_res = {
            "per_period": {
                lbl: (True, {"overall_accuracy": 0.84, "kappa": 0.79})
                for lbl in labels
                if validation_by_period.get(lbl) is not None
            },
            "assessed_periods": [lbl for lbl in labels if validation_by_period.get(lbl) is not None],
            "skipped_periods": [lbl for lbl in labels if validation_by_period.get(lbl) is None],
        }

        with patch("luma_ge.multitemporal.ensure_ee_initialized"), \
             patch("luma_ge.data_acquisition.Reflectance_Data") as mock_rd_cls, \
             patch("luma_ge.data_acquisition.final_Image") as mock_fi_cls, \
             patch("luma_ge.sample_data.SyncTrainData.LoadMultitemporalTrainData", return_value=load_res), \
             patch("luma_ge.sample_data_quality.sample_quality.run_multitemporal_separability", return_value={"analyzed_periods": provided}) as mock_sep, \
             patch("luma_ge.multitemporal.compute_multitemporal_predictors", return_value=predictors_res) as mock_pred, \
             patch("luma_ge.classification.FeatureExtraction") as mock_fe_cls, \
             patch("luma_ge.classification.Generate_LULC") as mock_lulc_cls, \
             patch("luma_ge.accuracy.thematic_accuracy") as mock_ta_cls:

            mock_rd_cls.return_value.get_multitemporal_optical_data.return_value = optical_mock
            mock_fi_cls.return_value.get_multitemporal_composites.return_value = composites_mock
            mock_fe_cls.return_value.extract_multitemporal_samples.return_value = fe_res
            mock_lulc_cls.return_value.classify_multitemporal.return_value = clf_res
            mock_ta_cls.return_value.run_multitemporal_assessment.return_value = acc_res

            mt = MultitemporalAnalysis(
                aoi=aoi,
                periods=periods_cfg,
                reference_period=reference_period,
                apply_brdf=True,
                histogram_match=True,
            )
            out = mt.run_pipeline(
                training_sources=training_sources,
                landcover_df=_scheme_df(),
                predictor_config={"individual_predictors": {"elevation": True, "spectral_indices": ["NDVI"]}},
                validation_by_period=validation_by_period,
                class_field="ID_Kelas",
            )
            return out, mock_fe_cls.return_value, mock_lulc_cls.return_value, mock_sep, mock_pred, mock_ta_cls.return_value

    def test_two_period_model_transfer_cross_sensor_end_to_end(self):
        periods = {
            "baseline": {"start_date": 2025, "end_date": 2025, "sensor": "L8_SR"},
            "historical": {"start_date": 2010, "end_date": 2010, "sensor": "L5_SR"},
        }
        out, fe_inst, lulc_inst, _, _, ta_inst = self._run_mocked_pipeline(
            periods_cfg=periods,
            reference_period="baseline",
            training_sources={"historical": None, "baseline": MagicMock()},
            validation_by_period={"baseline": MagicMock(), "historical": None},
        )
        assert out["period_order"] == ["historical", "baseline"]
        assert out["mode"] == "model_transfer"
        assert out["feature_extraction"]["extraction_type"] == "single"
        assert out["feature_extraction"]["extracted_periods"] == ["baseline"]
        assert set(out["classification"]["maps"].keys()) == {"historical", "baseline"}
        assert fe_inst.extract_multitemporal_samples.call_args.kwargs["mode"] == "model_transfer"
        assert lulc_inst.classify_multitemporal.call_args.kwargs["mode"] == "model_transfer"
        assert fe_inst.extract_multitemporal_samples.call_args.kwargs["training_splits"] is not None
        assert lulc_inst.classify_multitemporal.call_args.kwargs["class_names"] is not None
        assert ta_inst.run_multitemporal_assessment.call_args.kwargs["required_periods"] == ["baseline"]

    def test_two_period_batch_mode_runs_batch_feature_extraction(self):
        periods = {
            "historical": {"start_date": 2010, "end_date": 2010, "sensor": "L5_SR"},
            "baseline": {"start_date": 2025, "end_date": 2025, "sensor": "L8_SR"},
        }
        out, fe_inst, lulc_inst, _, _, _ = self._run_mocked_pipeline(
            periods_cfg=periods,
            reference_period="baseline",
            training_sources={"historical": MagicMock(), "baseline": MagicMock()},
            validation_by_period={"historical": MagicMock(), "baseline": MagicMock()},
        )
        assert out["mode"] == "batch"
        assert out["feature_extraction"]["extraction_type"] == "batch"
        assert out["feature_extraction"]["extracted_periods"] == ["historical", "baseline"]
        assert fe_inst.extract_multitemporal_samples.call_args.kwargs["mode"] == "batch"
        assert lulc_inst.classify_multitemporal.call_args.kwargs["mode"] == "batch"

    def test_four_period_2010_to_2025_model_transfer(self):
        periods = [
            PeriodConfig("t2025", 2025, 2025, "HLS"),
            PeriodConfig("t2010", 2010, 2010, "L5_SR"),
            PeriodConfig("t2020", 2020, 2020, "S2_SR"),
            PeriodConfig("t2015", 2015, 2015, "L8_SR"),
        ]
        out, _, _, _, _, _ = self._run_mocked_pipeline(
            periods_cfg=periods,
            reference_period="t2025",
            training_sources={"t2010": None, "t2015": None, "t2020": None, "t2025": MagicMock()},
            validation_by_period={"t2025": MagicMock()},
        )
        assert out["period_order"] == ["t2010", "t2015", "t2020", "t2025"]
        assert out["feature_extraction"]["extraction_type"] == "single"
        assert out["feature_extraction"]["skipped_periods"] == ["t2010", "t2015", "t2020"]
        assert list(out["classification"]["maps"].keys()) == ["t2010", "t2015", "t2020", "t2025"]
