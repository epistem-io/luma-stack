"""Unit tests for Phase 2: Multitemporal Training Validation & Separability (Modules 3 & 4)."""

from unittest.mock import MagicMock, patch
import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point

from luma_ge.sample_data import SyncTrainData, SplitTrainData
from luma_ge.sample_data_quality import sample_quality


def _make_sample_gdf(class_ids, col_name="class_id"):
    rows = []
    for i, cid in enumerate(class_ids):
        rows.append(
            {
                col_name: cid,
                "class_name": f"Class_{cid}",
                "geometry": Point(106.0 + i * 0.001, -6.5),
            }
        )
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")


def _make_scheme_df():
    return pd.DataFrame(
        [
            {"ID": 1, "Class Name": "Forest", "Color Code": "#006400"},
            {"ID": 2, "Class Name": "Cropland", "Color Code": "#FFFF00"},
            {"ID": 3, "Class Name": "Settlement", "Color Code": "#FF0000"},
        ]
    )


class TestMultitemporalTrainingData:
    def test_single_dataset_on_reference_routes_to_model_transfer(self):
        scheme_df = _make_scheme_df()
        baseline_gdf = _make_sample_gdf([1] * 25 + [2] * 25 + [99] * 2)  # 99 is invalid class

        res = SyncTrainData.LoadMultitemporalTrainData(
            training_sources={"historical": None, "baseline": baseline_gdf},
            landcover_df=scheme_df,
            period_order=["historical", "baseline"],
            reference_period="baseline",
            use_class_ids=True,
            min_samples=20,
        )

        assert res["mode"] == "model_transfer"
        assert res["provided_periods"] == ["baseline"]
        assert "historical" not in res["training_by_period"]
        # Invalid class 99 should be filtered out
        assert res["validation_summary"]["baseline"]["invalid_classes"] == [99]
        assert res["validation_summary"]["baseline"]["points_after_class_filter"] == 50
        assert res["validation_summary"]["baseline"]["insufficient_samples"] == []

    def test_all_periods_provided_routes_to_batch(self):
        scheme_df = _make_scheme_df()
        hist_gdf = _make_sample_gdf([1] * 20 + [2] * 10)  # class 2 < 20 -> insufficient
        base_gdf = _make_sample_gdf([1] * 22 + [3] * 22)

        res = SyncTrainData.LoadMultitemporalTrainData(
            training_sources={"historical": hist_gdf, "baseline": base_gdf},
            landcover_df=scheme_df,
            period_order=["historical", "baseline"],
            reference_period="baseline",
            use_class_ids=True,
            min_samples=20,
        )

        assert res["mode"] == "batch"
        assert res["provided_periods"] == ["historical", "baseline"]
        assert res["validation_summary"]["historical"]["insufficient_samples"] == [2]
        assert res["validation_summary"]["baseline"]["insufficient_samples"] == []

    def test_single_dataset_on_non_reference_raises(self):
        scheme_df = _make_scheme_df()
        hist_gdf = _make_sample_gdf([1] * 25 + [2] * 25)

        with pytest.raises(ValueError, match="reference_period 'baseline'"):
            SyncTrainData.LoadMultitemporalTrainData(
                training_sources={"historical": hist_gdf, "baseline": None},
                landcover_df=scheme_df,
                period_order=["historical", "baseline"],
                reference_period="baseline",
            )

    def test_partial_periods_for_n_greater_than_2_raises(self):
        scheme_df = _make_scheme_df()
        gdf = _make_sample_gdf([1] * 25 + [2] * 25)

        with pytest.raises(ValueError, match="missing periods"):
            SyncTrainData.LoadMultitemporalTrainData(
                training_sources={"t2010": None, "t2015": gdf, "t2025": gdf},
                landcover_df=scheme_df,
                period_order=["t2010", "t2015", "t2025"],
                reference_period="t2025",
            )

    def test_module3_does_not_perform_multitemporal_splitting(self):
        assert not hasattr(SplitTrainData, "SplitMultitemporalProcess")

    def test_ee_feature_collection_validates_class_field(self):
        class FakeFC:
            def __init__(self, props):
                self._props = props

            def filterBounds(self, geom):
                return self

            def first(self):
                return MagicMock(propertyNames=MagicMock(return_value=MagicMock(getInfo=MagicMock(return_value=self._props))))

            def size(self):
                return MagicMock(getInfo=MagicMock(return_value=100))

            def filter(self, *args, **kwargs):
                return self

            def flatten(self):
                return self

            def aggregate_array(self, prop):
                def _assert_unary_map(fn):
                    assert fn.__code__.co_argcount == 1
                    return []

                return MagicMock(distinct=MagicMock(return_value=MagicMock(map=_assert_unary_map)))

            def aggregate_histogram(self, prop):
                return MagicMock(getInfo=MagicMock(return_value={"1": 50, "2": 50}))

        scheme_df = _make_scheme_df()
        fc = FakeFC(["ID_Kelas", "KelasTupla"])

        with patch("luma_ge.sample_data.ee.FeatureCollection", FakeFC), patch(
            "luma_ge.sample_data.ee.Filter"
        ):
            # Missing class_field raises helpful ValueError instead of silent 0 features
            with pytest.raises(ValueError, match="Available properties"):
                SyncTrainData.LoadMultitemporalTrainData(
                    training_sources={"historical": None, "baseline": fc},
                    landcover_df=scheme_df,
                    period_order=["historical", "baseline"],
                    reference_period="baseline",
                    class_field="ID",
                )

            # Valid class_field + split works for ee.FeatureCollection
            with patch.object(SyncTrainData, "ValidClass", side_effect=lambda d, **kw: d):
                loaded = SyncTrainData.LoadMultitemporalTrainData(
                    training_sources={"historical": None, "baseline": fc},
                    landcover_df=scheme_df,
                    period_order=["historical", "baseline"],
                    reference_period="baseline",
                    class_field="ID_Kelas",
                )
            assert loaded["mode"] == "model_transfer"
            assert loaded["training_by_period"]["baseline"]["class_field"] == "ID_Kelas"
            assert isinstance(loaded["training_by_period"]["baseline"]["training_data"], FakeFC)


class TestMultitemporalSeparability:
    def test_model_transfer_skips_target_period_without_training_data(self):
        mock_instance = MagicMock()
        mock_spectral = pd.DataFrame(
            {"class_id": [1, 1, 2, 2], "BLUE": [0.1, 0.12, 0.3, 0.32], "NIR": [0.5, 0.52, 0.2, 0.22]}
        )
        mock_sep = pd.DataFrame([{"Class1_ID": "1", "Class2_ID": "2", "TD_Distance": 1.95}])
        mock_sum = pd.DataFrame([{"Total Pairs": 1, "Good Separability Pairs": 1}])
        mock_instance.extract_spectral_values.return_value = mock_spectral
        mock_instance.get_separability_df.return_value = mock_sep
        mock_instance.sum_separability.return_value = mock_sum

        composites_payload = {
            "composites": {"historical": MagicMock(name="img_hist"), "baseline": MagicMock(name="img_base")},
            "period_order": ["historical", "baseline"],
        }
        training_by_period = {
            "baseline": {"training_data": MagicMock(spec=["size"]), "class_field": "class_id"}
        }
        run_fn = sample_quality.run_multitemporal_separability
        with patch("luma_ge.sample_data_quality.ee.FeatureCollection", MagicMock), patch(
            "luma_ge.sample_data_quality.sample_quality", return_value=mock_instance
        ) as mock_sq_cls:
            res = run_fn(
                composites_by_period=composites_payload,
                training_by_period=training_by_period,
                scale=30,
                max_pixels_per_class=5000,
            )

        assert res["analyzed_periods"] == ["baseline"]
        assert res["skipped_periods"] == ["historical"]
        assert mock_sq_cls.call_count == 1
        assert "baseline" in res["per_period"]

    def test_batch_mode_analyzes_all_periods(self):
        mock_instance = MagicMock()
        mock_instance.extract_spectral_values.return_value = pd.DataFrame()
        mock_instance.get_separability_df.return_value = pd.DataFrame()
        mock_instance.sum_separability.return_value = pd.DataFrame()

        composites = {"historical": MagicMock(name="img_hist"), "baseline": MagicMock(name="img_base")}
        training_by_period = {
            "historical": MagicMock(name="fc_hist"),
            "baseline": MagicMock(name="fc_base"),
        }
        run_fn = sample_quality.run_multitemporal_separability
        with patch("luma_ge.sample_data_quality.ee.FeatureCollection", MagicMock), patch(
            "luma_ge.sample_data_quality.sample_quality", return_value=mock_instance
        ) as mock_sq_cls:
            res = run_fn(
                composites_by_period=composites,
                training_by_period=training_by_period,
                period_order=["historical", "baseline"],
            )

        assert res["analyzed_periods"] == ["historical", "baseline"]
        assert res["skipped_periods"] == []
        assert mock_sq_cls.call_count == 2
