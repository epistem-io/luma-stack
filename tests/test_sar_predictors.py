"""Unit tests for SAR predictors (Phase 1: Sentinel-1 & Phase 2: ALOS PALSAR). Earth Engine calls are mocked."""
from unittest.mock import MagicMock, patch
import pytest

from luma_ge import sar as sar_module
from luma_ge import predictor as predictor_module


@pytest.fixture
def sar():
    with patch.object(sar_module, "ensure_ee_initialized"):
        yield sar_module.get_sar_data()


@pytest.fixture
def pc():
    with patch.object(predictor_module, "ensure_ee_initialized"), \
            patch.object(sar_module, "ensure_ee_initialized"):
        yield predictor_module.PredictorCalculation()


# ---------------------------------------------------------------- get_sentinel1
class TestGetSentinel1:
    def test_pre_sentinel1_range_returns_none(self, sar):
        assert sar.get_sentinel1(MagicMock(), "2010-01-01", "2010-12-31") is None

    def test_invalid_arguments(self, sar):
        with pytest.raises(ValueError):
            sar.get_sentinel1(MagicMock(), 2022, 2022, orbit_pass="UP")
        with pytest.raises(ValueError):
            sar.get_sentinel1(MagicMock(), 2022, 2022, reducer="max")

    def test_empty_collection_returns_none(self, sar):
        with patch.object(sar_module, "ee") as ee_mock:
            col = ee_mock.ImageCollection.return_value
            chain = col.filterBounds.return_value.filterDate.return_value
            for _ in range(3):
                chain = chain.filter.return_value
            chain.size.return_value.getInfo.return_value = 0
            assert sar.get_sentinel1(MagicMock(), 2022, 2022) is None

    def test_filters_applied_and_bands_named(self, sar):
        with patch.object(sar_module, "ee") as ee_mock:
            col = ee_mock.ImageCollection.return_value
            chain = col.filterBounds.return_value.filterDate.return_value
            for _ in range(3):
                chain = chain.filter.return_value
            chain.size.return_value.getInfo.return_value = 12

            with patch.object(sar, "apply_speckle_filter") as sf:
                image = MagicMock()
                sf.return_value = image
                result = sar.get_sentinel1(MagicMock(), "2022-01-01", "2022-06-30")

            ee_mock.ImageCollection.assert_called_with("COPERNICUS/S1_GRD")
            col.filterBounds.return_value.filterDate.assert_called_with("2022-01-01", "2022-06-30")
            image.select.return_value.rename.assert_called_with(["S1_VV", "S1_VH"])
            assert result is not None

    def test_ratio_is_optional(self, sar):
        with patch.object(sar_module, "ee") as ee_mock:
            col = ee_mock.ImageCollection.return_value
            chain = col.filterBounds.return_value.filterDate.return_value
            for _ in range(3):
                chain = chain.filter.return_value
            chain.size.return_value.getInfo.return_value = 5
            with patch.object(sar, "apply_speckle_filter") as sf:
                renamed = sf.return_value.select.return_value.rename.return_value
                sar.get_sentinel1(MagicMock(), 2022, 2022, include_ratio=False)
                renamed.addBands.assert_not_called()
                sar.get_sentinel1(MagicMock(), 2022, 2022, include_ratio=True)
                renamed.addBands.assert_called_once()


class TestSpeckleFilter:
    def test_none_returns_input(self, sar):
        img = MagicMock()
        assert sar.apply_speckle_filter(img, "none") is img

    def test_unknown_method(self, sar):
        with pytest.raises(ValueError):
            sar.apply_speckle_filter(MagicMock(), "bogus")


# ---------------------------------------------------------------- get_palsar (Phase 2)
class TestGetPalsar:
    def test_invalid_reducer(self, sar):
        with pytest.raises(ValueError):
            sar.get_palsar(MagicMock(), 2020, 2020, reducer="sum")

    def test_year_resolution_direct(self, sar):
        years = sar._resolve_palsar_years(2017, 2019)
        assert years == [2017, 2018, 2019]

    def test_year_resolution_gap_fallback(self, sar):
        # 2012 is inside the 2011-2014 gap; nearest available year is 2010
        years = sar._resolve_palsar_years(2012, 2012, allow_nearest_year_fallback=True)
        assert years == [2010]

    def test_year_resolution_gap_no_fallback(self, sar):
        years = sar._resolve_palsar_years(2012, 2012, allow_nearest_year_fallback=False)
        assert years == []

    def test_year_resolution_post_2024_fallback(self, sar):
        years = sar._resolve_palsar_years(2025, 2026, allow_nearest_year_fallback=True)
        assert years == [2024]

    def test_palsar_filters_applied_and_bands_named(self, sar):
        with patch.object(sar_module, "ee") as ee_mock:
            col = ee_mock.ImageCollection.return_value
            chain = col.filterBounds.return_value.filter.return_value
            chain.size.return_value.getInfo.return_value = 1

            with patch.object(sar, "apply_speckle_filter") as sf:
                image = MagicMock()
                sf.return_value = image
                result = sar.get_palsar(MagicMock(), "2020-01-01", "2020-12-31")

            ee_mock.ImageCollection.assert_called_with("JAXA/ALOS/PALSAR/YEARLY/SAR_EPOCH")
            image.select.return_value.rename.assert_called_with(["PALSAR_HH", "PALSAR_HV"])
            assert result is not None

    def test_palsar_empty_collection_returns_none(self, sar):
        with patch.object(sar_module, "ee") as ee_mock:
            col = ee_mock.ImageCollection.return_value
            chain = col.filterBounds.return_value.filter.return_value
            chain.size.return_value.getInfo.return_value = 0
            # Both EPOCH and v1 fallback report 0
            assert sar.get_palsar(MagicMock(), 2020, 2020) is None


# ------------------------------------------------------- PredictorCalculation
class TestSarSelection:
    def test_sensor_flag_selects_all(self, pc):
        s1, palsar, rest = pc._resolve_sar_selection({"sentinel1": True, "palsar": True}, ["NDVI"])
        assert s1 == ["S1_VV", "S1_VH", "S1_VV_VH"]
        assert palsar == ["PALSAR_HH", "PALSAR_HV", "PALSAR_HH_HV"]
        assert rest == ["NDVI"]

    def test_luma_be_style_band_names_removed_from_spectral(self, pc):
        cfg = {"individual_predictors": {"spectral_indices": ["NDVI", "S1_VH", "PALSAR_HH"]}}
        s1, palsar, rest = pc._resolve_sar_selection(cfg, ["NDVI", "S1_VH", "PALSAR_HH"])
        assert s1 == ["S1_VH"]
        assert palsar == ["PALSAR_HH"]
        assert rest == ["NDVI"]

    def test_individual_predictors_list(self, pc):
        cfg = {"individual_predictors": {"sentinel1": ["s1_vv_vh"], "palsar": ["palsar_hh_hv"]}}
        s1, palsar, _ = pc._resolve_sar_selection(cfg, [])
        assert s1 == ["S1_VV_VH"]
        assert palsar == ["PALSAR_HH_HV"]

    def test_nothing_selected(self, pc):
        assert pc._resolve_sar_selection({}, ["NDVI"]) == ([], [], ["NDVI"])


class TestDateInheritance:
    def test_config_has_priority(self, pc):
        cfg = {"start_date": "2020-01-01", "end_date": "2020-12-31"}
        assert pc._resolve_date_range(cfg, collection=MagicMock()) == ("2020-01-01", "2020-12-31")

    def test_collection_properties(self, pc):
        with patch.object(predictor_module, "ee") as ee_mock:
            ee_mock.Dictionary.return_value.getInfo.return_value = {
                "s": "2021-01-01", "e": "2021-12-31"}
            assert pc._resolve_date_range({}, collection=MagicMock()) == ("2021-01-01", "2021-12-31")

    def test_unresolved(self, pc):
        assert pc._resolve_date_range({}) == (None, None)


class TestComputePredictorsStacking:
    def _run(self, pc, s1_return, palsar_return, cfg):
        composite = MagicMock()
        composite.bandNames.return_value.getInfo.return_value = ["B2", "B3"]
        stacked = composite.addBands.return_value
        stacked.bandNames.return_value.size.return_value.getInfo.return_value = 8
        with patch.object(pc.sar_calc, "get_sentinel1", return_value=s1_return) as gs1, \
             patch.object(pc.sar_calc, "get_palsar", return_value=palsar_return) as gpalsar:
            result = pc.compute_predictors(
                composite, MagicMock(), cfg, collection=None)
        return result, gs1, gpalsar

    def test_s1_and_palsar_stacked(self, pc):
        s1 = MagicMock()
        palsar = MagicMock()
        cfg = {
            "sentinel1": True,
            "palsar": True,
            "start_date": "2020-01-01",
            "end_date": "2020-12-31"
        }
        result, gs1, gpalsar = self._run(pc, s1, palsar, cfg)
        assert result["success"]
        assert "sentinel1" in result["predictors"]
        assert "palsar" in result["predictors"]
        s1.select.assert_called_with(["S1_VV", "S1_VH", "S1_VV_VH"])
        palsar.select.assert_called_with(["PALSAR_HH", "PALSAR_HV", "PALSAR_HH_HV"])

    def test_palsar_unavailable_adds_warning_only(self, pc):
        cfg = {"palsar": True, "start_date": "2000-01-01", "end_date": "2000-12-31"}
        result, _, _ = self._run(pc, None, None, cfg)
        assert result["success"]
        assert "palsar" not in result["predictors"]
        assert any("PALSAR" in w for w in result["warnings"])
