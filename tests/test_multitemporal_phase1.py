"""Unit tests for Phase 1: Imagery Correction (Topographic & BRDF) and Multitemporal Data Acquisition."""

from unittest.mock import MagicMock, patch
import pytest

from luma_ge.imagery_correction import Topographic_Correction, BRDF_Correction
from luma_ge.data_acquisition import Reflectance_Data, final_Image


def _make_chainable_ee_image(name="img"):
    """Create a chainable MagicMock behaving like an ee.Image for unit tests."""
    img = MagicMock(name=name)
    for method in (
        "select", "rename", "addBands", "multiply", "divide", "add", "subtract",
        "cos", "sin", "acos", "asin", "atan", "clamp", "gt", "gte", "lt", "lte",
        "eq", "neq", "And", "Or", "mask", "unmask", "updateMask", "clip",
        "copyProperties", "set", "toFloat", "toInt", "round", "byte",
        "interpolate", "bitwiseAnd", "rightShift",
    ):
        getattr(img, method).return_value = img
    img.date.return_value = MagicMock(name="ee.Date")
    img.geometry.return_value.bounds.return_value.bounds.return_value.coordinates.return_value.get.return_value = MagicMock()
    img.geometry.return_value.buffer.return_value = MagicMock()
    img.propertyNames.return_value = MagicMock()
    img.bandNames.return_value.removeAll.return_value = MagicMock()
    img.bandNames.return_value.getInfo.return_value = ["BLUE", "GREEN", "RED", "NIR", "SWIR1", "SWIR2"]
    return img


class TestBRDFCorrection:
    @patch("luma_ge.imagery_correction.ensure_ee_initialized")
    def test_band_constants_and_coefficients(self, mock_init):
        brdf = BRDF_Correction()
        mock_init.assert_called_once()
        assert brdf.DEFAULT_OPTICAL_BANDS == ["BLUE", "GREEN", "RED", "NIR", "SWIR1", "SWIR2"]
        assert len(brdf.S2_OPTICAL_BANDS) == 10
        assert set(brdf.S2_OPTICAL_BANDS) == set(brdf.BAND_COEFFICIENTS.keys())
        assert brdf.BAND_COEFFICIENTS["NIR"]["f_iso"] == pytest.approx(0.3093)

    @patch("luma_ge.imagery_correction.ensure_ee_initialized")
    def test_invalid_band_list_raises(self, mock_init):
        brdf = BRDF_Correction()
        img = _make_chainable_ee_image()
        with pytest.raises(ValueError, match="No supported BRDF bands"):
            brdf.correct_image(img, band_list=["UNKNOWN_BAND"])

    @patch("luma_ge.imagery_correction.ee")
    @patch("luma_ge.imagery_correction.ensure_ee_initialized")
    def test_correct_image_uses_dynamic_date_and_preserves_properties(self, mock_init, mock_ee):
        mock_ee.Image.side_effect = lambda x=None: x if isinstance(x, MagicMock) else _make_chainable_ee_image()
        mock_ee.Image.pixelLonLat.return_value = _make_chainable_ee_image("lonlat")
        mock_ee.Image.cat.return_value = _make_chainable_ee_image("cat")

        brdf = BRDF_Correction()
        img = _make_chainable_ee_image("scene")
        result = brdf.correct_image(img, band_list=["BLUE", "NIR"], verbose=False)

        # Must query image.date() dynamically rather than a hardcoded date
        img.date.assert_called_once()
        img.addBands.assert_called()
        img.copyProperties.assert_called()
        assert result is not None

    @patch("luma_ge.imagery_correction.ensure_ee_initialized")
    def test_correct_collection_maps_lazily(self, mock_init):
        brdf = BRDF_Correction()
        col = MagicMock()
        col.map.return_value = "mapped_collection"
        out = brdf.correct_collection(col, verbose=False)
        col.map.assert_called_once()
        assert out == "mapped_collection"


class TestTopographicCorrection:
    @patch("luma_ge.imagery_correction.ensure_ee_initialized")
    def test_init_and_defaults(self, mock_init):
        tc = Topographic_Correction()
        mock_init.assert_called_once()
        assert tc.DEFAULT_DEM_ASSET == "NASA/NASADEM_HGT/001"
        assert tc.DEFAULT_OPTICAL_BANDS == ["BLUE", "GREEN", "RED", "NIR", "SWIR1", "SWIR2"]

    @patch("luma_ge.imagery_correction.ee")
    @patch("luma_ge.imagery_correction.ensure_ee_initialized")
    def test_correct_collection_maps_over_collection(self, mock_init, mock_ee):
        mock_ee.Image.return_value = _make_chainable_ee_image("dem")
        tc = Topographic_Correction()
        col = MagicMock()
        col.map.return_value = "topo_corrected_collection"
        out = tc.correct_collection(col, scale=30, verbose=False)
        col.map.assert_called_once()
        assert out == "topo_corrected_collection"


class TestHLSAndRoy2016Harmonization:
    @patch("luma_ge.data_acquisition.ensure_ee_initialized")
    def test_hls_datasets_registry(self, mock_init):
        rd = Reflectance_Data()
        assert "HLS" in rd.HLS_DATASETS
        assert "HLS_L30" in rd.HLS_DATASETS
        assert "HLS_S30" in rd.HLS_DATASETS
        assert rd.HLS_DATASETS["HLS"]["start_date"] == "2013-04-01"
        assert rd.HLS_DATASETS["HLS_S30"]["start_date"] == "2015-11-28"

    @patch("luma_ge.data_acquisition.ensure_ee_initialized")
    def test_hls_date_validation_rejects_pre_2013(self, mock_init):
        rd = Reflectance_Data()
        aoi = MagicMock()
        with pytest.raises(ValueError, match="2013-04-01"):
            rd.get_hls_optical_data(aoi, 2010, 2010, dataset="HLS")

        with pytest.raises(ValueError, match="2015-11-28"):
            rd.get_hls_optical_data(aoi, 2014, 2014, dataset="HLS_S30")

    @patch("luma_ge.data_acquisition.ee")
    @patch("luma_ge.data_acquisition.ensure_ee_initialized")
    def test_rename_hls_bands_l30_and_s30(self, mock_init, mock_ee):
        mock_ee.Image.side_effect = lambda x: x
        rd = Reflectance_Data()
        img = _make_chainable_ee_image("hls")

        rd.rename_hls_bands(img, hls_type="HLS_L30")
        img.select.assert_called_with(
            ["B2", "B3", "B4", "B5", "B6", "B7"],
            ["BLUE", "GREEN", "RED", "NIR", "SWIR1", "SWIR2"],
        )

        rd.rename_hls_bands(img, hls_type="HLS_S30")
        img.select.assert_called_with(
            ["B2", "B3", "B4", "B8A", "B11", "B12"],
            ["BLUE", "GREEN", "RED", "NIR", "SWIR1", "SWIR2"],
        )

    @patch("luma_ge.data_acquisition.ee")
    @patch("luma_ge.data_acquisition.ensure_ee_initialized")
    def test_roy2016_harmonization_applies_only_to_l4_l5_l7(self, mock_init, mock_ee):
        mock_ee.Image.side_effect = lambda x=None: x if isinstance(x, MagicMock) else _make_chainable_ee_image()
        mock_ee.Image.constant.return_value = _make_chainable_ee_image("const")

        rd = Reflectance_Data()
        img_l5 = _make_chainable_ee_image("l5")
        out_l5 = rd.apply_brdf_harmonization(img_l5, "L5")
        img_l5.select.assert_called_with(["BLUE", "GREEN", "RED", "NIR", "SWIR1", "SWIR2"])
        assert out_l5 is not None

        img_l8 = _make_chainable_ee_image("l8")
        out_l8 = rd.apply_brdf_harmonization(img_l8, "L8")
        img_l8.select.assert_not_called()
        assert out_l8 is img_l8

    @patch("luma_ge.data_acquisition.ensure_ee_initialized")
    def test_single_period_landsat8_preserves_aerosol_contract(self, mock_init):
        rd = Reflectance_Data()
        img = _make_chainable_ee_image("l8_single")
        rd.rename_landsat_bands(img, "L8")
        img.select.assert_called_with(
            ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"],
            ["AEROSOL", "BLUE", "GREEN", "RED", "NIR", "SWIR1", "SWIR2"],
        )


class TestMultitemporalAcquisitionAndComposites:
    @patch("luma_ge.data_acquisition.ensure_ee_initialized")
    def test_get_multitemporal_optical_data_sorts_chronologically_and_routes(self, mock_init):
        rd = Reflectance_Data()
        col_opt = MagicMock(name="col_opt")
        col_hls = MagicMock(name="col_hls")
        rd.get_optical_data = MagicMock(return_value=(col_opt, {"total_images": 5}))
        rd.get_hls_optical_data = MagicMock(return_value=(col_hls, {"total_images": 8}))

        aoi = MagicMock()
        periods = {
            "baseline": {"start_date": 2024, "end_date": 2024, "sensor": "HLS", "cloud_cover": 20},
            "historical": {"start_date": 1995, "end_date": 1995, "sensor": "L5_SR", "cloud_cover": 30},
        }
        res = rd.get_multitemporal_optical_data(
            aoi=aoi,
            periods=periods,
            harmonize_landsat=True,
            verbose=False,
            compute_detailed_stats=False,
        )
        # Historical (1995) must be ordered before Baseline (2024)
        assert res["period_order"] == ["historical", "baseline"]
        assert set(res["periods"].keys()) == {"historical", "baseline"}
        rd.get_optical_data.assert_called_once()
        rd.get_hls_optical_data.assert_called_once()
        # L5 collection should have Roy 2016 mapped with unary callback (co_argcount == 1)
        col_opt.map.assert_called_once()
        mapped_fn = col_opt.map.call_args.args[0]
        assert mapped_fn.__code__.co_argcount == 1

    @patch("luma_ge.data_acquisition.ee.Date")
    @patch("luma_ge.data_acquisition.ee.Reducer")
    @patch("luma_ge.data_acquisition.ensure_ee_initialized")
    def test_get_temporal_composite_preserves_dict_contract_and_supports_return_image_only(
        self, mock_init, mock_reducer, mock_date
    ):
        mock_reducer.median.return_value = "median"
        mock_date.return_value.format.return_value = "2024-01-01"

        fi = final_Image()
        col = MagicMock()
        img = _make_chainable_ee_image("comp")
        col.reduce.return_value = img
        aoi = MagicMock()

        res_dict = fi.get_temporal_composite(col, aoi, verbose=False)
        assert isinstance(res_dict, dict)
        assert set(res_dict.keys()) == {"image", "metadata", "coverage"}

        res_img = fi.get_temporal_composite(col, aoi, verbose=False, return_image_only=True)
        assert res_img is img

    @patch("luma_ge.data_acquisition.ee.Image")
    @patch("luma_ge.data_acquisition.ensure_ee_initialized")
    def test_get_multitemporal_composites_drops_aerosol_and_intersects_bands(
        self, mock_init, mock_ee_image
    ):
        mock_ee_image.side_effect = lambda x=None: x if isinstance(x, MagicMock) else _make_chainable_ee_image()
        fi = final_Image()
        fi.get_temporal_composite = MagicMock(
            side_effect=lambda collection, aoi, **kwargs: {
                "image": _make_chainable_ee_image("comp"),
                "metadata": {},
                "coverage": None,
            }
        )
        fi.histogram_match = MagicMock(side_effect=lambda source_image, **kwargs: source_image)

        aoi = MagicMock()
        # Case 1: L8_SR (has AEROSOL) + S2_SR (has RED_EDGE1..4) -> common 6 bands without AEROSOL
        mt_input = {
            "periods": {
                "historical": {"collection": MagicMock(), "sensor": "L8_SR"},
                "baseline": {"collection": MagicMock(), "sensor": "S2_SR"},
            },
            "period_order": ["historical", "baseline"],
        }
        out = fi.get_multitemporal_composites(
            mt_input,
            aoi=aoi,
            reference_period="baseline",
            histogram_match=True,
            verbose=False,
        )
        assert out["common_bands"] == ["BLUE", "GREEN", "RED", "NIR", "SWIR1", "SWIR2"]
        assert "AEROSOL" not in out["common_bands"]
        fi.histogram_match.assert_called_once()

        # Case 2: L1_RAW (MSS: GREEN, RED, NIR1, NIR2->NIR) + L5_SR -> common 3 bands ['GREEN', 'RED', 'NIR']
        mt_mss = {
            "periods": {
                "historical": {"collection": MagicMock(), "sensor": "L1_RAW"},
                "baseline": {"collection": MagicMock(), "sensor": "L5_SR"},
            },
            "period_order": ["historical", "baseline"],
        }
        out_mss = fi.get_multitemporal_composites(
            mt_mss,
            aoi=aoi,
            reference_period="baseline",
            histogram_match=True,
            verbose=False,
        )
        assert out_mss["common_bands"] == ["GREEN", "RED", "NIR"]

    @patch("luma_ge.data_acquisition.ee")
    @patch("luma_ge.data_acquisition.ensure_ee_initialized")
    def test_histogram_match_single_reduce_region_and_two_step_interpolate(
        self, mock_init, mock_ee
    ):
        mock_ee.FeatureCollection = type("FakeFC", (), {})
        mock_ee.Image.cat.return_value = _make_chainable_ee_image("cat_matched")
        mock_ee.Image.side_effect = lambda x=None: x if isinstance(x, MagicMock) else _make_chainable_ee_image()

        fi = final_Image()
        src = _make_chainable_ee_image("src")
        ref = _make_chainable_ee_image("ref")
        region = MagicMock()

        out = fi.histogram_match(
            source_image=src,
            reference_image=ref,
            region=region,
            bands=["BLUE", "RED"],
            scale=30,
            max_buckets=128,
        )
        # Single reduceRegion per image (not per band)
        assert src.reduceRegion.call_count == 1
        assert ref.reduceRegion.call_count == 1
        # Two interpolate() calls per band (src CDF + inverse ref CDF) -> 4 calls for 2 bands
        assert src.interpolate.call_count == 4
        assert out is not None
