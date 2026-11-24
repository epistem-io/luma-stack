"""
Imagery Correction Module

Pre-processing corrections applied to satellite imagery after acquisition
and cloud masking, before compositing. Includes:

- Topographic correction (SCSc method)
- BRDF normalisation for cross-sensor harmonisation

Source for SCSc & BRDF: Poortinga et al. (2018) https://doi.org/10.3390/rs11070831
"""

import math
import logging
import ee
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from .ee_config import ensure_ee_initialized

# IMAGERY CORRECTION MODULES
logger = logging.getLogger(__name__)

#core methods for topographic correction
class Topographic_Correction:
    """
    Topographic correction for satellite imagery using the SCSc (Sun-Canopy-Sensor
    with c-correction) method. Corrects for differential illumination caused by terrain by comparing the
    measured radiance at each pixel to the radiance that would be measured on
    a flat, horizontal surface under the same solar illumination.

    The workflow for a single image is:
    1. ``compute_illumination_condition()``  — append IC, cosZ, cosS, slope bands
    2. ``apply_scsc_correction()``           — apply per-band SCSc formula
    3. ``correct_image()``                   — convenience wrapper for steps 1 + 2
    4. ``correct_collection()``              — map correction across an ee.ImageCollection

    Notes
    -----
    - Correction is only applied to pixels on slopes >= 5 deg with positive IC values.
      Flat / negative-IC pixels are filled with the original (uncorrected) values so
      the output always has full spatial coverage.
    - The ``scale`` parameter must match the native resolution of the input imagery:
      ``30`` for Landsat, ``10`` for Sentinel-2.
    - Bands that are not in ``band_list`` (e.g. thermal, QA) are passed through
      unchanged.

    References
    ----------
    Soenen, S.A., Peddle, D.R. & Coburn, C.A. (2005). SCSc: A modified
    Sun-Canopy-Sensor topographic correction in forested terrain.
    *IEEE Transactions on Geoscience and Remote Sensing*, 43(9), 2148–2159.
    https://doi.org/10.3390/rs11070831

    Example
    -------
    >>> tc = Topographic_Correction()
    >>>
    >>> # Correct a single image (Landsat, 30 m) — uses NASADEM by default
    >>> corrected_img = tc.correct_image(image, scale=30)
    >>>
    >>> # Override with a custom DEM
    >>> custom_dem = ee.Image('JAXA/ALOS/AW3D30/V3_2').select('DSM')
    >>> corrected_img = tc.correct_image(image, dem=custom_dem, scale=30)
    >>>
    >>> # Correct an entire collection (Sentinel-2, 10 m)
    >>> corrected_col = tc.correct_collection(collection, scale=10)
    """

    # Use NASADEM as default option
    # Override by passing dem= to any method when a higher-resolution or
    # region-specific DEM is available (e.g. ALOS World 3D, a custom asset).
    DEFAULT_DEM_ASSET: str = 'NASA/NASADEM_HGT/001'
    DEFAULT_DEM_BAND: str  = 'elevation'

    # Bands that receive topographic correction.
    # Landsat 4-9 SR and Sentinel-2 share these standardised names
    DEFAULT_OPTICAL_BANDS: List[str] = [
        'BLUE', 'GREEN', 'RED', 'NIR', 'SWIR1', 'SWIR2'
    ]
    S2_OPTICAL_BANDS: List[str] = [
        'BLUE', 'GREEN', 'RED', 'RED_EDGE1', 'RED_EDGE2',
        'RED_EDGE3', 'NIR', 'RED_EDGE4', 'SWIR1', 'SWIR2'
    ]

    def __init__(self, log_level: int = logging.INFO) -> None:
        """
        Initialise Topographic_Correction and ensure Earth Engine is ready.

        Parameters
        ----------
        log_level : int, optional
            Logging verbosity level (default: ``logging.INFO``).
        """
        ensure_ee_initialized()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)
        self.logger.info("Topographic_Correction initialised.")

    # ------------------------------------------------------------------
    # Step 1: Define Illumination condition
    # ------------------------------------------------------------------
    def compute_illumination_condition(
        self,
        image: ee.Image,
        dem: Optional[ee.Image] = None,
        buffer_m: int = 10_000,
    ) -> ee.Image:
        """
        Compute the illumination condition (IC) and append it to the image.

        The IC represents the cosine of the angle between the solar illumination
        vector and the terrain surface normal. It is computed as:

            IC = cos(Z)*cos(S) + sin(Z)*sin(S)*cos(phi_sun - phi_aspect)

        where Z is the solar zenith, S is the terrain slope, and phi denotes azimuths.

        Parameters
        ----------
        image : ee.Image
            Single image carrying solar-angle properties (in degrees).
            Landsat Collection-2: ``'SUN_ELEVATION'`` (converted via 90 - SUN_ELEVATION) / ``'SUN_AZIMUTH'``.
            Landsat Collection-1: ``'SOLAR_ZENITH_ANGLE'`` / ``'SOLAR_AZIMUTH_ANGLE'``.
            Sentinel-2: ``'MEAN_SOLAR_ZENITH_ANGLE'`` / ``'MEAN_SOLAR_AZIMUTH_ANGLE'``.
            All naming conventions are resolved automatically.
        dem : ee.Image, optional
            Defaults to ``NASA/NASADEM_HGT/001`` (30 m global).
            Override with a higher-resolution or region-specific DEM when needed.
        buffer_m : int, optional
            Buffer in metres applied to the image geometry before clipping
            terrain layers. Prevents edge artefacts. Default: ``10 000``.

        Returns
        -------
        ee.Image
            Original image with four additional bands appended:
            ``'IC'``, ``'cosZ'``, ``'cosS'``, ``'slope'`` (slope in degrees).
        """
        try:
            if dem is None:
                dem = ee.Image(self.DEFAULT_DEM_ASSET).select(self.DEFAULT_DEM_BAND)
            buffer_geom = image.geometry().buffer(buffer_m, 100)

            prop_names = image.propertyNames()
            sun_elev = ee.Number(
                ee.List([image.get('SUN_ELEVATION'), 0]).reduce(ee.Reducer.firstNonNull())
            )
            sz_from_elev = ee.Number(90).subtract(sun_elev)
            sz_direct = ee.List([
                image.get('MEAN_SOLAR_ZENITH_ANGLE'),
                image.get('SOLAR_ZENITH_ANGLE'),
            ]).reduce(ee.Reducer.firstNonNull())

            sz_deg = ee.Number(
                ee.Algorithms.If(
                    prop_names.contains('SUN_ELEVATION'),
                    sz_from_elev,
                    sz_direct,
                )
            )
            sa_deg = ee.Number(
                ee.List([
                    image.get('MEAN_SOLAR_AZIMUTH_ANGLE'),
                    image.get('SUN_AZIMUTH'),
                    image.get('SOLAR_AZIMUTH_ANGLE'),
                ]).reduce(ee.Reducer.firstNonNull())
            )
            sz_rad = (
                ee.Image.constant(sz_deg.multiply(math.pi / 180))
                .clip(buffer_geom)
            )
            sa_rad = (
                ee.Image.constant(sa_deg.multiply(math.pi / 180))
                .clip(buffer_geom)
            )

            # --- Terrain layers ----------------------------------------------
            slp_deg = ee.Terrain.slope(dem).clip(buffer_geom)
            slp_rad = slp_deg.multiply(math.pi / 180)
            asp_rad = ee.Terrain.aspect(dem).multiply(math.pi / 180).clip(buffer_geom)

            # --- IC components -----------------------------------------------
            cos_z = sz_rad.cos()
            cos_s = slp_rad.cos()
            sin_z = sz_rad.sin()
            sin_s = slp_rad.sin()

            slope_term  = cos_z.multiply(cos_s)
            aspect_term = sin_z.multiply(sin_s).multiply(sa_rad.subtract(asp_rad).cos())
            ic = slope_term.add(aspect_term).rename('IC')

            return ee.Image(
                image
                .addBands(ic)
                .addBands(cos_z.rename('cosZ'))
                .addBands(cos_s.rename('cosS'))
                .addBands(slp_deg.rename('slope'))
            )

        except ee.EEException as e:
            self.logger.error(f"EEException in compute_illumination_condition: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error in compute_illumination_condition: {e}")
            raise

    # ------------------------------------------------------------------
    # Step 2: Applied SCSc correction
    # ------------------------------------------------------------------

    def apply_scsc_correction(
        self,
        image: ee.Image,
        scale: int = 30,
        band_list: Optional[List[str]] = None,
        slope_threshold: float = 5.0,
        max_pixels: int = 1_000_000_000,
        tile_scale: int = 4,
    ) -> ee.Image:
        """
        Apply the SCSc topographic correction to an image that already carries
        the IC, cosZ, cosS, and slope bands (output of ``compute_illumination_condition``).
        """
        if band_list is None:
            band_list = self.DEFAULT_OPTICAL_BANDS

        try:
            mask = (
                image.select('slope').gte(slope_threshold)
                .And(image.select('IC').gte(0))
                .And(image.select('NIR').gt(0))
            )
            img_masked = ee.Image(image.updateMask(mask))

            reg_geometry = ee.Geometry(image.geometry().buffer(-100, 10))

            reg_scale = max(scale * 10, 300)
            reg_input = (
                ee.Image.constant(1).rename('constant')
                .addBands(img_masked.select(['IC'] + band_list))
                .updateMask(mask)
            )
            lr = reg_input.reduceRegion(
                reducer    = ee.Reducer.linearRegression(numX=2, numY=len(band_list)),
                geometry   = reg_geometry,
                scale      = reg_scale,
                maxPixels  = max_pixels,
                bestEffort = True,
                tileScale  = tile_scale,
            )

            default_coeffs = ee.Array([[0.0] * len(band_list), [1.0] * len(band_list)])
            coeffs = ee.Array(
                ee.List([lr.get('coefficients'), default_coeffs]).reduce(ee.Reducer.firstNonNull())
            )
            coeffs = ee.Array(
                ee.Algorithms.If(coeffs.length().get([0]).eq(2), coeffs, default_coeffs)
            )

            b_arr  = coeffs.slice(0, 0, 1).project([1])
            a_arr  = coeffs.slice(0, 1, 2).project([1])
            a_safe = a_arr.add(a_arr.eq(0))
            c_arr  = b_arr.divide(a_safe).multiply(a_arr.neq(0))

            c_img = ee.Image.cat([
                ee.Image.constant(c_arr.get([i])).rename(b)
                for i, b in enumerate(band_list)
            ])

            bands_img = img_masked.select(band_list)
            cos_sz    = img_masked.select('cosS').multiply(img_masked.select('cosZ'))
            ic_img    = img_masked.select('IC')

            corrected_bands = (
                bands_img.multiply(cos_sz.add(c_img))
                .divide(ic_img.add(c_img))
                .rename(band_list)
            )

            corrected_filled = corrected_bands.unmask(image.select(band_list))

            orig_bands = image.bandNames().removeAll(ee.List(['IC', 'cosZ', 'cosS', 'slope']))
            return ee.Image(
                image
                .addBands(corrected_filled, band_list, True)
                .select(orig_bands)
            )

        except ee.EEException as e:
            self.logger.error(f"EEException in apply_scsc_correction: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error in apply_scsc_correction: {e}")
            raise

    # -----------------------------------------------------------------
    # Step 3: Single-image convenience wrapper
    # -----------------------------------------------------------------
    def correct_image(
        self,
        image: ee.Image,
        dem: Optional[ee.Image] = None,
        scale: int = 30,
        band_list: Optional[List[str]] = None,
        slope_threshold: float = 5.0,
        buffer_m: int = 10_000,
        verbose: bool = True,
    ) -> ee.Image:
        """
        Apply SCSc topographic correction to a single image in one call.
        """
        if band_list is None:
            band_list = self.DEFAULT_OPTICAL_BANDS

        if verbose:
            self.logger.info(
                f"Applying SCSc topographic correction "
                f"(scale={scale}m, slope_threshold={slope_threshold} deg)"
            )
            if scale == 30:
                self.logger.info(
                    "Note: scale=30m (Landsat default). "
                    "Pass scale=10 for Sentinel-2 imagery."
                )

        img_with_ic = self.compute_illumination_condition(image, dem, buffer_m=buffer_m)
        corrected   = self.apply_scsc_correction(
            img_with_ic,
            scale           = scale,
            band_list       = band_list,
            slope_threshold = slope_threshold,
        )

        if verbose:
            self.logger.info("SCSc correction applied to single image.")

        return ee.Image(image.addBands(corrected.select(band_list), band_list, True))

    # ------------------------------------------------------------------
    # Step 4: Collection-level wrapper
    # ------------------------------------------------------------------

    def correct_collection(
        self,
        collection: ee.ImageCollection,
        dem: Optional[ee.Image] = None,
        scale: int = 30,
        band_list: Optional[List[str]] = None,
        slope_threshold: float = 5.0,
        buffer_m: int = 10_000,
        verbose: bool = True,
    ) -> ee.ImageCollection:
        """
        Apply SCSc topographic correction to every image in a collection.
        """
        if band_list is None:
            band_list = self.DEFAULT_OPTICAL_BANDS

        if dem is None:
            dem = ee.Image(self.DEFAULT_DEM_ASSET).select(self.DEFAULT_DEM_BAND)
            if verbose:
                self.logger.info(f"Using default DEM: {self.DEFAULT_DEM_ASSET} (band: {self.DEFAULT_DEM_BAND})")
        if verbose:
            self.logger.info(
                f"Applying SCSc topographic correction to collection "
                f"(scale={scale}m, slope_threshold={slope_threshold} deg)"
            )
            if scale == 30:
                self.logger.info(
                    "Note: scale=30m (Landsat default). "
                    "Pass scale=10 for Sentinel-2 imagery."
                )

        def _correct(img: ee.Image) -> ee.Image:
            img_with_ic = self.compute_illumination_condition(img, dem, buffer_m=buffer_m)
            corrected = self.apply_scsc_correction(
                img_with_ic,
                scale           = scale,
                band_list       = band_list,
                slope_threshold = slope_threshold,
            )
            return ee.Image(img.addBands(corrected.select(band_list), band_list, True))

        corrected = collection.map(_correct)

        if verbose:
            self.logger.info("SCSc correction mapped over collection.")

        return corrected

#core methods for Radiometric normalzation using BRDF
class BRDF_Correction:
    """
    Bidirectional Reflectance Distribution Function (BRDF) c-factor normalisation
    for Landsat and Sentinel-2 surface reflectance imagery.

    Normalises surface reflectance to a nadir-viewing geometry (view zenith = 0)
    using the Ross-Thick volumetric kernel (`kvol`) and c-factor coefficients
    following Poortinga et al. (2018) and Roy et al. (2016, 2017).

    The workflow for a single image is:
    1. ``compute_sun_angles()``   — per-pixel solar azimuth and zenith from acquisition date & lat/lon
    2. ``compute_view_angles()``  — per-pixel sensor view azimuth and zenith from scene footprint
    3. ``compute_kvol()``         — Ross-Thick volumetric scattering kernel (`kvol` and nadir `kvol0`)
    4. ``correct_image()``        — apply per-band c-factor correction while preserving ancillary bands & metadata
    5. ``correct_collection()``   — map ``correct_image`` lazily across an ``ee.ImageCollection``

    References
    ----------
    - Poortinga, A. et al. (2018 / 2019). Mapping plantations in Myanmar by fusing
      Landsat-8, Sentinel-2 and Sentinel-1 data along with systematic error
      quantification. *Remote Sensing*, 11(7), 831. https://doi.org/10.3390/rs11070831
    - Lucht, W., Schaaf, C.B. & Strahler, A.H. (2000). An algorithm for the retrieval
      of albedo from space using semiempirical BRDF models. *IEEE TGRS*, 38(2), 977–998.
    - Roy, D.P. et al. (2016). A general method to normalize Landsat reflectance data
      to nadir BRDF adjusted reflectance. *Remote Sensing of Environment*, 176, 255–271.
    """
    #default configurations value
    MAX_SATELLITE_ZENITH: float = 7.5
    MAX_DISTANCE: int = 1_000_000
    UPPER_LEFT: int = 0
    LOWER_LEFT: int = 1
    LOWER_RIGHT: int = 2
    UPPER_RIGHT: int = 3

    DEFAULT_OPTICAL_BANDS: List[str] = [
        'BLUE', 'GREEN', 'RED', 'NIR', 'SWIR1', 'SWIR2'
    ]
    S2_OPTICAL_BANDS: List[str] = [
        'BLUE', 'GREEN', 'RED', 'RED_EDGE1', 'RED_EDGE2',
        'RED_EDGE3', 'NIR', 'RED_EDGE4', 'SWIR1', 'SWIR2'
    ]

    # Per-band BRDF model parameters (f_iso, f_geo, f_vol) from Poortinga et al. (2018)
    BAND_COEFFICIENTS: Dict[str, Dict[str, float]] = {
        'BLUE':      {'f_iso': 0.0774, 'f_geo': 0.0079, 'f_vol': 0.0372},
        'GREEN':     {'f_iso': 0.1306, 'f_geo': 0.0178, 'f_vol': 0.0580},
        'RED':       {'f_iso': 0.1690, 'f_geo': 0.0227, 'f_vol': 0.0574},
        'RED_EDGE1': {'f_iso': 0.2085, 'f_geo': 0.0256, 'f_vol': 0.0845},
        'RED_EDGE2': {'f_iso': 0.2316, 'f_geo': 0.0273, 'f_vol': 0.1003},
        'RED_EDGE3': {'f_iso': 0.2599, 'f_geo': 0.0294, 'f_vol': 0.1197},
        'NIR':       {'f_iso': 0.3093, 'f_geo': 0.0330, 'f_vol': 0.1535},
        'RED_EDGE4': {'f_iso': 0.2907, 'f_geo': 0.0410, 'f_vol': 0.1611},
        'SWIR1':     {'f_iso': 0.3430, 'f_geo': 0.0453, 'f_vol': 0.1154},
        'SWIR2':     {'f_iso': 0.2658, 'f_geo': 0.0387, 'f_vol': 0.0639},
    }

    def __init__(self, log_level: int = logging.INFO) -> None:
        """
        Initialise BRDF_Correction and ensure Earth Engine is ready.

        Parameters
        ----------
        log_level : int, optional
            Logging verbosity level (default: ``logging.INFO``).
        """
        ensure_ee_initialized()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)
        self.logger.info("BRDF_Correction initialised.")

    # ------------------------------------------------------------------
    # Internal server-side geometry & conditional helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _line_from_coords(coordinates: ee.List, from_index: int, to_index: int) -> ee.Geometry:
        return ee.Geometry.LineString(
            ee.List([coordinates.get(from_index), coordinates.get(to_index)])
        )

    @staticmethod
    def _invert_mask(mask: ee.Image) -> ee.Image:
        return mask.multiply(-1).add(1)

    @classmethod
    def _where(cls, condition: ee.Image, true_value: ee.Image, false_value: ee.Image) -> ee.Image:
        true_masked = true_value.mask(condition)
        false_masked = false_value.mask(cls._invert_mask(condition))
        return true_masked.unmask(false_masked)

    @staticmethod
    def _value(ee_list: ee.List, index: int) -> ee.Number:
        return ee.Number(ee_list.get(index))

    # ------------------------------------------------------------------
    # Step 1: Solar geometry
    # ------------------------------------------------------------------
    def compute_sun_angles(self, date: ee.Date, footprint: ee.List) -> Tuple[ee.Image, ee.Image]:
        """
        Compute per-pixel solar azimuth and solar zenith angles (in radians).

        Parameters
        ----------
        date : ee.Date
            Acquisition timestamp of the image (``image.date()``).
        footprint : ee.List
            Outer ring coordinates of the bounding footprint polygon.

        Returns
        -------
        tuple of (ee.Image, ee.Image)
            ``(sunAz, sunZen)`` clipped to ``footprint`` with band names
            ``'sunAz'`` and ``'sunZen'``.
        """
        pi = ee.Number(math.pi)
        jdp = date.getFraction('year')
        seconds_in_hour = 3600
        hour_gmt = ee.Number(date.getRelative('second', 'day')).divide(seconds_in_hour)

        lon_lat = ee.Image.pixelLonLat()
        lat_rad = lon_lat.select('latitude').multiply(pi.divide(180))
        long_deg = lon_lat.select('longitude')

        # Julian day proportion in radians
        jdpr = jdp.multiply(pi).multiply(2)

        a = ee.List([0.000075, 0.001868, 0.032077, 0.014615, 0.040849])
        mean_solar_time = long_deg.divide(15.0).add(hour_gmt)
        local_solar_diff1 = (
            self._value(a, 0)
            .add(self._value(a, 1).multiply(jdpr.cos()))
            .subtract(self._value(a, 2).multiply(jdpr.sin()))
            .subtract(self._value(a, 3).multiply(jdpr.multiply(2).cos()))
            .subtract(self._value(a, 4).multiply(jdpr.multiply(2).sin()))
        )
        local_solar_diff2 = local_solar_diff1.multiply(12 * 60)
        local_solar_diff = local_solar_diff2.divide(pi)
        true_solar_time = (
            mean_solar_time
            .add(local_solar_diff.divide(60))
            .subtract(12.0)
        )

        # Hour as an angle
        ah = true_solar_time.multiply(
            ee.Number(self.MAX_SATELLITE_ZENITH * 2).multiply(pi.divide(180))
        )
        b = ee.List([0.006918, 0.399912, 0.070257, 0.006758, 0.000907, 0.002697, 0.001480])
        delta = (
            self._value(b, 0)
            .subtract(self._value(b, 1).multiply(jdpr.cos()))
            .add(self._value(b, 2).multiply(jdpr.sin()))
            .subtract(self._value(b, 3).multiply(jdpr.multiply(2).cos()))
            .add(self._value(b, 4).multiply(jdpr.multiply(2).sin()))
            .subtract(self._value(b, 5).multiply(jdpr.multiply(3).cos()))
            .add(self._value(b, 6).multiply(jdpr.multiply(3).sin()))
        )

        cos_sun_zen = (
            lat_rad.sin().multiply(delta.sin())
            .add(lat_rad.cos().multiply(ah.cos()).multiply(delta.cos()))
        )
        sun_zen = cos_sun_zen.acos()

        # Sun azimuth from south, turning west
        sin_sun_az_sw = ah.sin().multiply(delta.cos()).divide(sun_zen.sin()).clamp(-1.0, 1.0)
        cos_sun_az_sw = (
            lat_rad.cos().multiply(-1).multiply(delta.sin())
            .add(lat_rad.sin().multiply(delta.cos()).multiply(ah.cos()))
            .divide(sun_zen.sin())
        )
        sun_az_sw = sin_sun_az_sw.asin()
        sun_az_sw = self._where(cos_sun_az_sw.lte(0), sun_az_sw.multiply(-1).add(pi), sun_az_sw)
        sun_az_sw = self._where(
            cos_sun_az_sw.gt(0).And(sin_sun_az_sw.lte(0)),
            sun_az_sw.add(pi.multiply(2)),
            sun_az_sw,
        )

        sun_az = sun_az_sw.add(pi)
        sun_az = self._where(
            sun_az.gt(pi.multiply(2)),
            sun_az.subtract(pi.multiply(2)),
            sun_az,
        )

        footprint_polygon = ee.Geometry.Polygon(footprint)
        sun_az = sun_az.clip(footprint_polygon).rename(['sunAz'])
        sun_zen = sun_zen.clip(footprint_polygon).rename(['sunZen'])
        return sun_az, sun_zen

    # ------------------------------------------------------------------
    # Step 2: Sensor view geometry
    # ------------------------------------------------------------------
    def compute_view_angles(self, footprint: ee.List) -> Tuple[ee.Image, ee.Image]:
        """
        Compute per-pixel sensor view azimuth and view zenith angles (in radians)
        from the scene bounding footprint.

        Parameters
        ----------
        footprint : ee.List
            Outer ring coordinates of the bounding footprint polygon.

        Returns
        -------
        tuple of (ee.Image, ee.Image)
            ``(viewAz, viewZen)`` with band names ``'viewAz'`` and ``'viewZen'``.
        """
        pi = ee.Number(math.pi)

        def _x(point):
            return ee.Number(ee.List(point).get(0))

        def _y(point):
            return ee.Number(ee.List(point).get(1))

        upper_center = self._line_from_coords(
            footprint, self.UPPER_LEFT, self.UPPER_RIGHT
        ).centroid().coordinates()
        lower_center = self._line_from_coords(
            footprint, self.LOWER_LEFT, self.LOWER_RIGHT
        ).centroid().coordinates()

        slope = (
            _y(lower_center).subtract(_y(upper_center))
            .divide(_x(lower_center).subtract(_x(upper_center)))
        )
        slope_perp = ee.Number(-1).divide(slope)
        view_az = ee.Image(pi.divide(2).subtract(slope_perp.atan())).rename(['viewAz'])

        left_line = self._line_from_coords(footprint, self.UPPER_LEFT, self.LOWER_LEFT)
        right_line = self._line_from_coords(footprint, self.UPPER_RIGHT, self.LOWER_RIGHT)
        left_distance = ee.FeatureCollection(left_line).distance(self.MAX_DISTANCE)
        right_distance = ee.FeatureCollection(right_line).distance(self.MAX_DISTANCE)

        view_zen = (
            right_distance.multiply(ee.Number(self.MAX_SATELLITE_ZENITH * 2))
            .divide(right_distance.add(left_distance))
            .subtract(ee.Number(self.MAX_SATELLITE_ZENITH))
            .clip(ee.Geometry.Polygon(footprint))
            .rename(['viewZen'])
            .multiply(pi.divide(180))
        )
        return view_az, view_zen

    # ------------------------------------------------------------------
    # Step 3: Ross-Thick volumetric kernel (Lucht et al. 2000)
    # ------------------------------------------------------------------
    def compute_kvol(
        self,
        sun_az: ee.Image,
        sun_zen: ee.Image,
        view_az: ee.Image,
        view_zen: ee.Image,
    ) -> Tuple[ee.Image, ee.Image]:
        """
        Compute Ross-Thick volumetric scattering kernel at actual view zenith (`kvol`)
        and at nadir view zenith = 0 (`kvol0`).

        Parameters
        ----------
        sun_az, sun_zen, view_az, view_zen : ee.Image
            Solar and view angle images in radians.

        Returns
        -------
        tuple of (ee.Image, ee.Image)
            ``(kvol, kvol0)`` with band names ``'kvol'`` and ``'kvol0'``.
        """
        pi = ee.Number(math.pi)
        relative_azimuth = sun_az.subtract(view_az).rename(['relAz'])

        pa1 = view_zen.cos().multiply(sun_zen.cos())
        pa2 = view_zen.sin().multiply(sun_zen.sin()).multiply(relative_azimuth.cos())
        phase_angle1 = pa1.add(pa2)
        phase_angle = phase_angle1.acos()
        p1 = ee.Image(pi.divide(2)).subtract(phase_angle)
        p2 = p1.multiply(phase_angle1)
        p3 = p2.add(phase_angle.sin())
        p4 = sun_zen.cos().add(view_zen.cos())
        p5 = ee.Image(pi.divide(4))
        kvol = p3.divide(p4).subtract(p5).rename(['kvol'])

        view_zen0 = ee.Image(0)
        pa10 = view_zen0.cos().multiply(sun_zen.cos())
        pa20 = view_zen0.sin().multiply(sun_zen.sin()).multiply(relative_azimuth.cos())
        phase_angle10 = pa10.add(pa20)
        phase_angle0 = phase_angle10.acos()
        p10 = ee.Image(pi.divide(2)).subtract(phase_angle0)
        p20 = p10.multiply(phase_angle10)
        p30 = p20.add(phase_angle0.sin())
        p40 = sun_zen.cos().add(view_zen0.cos())
        p50 = ee.Image(pi.divide(4))
        kvol0 = p30.divide(p40).subtract(p50).rename(['kvol0'])

        return kvol, kvol0

    # ------------------------------------------------------------------
    # Step 4: Per-band c-factor correction & single-image wrapper
    # ------------------------------------------------------------------
    def _correct_band(
        self,
        image: ee.Image,
        band_name: str,
        kvol_pi: ee.Image,
        kvol0_pi: ee.Image,
    ) -> ee.Image:
        coeffs = self.BAND_COEFFICIENTS[band_name]
        iso = ee.Image(coeffs['f_iso'])
        geo = ee.Image(coeffs['f_geo'])
        vol = ee.Image(coeffs['f_vol'])

        pred = vol.multiply(kvol_pi).add(geo.multiply(kvol_pi)).add(iso).rename(['pred'])
        pred0 = vol.multiply(kvol0_pi).add(geo.multiply(kvol0_pi)).add(iso).rename(['pred0'])
        cfac = pred0.divide(pred).rename(['cfac'])
        return image.select(band_name).multiply(cfac).rename([band_name])

    def correct_image(
        self,
        image: ee.Image,
        band_list: Optional[List[str]] = None,
        verbose: bool = True,
    ) -> ee.Image:
        """
        Apply BRDF c-factor correction to a single image.

        Always uses ``image.date()`` dynamically (supports both Landsat 4–9 and
        Sentinel-2). Overwrites only ``band_list`` in-place on ``image`` so
        ancillary bands (e.g. ``THERMAL``, ``QA_PIXEL``) and all image properties
        (``system:time_start``, solar angles, ``system:index``) are preserved.

        Parameters
        ----------
        image : ee.Image
            Pre-processed image with standardised band names (output of
            ``rename_landsat_bands`` or ``rename_s2_bands``).
        band_list : list of str, optional
            Optical bands to correct. Defaults to ``DEFAULT_OPTICAL_BANDS``
            (``['BLUE','GREEN','RED','NIR','SWIR1','SWIR2']``). Pass
            ``BRDF_Correction.S2_OPTICAL_BANDS`` to also correct Sentinel-2 Red-Edge bands.
        verbose : bool, optional
            Log progress messages. Default: ``True``.

        Returns
        -------
        ee.Image
            BRDF-corrected image with original band structure and properties preserved.
        """
        if band_list is None:
            band_list = self.DEFAULT_OPTICAL_BANDS

        valid_bands = [b for b in band_list if b in self.BAND_COEFFICIENTS]
        if not valid_bands:
            raise ValueError(
                f"No supported BRDF bands in band_list={band_list}. "
                f"Supported: {list(self.BAND_COEFFICIENTS.keys())}"
            )

        try:
            if verbose:
                self.logger.info(f"Applying BRDF correction to bands: {valid_bands}")

            date = image.date()
            footprint = ee.List(image.geometry().bounds().bounds().coordinates().get(0))

            sun_az, sun_zen = self.compute_sun_angles(date, footprint)
            view_az, view_zen = self.compute_view_angles(footprint)
            kvol, kvol0 = self.compute_kvol(sun_az, sun_zen, view_az, view_zen)

            pi = ee.Number(math.pi)
            kvol_pi = kvol.multiply(pi)
            kvol0_pi = kvol0.multiply(pi)

            corrected_bands = [
                self._correct_band(image, band_name, kvol_pi, kvol0_pi)
                for band_name in valid_bands
            ]
            corrected_stack = ee.Image.cat(corrected_bands)

            return ee.Image(
                image
                .addBands(corrected_stack.select(valid_bands), valid_bands, True)
                .copyProperties(image, image.propertyNames())
            )
        except ee.EEException as e:
            self.logger.error(f"EEException in BRDF_Correction.correct_image: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error in BRDF_Correction.correct_image: {e}")
            raise

    # ------------------------------------------------------------------
    # Step 5: Collection-level wrapper
    # ------------------------------------------------------------------
    def correct_collection(
        self,
        collection: ee.ImageCollection,
        band_list: Optional[List[str]] = None,
        verbose: bool = True,
    ) -> ee.ImageCollection:
        """
        Apply BRDF c-factor correction lazily across an ``ee.ImageCollection``.

        Parameters
        ----------
        collection : ee.ImageCollection
            Filtered and pre-processed image collection.
        band_list : list of str, optional
            Optical bands to correct. Defaults to ``DEFAULT_OPTICAL_BANDS``.
        verbose : bool, optional
            Log progress messages. Default: ``True``.

        Returns
        -------
        ee.ImageCollection
            Collection with every image BRDF-corrected.
        """
        if band_list is None:
            band_list = self.DEFAULT_OPTICAL_BANDS

        if verbose:
            self.logger.info(f"Mapping BRDF correction over collection (bands={band_list}).")

        return collection.map(lambda img: self.correct_image(img, band_list=band_list, verbose=False))

class PIF_Normalization:
    """
    Unsupervised Pseudo-Invariant Feature (PIF) relative radiometric normalization
    between a historical/source composite and a baseline/reference composite.

    Follows the Google Earth Engine community tutorial (*Pseudo-Invariant Feature
    Matching*) with production guards for Landsat Collection 2 SR / HLS / Sentinel-2:

    1. **Spectral Distance (``ee.Image.spectralDistance``):** Computes per-pixel
       spectral dissimilarity (``'sid'`` Spectral Information Divergence, ``'sam'``
       Spectral Angle Mapper, or ``'sed'`` Squared Euclidean Distance) across common
       optical bands. For ``'sid'``, bands are clamped to ``>= 1e-4`` so slight
       negative Landsat C2 SR pixels over water/shadows do not produce ``NaN``.
    2. **Automatic PIF Mask (``ee.Reducer.percentile``):** Selects the most
       spectrally invariant ``pif_percentile``% of pixels (default ``10%``) as
       pseudo-invariant features — zero manual polygons required.
    3. **Server-Side Linear Fit (``ee.Reducer.linearFit``):** Fits a per-band linear
       transform (``y = scale * x + offset``) strictly on PIF pixels and applies it
       to the full ``source_image``. Any band with a null or out-of-bounds slope
       (outside ``scale_bounds``, default ``(0.5, 2.0)``) safely falls back to
       identity (``scale=1.0, offset=0.0``) purely server-side.

    Reference
    ---------
    https://developers.google.com/earth-engine/tutorials/community/pseudo-invariant-feature-matching
    """

    DEFAULT_OPTICAL_BANDS: List[str] = [
        'BLUE', 'GREEN', 'RED', 'NIR', 'SWIR1', 'SWIR2'
    ]
    SUPPORTED_METRICS = ('sid', 'sam', 'sed')

    def __init__(self, log_level: int = logging.INFO) -> None:
        ensure_ee_initialized()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)
        self.logger.info("PIF_Normalization initialised.")

    @classmethod
    def _validate_params(
        cls,
        metric: str,
        pif_percentile: float,
        scale_bounds: Tuple[float, float],
    ) -> str:
        metric_key = str(metric).strip().lower()
        if metric_key not in cls.SUPPORTED_METRICS:
            raise ValueError(
                f"Unsupported PIF spectralDistance metric '{metric}'. "
                f"Supported: {list(cls.SUPPORTED_METRICS)}"
            )
        if not (0.0 < float(pif_percentile) < 100.0):
            raise ValueError(
                f"pif_percentile must be strictly between 0 and 100, got {pif_percentile}."
            )
        lo, hi = float(scale_bounds[0]), float(scale_bounds[1])
        if lo <= 0.0 or hi <= lo:
            raise ValueError(
                f"scale_bounds must satisfy 0 < min_scale < max_scale, got {scale_bounds}."
            )
        return metric_key

    def compute_pif_mask(
        self,
        source_image: ee.Image,
        reference_image: ee.Image,
        region: Any,
        bands: Optional[Sequence[str]] = None,
        metric: str = 'sid',
        pif_percentile: float = 10.0,
        scale: int = 30,
        max_pixels: float = 1e9,
        tile_scale: int = 4,
    ) -> Tuple[ee.Image, ee.Image]:
        """
        Compute the spectral distance image and binary PIF mask isolating the most
        spectrally invariant ``pif_percentile``% of pixels within ``region``.

        Returns
        -------
        tuple of (ee.Image, ee.Image)
            ``(pif_mask, distance_image)``.
        """
        metric_key = self._validate_params(metric, pif_percentile, (0.5, 2.0))
        band_list = list(bands) if bands is not None else list(self.DEFAULT_OPTICAL_BANDS)
        if not band_list:
            raise ValueError("bands must contain at least one band name.")

        geometry = region.geometry() if isinstance(region, ee.FeatureCollection) else region

        src_sel = source_image.select(band_list).toFloat()
        ref_sel = reference_image.select(band_list).toFloat()

        # Require valid (unmasked) pixels in both images across all selected bands
        joint_mask = src_sel.mask().reduce(ee.Reducer.min()).And(
            ref_sel.mask().reduce(ee.Reducer.min())
        )
        src_clean = src_sel.updateMask(joint_mask)
        ref_clean = ref_sel.updateMask(joint_mask)

        # SID takes log(p/q), so clamp <= 0 reflectance (e.g. dark water in Landsat C2 SR)
        if metric_key == 'sid':
            src_for_dist = src_clean.max(1e-4)
            ref_for_dist = ref_clean.max(1e-4)
        else:
            src_for_dist = src_clean
            ref_for_dist = ref_clean

        distance_img = (
            src_for_dist.spectralDistance(ref_for_dist, metric_key)
            .rename('distance')
            .updateMask(joint_mask)
        )

        pct_dict = distance_img.reduceRegion(
            reducer=ee.Reducer.percentile([float(pif_percentile)]),
            geometry=geometry,
            scale=scale,
            maxPixels=max_pixels,
            bestEffort=True,
            tileScale=tile_scale,
        )
        threshold = ee.Number(
            ee.List([pct_dict.get('distance'), 0.0]).reduce(ee.Reducer.firstNonNull())
        )
        pif_mask = distance_img.lte(threshold).selfMask().rename('pif_mask')
        return pif_mask, distance_img

    def normalize_image(
        self,
        source_image: ee.Image,
        reference_image: ee.Image,
        region: Any,
        bands: Optional[Sequence[str]] = None,
        metric: str = 'sid',
        pif_percentile: float = 10.0,
        scale: int = 30,
        max_pixels: float = 1e9,
        tile_scale: int = 4,
        scale_bounds: Tuple[float, float] = (0.5, 2.0),
        return_diagnostics: bool = False,
        verbose: bool = True,
    ) -> Union[ee.Image, Dict[str, Any]]:
        """
        Radiometrically normalize ``source_image`` onto ``reference_image`` via
        server-side per-band linear regression over automatically detected PIF pixels.

        Parameters
        ----------
        source_image : ee.Image
            Historical / target composite to normalize.
        reference_image : ee.Image
            Baseline / reference composite.
        region : ee.Geometry or ee.FeatureCollection
            Area of interest used to locate PIFs and fit linear coefficients.
        bands : sequence of str, optional
            Optical bands to normalize (defaults to ``DEFAULT_OPTICAL_BANDS``).
            Any other bands on ``source_image`` (e.g. ``THERMAL``) pass through unchanged.
        metric : {'sid', 'sam', 'sed'}, default='sid'
            Spectral distance metric passed to ``ee.Image.spectralDistance``.
        pif_percentile : float, default=10.0
            Lower percentile (0-100) of spectral distance kept as PIF pixels.
        scale_bounds : tuple of (float, float), default=(0.5, 2.0)
            Valid physical range for fitted linear slope. Bands whose fit falls
            outside this range (or is null) fall back to identity ``(1.0, 0.0)``.
        return_diagnostics : bool, default=False
            If True, returns ``{'image', 'pif_mask', 'distance_image', 'coefficients'}``
            instead of only the normalized ``ee.Image``.
        """
        metric_key = self._validate_params(metric, pif_percentile, scale_bounds)
        band_list = list(bands) if bands is not None else list(self.DEFAULT_OPTICAL_BANDS)
        geometry = region.geometry() if isinstance(region, ee.FeatureCollection) else region

        if verbose:
            self.logger.info(
                f"Applying PIF normalization (metric={metric_key}, "
                f"pif_percentile={pif_percentile}%, bands={band_list}, scale={scale}m)"
            )

        pif_mask, distance_img = self.compute_pif_mask(
            source_image=source_image,
            reference_image=reference_image,
            region=geometry,
            bands=band_list,
            metric=metric_key,
            pif_percentile=pif_percentile,
            scale=scale,
            max_pixels=max_pixels,
            tile_scale=tile_scale,
        )

        min_slope, max_slope = float(scale_bounds[0]), float(scale_bounds[1])
        normalized_bands: List[ee.Image] = []
        coeffs_server: Dict[str, ee.Dictionary] = {}

        for b in band_list:
            pair_img = (
                ee.Image.cat([
                    source_image.select(b).toFloat().rename('x'),
                    reference_image.select(b).toFloat().rename('y'),
                ])
                .updateMask(pif_mask)
            )
            fit = pair_img.reduceRegion(
                reducer=ee.Reducer.linearFit(),
                geometry=geometry,
                scale=scale,
                maxPixels=max_pixels,
                bestEffort=True,
                tileScale=tile_scale,
            )
            raw_scale = ee.Number(
                ee.List([fit.get('scale'), 1.0]).reduce(ee.Reducer.firstNonNull())
            )
            raw_offset = ee.Number(
                ee.List([fit.get('offset'), 0.0]).reduce(ee.Reducer.firstNonNull())
            )
            valid_slope = raw_scale.gte(min_slope).And(raw_scale.lte(max_slope))
            safe_scale = ee.Number(ee.Algorithms.If(valid_slope, raw_scale, 1.0))
            safe_offset = ee.Number(ee.Algorithms.If(valid_slope, raw_offset, 0.0))

            coeffs_server[b] = ee.Dictionary({
                'scale': safe_scale,
                'offset': safe_offset,
                'raw_scale': raw_scale,
                'raw_offset': raw_offset,
                'used_fallback': valid_slope.Not(),
            })
            norm_b = (
                source_image.select(b)
                .toFloat()
                .multiply(safe_scale)
                .add(safe_offset)
                .rename(b)
            )
            normalized_bands.append(norm_b)

        norm_stack = ee.Image.cat(normalized_bands)
        result_img = ee.Image(
            source_image
            .addBands(norm_stack.select(band_list), band_list, True)
            .copyProperties(source_image, source_image.propertyNames())
        )

        if return_diagnostics:
            return {
                'image': result_img,
                'pif_mask': pif_mask,
                'distance_image': distance_img,
                'coefficients': ee.Dictionary(coeffs_server),
            }
        return result_img
