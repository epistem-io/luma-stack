"""
Imagery Correction Module

Pre-processing corrections applied to satellite imagery after acquisition
and cloud masking, before compositing. Includes:

- Topographic correction  (SCSc method)
- [Planned] BRDF normalisation for cross-sensor harmonisation

Source for SCSc: https://doi.org/10.3390/rs11070831
"""

import math
import logging
import ee
from typing import List, Optional, Union
from .ee_config import ensure_ee_initialized

#IMAGERY CORRECTION MODULES
logger = logging.getLogger(__name__)

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
    - Correction is only applied to pixels on slopes ≥ 5° with positive IC values.
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

    #Use NASADEM as default option
    #Override by passing dem= to any method when a higher-resolution or
    #region-specific DEM is available (e.g. ALOS World 3D, a custom asset).
    DEFAULT_DEM_ASSET: str = 'NASA/NASADEM_HGT/001'
    DEFAULT_DEM_BAND: str  = 'elevation'

    #Bands that receive topographic correction.
    #Landsat 4-9 SR and Sentinel-2 share these standardised names after
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

        Example
        -------
        >>> tc = Topographic_Correction()
        >>> img_with_ic = tc.compute_illumination_condition(image)
        >>> # Override DEM
        >>> custom_dem = ee.Image('JAXA/ALOS/AW3D30/V3_2').select('DSM')
        >>> img_with_ic = tc.compute_illumination_condition(image, dem=custom_dem)
        """
        try:
            if dem is None:
                dem = ee.Image(self.DEFAULT_DEM_ASSET).select(self.DEFAULT_DEM_BAND)
            buffer_geom = image.geometry().buffer(buffer_m, 100)

            # --- Solar geometry: degrees to radians (server-side constants) ---
            # Sentinel-2 uses MEAN_SOLAR_ZENITH_ANGLE / MEAN_SOLAR_AZIMUTH_ANGLE.
            # Landsat Collection 2 uses SUN_ELEVATION (zenith = 90 - SUN_ELEVATION) / SUN_AZIMUTH.
            # Landsat Collection 1 uses SOLAR_ZENITH_ANGLE / SOLAR_AZIMUTH_ANGLE.
            # Resolve whichever is present safely on the server without null arithmetic errors.
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
            slp_deg = ee.Terrain.slope(dem).clip(buffer_geom)   # degrees, for mask
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

    def apply_scsc_correction(self, image: ee.Image, scale: int = 30,
        band_list: Optional[List[str]] = None,
        slope_threshold: float = 5.0,
        max_pixels: int = 1_000_000_000,
        tile_scale: int = 4,
    ) -> ee.Image:
        """
        Apply the SCSc topographic correction to an image that already carries
        the IC, cosZ, cosS, and slope bands (output of ``compute_illumination_condition``).
        For each band *b* in ``band_list`` the correction is:

            c  = b / a                   (regression offset / slope)
            ρ* = ρ · (cosZ·cosS + c) / (IC + c)

        where *a* and *b* are the slope and intercept of the linear regression
        between IC and ρ, computed from pixels where slope ≥ ``slope_threshold``
        and IC ≥ 0.

        Flat-terrain pixels (slope < threshold) and pixels with negative IC are
        filled with the original uncorrected reflectance so the output retains
        full spatial coverage.

        Notes
        -----
        **Single-pass multi-band regression:** Uses a single
        ``ee.Reducer.linearRegression(numX=2, numY=len(band_list))`` call per
        image rather than one ``reduceRegion`` per band, preventing GEE's
        ``Too many concurrent aggregations`` error when mapped over large
        image collections.

        **Red Edge bands (Sentinel-2):** ``RED_EDGE1``, ``RED_EDGE2``,
        ``RED_EDGE3``, ``RED_EDGE4`` are not in ``DEFAULT_OPTICAL_BANDS`` and
        therefore pass through uncorrected but intact as ancillary bands unless
        ``band_list=Topographic_Correction.S2_OPTICAL_BANDS`` is explicitly supplied.
        This is intentional — the SCSc regression is fit on the six common optical
        bands by default; applying the same c-value to Red Edge bands would introduce
        sensor-specific bias.

        **NIR validity mask:** uses ``NIR > 0`` which correctly excludes fill
        pixels for both scaled reflectance (0.0–1.0) and raw DN (0–10 000).

        Parameters
        ----------
        image : ee.Image
            Image output of ``compute_illumination_condition`` — must carry
            ``'IC'``, ``'cosZ'``, ``'cosS'``, and ``'slope'`` bands alongside
            the optical reflectance bands.
        scale : int, optional
            Pixel resolution (metres) of the input imagery.
            Use ``30`` for Landsat, ``10`` for Sentinel-2. Default: ``30``.
        band_list : list of str, optional
            Bands to correct. Defaults to ``DEFAULT_OPTICAL_BANDS``
            (``['BLUE','GREEN','RED','NIR','SWIR1','SWIR2']``).
        slope_threshold : float, optional
            Minimum slope (degrees) for a pixel to be included in the
            regression and correction. Default: ``5.0``.
        max_pixels : int, optional
            ``maxPixels`` argument passed to ``reduceRegion``. Default: ``1e9``.
        tile_scale : int, optional
            ``tileScale`` argument passed to ``reduceRegion`` to prevent memory
            errors on high-resolution scenes (e.g. 10 m Sentinel-2). Default: ``4``.

        Returns
        -------
        ee.Image
            Topographically corrected image. Non-optical bands (thermal, QA,
            etc.) are preserved unchanged. ``system:time_start`` and all other
            image properties are copied to the output.

        Example
        -------
        >>> tc = Topographic_Correction()
        >>> dem = ee.Image('NASA/NASADEM_HGT/001').select('elevation')
        >>> img_with_ic = tc.compute_illumination_condition(image, dem)
        >>> corrected = tc.apply_scsc_correction(img_with_ic, scale=30)
        """
        if band_list is None:
            band_list = self.DEFAULT_OPTICAL_BANDS

        try:
            # Mask to sloped pixels with positive IC and valid NIR.
            # Threshold is 0 rather than -0.1 so it works correctly for both:
            #   - Landsat / S2 scaled reflectance (0.0–1.0): excludes nodata fill
            #   - S2 raw DN (0–10000): excludes only DN=0 (invalid/fill pixels)
            # Using -0.1 would be a no-op on raw DN — every valid pixel > -0.1.
            mask = (
                image.select('slope').gte(slope_threshold)
                .And(image.select('IC').gte(0))
                .And(image.select('NIR').gt(0))
            )
            img_masked = ee.Image(image.updateMask(mask))

            # Geometry slightly inset to avoid edge noise in regression
            reg_geometry = ee.Geometry(image.geometry().buffer(-100, 10))

            # --- Single-pass multi-band SCSc regression ----------------------
            # Fit all bands in a single reduceRegion call (X = [1, IC], Y = band_list)
            # to avoid triggering "Too many concurrent aggregations" across collections.
            # Sampling at reg_scale >= 300m (~134k pixels per S2 tile) accurately
            # estimates the two scalar linear-fit parameters (a, b) while the
            # pixel-level SCSc correction below is applied at full native resolution.
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

            # Fallback 2xN array (b=0, a=1 -> c=0) when all pixels in scene are masked/flat
            default_coeffs = ee.Array([[0.0] * len(band_list), [1.0] * len(band_list)])
            coeffs = ee.Array(
                ee.List([lr.get('coefficients'), default_coeffs]).reduce(ee.Reducer.firstNonNull())
            )
            coeffs = ee.Array(
                ee.Algorithms.If(coeffs.length().get([0]).eq(2), coeffs, default_coeffs)
            )

            # Row 0: intercept (b), Row 1: slope (a); c = b / a
            b_arr  = coeffs.slice(0, 0, 1).project([1])
            a_arr  = coeffs.slice(0, 1, 2).project([1])
            a_safe = a_arr.add(a_arr.eq(0))
            c_arr  = b_arr.divide(a_safe).multiply(a_arr.neq(0))

            # Build multi-band constant image with statically known band names
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

            # Fill flat / negative-IC pixels with original reflectance
            corrected_filled = corrected_bands.unmask(image.select(band_list))

            # Overwrite corrected bands on original image and drop helper IC bands
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
    # ----------------------------------------------------------------
    def correct_image(self, image: ee.Image, dem: Optional[ee.Image] = None, scale: int = 30,
        band_list: Optional[List[str]] = None,
        slope_threshold: float = 5.0,
        buffer_m: int = 10_000,
        verbose: bool = True,
    ) -> ee.Image:
        """
        Apply SCSc topographic correction to a single image in one call.

        Runs ``compute_illumination_condition`` followed by
        ``apply_scsc_correction`` and returns the corrected image.

        Parameters
        ----------
        image : ee.Image
            Pre-processed image with standardised band names (output of
            ``rename_landsat_bands`` or ``rename_s2_bands``).  Must carry
            solar angle properties (Landsat C2 ``'SUN_ELEVATION'`` / ``'SUN_AZIMUTH'``,
            Landsat C1 ``'SOLAR_ZENITH_ANGLE'`` / ``'SOLAR_AZIMUTH_ANGLE'``, or
            Sentinel-2 ``'MEAN_SOLAR_ZENITH_ANGLE'`` / ``'MEAN_SOLAR_AZIMUTH_ANGLE'``).
        dem : ee.Image, optional
            Digital elevation model covering the image footprint.
            Defaults to ``NASA/NASADEM_HGT/001`` (30 m global).
            Override with a higher-resolution or region-specific DEM when needed.
        scale : int, optional
            Pixel resolution in metres. ``30`` for Landsat, ``10`` for
            Sentinel-2. Default: ``30``.
        band_list : list of str, optional
            Optical bands to correct. Defaults to
            ``['BLUE','GREEN','RED','NIR','SWIR1','SWIR2']``.
        slope_threshold : float, optional
            Minimum slope (degrees) for correction to be applied. Default: ``5.0``.
        buffer_m : int, optional
            DEM buffer around the image geometry (metres). Default: ``10 000``.
        verbose : bool, optional
            Log progress messages. Default: ``True``.

        Returns
        -------
        ee.Image
            Topographically corrected image with original band structure.

        Example
        -------
        >>> tc = Topographic_Correction()
        >>> # Uses NASADEM by default
        >>> corrected = tc.correct_image(image, scale=30)
        >>> # Override with a custom DEM
        >>> custom_dem = ee.Image('JAXA/ALOS/AW3D30/V3_2').select('DSM')
        >>> corrected = tc.correct_image(image, dem=custom_dem, scale=30)
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

        # Overwrite band_list directly on input image so band names are statically known
        return ee.Image(image.addBands(corrected.select(band_list), band_list, True))

    # ------------------------------------------------------------------
    # Step 4: Collection-level wrapper
    # ------------------------------------------------------------------

    def correct_collection(self, collection: ee.ImageCollection, dem: Optional[ee.Image] = None, scale: int = 30,
        band_list: Optional[List[str]] = None,
        slope_threshold: float = 5.0,
        buffer_m: int = 10_000,
        verbose: bool = True,
    ) -> ee.ImageCollection:
        """
        Apply SCSc topographic correction to every image in a collection.

        Maps ``correct_image`` over the collection. Because GEE evaluates
        lazily, no computation is triggered until the collection is consumed
        (e.g. by a reducer or export).

        Parameters
        ----------
        collection : ee.ImageCollection
            Filtered and pre-processed image collection (output of
            ``Reflectance_Data.get_optical_data`` or ``get_s2_optical_data``).
        dem : ee.Image, optional
            Digital elevation model covering the full collection footprint.
            Defaults to ``NASA/NASADEM_HGT/001`` (30 m global NASADEM).
            Override with a higher-resolution or region-specific DEM when needed
            (e.g. ``ee.Image('JAXA/ALOS/AW3D30/V3_2').select('DSM')``).
        scale : int, optional
            Pixel resolution in metres. ``30`` for Landsat, ``10`` for
            Sentinel-2. Default: ``30``.
        band_list : list of str, optional
            Optical bands to correct. Defaults to
            ``['BLUE','GREEN','RED','NIR','SWIR1','SWIR2']``.
        slope_threshold : float, optional
            Minimum slope (degrees) for a pixel to be included in the regression
            and receive a correction. Default: ``5.0``.
        buffer_m : int, optional
            DEM buffer around each image geometry in metres. Default: ``10 000``.
        verbose : bool, optional
            Log collection size and progress. Default: ``True``.

        Returns
        -------
        ee.ImageCollection
            Collection with every image topographically corrected.

        Example
        -------
        >>> from luma_ge.data_acquisition import Reflectance_Data, final_Image
        >>> from luma_ge.imagery_correction import Topographic_Correction
        >>>
        >>> rd = Reflectance_Data()
        >>> tc = Topographic_Correction()
        >>>
        >>> # 1. Acquire
        >>> collection, stats = rd.get_optical_data(aoi, 2020, 2023, 'L8_SR')
        >>>
        >>> # 2. Correct — NASADEM used by default, no dem= argument needed
        >>> corrected_col = tc.correct_collection(collection, scale=30)
        >>>
        >>> # 2b. Override with a custom DEM
        >>> custom_dem = ee.Image('JAXA/ALOS/AW3D30/V3_2').select('DSM')
        >>> corrected_col = tc.correct_collection(collection, dem=custom_dem, scale=30)
        >>>
        >>> # 3. Composite
        >>> result = final_Image().get_temporal_composite(corrected_col, aoi)
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
            # Overwrite band_list directly on img so band names and properties remain static
            return ee.Image(img.addBands(corrected.select(band_list), band_list, True))

        corrected = collection.map(_correct)

        if verbose:
            self.logger.info("SCSc correction mapped over collection.")

        return corrected
