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

# ---------------------------------------------------------------------------
# Module 1.5: Imagery Correction
## System Response 1.5.1 – Topographic Correction (SCSc)
# ---------------------------------------------------------------------------

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
    >>> dem = ee.Image('NASA/NASADEM_HGT/001').select('elevation')
    >>>
    >>> # Correct a single image (Landsat, 30 m)
    >>> corrected_img = tc.correct_image(image, dem, scale=30)
    >>>
    >>> # Correct an entire collection (Sentinel-2, 10 m)
    >>> corrected_col = tc.correct_collection(collection, dem, scale=10)
    """

    # Bands that receive topographic correction.
    # Landsat 4-9 SR and Sentinel-2 share these standardised
    
    DEFAULT_OPTICAL_BANDS: List[str] = [
        'BLUE', 'GREEN', 'RED', 'NIR', 'SWIR1', 'SWIR2'
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
    # Step 1: Illumination condition
    # ------------------------------------------------------------------

    def compute_illumination_condition(
        self,
        image: ee.Image,
        dem: ee.Image,
        buffer_m: int = 10_000,
    ) -> ee.Image:
        """
        Compute the illumination condition (IC) and append it to the image.

        The IC represents the cosine of the angle between the solar illumination
        vector and the terrain surface normal. It is computed as:

            IC = cos(Z)·cos(S) + sin(Z)·sin(S)·cos(φ_sun − φ_aspect)

        where Z is the solar zenith, S is the terrain slope, and φ denotes azimuths.

        Parameters
        ----------
        image : ee.Image
            Single image carrying ``'SOLAR_ZENITH_ANGLE'`` and
            ``'SOLAR_AZIMUTH_ANGLE'`` properties (in degrees). Present on all
            Landsat Collection-2 and Sentinel-2 images.
        dem : ee.Image
            Digital elevation model covering the image footprint (e.g.
            ``ee.Image('NASA/NASADEM_HGT/001').select('elevation')``).
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
        >>> dem = ee.Image('NASA/NASADEM_HGT/001').select('elevation')
        >>> img_with_ic = tc.compute_illumination_condition(image, dem)
        """
        try:
            buffer_geom = image.geometry().buffer(buffer_m)

            # --- Solar geometry: degrees to radians (server-side constants) ---
            sz_rad = (
                ee.Image.constant(ee.Number(image.get('SOLAR_ZENITH_ANGLE')))
                .multiply(math.pi / 180)
                .clip(buffer_geom)
            )
            sa_rad = (
                ee.Image.constant(
                    ee.Number(image.get('SOLAR_AZIMUTH_ANGLE'))
                    .multiply(math.pi / 180)
                )
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
    # Step 2: SCSc correction
    # ------------------------------------------------------------------

    def apply_scsc_correction(
        self,
        image: ee.Image,
        scale: int = 30,
        band_list: Optional[List[str]] = None,
        slope_threshold: float = 5.0,
        max_pixels: int = 1_000_000_000,
    ) -> ee.Image:
        """
        Apply the SCSc topographic correction to an image that already carries
        the IC, cosZ, cosS, and slope bands (output of
        ``compute_illumination_condition``).

        For each band *b* in ``band_list`` the correction is:

            c  = b / a                   (regression offset / slope)
            ρ* = ρ · (cosZ·cosS + c) / (IC + c)

        where *a* and *b* are the slope and intercept of the linear regression
        between IC and ρ, computed from pixels where slope ≥ ``slope_threshold``
        and IC ≥ 0.

        Flat-terrain pixels (slope < threshold) and pixels with negative IC are
        filled with the original uncorrected reflectance so the output retains
        full spatial coverage.

        Parameters
        ----------
        image : ee.Image
            Image output of ``compute_illumination_condition`` — must carry
            ``'IC'``, ``'cosZ'``, ``'cosS'``, and ``'slope'`` bands alongside
            the optical reflectance bands.
        scale : int, optional
            Pixel resolution (metres) used for the ``reduceRegion`` regression.
            Use ``30`` for Landsat, ``10`` for Sentinel-2. Default: ``30``.
        band_list : list of str, optional
            Bands to correct. Defaults to ``DEFAULT_OPTICAL_BANDS``
            (``['BLUE','GREEN','RED','NIR','SWIR1','SWIR2']``).
        slope_threshold : float, optional
            Minimum slope (degrees) for a pixel to be included in the
            regression and correction. Default: ``5.0``.
        max_pixels : int, optional
            ``maxPixels`` argument passed to ``reduceRegion``. Default: ``1e9``.

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
            props = image.toDictionary()
            st    = image.get('system:time_start')

            # Mask to sloped pixels with positive IC and valid NIR
            mask = (
                image.select('slope').gte(slope_threshold)
                .And(image.select('IC').gte(0))
                .And(image.select('NIR').gt(-0.1))
            )
            img_masked = ee.Image(image.updateMask(mask))

            # Geometry slightly inset to avoid edge noise in regression
            reg_geometry = ee.Geometry(image.geometry().buffer(-100))

            # --- Per-band SCSc correction (Python loop — avoids ee.List overhead) ---
            def _scsc_band(band: str) -> ee.Image:
                regression = img_masked.select('IC', band).reduceRegion(
                    reducer   = ee.Reducer.linearFit(),
                    geometry  = reg_geometry,
                    scale     = scale,
                    maxPixels = max_pixels,
                )
                a = ee.Number(regression.get('scale'))    # slope
                b = ee.Number(regression.get('offset'))   # intercept
                c = b.divide(a)                           # empirical c-value

                return img_masked.expression(
                    '(image * (cosB * cosZ + c)) / (ic + c)',
                    {
                        'image': img_masked.select(band),
                        'ic'   : img_masked.select('IC'),
                        'cosB' : img_masked.select('cosS'),
                        'cosZ' : img_masked.select('cosZ'),
                        'c'    : c,
                    },
                ).rename(band)

            corrected_bands = ee.Image([_scsc_band(b) for b in band_list])

            # Add IC band so unmask can fill from original IC-band image
            corrected_with_ic = corrected_bands.addBands(image.select('IC'))
            fill_bands = band_list + ['IC']

            # Fill flat / negative-IC pixels with original reflectance
            corrected_filled = (
                corrected_with_ic
                .unmask(image.select(fill_bands))
                .select(band_list)                  # drop IC from final output
            )

            # Pass through non-corrected bands (thermal, QA, extra indices, etc.)
            all_bands     = image.bandNames()
            ancillary_names = (
                all_bands
                .removeAll(ee.List(band_list))
                .removeAll(ee.List(['IC', 'cosZ', 'cosS', 'slope']))
            )
            ancillary = image.select(ancillary_names)

            return (
                corrected_filled
                .addBands(ancillary)
                .setMulti(props)
                .set('system:time_start', st)
            )

        except ee.EEException as e:
            self.logger.error(f"EEException in apply_scsc_correction: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error in apply_scsc_correction: {e}")
            raise

    # ------------------------------------------------------------------
    # Step 3: Single-image convenience wrapper
    # ------------------------------------------------------------------

    def correct_image(
        self,
        image: ee.Image,
        dem: ee.Image,
        scale: int = 30,
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
            ``'SOLAR_ZENITH_ANGLE'`` and ``'SOLAR_AZIMUTH_ANGLE'`` properties.
        dem : ee.Image
            Digital elevation model covering the image footprint.
        scale : int, optional
            Pixel resolution in metres. ``30`` for Landsat, ``10`` for
            Sentinel-2. Default: ``30``.
        band_list : list of str, optional
            Optical bands to correct. Defaults to
            ``['BLUE','GREEN','RED','NIR','SWIR1','SWIR2']``.
        slope_threshold : float, optional
            Minimum slope (°) for correction to be applied. Default: ``5.0``.
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
        >>> dem = ee.Image('NASA/NASADEM_HGT/001').select('elevation')
        >>> corrected = tc.correct_image(image, dem, scale=30)
        """
        if verbose:
            self.logger.info(
                f"Applying SCSc topographic correction "
                f"(scale={scale}m, slope_threshold={slope_threshold}°)"
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

        return corrected

    # ------------------------------------------------------------------
    # Step 4: Collection-level wrapper
    # ------------------------------------------------------------------

    def correct_collection(
        self,
        collection: ee.ImageCollection,
        dem: ee.Image,
        scale: int = 30,
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
        dem : ee.Image
            Digital elevation model covering the full collection footprint.
            ``ee.Image('NASA/NASADEM_HGT/001').select('elevation')`` is a
            reliable global choice at 30 m resolution.
        scale : int, optional
            Pixel resolution in metres. ``30`` for Landsat, ``10`` for
            Sentinel-2. Default: ``30``.
        band_list : list of str, optional
            Optical bands to correct. Defaults to
            ``['BLUE','GREEN','RED','NIR','SWIR1','SWIR2']``.
        slope_threshold : float, optional
            Minimum slope (°) for a pixel to be included in the regression
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
        >>> dem = ee.Image('NASA/NASADEM_HGT/001').select('elevation')
        >>>
        >>> # 1. Acquire
        >>> collection, stats = rd.get_optical_data(aoi, 2020, 2023, 'L8_SR')
        >>>
        >>> # 2. Correct
        >>> corrected_col = tc.correct_collection(collection, dem, scale=30)
        >>>
        >>> # 3. Composite
        >>> result = final_Image().get_temporal_composite(corrected_col, aoi)
        """
        if verbose:
            size = collection.size().getInfo()
            self.logger.info(
                f"Applying SCSc topographic correction to {size} images "
                f"(scale={scale}m, slope_threshold={slope_threshold}°)"
            )

        def _correct(img: ee.Image) -> ee.Image:
            img_with_ic = self.compute_illumination_condition(img, dem, buffer_m=buffer_m)
            return self.apply_scsc_correction(
                img_with_ic,
                scale           = scale,
                band_list       = band_list,
                slope_threshold = slope_threshold,
            )

        corrected = collection.map(_correct)

        if verbose:
            self.logger.info("SCSc correction mapped over collection.")

        return corrected
