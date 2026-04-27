"""
1. Module Helper
This file is design to incorporate features and function by multiple modules.
    1.1 Earth Engine Export Managaer
    Provides centralized export functionality for Earth Engine images via the
    EE_Export_Manager class. Supports three export methods:
    - direct: Synchronous export using getDownloadURL()
    - gcs:    Asynchronous export to Google Cloud Storage
    - oauth2: Asynchronous export to Google Drive (in development)

    The class handles parameter validation, geometry extraction, name sanitization,
    and task monitoring with caching to minimise Earth Engine API calls.

    Usage:
        from modules.ee_export_manager import EE_Export_Manager

        manager = EE_Export_Manager()
        result  = manager.export_image(image, aoi, "gcs", "my_export", 30, bucket="my-bucket")
        tasks   = manager.get_active_tasks()
"""
import datetime
from typing import Any, Dict, List, Optional, Union
import ee
from .ee_config import ensure_ee_initialized

#Module-level constants
DEFAULT_CACHE_TTL   = 30    #seconds
MAX_PIXELS          = 1e13  #maximum pixels for large-area exports
DEFAULT_CRS         = "EPSG:4326" #Default Coordinate Reference System (CRS) set to WGS 1984
DEFAULT_FILE_FORMAT = "GeoTIFF"

# ---------------------------------------------------------------------------
#Earth Engine Export Manager
class EE_Export_Manager:
    """
    Manages Earth Engine image exports and task tracking.

    Instance state replaces the streamlit's session_state pattern:
        self.export_tasks       – list of task_info dicts
        self.task_cache         – {task_id: status_dict}
        self.last_cache_update  – {task_id: datetime}

    """

    def __init__(self):
        self.export_tasks: List[Dict[str, Any]]       = []
        self.task_cache: Dict[str, Any]               = {}
        self.last_cache_update: Dict[str, datetime.datetime] = {}

    #Utility helpers
    #standarized export name for consistent default naming
    @staticmethod
    def sanitize_export_name(name: str) -> str:
        """
        Sanitize export name for file-system compatibility.

        Replaces spaces with underscores and removes characters that are
        invalid for GCS, Drive, and local file systems.

        Args:
            name: Original export name.

        Returns:
            Sanitized name. Returns ``"export"`` for empty / blank input.

        Examples:
            >>> EE_Export_Manager.sanitize_export_name("My Export 2024")
            'My_Export_2024'
            >>> EE_Export_Manager.sanitize_export_name("file/with:special*chars?")
            'file_with_special_chars_'
            >>> EE_Export_Manager.sanitize_export_name("")
            'export'
        """
        if not name or not name.strip():
            return "export"

        sanitized = name.replace(" ", "_")
        for char in ['/', '\\', ':', '*', '?', '"', '<', '>', '|']:
            sanitized = sanitized.replace(char, "_")

        if not sanitized.strip("_"):
            return "export"
        return sanitized

    @staticmethod
    #get the geometry from AOI. 
    #support ee geometry, feature, and feature collection
    def extract_geometry(
        aoi: Union[ee.FeatureCollection, ee.Feature, ee.Geometry]
    ) -> Union[ee.Geometry, Dict[str, Any]]:
        """
        Extract ``ee.Geometry`` from various AOI types.

        Args:
            aoi: An ``ee.Geometry``, ``ee.Feature``, or ``ee.FeatureCollection``.

        Returns:
            ``ee.Geometry`` on success, or ``{"success": False, "error": "…"}``
            on failure.
        """
        try:
            if isinstance(aoi, ee.Geometry):
                return aoi
            if isinstance(aoi, ee.Feature):
                return aoi.geometry()
            if isinstance(aoi, ee.FeatureCollection):
                return aoi.geometry()
            return {
                "success": False,
                "error": f"Cannot extract geometry from AOI of type: {type(aoi)}"
            }
        except Exception as e:
            return {"success": False, "error": f"Error extracting geometry: {str(e)}"}

    #Input data validation
    @staticmethod
    def _validate_image(image: Any) -> Optional[str]:
        if not isinstance(image, ee.Image):
            return f"Image must be an ee.Image object, got: {type(image)}"
        return None

    @staticmethod
    def _validate_scale(scale: Any) -> Optional[str]:
        try:
            if float(scale) <= 0:
                return f"Scale must be a positive number, got: {scale}"
            return None
        except (TypeError, ValueError):
            return f"Scale must be a number, got: {type(scale).__name__}"

    @staticmethod
    def _validate_crs(crs: str) -> Optional[str]:
        if not isinstance(crs, str):
            return f"CRS must be a string, got: {type(crs).__name__}"
        if not crs.strip():
            return "CRS cannot be empty"
        crs_upper = crs.upper()
        if not any(crs_upper.startswith(p) for p in ["EPSG:", "ESRI:", "SR-ORG:"]):
            return (
                f"Invalid CRS format: '{crs}'. "
                "Expected format like 'EPSG:4326', 'ESRI:54009', or 'SR-ORG:6864'"
            )
        parts = crs.split(":")
        if len(parts) != 2 or not parts[1].strip():
            return f"Invalid CRS format: '{crs}'. Expected format like 'EPSG:4326'"
        try:
            int(parts[1])
        except ValueError:
            return f"Invalid CRS format: '{crs}'. Code after ':' must be numeric"
        return None

    @staticmethod
    def _validate_bucket(bucket: Optional[str], export_method: str) -> Optional[str]:
        if export_method == "gcs":
            if not bucket:
                return "Bucket name is required for GCS export"
            if not isinstance(bucket, str) or not bucket.strip():
                return "Bucket name must be a non-empty string"
        return None

    def _validate_export_parameters(
        self,
        image: Any,
        scale: Any,
        crs: str,
        bucket: Optional[str],
        export_method: str,
    ) -> Optional[str]:
        return (
            self._validate_image(image)
            or self._validate_scale(scale)
            or self._validate_crs(crs)
            or self._validate_bucket(bucket, export_method)
        )

    #Format / scale helpers (private)
    #dedicated for module 5 only
    @staticmethod
    def _predictor_options(export_type: Optional[str], scale: float) -> Dict[str, Any]:
        """Return format_options and adjusted scale for a given export type."""
        format_options   = {"cloudOptimized": True}
        adjusted_scale   = scale

        if export_type == "terrain":
            if scale > 100:
                adjusted_scale = 30
        elif export_type == "stacked_predictors":
            if scale != 30:
                adjusted_scale = 30
        elif export_type == "classification":
            format_options["noData"] = 0
        # spectral_index and others: use defaults
        return {"format_options": format_options, "adjusted_scale": adjusted_scale}

    #Export helpers
    #direct download export
    def _export_direct_download(
        self,
        image: ee.Image,
        geometry: ee.Geometry,
        sanitized_name: str,
        scale: float,
        crs: str,
        format_options: Dict[str, Any],
        file_per_band: bool = False,
    ) -> Dict[str, Any]:
        try:
            url = image.getDownloadURL({
                "name":          sanitized_name,
                "crs":           crs,
                "scale":         scale,
                "region":        geometry,
                "fileFormat":    DEFAULT_FILE_FORMAT,
                "formatOptions": format_options,
                "filePerBand":   file_per_band,
            })
            return {
                "success": True, "method": "direct",
                "url": url, "name": sanitized_name,
                "format": DEFAULT_FILE_FORMAT, "crs": crs, "scale": scale,
            }
        except Exception as e:
            msg = str(e).lower()
            if any(k in msg for k in ("too large", "size", "limit")):
                return {
                    "success": False, "method": "direct",
                    "error": "Direct download failed: Image too large. "
                             "Try reducing the area or use Google Cloud Storage export.",
                }
            return {"success": False, "method": "direct", "error": f"Direct download failed: {e}"}
    #google cloud storage export
    #not fully operational
    def _export_gcs(
        self,
        image: ee.Image,
        geometry: ee.Geometry,
        sanitized_name: str,
        scale: float,
        crs: str,
        bucket: str,
        path_prefix: str,
        format_options: Dict[str, Any],
        export_type: Optional[str],
        original_name: str,
    ) -> Dict[str, Any]:
        try:
            task = ee.batch.Export.image.toCloudStorage(
                image          = image,
                description    = sanitized_name,
                bucket         = bucket,
                fileNamePrefix = f"{path_prefix}{sanitized_name}" if path_prefix else sanitized_name,
                scale          = scale,
                crs            = crs,
                region         = geometry,
                maxPixels      = MAX_PIXELS,
                fileFormat     = DEFAULT_FILE_FORMAT,
                formatOptions  = format_options,
            )
            task.start()
            task_info = {
                "id":            task.id,
                "name":          original_name,
                "destination":   "gcs",
                "folder":        bucket,
                "crs":           crs,
                "scale":         scale,
                "start_time":    datetime.datetime.now(),
                "last_progress": 0,
                "last_update":   datetime.datetime.now(),
            }
            if export_type:
                task_info["type"] = export_type
            return {"success": True, "method": "gcs", "task_info": task_info}
        except Exception as e:
            return {"success": False, "method": "gcs", "error": f"GCS export failed: {e}"}
    #Google drive export using OAuth2 authentication
    #the user needs to login and authenticate the app to access their drive. This is a placeholder for the full OAuth2 flow.
    #not yet operational
    def _export_oauth2(
        self,
        image: ee.Image,
        geometry: ee.Geometry,
        sanitized_name: str,
        scale: float,
        crs: str,
        format_options: Dict[str, Any],
        export_type: Optional[str],
        original_name: str,
    ) -> Dict[str, Any]:
        # TODO: full OAuth2 authentication flow
        try:
            task = ee.batch.Export.image.toDrive(
                image          = image,
                description    = sanitized_name,
                scale          = scale,
                crs            = crs,
                region         = geometry,
                maxPixels      = MAX_PIXELS,
                fileFormat     = DEFAULT_FILE_FORMAT,
                formatOptions  = format_options,
            )
            task.start()
            task_info = {
                "id":            task.id,
                "name":          original_name,
                "destination":   "oauth2",
                "folder":        "Drive",
                "crs":           crs,
                "scale":         scale,
                "start_time":    datetime.datetime.now(),
                "last_progress": 0,
                "last_update":   datetime.datetime.now(),
            }
            if export_type:
                task_info["type"] = export_type
            return {"success": True, "method": "oauth2", "task_info": task_info}
        except Exception as e:
            msg = str(e).lower()
            if any(k in msg for k in ("auth", "credential", "permission")):
                return {
                    "success": False, "method": "oauth2",
                    "error": "OAuth2 authentication failed: Invalid credentials. Please re-authenticate.",
                }
            return {"success": False, "method": "oauth2", "error": f"OAuth2 export failed: {e}"}

    # -----------------------------------------------------------------------
    # Public export API
    def export_image(
        self,
        image: ee.Image,
        region: Union[ee.FeatureCollection, ee.Feature, ee.Geometry],
        export_method: str,
        name: str,
        scale: float,
        crs: str = DEFAULT_CRS,
        bucket: Optional[str] = None,
        path_prefix: Optional[str] = "",
        format_options: Optional[Dict] = None,
        export_type: Optional[str] = None,
        file_per_band: bool = True,
    ) -> Dict[str, Any]:
        """
        Export an Earth Engine image.

        Args:
            image:         ``ee.Image`` to export.
            region:        Area of interest.
            export_method: ``"direct"``, ``"gcs"``, or ``"oauth2"``.
            name:          Export name (will be sanitized).
            scale:         Resolution in metres.
            crs:           Coordinate reference system (default ``EPSG:4326``).
            bucket:        GCS bucket (required for ``"gcs"``).
            path_prefix:   GCS path prefix.
            format_options: Extra format options merged with defaults.
            export_type:   Optional label (``"terrain"``, ``"spectral_index"``,
                           ``"stacked_predictors"``, ``"classification"``).
            file_per_band: Separate file per band for direct downloads.

        Returns:
            ``{"success": True,  "method": …, "url": …}``          (direct)
            ``{"success": True,  "method": …, "task_info": …}``     (async)
            ``{"success": False, "method": …, "error": …}``         (any failure)
        """
        try:
            err = self._validate_export_parameters(image, scale, crs, bucket, export_method)
            if err:
                return {"success": False, "method": export_method, "error": err}

            geom = self.extract_geometry(region)
            if isinstance(geom, dict) and not geom.get("success", True):
                return {"success": False, "method": export_method, "error": geom["error"]}

            sanitized_name = self.sanitize_export_name(name)
            opts           = self._predictor_options(export_type, scale)
            adj_scale      = opts["adjusted_scale"]
            base_fmt       = opts["format_options"]
            final_fmt      = {**base_fmt, **(format_options or {})}

            if export_method == "direct":
                return self._export_direct_download(
                    image, geom, sanitized_name, adj_scale, crs, final_fmt, file_per_band
                )
            elif export_method == "gcs":
                return self._export_gcs(
                    image, geom, sanitized_name, adj_scale, crs,
                    bucket, path_prefix, final_fmt, export_type, name
                )
            elif export_method == "oauth2":
                return self._export_oauth2(
                    image, geom, sanitized_name, adj_scale, crs,
                    final_fmt, export_type, name
                )
            else:
                return {
                    "success": False,
                    "method":  export_method,
                    "error":   (
                        f"Unsupported export method: '{export_method}'. "
                        "Supported methods: 'direct', 'gcs', 'oauth2'"
                    ),
                }
        except Exception as e:
            return {"success": False, "method": export_method, "error": f"Export failed: {e}"}

    #Task status monitoring
    #simple task monitoring for exports
    #More comprehensive monitoring is implemented in earth engine task export
    def get_task_status(
        self,
        task_id: str,
        cache_ttl: int = DEFAULT_CACHE_TTL,
    ) -> Optional[Dict[str, Any]]:
        """
        Return task status, using a cache to reduce EE API calls.

        Terminal states (COMPLETED, FAILED, CANCELLED) are cached indefinitely.
        Active states (READY, RUNNING) are cached for ``cache_ttl`` seconds.
        Returns ``None`` on API error.
        """
        now = datetime.datetime.now()

        if task_id in self.task_cache and task_id in self.last_cache_update:
            cached = self.task_cache[task_id]
            if cached.get("state") in ("COMPLETED", "FAILED", "CANCELLED"):
                return cached
            if (now - self.last_cache_update[task_id]).total_seconds() < cache_ttl:
                return cached

        try:
            status = ee.data.getTaskStatus(task_id)[0]
            self.task_cache[task_id]        = status
            self.last_cache_update[task_id] = now
            return status
        except Exception:
            return None

    def get_active_tasks(self) -> List[Dict[str, Any]]:
        """
        Return task_info dicts for tasks in READY or RUNNING state.
        """
        active = []
        for task_info in self.export_tasks:
            task_id = task_info.get("id")
            if not task_id:
                continue
            status = self.get_task_status(task_id)
            if status is None:
                continue
            state = status.get("state", "").upper()
            if state in ("READY", "RUNNING"):
                entry = task_info.copy()
                entry["current_state"]    = state
                entry["current_progress"] = status.get("progress", 0)
                entry["last_update"]      = datetime.datetime.now()
                active.append(entry)
        return active

    # -----------------------------------------------------------------------
    # Task management
    # -----------------------------------------------------------------------

    def remove_task(self, task_id: str) -> bool:
        """
        Remove a task from local tracking (does not cancel it in EE).

        Returns ``True`` if the task was found and removed, ``False`` otherwise.
        """
        before = len(self.export_tasks)
        self.export_tasks = [t for t in self.export_tasks if t.get("id") != task_id]
        return len(self.export_tasks) < before

    def remove_completed_tasks(self) -> int:
        """
        Remove all tasks in terminal states (COMPLETED, FAILED, CANCELLED).

        Returns the number of tasks removed.
        """
        terminal = {"COMPLETED", "FAILED", "CANCELLED"}
        keep     = []
        for task_info in self.export_tasks:
            task_id = task_info.get("id")
            if not task_id:
                keep.append(task_info)
                continue
            status = self.get_task_status(task_id)
            if status is None or status.get("state", "").upper() not in terminal:
                keep.append(task_info)
        removed           = len(self.export_tasks) - len(keep)
        self.export_tasks = keep
        return removed
