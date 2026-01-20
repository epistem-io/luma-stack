import streamlit as st
import geopandas as gpd
import pandas as pd
import json
import ee 
import geemap
from .ee_config import ensure_ee_initialized
from shapely.validation import make_valid
from shapely.geometry import MultiPoint

# Do not initialize Earth Engine at import time. Initialize when classes are instantiated.

#Based on early experiments, shapefile with complex geometry often cause issues in GEE
#The following functions are used to handle the common geometry issues

# Module 1: Cloudless Image Mosaic
## System Response 1.1: Area of Interest Definition

#1. Shapefile Validation. Validate common shapefile issues, such as too complex geometry, null geom, or invalid CRS
class shapefile_validator:
    """
    Handle shapefile validation from user upload. Used in module 1 for AOI, and module 3 for ROI and 7 for thematic accuracy
    Designed for point, multipoint, polygon, and multipolygon
    """
    def __init__(self, verbose=True):
        """
        Initialize the validator
        Ensure Earth Engine is initialized lazily (avoids import-time failures).
        """
        # Ensure Earth Engine is initialized when first used (raises helpful error if not)
        ensure_ee_initialized()
        
        self.verbose = verbose
    def log(self, message, level = "info"):
        if self.verbose:
            if level == "error":
                st.error(message)
            elif level == "warning":
                st.warning(message)
            elif level == "success":
                st.success(message)
            else:
                st.info(message)
        #Core validation and fixing functions
    def validate_and_fix_geometry(self, gdf, geometry = "mixed"):
        """
        Validate and fix geometry issues
        """
        self.log("Validating geometry...")
        #Checking the basic shapefile info
        self.log(f"Original CRS: {gdf.crs}")
        self.log(f"Number of features: {len(gdf)}")
        self.log(f"Geometry types: {gdf.geometry.geom_type.unique()}")
        #handle coordinate reference system (CRS) conversion
        gdf = self._fix_crs(gdf)
        if gdf is None:
            return None
        # Remove invalid, empty, and null geometries
        gdf = self._clean_geometries(gdf)
        if gdf is None or len(gdf) == 0:
            return None
        # Geometry-specific validation
        if geometry in ["point", "mixed"]:
            gdf = self._validate_points(gdf)
        if geometry in ["polygon", "mixed"]:
            gdf = self._validate_polygons(gdf)
        # Final validation
        if not self._final_validation(gdf):
            return None
        self.log("Geometry validation completed!", "success")
        self.log(f"Final geometry types: {gdf.geometry.geom_type.value_counts().to_dict()}")
        self.log(f"Valid features: {len(gdf)}")
        return gdf
    #All shapefile should be WGS 1984 (EPSG:4326)
    def _fix_crs(self, gdf):
        """Fix coordinate reference system issues"""
        if gdf.crs is None:
            self.log("No CRS information found. Assuming WGS84...", "warning")
            gdf = gdf.set_crs('EPSG:4326')
        elif gdf.crs != 'EPSG:4326':
            self.log(f"Converting from {gdf.crs} to EPSG:4326 (WGS84)")
            try:
                gdf = gdf.to_crs('EPSG:4326')
                self.log("CRS conversion successful!", "success")
            except Exception as e:
                self.log(f"CRS conversion failed: {e}", "error")
                return None
        return gdf
     #Function to support the validation (validate_and_fix_geometry)
    def _clean_geometries(self, gdf):
        """Remove and fix invalid geometries"""
        original_count = len(gdf)
        
        #Fix invalid geometries
        invalid_mask = ~gdf.geometry.is_valid
        if invalid_mask.any():
            self.log(f"Found {invalid_mask.sum()} invalid geometries. Fixing...", "warning")
            gdf.loc[invalid_mask, 'geometry'] = gdf.loc[invalid_mask, 'geometry'].apply(make_valid)
        
        #Remove empty geometries
        empty_mask = gdf.geometry.is_empty
        if empty_mask.any():
            self.log(f"Found {empty_mask.sum()} empty geometries. Removing...", "warning")
            gdf = gdf[~empty_mask].copy()
        
        #Remove null geometries
        null_mask = gdf.geometry.isnull()
        if null_mask.any():
            self.log(f"Found {null_mask.sum()} null geometries. Removing...", "warning")
            gdf = gdf[~null_mask].copy()
        if len(gdf) < original_count:
            gdf = gdf.reset_index(drop=True)
            self.log(f"Removed {original_count - len(gdf)} invalid geometries")
            
        return gdf
     #Function to support the validation (validate_and_fix_geometry). For point data
    def _validate_points(self, gdf):
        """Validate point geometries"""
        point_mask = gdf.geometry.geom_type.isin(['Point', 'MultiPoint'])
        if not point_mask.any():
            return gdf
            
        self.log("Validating point coordinates...")
        
        indices_to_remove = []
        #fixed and validate geometry for every single point data
        for idx in gdf[point_mask].index:
            geom = gdf.loc[idx, 'geometry']
            
            if geom.geom_type == 'Point':
                if not self._is_valid_coordinate(geom.x, geom.y):
                    self.log(f"Invalid coordinates at index {idx}: ({geom.x}, {geom.y})", "warning")
                    indices_to_remove.append(idx)
                    
            elif geom.geom_type == 'MultiPoint':
                valid_points = []
                for pt in geom.geoms:
                    if self._is_valid_coordinate(pt.x, pt.y):
                        valid_points.append(pt)
                    else:
                        self.log(f"Invalid coordinates in MultiPoint at index {idx}: ({pt.x}, {pt.y})", "warning")
                
                if valid_points:
                    gdf.loc[idx, 'geometry'] = MultiPoint(valid_points)
                else:
                    indices_to_remove.append(idx)
        
        if indices_to_remove:
            gdf = gdf.drop(indices_to_remove).reset_index(drop=True)
            self.log(f"Removed {len(indices_to_remove)} features with invalid coordinates")
            
        return gdf
    #Function to support the validation (validate_and_fix_geometry). For polygon data. Complex polygon shape (with a lot of vertex, will be simplified here)
    def _validate_polygons(self, gdf):
        """Validate and simplify polygon geometries"""
        poly_mask = gdf.geometry.geom_type.isin(['Polygon', 'MultiPolygon'])
        if not poly_mask.any():
            return gdf
            
        self.log("Checking polygon complexity...")
        #validate the polygon for each ID
        for idx in gdf[poly_mask].index:
            geom = gdf.loc[idx, 'geometry']
            vertex_count = self._count_vertices(geom)
            
            if vertex_count > 3000:  #Complexity threshold (number of vertex)
                self.log(f"Complex geometry at index {idx} ({vertex_count} vertices). Simplifying...", "warning")
                gdf.loc[idx, 'geometry'] = geom.simplify(tolerance=0.001, preserve_topology=True)
                
        return gdf
    #check valid coordinate range 
    def _is_valid_coordinate(self, x, y):
        """Check if coordinates are within valid lat/lon ranges"""
        return (-180 <= x <= 180) and (-90 <= y <= 90)
    #vertex calculation. Determine complexity of a geometry
    def _count_vertices(self, geom):
        """Count vertices in a geometry"""
        if hasattr(geom, 'exterior'):
            return len(geom.exterior.coords)
        elif hasattr(geom, 'geoms'):  # MultiPolygon
            return sum(len(poly.exterior.coords) for poly in geom.geoms)
        else:
            return 0
    
    def _final_validation(self, gdf):
        """Perform final validation checks"""
        #Check if any geometries remain
        if len(gdf) == 0:
            self.log("No valid geometries remaining after validation!", "error")
            return False
        
        #Check geometry validity
        if not gdf.geometry.is_valid.all():
            self.log("Could not fix all geometry issues!", "error")
            return False
        
        #Check coordinate bounds
        bounds = gdf.total_bounds
        if not (-180 <= bounds[0] <= 180 and -180 <= bounds[2] <= 180 and 
                -90 <= bounds[1] <= 90 and -90 <= bounds[3] <= 90):
            self.log(f"Coordinates out of valid range: {bounds}", "error")
            self.log("Longitude must be between -180 and 180, Latitude between -90 and 90", "error")
            return False
            
        return True
    
#2. Functions to convert geodataframe into EE geometry
#Several option is presented here:
#a. direct conversion for single geometry (geemap function is utilzed here)
#b. if failed, use manual geojson conversion. 
#c. If that failed too, use a failback system by creating a bounding box around AOI, while using a simplified version of the AOI
class EE_converter:
    """
    Handle shapefile conversion from geodataframe to earth engine object
    """
    def __init__(self, verbose=True):
        """
        Initialize the EE converter
        Ensure Earth Engine is initialized lazily (avoids import-time failures).
        """
        # Ensure Earth Engine is initialized when first used (raises helpful error if not)
        ensure_ee_initialized()
        
        self.verbose = verbose

    def log(self, message, level = "info"):
        if self.verbose:
            if level == "error":
                st.error(message)
            elif level == "warning":
                st.warning(message)
            elif level == "success":
                st.success(message)
            else:
                st.info(message)  
    #Conversion for Single Geometry (AOI)      
    def convert_aoi_gdf(self, gdf):
        """
        GeoDataFrame for single geometry (AOI) to EE geometry conversion with option if failed
        """
        try:
            #Use geemap's built-in function
            aoi = geemap.gdf_to_ee(gdf)
            self.log("Successfully converted to EE geometry using geemap")
            return aoi
        except Exception as e:
            self.log(f"geemap conversion failed: {e}", "warning")
                
            try:
                    #If failed, use Manual conversion via GeoJSON
                self.log("Trying manual GeoJSON conversion...")
                    
                    # Union multiple geometries if present
                if len(gdf) > 1:
                        self.log("Multiple features found. Creating union...")
                        union_geom = gdf.geometry.unary_union
                        gdf = gpd.GeoDataFrame([{'geometry': union_geom}], crs=gdf.crs)
                    
                    # Convert to GeoJSON
                geojson = json.loads(gdf.to_json())
                    
                if geojson['features']:
                    geometry = geojson['features'][0]['geometry']
                        
                    if geometry['type'] == 'Polygon':
                            coords = geometry['coordinates']
                            aoi = ee.Geometry.Polygon(coords)
                    elif geometry['type'] == 'MultiPolygon':
                            coords = geometry['coordinates']
                            aoi = ee.Geometry.MultiPolygon(coords)
                    else:
                            self.log(f"Unsupported geometry type: {geometry['type']}", "error")
                            return None
                        
                    self.log("Successfully converted using manual method", "success")
                    return aoi
                        
            except Exception as e2:
                self.log(f"Manual conversion also failed: {e2}", "error")
                    
                try:
                    #Final attempt, create bounding box around the geometries
                    self.log("Trying bounding box conversion final attempt...")
                    bounds = gdf.total_bounds
                    aoi = ee.Geometry.Rectangle([bounds[0], bounds[1], bounds[2], bounds[3]])
                    self.log("Using bounding box as AOI (rectangular approximation)", "warning")
                    return aoi
                        
                except Exception as e3:
                    self.log(f"All conversion methods failed: {e3}", "error")
                    return None      
#Conversion for Multi Geometry (ROI)     
    def convert_roi_gdf(self, gdf):
        """
    Convert geodataframe into EE feture collection, build for the region of interest (ROI) data
         """
        try:
            self.log("Converting training data to Earth Engine FeatureCollection...")
            
            features = []
            conversion_errors = 0
            
            for idx, row in gdf.iterrows():
                try:
                    geom = row.geometry
                    # Get all non-geometry columns as properties
                    props = {k: v for k, v in row.drop('geometry').to_dict().items() 
                            if pd.notna(v)}  # Remove NaN values
                    
                    # Convert different geometry types
                    if geom.geom_type == "MultiPoint":
                        # Convert each point in MultiPoint separately
                        for i, pt in enumerate(geom.geoms):
                            pt_props = props.copy()
                            pt_props['point_id'] = f"{idx}_{i}"
                            features.append(ee.Feature(
                                ee.Geometry.Point([pt.x, pt.y]), 
                                pt_props
                            ))
                            
                    elif geom.geom_type == "Point":
                        features.append(ee.Feature(
                            ee.Geometry.Point([geom.x, geom.y]), 
                            props
                        ))
                        
                    elif geom.geom_type == "Polygon":
                        coords = list(geom.exterior.coords)
                        features.append(ee.Feature(
                            ee.Geometry.Polygon([coords]), 
                            props
                        ))
                        
                    elif geom.geom_type == "MultiPolygon":
                        polygons = []
                        for poly in geom.geoms:
                            coords = list(poly.exterior.coords)
                            polygons.append([coords])
                        features.append(ee.Feature(
                            ee.Geometry.MultiPolygon(polygons), 
                            props
                        ))
                        
                    else:
                        # Fallback: try using __geo_interface__
                        features.append(ee.Feature(geom.__geo_interface__, props))
                        
                except Exception as e:
                    conversion_errors += 1
                    self.log(f"Failed to convert feature {idx}: {e}", "warning")
                    continue
            
            if conversion_errors > 0:
                self.log(f"Failed to convert {conversion_errors} out of {len(gdf)} features", "warning")
            
            if not features:
                self.log("No features could be converted to Earth Engine format", "error")
                return None
            
            # Create FeatureCollection
            fc = ee.FeatureCollection(features)
            self.log(f"Successfully converted {len(features)} features to Earth Engine FeatureCollection", "success")
            
            return fc
            
        except Exception as e:
            self.log(f"Failed to convert training data: {e}", "error")
            return None