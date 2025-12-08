"""
Module 1: Acquisition of Near cloud free satellite imagery

This module provides a user interface for fetching satellite imagery from google earth engine python API

Architecture:
- Backend (module_1.py): Pure earth engine process without UI dependencies
- Frontend (this file): Streamlit UI with session state management
- State synchronization ensures data persistence across page interactions
"""
import streamlit as st
import geemap.foliumap as geemap
import geopandas as gpd
from epistemx.module_1 import Reflectance_Data, Reflectance_Stats
from epistemx.shapefile_utils import shapefile_validator, EE_converter
from modules.nav import Navbar
import tempfile
import zipfile
import os
import ee
import datetime
import pandas as pd
from ui_helper import show_header

# Page configuration
st.set_page_config(
    page_title="Epistem-X Modul 1",
    page_icon="logos/logo_epistem_crop.png",
    layout="wide"
)

# Load custom CSS
def load_css():
    """Load custom CSS for EpistemX theme"""
    try:
        with open('.streamlit/style.css') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        pass  # CSS file not found, continue without custom styling

# Apply custom theme
load_css()

show_header()

st.markdown("""
<div class="breadcrumb">Module 1 › Buat Gabungan Citra</div>
""", unsafe_allow_html=True)

#=========Page requirements (title, description, session state)===========
#title of the module
st.markdown("""
<style>
.gradient-title {
  font-size: 2.5em;
  font-weight: 700;
  text-align: left;
  background: linear-gradient(90deg, var(--pink), var(--purple));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  color: transparent;
  margin-bottom: 0.4em;
}
</style>

<h1 class="gradient-title">Cari dan Buat Gabungan Citra Satelit</h1>
""", unsafe_allow_html=True)
st.divider()
#module name
markdown = """
Modul ini memungkinkan pengguna untuk mencari dan menghasilkan gabungan citra satelit untuk wilayah kajian dan rentang waktu yang anda tentukan, menggunakan data katalog Google Earth Engine (GEE).
"""
# Add navigation sidebar
Navbar()
#Initialize session state for storing collection, composite, aoi, AOI that has been converted to gdf, and export task
#similar to a python dict, we fill it later
if 'collection' not in st.session_state:
    st.session_state.collection = None
if 'composite' not in st.session_state:
    st.session_state.composite = None
if 'aoi' not in st.session_state:
    st.session_state.aoi = None
if 'gdf' not in st.session_state:
    st.session_state.gdf = None
if 'export_tasks' not in st.session_state:
    st.session_state.export_tasks = []
# Store search results to prevent re-searching on visualization changes
if 'search_results' not in st.session_state:
    st.session_state.search_results = None
if 'thermal_collection' not in st.session_state:
    st.session_state.thermal_collection = None
if 'detailed_stats' not in st.session_state:
    st.session_state.detailed_stats = None

#Task status caching to reduce API calls
#Lesson learn from multiple exports, since it conflict between user session state
#Cache aim to reduce the conflict between user
if 'task_cache' not in st.session_state:
    st.session_state.task_cache = {}
if 'last_cache_update' not in st.session_state:
    st.session_state.last_cache_update = {}

#Cache task status with time to live to reduce API calls
def get_cached_task_status(task_id, cache_ttl=30):
    """Get task status with caching to reduce API calls"""
    now = datetime.datetime.now()
    
    # Check if we have cached data that's still fresh
    if (task_id in st.session_state.task_cache and 
        task_id in st.session_state.last_cache_update):
        
        last_update = st.session_state.last_cache_update[task_id]
        if (now - last_update).seconds < cache_ttl:
            return st.session_state.task_cache[task_id]
    
    # Fetch fresh data
    try:
        status = ee.data.getTaskStatus(task_id)[0]
        st.session_state.task_cache[task_id] = status
        st.session_state.last_cache_update[task_id] = now
        return status
    except Exception as e:
        return None

def get_active_tasks():
    """Return only tasks that need monitoring"""
    active_tasks = []
    for task_info in st.session_state.export_tasks:
        # Skip if we know it's completed/failed from cache
        cached_status = st.session_state.task_cache.get(task_info['id'])
        if cached_status:
            state = cached_status.get('state', 'UNKNOWN')
            if state in ['COMPLETED', 'FAILED', 'CANCELLED']:
                continue
        active_tasks.append(task_info)
    return active_tasks

#Based on early experiments, shapefile with complex geometry often cause issues in GEE
#User input, AOI upload
st.subheader("Unggah berkas wilayah kajian (Shapefile)")
st.markdown("Saat ini, wahana hanya mendukung shapefile dalam format berkas .zip.")


#=========1. Area of Interest Definition (upload an AOI)===========
uploaded_file = st.file_uploader("Unggah shapefile dalam berkas .zip", type=["zip"])
aoi = None
#create a code for uploading the shapefile (what happen if the shapefile is uploaded)
if uploaded_file:
    # Extract the uploaded zip file to a temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # write uploaded bytes to disk (required before reading zip)
        zip_path = os.path.join(tmpdir, "upload.zip")
        with open(zip_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(tmpdir)

        # Find the .shp file in the extracted files (walk subfolders)
        shp_files = []
        for root, _, files in os.walk(tmpdir):
            for fname in files:
                if fname.lower().endswith(".shp"):
                    shp_files.append(os.path.join(root, fname))

        if len(shp_files) == 0:
            st.error("Shapefile tidak ditemukan dalam berkas zip yang diunggah.")
        else:
            #read the shapefile using geopandas
            try:
                gdf = gpd.read_file(shp_files[0])
                st.success("Berkas (Shapefile) berhasil dimuat!")
                #(System Response 1.1: Area of Interest Definition)
                #initialize the validator and converter 
                validate = shapefile_validator(verbose=False)
                converter = EE_converter(verbose=False) 
                #Validate and fix geometry
                gdf_cleaned = validate.validate_and_fix_geometry(gdf)
                #convert geodataframe to ee geometry, several option avaliable if the conversion failed
                if gdf_cleaned is not None:
                    aoi = converter.convert_aoi_gdf(gdf_cleaned)
                    if aoi is not None:
                        st.success("Wilayah kajian berhasil diproses!")
                        st.session_state.aoi = aoi
                        st.session_state.gdf = gdf_cleaned
                        
                        #Show a small preview map centered on AOI
                        st.text("Pratayang wilayah kajian:")
                        centroid = gdf_cleaned.geometry.centroid.iloc[0]
                        preview_map = geemap.Map(center=[centroid.y, centroid.x], zoom=7)
                        preview_map.add_geojson(gdf_cleaned.__geo_interface__, layer_name="AOI")
                        preview_map.to_streamlit(height=500)
                    else:
                        st.error("Gagal memuat wilayah kajian ke server")
                else:
                    st.error("Validasi geometri gagal.")
                    
            except Exception as e:
                st.error(f"Gagal membaca shapefile: {e}")
                st.info("Pastikan Shapefile Anda memuat semua berkas yang diperlukan (.shp, .shx, .dbf, .prj).")


#=========2. User input for image search criteria===========
st.divider()
#User input, search criteria
st.subheader("Tentukan Kriteria Pencarian Citra")

#Dump some information for supported imagery in the platform
st.markdown("""
Masukkan rentang tanggal perekaman, persentase tutupan awan, dan jenis misi Landsat.
Platform saat ini mendukung Landsat 1–3 pada level radiansi sensor, serta Landsat 4–9 pada Data Reflektansi Permukaan (Surface Reflectance) Collection 2 yang sudah berstatus Analysis Ready Data (ARD), tidak termasuk kanal termal.

Resolusi spasial untuk Landsat 1–3 adalah 60 meter, sedangkan seri lainnya memiliki resolusi 30 meter.
Ketersediaan misi Landsat adalah sebagai berikut:""")
st.markdown("""
1. Landsat 1 — Multispectral Scanner (MSS), periode 1972–1978
2. Landsat 2 — Multispectral Scanner (MSS), periode 1978–1982
3. Landsat 3 — Multispectral Scanner (MSS), periode 1978–1983
4. Landsat 4 — Thematic Mapper (TM), periode 1982–1993
5. Landsat 5 — Thematic Mapper (TM), periode 1984–2012
6. Landsat 7 — Enhanced Thematic Mapper Plus (ETM+), periode 1999–2021
7. Landsat 8 — Operational Land Imager (OLI), periode 2013–sekarang
8. Landsat 9 — Operational Land Imager-2 (OLI-2), periode 2021–sekarang
""")
#Date selection first - this will filter available sensors
st.subheader("Pilih Periode Waktu")
st.markdown("Jika opsi 'pilih berdasarkan tahun' dipilih, sistem akan secara otomatis mencari citra dari tanggal 1 Januari hingga 31 Desember.")
date_mode = st.radio(
    "Pilih mode penentuan tanggal:",
    ["Pilih berdasarkan tahun", "Rentang tanggal pilihan sendiri"],
    index=0
)

if date_mode == "Pilih berdasarkan tahun":
    # Just year input
    years = list(range(1972, datetime.datetime.now().year + 1))
    years.reverse()  #Newest First

    year = st.selectbox("Pilih tahun", years, index=years.index(2020))
    start_date = str(year)
    end_date = str(year)
    selected_year = year
#Full date
else:
    # Full date inputs
    default_start = datetime.date(2020, 1, 1)
    default_end = datetime.date(2020, 12, 31)
    start_date_dt = st.date_input("Tanggal mulai:", default_start)
    end_date_dt = st.date_input("Tanggal selesai:", default_end)
    start_date = start_date_dt.strftime("%Y-%m-%d")
    end_date = end_date_dt.strftime("%Y-%m-%d")
    selected_year = start_date_dt.year

# Filter sensors based on selected year/date range
sensor_dict = {
    "Landsat 1 MSS": {"code": "L1_RAW", "start_year": 1972, "end_year": 1978},
    "Landsat 2 MSS": {"code": "L2_RAW", "start_year": 1975, "end_year": 1982},
    "Landsat 3 MSS": {"code": "L3_RAW", "start_year": 1978, "end_year": 1983},
    "Landsat 4 TM": {"code": "L4_SR", "start_year": 1982, "end_year": 1993},
    "Landsat 5 TM": {"code": "L5_SR", "start_year": 1984, "end_year": 2012},
    "Landsat 7 ETM+": {"code": "L7_SR", "start_year": 1999, "end_year": 2021},
    "Landsat 8 OLI": {"code": "L8_SR", "start_year": 2013, "end_year": datetime.datetime.now().year},
    "Landsat 9 OLI-2": {"code": "L9_SR", "start_year": 2021, "end_year": datetime.datetime.now().year}
}

# Filter available sensors based on selected year
available_sensors = {}
for sensor_name, sensor_info in sensor_dict.items():
    if sensor_info["start_year"] <= selected_year <= sensor_info["end_year"]:
        available_sensors[sensor_name] = sensor_info["code"]

if not available_sensors:
    st.warning(f"Tidak ada sensor Landsat yang beroperasi pada tahun {selected_year}. Silakan pilih tahun lain.")
    st.stop()

# Create sensor selection based on filtered options
sensor_names = list(available_sensors.keys())
default_index = 0
# Try to set a reasonable default (prefer newer sensors)
if "Landsat 8 OLI" in sensor_names:
    default_index = sensor_names.index("Landsat 8 OLI")
elif "Landsat 7 ETM+" in sensor_names:
    default_index = sensor_names.index("Landsat 7 ETM+")

selected_sensor_name = st.selectbox(
    f"Pilih Sensor Landsat (Tersedia untuk tahun {selected_year}):", 
    sensor_names, 
    index=default_index
)
optical_data = available_sensors[selected_sensor_name]  #passing to backend process

#cloud cover slider
cloud_cover = st.slider("Batas Maksimum Tutupan Awan (%):", 0, 100, 30)

#=========3. Passing user input to backend codes ===========
#What happend when the button is pres by the user
if st.button("Cari citra satelit", type="primary") and st.session_state.aoi is not None:
    with st.spinner("Mencari citra Landsat..."):
        #first, search multispectral data (Collection 2 Tier 1, SR data)

        #(System Response 1.2: Search and Filter Imagery)
        reflectance = Reflectance_Data()
        collection, meta = reflectance.get_optical_data(
            aoi=aoi,
            start_date=start_date,
            end_date=end_date,
            optical_data=optical_data,
            cloud_cover=cloud_cover,
            verbose=False,
            compute_detailed_stats=False
        )
        #Second, use the same parameter as multispectral data and use it to search collection 2 TOA data. Retrive thermal band only
        #Skip thermal bands for Landsat 1-3 MSS (no thermal capability)
        thermal_collection = None
        if optical_data not in ['L1_RAW', 'L2_RAW', 'L3_RAW']:
            thermal_data = optical_data.replace('_SR', '_TOA')  # match Landsat pair automatically
            thermal_collection, meta = reflectance.get_thermal_bands(
                aoi=aoi,
                start_date=start_date,
                end_date=end_date,
                thermal_data=thermal_data,
                cloud_cover=cloud_cover,
                verbose=False,
                compute_detailed_stats=False
            )
        else:
            st.info("ℹ️ Catatan: Sensor MSS pada Landsat 1–3 tidak memiliki kanal termal. Hanya kanal multispektral yang akan diproses.")
        #Get collection retrival statistic
        stats = Reflectance_Stats()
        detailed_stats = stats.get_collection_statistics(collection, compute_stats=True, print_report=True)
        
        # Safely get total images count with fallback
        total_images = detailed_stats.get('total_images', 0)
        if total_images == 0:
            # Try alternative keys that might contain the count
            total_images = detailed_stats.get('num_images', 0)
            if total_images == 0:
                # Fallback to collection size
                try:
                    total_images = int(collection.size().getInfo())
                except:
                    total_images = 0
        
        st.success(f"Tersedia {total_images} grid lembar citra.")

        #Store the metadata for export
        st.session_state.search_metadata = {
            'sensor': optical_data,
            'start_date': start_date,
            'end_date': end_date,
            'num_images': total_images
            }
        try:
            coll_size = int(collection.size().getInfo())
        except Exception as e:
            st.error(f"Gagal mengambil ukuran koleksi data: {e}")
            coll_size = 0

        if coll_size == 0:
            st.warning("Tidak ada citra yang ditemukan. Coba: a) Naikkan batas tutupan awan, atau b) Ubah periode waktu/tanggal.")
        else:
            # Store search results in session state to prevent re-searching
            st.session_state.search_results = collection
            st.session_state.thermal_collection = thermal_collection
            st.session_state.detailed_stats = detailed_stats


#=========4. Displaying the result of the search===========
# Display results if they exist in session state (either from new search or previous search)
if st.session_state.search_results is not None and st.session_state.detailed_stats is not None:
    collection = st.session_state.search_results
    thermal_collection = st.session_state.thermal_collection
    detailed_stats = st.session_state.detailed_stats
    aoi = st.session_state.aoi
    gdf = st.session_state.gdf
    
    #Display the search information as report
    summary_md = f"""
    ### Ringkasan Pencarian Citra Landsat

    - **Total Citra Ditemukan:** {detailed_stats.get('total_images', 'N/A')}
    - **Rentang Tanggal Tersedia:** {detailed_stats.get('date_range', 'N/A')}
    """
    st.markdown(summary_md)
    #Path/Row information in expandable section
    path_row_tiles = detailed_stats.get('path_row_tiles', [])
    if path_row_tiles:
        with st.expander(f"Cakupan WRS Path/Row Coverage ({len(path_row_tiles)} lembar citra)"):
            # Create columns for better display
            num_cols = 3
            cols = st.columns(num_cols)
            
            for idx, (path, row) in enumerate(path_row_tiles):
                col_idx = idx % num_cols
                cols[col_idx].write(f" Path {path:03d} / Row {row:03d}")

    #Detailed Scene Information with cloud cover information
    with st.expander("ID Citra, Tanggal Perekaman, dan Tutupan Awan"):
        scene_ids = detailed_stats.get('Scene_ids', [])
        acquisition_dates = detailed_stats.get('individual_dates', [])
        cloud_covers = detailed_stats.get('cloud_cover', {}).get('values', [])
        
        if scene_ids and acquisition_dates:
            #Create a dataframe with all information
            scene_df = pd.DataFrame({
                '#': range(1, len(scene_ids) + 1),
                'Scene ID': scene_ids,
                'Tanggal Perekaman': acquisition_dates,
                'Tutupan Awan (%)': [round(cc, 2) for cc in cloud_covers] if cloud_covers else ['N/A'] * len(scene_ids)
            })
            
            #Display the table with formatting
            st.dataframe(
                scene_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    '#': st.column_config.NumberColumn('#', width='small'),
                    'Scene ID': st.column_config.TextColumn('Scene ID', width='large'),
                    'Tanggal Perekaman': st.column_config.TextColumn('Tanggal Perekaman', width='medium'),
                    'Tutupan Awan (%)': st.column_config.NumberColumn(
                        'Tutupan Awan (%)',
                        width='medium',
                        format="%.2f"
                    )
                }
            )
            #Show cloud cover statistics
            if cloud_covers:
                st.markdown("#### Statistik Tutupan Awan")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Minimum", f"{min(cloud_covers):.2f}%")
                with col2:
                    st.metric("Rata-rata", f"{sum(cloud_covers)/len(cloud_covers):.2f}%")
                with col3:
                    st.metric("Maksimum", f"{max(cloud_covers):.2f}%")
            
            #Download button
            csv = scene_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Unduh Daftar Citra sebagai CSV",
                data=csv,
                file_name=f"landsat_scenes_{start_date}_{end_date}.csv",
                mime="text/csv"
            )
        else:
            st.info("Tidak ada data citra untuk ditampilkan")
    #st.subheader("Detailed Statistics") {'bands': ['RED', 'GREEN', 'BLUE'], 'min': 0, 'max': 0.3}
    #st.write(detailed_stats)
    
    # Safely get total images count with fallback
    total_images = detailed_stats.get('total_images', 0)
    if total_images == 0:
        # Try alternative keys that might contain the count
        total_images = detailed_stats.get('num_images', 0)
        if total_images == 0:
            # Fallback to collection size
            try:
                total_images = int(collection.size().getInfo())
            except:
                total_images = 0
    
    if total_images > 0:
        #Create and image composite/mosaic for thermal bands (if available)
        if thermal_collection is not None:
            thermal_median = thermal_collection.median().clip(aoi)
            #composite for multispectral data and stacked them with thermal bands. Also convert to float()
            composite = collection.median().clip(aoi).addBands(thermal_median).toFloat()
        else:
            #For Landsat 1-3 MSS: no thermal bands available
            composite = collection.median().clip(aoi).toFloat()
        
        #Add section for visualization control
        st.subheader("Kombinasi Kanal Majemuk")
        #Add commonly used band combination for Landsat
        band_combinations = {
            "True Color (RGB)": {
                'bands': ['RED', 'GREEN', 'BLUE'],
                'min': 0.0,
                'max': 0.3,
                'gamma': 1.4
            },
            "False Color Infrared (NIR/Red/Green)": {
                'bands': ['NIR', 'RED', 'GREEN'],
                'min': 0,
                'max': 0.4,
                'gamma': 1.1
            },
            "Short-wave Infrared (SWIR2/NIR/RED)": {
                'bands': ['SWIR2', 'NIR', 'RED'],
                'min': 0,
                'max': 0.4,
                'gamma': 1.2
            },
            "Land/Water (NIR/SWIR1/RED)": {
                'bands': ['NIR','SWIR1','RED'],
                'min': 0,
                'max': 0.4,
                'gamma': [0.95, 1.1, 1]
            },
            "Kombinasi saluran bebas": {
                'bands': ['NIR', 'RED', 'GREEN'],  # Default for custom
                'min': 0.0,
                'max': 0.4,
                'gamma': 1.0
            }
        }
        
        # Get available bands from the composite
        available_bands = composite.bandNames().getInfo()
        
        #create a select box for the user to select the band combination
        selected_combination = st.selectbox(
            "Pilih Kombinasi Kanal:",
            list(band_combinations.keys()),
            index=0  #True color as default value
        )
        
        # Get the selected visualization parameters
        vis_params = band_combinations[selected_combination].copy()
        
        # If custom is selected, show band selection interface
        if selected_combination == "Kombinasi saluran bebas":
            st.info("💡 Pilih satu kanal untuk kombinasi hitam putih dan 3 kanal untuk visualisasi berwarna")
            
            col_band1, col_band2, col_band3 = st.columns(3)
            
            with col_band1:
                band1 = st.selectbox(
                    "Red Channel (or Single Band):",
                    available_bands,
                    index=available_bands.index('RED') if 'RED' in available_bands else 0,
                    key="custom_band1"
                )
            
            with col_band2:
                band2 = st.selectbox(
                    "Green Channel (optional):",
                    ['None'] + available_bands,
                    index=available_bands.index('GREEN') + 1 if 'GREEN' in available_bands else 0,
                    key="custom_band2"
                )
            
            with col_band3:
                band3 = st.selectbox(
                    "Blue Channel (optional):",
                    ['None'] + available_bands,
                    index=available_bands.index('BLUE') + 1 if 'BLUE' in available_bands else 0,
                    key="custom_band3"
                )
            
            # Build custom band list
            custom_bands = [band1]
            if band2 != 'None':
                custom_bands.append(band2)
            if band3 != 'None':
                custom_bands.append(band3)
            
            # Update vis_params with custom bands
            vis_params['bands'] = custom_bands
            
            # Show band combination info
            if len(custom_bands) == 1:
                st.caption(f"Grayscale visualization using: **{custom_bands[0]}**")
            elif len(custom_bands) == 3:
                st.caption(f"RGB visualization: R={custom_bands[0]}, G={custom_bands[1]}, B={custom_bands[2]}")
            else:
                st.warning("⚠️ Please select either 1 band (grayscale) or 3 bands (RGB)")
                vis_params['bands'] = [band1, band1, band1]  # Fallback to grayscale
        
        # Advanced visualization controls in expander
        with st.expander("Pengaturan visualisasi citra", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                # Min/Max value controls
                min_val = st.slider(
                    "Nilai Minimal:",
                    0.0,
                    0.5,
                    float(vis_params['min']),
                    0.01,
                    help="Adjust the minimum display value",
                    key="vis_min_slider"
                )
                max_val = st.slider(
                    "Nilai Maksimal:",
                    0.1,
                    1.0,
                    float(vis_params['max']),
                    0.01,
                    help="Adjust the maximum display value",
                    key="vis_max_slider"
                )
                
                # Update vis_params with user adjustments
                vis_params['min'] = min_val
                vis_params['max'] = max_val
            
            with col2:
                # Gamma controls - adapt based on number of bands
                num_bands = len(vis_params['bands'])
                
                if num_bands == 3 and isinstance(vis_params['gamma'], list):
                    st.write("**Nillai Gamma setiap kanal (R, G, B):**")
                    gamma_r = st.slider("Red Gamma:", 0.1, 2.0, float(vis_params['gamma'][0]), 0.1, key="gamma_r")
                    gamma_g = st.slider("Green Gamma:", 0.1, 2.0, float(vis_params['gamma'][1]), 0.1, key="gamma_g")
                    gamma_b = st.slider("Blue Gamma:", 0.1, 2.0, float(vis_params['gamma'][2]), 0.1, key="gamma_b")
                    vis_params['gamma'] = [gamma_r, gamma_g, gamma_b]
                elif num_bands == 3:
                    # RGB but single gamma value
                    use_per_band = st.checkbox("Gunakan nilai gamma setiap kanal", value=False, key="use_per_band_gamma")
                    if use_per_band:
                        st.write("**Gamma setiap kanal (R, G, B):**")
                        gamma_r = st.slider("Red Gamma:", 0.1, 2.0, 1.0, 0.1, key="gamma_r2")
                        gamma_g = st.slider("Green Gamma:", 0.1, 2.0, 1.0, 0.1, key="gamma_g2")
                        gamma_b = st.slider("Blue Gamma:", 0.1, 2.0, 1.0, 0.1, key="gamma_b2")
                        vis_params['gamma'] = [gamma_r, gamma_g, gamma_b]
                    else:
                        gamma = st.slider(
                            "Gamma:",
                            0.1,
                            2.0,
                            float(vis_params['gamma']) if not isinstance(vis_params['gamma'], list) else 1.0,
                            0.1,
                            help="Adjust image brightness/contrast",
                            key="gamma_single"
                        )
                        vis_params['gamma'] = gamma
                else:
                    # Single band (grayscale)
                    gamma = st.slider(
                        "Gamma:",
                        0.1,
                        2.0,
                        float(vis_params['gamma']) if not isinstance(vis_params['gamma'], list) else 1.0,
                        0.1,
                        help="Adjust image brightness/contrast",
                        key="gamma_grayscale"
                    )
                    vis_params['gamma'] = gamma
            
        # Thermal band visualization parameters
        thermal_vis = {
            'min': 286,
            'max': 300,
            'gamma': 0.4
        }
        
        #Store in session state for use in other modules
        st.session_state['composite'] = composite
        st.session_state['Image_metadata'] = detailed_stats
        st.session_state['AOI'] = aoi
        st.session_state['visualization'] = vis_params
        
        #Display the image using geemap, center around the AOI
        st.subheader("Pratayang Gabungan Citra")
        centroid = gdf.geometry.centroid.iloc[0]
        m = geemap.Map(center=[centroid.y, centroid.x], zoom=10)
        
        #Add layers with visibility controls
        if thermal_collection is not None:
            m.addLayer(thermal_median, thermal_vis, "Landsat Thermal Band", shown=False) #not shown
        m.addLayer(collection, vis_params, 'Landsat Collection', shown=False) #not shown
        m.addLayer(composite, vis_params, f'Composite - {selected_combination}', shown=True) #shown
        m.add_geojson(gdf.__geo_interface__, layer_name="AOI", shown=False) #not shown
        #cast the map to streamlit
        m.to_streamlit(height=600)
        
        #Add a button to clear search results and start over
        if st.button("🔄 Clear Results", type="secondary"):
            st.session_state.search_results = None
            st.session_state.thermal_collection = None
            st.session_state.detailed_stats = None
            st.session_state.composite = None
            st.session_state.Image_metadata = None
            st.rerun()
            
elif st.session_state.aoi is not None:
    st.info("Click 'Search Landsat Imagery' button above to begin searching.")
else:
    st.info("Upload an AOI and specify search criteria to begin.")

#=========5. Exporting the image collection===========
#current version only support direct download for data no larger in 32mb 
#if the data can be downloaded, the image is split into various bands. Dont do that. Make it one data
#check if the session state is not empty
if st.session_state.composite is not None and st.session_state.aoi is not None:
    st.subheader("Simpan Gabungan Citra")
    
    # Export destination selection
    export_destination = st.radio(
        "Pilih tujuan ekspor:",
        ["Unduh Langsung", "Google Cloud Storage"],
        index=0,
        help="Pilih lokasi untuk menyimpan hasil gabungan citra"
    )
    #Create an export setting for the user to filled
    with st.expander("Pengaturan ekspor", expanded=True):
        col1 = st.columns(1)
        #File Naming
        default_name = f"Landsat_{st.session_state.search_metadata.get('sensor', 'unknown')}_{st.session_state.search_metadata.get('start_date', '')}_{st.session_state.search_metadata.get('end_date', '')}_mosaic"
        export_name = st.text_input(
                "Nama berkas ekspor:",
                value=default_name,
                help="Hasil akan disimpan dalam format GeoTIFF (.tif)"
            )
        # Export destination specific settings
        if export_destination == "Unduh Langsung":
            st.info("📥 Berkas akan diunduh langsung ke komputer Anda dalam format GeoTIFF")
            st.warning("⚠️ Catatan: Unduhan langsung dibatasi maksimal 32 MB. Untuk area yang lebih besar, gunakan Google Cloud Storage.")
        
        else:  # Google Cloud Storage
            st.subheader("Pengaturan Google Cloud Storage")
            
            # Nama GCS Bucket
            gcs_bucket = st.text_input(
                "Nama GCS Bucket:",
                value="epistemx",
                placeholder="epistemx",
                help="Masukkan nama bucket Google Cloud Storage Anda"
            )
            
            # Awalan jalur file GCS
            gcs_path_prefix = st.text_input(
                "Awalan Jalur File (opsional):",
                value="landsat_exports/",
                help="Awalan jalur opsional di dalam bucket (misal: 'landsat_exports/' atau 'data/imagery/')"
            )
            
            # Email Akun Layanan (opsional - hanya untuk tampilan, sebagian disembunyikan)
            service_account_email = st.text_input(
                "Email Akun Layanan:",
                value="epistemx@ee-xxx.iam.gserviceaccount.com",
                placeholder="epistemx@ee-xxx.iam.gserviceaccount.com",
                help="Email akun layanan untuk autentikasi (disetel secara terpisah)"
            )
            
            if not gcs_bucket:
                st.warning("⚠️ Nama GCS Bucket wajib diisi untuk ekspor ke Cloud Storage")
            else:
                st.info(f"Berkas akan diekspor ke: gs://{gcs_bucket}/{gcs_path_prefix}{export_name}.tif")
        #Coordinate Reference System (CRS)
        #User can define their own CRS using EPSG code, if not, used WGS 1984 as default option    
        crs_options = {
                "WGS 84 (EPSG:4326)": "EPSG:4326",
                "Custom EPSG": "CUSTOM"
            }
        crs_choice = st.selectbox(
                "Coordinate Reference System:",
                options=list(crs_options.keys()),
                index=0
            )
            
        if crs_choice == 'Custom EPSG':
            custom_epsg = st.text_input(
                "Masukkan EPSG Code:",
                value="4326",
                help="Contoh: 32748 (UTM Zona 48S)"
                )
            export_crs = f"EPSG:{custom_epsg}"
        else:
            export_crs = crs_options[crs_choice]
            #Define the scale/spatial resolution of the imagery
        scale = st.number_input(
                "Ukuran piksel (meter):",
                min_value=10,
                max_value=1000,
                value=30,
                step=10
            )
        #Button to start export the composite
        #System Response 1.3: Imagery Download
        export_button_text = f"Mulai ekspor ke {export_destination}"
        export_disabled = False
        
        # Disable button if GCS is selected but bucket name is missing
        if export_destination == "Google Cloud Storage" and not gcs_bucket:
            export_disabled = True
            
        if st.button(export_button_text, type="primary", disabled=export_disabled):
            try:
                with st.spinner("Menyiapkan tugas ekspor…"):
                    #Use the composite from session state
                    export_image = st.session_state.composite
                    
                    #Valid Band Names 
                    band_names = export_image.bandNames()
                    export_image = export_image.select(band_names)
                    
                    #Get the AOI from geometry
                    aoi_obj = st.session_state.aoi

                    if isinstance(aoi_obj, ee.FeatureCollection):
                        export_region = aoi_obj.geometry()
                    elif isinstance(aoi_obj, ee.Feature):
                        export_region = aoi_obj.geometry()
                    elif isinstance(aoi_obj, ee.Geometry):
                        export_region = aoi_obj
                    else:
                        #If all else fails, try to get bounds
                        try:
                            export_region = aoi_obj.geometry()
                        except:
                            raise ValueError(f"Tidak dapat mengekstrak geometri dari objek wilayah kajian bertipe: {type(aoi_obj)}")
                    
                    # Configure export parameters based on destination
                    if export_destination == "Unduh Langsung":
                        # Use getDownloadURL for direct download
                        try:
                            download_params = {
                                "name": export_name,
                                "crs": export_crs,
                                "scale": scale,
                                "region": export_region,
                                "filePerBand": False,
                                "fileFormat": "GEO_TIFF",
                                "formatOptions": {"cloudOptimized": True}
                            }
                            
                            # Get download URL
                            download_url = export_image.getDownloadURL(download_params)
                            
                            if download_url:
                                st.success("✅ URL unduhan berhasil dibuat!")
                                st.markdown(f"""
                                **Detail Unduhan:**
                                - Format: GeoTIFF
                                - CRS: {export_crs}
                                - Resolusi: {scale} meter
                                - Nama berkas: {export_name}.tif
                                """)
                                
                                # Create download link with button styling
                                st.markdown(f"""
                                <div style="margin: 1rem 0;">
                                    <a href="{download_url}" download="{export_name}.tif" target="_blank" style="text-decoration: none;">
                                        <div style="
                                            background-color: #ff4b4b;
                                            color: white;
                                            padding: 0.75rem 1.5rem;
                                            border-radius: 0.5rem;
                                            text-align: center;
                                            font-weight: 600;
                                            font-size: 1rem;
                                            cursor: pointer;
                                            display: inline-block;
                                            min-width: 200px;
                                            transition: background-color 0.3s;
                                        " onmouseover="this.style.backgroundColor='#e63946'" onmouseout="this.style.backgroundColor='#ff4b4b'">
                                            📥 Unduh GeoTIFF
                                        </div>
                                    </a>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Also provide text link as backup
                                st.markdown(f"**Link alternatif:** [Unduh {export_name}.tif]({download_url})")
                                st.info("💡 Klik tombol merah di atas untuk mengunduh berkas GeoTIFF ke komputer Anda.")
                                
                            else:
                                st.error("❌ Gagal membuat URL unduhan. Coba kurangi area atau gunakan Google Cloud Storage.")
                                
                        except Exception as download_error:
                            st.error(f"❌ Error saat membuat unduhan: {str(download_error)}")
                            st.info("💡 Coba kurangi area kajian atau gunakan Google Cloud Storage untuk area yang lebih besar.")
                        
                        # Direct download completed - no task needed
                        
                    else:  # Google Cloud Storage
                        #Summarize the export parameter from user input for GCS
                        export_params = {
                            "image": export_image,
                            "description": export_name.replace(" ", "_"),  #Remove spaces from description
                            "bucket": gcs_bucket,
                            "fileNamePrefix": f"{gcs_path_prefix}{export_name}",
                            "scale": scale,
                            "crs": export_crs,
                            "maxPixels": 1e13,
                            "fileFormat": "GeoTIFF",
                            "formatOptions": {"cloudOptimized": True},
                            "region": export_region
                        }
                        
                        #Pass the parameters to earth engine export for Cloud Storage
                        task = ee.batch.Export.image.toCloudStorage(**export_params)
                        
                        task.start()
                        
                        #Store task info in session state for monitoring
                        task_info = {
                            'id': task.id,
                            'name': export_name,
                            'destination': export_destination,
                            'folder': gcs_bucket,
                            'crs': export_crs,
                            'scale': scale,
                            'start_time': datetime.datetime.now(),
                            'last_progress': 0,
                            'last_update': datetime.datetime.now()
                        }
                        #Append to export tasks list
                        st.session_state.export_tasks.append(task_info)
                        #note, here the task is submitted, but not yet done
                        st.success(f"✅ Tugas ekspor '{export_name}' berhasil dikirim!")
                        st.info(f"ID Tugas: {task.id}")
                        
                        # Display export details for Google Cloud Storage
                        st.markdown(f"""
                        **Detail Ekspor:**
                        - Tujuan: Google Cloud Storage
                        - Lokasi berkas: gs://{gcs_bucket}/{gcs_path_prefix}{export_name}.tif
                        - CRS: {export_crs}
                        - Resolusi: {scale}m
                        
                        Periksa progres di [Earth Engine Task Manager](https://code.earthengine.google.com/tasks) atau gunakan pemantau tugas di bawah ini.
                        """)
                    
            except Exception as e:
                st.error(f"Gagal mengekspor: {str(e)}")
                st.info("Informasi Pemecahan Masalah:")
                st.write(f"Jenis wilayah kajian: {type(st.session_state.aoi)}")
                st.write(f"Komposit tersedia: {st.session_state.composite is not None}")

    #Earth Engine Export Task Monitor
    if st.session_state.export_tasks:
        st.subheader("Pantau Ekspor Earth Engine")
        
        # Manual refresh options with cache control
        col_refresh1, col_refresh2 = st.columns([1, 3])
        with col_refresh1:
            if st.button("🔄 Refresh Semua"):
                # Clear cache to force fresh data
                st.session_state.task_cache.clear()
                st.session_state.last_cache_update.clear()
                st.rerun()
        
        with col_refresh2:
            # Show cache status
            active_tasks_count = len(get_active_tasks())
            total_tasks_count = len(st.session_state.export_tasks)
            st.caption(f"Pantau {active_tasks_count}/{total_tasks_count} tugas aktif | segarkan ulang manual")
        
        #Summary of active tasks using cached status
        running_tasks = 0
        completed_tasks_count = 0
        failed_tasks = 0
        
        for task_info in st.session_state.export_tasks:
            try:
                status = get_cached_task_status(task_info['id'])
                if status:
                    state = status.get('state', 'UNKNOWN')
                    if state == 'RUNNING':
                        running_tasks += 1
                    elif state == 'COMPLETED':
                        completed_tasks_count += 1
                    elif state == 'FAILED':
                        failed_tasks += 1
            except:
                pass
        #Display task status for each task
        for i, task_info in enumerate(st.session_state.export_tasks):
            with st.expander(f"Task: {task_info['name']}", expanded=True):
                try:
                    # Get task status from cache or Earth Engine
                    status = get_cached_task_status(task_info['id'])
                    if not status:
                        st.error("Failed to get task status")
                        continue
                    
                    # Create columns for better layout
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.write(f"**Task ID:** {task_info['id']}")
                        st.write(f"**Name:** {task_info['name']}")
                        
                        # Individual task refresh button
                        if st.button(f"🔄", key=f"refresh_{i}", help="Refresh this task"):
                            # Clear cache for this specific task
                            if task_info['id'] in st.session_state.task_cache:
                                del st.session_state.task_cache[task_info['id']]
                            if task_info['id'] in st.session_state.last_cache_update:
                                del st.session_state.last_cache_update[task_info['id']]
                            st.rerun()
                    
                    with col2:
                        # Status with color coding
                        state = status.get('state', 'UNKNOWN')
                        if state == 'COMPLETED':
                            st.success(f"**Status:** {state}")
                        elif state == 'RUNNING':
                            st.info(f"**Status:** {state}")
                        elif state == 'FAILED':
                            st.error(f"**Status:** {state}")
                        elif state == 'CANCELLED':
                            st.warning(f"**Status:** {state}")
                        else:
                            st.write(f"**Status:** {state}")
                        
                        # Enhanced progress tracking with time estimates
                        progress = status.get('progress', 0)
                        current_time = datetime.datetime.now()
                        
                        if state == 'RUNNING' and progress > 0:
                            # Only update timing calculations if progress has changed
                            progress_changed = progress != task_info.get('last_progress', 0)
                            
                            if progress_changed:
                                task_info['last_progress'] = progress
                                task_info['last_update'] = current_time
                                #Calculate the progress ETA if only session state is changed
                                elapsed_time = current_time - task_info['start_time']
                                elapsed_minutes = elapsed_time.total_seconds() / 60
                                
                                if progress > 0:
                                    rate = progress / elapsed_minutes
                                    task_info['current_rate'] = rate
                                    if progress > 5:  # Only calculate ETA after 5% progress
                                        estimated_total_time = elapsed_minutes * (100 / progress)
                                        remaining_time = estimated_total_time - elapsed_minutes
                                        task_info['remaining_time'] = remaining_time
                                        task_info['elapsed_minutes'] = elapsed_minutes
                            
                            # Use cached calculations for display
                            progress_bar = st.progress(progress / 100.0)
                            
                            # Time information
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.write(f"**Progress:** {progress:.1f}%")
                            
                            if progress > 5:
                                with col_b:
                                    remaining_time = task_info.get('remaining_time', 0)
                                    if remaining_time > 60:
                                        st.write(f"**ETA:** ~{remaining_time/60:.0f}h {remaining_time%60:.0f}m")
                                    elif remaining_time > 1:
                                        st.write(f"**ETA:** ~{remaining_time:.0f} min")
                                    else:
                                        st.write("**ETA:** <1 min")
                                
                                # Additional timing info using cached values
                                elapsed_minutes = task_info.get('elapsed_minutes', 0)
                                rate = task_info.get('current_rate', 0)
                                st.caption(f"Elapsed: {elapsed_minutes:.0f} min | Rate: {rate:.1f}%/min")
                            else:
                                # Basic progress for early stages
                                st.progress(progress / 100.0)
                                st.write(f"**Progress:** {progress:.1f}% (calculating ETA...)")
                                st.caption(f"Elapsed: {task_info.get('elapsed_minutes', 0):.0f} min")
                        
                        elif state == 'RUNNING':
                            # Task is running but no progress reported yet
                            st.progress(0)
                            elapsed_time = current_time - task_info['start_time']
                            elapsed_minutes = elapsed_time.total_seconds() / 60
                            st.write("**Progress:** Initializing...")
                            st.caption(f"Elapsed: {elapsed_minutes:.0f} min")
                        
                        elif progress > 0 and state not in ['COMPLETED', 'FAILED', 'CANCELLED']:
                            # Show progress for other states
                            st.progress(progress / 100.0)
                            st.write(f"**Progress:** {progress:.1f}%")
                    
                    with col3:
                        # Format timestamps more readably
                        creation_ts = status.get('creation_timestamp_ms')
                        update_ts = status.get('update_timestamp_ms')
                        
                        if creation_ts:
                            creation_time = datetime.datetime.fromtimestamp(creation_ts / 1000)
                            st.write(f"**Mulai:** {creation_time.strftime('%H:%M:%S')}")
                        else:
                            st.write("**Mulai:** N/A")
                        
                        if update_ts:
                            update_time = datetime.datetime.fromtimestamp(update_ts / 1000)
                            st.write(f"**Waktu terkini:** {update_time.strftime('%H:%M:%S')}")
                        else:
                            st.write("**Waktu terkini:** N/A")
                        
                        # Show total runtime for completed tasks
                        if state == 'COMPLETED' and creation_ts and update_ts:
                            total_runtime = (update_ts - creation_ts) / 1000 / 60  # minutes
                            if total_runtime > 60:
                                st.caption(f"Total waktu: {total_runtime/60:.1f}jam {total_runtime%60:.0f}menit")
                            else:
                                st.caption(f"Total waktu: {total_runtime:.0f} menit")
                        
                        # Show cache status
                        if task_info['id'] in st.session_state.last_cache_update:
                            cache_age = (datetime.datetime.now() - st.session_state.last_cache_update[task_info['id']]).seconds
                            if cache_age < 60:
                                st.caption(f"📊 Data: {cache_age}detik yang lalu")
                            else:
                                st.caption(f"📊 Data: {cache_age//60}menit yang lalu")
                    
                    # Show error message if failed
                    if state == 'FAILED' and 'error_message' in status:
                        st.error(f"Error: {status['error_message']}")
                    
                    # Show completion details
                    if state == 'COMPLETED':
                        st.success("✅ Ekspor berhasil!")
                        drive_url = "https://drive.google.com/drive/folders/1JKwqv3q3JyQnkIEuIqTQ2hlwPmM-FQaF?usp=sharing"
                        st.success(f"Berkas disimpan di: [EPISTEM/EPISTEMX_Landsat_Export Folder]({drive_url})")
                        
                        #Option to remove completed task from monitor
                        if st.button(f"Hapus dari pantauan", key=f"remove_{i}"):
                            st.session_state.export_tasks.pop(i)
                            st.rerun()
                
                except Exception as e:
                    st.error(f"Gagal memuat status tugas: {str(e)}")
                    st.write(f"ID tugas: {task_info['id']}")
        
        # Clear all completed tasks button
        completed_tasks = []
        for task_info in st.session_state.export_tasks:
            try:
                status = get_cached_task_status(task_info['id'])
                if status and status.get('state') == 'COMPLETED':
                    completed_tasks.append(task_info)
            except:
                pass
        
        if completed_tasks:
            if st.button("🗑️ Hapus semua tugas yang selesai"):
                st.session_state.export_tasks = [
                    task for task in st.session_state.export_tasks 
                    if task not in completed_tasks
                ]
                st.rerun()

# Navigation
st.divider()
st.subheader("Navigasi modul")

if st.session_state.composite is not None:
    if st.button("Lanjut ke Modul 2: Tentukan Skema Klasifikasi"):
        st.switch_page("pages/2_Module_2_Classification_scheme.py")
else:
    st.button("🔒 Selesaikan Modul 1 terlebih dahulu", disabled=True)
    st.info("Buat gabungan citra terlebih dahulu sebelum melanjutkan.")