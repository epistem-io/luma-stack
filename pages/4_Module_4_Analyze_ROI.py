"""
Module 4: Spectral Separability Analysis

This module facilitate the user to perform spectral separability analysis for their training data

Architecture:
- Backend (module_4.py and module_4_part2.py): Pure backend process without UI dependencies
- Frontend (this file): Streamlit UI with session state management
- State synchronization ensures data persistence across page interactions
"""

import streamlit as st
from epistemx.shapefile_utils import shapefile_validator, EE_converter
from epistemx.module_4 import sample_quality
from epistemx.module_4_part2 import spectral_plotter
from modules.nav import Navbar
import matplotlib.pyplot as plt
import numpy as np
import geemap.foliumap as geemap
import geopandas as gpd
import traceback
import tempfile
import zipfile
import os
from ui_helper import show_footer, show_header

#Page configuration
st.set_page_config(
    page_title="Epistem-X Modul 4", #visible in the browser
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
        pass

# Apply custom theme
load_css()
show_header()

st.markdown("""
<div class="breadcrumb">Modul 4 › Analisis Area Sampel</div>
""", unsafe_allow_html=True)

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

<h1 class="gradient-title">Analisis Pemisahan Data Area Sampel (Region of Interest/ROI)</h1>
""", unsafe_allow_html=True)
st.divider()
st.markdown("""Modul ini memungkinkan pengguna melakukan analisis keterpisahan antar kelas tutupan/penggunaan lahan di dalam wilayah analisis yang anda tentukan. 
Sebelum melakukan analisis, pengguna harus mengunggah data referensi tiap-tiap kelas tutupan/penggunaan lahan yang sudah ditentukan dalam format yang kompatibel. 
Data ini harus memuat ID kelas unik dan nama kelas yang sesuai. Setelah data referensi diunggah, 
pengguna dapat melakukan analisis keterpisahan dengan langkah-langkah berikut:""")
st.markdown("1. Pilih atribut data latih, yaitu ID dan nama kelas")
st.markdown("2. Pilih parameter keterpisahan yang terdiri dari resolusi spasial dan jumlah maksimum piksel untuk setiap kelas. Platform ini menggunakan metode Transformed Divergence (TD) untuk melakukan analisis keterpisahan")

# Add navigation sidebar
Navbar()
st.markdown("Ketersediaan gabungan citra satelit dari modul 1")
#Check if landsat data from module 1 is available
if 'composite' in st.session_state:
    st.success("Gabungan citra satelit tersedia!")
    # Display information about available imagery
    if 'collection_metadata' in st.session_state:
        metadata = st.session_state['Image_metadata']
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Jumlah Citra", metadata.get('total_images', 'N/A'))
        with col2:
            st.metric("Periode Perekaman", metadata.get('date_range', 'N/A'))
        with col3:
            st.metric("Rata - Rata Tutupan Awan", f"{metadata.get('cloud_cover', {}).get('mean', 'N/A')}%")
#Preview the landsat mosaic
    if st.checkbox("Pratinjau Gabungan Citra Satelit"):
        composite = st.session_state['composite']
        vis_params = st.session_state['visualization']
        aoi_ee = st.session_state['AOI']  # This is ee.Geometry
        aoi_gdf = st.session_state.get('gdf')  # This is GeoDataFrame
        
        # Get centroid from GeoDataFrame if available, otherwise use default
        if aoi_gdf is not None:
            centroid = aoi_gdf.geometry.centroid.iloc[0]
            center = [centroid.y, centroid.x]
        else:
            # Fallback to Indonesia center if GeoDataFrame not available
            center = [-2.5, 118.0]
        
        m = geemap.Map(center=center, zoom=7)
        m.addLayer(composite, vis_params, "Landsat Mosaic")
        
        # Add AOI layer - use GeoDataFrame if available, otherwise EE geometry
        if aoi_gdf is not None:
            m.add_geojson(aoi_gdf.__geo_interface__, layer_name="AOI")
        else:
            m.addLayer(aoi_ee, {}, "AOI", True, 0.5)
        
        m.to_streamlit(height=600)
else:
    st.warning("Gabungan Citra Satelit tidak ditemukan. Mohon jalankan Modul 1 terlebih dahulu.")
    st.stop()

st.markdown("Ketersediaan data latih dari modul 3")
if 'train_final' in st.session_state:
    st.success("Data latih tersedia!")

else:
    st.warning("Data latih tidak ditemukan. Mohon jalankan Modul 3 terlebih dahulu.")
    st.stop()

st.divider()
#Training data separability analysis
if "train_final" in st.session_state:
    gdf_cleaned = st.session_state["train_final"]

    st.subheader("A. Lakukan Analisis Keterpisahan Sampel Data Latih")
    
    # Automatically use standardized column names from Module 3
    class_property = "LULC_ID"
    class_name_property = "Class_Name"
    
    # Verify that required columns exist
    if class_property not in gdf_cleaned.columns:
        st.error(f"❌ Kolom '{class_property}' tidak ditemukan dalam data latih.")
        st.error("Pastikan Anda telah menyelesaikan Modul 3 dengan benar dan menggunakan tombol 'Gunakan Data Latih Ini'.")
        st.stop()
    
    if class_name_property not in gdf_cleaned.columns:
        st.error(f"❌ Kolom '{class_name_property}' tidak ditemukan dalam data latih.")
        st.error("Pastikan Anda telah menyelesaikan Modul 3 dengan benar dan menggunakan tombol 'Gunakan Data Latih Ini'.")
        st.stop()

    st.session_state["selected_class_property"] = class_property
    st.session_state["selected_class_name_property"] = class_name_property

    
    #Separability Parameters
    st.subheader("Parameter Analisis")
    #Hardcode method to Transformed Divergence
    method = "TD"
    #Add information for separability approach
    st.info("Metode Analisis Keterpisahan: *Transformed Divergence* (TD)")
    st.markdown("""
    Metode ini mengukur keterpisahan statisk antara kelas tutupan lahan melalui analisis perbedaan nilai rata-rata dan struktur kovarian. 
    Nilai TD memiliki rentang 0 - 2, dimana nilai yang lebih tinggi mengacu kepada keterpisahan yang lebih baik.
    """)
    #user input (scale and max pixels). Note user did not have a lot to do, just make it default
    scale = st.number_input("Resolusi Spasial:", min_value=10, max_value=1000, value=30, step=10, 
                            help="Nilai lebih tinggi = resolusi lebih rendah tetapi waktu pemrosesan lebih cepat")
    max_pixels = st.number_input("Jumlah piksel maksimum per kelas:", min_value=1000, max_value=10000, value=5000, step=500,
                                help="Nilai lebih rendah = waktu pemrosesan lebih cepat dengan risiko sampel kurang representatif")

    #Single command to complete the analysis
    if st.button("Jalankan Analisis Keterpisahan", type="primary", use_container_width=True):
        if "train_final" not in st.session_state:
            st.error("Mohon unggah Shapefile data latih yang benar terlebih dahulu.")
        else:
            try:
                #get the properties 
                class_prop = st.session_state["selected_class_property"]
                class_name_prop = st.session_state["selected_class_name_property"]
                #Create progress bar
                progress = st.progress(0)
                status_text = st.empty()
                status_text.text("Langkah 1/7: Memvalidasi geometri data latih...")
                validate = shapefile_validator(verbose=False)
                converter = EE_converter(verbose=False)
                train_gdf = st.session_state["train_final"]
                train_gdf_cleaned = validate.validate_and_fix_geometry(train_gdf, geometry="mixed")
                
                if train_gdf_cleaned is None or train_gdf_cleaned.empty:
                    st.error("❌ Validasi geometri data latih gagal. Data tidak valid atau kosong.")
                    st.stop()
                progress.progress(10)
                
                status_text.text("Langkah 2/7: Mengonversi data latih ke format Earth Engine...")
                train_ee = converter.convert_roi_gdf(train_gdf_cleaned)
                
                if train_ee is None:
                    st.error("❌ Gagal mengonversi data latih ke format Google Earth Engine.")
                    st.stop()
                progress.progress(15)
                st.session_state["train_final_ee"] = train_ee
                
                #Intialize analyzer
                status_text.text("Langkah 3/7: Memulai proses analisis...")
                analyzer = sample_quality(
                    training_data=train_ee,
                    image=st.session_state["composite"],
                    class_property=class_prop,
                    region=st.session_state["AOI"],
                    class_name_property=class_name_prop,           
                )
                st.session_state["analyzer"] = analyzer
                st.session_state["analyzer_class_property"] = class_prop
                st.session_state["analyzer_class_name_property"] = class_name_prop
                progress.progress(25)
                #ROI statistics
                status_text.text("Langkah 4/7: Menghitung statistik area sampel")
                sample_stats_df = analyzer.get_sample_stats_df()
                st.session_state["sample_stats"] = sample_stats_df
                progress.progress(40)

                #Extract spectral values
                status_text.text("Langkah 5/7: Mengekstrak nilai spektral dari berbagai kanal/band… Proses ini memakan waktu beberapa menit.")
                try:
                    print(f"Debug: Akan mengekstrak nilai spektral dengan resolusi ={scale}, man jumlah maksimum piksel ={max_pixels}")
                    print(f"Debug: Properti kelas (class_property) yang digunakan ={analyzer.class_property}")
                    print(f"Properti nama kelas (class_name_property) yang digunakan ={analyzer.class_name_property}")
                        
                    pixel_extract = analyzer.extract_spectral_values(scale=scale, max_pixels_per_class=max_pixels)
                        
                    if pixel_extract.empty:
                            st.error("Tidak ada data spektral yang berhasil diekstrak. Periksa kembali apakah data latih Anda berada di dalam wilayah kajian (AOI).")
                            st.stop()
                        
                    print(f"Debug: Berhasil mengekstrak {len(pixel_extract)} piksel")
                    st.session_state["pixel_extract"] = pixel_extract
                    progress.progress(60)
                        
                except Exception as extract_error:
                        print(f"Debug: Rincian kesalahan ekstraksi {extract_error}")
                        print(f"Debug: Jenis kesalahan ekstraksi {type(extract_error)}")
                        traceback.print_exc()
                        raise extract_error

                #Compute pixel statistics
                status_text.text("Langkah 6/7: Menghitung statistik piksel...")
                try:
                    print("Debug: Sedang memulai perhitungan statistik piksel")
                    pixel_stats_df = analyzer.get_sample_pixel_stats_df(pixel_extract)
                    print(f"Debug: perhitungan statistik piksel berhasil, bentuk: {pixel_stats_df.shape}")
                    st.session_state["pixel_stats"] = pixel_stats_df
                    progress.progress(80)
                        
                except Exception as stats_error:
                    print(f"Debug: Error perhitungan statistik piksel: {stats_error}")
                    traceback.print_exc()
                    raise stats_error
                #Run separability analysis
                status_text.text("Langkah 7/7: Menjalankan analisis keterpisahan…")
                separability_df = analyzer.get_separability_df(pixel_extract, method=method)
                lowest_sep = analyzer.lowest_separability(pixel_extract, method=method)
                summary_df = analyzer.sum_separability(pixel_extract)
                #store all separability data
                st.session_state["separability_results"] = separability_df
                st.session_state["lowest_separability"] = lowest_sep
                st.session_state["separability_summary"] = summary_df
                st.session_state["separability_method"] = method
                st.session_state["analysis_complete"] = True
                progress.progress(100)
                st.success("Analisis Selesai!")
            except Exception as e:
                st.error(f"Terjadi kesalahan saat analisis: {str(e)}")
                st.write("Mohon periksa kembali data latih dan parameter analisis Anda, lalu coba lagi.")
    #display the result
if st.session_state.get("analysis_complete", False):
    st.divider()
    st.subheader("B. Hasil Analisis")
    #Display the results in table format
    #ROI Stats
    with st.expander("Statistik Sampel Area (ROI Statistics)", expanded=False):
        if "sample_stats" in st.session_state:
            st.markdown("""
            **Penjelasan Statistik Sampel:**
            
            Tabel ini menampilkan informasi dasar tentang area sampel (ROI) yang Anda buat untuk setiap kelas tutupan lahan:
            - **Class**: Nama kelas tutupan/penggunaan lahan
            - **Class ID**: Nomor identifikasi unik untuk setiap kelas
            - **Sample Count**: Jumlah area sampel yang dibuat untuk kelas tersebut
            - **Proportion**: Proporsi kelas tersebut terhadap keseluruhan sampel
            
            💡 Pastikan setiap kelas memiliki jumlah sampel yang cukup dan tersebar merata di wilayah kajian untuk hasil klasifikasi yang optimal.
            """)
            st.dataframe(st.session_state["sample_stats"], use_container_width=True)
        else:
            st.write("Statistik sampel tidak tersedia.")
    #Pixel stats
    with st.expander("Statistik Piksel Sampel", expanded=True):  
        if "pixel_stats" in st.session_state:
            st.markdown("""
            **Penjelasan Statistik Piksel:**
            
            Tabel ini menampilkan ringkasan nilai spektral dari piksel-piksel yang diambil dari setiap kelas. Statistik ini membantu memahami karakteristik spektral setiap kelas tutupan lahan:
            
            - **Class**: Nama kelas tutupan/penggunaan lahan
            - **Band**: Kanal spektral citra satelit (contoh: BLUE, GREEN, RED, NIR, SWIR)
            - **Count**: Jumlah piksel yang berhasil diekstrak untuk analisis
            - **Mean**: Nilai rata-rata reflektansi spektral kelas tersebut pada band tertentu
            - **Std (Standard Deviation)**: Ukuran sebaran/variasi nilai spektral (nilai lebih tinggi = lebih beragam)
            - **Min/Max**: Nilai reflektansi terendah dan tertinggi yang tercatat
            
            💡 **Cara Membaca:**
            - Nilai **Mean** yang berbeda antar kelas pada band yang sama menunjukkan potensi keterpisahan yang baik
            - **Std** yang rendah menunjukkan kelas tersebut homogen (konsisten secara spektral)
            - Perhatikan band mana yang paling membedakan antar kelas (perbedaan Mean yang besar)
            
            **Contoh Interpretasi:**
            - Vegetasi biasanya memiliki nilai NIR tinggi dan RED rendah
            - Air biasanya memiliki nilai rendah di semua band, terutama NIR dan SWIR
            """)
            st.dataframe(st.session_state["pixel_stats"], use_container_width=True)
        else:
            st.write("Statistik piksel tidak tersedia.")
    #Separability summary
    with st.expander("Ringkasan Analisis Keterpisahan", expanded=True):
        if "separability_summary" in st.session_state:
            st.markdown("""
            **Penjelasan Ringkasan Keterpisahan:**
            
            Tabel ini memberikan gambaran umum tentang seberapa baik kelas-kelas tutupan lahan Anda dapat dibedakan secara spektral:
            
            - **Total Pairs**: Jumlah total pasangan kelas yang dianalisis (jika ada N kelas, maka ada N×(N-1)/2 pasangan)
            - **Good Separability Pairs (TD ≥ 1.8)**: Jumlah pasangan kelas yang dapat dipisahkan dengan baik
            - **Weak Separability Pairs (1.0 ≤ TD < 1.8)**: Jumlah pasangan kelas dengan pemisahan lemah/marginal
            - **Poor Separability Pairs (TD < 1.0)**: Jumlah pasangan kelas yang sulit dipisahkan
            - **Mean TD**: Rata-rata nilai keterpisahan dari semua pasangan kelas
            - **Min/Max TD**: Nilai keterpisahan terendah dan tertinggi
            
            💡 **Apa artinya bagi Anda?**
            Semakin banyak pasangan dengan "Good Separability", semakin akurat hasil klasifikasi yang akan Anda dapatkan.
            """)
            st.dataframe(st.session_state["separability_summary"], use_container_width=True)
            
            #Add interpretation
            summary = st.session_state["separability_summary"].iloc[0]
            total_pairs = summary.get('Total Pairs', 0)
            good_pairs = summary.get('Good Separability Pairs', 0)
            weak_pairs = summary.get('Weak Separability Pairs', 0)
            poor_pairs = summary.get('Poor Separability Pairs', 0)
            
            st.markdown("---")
            st.markdown("### 📊 Evaluasi Kualitas Data Latih")
            
            if total_pairs > 0:
                good_pct = (good_pairs / total_pairs) * 100
                weak_pct = (weak_pairs / total_pairs) * 100
                poor_pct = (poor_pairs / total_pairs) * 100
                
                # Visual representation
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("🟢 Baik", f"{good_pairs}/{total_pairs}", f"{good_pct:.1f}%")
                with col2:
                    st.metric("🟡 Lemah/Marjinal", f"{weak_pairs}/{total_pairs}", f"{weak_pct:.1f}%")
                with col3:
                    st.metric("🔴 Buruk", f"{poor_pairs}/{total_pairs}", f"{poor_pct:.1f}%")
                
                st.markdown("---")
                
                # Overall assessment
                if good_pct >= 70:
                    st.success(f"""
                    ✅ **Kualitas Data Latih: SANGAT BAIK**
                    
                    {good_pct:.1f}% dari pasangan kelas memiliki keterpisahan yang baik. Data latih Anda berkualitas tinggi dan siap untuk proses klasifikasi!
                    
                    **Rekomendasi:** Lanjutkan ke Modul 6 untuk membuat peta tutupan lahan.
                    """)
                elif good_pct >= 50:
                    st.warning(f"""
                    ⚠️ **Kualitas Data Latih: MENENGAH**
                    
                    {good_pct:.1f}% dari pasangan kelas memiliki keterpisahan yang baik. Hasil klasifikasi mungkin cukup akurat, tetapi ada ruang untuk perbaikan.
                    
                    **Rekomendasi:** 
                    - Periksa pasangan kelas dengan keterpisahan rendah di bawah
                    - Pertimbangkan untuk menambah atau memperbaiki sampel untuk kelas yang bermasalah
                    - Atau lanjutkan ke Modul 6 jika hasil ini sudah memadai untuk kebutuhan Anda
                    """)
                else:
                    st.error(f"""
                    ❌ **Kualitas Data Latih: PERLU PERBAIKAN**
                    
                    Hanya {good_pct:.1f}% dari pasangan kelas yang memiliki keterpisahan baik. Hasil klasifikasi kemungkinan akan memiliki banyak kesalahan.
                    
                    **Rekomendasi Perbaikan:**
                    1. Periksa pasangan kelas dengan keterpisahan rendah di bagian "Pasangan Kelas dengan Keterpisahan Terendah"
                    2. Kembali ke Modul 3 untuk:
                       - Menambah jumlah sampel untuk kelas yang bermasalah
                       - Memastikan sampel lebih representatif dan homogen
                       - Mempertimbangkan penggabungan kelas yang sangat mirip secara spektral
                    3. Periksa apakah definisi kelas Anda terlalu mirip (contoh: "Hutan Lebat" vs "Hutan Sedang")
                    """)
            
            # Add detailed interpretation guide
            st.markdown("---")
            st.markdown("### 📖 Panduan Interpretasi Nilai TD (Transformed Divergence)")
            st.markdown("""
            Nilai TD mengukur seberapa berbeda dua kelas secara spektral. Rentang nilai: 0 - 2
            
            | Nilai TD | Kategori | Interpretasi | Dampak pada Klasifikasi |
            |----------|----------|--------------|-------------------------|
            | **≥ 1.8** | 🟢 **Baik** | Kedua kelas sangat berbeda secara spektral | Dapat diklasifikasikan dengan akurat |
            | **1.0 - 1.8** | 🟡 **Lemah/Marjinal** | Ada tumpang tindih spektral | Potensi kesalahan klasifikasi sedang |
            | **< 1.0** | 🔴 **Buruk** | Sangat mirip secara spektral | Tingkat kesalahan klasifikasi tinggi |
            
            **Contoh Kasus:**
            - **TD = 1.95** antara "Hutan" dan "Lahan Terbangun" → Sangat baik, mudah dibedakan
            - **TD = 1.45** antara "Sawah" dan "Lahan Pertanian Kering" → Lemah, perlu perhatian khusus
            - **TD = 0.75** antara "Tanaman Sawit" dan "Tanaman Karet" → Buruk, sulit dibedakan
            """)
            
        else:
            st.write("Ringkasan analisis keterpisahan data tidak tersedia.")           
    # Detailed Separability Results
    with st.expander("Rincian Keterpisahan Setiap Pasangan Kelas", expanded=False):
        if "separability_results" in st.session_state:
            st.markdown("""
            **Penjelasan Tabel Rincian:**
            
            Tabel ini menampilkan nilai keterpisahan (TD) untuk setiap pasangan kelas yang mungkin. Gunakan tabel ini untuk:
            
            - **Mengidentifikasi pasangan kelas yang bermasalah**: Cari nilai TD < 1.0
            - **Memahami hubungan antar kelas**: Kelas mana yang paling mirip secara spektral?
            - **Merencanakan perbaikan**: Fokus pada pasangan kelas yang memiliki nilai TD rendah
            
            **Kolom dalam tabel:**
            - **Class 1 / Class 2**: Pasangan kelas yang dibandingkan
            - **TD Value**: Nilai Transformed Divergence (0-2, semakin tinggi semakin baik)
            - **Category**: Kategori keterpisahan (Good/Weak/Poor)
            
            💡 **Tips:** Urutkan tabel berdasarkan "TD Value" untuk melihat pasangan yang paling bermasalah terlebih dahulu.
            """)
            st.dataframe(st.session_state["separability_results"], use_container_width=True)
        else:
            st.write("Tidak ada hasil analisis keterpisahan terperinci yang tersedia")
    # Most Problematic Class Pairs
    with st.expander("⚠️ Pasangan Kelas dengan Keterpisahan Terendah (Perlu Perhatian)", expanded=True):
        if "lowest_separability" in st.session_state:
            st.markdown("""
            **Mengapa Ini Penting?**
            
            Pasangan kelas di bawah ini memiliki nilai keterpisahan terendah, yang berarti mereka paling sulit dibedakan secara spektral. 
            Oleh karena itu, terdapat kemungkinan kesalahan klasifikasi diantara kedua kelas tersebut
            
            **Apa yang Harus Dilakukan?**
            """)
            
            lowest_sep = st.session_state["lowest_separability"]
            st.dataframe(lowest_sep, use_container_width=True)
            
            # Provide specific recommendations based on the results
            st.markdown("---")
            st.markdown("### 🔧 Rekomendasi Perbaikan")
            
            # Check if there are any poor separability pairs
            if not lowest_sep.empty:
                min_td = lowest_sep['TD_Distance'].min()
                
                if min_td < 1.0:
                    st.error("""
                    **🚨 Tindakan Diperlukan - Keterpisahan Buruk Terdeteksi**
                    
                    Beberapa pasangan kelas memiliki keterpisahan sangat rendah (TD < 1.0). Pertimbangkan langkah berikut:
                    """)
                    
                    st.markdown("""
                    **Opsi 1: Perbaiki Data Latih (Direkomendasikan)**
                    - Tambahkan lebih banyak sampel untuk kelas yang bermasalah
                    - Pastikan sampel lebih homogen dan tidak bercampur dengan kelas lain
                    
                    **Opsi 2: Gabungkan Kelas yang Mirip**
                    - Jika dua kelas memang sangat mirip secara spektral, pertimbangkan untuk menggabungkannya
                    - Contoh: "Tanaman Sawit" dan "Tanaman karet" → "Perkebunan Berkayu Keras"
                    
                    **Opsi 3: Gunakan Data Tambahan**
                    - Pertimbangkan menggunakan data non-spektral (NDVI, EVI, dll)
                    - Gunakan data elevasi atau kemiringan lereng sebagai data pendukung
                    - Saat ini platform Epistem belum mendukung opsi ini, nantikan di fase selanjutnya
                    
                    **Opsi 4: Lanjutkan dengan Catatan**
                    - Anda dapat melanjutkan modul, namun kelas yang memiliki keterpisahan yang rendah perlu diperhatikan
                    """)
                    
                elif min_td < 1.8:
                    st.warning("""
                    **⚠️ Perhatian - Keterpisahan Lemah Terdeteksi**
                    
                    Beberapa pasangan kelas memiliki keterpisahan lemah (TD 1.0-1.8). Hasil klasifikasi mungkin masih dapat diterima, 
                    tetapi pertimbangkan untuk:
                    """)
                    
                    st.markdown("""
                    - Menambah jumlah sampel untuk kelas yang bermasalah
                    - Memastikan sampel lebih representatif
                    - Melakukan validasi akurasi yang teliti di Modul 7
                    - Atau lanjutkan ke Modul 6 jika hasil ini sudah memadai
                    """)
                else:
                    st.success("""
                    **✅ Keterpisahan Baik**
                    
                    Semua pasangan kelas memiliki keterpisahan yang baik (TD ≥ 1.8). Data latih Anda berkualitas tinggi!
                    Anda dapat melanjutkan ke Modul 6
                    """)
            
            # Add visual guide
            st.markdown("---")
            st.markdown("### 📊 Visualisasi untuk Analisis Lebih Lanjut")
            st.info("""
            Gunakan tab **Visualisasi** di bawah untuk melihat secara visual bagaimana kelas-kelas Anda terpisah:
            - **Histogram**: Lihat distribusi nilai spektral setiap kelas
            - **Box Plot**: Bandingkan rentang nilai spektral antar kelas
            - **Scatter Plot 2D**: Lihat tumpang tindih antar kelas dalam ruang 2 dimensi
            - **Scatter Plot 3D**: Eksplorasi keterpisahan dalam ruang 3 dimensi
            
            Visualisasi ini akan membantu Anda memahami mengapa beberapa kelas sulit dipisahkan.
            """)
            
        else:
            st.write("Tidak ada data pasangan kelas yang rancu.")            

st.divider()
st.subheader("C. Visualisasi keterpisahan antar kelas dalam data latih")
st.markdown("Anda dapat menampilkan keterpisahan antar kelas dalam data latih menggunakan beberapa plot, yaitu histogram, plot kotak (*box plot*), dan plot sebar (*scatter plot*). Ini memungkinkan pengguna untuk mengevaluasi sebaran tumpang tindih data antar kelas, yang berpotensi mengurangi akurasi proses klasifikasi di langkah selanjutnya.")
if (st.session_state.get("analysis_complete", False) and 
    "pixel_extract" in st.session_state and
    "analyzer" in st.session_state and
    not st.session_state["pixel_extract"].empty):

    #initialize the plotter
    try:
        plotter = spectral_plotter(st.session_state["analyzer"])
        pixel_data = st.session_state["pixel_extract"]
        #verification
        available_bands = [b for b in plotter.band_names if b in pixel_data.columns]
        if not available_bands:
            st.error("Tidak ada kanal/band spektral yang ditemukan dalam piksel yang diekstrak.")
            st.stop()
        #Tabs for different visualization
        viz1, viz2, viz3, viz4 = st.tabs([
            "Histogram",
            "Box Plot",
            "Scatter Plot",
            "3D Scatter Plot"
        ])
        #Tab 1: Facet Histograms
        with viz1:
            st.markdown("### Distribusi Nilai Spektral per Kelas")
            st.markdown("Histogram interaktif yang menampilkan semua kelas untuk perbandingan mudah. " \
                       "Klik item legenda untuk menampilkan/menyembunyikan kelas tertentu.")
            
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                selected_hist_bands = st.multiselect(
                    "Pilih kanal/band untuk diplot:",
                    options=available_bands,
                    default=available_bands[:3] if len(available_bands) >= 3 else available_bands,
                    key="hist_bands"
                )
            with col2:
                bins = st.slider("Jumlah bin:", min_value=10, max_value=50, value=30, step=5)
            with col3:
                hist_opacity = st.slider("Opasitas:", 0.3, 0.9, 0.6, 0.1, key="hist_opacity")
            
            if st.button("Buat Histogram", key="btn_histogram", type="primary"):
                if selected_hist_bands:
                    with st.spinner("Membuat histogram interaktif..."):
                        try:
                            figures = plotter.plot_histogram(
                                pixel_data, 
                                bands=selected_hist_bands, 
                                bins=bins,
                                opacity=hist_opacity
                            )
                            for fig in figures:
                                st.plotly_chart(fig,width='stretch')
                            st.success("✅ Histogram berhasil dibuat!")
                            st.info("💡 **Tips:** Klik item legenda untuk menampilkan/menyembunyikan kelas.")
                        except Exception as e:
                            st.error(f"Terjadi kesalahan dalam menampilkan histogram: {str(e)}")
                else:
                    st.warning("Pilih setidaknya satu kanal/band.")
        
        #Tab 2: Box Plots
        with viz2:
            st.markdown("### Plot Kotak - Distribusi Nilai Spektral")
            st.markdown("Plot kotak interaktif yang menampilkan nilai spektral median, kuartil, dan pencilan untuk setiap kelas. " \
                       "Arahkan kursor ke kotak untuk melihat rincian statistik.")
            
            selected_box_bands = st.multiselect(
                "Pilih kanal/band untuk ditampilkan:",
                options=available_bands,
                default=available_bands[:5] if len(available_bands) >= 5 else available_bands,
                key="box_bands"
            )
            
            if st.button("Buat Plot Kotak", key="btn_boxplot", type="primary"):
                if selected_box_bands:
                    with st.spinner("Membuat plot kotak interaktif..."):
                        try:
                            figures = plotter.plot_boxplot(
                                pixel_data, 
                                bands=selected_box_bands
                            )
                            for fig in figures:
                                st.plotly_chart(fig,width='stretch')
                            st.success("✅ Plot kotak berhasil dibuat!")
                            st.info("💡 **Tips:** Arahkan kursor ke kotak untuk melihat nilai min, max, median, dan kuartil.")
                        except Exception as e:
                            st.error(f"Terjadi kesalahan dalam menampilkan plot kotak: {str(e)}")
                else:
                    st.warning("Pilih setidaknya satu kanal/band.")
        
        # TAB 3: SINGLE SCATTER PLOT
        with viz3:
            st.markdown("### Plot sebar antar kanal/band")
            st.markdown("Gunakan dua band spektral untuk menilai seberapa jelas perbedaan antar kelas pada ruang fitur dua dimensi.")
            
            available_bands = [b for b in plotter.band_names if b in pixel_data.columns]
            
            if len(available_bands) >= 2:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    x_band = st.selectbox(
                        "Band sumbu-X:",
                        options=available_bands,
                        index=0 if "RED" not in available_bands else available_bands.index("RED"),
                        key="scatter_x"
                    )
                
                with col2:
                    y_band = st.selectbox(
                        "Band sumbu-Y:",
                        options=available_bands,
                        index=1 if "NIR" not in available_bands else available_bands.index("NIR"),
                        key="scatter_y"
                    )
                
                with col3:
                    alpha = st.slider("Transparansi titik:", 0.1, 1.0, 0.6, 0.1)
                
                # Additional options
                col4, col5 = st.columns(2)
                with col4:
                    add_ellipse = st.checkbox("Tampilkan rentang kepercayaan", value=False, 
                                            help="Menampilkan elipsoid rentang kepercayaan (2-sigma) untuk setiap kelas")
                with col5:
                    color_palette = st.selectbox("Palet warna:", 
                                                ["tab10", "Set3", "Paired", "husl", "Accent"], 
                                                index=0,  help="Pratinjau akan diperbarui saat Anda mengubah pilihan")
                    colors = plt.cm.get_cmap(color_palette)(np.linspace(0, 1, 10))
                    color_boxes = ""
                    for i in range(10):
                        color_hex = '#{:02x}{:02x}{:02x}'.format(
                            int(colors[i][0]*255), 
                            int(colors[i][1]*255), 
                            int(colors[i][2]*255),
                            int(colors[i][2]*255)
                        )
                        color_boxes += f'<span style="display:inline-block; width:20px; height:20px; background-color:{color_hex}; margin:2px; border:1px solid #ddd; border-radius:2px;"></span>'
                    
                    st.markdown(color_boxes, unsafe_allow_html=True)
                
                if st.button("Buat plot sebar", key="btn_scatter"):
                    with st.spinner("Membuat plot sebar..."):
                        try:
                            fig = plotter.static_scatter_plot(
                                pixel_data,
                                x_band=x_band,
                                y_band=y_band,
                                alpha=alpha,
                                figsize=(12, 8),
                                color_palette=color_palette,
                                add_legend=True,
                                add_ellipse=add_ellipse
                            )
                            if fig:
                                #st.pyplot(fig)
                                st.success("Plot sebar berhasil dibuat!")
                                st.pyplot(plt.gcf())
                                plt.close()
                                # Add interpretation
                                st.info("""
                                **Tips interpretasi:**
                                - Kluster yang terpisah dengan jelas menunjukkan keterpisahan kelas yang baik
                                - Kluster yang saling tumpang tindih menandakan kemungkinan terjadinya kekeliruan dalam klasifikasi
                                - Bulatan elips menunjukkan sebaran dan korelasi data dari setiap kelas
                                """)
                        except Exception as e:
                            st.error(f"Error generating scatter plot: {str(e)}")
            else:
                st.warning("Memerlukan setidaknya 2 kanal/band untuk visualisasi plot sebar.")
   
        # TAB 4: MULTI-BAND SCATTER COMBINATIONS
        with viz4:
            st.markdown("### Eksplorasi Ruang Fitur 3D")
            st.markdown("Jelajahi tanda spektral dalam ruang 3D. Putar, zoom, dan geser untuk memahami hubungan kelas dalam ruang fitur tiga-kanal/band.")
            
            if len(available_bands) >= 3:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    x_band_3d = st.selectbox(
                        "Band sumbu-X:",
                        options=available_bands,
                        index=available_bands.index("RED") if "RED" in available_bands else 0,
                        key="scatter_3d_x"
                    )
                
                with col2:
                    y_band_3d = st.selectbox(
                        "Band sumbu-Y:",
                        options=available_bands,
                        index=available_bands.index("GREEN") if "GREEN" in available_bands else (1 if len(available_bands) > 1 else 0),
                        key="scatter_3d_y"
                    )
                
                with col3:
                    z_band_3d = st.selectbox(
                        "Band sumbu-Z:",
                        options=available_bands,
                        index=available_bands.index("NIR") if "NIR" in available_bands else (2 if len(available_bands) > 2 else 0),
                        key="scatter_3d_z"
                    )
                
                col4, col5 = st.columns(2)
                with col4:
                    marker_size_3d = st.slider("Ukuran titik:", 2, 8, 4, 1, key="marker_3d")
                with col5:
                    opacity_3d = st.slider("Transparansi titik:", 0.2, 1.0, 0.7, 0.1, key="opacity_3d")
                
                if st.button("Buat plot sebar 3-dimensi", key="btn_3d_scatter", type="primary"):
                    with st.spinner("Membuat plot sebar 3-dimensi..."):
                        try:
                            fig = plotter.scatter_plot_3d(
                                pixel_data,
                                x_band=x_band_3d,
                                y_band=y_band_3d,
                                z_band=z_band_3d,
                                marker_size=marker_size_3d,
                                opacity=opacity_3d
                            )
                            if fig:
                                st.plotly_chart(fig,width='stretch')
                                st.success("✅ Plot sebar 3-dimensi berhasil dibuat!")
                                
                                st.info("""
                                **💡 Fitur-fitur interaktif:**
                                - **Putar (Rotate)**: Klik dan seret untuk memutar tampilan 3D  
                                - **Perbesar (Zoom)**: Gunakan *scroll* untuk memperbesar/memperkecil tampilan  
                                - **Geser (Pan)**: Klik kanan dan seret untuk menggeser tampilan  
                                - **Arahkan (Hover)**: Arahkan kursor untuk melihat nilai pasti dari ketiga band  
                                - **Legenda (Legend)**: Klik untuk menampilkan atau menyembunyikan kelas  
                                - **Atur Ulang (Reset)**: Klik dua kali untuk mengembalikan tampilan semula  
                                
                                **Tips Analisis:**
                                - Perhatikan sebaran kelompok (cluster) yang terpisah dengan jelas dalam ruang 3D  
                                - Putar tampilan untuk menemukan sudut pandang yang paling menunjukkan keterpisahan antar kelas  
                                - Kelas yang tumpang tindih pada ruang 2D mungkin dapat terpisah di ruang 3D  
                                - Kombinasi band yang umum digunakan: **RGB**, **NIR-RED-GREEN**, **SWIR-NIR-RED**
                                """)
                        except Exception as e:
                            st.error(f"Terjadi kesalahan dalam membuat plot sebar 3-dimensi: {str(e)}")
            else:
                st.warning("Memerlukan setidaknya 3 kanal/band untuk visualisasi plot sebar 3-dimensi.")
                st.info("Plot sebar 3-dimensi memerlukan setidaknya tiga band spektral. " \
                    "Pastikan analisis Anda mencakup kanal/band yang cukup.")        
                st.markdown("---")
                st.markdown("**Klik kanan pada plot mana pun dan pilih 'Save image as...' untuk mengunduh")
            
    except Exception as e:
                st.error(f"Terjadi kesalahan dalam menyiapkan plot sebar 3-dimensi: {str(e)}")
                st.info("Pastikan analisis keterpisahan telah berhasil diselesaikan.")
else:
    st.info("Selesaikan analisis keterpisahan terlebih dahulu untuk memvisualisasikan data latihan.")
    st.markdown("""
    **Visualisasi yang tersedia setelah analisis:**
    - **Histogram**: Distribusi nilai spektral per kelas
    - **Plot kotak**: Ringkasan statistik nilai spektral
    - **Plot sebar**: Visualisasi ruang fitur 2D
    - **Plot sebar 3-dimensi**: Visualisasi ruang fitur 3D
    """)        
st.divider()
st.subheader("Navigasi modul")

# Check if Module 2 is completed (has at least one class)
module_2_completed = len(st.session_state.get("classes", [])) > 0

# Create two columns for navigation buttons
col1, col2 = st.columns(2)

with col1:
    # Back to Module 3 button (always available)
    if st.button("⬅️ Kembali ke Modul 3: Penentuan Data Latih", use_container_width=True):
        st.switch_page("pages/3_Module_3_Generate_ROI.py")

with col2:
    # Forward to Module 6 button (conditional)
    if module_2_completed:
        if st.button("➡️ Lanjut ke Modul 6: Buat Peta Tutupan Lahan", type="primary", use_container_width=True):
            st.switch_page("pages/5_Module_6_Classification_and_LULC_Creation.py")
    else:
        st.button("🔒 Selesaikan Modul 4 Terlebih Dahulu", disabled=True, use_container_width=True, 
                 help="Analisis area sampel untuk melanjutkan")

# Optional: Show completion status
if module_2_completed:
    st.success(f"✅ Analisis Selesai")
else:
    st.info("Analisis area sampel untuk melanjutkan")