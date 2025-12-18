"""
Module 6: Supervised Classification

This module facilitate the user to perform supervised classification using random forest classifier

Architecture:
- Backend (module_6_phase1.py): Pure backend process without UI dependencies
- Frontend (this file): Streamlit UI with session state management
- State synchronization ensures data persistence across page interactions
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import geemap.foliumap as geemap
from epistemx.module_6_phase1 import FeatureExtraction, Generate_LULC
from modules.nav import Navbar
import numpy as np
import traceback
import ee
import datetime
from ui_helper import show_footer, show_header

#Page configuration
st.set_page_config(
    page_title="Epistem-X Modul 6",
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
<div class="breadcrumb">Modul 6 › Buat Peta Tutupan Lahan</div>
""", unsafe_allow_html=True)


#Set the page title (for the canvas)
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

<h1 class="gradient-title">Pembuatan Peta Tutupan Lahan</h1>
""", unsafe_allow_html=True)
st.divider()
st.markdown("""
Modul ini melakukan klasifikasi tutupan lahan menggunakan metode Random Forest. 
Untuk menggunakan modul ini, Anda harus menyelesaikan Modul 1 hingga 4. 
Modul 1 menghasilkan gabungan citra, Modul 2 mendefinisikan skema kelas, Modul 3 membuat data latihan (Area Sampel), dan Modul 4 menganalisis kualitas data latihan.
""")

# Add navigation sidebar
Navbar()

#Check prerequisites from previous modules. The module cannot open if the previous modules is not complete.
#add module 2 check and module 3 (for training data not analysis)
st.subheader("Cek Prasyarat")

col1, col2 = st.columns(2)

#Check for image composite from Module 1
with col1:
    if 'composite' in st.session_state and st.session_state.composite is not None:
        st.success("✅ Gabungan citra tersedia dari modul 1")
        image = st.session_state['composite']
        
        # Display metadata if available
        if 'Image_metadata' in st.session_state:
            metadata = st.session_state['Image_metadata']
            with st.expander("Detail Citra"):
                st.write(f"**Sensor:** {st.session_state.get('search_metadata', {}).get('sensor', 'N/A')}")
                st.write(f"**Date Range:** {metadata.get('date_range', 'N/A')}")
                st.write(f"**Total Images:** {metadata.get('total_images', 'N/A')}")
    else:
        #Display error message if composite is not found
        st.error("❌ Gabungan citra tidak tersedia")
        st.warning("Mohon selesaikan modul 1 untuk menghasilkan gabungan citra")
        image = None

#Check for training data from Module 3/4
with col2:
    if 'train_final' in st.session_state and st.session_state.train_final is not None:
        st.success("✅ Data sampel tersedia")
        roi = st.session_state['train_final']
        
        # Display training data info if available
        if 'train_final' in st.session_state:
            gdf = st.session_state['train_final']
            with st.expander("Detail Data Pelatihan"):
                st.write(f"**Total Features:** {len(gdf)}")
                st.write(f"**Columns:** {', '.join(gdf.columns.tolist())}")
                
                # Show class distribution if class property is known
                if 'selected_class_property' in st.session_state:
                    class_prop = st.session_state['selected_class_property']
                    class_name = st.session_state['selected_class_name_property']
                    if class_prop in gdf.columns:
                        class_counts = gdf[class_prop].value_counts()
                        class_name = gdf[class_name].unique()
                        st.write("**Class Distribution:**")
                        st.dataframe(class_counts, use_container_width=True)
    else:
        st.error("❌ Data sampel tidak tersedia")
        st.warning("Mohon selesaikan modul 3 dan 4 untuk menghasilkan dan melakukan analisis data sampel")
        roi = None

#Stop if prerequisites are not met
if image is None or roi is None:
    st.divider()
    st.info("⚠️ Selesaikan modul-modul sebelumnya sebelum melanjutkan ke klasifikasi")
    st.markdown("""
    **Langkah yang Diperlukan:**
    1. **Module 1:** Buat gabungan citra
    2. **Module 2:** Definisikan skema klasifikasi 
    3. **Module 3:** Unggah dan validasi data sampe;
    4. **Module 4:** Analisis keterpisahan data sampel
    5. **Module 6:** Kembali ke sini untuk melakukan klasifikasi
    """)
    st.stop()

#Get AOI for clipping the result
aoi = st.session_state.get('AOI', None)


#Initialize session state for storing results
if 'extracted_training_data' not in st.session_state:
    st.session_state.extracted_training_data = None
if 'extracted_testing_data' not in st.session_state:
    st.session_state.extracted_testing_data = None
if 'classification_result' not in st.session_state:
    st.session_state.classification_result = None

# Initialize session state for export tasks (reuse from Module 1)
if 'export_tasks' not in st.session_state:
    st.session_state.export_tasks = []

# Task status caching to reduce API calls
if 'task_cache' not in st.session_state:
    st.session_state.task_cache = {}
if 'last_cache_update' not in st.session_state:
    st.session_state.last_cache_update = {}

# Cache task status with time to live to reduce API calls
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

st.divider()

#Main content tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Ekstraksi Fitur/Nilai Piksel", "Latih Model Klasifikasi", "Ringkasan Hasil Latih dan Evaluasi Model Klasifikasi", "Visualisasi", "Unduh Hasil Klasifikasi"])
#write each the content for each tab
# ==================== Tab 1: Feature Extraction ====================
#Option to either use all of the training data for classification, or split them into train and test data
#This section can be change to module 3 (?)
with tab1:
    st.header("Pengaturan ekstraksi data citra")
    markdown = """ 
    Langkah pertama dalam klasifikasi adalah mengekstrak nilai piksel dari data citra untuk setiap kelas ROI. Sebelum mengekstrak piksel, Anda harus menentukan apakah akan membagi ROI menjadi data pelatihan dan pengujian.
    Jika Anda memutuskan untuk membagi data, Anda akan dapat mengevaluasi model klasifikasi sebelum menghasilkan klasifikasi tutupan lahan.
    Jika Anda memutuskan untuk tidak membagi data, Anda tidak dapat mengevaluasi kualitas model, dan hanya dapat menghitung akurasi tematik di modul 7.
    """
    st.markdown(markdown)
    
    col1, col2 = st.columns([1, 1])
    #first column, provide option to split or not split
    with col1:
        st.subheader("Opsi Pembagian Data")
        # Option to split data
        split_data = st.checkbox(
            "Bagi data menjadi subset pelatihan dan pengujian",
            value=True,
            help="Jika tidak dicentang, seluruh data referensi akan digunakan untuk melatih model klasifikasi"
        )
        #What happened if the user choose to split the data
        if split_data:
            st.info("ROI dibagi menjadi data pelatihan dan pengujian menggunakan pendekatan pembagian acak berstrata")
            
            #Split ratio
            split_ratio = st.slider(
                "Training Data Ratio",
                min_value=0.5,
                max_value=0.9,
                value=0.7,
                step=0.05,
                help="Proporsi data yang digunakan untuk pelatihan"
            )
            #information about the proportion
            st.metric("Training", f"{split_ratio*100:.0f}%", delta=None)
            st.metric("Testing", f"{(1-split_ratio)*100:.0f}%", delta=None)
        #What happened if the user choose not to split the data    
        else:
            st.warning("Seluruh data ROI akan digunakan untuk pelatihan. Siapkan dataset pengujian independen.")
    #Second column, prepared the extraction parameters 
    with col2:
        st.subheader("Parameter ekstrasi data")
        #Get class property from previous module if available. What the user choose for separability analysis, will be used here
        default_class_prop = st.session_state.get('selected_class_property', 'class')
        #Class property name
        class_property = st.text_input(
            "Class ID",
            value=default_class_prop,
            help="Nama kolom tabel atribut yang berisi ID kelas numerik"
        )
        # Pixel size
        pixel_size = st.number_input(
            "Pixel Size (meters)",
            min_value=1,
            max_value=1000,
            value=30,
            help="Resolusi spasial untuk pengambilan sampel"
        )
    st.markdown("---")
    
    #Extract Features button
    if st.button("Ekstrak Fitur", type="primary", use_container_width=True):
        #Spinner to show progress
        with st.spinner("Mengekstrak fitur dari citra..."):
            try:
                if "train_final_ee" in st.session_state and st.session_state["train_final_ee"] is not None:
                    roi_ee = st.session_state["train_final_ee"]
                else:
                    st.warning("⚠️ Data latih belum divalidasi. Melakukan validasi sekarang...")
                    from epistemx.shapefile_utils import shapefile_validator, EE_converter
                    
                    validate = shapefile_validator(verbose=False)
                    converter = EE_converter(verbose=False)
                    train_gdf = st.session_state["train_final"]
                    train_gdf_cleaned = validate.validate_and_fix_geometry(train_gdf, geometry="mixed")
                    
                    if train_gdf_cleaned is None or train_gdf_cleaned.empty:
                        st.error("❌ Validasi geometri data latih gagal. Data tidak valid atau kosong.")
                        st.stop()
                    roi_ee = converter.convert_roi_gdf(train_gdf_cleaned)
                    
                    if roi_ee is None:
                        st.error("❌ Gagal mengonversi data latih ke format Google Earth Engine.")
                        st.stop()
                    st.session_state["train_final_ee"] = roi_ee
                    st.success("✅ Data latih berhasil divalidasi dan dikonversi ke format Earth Engine")
                
                #Use module 6 feature extraction class 
                fe = FeatureExtraction()
                #define the spliting function from the source code
                if split_data:
                    training_data, testing_data = fe.stratified_split(
                        roi=roi_ee,
                        image=image,
                        class_prop=class_property,
                        pixel_size=pixel_size,
                        train_ratio=split_ratio,
                    )
                    #stored the result in session state so that it can be used in classification
                    st.session_state.extracted_training_data = training_data
                    st.session_state.extracted_testing_data = testing_data
                    st.session_state.class_property = class_property
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Training Samples", training_data.size().getInfo())
                    with col2:
                        st.metric("Testing Samples", testing_data.size().getInfo())
                    st.success("✅ Ekstraksi fitur selesai!")
                else:
                    training_data = image.sampleRegions(
                        collection=roi_ee,
                        properties=[class_property],
                        scale=pixel_size,
                    )
                    #store the data for the classification
                    st.session_state.extracted_training_data = training_data
                    st.session_state.extracted_testing_data = None
                    st.session_state.class_property = class_property
                    st.info("ℹ️ Semua data ROI telah digunakan untuk pelatihan. Tidak ada set pengujian yang dibuat.")
            #error log if something fail    
            except Exception as e:
                st.error(f"Kesalahan saat ekstraksi fitur: {e}")
                import traceback
                st.code(traceback.format_exc())

# ==================== Tab 2: Model Learning ====================
with tab2:
    st.header("Membuat model klasifikasi")
    
    #introduction
    st.markdown("""
    Di bagian ini, dilakukan proses klasifikasi digital untuk mengelompokkan pola penutup lahan pada citra satelit.
    Bayangkan anda menyuruh komputer untuk mengenali pola - pola, selayaknya anda melihat pola penutup lahan yang berbeda secara visual.
    """)
    
    #Algorithm explanation with visual
    with st.expander("🤔 Bagaimana Model Random Forest Mengenali Pola? (Click to learn more)", expanded=False):
        st.markdown("""
        **Random Forest:** Bayangkan model ini sebagai sekelompok ilmuwan ('pohon') 
        yang memberikan suara (voting) terkait jenis piksel pada citra satelit. 
        Proses pengelompokan nilai piksel menjadi kelas penutup lahan adalah sebagai berikut:
        
        🌲 **Setiap "Pohon"** mempertimbangkan kombinasi nilai piksel yang berbeda pada setiap kanal spektral
        
        🗳️ **Pengambilan Keputusan** Pohon ini kemudian menentukan tipe penutup lahan yang diwakili oleh setiap nilai piksel
        
        📊 **Keputusan Akhir** ditetapkan melalui pengambilan suara terbanyak, apapun yang disetujui oleh sebagian besar pohon akan menjadi keputusan terakhir
        
        **Fun fact: Random Forest menjadi salah satu algoritma yang banyak digunakan dalam kajian penginderaan jauh**
        - Dapat diandalkan karena proses penentuan kelas dilakukan melalui kumpulan 'pendapat ahli'
        - Dapat menghadapi berbagai jenis kondisi data (tidak seimbang, atau penuh dengan noise)
        """)
    
    #Check if training data is available
    if st.session_state.extracted_training_data is None:
        st.warning("⚠️ Lakukan ekstraksi nilai piksel melalui 'ekstraksi fitur'")
    else:
        st.success("✅ Proses ekstraksi nilai piksel selesai. Proses pelatihan model dapat dilakukan")
        
        #Model Config with explanations
        st.subheader("⚙️ Pengaturan Model Klasifikasi")
        with st.expander("Kenapa Model klasifikasi perlu diatur?", expanded = False):
            st.markdown(""" 
            Setiap model machine learning memiliki beberapa parameter yang mengendalikan bagaimana mesin
            mempelajari hubungan antara variabel dan pola data yang diberikan. Oleh karena itu, 
            pengaturean parameter ini dapat mempengaruhi kualitas model dan klasifikasi yang dihasilkan.
            """
            )
            st.markdown("Algoritma Random Forest memiliki beberapa parameter utama yang mempengaruhi kemampuannya untuk mempelajari pola")
            st.markdown("1. Jumlah Pohon Keputusan (number of trees)")
            st.markdown("2. Jumlah variabel yang dipertimbangkan saat pengambilan keputusan (variable_per_split)")
            st.markdown("3. Jumlah sampel yang dipertimbangkan untuk memecah sebuah daun dalam pohon keputusan (min leaf population)")
        
        #Create tabs for preset value, or manuall setting
        config_tab1, config_tab2 = st.tabs(["Pengaturan Umum", "Pengaturan manual"])
        
        # Initialize default values before tabs
        ntrees = 150
        v_split = None
        min_leaf = 1
        use_auto_vsplit = True
        
        #Preset parameter value
        with config_tab1:
            col1, col2 = st.columns(2)
            st.markdown("Pengaturan umum yang diterapkan untuk kajian penginderaan jauh")
            with col1:
                st.markdown("### Jumlah Pohon Keputusan")
                st.markdown("*Berapa banyak 'pendapat ahli' diperlukan?*")
                
                #Predefined preset (?)
                tree_preset = st.radio(
                    "Preset:",
                    ["Stable: 50 trees (Ideal untuk klasifikasi penutup lahan dengan kompleksitas rendah)", 
                     "Balanced: 150 trees (Ideal untuk klasifikasi penutup lahan dengan kompleksitas menengah)", 
                     "Complex: 300 trees (Ideal untuk klasifikasi penutup lahan dengan kompleksitas tinggi) "],
                    index=1,
                    help="Semakin banyak jumlah pohon, akurasi dapat meningkat, namun beban komputasi yang semakin tinggi"
                )
                #translate the preset to the machine requirement
                if "Stable" in tree_preset:
                    ntrees = 50                    
                elif "Balanced" in tree_preset:
                    ntrees = 150                   
                else:
                    ntrees = 300
            #Other option beside number of trees
            with col2:
                st.markdown("### Pengaturan lainnya")
                st.markdown("*Parameter lainnya menggunakan nilai bawaan (default)*")
                #default variable persplit
                use_auto_vsplit = True
                v_split = None
                #default min leaf
                min_leaf = 1
                
                st.success("✅ *Variables per split*: default (akar dari jumlah total variabel)")
                st.success("✅ *Minimum samples*: 1 (default)")
        #advance option, used for manually specified parameter
        with config_tab2:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("### Number of Trees")
                ntrees = st.number_input(
                    "Number of Trees",
                    min_value=10,
                    max_value=900,
                    value=100,
                    step=10,
                    help="Jumlah pohon keputusan"
                )
                
            with col2:
                st.markdown("### Variable per split")
                use_auto_vsplit = st.checkbox(
                    "Menggunakan nilai bawaan",
                    value=True,
                    help="Menggunakan nilai bawaan berdasarkan data yang digunakan"
                )
                
                if not use_auto_vsplit:
                    v_split = st.number_input(
                        "Variables Per Split",
                        min_value=1,
                        max_value=50,
                        value=5,
                        help="Berapa banyak variabel yang dipertimbangkan saat pengambilan keputusan (split)"
                    )
                else:
                    v_split = None
                    st.success("✅ Menggunakan √(dari jumlah variabel/prediktor)")
            
            with col3:
                st.markdown("### Minimum Sample Leaf")
                min_leaf = st.number_input(
                    "Minimum Samples per Leaf",
                    min_value=1,
                    max_value=100,
                    value=1,
                    help= "Jumlah minimal sampel yang dipertimbangkan saat mengambil keputusan"
                )
                
        #Ready to train section
        st.markdown("---")
        
        # Show current configuration summary
        with st.expander("📋 Konfigurasi Model", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🌲 Jumlah Pohon", ntrees)
            with col2:
                if v_split is None:
                    st.metric("🔀 Variables per Split", "Default")
                else:
                    st.metric("🔀 Variables per Split", v_split)
            with col3:
                st.metric("🍃 Min Samples per Leaf", min_leaf)
        
        # classification button
        if st.button(" Latih Model Klasifikasi", type="primary", use_container_width=True):
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("🔄 Initializing Random Forest model...")
            progress_bar.progress(10)
            
            try:
                #Initialize the generate lulc class from module 6
                lulc = Generate_LULC()
                #Get the class property used during extraction
                clf_class_property = st.session_state.get('class_property')
                
                status_text.text(f"🌱 Training {ntrees} decision trees...")
                progress_bar.progress(30)
                
                #Run the hard classification
                classification_result, trained_model = lulc.hard_classification(
                    training_data=st.session_state.extracted_training_data,
                    class_property=clf_class_property,
                    image=image,
                    ntrees=ntrees,
                    v_split=v_split,
                    min_leaf=min_leaf,
                    return_model=True
                )
                
                status_text.text("Saving results...")
                progress_bar.progress(80)
                
                #Store the results for visualization and evaluation
                st.session_state.classification_mode = "Hard Classification"
                st.session_state.trained_model = trained_model
                st.session_state.classification_result = classification_result
                st.session_state.classification_params = {
                    'mode': 'Hard Classification',
                    'ntrees': ntrees,
                    'v_split': v_split,
                    'min_leaf': min_leaf,
                    'class_property': clf_class_property
                }
                
                progress_bar.progress(100)
                
                # Success message with next steps
                st.success("🎉 **Selamat!** Model klasifikasi telah berhasil dilatih!")
                st.info("👉 **Apa selanjutnya?** Pergi ke sub-bagian 'Ringkasan Hasil Latih dan Evaluasi Model Klasifikasi' untuk melihat performa model klasifikasi!")
                
            except Exception as e:
                progress_bar.progress(0)
                status_text.text("")
                st.error("❌ **Ups! Terjadi kesalahan saat pelatihan.**")
                st.error("**Detail kesalahan:** " + str(e))
                
                with st.expander("🔧 Detail Teknis"):
                    import traceback
                    st.code(traceback.format_exc())
# ==================== TAB 3 Summary Result ====================
with tab3:
    #Lets dump some exposition for this tab
    st.header("Ulasan Model Klasifikasi")
    st.markdown("""
    Melalui bagian ini, anda dapat mengulas proses latih model klasifikasi yang telah dilakukan
    Platform EPISTEM mendukung dua pendekatan untuk mengulas kemampuan pembelajaran mesin:""")
    st.markdown("1. Feature Importance: Kanal citra mana yang mengandung informasi terpenting untuk pembelajaran model?")
    st.markdown("2. Akurasi Model: Bagaimana model klasifikasi menghadapi data yang baru?")
    
    #Column for feature importance and Model accuracy explanation
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**📊 Feature Importance**")
        with st.expander("Apa itu Feature Importance?", expanded=False):
            st.markdown("""
             Analisis tingkat kepentingan fitur merupakan salah satu umpan balik model
             yang bertujuan untuk memberikan informasi kontribusi setiap fitur (dalam konteks ini adalah kanal citra satelit)
             terhadap pembelajaran mesin. Kanal yang memberikan kontribusi paling kecil terhadap model dapat dihilangkan sehingga
             kemampuan pembelajaran model dapat meningkat
            """
                )
    with col2:
        st.markdown("**🎯 Akurasi Model**") 
        with st.expander("Apa itu evaluasi model?", expanded=False):
            st.markdown("""
            Salah satu kelebihan klasifikasi berbasis pembelajaran mesin adalah kemampuan untuk 
            melakukan evaluasi proses pembelajaran sebelum menghasilkan klasifikasi untuk seluruh citra.
            Evaluasi ini bertujuan untuk melihat bagaimana model melakukan klasifikasi terhadap data yang baru.
            Pendekatan evaluasi ini mirip dengan pengujian akurasi pada peta, namun hal yang membedakan adalah 
            objek yang diuji. Dalam konteks evaluasi model, objek yang diuji adalah prediksi statistik. 
            Jika model belum menghasilkan akurasi yang memuaskan, maka dapat dilakukan pelatihan ulang terhadap model 
            
            """
                )
    
    # Check if classification model is available
    if st.session_state.classification_result is None:
        st.warning("Selesaikan proses pembelajaran model terlebih dahulu!")
        st.stop()
    
    # Check if trained model exists
    if 'trained_model' not in st.session_state:
        st.error("Model terlatih tidak ditemukan. Silakan jalankan ulang klasifikasi.")
        st.stop()
    
    st.divider()
    
    # ==== Feature Importance ====
    st.subheader("📊 Feature Importance Analysis")
    
    with st.expander("Apa yang ditunjukan grafik ini?", expanded=False):
        st.markdown("""
        Grafik ini menunjukan kanal mana yang sangat berguna untuk identifikasi kelas penutup lahan 
        
        - **Nilai yang tinggi** = Lebih penting untuk klasifikasi 
        - **Nilai yang rendah** = Kurang penting untuk proses klasifikasi
        """)
    
    try:
        lulc = Generate_LULC()
        # Get additional parameters for fallback method
        training_data = st.session_state.get('extracted_training_data')
        class_property = st.session_state.get('class_property')
        
        importance_df = lulc.get_feature_importance(
            st.session_state.trained_model,
            training_data=training_data,
            class_property=class_property
        )
        st.session_state.importance_df = importance_df
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = px.bar(
                importance_df,
                x='Importance',
                y='Band',
                orientation='h',
                title='Kanal mana yang paling penting?',
                color='Importance',
                color_continuous_scale='Viridis',
                text='Importance'
            )
            
            fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
            fig.update_layout(
                yaxis={'categoryorder': 'total ascending'},
                height=max(400, len(importance_df) * 30),
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("**Kanal yang paling penting:**")
            for i, row in importance_df.head(5).iterrows():
                st.write(f"{i+1}. {row['Band']}")
            
            with st.expander("View All Bands"):
                st.dataframe(
                    importance_df.style.background_gradient(
                        subset=['Importance'],
                        cmap='YlGn'
                    ),
                    use_container_width=True,
                    hide_index=True
                )
    except Exception as e:
        st.error(f"Could not analyze feature importance: {e}")
    
    st.divider()
    
    # ==== Model Evaluation ====
    st.subheader("🎯 Model Accuracy Assessment")
    
    with st.expander("Bagaimana model diuji?", expanded=False):
        st.markdown("""
        Pengujian model dilakukan dengan menerapkan model kepada data yang tidak digunakan dalam proses pembelajaran pola
        sehingga kualitas hasil pembelajaran model dapat diketahui. Hasil pengujian kemudian dilaporkan melalui metrik akurasi
        pada tingkat keseluruhan maupun per-kelas penutup penggunaan lahan.
        
        **Metric Akurasi:**
        - **Akurasi Keseluruhan/Overall Accuracy**: Persentasi piksels yang diklasifikasikan secara benar
        - **Koefisien Kappa**: Tingkat kesepakatan antara model dan data penguji
        - **F1-Score**: Tingkat rata - rata harmonik antara metrik presisi (precision) dan sensitivitas (sensitivity)
        - **G-mean**: 
        """)
    #check the model test data avaliability
    have_test_data = st.session_state.extracted_testing_data is not None
    #if its not there
    #user still able to visualize the map
    if not have_test_data:
        st.info("💡 Tidak ada data pengujian untuk penilaian akurasi")
        st.markdown("""
        **Untuk mengevaluasi akurasi model:**
        1. Kembali ke tab 'Ekstraksi Fitur/Nilai Piksel'
        2. Centang 'Bagi data menjadi subset pelatihan dan pengujian'
        3. Jalankan ulang ekstraksi fitur dan pelatihan model
        4. Kembali ke sini untuk melihat hasil akurasi
        """)
    #If there's data, capability to calculate model accuracy
    else:
        if st.button("Hitung Akurasi Model", type="primary"):
            with st.spinner("menguji model..."):
                try:
                    lulc = Generate_LULC()
                    class_prop = st.session_state.get('classification_params', {}).get('class_property')
                    
                    model_quality = lulc.evaluate_model(
                        trained_model=st.session_state.trained_model,
                        test_data=st.session_state.extracted_testing_data,
                        class_property=class_prop
                    )
                    #stored the model for model accuracy assessment
                    st.session_state.model_quality = model_quality
                    st.success("✅ Penilaian akurasi selesai!")
                    
                except Exception as e:
                    st.error(f"Error during evaluation: {e}")
                    st.code(traceback.format_exc())
        
        # Show results if available
        if "model_quality" in st.session_state:
            st.subheader("📈 Akurasi Tingkat Keseluruhan Model")
            st.markdown("Berikut adalah kualitas model pada tingkat keseluruhan. Akurasi pada tingkat kelas disajikan pada bagian setelah ini")

            #get model quality stored in st session state
            acc = st.session_state.model_quality
            
            #Overall metrics (OA, kappa, f1, gmean)
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                oa = acc['overall_accuracy'] * 100
                st.metric("Overall Accuracy", f"{oa:.1f}%")
            
            with col2:
                kappa = acc['kappa']
                st.metric("Kappa Coefficient", f"{kappa:.3f}")
            
            with col3:
                mean_f1 = sum(acc['f1_scores']) / len(acc['f1_scores'])
                st.metric("Average F1-Score", f"{mean_f1:.3f}")
            
            with col4:
                overall_gmean = acc.get('overall_gmean', 0)
                st.metric("G-Mean Score", f"{overall_gmean:.3f}")
            
            # Interpretation guidelines
            with st.expander("📖 Contoh Interpretasi Akurasi Model", expanded=False):
                st.markdown("""
                **Overall Accuracy (Akurasi Keseluruhan):**
                - **≥ 85%**: Akurasi yang baik untuk sebagian besar kajian penutup/penggunaan lahan
                - **70-84%**: Akurasi sedang, dapat digunakan untuk kajian tertentu 
                - **< 70%**: Akurasi rendah, disarankan untuk melatih ulang model
                
                **Kappa Coefficient:**
                - **≥ 0.8**: Kesepakatan yang kuat antara model dan data referensi
                - **0.6-0.79**: Kesepakatan sedang antara model dan data referensi
                - **< 0.6**: Kesepakatan lemah antara model dan data referensi
                
                **F1-Score & G-Mean:**
                - **Nilai mendekati 1.0**: Performa yang baik. Model mampu menangkap pola baik kelas dominan maupun minoritas, sehingga dapat melakukan klasifikasi dengan ideal
                - **Nilai mendekati 0.5**: Performa sedang. Model menangkap pola yang kurang baik, sehingga terdapat kemungkinan kesalahan klasifikasi
                - **Nilai mendekati 0.0**: Performa rendah. Model belum menangkap pola data, sehingga terdapat kemungkinan besar kesalahan kalsifikasi 
                
                💡 **Catatan:** Interpretasi ini bersifat umum. Standar akurasi dapat bervariasi tergantung studi yang dilakukan dan kompleksitas skema klasifikasi.
                """)
            
            st.markdown("---")
            
            # Class-level results
            st.subheader("📋 Akurasi Tingkat Kelas")
            st.markdown("""
            Akurasi pada tingkat kelas dapat digunakan untuk menilai kualitas model pada kelas tertentu. 
            Selain dari F1-score and G-mean, terdapat metrik akurasi lain yang digunakan untuk menilai akurasi pada tingkat kelas:
            - **Recall/Producer's Accuracy**:  Akurasi ini menjawab pertanyaan 'Seberapa baik algoritma memetakan kelas yang ada di lapangan?'.
            Metrik ini memberikan informasi mengenai kesalahan omisi, yaitu ketika data dari kelas yang benar tidak terdeteksi atau terlewat oleh model.
            - **Precision/User's Accuracy**: Akurasi ini menjawab pertanyaan 'Seberapa dipercayanya hasil klasifikasi kelas tertentu?'
            Metrik ini memberikan informasi mengenai kesalahan komisi, yaitu ketika model melakukan kesalahan klasifikasi dengan memasukkan data dari kelas lain ke dalam kelas tersebut.
            """)
            #Get class names from Module 2 if available
            class_names = []
            if 'lulc_classes_final' in st.session_state:
                # Create a mapping from class ID to class name
                class_id_to_name = {}
                for cls in st.session_state['lulc_classes_final']:
                    class_id = cls.get('ID', cls.get('Class ID'))
                    class_name = cls.get('Class Name', cls.get('Land Cover Class', f'Class {class_id}'))
                    class_id_to_name[class_id] = class_name
                
                # Create class names list
                for i in range(len(acc["precision"])):
                    if i in class_id_to_name:
                        class_names.append(class_id_to_name[i])
                    else:
                        class_names.append(f"Class {i}")
            else:
                # Fallback to generic class names if Module 2 data not available
                class_names = [f"Class {i}" for i in range(len(acc["precision"]))]
            
            df_metrics = pd.DataFrame({
                "Class ID": range(len(acc["precision"])),
                "Class Name": class_names,
                "Recall/Producer's Accuracy (%)": np.round(np.array(acc["recall"]) * 100, 1),
                "Precision/User's Accuracy (%)": np.round(np.array(acc["precision"]) * 100, 1),
                "F1-Score (%)": np.round(np.array(acc["f1_scores"]) * 100, 1),
                "G-Mean Score (%)": np.round(np.array(acc["gmean_per_class"]) * 100, 1)
            })
            
            st.dataframe(df_metrics, use_container_width=True)
            
            # Confusion Matrix
            st.subheader("🔍 Confusion Matrix")
            st.markdown("Menunjukkan seberapa sering setiap kelas berhasil diidentifikasi dengan benar dibandingkan dengan yang keliru diklasifikasikan sebagai kelas lain.")
            
            # Get class names from Module 2 if available
            class_labels = []
            if 'lulc_classes_final' in st.session_state:
                # Create a mapping from class ID to class name
                class_id_to_name = {}
                for cls in st.session_state['lulc_classes_final']:
                    class_id = cls.get('ID', cls.get('Class ID'))
                    class_name = cls.get('Class Name', cls.get('Land Cover Class', f'Class {class_id}'))
                    class_id_to_name[class_id] = class_name
                
                # Create labels for confusion matrix (ID: Name format)
                for i in range(len(acc["confusion_matrix"])):
                    if i in class_id_to_name:
                        class_labels.append(f"{i}: {class_id_to_name[i]}")
                    else:
                        class_labels.append(f"Class {i}")
            else:
                # Fallback to generic class labels if Module 2 data not available
                class_labels = [f"Class {i}" for i in range(len(acc["confusion_matrix"]))]
            
            cm = pd.DataFrame(
                acc["confusion_matrix"],
                columns=[f"Predicted {label}" for label in class_labels],
                index=[f"Actual {label}" for label in class_labels]
            )
            
            # Calculate dynamic height based on number of classes
            num_classes = len(acc["confusion_matrix"])
            base_height = max(500, num_classes * 60)  # Minimum 500px, 60px per class
            
            fig = px.imshow(
                cm,
                text_auto=True,
                aspect="auto",
                color_continuous_scale="Blues",
                title="Confusion Matrix: Kelas Aktual vs Prediksi"
            )
            
            # Improve layout for better readability
            fig.update_layout(
                height=base_height,
                width=None,  # Let it use container width
                title={
                    'text': "Confusion Matrix: Actual vs Predicted Classes",
                    'x': 0.5,
                    'xanchor': 'center'
                },
                xaxis={
                    'tickangle': 45,
                    'side': 'bottom'
                },
                yaxis={
                    'tickangle': 0
                },
                font=dict(size=10),
                margin=dict(l=150, r=50, t=80, b=150)  # Add margins for labels
            )
            
            # Update text annotations for better visibility
            fig.update_traces(
                texttemplate="%{z}",
                textfont={"size": max(8, 14 - num_classes)},  # Smaller text for more classes
                hovertemplate="<b>Actual:</b> %{y}<br><b>Predicted:</b> %{x}<br><b>Count:</b> %{z}<extra></extra>"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add interpretation help
            with st.expander("📖 Cara membaca matriks kesalahan (error matrix)"):
                st.markdown("""
                **Memahami matriks kesalahan:**
                - **Baris (Actual)**: Label kelas sebenarnya dari data pengujian Anda
                - **Kolom (Predicted)**: Kelas yang diprediksi oleh model Anda
                - **Nilai diagonal**: Prediksi yang benar (semakin tinggi semakin baik)
                - **Nilai di luar diagonal**: Kesalahan klasifikasi (semakin rendah semakin baik)
                
                **Klasifikasi Sempurna**: Semua nilai akan berada di diagonal dengan nilai nol di tempat lain.
                
                **Masalah Umum yang Perlu Diperhatikan:**
                - Nilai yang tinggi di luar diagonal menunjukkan kebingungan antara kelas tertentu
                - Nilai yang rendah dalam satu baris menunjukkan model kesulitan mendeteksi kelas tersebut
                - Nilai yang tinggi dalam satu kolom menunjukkan model terlalu sering memprediksi kelas tersebut
                """)
            
            # Add summary statistics
            st.markdown("#### Ringkasan Matriks Kesalahan")
            col1, col2 = st.columns(2)
            
            with col1:
                # Calculate per-class accuracy (diagonal / row sum)
                cm_array = np.array(acc["confusion_matrix"])
                row_sums = cm_array.sum(axis=1)
                diagonal = np.diag(cm_array)
                per_class_accuracy = np.divide(diagonal, row_sums, out=np.zeros_like(diagonal, dtype=float), where=row_sums!=0) * 100
                
                accuracy_df = pd.DataFrame({
                    "Class": class_labels,
                    "Correct Predictions": diagonal,
                    "Total Samples": row_sums,
                    "Class Accuracy (%)": np.round(per_class_accuracy, 1)
                })
                
                st.markdown("**Ringkasan Tingkat Kelas:**")
                st.dataframe(accuracy_df, use_container_width=True, hide_index=True)
            
            with col2:
                # Show most confused classes
                st.markdown("**Kesalahan klasifikasi yang paling umum:**")
                misclass_data = []
                
                for i in range(len(cm_array)):
                    for j in range(len(cm_array)):
                        if i != j and cm_array[i][j] > 0:  # Off-diagonal elements
                            misclass_data.append({
                                'Actual': class_labels[i],
                                'Predicted': class_labels[j], 
                                'Count': cm_array[i][j]
                            })
                
                if misclass_data:
                    misclass_df = pd.DataFrame(misclass_data)
                    misclass_df = misclass_df.sort_values('Count', ascending=False).head(5)
                    st.dataframe(misclass_df, use_container_width=True, hide_index=True)
                else:
                    st.success("🎉 Kelasahan klasifikasi minimal.")

# ==================== TAB 4 Visualization ====================
#USE MODULE 2 CLASSIFICATION SCHEME 
with tab4:
    st.header("Visualisasi")
    st.markdown("""
    Pada bagian ini anda dapat melihat hasil klasifikasi yang telah dilakukan oleh model yang telah dilatih.
    Sistem akan menggunakan warna yang telah ditentukan di modul 2 untuk visualisasi hasil klasifikasi,
    namun anda masih bisa melakukan penyesuaian jika memang diperlukan
    """)
    if st.session_state.classification_result is None:
        st.info("ℹ️ Klasifikasi belum dilakukan, silahkan lakukan klasifikasi terlebih dahulu.")
    else:
        # Visualization section
        st.subheader("Pratinjau Hasil Klasifikasi")
        if st.checkbox("Tunjukan klasifikasi tutupan lahan", value=True):
            try:
                #Prepare visualization
                classification_map = st.session_state.classification_result
                # Get class information from Module 2 and training data
                class_info = {}
                palette = []
                unique_classes = []
                
                #First, try to get class info from Module 2
                if 'lulc_classes_final' in st.session_state:
                    lulc_classes = st.session_state['lulc_classes_final']
                    for cls in lulc_classes:
                        class_id = cls.get('ID', cls.get('Class ID'))
                        class_name = cls.get('Class Name', cls.get('Land Cover Class', f'Class {class_id}'))
                        color_code = cls.get('Color Code', cls.get('Color', '#228B22'))
                        class_info[class_id] = {
                            'name': class_name,
                            'color': color_code
                        }
                    unique_classes = sorted(class_info.keys())
                    palette = [class_info[cls]['color'] for cls in unique_classes]
                
                #if failed, use a default and or random color palette
                elif 'train_final' in st.session_state and 'selected_class_property' in st.session_state:
                    class_prop = st.session_state['selected_class_property']
                    class_name_prop = st.session_state.get('selected_class_name_property', None)
                    gdf = st.session_state['train_final']
                    unique_classes = sorted(gdf[class_prop].unique())
                    
                    # Create default color mapping if not from Module 2
                    default_colors = ['#228B22', '#0000FF', '#FF0000', '#FFFF00', '#8B4513', 
                                    '#808080', '#FFA500', '#00FFFF', '#FF00FF', '#90EE90']
                    
                    for idx, class_id in enumerate(unique_classes):
                        if class_name_prop and class_name_prop in gdf.columns:
                            class_name = gdf[gdf[class_prop] == class_id][class_name_prop].iloc[0]
                        else:
                            class_name = f"Class {class_id}"
                        
                        class_info[class_id] = {
                            'name': class_name,
                            'color': default_colors[idx % len(default_colors)]
                        }
                    
                    palette = [class_info[cls]['color'] for cls in unique_classes]
                
                # Create visualization parameters
                if unique_classes and palette:
                    vis_params = {
                        'min': min(unique_classes),
                        'max': max(unique_classes),
                        'palette': palette
                    }
                    
                    # Display legend before the map
                    st.subheader("🗺️ Legenda Peta")
                    
                    # Create legend in columns for better layout
                    num_cols = min(4, len(unique_classes))  # Max 4 columns
                    cols = st.columns(num_cols)
                    
                    for idx, class_id in enumerate(unique_classes):
                        with cols[idx % num_cols]:
                            class_name = class_info[class_id]['name']
                            color = class_info[class_id]['color']
                            
                            # Create colored legend item
                            st.markdown(
                                f"""
                                <div style='display: flex; align-items: center; margin-bottom: 8px;'>
                                    <div style='background-color: {color}; 
                                                width: 20px; height: 20px; 
                                                border: 1px solid #ccc; 
                                                margin-right: 8px; 
                                                border-radius: 3px;'></div>
                                    <span style='font-size: 14px;'><strong>{class_id}:</strong> {class_name}</span>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                    
                    # Option to customize colors (expandable section)
                    with st.expander("🎨 Rubah warna kelas", expanded=False):
                        st.markdown("Adjust colors for each land cover class:")
                        
                        # Create color pickers for each class
                        color_cols = st.columns(min(3, len(unique_classes)))
                        updated_colors = {}
                        
                        for idx, class_id in enumerate(unique_classes):
                            with color_cols[idx % len(color_cols)]:
                                class_name = class_info[class_id]['name']
                                current_color = class_info[class_id]['color']
                                
                                # Color picker
                                new_color = st.color_picker(
                                    f"Class {class_id}: {class_name}",
                                    value=current_color,
                                    key=f"viz_color_{class_id}"
                                )
                                updated_colors[class_id] = new_color
                        
                        # Update colors if changed
                        if st.button("🔄 Apply Color Changes"):
                            for class_id in unique_classes:
                                class_info[class_id]['color'] = updated_colors[class_id]
                            palette = [class_info[cls]['color'] for cls in unique_classes]
                            vis_params['palette'] = palette
                            st.success("Colors updated! Map will refresh automatically.")
                            st.rerun()
                        
                        # Reset to Module 2 colors button
                        if 'lulc_classes_final' in st.session_state:
                            if st.button("🔄 Reset to Module 2 Colors"):
                                # Reload colors from Module 2
                                lulc_classes = st.session_state['lulc_classes_final']
                                for cls in lulc_classes:
                                    class_id = cls.get('ID', cls.get('Class ID'))
                                    original_color = cls.get('Color Code', cls.get('Color', '#228B22'))
                                    class_info[class_id]['color'] = original_color
                                palette = [class_info[cls]['color'] for cls in unique_classes]
                                vis_params['palette'] = palette
                                st.success("Colors reset to Module 2 scheme!")
                                st.rerun()
                    
                    st.markdown("---")

                    # Create map
                    if 'train_final' in st.session_state:
                        gdf = st.session_state['train_final']
                        centroid = gdf.geometry.centroid.iloc[0]
                        Map = geemap.Map(center=[centroid.y, centroid.x], zoom=10)
                    else:
                        Map = geemap.Map()
                    
                    # Add layers
                    Map.addLayer(classification_map, vis_params, 'Land Cover Classification', True)
                    Map.addLayer(image, st.session_state.get('visualization', {}), 'Image Composite', False)
                    
                    # Add training data as overlay (optional)
                    if 'train_final' in st.session_state:
                        try:
                            Map.add_geojson(
                                st.session_state['train_final'].__geo_interface__, 
                                layer_name="Training Data", 
                                style={'color': 'yellow', 'weight': 2, 'fillOpacity': 0},
                                shown=False
                            )
                        except:
                            pass  # Skip if geojson conversion fails
                    
                    # Display the map
                    Map.to_streamlit(height=600)
                    
                    # Add map information
                    st.info("""
                    **Data yang ditampilkan di kanvas peta:**
                    - **Land Cover Classification**: Peta hasil klasifikasi anda
                    - **Image Composite**: Citra satelit yang digunakan untuk proses klasifikasi (aktifkan melalui kendali layar kanan atas kanvas peta)
                    - **Training Data**: data sampel yang digunakan untuk melatih model (aktifkan melalui kendali layar kanan atas kanvas peta)
                    """)
                    
                else:
                    st.error("Informasi kelas tidak tersedia")
                    st.info("Pastikan modul 2 telah selesai atau data sampel memiliki ID dan nama kelas yang unik")
                    
                
            except Exception as e:
                st.error(f"Error displaying map: {e}")
                st.code(traceback.format_exc())

# ==================== TAB 5 Export Classification ====================
#REUSE THE LOGIC FROM MODULE 1
with tab5:
    st.header("Simpan Hasil Klasifikasi")
    st.markdown("""
    Pada bagian ini anda dapat menyimpan hasil klasifikasi dengan dua pilihan:
    
    1. **Unduh Langsung**: Unduh berkas GeoTIFF langsung ke komputer Anda (Maksimal 32 Mb)
    2. **Google Cloud Storage**: Untuk area yang lebih besar atau jika unduhan langsung gagal
    
    **Tips penamaan berkas:** Gunakan format yang mudah dikenali seperti LULC_Area_Studi_Tahun_citra, contoh: LULC_Sumsel_2024_Landsat8
    """)
    
    #check if the classification result is complete
    if st.session_state.classification_result is None:
        st.info("ℹ️ Klasifikasi belum tersedia, silahkan jalankan klasifikasi terlebih dahulu")
    else:
        st.success("✅ Klasifikasi selesai!")
        
        # Export destination selection
        export_destination = st.radio(
            "Pilih tujuan ekspor:",
            ["Unduh Langsung", "Google Cloud Storage"],
            index=0,
            help="Pilih lokasi penyimpanan hasil klasifikasi"
        )
        
        #Create export settings
        with st.expander("Pengaturan Penyimpanan", expanded=True):
            #Classification Naming
            classification_params = st.session_state.get('classification_params', {})
            sensor = st.session_state.get('search_metadata', {}).get('sensor', 'unknown')
            start_date = st.session_state.get('search_metadata', {}).get('start_date', '')
            end_date = st.session_state.get('search_metadata', {}).get('end_date', '')
            #Default classification name
            default_name = f"LULC_{sensor}_{start_date}_{end_date}"
            export_name = st.text_input(
                "Nama berkas ekspor:",
                value=default_name,
                help="Hasil akan disimpan dalam format GeoTIFF (.tif)"
            )
            
            # Export destination specific settings
            if export_destination == "Unduh Langsung":
                st.info("Berkas akan diunduh langsung ke komputer Anda dalam format GeoTIFF")
                st.warning("Catatan: Unduhan langsung dibatasi maksimal 32 MB. Untuk area yang lebih besar, gunakan Google Cloud Storage.")
            
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
                    value="classification_exports/",
                    help="Awalan jalur opsional di dalam bucket (misal: 'classification_exports/' atau 'data/lulc/')"
                )
                
                if not gcs_bucket:
                    st.warning("⚠️ Nama GCS Bucket wajib diisi untuk ekspor ke Cloud Storage")
                else:
                    st.info(f"Berkas akan diekspor ke: gs://{gcs_bucket}/{gcs_path_prefix}{export_name}.tif")
            
            #Coordinate Reference System (CRS)
            crs_options = {
                "WGS 84 (EPSG:4326)": "EPSG:4326",
                "Custom EPSG": "CUSTOM"
            }
            crs_choice = st.selectbox(
                "Sistem Referensi Koordinat:",
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
            
            #Define the spatial resolution
            scale = st.number_input(
                "Ukuran Piksel (meter):",
                min_value=10,
                max_value=1000,
                value=30,
                step=10
            )
            
            # Button to start export
            export_button_text = f"Mulai ekspor ke {export_destination}"
            export_disabled = False
            
            # Disable button if GCS is selected but bucket name is missing
            if export_destination == "Google Cloud Storage" and not gcs_bucket:
                export_disabled = True
            
            if st.button(export_button_text, type="primary", disabled=export_disabled):
                try:
                    with st.spinner("Menyiapkan tugas ekspor…"):
                        #Use the classification result from session state
                        export_image = st.session_state.classification_result
                        
                        #Convert to integer format for classification maps
                        export_image = export_image.toInt()
                        
                        #Get the AOI from session state
                        aoi_obj = st.session_state.get('AOI') or st.session_state.get('aoi')
                        
                        if isinstance(aoi_obj, ee.FeatureCollection):
                            export_region = aoi_obj.geometry()
                        elif isinstance(aoi_obj, ee.Feature):
                            export_region = aoi_obj.geometry()
                        elif isinstance(aoi_obj, ee.Geometry):
                            export_region = aoi_obj
                        else:
                            # If all else fails, try to get bounds
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
                                    "fileFormat": "GEO_TIFF",
                                    "formatOptions": {"cloudOptimized": True, "noData": 0}
                                }
                                
                                # Get download URL
                                download_url = export_image.getDownloadURL(download_params)
                                
                                if download_url:
                                    st.success("✅ URL unduhan berhasil dibuat!")
                                    st.markdown(f"""
                                    **Detail Unduhan:**
                                    - Format: GeoTIFF (Integer)
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
                                                📥 Unduh Peta Tutupan Lahan
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
                            #Set format options for integer classification maps
                            format_options = {"cloudOptimized": True, "noData": 0}
                            
                            #Summarize the export parameters
                            export_params = {
                                "image": export_image,
                                "description": export_name.replace(" ", "_"),  
                                "bucket": gcs_bucket,
                                "fileNamePrefix": f"{gcs_path_prefix}{export_name}",
                                "scale": scale,
                                "crs": export_crs,
                                "maxPixels": 1e13,
                                "fileFormat": "GeoTIFF",
                                "formatOptions": format_options,
                                "region": export_region
                            }
                            
                            # Pass the parameters to earth engine export for Cloud Storage
                            task = ee.batch.Export.image.toCloudStorage(**export_params)
                            task.start()
                            
                            # Store task info in session state for monitoring
                            task_info = {
                                'id': task.id,
                                'name': export_name,
                                'destination': export_destination,
                                'folder': gcs_bucket,
                                'crs': export_crs,
                                'scale': scale,
                                'type': 'Classification',
                                'start_time': datetime.datetime.now(),
                                'last_progress': 0,
                                'last_update': datetime.datetime.now()
                            }
                            
                            # Append to export tasks list
                            st.session_state.export_tasks.append(task_info)
                            
                            st.success(f"✅ Tugas ekspor '{export_name}' berhasil dikirim!")
                            st.info(f"ID Tugas: {task.id}")
                            st.markdown(f"""
                            **Detail Ekspor:**
                            - Tujuan: Google Cloud Storage
                            - Lokasi berkas: gs://{gcs_bucket}/{gcs_path_prefix}{export_name}.tif
                            - CRS: {export_crs}
                            - Resolusi: {scale}m
                            - Format: GeoTIFF (Integer)
                            
                            Periksa progres di [Earth Engine Task Manager](https://code.earthengine.google.com/tasks) atau gunakan pemantau tugas di bawah ini.
                            """)
                        
                except Exception as e:
                    st.error(f"Gagal mengekspor: {str(e)}")
                    st.info("Informasi Pemecahan Masalah:")
                    st.write(f"Jenis wilayah kajian: {type(st.session_state.get('AOI', st.session_state.get('aoi')))}")
                    st.write(f"Hasil klasifikasi tersedia: {st.session_state.classification_result is not None}")
        
        #Earth Engine Export Task Monitor (reuse from Module 1 logic)
        if st.session_state.export_tasks:
            st.subheader("Earth Engine Export Monitor")
            
            #Manual refresh options with cache control
            col_refresh1, col_refresh2 = st.columns([1, 3])
            with col_refresh1:
                if st.button("🔄 Refresh All", key="refresh_all_classification"):
                    # Clear cache to force fresh data
                    st.session_state.task_cache.clear()
                    st.session_state.last_cache_update.clear()
                    st.rerun()
            
            with col_refresh2:
                # Show cache status
                active_tasks_count = len(get_active_tasks())
                total_tasks_count = len(st.session_state.export_tasks)
                st.caption(f"Monitoring {active_tasks_count}/{total_tasks_count} active tasks | Manual refresh only")
            
            # Display task status for each task
            for i, task_info in enumerate(st.session_state.export_tasks):
                # Only show classification tasks in this tab
                if task_info.get('type') == 'Classification':
                    with st.expander(f"Classification Task: {task_info['name']}", expanded=True):
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
                                st.write(f"**Type:** {task_info.get('type', 'N/A')}")
                                
                                # Individual task refresh button
                                if st.button(f"🔄", key=f"refresh_class_{i}", help="Refresh this task"):
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
                                
                                # Progress tracking
                                progress = status.get('progress', 0)
                                if progress > 0:
                                    st.progress(progress / 100.0)
                                    st.write(f"**Progress:** {progress:.1f}%")
                                elif state == 'RUNNING':
                                    st.progress(0)
                                    st.write("**Progress:** Initializing...")
                            
                            with col3:
                                # Format timestamps
                                creation_ts = status.get('creation_timestamp_ms')
                                update_ts = status.get('update_timestamp_ms')
                                
                                if creation_ts:
                                    creation_time = datetime.datetime.fromtimestamp(creation_ts / 1000)
                                    st.write(f"**Started:** {creation_time.strftime('%H:%M:%S')}")
                                
                                if update_ts:
                                    update_time = datetime.datetime.fromtimestamp(update_ts / 1000)
                                    st.write(f"**Updated:** {update_time.strftime('%H:%M:%S')}")
                                
                                # Show export details
                                st.write(f"**Format:** {task_info.get('format', 'N/A')}")
                                st.write(f"**Scale:** {task_info.get('scale', 'N/A')}m")
                            
                            # Show error message if failed
                            if state == 'FAILED' and 'error_message' in status:
                                st.error(f"Error: {status['error_message']}")
                            
                            # Show completion details
                            if state == 'COMPLETED':
                                st.success("✅ Classification export completed successfully!")
                                st.success(f"File saved to: [EPISTEM/EPISTEMX_Classification_Export Folder]({drive_url})")
                                
                                # Option to remove completed task from monitor
                                if st.button(f"Remove from monitor", key=f"remove_class_{i}"):
                                    st.session_state.export_tasks.pop(i)
                                    st.rerun()
                
                        except Exception as e:
                            st.error(f"Failed to get task status: {str(e)}")
                            st.write(f"Task ID: {task_info['id']}")
            
            # Clear completed classification tasks button
            completed_classification_tasks = []
            for task_info in st.session_state.export_tasks:
                if task_info.get('type') == 'Classification':
                    try:
                        status = get_cached_task_status(task_info['id'])
                        if status and status.get('state') == 'COMPLETED':
                            completed_classification_tasks.append(task_info)
                    except:
                        pass
            
            if completed_classification_tasks:
                if st.button("🗑️ Clear Completed Classification Tasks", key="clear_completed_class"):
                    st.session_state.export_tasks = [
                        task for task in st.session_state.export_tasks 
                        if task not in completed_classification_tasks
                    ]
                    st.rerun()

# Footer with navigation
st.divider()
st.subheader("Navigasi modul")

col1, col2 = st.columns(2)

with col1:
    if st.button("⬅️ Back to Module 3: Analyze ROI", use_container_width=True):
        st.switch_page("pages/4_Module_4_Analyze_ROI.py")

with col2:
    if st.session_state.classification_result is not None:
        if st.button("➡️ Go to Module 7: Thematic Accuracy Assessment", use_container_width=True):
            st.switch_page("pages/6_Module_7_Thematic_Accuracy.py")
            st.info("Modul uji akurasi akan segera tersedia!")
    else:
        st.button("🔒 Complete Classification First", disabled=True, use_container_width=True)

# Show completion status
if st.session_state.classification_result is not None:
    st.success("Anda telah menyelesaikan modul 6. Silahkan lanjut ke modul berikutnya")
else:
    st.info("💡 Complete feature extraction and classification to proceed")