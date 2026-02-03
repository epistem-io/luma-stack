"""
Module 6: Supervised Classification (Redesigned)

This module facilitates supervised classification using random forest classifier
with improved user journey and experience.

Architecture:
- Backend (classification.py): Pure backend process without UI dependencies
- Frontend (this file): Streamlit UI with session state management
- State synchronization ensures data persistence across page interactions

Key improvements:
- Replaced tabs with expanders for linear workflow
- Added floating table of contents
- Clearer step-by-step progression
- Better visual hierarchy
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import geemap.foliumap as geemap
from luma_ge.classification import FeatureExtraction, Generate_LULC
from modules.nav import Navbar
import numpy as np
import traceback
import ee
import datetime
from ui_helper import show_footer, show_header

# Page configuration
st.set_page_config(
    page_title="Luma Modul 6",
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

# Custom CSS for floating TOC and improved styling
st.markdown("""
<style>
/* Floating Table of Contents */
.floating-toc {
    position: fixed;
    top: 120px;
    right: 20px;
    width: 250px;
    background: rgba(255, 255, 255, 0.95);
    border-radius: 10px;
    padding: 20px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    z-index: 999;
    max-height: calc(100vh - 140px);
    overflow-y: auto;
}

.floating-toc h3 {
    font-size: 1.1em;
    margin-bottom: 15px;
    color: #1f1f1f;
    border-bottom: 2px solid #e0e0e0;
    padding-bottom: 8px;
}

.floating-toc a {
    display: block;
    padding: 8px 12px;
    margin: 4px 0;
    text-decoration: none;
    color: #555;
    border-radius: 5px;
    transition: all 0.2s;
    font-size: 0.9em;
}

.floating-toc a:hover {
    background: linear-gradient(90deg, rgba(255, 75, 145, 0.1), rgba(138, 43, 226, 0.1));
    color: #1f1f1f;
    transform: translateX(5px);
}

.floating-toc .completed {
    color: #00a67e;
    font-weight: 500;
}

.floating-toc .active {
    background: linear-gradient(90deg, rgba(255, 75, 145, 0.15), rgba(138, 43, 226, 0.15));
    font-weight: 600;
    color: #1f1f1f;
}

/* Step indicators */
.step-indicator {
    display: inline-block;
    width: 28px;
    height: 28px;
    border-radius: 50%;
    background: linear-gradient(135deg, var(--pink), var(--purple));
    color: white;
    text-align: center;
    line-height: 28px;
    font-weight: bold;
    margin-right: 10px;
    font-size: 0.9em;
}

.step-header {
    display: flex;
    align-items: center;
    margin-bottom: 15px;
}

.step-title {
    font-size: 1.3em;
    font-weight: 600;
    color: #1f1f1f;
}

/* Progress indicator */
.progress-bar {
    height: 8px;
    background: #e0e0e0;
    border-radius: 10px;
    margin: 20px 0;
    overflow: hidden;
}

.progress-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--pink), var(--purple));
    transition: width 0.3s ease;
    border-radius: 10px;
}

/* Metric cards */
.metric-card {
    background: white;
    border-radius: 10px;
    padding: 20px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    border-left: 4px solid transparent;
    border-image: linear-gradient(to bottom, var(--pink), var(--purple)) 1;
}

/* Responsive TOC */
@media (max-width: 1200px) {
    .floating-toc {
        display: none;
    }
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="breadcrumb">Modul 6 › Buat Peta Tutupan Lahan</div>
""", unsafe_allow_html=True)

# Title
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

# ========== Initialize Session State ==========
if 'extracted_training_data' not in st.session_state:
    st.session_state.extracted_training_data = None
if 'extracted_testing_data' not in st.session_state:
    st.session_state.extracted_testing_data = None
if 'classification_result' not in st.session_state:
    st.session_state.classification_result = None
if 'trained_classifier' not in st.session_state:
    st.session_state.trained_classifier = None
if 'export_tasks' not in st.session_state:
    st.session_state.export_tasks = []
if 'task_cache' not in st.session_state:
    st.session_state.task_cache = {}
if 'last_cache_update' not in st.session_state:
    st.session_state.last_cache_update = {}

# ========== Calculate Progress ==========
def calculate_progress():
    """Calculate workflow completion percentage"""
    steps_completed = 0
    total_steps = 4
    
    # Step 1: Prerequisites
    if ('composite' in st.session_state and st.session_state.composite is not None and
        'train_final' in st.session_state and st.session_state.train_final is not None):
        steps_completed += 1
    
    # Step 2: Feature Extraction
    if st.session_state.extracted_training_data is not None:
        steps_completed += 1
    
    # Step 3: Classification
    if st.session_state.classification_result is not None:
        steps_completed += 1
    
    # Step 4: Export (if tasks exist)
    if len(st.session_state.export_tasks) > 0:
        steps_completed += 1
    
    return (steps_completed / total_steps) * 100

progress_percentage = calculate_progress()

# ========== Floating Table of Contents ==========
st.markdown(f"""
<div class="floating-toc">
    <h3>📑 Daftar Isi</h3>
    <a href="#prasyarat-modul" class="{'completed' if progress_percentage > 0 else ''}">
        {'✓' if progress_percentage > 0 else '1.'} Prasyarat Modul
    </a>
    <a href="#ekstraksi-fitur" class="{'completed' if st.session_state.extracted_training_data is not None else ''}">
        {'✓' if st.session_state.extracted_training_data is not None else '2.'} Ekstraksi Fitur
    </a>
    <a href="#klasifikasi" class="{'completed' if st.session_state.classification_result is not None else ''}">
        {'✓' if st.session_state.classification_result is not None else '3.'} Klasifikasi
    </a>
    <a href="#visualisasi" class="{'completed' if st.session_state.classification_result is not None else ''}">
        {'✓' if st.session_state.classification_result is not None else '4.'} Visualisasi Hasil
    </a>
    <a href="#ekspor" class="{'completed' if len(st.session_state.export_tasks) > 0 else ''}">
        {'✓' if len(st.session_state.export_tasks) > 0 else '5.'} Ekspor Hasil
    </a>
    <div style="margin-top: 15px; padding-top: 15px; border-top: 1px solid #e0e0e0;">
        <div style="font-size: 0.85em; color: #666; margin-bottom: 5px;">Progres:</div>
        <div class="progress-bar">
            <div class="progress-fill" style="width: {progress_percentage}%;"></div>
        </div>
        <div style="font-size: 0.85em; color: #666; text-align: center;">{int(progress_percentage)}%</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ========== SECTION 1: Prerequisites Check ==========
st.markdown('<a id="prasyarat-modul"></a>', unsafe_allow_html=True)
st.markdown("""
<div class="step-header">
    <span class="step-indicator">1</span>
    <span class="step-title">Prasyarat Modul</span>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

# Check for image composite from Module 1
with col1:
    if 'composite' in st.session_state and st.session_state.composite is not None:
        st.success("✅ Citra satelit tersedia dari modul 1")
        image = st.session_state['composite']
        
        if 'Image_metadata' in st.session_state:
            metadata = st.session_state['Image_metadata']
            with st.expander("📊 Detail Citra"):
                st.write(f"**Sensor:** {st.session_state.get('search_metadata', {}).get('sensor', 'N/A')}")
                st.write(f"**Rentang Tanggal:** {metadata.get('date_range', 'N/A')}")
                st.write(f"**Total Citra:** {metadata.get('total_images', 'N/A')}")
    else:
        st.error("❌ Gabungan citra tidak tersedia")
        st.warning("Mohon selesaikan modul 1 untuk menghasilkan gabungan citra")
        image = None

# Check for training data from Module 3/4
with col2:
    if 'train_final' in st.session_state and st.session_state.train_final is not None:
        st.success("✅ Data sampel tersedia")
        roi = st.session_state['train_final']
        
        if 'train_final' in st.session_state:
            gdf = st.session_state['train_final']
            with st.expander("📊 Detail Data Pelatihan"):
                st.write(f"**Total Fitur:** {len(gdf)}")
                st.write(f"**Kolom:** {', '.join(gdf.columns.tolist())}")
                
                if 'selected_class_property' in st.session_state:
                    class_prop = st.session_state['selected_class_property']
                    class_name = st.session_state['selected_class_name_property']
                    if class_prop in gdf.columns:
                        class_counts = gdf[class_prop].value_counts()
                        st.write("**Distribusi Kelas:**")
                        st.dataframe(class_counts, use_container_width=True)
    else:
        st.error("❌ Data sampel tidak tersedia")
        st.warning("Mohon selesaikan modul 3 dan 4 untuk menghasilkan dan melakukan analisis data sampel")
        roi = None

# Stop if prerequisites are not met
if image is None or roi is None:
    st.divider()
    st.info("⚠️ Selesaikan modul-modul sebelumnya sebelum melanjutkan ke klasifikasi")
    st.markdown("""
    **Langkah yang Diperlukan:**
    1. **Modul 1:** Buat gabungan citra
    2. **Modul 2:** Definisikan skema klasifikasi 
    3. **Modul 3:** Unggah dan validasi data sampel
    4. **Modul 4:** Analisis keterpisahan data sampel
    5. **Modul 6:** Kembali ke sini untuk melakukan klasifikasi
    """)
    st.stop()

# Get AOI for clipping the result
aoi = st.session_state.get('AOI', None)

# Check if Module 5 has been completed with stacked predictors (optional)
module5_available = False
stacked_predictors_for_classification = None
if st.session_state.get("predictors_calculated", False) and st.session_state.get("stacked_predictors") is not None:
    module5_available = True
    stacked_predictors_for_classification = st.session_state["stacked_predictors"]
    st.info("ℹ️ Prediktor tambahan dari Modul 5 terdeteksi dan akan digunakan dalam klasifikasi")

st.divider()

# ========== SECTION 2: Feature Extraction ==========
st.markdown('<a id="ekstraksi-fitur"></a>', unsafe_allow_html=True)

with st.expander("**2️⃣ Ekstraksi Fitur Spektral**", expanded=st.session_state.extracted_training_data is None):
    st.markdown("""
    <div class="step-header">
        <span class="step-indicator">2</span>
        <span class="step-title">Ekstraksi Fitur Spektral</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    Ekstraksi fitur adalah proses pengambilan nilai spektral dari citra satelit pada lokasi sampel pelatihan. 
    Proses ini menghasilkan dataset yang berisi nilai-nilai spektral untuk setiap sampel, yang akan digunakan untuk melatih model klasifikasi.
    """)
    
    st.markdown("### Parameter Ekstraksi")
    
    col1, col2 = st.columns(2)
    
    with col1:
        scale_extraction = st.number_input(
            "Resolusi Spasial (meter):",
            min_value=10,
            max_value=1000,
            value=30,
            step=10,
            help="Resolusi spasial untuk ekstraksi fitur. Nilai default 30m sesuai dengan Landsat."
        )
    
    with col2:
        split_ratio = st.slider(
            "Rasio Pembagian Data (Training/Testing):",
            min_value=0.5,
            max_value=0.9,
            value=0.7,
            step=0.05,
            help="Proporsi data untuk pelatihan. Sisanya akan digunakan untuk testing."
        )
    
    st.info(f"📊 Data akan dibagi: **{int(split_ratio*100)}%** untuk pelatihan, **{int((1-split_ratio)*100)}%** untuk testing")
    
    if st.button("🚀 Jalankan Ekstraksi Fitur", type="primary", use_container_width=True):
        with st.spinner("Mengekstrak fitur dari citra satelit..."):
            try:
                # Get class properties
                class_property = st.session_state.get('selected_class_property', 'LULC_ID')
                
                # Determine which image to use
                if module5_available:
                    image_for_extraction = stacked_predictors_for_classification
                    st.info("✓ Menggunakan citra dengan prediktor tambahan dari Modul 5")
                else:
                    image_for_extraction = image
                
                # Initialize feature extractor
                extractor = FeatureExtraction(
                    image=image_for_extraction,
                    roi=roi,
                    class_property=class_property,
                    scale=scale_extraction
                )
                
                # Extract features
                training_fc = extractor.extract_features()
                
                if training_fc is None:
                    st.error("❌ Gagal mengekstrak fitur. Periksa kembali citra dan sampel Anda.")
                    st.stop()
                
                # Split into training and testing
                training_data, testing_data = extractor.split_sample(
                    training_fc,
                    split_ratio=split_ratio
                )
                
                # Store in session state
                st.session_state.extracted_training_data = training_data
                st.session_state.extracted_testing_data = testing_data
                
                st.success("✅ Ekstraksi fitur berhasil!")
                
                # Display statistics
                st.markdown("### Statistik Ekstraksi")
                
                col1, col2, col3 = st.columns(3)
                
                try:
                    total_samples = training_fc.size().getInfo()
                    training_samples = training_data.size().getInfo()
                    testing_samples = testing_data.size().getInfo()
                    
                    with col1:
                        st.metric("Total Sampel", total_samples)
                    with col2:
                        st.metric("Data Pelatihan", training_samples)
                    with col3:
                        st.metric("Data Testing", testing_samples)
                    
                except Exception as e:
                    st.warning(f"Tidak dapat menampilkan statistik detail: {str(e)}")
                
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error saat ekstraksi fitur: {str(e)}")
                st.code(traceback.format_exc())
    
    # Show extraction results if available
    if st.session_state.extracted_training_data is not None:
        st.success("✅ Fitur telah diekstrak dan siap untuk klasifikasi")
        
        with st.expander("📊 Lihat Informasi Data Ekstraksi"):
            try:
                # Get band names
                band_names = st.session_state.extracted_training_data.first().propertyNames().getInfo()
                st.write(f"**Fitur yang diekstrak:** {', '.join([b for b in band_names if b != st.session_state.get('selected_class_property', 'LULC_ID')])}")
                
                # Sample data preview
                sample_size = 5
                sample_data = st.session_state.extracted_training_data.limit(sample_size).getInfo()
                
                if sample_data and 'features' in sample_data:
                    df_sample = pd.DataFrame([f['properties'] for f in sample_data['features']])
                    st.dataframe(df_sample, use_container_width=True)
            except Exception as e:
                st.warning(f"Tidak dapat menampilkan preview data: {str(e)}")

st.divider()

# ========== SECTION 3: Classification ==========
st.markdown('<a id="klasifikasi"></a>', unsafe_allow_html=True)

with st.expander("**3️⃣ Pelatihan Model dan Klasifikasi**", expanded=st.session_state.extracted_training_data is not None and st.session_state.classification_result is None):
    st.markdown("""
    <div class="step-header">
        <span class="step-indicator">3</span>
        <span class="step-title">Pelatihan Model dan Klasifikasi</span>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.extracted_training_data is None:
        st.warning("⚠️ Mohon lakukan ekstraksi fitur terlebih dahulu")
    else:
        st.markdown("""
        Pada tahap ini, model Random Forest akan dilatih menggunakan data pelatihan yang telah diekstrak, 
        kemudian model tersebut akan digunakan untuk mengklasifikasikan seluruh area kajian.
        """)
        
        st.markdown("### Parameter Random Forest")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            n_trees = st.number_input(
                "Jumlah Pohon (*trees*):",
                min_value=10,
                max_value=500,
                value=100,
                step=10,
                help="Jumlah pohon keputusan dalam Random Forest. Lebih banyak pohon = lebih akurat tapi lebih lambat."
            )
        
        with col2:
            variables_per_split = st.number_input(
                "Variabel per Split:",
                min_value=1,
                max_value=20,
                value=None,
                help="Jumlah variabel yang dipertimbangkan pada setiap split. None = akar kuadrat dari total variabel (default)."
            )
        
        with col3:
            min_leaf_population = st.number_input(
                "Minimum Populasi Leaf:",
                min_value=1,
                max_value=100,
                value=1,
                help="Jumlah minimum sampel yang diperlukan untuk membentuk leaf node."
            )
        
        bag_fraction = st.slider(
            "Bag Fraction:",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="Fraksi data pelatihan yang digunakan untuk setiap pohon (bootstrap sampling)."
        )
        
        max_nodes = st.number_input(
            "Maximum Nodes:",
            min_value=1,
            max_value=1000,
            value=None,
            help="Jumlah maksimum node dalam setiap pohon. None = tidak dibatasi."
        )
        
        st.info(f"""
        **Parameter yang dipilih:**
        - Jumlah Pohon: {n_trees}
        - Variabel per Split: {variables_per_split if variables_per_split else 'Auto (√n)'}
        - Min Leaf Population: {min_leaf_population}
        - Bag Fraction: {bag_fraction}
        - Max Nodes: {max_nodes if max_nodes else 'Unlimited'}
        """)
        
        if st.button("🎯 Latih Model dan Klasifikasi", type="primary", use_container_width=True):
            with st.spinner("Melatih model Random Forest dan melakukan klasifikasi..."):
                try:
                    # Get class property
                    class_property = st.session_state.get('selected_class_property', 'LULC_ID')
                    
                    # Determine which image to use
                    if module5_available:
                        image_for_classification = stacked_predictors_for_classification
                    else:
                        image_for_classification = image
                    
                    # Get band names from extracted data
                    band_names = st.session_state.extracted_training_data.first().propertyNames().getInfo()
                    input_properties = [b for b in band_names if b != class_property]
                    
                    # Initialize classifier
                    lulc_generator = Generate_LULC(
                        image=image_for_classification,
                        training_data=st.session_state.extracted_training_data,
                        testing_data=st.session_state.extracted_testing_data,
                        class_property=class_property,
                        input_properties=input_properties
                    )
                    
                    # Train classifier
                    trained_classifier = lulc_generator.train_classifier(
                        numberOfTrees=n_trees,
                        variablesPerSplit=variables_per_split,
                        minLeafPopulation=min_leaf_population,
                        bagFraction=bag_fraction,
                        maxNodes=max_nodes
                    )
                    
                    # Classify
                    classified = lulc_generator.classify()
                    
                    # Clip to AOI if available
                    if aoi is not None:
                        classified = classified.clip(aoi)
                    
                    # Store results
                    st.session_state.classification_result = classified
                    st.session_state.trained_classifier = trained_classifier
                    
                    st.success("✅ Klasifikasi berhasil!")
                    
                    # Get variable importance if available
                    try:
                        importance_dict = trained_classifier.explain().getInfo()
                        if 'importance' in importance_dict:
                            st.markdown("### Pentingnya Variabel (*Feature Importance*)")
                            
                            importance_data = []
                            for band, importance in zip(input_properties, importance_dict['importance']):
                                importance_data.append({'Band': band, 'Importance': importance})
                            
                            df_importance = pd.DataFrame(importance_data).sort_values('Importance', ascending=False)
                            
                            # Create bar chart
                            fig = px.bar(
                                df_importance,
                                x='Importance',
                                y='Band',
                                orientation='h',
                                title='Pentingnya Fitur dalam Model Random Forest',
                                labels={'Importance': 'Nilai Pentingnya', 'Band': 'Fitur'}
                            )
                            fig.update_layout(height=400, showlegend=False)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            st.info("""
                            **Interpretasi Feature Importance:**
                            - Nilai lebih tinggi = fitur lebih penting untuk klasifikasi
                            - Fitur dengan pentingnya rendah mungkin dapat dihilangkan untuk menyederhanakan model
                            """)
                    except Exception as e:
                        st.warning(f"Tidak dapat menampilkan feature importance: {str(e)}")
                    
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error saat klasifikasi: {str(e)}")
                    st.code(traceback.format_exc())
        
        # Show classification results if available
        if st.session_state.classification_result is not None:
            st.success("✅ Klasifikasi telah selesai!")
            
            try:
                # Get class names and colors from Module 2
                if 'classes' in st.session_state and len(st.session_state.classes) > 0:
                    classes = st.session_state.classes
                    
                    st.markdown("### Skema Klasifikasi")
                    
                    # Create color palette for visualization
                    class_colors = []
                    class_values = []
                    class_names = []
                    
                    for cls in classes:
                        class_values.append(cls['id'])
                        class_names.append(cls['name'])
                        class_colors.append(cls['color'])
                    
                    # Display class legend
                    cols = st.columns(min(len(classes), 4))
                    for i, cls in enumerate(classes):
                        with cols[i % len(cols)]:
                            st.markdown(f"""
                            <div style="display: flex; align-items: center; margin-bottom: 10px;">
                                <div style="width: 30px; height: 30px; background-color: {cls['color']}; 
                                     border-radius: 5px; margin-right: 10px; border: 1px solid #ddd;"></div>
                                <div>
                                    <strong>{cls['name']}</strong><br>
                                    <small>ID: {cls['id']}</small>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                    
            except Exception as e:
                st.warning(f"Tidak dapat menampilkan skema klasifikasi: {str(e)}")

st.divider()

# ========== SECTION 4: Visualization ==========
st.markdown('<a id="visualisasi"></a>', unsafe_allow_html=True)

with st.expander("**4️⃣ Visualisasi Hasil Klasifikasi**", expanded=st.session_state.classification_result is not None):
    st.markdown("""
    <div class="step-header">
        <span class="step-indicator">4</span>
        <span class="step-title">Visualisasi Hasil Klasifikasi</span>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.classification_result is None:
        st.warning("⚠️ Mohon lakukan klasifikasi terlebih dahulu")
    else:
        st.markdown("Visualisasi hasil klasifikasi pada peta interaktif.")
        
        try:
            classified = st.session_state.classification_result
            aoi_ee = st.session_state.get('AOI')
            aoi_gdf = st.session_state.get('gdf')
            
            # Prepare visualization parameters
            if 'classes' in st.session_state and len(st.session_state.classes) > 0:
                classes = st.session_state.classes
                palette = [cls['color'] for cls in classes]
                min_val = min([cls['id'] for cls in classes])
                max_val = max([cls['id'] for cls in classes])
            else:
                palette = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00']
                min_val = 1
                max_val = 4
            
            vis_params = {
                'min': min_val,
                'max': max_val,
                'palette': palette
            }
            
            # Create map
            if aoi_gdf is not None:
                centroid = aoi_gdf.geometry.centroid.iloc[0]
                center = [centroid.y, centroid.x]
            else:
                center = [-2.5, 118.0]
            
            m = geemap.Map(center=center, zoom=10)
            m.addLayer(classified, vis_params, "Hasil Klasifikasi")
            
            # Add base image for comparison
            if 'composite' in st.session_state:
                composite = st.session_state['composite']
                rgb_vis = st.session_state.get('visualization', {})
                m.addLayer(composite, rgb_vis, "Citra Asli", False)
            
            # Add AOI
            if aoi_gdf is not None:
                m.add_geojson(aoi_gdf.__geo_interface__, layer_name="Area Kajian")
            elif aoi_ee is not None:
                m.addLayer(aoi_ee, {}, "Area Kajian", True, 0.3)
            
            # Add legend
            if 'classes' in st.session_state:
                legend_dict = {cls['name']: cls['color'] for cls in st.session_state.classes}
                m.add_legend(title="Kelas Tutupan Lahan", legend_dict=legend_dict)
            
            m.to_streamlit(height=700)
            
            st.info("""
            **💡 Tips Visualisasi:**
            - Gunakan layer control (⊟) di kanan atas untuk mengaktifkan/menonaktifkan layer
            - Bandingkan hasil klasifikasi dengan citra asli untuk validasi visual
            - Gunakan zoom dan pan untuk memeriksa detail klasifikasi
            """)
            
        except Exception as e:
            st.error(f"❌ Error saat memvisualisasikan hasil: {str(e)}")
            st.code(traceback.format_exc())

st.divider()

# ========== SECTION 5: Export ==========
st.markdown('<a id="ekspor"></a>', unsafe_allow_html=True)

with st.expander("**5️⃣ Ekspor Hasil Klasifikasi**", expanded=False):
    st.markdown("""
    <div class="step-header">
        <span class="step-indicator">5</span>
        <span class="step-title">Ekspor Hasil Klasifikasi</span>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.classification_result is None:
        st.warning("⚠️ Mohon lakukan klasifikasi terlebih dahulu")
    else:
        st.markdown("""
        Ekspor hasil klasifikasi ke Google Drive atau Google Cloud Storage untuk penggunaan lebih lanjut.
        File akan diekspor dalam format GeoTIFF yang dapat dibuka di software GIS seperti QGIS atau ArcGIS.
        """)
        
        # Export type selection
        export_type = st.radio(
            "Pilih Tujuan Ekspor:",
            ["Google Drive", "Google Cloud Storage"],
            horizontal=True
        )
        
        st.markdown("### Parameter Ekspor")
        
        col1, col2 = st.columns(2)
        
        with col1:
            export_name = st.text_input(
                "Nama File:",
                value=f"LULC_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
                help="Nama file untuk hasil ekspor (tanpa ekstensi)"
            )
            
            export_scale = st.number_input(
                "Resolusi Spasial (meter):",
                min_value=10,
                max_value=1000,
                value=30,
                step=10,
                help="Resolusi spasial untuk file hasil"
            )
        
        with col2:
            export_crs = st.text_input(
                "Sistem Koordinat (CRS):",
                value="EPSG:4326",
                help="Contoh: EPSG:4326 (WGS84), EPSG:32749 (UTM Zone 49S)"
            )
            
            export_maxPixels = st.number_input(
                "Maximum Pixels:",
                min_value=1e8,
                max_value=1e13,
                value=1e13,
                format="%.0e",
                help="Batas maksimum piksel untuk ekspor"
            )
        
        # Additional parameters based on export type
        if export_type == "Google Drive":
            folder_name = st.text_input(
                "Folder Google Drive:",
                value="EarthEngine",
                help="Nama folder di Google Drive (akan dibuat jika belum ada)"
            )
            
            if st.button("📤 Ekspor ke Google Drive", type="primary", use_container_width=True):
                with st.spinner("Mengirim tugas ekspor ke Earth Engine..."):
                    try:
                        task = ee.batch.Export.image.toDrive(
                            image=st.session_state.classification_result,
                            description=export_name,
                            folder=folder_name,
                            fileNamePrefix=export_name,
                            region=aoi.geometry() if hasattr(aoi, 'geometry') else aoi,
                            scale=export_scale,
                            crs=export_crs,
                            maxPixels=int(export_maxPixels),
                            fileFormat='GeoTIFF',
                            formatOptions={'cloudOptimized': True}
                        )
                        task.start()
                        
                        # Add to export tasks
                        st.session_state.export_tasks.append({
                            'id': task.id,
                            'name': export_name,
                            'type': 'Classification',
                            'destination': 'Google Drive',
                            'folder': folder_name,
                            'scale': export_scale,
                            'crs': export_crs,
                            'format': 'GeoTIFF',
                            'timestamp': datetime.datetime.now()
                        })
                        
                        st.success("✅ Tugas ekspor berhasil dikirim!")
                        st.info(f"""
                        **Detail Ekspor:**
                        - ID Tugas: {task.id}
                        - Tujuan: Google Drive
                        - Folder: {folder_name}
                        - Nama File: {export_name}.tif
                        - Resolusi: {export_scale}m
                        - CRS: {export_crs}
                        
                        Periksa progres di [Earth Engine Task Manager](https://code.earthengine.google.com/tasks)
                        """)
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Gagal mengekspor: {str(e)}")
        
        else:  # Google Cloud Storage
            col_gcs1, col_gcs2 = st.columns(2)
            
            with col_gcs1:
                gcs_bucket = st.text_input(
                    "Bucket Name:",
                    help="Nama bucket GCS (tanpa gs://)"
                )
            
            with col_gcs2:
                gcs_path_prefix = st.text_input(
                    "Path Prefix (opsional):",
                    value="luma/",
                    help="Prefix path di dalam bucket"
                )
            
            if st.button("📤 Ekspor ke GCS", type="primary", use_container_width=True):
                if not gcs_bucket:
                    st.error("❌ Mohon isi nama bucket GCS")
                else:
                    with st.spinner("Mengirim tugas ekspor ke Earth Engine..."):
                        try:
                            task = ee.batch.Export.image.toCloudStorage(
                                image=st.session_state.classification_result,
                                description=export_name,
                                bucket=gcs_bucket,
                                fileNamePrefix=gcs_path_prefix + export_name,
                                region=aoi.geometry() if hasattr(aoi, 'geometry') else aoi,
                                scale=export_scale,
                                crs=export_crs,
                                maxPixels=int(export_maxPixels),
                                fileFormat='GeoTIFF',
                                formatOptions={'cloudOptimized': True}
                            )
                            task.start()
                            
                            # Add to export tasks
                            st.session_state.export_tasks.append({
                                'id': task.id,
                                'name': export_name,
                                'type': 'Classification',
                                'destination': 'Google Cloud Storage',
                                'bucket': gcs_bucket,
                                'path': gcs_path_prefix,
                                'scale': export_scale,
                                'crs': export_crs,
                                'format': 'GeoTIFF',
                                'timestamp': datetime.datetime.now()
                            })
                            
                            st.success("✅ Tugas ekspor berhasil dikirim!")
                            st.info(f"""
                            **Detail Ekspor:**
                            - ID Tugas: {task.id}
                            - Tujuan: Google Cloud Storage
                            - Lokasi: gs://{gcs_bucket}/{gcs_path_prefix}{export_name}.tif
                            - Resolusi: {export_scale}m
                            - CRS: {export_crs}
                            
                            Periksa progres di [Earth Engine Task Manager](https://code.earthengine.google.com/tasks)
                            """)
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"❌ Gagal mengekspor: {str(e)}")

# ========== Task Monitor ==========
if st.session_state.export_tasks:
    st.divider()
    st.markdown("### 📊 Monitor Tugas Ekspor")
    
    # Helper functions for task monitoring
    def get_active_tasks():
        """Get list of currently active tasks"""
        active_tasks = []
        for task_info in st.session_state.export_tasks:
            try:
                status = get_cached_task_status(task_info['id'])
                if status and status.get('state') in ['READY', 'RUNNING']:
                    active_tasks.append(task_info)
            except:
                pass
        return active_tasks
    
    def get_cached_task_status(task_id, cache_duration=10):
        """Get task status with caching"""
        import time
        current_time = time.time()
        
        # Check if we have cached data
        if task_id in st.session_state.task_cache:
            last_update = st.session_state.last_cache_update.get(task_id, 0)
            if current_time - last_update < cache_duration:
                return st.session_state.task_cache[task_id]
        
        # Fetch fresh data
        try:
            task = ee.data.getTaskStatus(task_id)[0]
            st.session_state.task_cache[task_id] = task
            st.session_state.last_cache_update[task_id] = current_time
            return task
        except Exception as e:
            return None
    
    # Refresh button
    col_refresh1, col_refresh2 = st.columns([1, 3])
    with col_refresh1:
        if st.button("🔄 Refresh", key="refresh_all"):
            st.session_state.task_cache.clear()
            st.session_state.last_cache_update.clear()
            st.rerun()
    
    with col_refresh2:
        active_count = len(get_active_tasks())
        total_count = len(st.session_state.export_tasks)
        st.caption(f"Memantau {active_count}/{total_count} tugas aktif")
    
    # Display tasks
    for i, task_info in enumerate(st.session_state.export_tasks):
        if task_info.get('type') == 'Classification':
            with st.container():
                try:
                    status = get_cached_task_status(task_info['id'])
                    if not status:
                        st.error(f"Gagal mendapatkan status untuk: {task_info['name']}")
                        continue
                    
                    col1, col2, col3 = st.columns([2, 1, 1])
                    
                    with col1:
                        st.write(f"**{task_info['name']}**")
                        st.caption(f"Tujuan: {task_info.get('destination', 'N/A')}")
                    
                    with col2:
                        state = status.get('state', 'UNKNOWN')
                        if state == 'COMPLETED':
                            st.success(f"✅ {state}")
                        elif state == 'RUNNING':
                            progress = status.get('progress', 0)
                            st.progress(progress / 100.0 if progress > 0 else 0)
                            st.caption(f"Progress: {progress:.1f}%" if progress > 0 else "Initializing...")
                        elif state == 'FAILED':
                            st.error(f"❌ {state}")
                        else:
                            st.info(f"⏳ {state}")
                    
                    with col3:
                        if st.button("🔄", key=f"refresh_{i}", help="Refresh"):
                            if task_info['id'] in st.session_state.task_cache:
                                del st.session_state.task_cache[task_info['id']]
                            if task_info['id'] in st.session_state.last_cache_update:
                                del st.session_state.last_cache_update[task_info['id']]
                            st.rerun()
                    
                    # Show error if failed
                    if state == 'FAILED' and 'error_message' in status:
                        st.error(f"Error: {status['error_message']}")
                    
                    # Option to remove completed tasks
                    if state == 'COMPLETED':
                        if st.button(f"Hapus dari Monitor", key=f"remove_{i}"):
                            st.session_state.export_tasks.pop(i)
                            st.rerun()
                    
                    st.divider()
                    
                except Exception as e:
                    st.error(f"Error mendapatkan status tugas: {str(e)}")
    
    # Clear completed tasks
    completed_tasks = [t for t in st.session_state.export_tasks 
                      if t.get('type') == 'Classification' and 
                      get_cached_task_status(t['id']) and 
                      get_cached_task_status(t['id']).get('state') == 'COMPLETED']
    
    if completed_tasks:
        if st.button("🗑️ Hapus Semua Tugas Selesai", use_container_width=True):
            st.session_state.export_tasks = [
                t for t in st.session_state.export_tasks 
                if t not in completed_tasks
            ]
            st.rerun()

# ========== Footer Navigation ==========
st.divider()
st.subheader("Navigasi Modul")

col1, col2 = st.columns(2)

with col1:
    if st.button("⬅️ Kembali ke Modul 4: Analisis Sampel", use_container_width=True):
        st.switch_page("pages/4_Module_4_Analyze_ROI.py")

with col2:
    if st.session_state.classification_result is not None:
        if st.button("➡️ Lanjut ke Modul 7: Uji Akurasi", type="primary", use_container_width=True):
            st.switch_page("pages/6_Module_7_Thematic_Accuracy.py")
    else:
        st.button("🔒 Selesaikan Klasifikasi Terlebih Dahulu", disabled=True, use_container_width=True)

# Show completion status
if st.session_state.classification_result is not None:
    st.success("✅ Anda telah menyelesaikan Modul 6. Silakan lanjut ke modul berikutnya")
else:
    st.info("💡 Selesaikan ekstraksi fitur dan klasifikasi untuk melanjutkan")

# Footer
show_footer()
