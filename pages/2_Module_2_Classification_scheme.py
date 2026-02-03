"""
Module 2: Land Cover Classification Scheme Definition

This module provides a user interface for defining land cover classification schemes
through three methods: manual input, CSV upload, or default templates.

Architecture:
- Backend (src_modul_2.py): Pure business logic without UI dependencies
- Frontend (this file): Streamlit UI with session state management
- State synchronization ensures data persistence across page interactions
"""

import streamlit as st
import pandas as pd
from luma_ge.classification_scheme import LULC_Scheme_Manager
from modules.nav import Navbar
from ui_helper import show_footer, show_header

#Page configuration
st.set_page_config(
    page_title="Epistem-X Modul 2",
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
<div class="breadcrumb">Modul 2 › Tentukan Skema Klasifikasi</div>
""", unsafe_allow_html=True)

# Initialize session state for persistence
def init_session_state():
    """Initialize session state variables for LULC scheme management."""
    session_vars = {
        'lulc_classes': [],
        'lulc_next_id': 1,
        'lulc_edit_mode': False,
        'lulc_edit_idx': None,
        'csv_temp_classes': []
    }
    
    for var, default_value in session_vars.items():
        if var not in st.session_state:
            st.session_state[var] = default_value

# Initialize session state
init_session_state()

# Create manager and sync with session state
manager = LULC_Scheme_Manager()

def sync_manager_from_session():
    """Sync manager state from Streamlit session state."""
    state = {
        'classes': st.session_state.lulc_classes,
        'next_id': st.session_state.lulc_next_id,
        'edit_mode': st.session_state.lulc_edit_mode,
        'edit_idx': st.session_state.lulc_edit_idx,
        'csv_temp_classes': st.session_state.csv_temp_classes
    }
    manager.set_state(state)

def sync_session_from_manager():
    """Sync Streamlit session state from manager state."""
    state = manager.get_state()
    st.session_state.lulc_classes = state['classes']
    st.session_state.lulc_next_id = state['next_id']
    st.session_state.lulc_edit_mode = state['edit_mode']
    st.session_state.lulc_edit_idx = state['edit_idx']
    st.session_state.csv_temp_classes = state['csv_temp_classes']

# Sync manager with current session state
sync_manager_from_session()

# Add navigation sidebar
Navbar()
#Page header
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
<h1 class="gradient-title">Menentukan Skema Klasifikasi<br>Peta Tutupan/Penggunaan Lahan</h1>
""", unsafe_allow_html=True)
st.divider()

st.markdown("""
<style>
.check-item {
  display: flex;
  align-items: flex-start;
  gap: 10px;
  margin-bottom: 8px;
}
.check-icon {
  width: 20px;
  height: 20px;
  background: linear-gradient(135deg, var(--pink), var(--purple));
  -webkit-mask: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" fill="white" viewBox="0 0 24 24"><path d="M9 16.172l-3.586-3.586-1.414 1.414L9 19 20 8l-1.414-1.414z"/></svg>') no-repeat center;
  mask: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" fill="white" viewBox="0 0 24 24"><path d="M9 16.172l-3.586-3.586-1.414 1.414L9 19 20 8l-1.414-1.414z"/></svg>') no-repeat center;
  -webkit-mask-size: contain;
  mask-size: contain;
  flex-shrink: 0;
}
</style>
<p>Dalam modul ini, Anda perlu menentukan skema klasifikasi yang akan digunakan untuk membuat peta tutupan lahan.</p>
<p>Terdapat tiga metode yang didukung dalam platform ini:</p>
<div class="check-item">
  <div class="check-icon"></div>
  <p><strong>Input Manual</strong>:<br>Tambahkan kelas satu per satu</p>
</div>
<div class="check-item">
  <div class="check-icon"></div>
  <p><strong>Unggah CSV</strong>:<br>Impor dari berkas klasifikasi yang sudah ada</p>
</div>
<div class="check-item">
  <div class="check-icon"></div>
  <p><strong>Skema Bawaan</strong>:<br>Gunakan kelas yang berdasarkan skema RESTORE+</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")


#Tab layout for different classification definition
tab1, tab2, tab3 = st.tabs(["➕ Input Manual", "📤 Unggah CSV", "📋 Skema Bawaan"])

#Create a function for manual input the class
def render_manual_input_form():
    """
    Render the manual class input form.
    
    Creates input fields for class ID, name, and color, with support for
    both adding new classes and editing existing ones.
    """
    st.markdown("#### Tambahkan sebuah kelas baru")
    #3 columns
    col1, col2, col3 = st.columns([1, 3, 2])
    
    # Sync manager state
    sync_manager_from_session()
    
    # Get current values for edit mode
    if manager.edit_mode and manager.edit_idx is not None:
        current_class = manager.classes[manager.edit_idx]
        default_id = current_class['ID']
        default_name = current_class['Class Name']
        default_color = current_class['Color Code']
        key_suffix = "edit"
    else:
        default_id = manager.next_id
        default_name = ""
        default_color = "#2e8540"
        key_suffix = "new"
    
    with col1:
        class_id = st.number_input(
            "ID Kelas", 
            value=default_id, 
            min_value=1, 
            step=1,
            key=f"{key_suffix}_class_id"
        )
    
    with col2:
        class_name = st.text_input(
            "Nama Kelas", 
            value=default_name,
            placeholder="e.g., Hutan, Pemukiman",
            key=f"{key_suffix}_class_name"
        )
    
    with col3:
        color_code = st.color_picker(
            "Kode Warna", 
            value=default_color,
            key=f"{key_suffix}_color_code"
        )
    
    # Action buttons
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 4])
    
    with col_btn1:
        button_text = "💾 Perbarui kelas" if manager.edit_mode else "➕ Tambahkan kelas"
        if st.button(button_text, type="primary", use_container_width=True):
            success, message = manager.add_class(class_id, class_name, color_code)
            if success:
                st.session_state['ReferenceDataSource'] = False
                sync_session_from_manager()  # Sync back to session state
                st.success(f"✅ {message}")
                st.rerun()
            else:
                st.error(f"❌ {message}")
    
    with col_btn2:
        if manager.edit_mode and st.button("❌ Batalkan", use_container_width=True):
            manager.cancel_edit()
            sync_session_from_manager()  # Sync back to session state
            st.rerun()

# Tab 1: Manual Input
with tab1:
    render_manual_input_form()

# Tab 2: Upload CSV
with tab2:
    st.markdown("#### Unggah Skema Klasifikasi")
    st.info("""
    *Persyaratan berkas CSV:**
    - **Kolom ID**: Pengenal numerik (misalnya: 'ID', 'Class ID', 'Kode')
    - **Kolom Nama**: Nama kelas (misalnya: 'Class Name', 'Kelas', 'Name')
    - **Kolom Warna** (Opsional): Kode warna heksadesimal (misalnya: 'Color', 'Color Code', 'Hex')
    
    Jika tidak ada kolom warna yang terdeteksi, warna yang berbeda akan ditetapkan secara otomatis.
    """)
    #Code to upload csv
    uploaded_file = st.file_uploader("Pilih sebuah berkas CSV", type=['csv'])

    if uploaded_file is not None:
        try:
            #make sure that python can read any form of delimiter
            df = pd.read_csv(uploaded_file, sep=None, engine="python")
            
            #Auto-detect columns
            auto_id, auto_name, auto_color = manager.auto_detect_csv_columns(df)
            
            st.markdown("### Pilih kolom")
            col1, col2, col3 = st.columns(3)
            #first column to select class ID's column
            with col1:
                id_col = st.selectbox(
                    "Pilih ID kolom *", 
                    df.columns, 
                    index=df.columns.get_loc(auto_id) if auto_id in df.columns else 0
                )
            #second column to select class name's column
            with col2:
                name_col = st.selectbox(
                    "Pilih nama kolom kelas *", 
                    df.columns,
                    index=df.columns.get_loc(auto_name) if auto_name in df.columns else 0
                )
            #new column for detecting color palette column
            with col3:
                color_options = ["< Tidak ada kolom warna >"] + list(df.columns)
                default_color_idx = 0
                if auto_color and auto_color in df.columns:
                    default_color_idx = color_options.index(auto_color)
                
                color_col_selection = st.selectbox(
                    "Pilih kolom warna (Opsional)",
                    color_options,
                    index=default_color_idx
                )
            #After selection, load the CSV
            if st.button("📤 Unggah berkas CSV", type="primary"):
                sync_manager_from_session()  # Sync current state
                color_col = None if color_col_selection == "< Tidak ada kolom warna >" else color_col_selection
                success, message = manager.process_csv_upload(df, id_col, name_col, color_col)
                if success:
                    st.session_state['ReferenceDataSource'] = False
                    # If colors were detected, finalize immediately
                    if color_col:
                        success_final, message_final = manager.finalize_csv_upload()
                        if success_final:
                            sync_session_from_manager()  # Sync back to session state
                            st.success(f"✅ {message_final}")
                        else:
                            st.error(f"❌ {message_final}")
                    else:
                        sync_session_from_manager()  # Sync back to session state
                        st.success(f"✅ {message}")
                    st.rerun()
                else:
                    st.error(f"❌ {message}")
                    
        except Exception as e:
            st.error(f"Berkas CSV error: {str(e)}")

    #Color assignment section (only show if no colors were auto-detected)
    sync_manager_from_session()  # Sync current state
    if manager.csv_temp_classes:
        st.markdown("---")
        st.markdown("### Tentukan warna untuk tiap kelas")
        st.info("Warna telah ditentukan secara acak. Anda dapat menyesuaikannya jika diperlukan.")
        
        color_assignments = []
        temp_classes = manager.csv_temp_classes
        
        for i, class_data in enumerate(temp_classes):
            col1, col2, col3 = st.columns([1, 3, 2])
            
            with col1:
                st.write(f"**ID: {class_data['ID']}**")
            with col2:
                st.write(f"**{class_data['Class Name']}**")
            with col3:
                color = st.color_picker(
                    f"Color", 
                    value=class_data.get('Color Code', '#2e8540'),
                    key=f"csv_color_{i}",
                    label_visibility="collapsed"
                )
                color_assignments.append(color)
        
        st.markdown("---")
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("✅ Finalisasi Skema", type="primary", use_container_width=True):
                success, message = manager.finalize_csv_upload(color_assignments)
                if success:
                    sync_session_from_manager()  # Sync back to session state
                    st.success(f"✅ {message}")
                    st.rerun()
                else:
                    st.error(f"❌ {message}")
        
        with col2:
            if st.button("❌ Batalkan unggah", use_container_width=True):
                manager.csv_temp_classes = []
                sync_session_from_manager()  # Sync back to session state
                st.rerun()

# Tab 3: Default Scheme
with tab3:
    st.markdown("####  Muat Skema Klasifikasi Bawaan")
    st.info("Mulai dengan kelas tutupan/penggunaan lahan RESTORE+")
    
    if 'ReferenceDataSource' not in st.session_state:
        st.session_state.ReferenceDataSource = False

    default_schemes = manager.get_default_schemes()
    
    selected_scheme = st.selectbox(
        "Pilih skema bawaan:",
        options=list(default_schemes.keys())
    )
    
    # Preview the selected scheme
    if selected_scheme:
        with st.expander("📋 Pratayang kelas"):
            preview_df = pd.DataFrame(default_schemes[selected_scheme])
            st.dataframe(preview_df, use_container_width=True)
    
    if st.button("📋 Muat skema bawaan", type="primary", use_container_width=True):
        sync_manager_from_session()  # Sync current state
        success, message = manager.load_default_scheme(selected_scheme)
        if success:
            sync_session_from_manager()  # Sync back to session state
            st.success(f"✅ {message}")
            st.session_state['ReferenceDataSource'] = True
            st.rerun()
        else:
            st.error(f"❌ {message}")

#new function to render selected classification scheme from one of the three methods
def render_class_display():
    """
    Render the current classification scheme display.
    
    Shows all defined classes in a table format with color previews,
    edit/delete buttons, and download functionality.
    """
    st.markdown("---")
    st.markdown("#### Skema Klasifikasi Saat Ini")

    # Sync manager state
    sync_manager_from_session()
    
    if not manager.classes:
        st.warning("⚠️ Belum ada kelas yang ditentukan. Tambahkan kelas pertama Anda di atas!")
        return

    # Display classes in a clean table format
    for idx, class_data in enumerate(manager.classes):
        col1, col2, col3, col4, col5 = st.columns([1, 3, 2, 1, 1])
        
        with col1:
            st.write(f"**{class_data['ID']}**")
        
        with col2:
            st.write(class_data['Class Name'])
        
        with col3:
            # Color preview with code
            st.markdown(
                f"""<div style='display: flex; align-items: center;'>
                    <div style='background-color: {class_data['Color Code']}; 
                                width: 40px; height: 25px; border: 1px solid #ccc; 
                                margin-right: 8px; border-radius: 3px;'></div>
                    <code>{class_data['Color Code']}</code>
                </div>""",
                unsafe_allow_html=True
            )
        
        with col4:
            if st.button("✏️", key=f"edit_{idx}", help="Ubah kelas"):
                manager.edit_class(idx)
                sync_session_from_manager()  # Sync back to session state
                st.rerun()
        
        with col5:
            if st.button("🗑️", key=f"delete_{idx}", help="Delete class"):
                success, message = manager.delete_class(idx)
                if success:
                    sync_session_from_manager()  # Sync back to session state
                    st.success(f"✅ {message}")
                    st.rerun()

    # Download and preview section
    st.markdown("---")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        csv_data = manager.get_csv_data()
        if csv_data:
            st.download_button(
                label="📥 Unduh sebagai berkas CSV",
                data=csv_data,
                file_name="classification_scheme.csv",
                mime="text/csv",
                type="primary",
                use_container_width=True
            )
    
    with col2:
        with st.expander("📋 Prayatang berkas"):
            st.dataframe(manager.get_dataframe(), use_container_width=True)

# Render the class display
render_class_display()

def render_navigation():
    """
    Render module navigation and completion status.
    
    Provides navigation buttons to previous/next modules and displays
    completion status based on whether classes have been defined.
    """
    st.divider()
    
    # Sync manager state and store classification data for other modules
    sync_manager_from_session()
    if manager.classes:
        # Store both the formatted DataFrame and raw classes for other modules
        st.session_state['classification_df'] = manager.get_dataframe()
        st.session_state['lulc_classes_final'] = manager.classes.copy()
        
        # Also store in the format expected by Module 3
        # Convert manager classes to the format Module 3 expects
        classes_for_module3 = []
        for cls in manager.classes:
            classes_for_module3.append({
                'ID': cls.get('ID', cls.get('Class ID', '')),
                'LULC_Type': cls.get('Class Name', cls.get('Land Cover Class', '')),
                'color_palette': cls.get('Color Code', cls.get('Color Palette', cls.get('Color', '#2e8540')))
            })
        st.session_state['classes'] = classes_for_module3
    
    # Module completion check
    module_completed = manager.has_classes()
    
    # Navigation buttons
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("⬅️ Kembali ke modul 1", use_container_width=True):
            st.switch_page("pages/1_Module_1_Generate_Image_Mosaic.py")
    
    with col2:
        if module_completed:
            if st.button("➡️ Buka Modul 3: Penentuan data latih", 
                        type="primary", use_container_width=True):
                st.switch_page("pages/3_Module_3_Generate_ROI.py")
        else:
            st.button("🔒 Selesaikan modul 2 terlebih dahulu", 
                     disabled=True, use_container_width=True,
                     help="Add at least one class to proceed")
    
    # Status indicator
    if module_completed:
        st.success(f"✅ Modul 2 selesai dengan {manager.get_class_count()} kelas")
    else:
        st.info("💡 Tambahkan setidaknya satu kelas untuk menyelesaikan modul ini")

# Render navigation
render_navigation()