import pandas as pd
import random
from typing import List, Dict, Optional, Any, Tuple


class LULC_Scheme_Manager:
    """
    Class for facilitating land cover classification scheme definition.
    
    This class manages the creation, editing, and validation of land use/land cover
    classification schemes through manual input, CSV upload, or default templates.
    Pure backend procedure, without any frontend 
    
    Attributes
    ----------
    classes : List[Dict[str, Any]]
        List of classification classes with ID, name, and color information
    next_id : int
        Next available ID for new classes
    edit_mode : bool
        Whether currently in edit mode
    edit_idx : Optional[int]
        Index of class being edited
    csv_temp_classes : List[Dict[str, Any]]
        Temporary storage for CSV classes during color assignment
    """
    
    def __init__(self):
        """Initialize the LULC scheme manager with empty state."""
        self.classes: List[Dict[str, Any]] = []
        self.next_id: int = 1
        self.edit_mode: bool = False
        self.edit_idx: Optional[int] = None
        self.csv_temp_classes: List[Dict[str, Any]] = []
    def get_state(self) -> Dict[str, Any]:
        """
        Get current state as dictionary for persistence.
        
        Returns
        -------
        Dict[str, Any]
            Current state including classes, next_id, edit_mode, etc.
        """
        return {
            'classes': self.classes,
            'next_id': self.next_id,
            'edit_mode': self.edit_mode,
            'edit_idx': self.edit_idx,
            'csv_temp_classes': self.csv_temp_classes
        }
    
    def set_state(self, state: Dict[str, Any]) -> None:
        """
        Set state from dictionary for persistence.
        
        Parameters
        ----------
        state : Dict[str, Any]
            State dictionary to restore
        """
        self.classes = state.get('classes', [])
        self.next_id = state.get('next_id', 1)
        self.edit_mode = state.get('edit_mode', False)
        self.edit_idx = state.get('edit_idx', None)
        self.csv_temp_classes = state.get('csv_temp_classes', [])
    
    def has_classes(self) -> bool:
        """
        Check if any classes are defined.
        
        Returns
        -------
        bool
            True if classes exist, False otherwise
        """
        return len(self.classes) > 0
    
    def get_class_count(self) -> int:
        """
        Get the number of defined classes.
        
        Returns
        -------
        int
            Number of classes currently defined
        """
        return len(self.classes)
    ## System Response 2.1b: Manual Scheme Definition
    def validate_class_input(self, class_id: Any, class_name: str) -> Tuple[bool, Optional[str]]:
        """
        Validate class input parameters.
        
        Parameters
        ----------
        class_id : Any
            Class ID to validate (will be converted to int)
        class_name : str
            Class name to validate
            
        Returns
        -------
        Tuple[bool, Optional[str]]
            Tuple of (is_valid, error_message). If valid, error_message is None.
        """
        # Validate class_id type
        try:
            class_id = int(class_id)
        except (ValueError, TypeError):
            return False, "Class ID must be a valid number!"
        
        # Validate class name
        class_name = class_name.strip()
        if not class_name:
            return False, "Class name cannot be empty!"
        
        # Check if ID already exists (only for new classes)
        if not self.edit_mode:
            if any(c['ID'] == class_id for c in self.classes):
                return False, f"Class ID {class_id} already exists!"
        
        return True, None
    #adapted from line 184, but tailored with streamlit compability    
    def add_class(self, class_id: Any, class_name: str, color_code: str) -> Tuple[bool, str]:
        """
        Add or update a class in the classification scheme.
        
        Parameters
        ----------
        class_id : Any
            Unique identifier for the class (will be converted to int)
        class_name : str
            Human-readable name for the class
        color_code : str
            Hex color code for visualization (e.g., '#FF0000')
            
        Returns
        -------
        Tuple[bool, str]
            Tuple of (success, message). Message contains success or error details.
        """
        # Validate input
        is_valid, error_msg = self.validate_class_input(class_id, class_name)
        if not is_valid:
            return False, error_msg
        
        class_id = int(class_id)
        class_name = class_name.strip()
        
        class_data = {
            'ID': class_id,
            'Class Name': class_name,
            'Color Code': color_code
        }
        
        # Update existing class
        if self.edit_mode and self.edit_idx is not None:
            self.classes[self.edit_idx] = class_data
            success_msg = f"KElas '{class_name}' (ID: {class_id}) berhasil diperbarui!"
            self._reset_edit_mode()
        else:
            # Add new class
            self.classes.append(class_data)
            success_msg = f"Kelas '{class_name}' (ID: {class_id}) berhasil ditambahkan!"
        
        self._sort_and_update_next_id()
        return True, success_msg

    #functions for manual input options
    def _reset_edit_mode(self) -> None:
        """
        Reset edit mode state.
        
        Clears the edit mode flags to return to normal add mode.
        """
        self.edit_mode = False
        self.edit_idx = None
    
    def _sort_and_update_next_id(self) -> None:
        """
        Sort classes by ID and update next available ID.
        
        Maintains classes in ascending ID order and calculates the next
        available ID for new classes.
        """
        self.classes = sorted(self.classes, key=lambda x: x['ID'])
        
        if self.classes:
            self.next_id = max([c['ID'] for c in self.classes]) + 1
        else:
            self.next_id = 1
    
    def _generate_random_color(self) -> str:
        """
        Generate a random hex color code.
        
        Returns
        -------
        str
            Random hex color code in format '#RRGGBB'
        """
        return f"#{random.randint(0, 0xFFFFFF):06x}"
    
    def _generate_distinct_colors(self, count: int) -> List[str]:
        """
        Generate a list of visually distinct colors.
        
        Parameters
        ----------
        count : int
            Number of distinct colors to generate
            
        Returns
        -------
        List[str]
            List of hex color codes that are visually distinct
        """
        if count <= 0:
            return []
        
        # Predefined distinct colors for better visual separation
        distinct_colors = [
            "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7",
            "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9",
            "#F8C471", "#82E0AA", "#F1948A", "#85C1E9", "#D7BDE2",
            "#A3E4D7", "#F9E79F", "#D5A6BD", "#AED6F1", "#A9DFBF"
        ]
        
        if count <= len(distinct_colors):
            return distinct_colors[:count]
        
        # If we need more colors than predefined, generate random ones
        colors = distinct_colors.copy()
        while len(colors) < count:
            colors.append(self._generate_random_color())
        
        return colors[:count]
    #adapted from line 229 onward
    def edit_class(self, idx: int) -> Optional[Dict[str, Any]]:
        """
        Set class for editing mode.
        
        Parameters
        ----------
        idx : int
            Index of the class to edit in the classes list
            
        Returns
        -------
        Optional[Dict[str, Any]]
            Class data if valid index, None otherwise
        """
        if 0 <= idx < len(self.classes):
            self.edit_mode = True
            self.edit_idx = idx
            return self.classes[idx]
        return None
    #adapted from line 247
    def delete_class(self, idx: int) -> Tuple[bool, str]:
        """
        Delete a class from the scheme.
        
        Parameters
        ----------
        idx : int
            Index of the class to delete
            
        Returns
        -------
        Tuple[bool, str]
            Tuple of (success, message). Message contains operation details.
        """
        if 0 <= idx < len(self.classes):
            class_to_delete = self.classes[idx]
            del self.classes[idx]
            success_msg = f"Kelas '{class_to_delete['Class Name']}' (ID: {class_to_delete['ID']}) berhasil dihapus!"
            return True, success_msg
        return False, "Invalid class index"
    
    def cancel_edit(self) -> None:
        """
        Cancel edit mode.
        
        Resets the edit mode flags and returns to normal add mode.
        """
        self._reset_edit_mode()

    #adapted from line 266 - 320
    #Change so that csv is more tolaratable 
# Module 2: Classification Scheme Definition
## System Response 2.1a: Upload Classification Scheme
    def process_csv_upload(self, df: pd.DataFrame, id_col: str, name_col: str, 
                          color_col: Optional[str] = None) -> Tuple[bool, str]:
        """
        Process CSV upload - validate and prepare for color assignment.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing the CSV data
        id_col : str
            Column name containing class IDs
        name_col : str
            Column name containing class names
        color_col : Optional[str], default=None
            Column name containing color codes (optional)
            
        Returns
        -------
        Tuple[bool, str]
            Tuple of (success, message). Message contains details about the operation.
        """
        try:
            class_list = []
            used_ids = set()

            for _, row in df.iterrows():
                class_id = row[id_col]
                class_name = row[name_col]

                # Skip empty rows
                if pd.isna(class_id) or pd.isna(class_name):
                    continue

                # Validate and convert class_id
                try:
                    class_id = int(class_id)
                except (ValueError, TypeError):
                    return False, f"Invalid Class ID format: {class_id}. Must be a number."

                # Check for duplicates
                if class_id in used_ids:
                    return False, f"Duplicate Class ID found: {class_id}"
                used_ids.add(class_id)

                # Handle color assignment
                if color_col and not pd.isna(row[color_col]):
                    color_code = str(row[color_col]).strip()
                    # Validate hex color format
                    if not color_code.startswith('#'):
                        color_code = f"#{color_code}"
                    if len(color_code) != 7:
                        color_code = "#2e8540"  # Fallback to default
                else:
                    color_code = "#2e8540"  # Default color

                class_list.append({
                    "ID": class_id,
                    "Class Name": str(class_name).strip(),
                    "Color Code": color_code
                })

            # If no color column was provided, assign distinct random colors
            if not color_col:
                distinct_colors = self._generate_distinct_colors(len(class_list))
                for i, class_data in enumerate(class_list):
                    class_data["Color Code"] = distinct_colors[i]

            self.csv_temp_classes = class_list
            color_msg = "with colors from CSV" if color_col else "with auto-generated colors"
            return True, f"Successfully loaded {len(class_list)} classes from CSV {color_msg}"

        except Exception as e:
            return False, f"Error processing CSV: {str(e)}"

    def finalize_csv_upload(self, color_assignments: Optional[List[str]] = None) -> Tuple[bool, str]:
        """
        Finalize CSV upload with user-assigned colors.
        
        Parameters
        ----------
        color_assignments : Optional[List[str]], default=None
            List of hex color codes assigned by user. If None, uses existing colors.
            
        Returns
        -------
        Tuple[bool, str]
            Tuple of (success, message). Message contains operation details.
        """
        try:
            # Update colors based on user assignments if provided
            if color_assignments:
                for i, class_data in enumerate(self.csv_temp_classes):
                    if i < len(color_assignments):
                        class_data["Color Code"] = color_assignments[i]
            
            # Save to main classes and sort
            self.classes = self.csv_temp_classes.copy()
            self._sort_and_update_next_id()
            
            # Clear temporary storage
            self.csv_temp_classes = []
            
            return True, f"Skema klasifikasi berhasil dibuat dengan {len(self.classes)} kelas"
            
        except Exception as e:
            return False, f"Error finalizing CSV upload: {str(e)}"
    ## System Response 2.1c: Download classification scheme
    def get_csv_data(self) -> Optional[bytes]:
        """
        Generate CSV data for download.
        
        Returns
        -------
        Optional[bytes]
            CSV data as bytes for download, None if no classes exist
        """
        if not self.classes:
            return None
        
        df = self.get_dataframe()
        return df.to_csv(index=False).encode('utf-8')
    
    def get_dataframe(self) -> pd.DataFrame:
        """
        Get the classification scheme as a normalized DataFrame.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with standardized column names for display and export
        """
        if not self.classes:
            return pd.DataFrame(columns=["ID", "Land Cover Class", "Color Palette"])

        df = pd.DataFrame(self.classes)

        # Normalize column names
        column_mapping = {
            "ID": "ID",
            "Class ID": "ID", 
            "Class Name": "Land Cover Class",
            "Land Cover Class": "Land Cover Class",
            "Color": "Color Palette",
            "Color Code": "Color Palette",
            "Color Palette": "Color Palette"
        }

        # Apply column renaming
        df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})

        # Ensure consistent column order
        expected_columns = ["ID", "Land Cover Class", "Color Palette"]
        available_columns = [col for col in expected_columns if col in df.columns]
        
        return df[available_columns]
    
    # Module 2: Classification Scheme Definition
    ## System Response 2.1c: Template Classification Scheme
    #Adapted from line 407
    def load_default_scheme(self, scheme_name: str) -> Tuple[bool, str]:
        """
        Load a predefined classification scheme.
        
        Parameters
        ----------
        scheme_name : str
            Name of the default scheme to load
            
        Returns
        -------
        Tuple[bool, str]
            Tuple of (success, message). Message contains operation details.
        """
        default_schemes = self.get_default_schemes()
        
        if scheme_name not in default_schemes:
            return False, f"Unknown scheme: {scheme_name}"
        
        self.classes = default_schemes[scheme_name].copy()
        self._sort_and_update_next_id()
        
        return True, f"Loaded {scheme_name} with {len(self.classes)} classes"
    #Add RESTORE+ classification scheme
    @staticmethod
    def get_default_schemes() -> Dict[str, List[Dict[str, Any]]]:
        """
        Return available default classification schemes.
        
        Returns
        -------
        Dict[str, List[Dict[str, Any]]]
            Dictionary mapping scheme names to lists of class definitions.
            Each class contains 'ID', 'Class Name', and 'Color Code'.
        """
        return {
            "RESTORE+ Project": [
                {'ID': 1, 'Class Name': 'Undisturbed dry-land forest', 'Color Code': '#006400'},
                {'ID': 2, 'Class Name': 'Logged-over dry-land forest', 'Color Code': '#228B22'},
                {'ID': 3, 'Class Name': 'Undisturbed mangrove', 'Color Code': '#4169E1'},
                {'ID': 4, 'Class Name': 'Logged-over mangrove', 'Color Code': '#87CEEB'},
                {'ID': 5, 'Class Name': 'Undisturbed swamp forest', 'Color Code': '#2E8B57'},
                {'ID': 6, 'Class Name': 'Logged-over swamp forest', 'Color Code': '#8FBC8F'},
                {'ID': 7, 'Class Name': 'Agroforestry', 'Color Code': '#9ACD32'},
                {'ID': 8, 'Class Name': 'Plantation forest', 'Color Code': '#32CD32'},
                {'ID': 9, 'Class Name': 'Rubber monoculture', 'Color Code': '#8B4513'},
                {'ID': 10, 'Class Name': 'Oil palm monoculture', 'Color Code': '#FF8C00'},
                {'ID': 11, 'Class Name': 'Other monoculture', 'Color Code': '#DAA520'},
                {'ID': 12, 'Class Name': 'Grass/savanna', 'Color Code': '#ADFF2F'},
                {'ID': 13, 'Class Name': 'Shrub', 'Color Code': '#90EE90'},
                {'ID': 14, 'Class Name': 'Cropland', 'Color Code': '#FFFF00'},
                {'ID': 15, 'Class Name': 'Settlement', 'Color Code': '#FF0000'},
                {'ID': 16, 'Class Name': 'Cleared land', 'Color Code': '#D2B48C'},
                {'ID': 17, 'Class Name': 'Waterbody', 'Color Code': '#0000FF'},
            ],
            "Epistem": [
                {'ID': 1, 'Class Name': 'Primary Dryland Forest', 'Color Code': '#006400'},
                {'ID': 2, 'Class Name': 'Secondary Dryland Forest', 'Color Code': '#228B22'},
                {'ID': 3, 'Class Name': 'Primary Mangrove Forest', 'Color Code': '#4169E1'},
                {'ID': 4, 'Class Name': 'Secondary Mangrove Forest', 'Color Code': '#87CEEB'},
                {'ID': 5, 'Class Name': 'Primary Swamp Forest', 'Color Code': '#2E8B57'},
                {'ID': 6, 'Class Name': 'Secondary Swamp Forest', 'Color Code': '#8FBC8F'},
                {'ID': 7, 'Class Name': 'Plantation Forest', 'Color Code': '#32CD32'},
                {'ID': 8, 'Class Name': 'Rubber Monoculture', 'Color Code': '#8B4513'},
                {'ID': 9, 'Class Name': 'Oil palm Monoculture', 'Color Code': '#FF8C00'},
                {'ID': 10, 'Class Name': 'Cacao Monoculture', 'Color Code': '#D2691E'},
                {'ID': 11, 'Class Name': 'Coconut monoculture', 'Color Code': '#F4A460'},
                {'ID': 12, 'Class Name': 'Other Monoculture', 'Color Code': '#DAA520'},
                {'ID': 13, 'Class Name': 'Other Cropland', 'Color Code': '#FFFF00'},
                {'ID': 14, 'Class Name': 'Coffee agroforestry', 'Color Code': '#6B8E23'},
                {'ID': 15, 'Class Name': 'Rubber agroforestry', 'Color Code': '#9ACD32'},
                {'ID': 16, 'Class Name': 'Mixed/home garden', 'Color Code': '#7CFC00'},
                {'ID': 17, 'Class Name': 'Paddy field', 'Color Code': '#EEE8AA'},
                {'ID': 18, 'Class Name': 'Grass or Savanna', 'Color Code': '#ADFF2F'},
                {'ID': 19, 'Class Name': 'Shrub', 'Color Code': '#90EE90'},
                {'ID': 20, 'Class Name': 'Settlement', 'Color Code': '#FF0000'},
                {'ID': 21, 'Class Name': 'Cleared Land', 'Color Code': '#D2B48C'},
                {'ID': 22, 'Class Name': 'Mining area', 'Color Code': '#808080'},
                {'ID': 23, 'Class Name': 'Waterbody', 'Color Code': '#0000FF'},
                {'ID': 24, 'Class Name': 'Fish pond', 'Color Code': '#00BFFF'},
            ]
        }
    
    @staticmethod
    def auto_detect_csv_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Auto-detect ID, Name, and Color columns in CSV.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to analyze for column detection
            
        Returns
        -------
        Tuple[Optional[str], Optional[str], Optional[str]]
            Tuple of (id_column, name_column, color_column). Any can be None if not found.
        """
        columns_lower = [c.lower().replace(" ", "").replace("_", "") for c in df.columns]
        
        def find_column(keywords: List[str]) -> Optional[str]:
            for keyword in keywords:
                for i, col in enumerate(columns_lower):
                    if keyword in col:
                        return df.columns[i]
            return None
        
        id_col = find_column(["id", "classid", "kode", "code"])
        name_col = find_column(["classname", "class", "kelas", "name", "nama"])
        color_col = find_column(["color", "colorcode", "warna", "colour", "hex", "palette"])
        
        return id_col, name_col, color_col 
    
    def store_classes_of_interest(
        self,
        scheme_name: str,
        classes_of_interest: List[int]
    ) -> Dict[str, Any]:
        """
        Store and validate selected classes of interest from the current classification scheme.

        Parameters
        ----------
        scheme_name : str
            Name of the classification scheme.
        classes_of_interest : List[int]
            List of class IDs selected by the user.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing scheme name and validated selected classes.
        """

        if not self.classes:
            raise ValueError("No classes are currently loaded in the scheme.")

        # Get valid IDs from current classes
        valid_ids = [c["ID"] for c in self.classes]

        # Validate selection
        invalid_ids = [cid for cid in classes_of_interest if cid not in valid_ids]
        if invalid_ids:
            raise ValueError(
                f"Selected class ID(s) not found in the classification scheme: {invalid_ids}"
            )

        # Store metadata
        selected_classes = {
            "scheme_name": scheme_name,
            "classes_of_interest": classes_of_interest
        }

        return selected_classes