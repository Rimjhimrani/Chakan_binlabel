import streamlit as st
import pandas as pd
import os
from reportlab.lib.pagesizes import landscape
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Spacer, Paragraph, PageBreak, Image
from reportlab.lib.units import cm, inch
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.lib.utils import ImageReader
from io import BytesIO
import subprocess
import sys
import re
import tempfile

# Define sticker dimensions
STICKER_WIDTH = 10 * cm
STICKER_HEIGHT = 15 * cm
STICKER_PAGESIZE = (STICKER_WIDTH, STICKER_HEIGHT)

# Define content box dimensions
CONTENT_BOX_WIDTH = 10 * cm  # Same width as page
CONTENT_BOX_HEIGHT = 7.2 * cm  # Half the page height

# Check for PIL and install if needed
try:
    from PIL import Image as PILImage
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    st.write("PIL not available. Installing...")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pillow'])
    from PIL import Image as PILImage
    PIL_AVAILABLE = True

# Check for QR code library and install if needed
try:
    import qrcode
    QR_AVAILABLE = True
except ImportError:
    QR_AVAILABLE = False
    st.write("qrcode not available. Installing...")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'qrcode'])
    import qrcode
    QR_AVAILABLE = True

# Define paragraph styles
bold_style = ParagraphStyle(name='Bold', fontName='Helvetica-Bold', fontSize=16, alignment=TA_CENTER, leading=14)
desc_style = ParagraphStyle(name='Description', fontName='Helvetica', fontSize=11, alignment=TA_CENTER, leading=12)
qty_style = ParagraphStyle(name='Quantity', fontName='Helvetica', fontSize=11, alignment=TA_CENTER, leading=12)

def find_bus_model_column(df_columns):
    """
    Enhanced function to find the bus model column with better detection
    """
    cols = [str(col).upper() for col in df_columns]
    
    # Priority order for bus model column detection
    patterns = [
        # Exact matches (highest priority)
        lambda col: col == 'BUS_MODEL',
        lambda col: col == 'BUSMODEL',
        lambda col: col == 'BUS MODEL',
        lambda col: col == 'MODEL',
        lambda col: col == 'BUS_TYPE',
        lambda col: col == 'BUSTYPE',
        lambda col: col == 'BUS TYPE',
        lambda col: col == 'VEHICLE_TYPE',
        lambda col: col == 'VEHICLETYPE',
        lambda col: col == 'VEHICLE TYPE',
        # Partial matches (lower priority)
        lambda col: 'BUS' in col and 'MODEL' in col,
        lambda col: 'BUS' in col and 'TYPE' in col,
        lambda col: 'VEHICLE' in col and 'MODEL' in col,
        lambda col: 'VEHICLE' in col and 'TYPE' in col,
        lambda col: 'MODEL' in col,
        lambda col: 'BUS' in col,
        lambda col: 'VEHICLE' in col,
    ]
    
    for pattern in patterns:
        for i, col in enumerate(cols):
            if pattern(col):
                return df_columns[i]  # Return original column name
    
    return None

def find_column_by_keywords(df_columns, keywords):
    """
    Find column by matching keywords (case-insensitive)
    """
    cols = [str(col).upper() for col in df_columns]
    
    for keyword in keywords:
        for i, col in enumerate(cols):
            if keyword.upper() in col:
                return df_columns[i]
    return None

def detect_bus_model_and_qty(row, qty_veh_col, bus_model_col=None):
    """
    Improved bus model detection that properly matches bus model to MTM box
    Returns a dictionary with keys '4W', '3WS', '3WM','3WC' and their respective quantities
    """
    # Initialize result dictionary
    result = {'4W': '', '3WS': '', '3WM': '', '3WC': ''}
    
    # Get quantity value
    qty_veh = ""
    if qty_veh_col and qty_veh_col in row and pd.notna(row[qty_veh_col]):
        qty_veh = str(row[qty_veh_col]).strip()
    
    if not qty_veh:
        return result
    
    # Method 1: Check if quantity already contains model info (e.g., "3WS:2", "4W: 3", "3WM: 5", "3WC: 6")
    qty_pattern = r'(\d+W[SMC]?)[:\-\s]*(\d+)'
    matches = re.findall(qty_pattern, qty_veh.upper())
    
    if matches:
        # If we found model-quantity pairs in the qty_veh field itself
        for model, quantity in matches:
            if model in result:
                result[model] = quantity
        return result
    
    # Method 2: Look for bus model in dedicated bus model column first
    detected_model = None
    if bus_model_col and bus_model_col in row and pd.notna(row[bus_model_col]):
        bus_model_value = str(row[bus_model_col]).strip().upper()
        
        # Check for exact matches first
        if bus_model_value in ['4W', '4']:
            detected_model = '4W'
        elif bus_model_value in ['3WS', '3S']:
            detected_model = '3WS'
        elif bus_model_value in ['3WM', '3M']:
            detected_model = '3WM'
        elif bus_model_value in ['3WC', '3C']:
            detected_model = '3WC'
        # Check for patterns within the text
        elif re.search(r'\b4W\b', bus_model_value):
            detected_model = '4W'
        elif re.search(r'\b3WS\b', bus_model_value):
            detected_model = '3WS'
        elif re.search(r'\b3WM\b', bus_model_value):
            detected_model = '3WM'
        elif re.search(r'\b3WC\b', bus_model_value):
            detected_model = '3WC'
        # Check for standalone numbers
        elif re.search(r'\b4\b', bus_model_value):
            detected_model = '4W'
        elif re.search(r'\b3S\b', bus_model_value):
            detected_model = '3WS'
        elif re.search(r'\b3M\b', bus_model_value):
            detected_model = '3WM'
        elif re.search(r'\b3C\b', bus_model_value):
            detected_model = '3WC'
    
    # If we found a model in the dedicated column, use it
    if detected_model:
        result[detected_model] = qty_veh
        return result
    
    # Method 3: Search through all columns systematically with priority
    # First, search in columns that are most likely to contain bus model info
    priority_columns = []
    other_columns = []
    
    for col in row.index:
        if pd.notna(row[col]):
            col_upper = str(col).upper()
            # High priority columns
            if any(keyword in col_upper for keyword in ['MODEL', 'BUS', 'VEHICLE', 'TYPE']):
                priority_columns.append(col)
            else:
                other_columns.append(col)
    
    # Search priority columns first
    for col in priority_columns:
        if pd.notna(row[col]):
            value_str = str(row[col]).upper()
            
            # Look for exact matches first
            if re.search(r'\b4W\b', value_str):
                result['4W'] = qty_veh
                return result
            elif re.search(r'\b3WS\b', value_str):
                result['3WS'] = qty_veh
                return result
            elif re.search(r'\b3WM\b', value_str):
                result['3WM'] = qty_veh
                return result
            elif re.search(r'\b3WC\b', value_str):
                result['3WC'] = qty_veh
                return result
            # Then look for standalone numbers in context
            elif re.search(r'\b4\b', value_str) and any(keyword in value_str for keyword in ['BUS', 'METER', 'M']):
                result['4W'] = qty_veh
                return result
            elif re.search(r'\b3S\b', value_str) and any(keyword in value_str for keyword in ['BUS', 'METER', 'M']):
                result['3WS'] = qty_veh
                return result
            elif re.search(r'\b3M\b', value_str) and any(keyword in value_str for keyword in ['BUS', 'METER', 'M']):
                result['3WM'] = qty_veh
                return result
            elif re.search(r'\b3C\b', value_str) and any(keyword in value_str for keyword in ['BUS', 'METER', 'M']):
                result['3WC'] = qty_veh
                return result
    
    # Method 4: Search in other columns as fallback
    detected_models = []
    for col in other_columns:
        if pd.notna(row[col]):
            value_str = str(row[col]).upper()
            
            # Use word boundaries to avoid false matches
            if re.search(r'\b4W\b', value_str):
                detected_models.append('4W')
            elif re.search(r'\b3WS\b', value_str):
                detected_models.append('3WS')
            elif re.search(r'\b3WM\b', value_str):
                detected_models.append('3WM')
            elif re.search(r'\b3WC\b', value_str):
                detected_models.append('3WC')
    
    # Remove duplicates while preserving order
    detected_models = list(dict.fromkeys(detected_models))
    
    if detected_models:
        # Use the first detected model
        result[detected_models[0]] = qty_veh
        return result
    
    # Method 5: Last resort - look for standalone numbers that might indicate bus length
    for col in row.index:
        if pd.notna(row[col]):
            value_str = str(row[col]).strip()
            
            # Look for exact matches of just the number
            if value_str == '4':
                result['4W'] = qty_veh
                return result
            elif value_str == '3S':
                result['3WS'] = qty_veh
                return result
            elif value_str == '3M':
                result['3WM'] = qty_veh
                return result
            elif value_str == '3C':
                result['3WC'] = qty_veh
                return result
    
    # Method 6: If still no model detected, return empty (no boxes filled)
    return result

def generate_qr_code(data_string):
    """
    Generate a QR code from the given data string
    """
    try:
        # Create QR code instance
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_M,
            box_size=10,
            border=4,
        )
        
        # Add data
        qr.add_data(data_string)
        qr.make(fit=True)
        
        # Create QR code image
        qr_img = qr.make_image(fill_color="black", back_color="white")
        
        # Convert PIL image to bytes that reportlab can use
        img_buffer = BytesIO()
        qr_img.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        
        # Create a QR code image with specified size
        return Image(img_buffer, width=2.2*cm, height=2.2*cm)
    except Exception as e:
        st.error(f"Error generating QR code: {e}")
        import traceback
        traceback.print_exc()
        return None

def extract_line_location_values(row, bus_model_col, station_no_col, rack_col, rack_no_col, level_col, cell_col):
    """
    Extract specific values for Line Location boxes from Excel columns
    """
    location_values = [''] * 7
    
    # 1st box: Bus Model
    if bus_model_col and bus_model_col in row and pd.notna(row[bus_model_col]):
        location_values[0] = str(row[bus_model_col]).strip()
    
    # 2nd box: Station No
    if station_no_col and station_no_col in row and pd.notna(row[station_no_col]):
        location_values[1] = str(row[station_no_col]).strip()
    
    # 3rd box: Rack
    if rack_col and rack_col in row and pd.notna(row[rack_col]):
        location_values[2] = str(row[rack_col]).strip()
    
    # 4th box: Rack No (1st digit)
    if rack_no_col and rack_no_col in row and pd.notna(row[rack_no_col]):
        rack_no_value = str(row[rack_no_col]).strip()
        if rack_no_value and len(rack_no_value) >= 1:
            location_values[3] = rack_no_value[0]
    
    # 5th box: Rack No (2nd digit)
    if rack_no_col and rack_no_col in row and pd.notna(row[rack_no_col]):
        rack_no_value = str(row[rack_no_col]).strip()
        if rack_no_value and len(rack_no_value) >= 2:
            location_values[4] = rack_no_value[1]
    
    # 6th box: Level
    if level_col and level_col in row and pd.notna(row[level_col]):
        location_values[5] = str(row[level_col]).strip()
    
    # 7th box: Cell
    if cell_col and cell_col in row and pd.notna(row[cell_col]):
        location_values[6] = str(row[cell_col]).strip()
    
    return location_values

def extract_store_location_values(row, zone_col, location_col, floor_col, rack_no_col, level_in_rack_col, cell_col, no_col):
    """
    Extract specific values for Store Location boxes from Excel columns
    """
    store_values = [''] * 7
    
    # 1st box: Zone
    if zone_col and zone_col in row and pd.notna(row[zone_col]):
        store_values[0] = str(row[zone_col]).strip()
    
    # 2nd box: Location
    if location_col and location_col in row and pd.notna(row[location_col]):
        store_values[1] = str(row[location_col]).strip()
    
    # 3rd box: Floor
    if floor_col and floor_col in row and pd.notna(row[floor_col]):
        store_values[2] = str(row[floor_col]).strip()
    
    # 4th box: Rack No
    if rack_no_col and rack_no_col in row and pd.notna(row[rack_no_col]):
        store_values[3] = str(row[rack_no_col]).strip()
    
    # 5th box: Level in Rack
    if level_in_rack_col and level_in_rack_col in row and pd.notna(row[level_in_rack_col]):
        store_values[4] = str(row[level_in_rack_col]).strip()
    
    # 6th box: Cell
    if cell_col and cell_col in row and pd.notna(row[cell_col]):
        store_values[5] = str(row[cell_col]).strip()
    
    # 7th box: No
    if no_col and no_col in row and pd.notna(row[no_col]):
        store_values[6] = str(row[no_col]).strip()
    
    return store_values

def parse_location_string(location_str):
    """Parse a location string into components for table display"""
    # Initialize with empty values
    location_parts = [''] * 7

    if not location_str or not isinstance(location_str, str):
        return location_parts

    # Remove any extra spaces
    location_str = location_str.strip()

    # Try to parse location components
    import re
    pattern = r'([^_\s]+)'
    matches = re.findall(pattern, location_str)

    # Fill the available parts
    for i, match in enumerate(matches[:7]):
        location_parts[i] = match

    return location_parts

def generate_sticker_labels(excel_file_path, output_pdf_path, status_callback=None):
    """Generate sticker labels with QR code from Excel data"""
    if status_callback:
        status_callback(f"Processing file: {excel_file_path}")
    else:
        st.write(f"Processing file: {excel_file_path}")

    # Create a function to draw the border box around content
    def draw_border(canvas, doc):
        canvas.saveState()
        # Draw border box around the content area (10cm x 7.5cm)
        # Position it at the top of the page with minimal margin
        x_offset = (STICKER_WIDTH - CONTENT_BOX_WIDTH) / 2
        y_offset = STICKER_HEIGHT - CONTENT_BOX_HEIGHT - 0.2*cm  # Position at top with minimal margin
        canvas.setStrokeColor(colors.Color(0, 0, 0, alpha=0.95))  # Slightly darker black (95% opacity)
        canvas.setLineWidth(1.8)  # Slightly thicker border
        canvas.rect(
            x_offset + doc.leftMargin,
            y_offset,
            CONTENT_BOX_WIDTH - 0.2*cm,  # Account for margins
            CONTENT_BOX_HEIGHT
        )
        canvas.restoreState()

    # Load the Excel data
    try:
        if excel_file_path.lower().endswith('.csv'):
            df = pd.read_csv(excel_file_path)
        else:
            try:
                df = pd.read_excel(excel_file_path)
            except Exception as e:
                try:
                    df = pd.read_excel(excel_file_path, engine='openpyxl')
                except Exception as e2:
                    df = pd.read_csv(excel_file_path, encoding='latin1')

        if status_callback:
            status_callback(f"Successfully read file with {len(df)} rows")
            status_callback(f"Columns found: {df.columns.tolist()}")
        else:
            st.write(f"Successfully read file with {len(df)} rows")
            st.write("Columns found:", df.columns.tolist())
    except Exception as e:
        error_msg = f"Error reading file: {e}"
        if status_callback:
            status_callback(error_msg)
        else:
            st.error(error_msg)
        return None

    # Identify columns (case-insensitive)
    original_columns = df.columns.tolist()
    df.columns = [col.upper() if isinstance(col, str) else col for col in df.columns]
    cols = df.columns.tolist()

    # Find relevant columns
    part_no_col = next((col for col in cols if 'PART' in col and ('NO' in col or 'NUM' in col or '#' in col)),
                   next((col for col in cols if col in ['PARTNO', 'PART']), cols[0]))

    desc_col = next((col for col in cols if 'DESC' in col),
                   next((col for col in cols if 'NAME' in col), cols[1] if len(cols) > 1 else part_no_col))

    # Look specifically for "QTY/BIN" column first, then fall back to general QTY column
    qty_bin_col = next((col for col in cols if 'QTY/BIN' in col or 'QTY_BIN' in col or 'QTYBIN' in col), 
                  next((col for col in cols if 'QTY' in col and 'BIN' in col), None))
    
    # If no specific QTY/BIN column is found, fall back to general QTY column
    if not qty_bin_col:
        qty_bin_col = next((col for col in cols if 'QTY' in col),
                      next((col for col in cols if 'QUANTITY' in col), None))
  
    loc_col = next((col for col in cols if 'LOC' in col or 'POS' in col or 'LOCATION' in col),
                   cols[2] if len(cols) > 2 else desc_col)

    # Improved detection of QTY/VEH column
    qty_veh_col = next((col for col in cols if any(term in col for term in ['QTY/VEH', 'QTY_VEH', 'QTY PER VEH', 'QTYVEH', 'QTYPERCAR', 'QTYCAR', 'QTY/CAR'])), None)

    # Find bus model column using the enhanced detection function
    bus_model_col = find_bus_model_column(original_columns)

    # Find Line Location specific columns
    station_no_col = find_column_by_keywords(original_columns, ['STATION_NO', 'STATION NO', 'STATION', 'STN_NO', 'STN NO'])
    rack_col = find_column_by_keywords(original_columns, ['RACK'])
    rack_no_col = find_column_by_keywords(original_columns, ['RACK_NO', 'RACK NO', 'RACKNO'])
    level_col = find_column_by_keywords(original_columns, ['LEVEL'])
    cell_col = find_column_by_keywords(original_columns, ['CELL'])

    # Find Store Location specific columns
    zone_col = find_column_by_keywords(original_columns, ['ZONE'])
    store_location_col = find_column_by_keywords(original_columns, ['STORE_LOCATION', 'STORE LOCATION', 'STORELOCATION'])
    floor_col = find_column_by_keywords(original_columns, ['FLOOR'])
    level_in_rack_col = find_column_by_keywords(original_columns, ['LEVEL_IN_RACK', 'LEVEL IN RACK', 'LEVELINRACK'])
    no_col = find_column_by_keywords(original_columns, ['NO'])

    if status_callback:
        status_callback(f"Using columns: Part No: {part_no_col}, Description: {desc_col}, Location: {loc_col}, Qty/Bin: {qty_bin_col}")
        if qty_veh_col:
            status_callback(f"Qty/Veh Column: {qty_veh_col}")
        if bus_model_col:
            status_callback(f"Bus Model Column: {bus_model_col}")
        if station_no_col:
            status_callback(f"Station No Column: {station_no_col}")
        if rack_col:
            status_callback(f"Rack Column: {rack_col}")
        if rack_no_col:
            status_callback(f"Rack No Column: {rack_no_col}")
        if level_col:
            status_callback(f"Level Column: {level_col}")
        if cell_col:
            status_callback(f"Cell Column: {cell_col}")
        if zone_col:
            status_callback(f"Zone Column: {zone_col}")
        if store_location_col:
            status_callback(f"Store Location Column: {store_location_col}")
        if floor_col:
            status_callback(f"Floor Column: {floor_col}")
        if level_in_rack_col:
            status_callback(f"Level in Rack Column: {level_in_rack_col}")
        if no_col:
            status_callback(f"No Column: {no_col}")
    else:
        st.write(f"Using columns: Part No: {part_no_col}, Description: {desc_col}, Location: {loc_col}, Qty/Bin: {qty_bin_col}")
        if qty_veh_col:
            st.write(f"Qty/Veh Column: {qty_veh_col}")
        if bus_model_col:
            st.write(f"Bus Model Column: {bus_model_col}")
        if station_no_col:
            st.write(f"Station No Column: {station_no_col}")
        if rack_col:
            st.write(f"Rack Column: {rack_col}")
        if rack_no_col:
            st.write(f"Rack No Column: {rack_no_col}")
        if level_col:
            st.write(f"Level Column: {level_col}")
        if cell_col:
            st.write(f"Cell Column: {cell_col}")
        if zone_col:
            st.write(f"Zone Column: {zone_col}")
        if store_location_col:
            st.write(f"Store Location Column: {store_location_col}")
        if floor_col:
            st.write(f"Floor Column: {floor_col}")
        if level_in_rack_col:
            st.write(f"Level in Rack Column: {level_in_rack_col}")
        if no_col:
            st.write(f"No Column: {no_col}")

    # Create document with minimal margins
    doc = SimpleDocTemplate(output_pdf_path, pagesize=STICKER_PAGESIZE,
                          topMargin=0.2*cm,  # Minimal top margin
                          bottomMargin=(STICKER_HEIGHT - CONTENT_BOX_HEIGHT - 0.2*cm),  # Adjust bottom margin accordingly
                          leftMargin=0.1*cm, rightMargin=0.1*cm)

    content_width = CONTENT_BOX_WIDTH - 0.2*cm
    all_elements = []

    # Process each row as a single sticker
    total_rows = len(df)
    for index, row in df.iterrows():
        # Update progress
        if status_callback:
            status_callback(f"Creating sticker {index+1} of {total_rows} ({int((index+1)/total_rows*100)}%)")
        
        elements = []

        # Extract data
        part_no = str(row[part_no_col])
        desc = str(row[desc_col])
        
        # Extract QTY/BIN properly
        qty_bin = ""
        if qty_bin_col and qty_bin_col in row and pd.notna(row[qty_bin_col]):
            qty_bin = str(row[qty_bin_col])
            
        # Extract QTY/VEH properly
        qty_veh = ""
        if qty_veh_col and qty_veh_col in row and pd.notna(row[qty_veh_col]):
            qty_veh = str(row[qty_veh_col])
        
        location_str = str(row[loc_col]) if loc_col and loc_col in row else ""
        
        # Extract Line Location values using new function
        location_parts = extract_line_location_values(row, bus_model_col, station_no_col, rack_col, rack_no_col, level_col, cell_col)
        
        # Extract Store Location values using new function
        store_location_values = extract_store_location_values(row, zone_col, store_location_col, floor_col, rack_no_col, level_in_rack_col, cell_col, no_col)

        # Use enhanced bus model detection
        mtm_quantities = detect_bus_model_and_qty(row, qty_veh_col, bus_model_col)

        # Generate QR code with part information
        qr_data = f"Part No: {part_no}\nDescription: {desc}\nLocation: {location_str}\n"
        qr_data += f"Store Location: {' '.join(store_location_values)}\nQTY/VEH: {qty_veh}\nQTY/BIN: {qty_bin}"
        
        qr_image = generate_qr_code(qr_data)
        if status_callback and qr_image:
            status_callback(f"QR code generated for part: {part_no}")
        
        # Define row heights
        header_row_height = 0.9*cm
        desc_row_height = 1.0*cm
        qty_row_height = 0.5*cm
        location_row_height = 0.5*cm

        # Main table data
        main_table_data = [
            ["Part No", Paragraph(f"{part_no}", bold_style)],
            ["Description", Paragraph(desc[:47] + "..." if len(desc) > 50 else desc, desc_style)],
            ["Qty/Bin", Paragraph(str(qty_bin), qty_style)]
        ]

        # Create main table
        main_table = Table(main_table_data,
                         colWidths=[content_width/3, content_width*2/3],
                         rowHeights=[header_row_height, desc_row_height, qty_row_height])

        main_table.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 1.2, colors.Color(0, 0, 0, alpha=0.95)),  # Darker grid lines
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (0, -1), 11),
        ]))

        elements.append(main_table)

        # Store Location section
        store_loc_label = Paragraph("Store Location", ParagraphStyle(
            name='StoreLoc', fontName='Helvetica-Bold', fontSize=11, alignment=TA_CENTER
        ))

        # Total width for the 7 inner columns (2/3 of full content width)
        inner_table_width = content_width * 2 / 3
        
        # Define proportional widths - same as Line Location for consistency
        col_widths = [inner_table_width/7] * 7
        
        store_loc_data = [
            ["Zone", "Location", "Floor", "Rack No", "Level in Rack", "Cell", "No"],
            store_location_values
        ]

        store_loc_table = Table(store_loc_data,
                              colWidths=col_widths,
                              rowHeights=[location_row_height, location_row_height])

        store_loc_table.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 1.2, colors.Color(0, 0, 0, alpha=0.95)),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
        ]))

        # Combine Store Location label and table
        store_loc_combined_data = [
            [store_loc_label, ""],
            ["", store_loc_table]
        ]

        store_loc_combined = Table(store_loc_combined_data,
                                 colWidths=[content_width/3, content_width*2/3],
                                 rowHeights=[location_row_height, location_row_height*2])

        store_loc_combined.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 1.2, colors.Color(0, 0, 0, alpha=0.95)),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('SPAN', (0, 0), (0, 1)),
            ('SPAN', (1, 0), (1, 1)),
        ]))

        elements.append(store_loc_combined)

        # Line Location section
        line_loc_label = Paragraph("Line Location", ParagraphStyle(
            name='LineLoc', fontName='Helvetica-Bold', fontSize=11, alignment=TA_CENTER
        ))

        line_loc_data = [
            ["Bus Model", "Station No", "Rack", "Rack No", "", "Level", "Cell"],
            [location_parts[0], location_parts[1], location_parts[2], 
             location_parts[3], location_parts[4], location_parts[5], location_parts[6]]
        ]

        line_loc_table = Table(line_loc_data,
                             colWidths=col_widths,
                             rowHeights=[location_row_height, location_row_height])

        line_loc_table.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 1.2, colors.Color(0, 0, 0, alpha=0.95)),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('SPAN', (3, 0), (4, 0)),  # Span "Rack No" header across two columns
        ]))

        # Combine Line Location label and table
        line_loc_combined_data = [
            [line_loc_label, ""],
            ["", line_loc_table]
        ]

        line_loc_combined = Table(line_loc_combined_data,
                                colWidths=[content_width/3, content_width*2/3],
                                rowHeights=[location_row_height, location_row_height*2])

        line_loc_combined.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 1.2, colors.Color(0, 0, 0, alpha=0.95)),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('SPAN', (0, 0), (0, 1)),
            ('SPAN', (1, 0), (1, 1)),
        ]))

        elements.append(line_loc_combined)

        # MTM section
        mtm_label = Paragraph("MTM", ParagraphStyle(
            name='MTM', fontName='Helvetica-Bold', fontSize=11, alignment=TA_CENTER
        ))

        mtm_data = [
            ["4W", "3WS", "3WM", "3WC"],
            [mtm_quantities.get('4W', ''), mtm_quantities.get('3WS', ''), 
             mtm_quantities.get('3WM', ''), mtm_quantities.get('3WC', '')]
        ]

        mtm_table = Table(mtm_data,
                        colWidths=[inner_table_width/4] * 4,
                        rowHeights=[location_row_height, location_row_height])

        mtm_table.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 1.2, colors.Color(0, 0, 0, alpha=0.95)),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
        ]))

        # Combine MTM label and table
        mtm_combined_data = [
            [mtm_label, ""],
            ["", mtm_table]
        ]

        mtm_combined = Table(mtm_combined_data,
                           colWidths=[content_width/3, content_width*2/3],
                           rowHeights=[location_row_height, location_row_height*2])

        mtm_combined.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 1.2, colors.Color(0, 0, 0, alpha=0.95)),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('SPAN', (0, 0), (0, 1)),
            ('SPAN', (1, 0), (1, 1)),
        ]))

        elements.append(mtm_combined)

        # QR Code section (if available)
        if qr_image:
            qr_label = Paragraph("QR Code", ParagraphStyle(
                name='QR', fontName='Helvetica-Bold', fontSize=11, alignment=TA_CENTER
            ))

            qr_combined_data = [
                [qr_label, ""],
                ["", qr_image]
            ]

            qr_combined = Table(qr_combined_data,
                              colWidths=[content_width/3, content_width*2/3],
                              rowHeights=[location_row_height, 2.5*cm])

            qr_combined.setStyle(TableStyle([
                ('GRID', (0, 0), (-1, -1), 1.2, colors.Color(0, 0, 0, alpha=0.95)),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('SPAN', (0, 0), (0, 1)),
                ('SPAN', (1, 0), (1, 1)),
            ]))

            elements.append(qr_combined)

        # Add all elements for this sticker
        all_elements.extend(elements)
        
        # Add page break if not the last sticker
        if index < total_rows - 1:
            all_elements.append(PageBreak())

    # Build PDF
    if status_callback:
        status_callback("Building PDF...")
    else:
        st.write("Building PDF...")
    
    doc.build(all_elements, onFirstPage=draw_border, onLaterPages=draw_border)
    
    if status_callback:
        status_callback(f"PDF generated successfully: {output_pdf_path}")
    else:
        st.success(f"PDF generated successfully: {output_pdf_path}")
    
    return output_pdf_path

def main():
    st.set_page_config(page_title="Sticker Label Generator", layout="wide")
    
    st.title("📋 Sticker Label Generator")
    st.markdown("Generate professional sticker labels with QR codes from Excel/CSV data")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose an Excel or CSV file",
        type=['xlsx', 'xls', 'csv'],
        help="Upload your Excel or CSV file containing part information"
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_input_path = tmp_file.name
        
        # Display file info
        st.success(f"File uploaded: {uploaded_file.name}")
        
        # Preview data
        try:
            if uploaded_file.name.lower().endswith('.csv'):
                df_preview = pd.read_csv(temp_input_path)
            else:
                df_preview = pd.read_excel(temp_input_path)
            
            st.subheader("📊 Data Preview")
            st.dataframe(df_preview.head(), use_container_width=True)
            st.info(f"Total rows: {len(df_preview)}")
            
        except Exception as e:
            st.error(f"Error reading file: {e}")
            return
        
        # Generate button
        if st.button("🏷️ Generate Sticker Labels", type="primary"):
            # Create temporary output file
            output_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
            output_path = output_file.name
            output_file.close()
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def update_status(message):
                status_text.text(message)
                # Extract percentage if available
                if "%" in message:
                    try:
                        percent = int(message.split("(")[1].split("%")[0])
                        progress_bar.progress(percent / 100)
                    except:
                        pass
            
            try:
                # Generate PDF
                result_path = generate_sticker_labels(
                    temp_input_path, 
                    output_path, 
                    status_callback=update_status
                )
                
                if result_path:
                    progress_bar.progress(100)
                    status_text.text("✅ PDF generation completed!")
                    
                    # Read the generated PDF
                    with open(result_path, 'rb') as pdf_file:
                        pdf_data = pdf_file.read()
                    
                    # Download button
                    st.download_button(
                        label="📥 Download Sticker Labels PDF",
                        data=pdf_data,
                        file_name=f"sticker_labels_{uploaded_file.name.split('.')[0]}.pdf",
                        mime="application/pdf"
                    )
                    
                    st.success("🎉 Sticker labels generated successfully!")
                    
            except Exception as e:
                st.error(f"Error generating PDF: {e}")
                import traceback
                st.error(traceback.format_exc())
            
            finally:
                # Clean up temporary files
                try:
                    os.unlink(temp_input_path)
                    os.unlink(output_path)
                except:
                    pass
    
    else:
        st.info("👆 Please upload an Excel or CSV file to get started")
        
        # Show example format
        st.subheader("📋 Expected File Format")
        st.markdown("""
        Your Excel/CSV file should contain columns like:
        - **Part No/Part Number**: Part identification
        - **Description**: Part description
        - **Location**: Storage location
        - **Qty/Bin**: Quantity per bin
        - **Qty/Veh**: Quantity per vehicle (optional)
        - **Bus Model**: Bus model information (optional)
        - **Station No, Rack, Level, Cell**: Line location details (optional)
        - **Zone, Floor**: Store location details (optional)
        """)

if __name__ == "__main__":
    main()
