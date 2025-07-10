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

def find_column_by_patterns(df_columns, patterns):
    """
    Find column by matching patterns (case-insensitive)
    """
    cols = [str(col).upper() for col in df_columns]
    
    for pattern in patterns:
        for i, col in enumerate(cols):
            if isinstance(pattern, str):
                if pattern.upper() in col:
                    return df_columns[i]
            elif callable(pattern):
                if pattern(col):
                    return df_columns[i]
    
    return None

def find_bus_model_column(df_columns):
    """
    Enhanced function to find the bus model column with better detection
    """
    patterns = [
        'BUS MODEL',
        'BUSMODEL',
        'BUS_MODEL',
        'MODEL',
        'BUS_TYPE',
        'BUSTYPE',
        'BUS TYPE',
        'VEHICLE_TYPE',
        'VEHICLETYPE',
        'VEHICLE TYPE',
        lambda col: 'BUS' in col and 'MODEL' in col,
        lambda col: 'BUS' in col and 'TYPE' in col,
        lambda col: 'VEHICLE' in col and 'MODEL' in col,
        lambda col: 'VEHICLE' in col and 'TYPE' in col,
        lambda col: 'MODEL' in col,
        lambda col: 'BUS' in col,
        lambda col: 'VEHICLE' in col,
    ]
    
    return find_column_by_patterns(df_columns, patterns)

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

def extract_line_location_values(row, df_columns):
    """
    Extract Line Location values from specific columns
    Returns a list of 7 values: [Bus Model, Station No, Rack, Rack No (1st digit), Rack No (2nd digit), Level, Cell]
    """
    location_values = [''] * 7
    
    # Define column patterns for Line Location
    line_location_patterns = {
        0: ['BUS MODEL', 'BUSMODEL', 'BUS_MODEL'],  # Bus Model
        1: ['STATION NO', 'STATION_NO', 'STATIONNO', 'STATION NUMBER', 'STATION'],  # Station No
        2: ['RACK', 'RACK_NO', 'RACKNO'],  # Rack (general rack column)
        3: ['RACK NO (1ST DIGIT)', 'RACK_NO_1ST_DIGIT', 'RACK NO 1ST DIGIT', 'RACK_NO_1ST', 'RACK_1ST_DIGIT'],  # Rack No (1st digit)
        4: ['RACK NO (2ND DIGIT)', 'RACK_NO_2ND_DIGIT', 'RACK NO 2ND DIGIT', 'RACK_NO_2ND', 'RACK_2ND_DIGIT'],  # Rack No (2nd digit)
        5: ['LEVEL', 'LEVEL_NO', 'LEVELNO'],  # Level
        6: ['CELL', 'CELL_NO', 'CELLNO']  # Cell
    }
    
    # Find and extract values for each position
    for position, patterns in line_location_patterns.items():
        col_name = find_column_by_patterns(df_columns, patterns)
        if col_name and col_name in row and pd.notna(row[col_name]):
            location_values[position] = str(row[col_name]).strip()
    
    return location_values

def extract_store_location_values(row, df_columns):
    """
    Extract Store Location values from specific ABB columns
    Returns a list of 7 values: [ABB ZONE, ABB LOCATION, ABB FLOOR, ABB RACK NO, ABB LEVEL IN RACK, ABB CELL, ABB NO]
    """
    store_values = [''] * 7
    
    # Define column patterns for Store Location
    store_location_patterns = {
        0: ['ABB ZONE', 'ABB_ZONE', 'ABBZONE'],  # ABB Zone
        1: ['ABB LOCATION', 'ABB_LOCATION', 'ABBLOCATION'],  # ABB Location
        2: ['ABB FLOOR', 'ABB_FLOOR', 'ABBFLOOR'],  # ABB Floor
        3: ['ABB RACK NO', 'ABB_RACK_NO', 'ABBRACKNO', 'ABB RACK'],  # ABB Rack No
        4: ['ABB LEVEL IN RACK', 'ABB_LEVEL_IN_RACK', 'ABBLEVELINRACK', 'ABB LEVEL'],  # ABB Level in Rack
        5: ['ABB CELL', 'ABB_CELL', 'ABBCELL'],  # ABB Cell
        6: ['ABB NO', 'ABB_NO', 'ABBNO']  # ABB No
    }
    
    # Find and extract values for each position
    for position, patterns in store_location_patterns.items():
        col_name = find_column_by_patterns(df_columns, patterns)
        if col_name and col_name in row and pd.notna(row[col_name]):
            store_values[position] = str(row[col_name]).strip()
    
    return store_values

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

    # Store original columns for pattern matching
    original_columns = df.columns.tolist()

    # Identify columns (case-insensitive)
    part_no_col = find_column_by_patterns(original_columns, [
        'PART NO', 'PART_NO', 'PARTNO', 'PART NUMBER', 'PART_NUMBER', 'PARTNUMBER', 'PART'
    ]) or original_columns[0]

    desc_col = find_column_by_patterns(original_columns, [
        'PART DESC', 'PART_DESC', 'PARTDESC', 'DESCRIPTION', 'DESC', 'PART DESCRIPTION', 'PART_DESCRIPTION'
    ]) or (original_columns[1] if len(original_columns) > 1 else part_no_col)

    # Look specifically for "QTY/BIN" column first, then fall back to general QTY column
    qty_bin_col = find_column_by_patterns(original_columns, [
        'QTY/BIN', 'QTY_BIN', 'QTYBIN', 'QTY BIN', 'QTY/BIN', 'QTY_PER_BIN'
    ]) or find_column_by_patterns(original_columns, [
        'QTY', 'QUANTITY', 'QTY_BIN'
    ])

    # Improved detection of QTY/VEH column
    qty_veh_col = find_column_by_patterns(original_columns, [
        'QTY/VEH', 'QTY_VEH', 'QTY PER VEH', 'QTYVEH', 'QTYPERCAR', 'QTYCAR', 'QTY/CAR', 'QTY_PER_VEH'
    ])

    # Find bus model column using the enhanced detection function
    bus_model_col = find_bus_model_column(original_columns)

    # Find bin type column
    bin_type_col = find_column_by_patterns(original_columns, [
        'BIN TYPE', 'BIN_TYPE', 'BINTYPE', 'BIN'
    ])

    if status_callback:
        status_callback(f"Using columns: Part No: {part_no_col}, Description: {desc_col}, Qty/Bin: {qty_bin_col}")
        if qty_veh_col:
            status_callback(f"Qty/Veh Column: {qty_veh_col}")
        if bus_model_col:
            status_callback(f"Bus Model Column: {bus_model_col}")
        if bin_type_col:
            status_callback(f"Bin Type Column: {bin_type_col}")
    else:
        st.write(f"Using columns: Part No: {part_no_col}, Description: {desc_col}, Qty/Bin: {qty_bin_col}")
        if qty_veh_col:
            st.write(f"Qty/Veh Column: {qty_veh_col}")
        if bus_model_col:
            st.write(f"Bus Model Column: {bus_model_col}")
        if bin_type_col:
            st.write(f"Bin Type Column: {bin_type_col}")

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
        part_no = str(row[part_no_col]) if part_no_col and part_no_col in row else ""
        desc = str(row[desc_col]) if desc_col and desc_col in row else ""
        
        # Extract QTY/BIN properly
        qty_bin = ""
        if qty_bin_col and qty_bin_col in row and pd.notna(row[qty_bin_col]):
            qty_bin = str(row[qty_bin_col])
            
        # Extract QTY/VEH properly
        qty_veh = ""
        if qty_veh_col and qty_veh_col in row and pd.notna(row[qty_veh_col]):
            qty_veh = str(row[qty_veh_col])

        # Extract Bin Type
        bin_type = ""
        if bin_type_col and bin_type_col in row and pd.notna(row[bin_type_col]):
            bin_type = str(row[bin_type_col])

        # Extract Line Location values using new function
        line_location_values = extract_line_location_values(row, original_columns)

        # Extract Store Location values using new function
        store_location_values = extract_store_location_values(row, original_columns)

        # Use enhanced bus model detection
        mtm_quantities = detect_bus_model_and_qty(row, qty_veh_col, bus_model_col)

        # Generate QR code with part information
        qr_data = f"Part No: {part_no}\nDescription: {desc}\n"
        qr_data += f"Line Location: {' | '.join(line_location_values)}\n"
        qr_data += f"Store Location: {' | '.join(store_location_values)}\n"
        qr_data += f"QTY/VEH: {qty_veh}\nQTY/BIN: {qty_bin}\nBin Type: {bin_type}"
        
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
        col_proportions = [1.5, 2, 0.7, 0.8, 1, 1, 0.9]
        total_proportion = sum(col_proportions)
        
        # Calculate column widths based on proportions 
        inner_col_widths = [w * inner_table_width / total_proportion for w in col_proportions]

        store_loc_inner_table = Table(
            [store_location_values],
            colWidths=inner_col_widths,
            rowHeights=[location_row_height]
        )

        store_loc_inner_table.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 1.2, colors.Color(0, 0, 0, alpha=0.95)),  # Darker grid lines
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),  # Make store location values bold
            ('FONTSIZE', (0, 0), (-1, -1), 9),
        ]))

        store_loc_table = Table(
            [[store_loc_label, store_loc_inner_table]],
            colWidths=[content_width/3, inner_table_width],
            rowHeights=[location_row_height]
        )

        store_loc_table.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 1.2, colors.Color(0, 0, 0, alpha=0.95)),  # Darker grid lines
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))

        elements.append(store_loc_table)

        # Line Location section
        line_loc_label = Paragraph("Line Location", ParagraphStyle(
            name='LineLoc', fontName='Helvetica-Bold', fontSize=11, alignment=TA_CENTER
        ))
        
        # Create the inner table with extracted values
        line_loc_inner_table = Table(
            [line_location_values],
            colWidths=inner_col_widths,
            rowHeights=[location_row_height]
        )
        
        line_loc_inner_table.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 1.2, colors.Color(0, 0, 0, alpha=0.95)),  # Darker grid lines
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),  # Make line location values bold
            ('FONTSIZE', (0, 0), (-1, -1), 9)
        ]))
        
        # Wrap the label and the inner table in a containing table
        line_loc_table = Table(
            [[line_loc_label, line_loc_inner_table]],
            colWidths=[content_width/3, inner_table_width],
            rowHeights=[location_row_height]
        )

        line_loc_table.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 1.2, colors.Color(0, 0, 0, alpha=0.95)),  # Darker grid lines
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))

        elements.append(line_loc_table)

        # Add smaller spacer between line location and bottom section
        elements.append(Spacer(1, 0.5*cm))

        # Bottom section - Enhanced with intelligent bus model detection
        mtm_box_width = 1.2*cm
        mtm_row_height = 1.5*cm

        # Create MTM boxes with detected quantities
        position_matrix_data = [
            ["4W", "3WS", "3WM", "3WC"],
            [
                Paragraph(f"<b>{mtm_quantities['4W']}</b>", ParagraphStyle(
                    name='Bold4W', fontName='Helvetica-Bold', fontSize=10, alignment=TA_CENTER
                )) if mtm_quantities['4W'] else "",
                Paragraph(f"<b>{mtm_quantities['3WS']}</b>", ParagraphStyle(
                    name='Bold3WS', fontName='Helvetica-Bold', fontSize=10, alignment=TA_CENTER
                )) if mtm_quantities['3WS'] else "",
                Paragraph(f"<b>{mtm_quantities['3WM']}</b>", ParagraphStyle(
                    name='Bold3WM', fontName='Helvetica-Bold', fontSize=10, alignment=TA_CENTER
                )) if mtm_quantities['3WM'] else "",
                Paragraph(f"<b>{mtm_quantities['3WC']}</b>", ParagraphStyle(
                    name='Bold3WC', fontName='Helvetica-Bold', fontSize=10, alignment=TA_CENTER
                )) if mtm_quantities['3WC'] else ""
            ]
        ]

        position_matrix_table = Table(
            position_matrix_data,
            colWidths=[mtm_box_width] * 4,
            rowHeights=[0.4*cm, mtm_row_height]
        )

        position_matrix_table.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 1.2, colors.Color(0, 0, 0, alpha=0.95)),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('FONTSIZE', (0, 1), (-1, 1), 10),
        ]))

        # Create bottom section with QR code, bin type, and MTM boxes
        bottom_section_data = []
        
        # If QR code exists, create a row with QR code and position matrix
        if qr_image:
            # Create a table for bin type info
            bin_info_data = [
                ["Bin Type", Paragraph(str(bin_type), ParagraphStyle(
                    name='BinType', fontName='Helvetica-Bold', fontSize=10, alignment=TA_CENTER
                ))]
            ]
            
            bin_info_table = Table(
                bin_info_data,
                colWidths=[1.5*cm, 2*cm],
                rowHeights=[0.5*cm]
            )
            
            bin_info_table.setStyle(TableStyle([
                ('GRID', (0, 0), (-1, -1), 1.2, colors.Color(0, 0, 0, alpha=0.95)),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
            ]))
            
            bottom_section_data = [[qr_image, bin_info_table, position_matrix_table]]
            
            bottom_section_table = Table(
                bottom_section_data,
                colWidths=[2.5*cm, 3.5*cm, 4.8*cm],
                rowHeights=[mtm_row_height]
            )
        else:
            # If no QR code, create a simpler layout
            bin_info_data = [
                ["Bin Type", Paragraph(str(bin_type), ParagraphStyle(
                    name='BinType', fontName='Helvetica-Bold', fontSize=10, alignment=TA_CENTER
                ))]
            ]
            
            bin_info_table = Table(
                bin_info_data,
                colWidths=[2*cm, 3*cm],
                rowHeights=[0.5*cm]
            )
            
            bin_info_table.setStyle(TableStyle([
                ('GRID', (0, 0), (-1, -1), 1.2, colors.Color(0, 0, 0, alpha=0.95)),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
            ]))
            
            bottom_section_data = [[bin_info_table, position_matrix_table]]
            
            bottom_section_table = Table(
                bottom_section_data,
                colWidths=[5*cm, 4.8*cm],
                rowHeights=[mtm_row_height]
            )

        bottom_section_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))

        elements.append(bottom_section_table)

        # Add all elements to the main list
        all_elements.extend(elements)
        
        # Add page break if not the last sticker
        if index < total_rows - 1:
            all_elements.append(PageBreak())

    # Build the PDF
    try:
        doc.build(all_elements, onFirstPage=draw_border, onLaterPages=draw_border)
        
        success_msg = f"Generated {total_rows} stickers successfully!"
        if status_callback:
            status_callback(success_msg)
        else:
            st.success(success_msg)
        
        return output_pdf_path
        
    except Exception as e:
        error_msg = f"Error generating PDF: {e}"
        if status_callback:
            status_callback(error_msg)
        else:
            st.error(error_msg)
        import traceback
        traceback.print_exc()
        return None

def main():
    st.set_page_config(page_title="Sticker Label Generator", layout="wide")
    
    st.title("🏷️ Sticker Label Generator")
    st.markdown("Generate professional sticker labels with QR codes from Excel/CSV data")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload your Excel or CSV file", 
        type=['xlsx', 'xls', 'csv'],
        help="Upload an Excel or CSV file containing part information"
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_input_path = tmp_file.name
        
        # Display file info
        st.success(f"File uploaded: {uploaded_file.name}")
        
        # Preview data
        try:
            if uploaded_file.name.lower().endswith('.csv'):
                preview_df = pd.read_csv(temp_input_path)
            else:
                preview_df = pd.read_excel(temp_input_path)
            
            st.subheader("📊 Data Preview")
            st.dataframe(preview_df.head(10))
            
            st.info(f"Total rows: {len(preview_df)}")
            
        except Exception as e:
            st.error(f"Error reading file: {e}")
            return
        
        # Generate stickers button
        if st.button("🚀 Generate Stickers", type="primary"):
            # Create output filename
            output_filename = f"stickers_{uploaded_file.name.split('.')[0]}.pdf"
            temp_output_path = os.path.join(tempfile.gettempdir(), output_filename)
            
            # Create progress bar and status text
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
            
            # Generate stickers
            result_path = generate_sticker_labels(
                temp_input_path, 
                temp_output_path, 
                status_callback=update_status
            )
            
            if result_path and os.path.exists(result_path):
                progress_bar.progress(1.0)
                status_text.success("✅ Stickers generated successfully!")
                
                # Provide download button
                with open(result_path, 'rb') as pdf_file:
                    pdf_bytes = pdf_file.read()
                
                st.download_button(
                    label="📥 Download Stickers PDF",
                    data=pdf_bytes,
                    file_name=output_filename,
                    mime="application/pdf"
                )
                
                # Clean up temporary files
                try:
                    os.unlink(temp_input_path)
                    os.unlink(result_path)
                except:
                    pass
            else:
                st.error("❌ Failed to generate stickers. Please check your file format and try again.")
    
    else:
        st.info("👆 Please upload an Excel or CSV file to get started")
        
        # Show example of expected format
        st.subheader("📋 Expected File Format")
        st.markdown("""
        Your file should contain the following columns (case-insensitive):
        - **Part No**: Part number/identifier
        - **Part Desc**: Part description
        - **QTY/BIN**: Quantity per bin
        - **QTY/VEH**: Quantity per vehicle (optional)
        - **Bus Model**: Bus model information (optional)
        - **Bin Type**: Type of bin (optional)
        - **Line Location columns**: Station No, Rack, Level, Cell, etc.
        - **Store Location columns**: ABB Zone, ABB Location, ABB Floor, etc.
        """)
        
        # Show sample data
        sample_data = {
            'Part No': ['P001', 'P002', 'P003'],
            'Part Desc': ['Brake Pad Assembly', 'Air Filter Element', 'Oil Filter Cartridge'],
            'QTY/BIN': [10, 5, 8],
            'QTY/VEH': [2, 1, 1],
            'Bus Model': ['4W', '3WS', '3WM'],
            'Bin Type': ['Small', 'Medium', 'Large']
        }
        
        sample_df = pd.DataFrame(sample_data)
        st.dataframe(sample_df)

if __name__ == "__main__":
    main()
