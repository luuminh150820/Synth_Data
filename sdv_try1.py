import pandas as pd
import json
import os
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata

# --- Configuration ---
SCHEMA_FILE = "enhanced_schema2.json"
ORIGINAL_DATA_FILE = "dim_retail_casa_sample_data.csv"
SYNTHETIC_DATA_FILE = "synth3.xlsx"
OUTPUT_FILE = "synth_updated_numeric.xlsx"
# CTGAN Hyperparameters (optional, adjust as needed)
CTGAN_EPOCHS = 200 # Default is 300
CTGAN_BATCH_SIZE = 300 # Default is 500
CTGAN_VERBOSE = True # Set to False to reduce training output

# --- Helper Functions ---

def load_schema_get_numeric_cols(schema_path):
    """Loads the schema JSON and identifies numeric columns (uppercase)."""
    print(f"Loading schema from: {schema_path}")
    if not os.path.exists(schema_path):
        print(f"Error: Schema file not found at {schema_path}")
        return None
    try:
        with open(schema_path, 'r', encoding='utf-8') as f:
            schema = json.load(f)
    except Exception as e:
        print(f"Error reading or parsing schema file {schema_path}: {e}")
        return None

    numeric_cols_upper = []
    if "columns" in schema:
        for col_name_upper, col_info in schema["columns"].items():
            # Prioritize the 'is_numeric' flag calculated during schema generation
            is_numeric_stat = col_info.get("stats", {}).get("is_numeric", False)
            # Fallback to inferred data type if 'is_numeric' is missing
            inferred_type = col_info.get("data_type", "").lower()
            is_numeric_type = inferred_type in ['integer', 'float', 'numerical']

            if is_numeric_stat or (not "is_numeric" in col_info.get("stats", {}) and is_numeric_type) :
                 # Check if it's not flagged as having leading zeros, which might indicate string type despite numeric appearance
                 had_leading_zero = col_info.get("stats", {}).get("had_leading_zero", False)
                 if not had_leading_zero:
                     numeric_cols_upper.append(col_name_upper.upper())
                     print(f"  Identified numeric column: {col_name_upper.upper()}")
                 else:
                     print(f"  Skipping potential numeric column '{col_name_upper.upper()}' due to 'had_leading_zero' flag.")

    if not numeric_cols_upper:
        print("Warning: No numeric columns identified in the schema.")
    else:
        print(f"Total numeric columns identified: {len(numeric_cols_upper)}")
    return numeric_cols_upper

def load_data(file_path):
    """Loads data from CSV or XLSX, returns DataFrame and original column names map."""
    print(f"Loading data from: {file_path}")
    if not os.path.exists(file_path):
        print(f"Error: Data file not found at {file_path}")
        return None, None

    original_columns = {}
    df = None
    try:
        file_extension = os.path.splitext(file_path)[1].lower()
        if file_extension == '.xlsx' or file_extension == '.xls':
            # Read header first to get original column names
            header_df = pd.read_excel(file_path, nrows=0)
            original_columns = {col.upper(): col for col in header_df.columns}
            # Read full data, attempting numeric conversion where possible
            df = pd.read_excel(file_path)
            print(f"Successfully loaded Excel data. Shape: {df.shape}")
        elif file_extension == '.csv':
             # Read header first to get original column names
            header_df = pd.read_csv(file_path, encoding='utf-8', nrows=0) # Try utf-8 first for header
            original_columns = {col.upper(): col for col in header_df.columns}
            try:
                # Read full data with potential type inference
                df = pd.read_csv(file_path, encoding='utf-8', low_memory=False)
                print(f"Successfully loaded CSV data with UTF-8 encoding. Shape: {df.shape}")
            except UnicodeDecodeError:
                print("UTF-8 decoding failed for full CSV, trying latin-1...")
                # Reread header with latin-1
                header_df = pd.read_csv(file_path, encoding='latin-1', nrows=0)
                original_columns = {col.upper(): col for col in header_df.columns}
                # Read full data with latin-1
                df = pd.read_csv(file_path, encoding='latin-1', low_memory=False)
                print(f"Successfully loaded CSV data with latin-1 encoding. Shape: {df.shape}")
        else:
            print(f"Error: Unsupported file format: {file_extension}")
            return None, None

        # Ensure column names are uppercase for processing consistency
        df.columns = df.columns.str.upper()
        return df, original_columns

    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return None, None

# --- Main Execution ---

if __name__ == "__main__":
    print("--- Starting CTGAN Numeric Column Replacement Process ---")

    # 1. Identify Numeric Columns from Schema
    numeric_cols_upper = load_schema_get_numeric_cols(SCHEMA_FILE)
    if not numeric_cols_upper:
        print("Exiting: No numeric columns found in schema.")
        exit()

    # 2. Load Original Data (for training)
    original_df, _ = load_data(ORIGINAL_DATA_FILE) # We don't need original case map here
    if original_df is None:
        print("Exiting: Failed to load original data.")
        exit()
        
    # Limit to 10,000 rows for training
    MAX_TRAINING_ROWS = 10000
    if len(original_df) > MAX_TRAINING_ROWS:
        print(f"Limiting training data to {MAX_TRAINING_ROWS} rows (original size: {len(original_df)} rows)")
        original_df = original_df.sample(n=MAX_TRAINING_ROWS, random_state=42).reset_index(drop=True)
    else:
        print(f"Using all {len(original_df)} rows for training")

    # 3. Validate and Extract Original Numeric Data
    missing_in_original = [col for col in numeric_cols_upper if col not in original_df.columns]
    if missing_in_original:
        print(f"Error: The following numeric columns from the schema are missing in the original data file ({ORIGINAL_DATA_FILE}): {missing_in_original}")
        print("Exiting.")
        exit()

    original_numeric_df = original_df[numeric_cols_upper].copy()
    # Convert columns to numeric, coercing errors (this helps CTGAN)
    for col in original_numeric_df.columns:
        original_numeric_df[col] = pd.to_numeric(original_numeric_df[col], errors='coerce')

    # Handle potential NaNs introduced by coercion - CTGAN might handle them,
    # or you could fill them (e.g., with mean/median) if appropriate.
    # For simplicity, we'll let CTGAN handle them for now.
    print(f"Extracted numeric data for training from original file. Shape: {original_numeric_df.shape}")
    print(f"Columns being used for training: {original_numeric_df.columns.tolist()}")

    # 4. Create SingleTableMetadata object for CTGAN using JSON schema
    print("\nCreating metadata for the CTGAN model from JSON schema...")
    try:
        # Load the schema JSON file
        with open(SCHEMA_FILE, 'r', encoding='utf-8') as f:
            schema_data = json.load(f)
        
        # Create and set up metadata from the schema
        metadata = SingleTableMetadata()
        
        # Set up column metadata from the schema
        for column in numeric_cols_upper:
            # Set all numeric columns as numerical type in metadata
            metadata.add_column(column, sdtype='numerical')
        
        print("Metadata creation from schema complete.")
    except Exception as e:
        print(f"Error during metadata creation from schema: {e}")
        print("Exiting.")
        exit()

    # 5. Train CTGAN Model
    print("\nInitializing and training CTGAN model...")
    try:
        model = CTGANSynthesizer(
            metadata=metadata,
            epochs=CTGAN_EPOCHS,
            batch_size=CTGAN_BATCH_SIZE,
            verbose=CTGAN_VERBOSE
        )
        model.fit(original_numeric_df)
        print("CTGAN model training complete.")
    except Exception as e:
        print(f"Error during CTGAN model training: {e}")
        print("Exiting.")
        exit()

    # 6. Load Existing Synthetic Data (for modification)
    synthetic_df, synthetic_original_case_map = load_data(SYNTHETIC_DATA_FILE)
    if synthetic_df is None:
        print("Exiting: Failed to load synthetic data file.")
        exit()

    # 7. Validate numeric columns in synthetic data
    missing_in_synthetic = [col for col in numeric_cols_upper if col not in synthetic_df.columns]
    if missing_in_synthetic:
        print(f"Warning: The following numeric columns identified for replacement are missing in the synthetic data file ({SYNTHETIC_DATA_FILE}): {missing_in_synthetic}")
        # Filter out columns that aren't in the synthetic file to avoid errors
        numeric_cols_upper = [col for col in numeric_cols_upper if col not in missing_in_synthetic]
        if not numeric_cols_upper:
            print("Error: No numeric columns to replace exist in the synthetic file. Exiting.")
            exit()
        print(f"Proceeding with replacement for existing columns: {numeric_cols_upper}")

    num_rows_to_generate = len(synthetic_df)
    print(f"\nGenerating {num_rows_to_generate} rows of new synthetic data for numeric columns...")

    # 8. Generate New Numeric Data
    try:
        generated_numeric_data = model.sample(num_rows=num_rows_to_generate)
        print(f"Successfully generated new numeric data. Shape: {generated_numeric_data.shape}")
        # Ensure generated data has the correct column names (it should, but double-check)
        generated_numeric_data.columns = numeric_cols_upper
    except Exception as e:
        print(f"Error during CTGAN data sampling: {e}")
        print("Exiting.")
        exit()

    # 9. Replace Columns in Synthetic Data
    print("Replacing numeric columns in the synthetic dataset...")
    # Separate non-numeric columns from the original synthetic data
    non_numeric_cols_upper = [col for col in synthetic_df.columns if col not in numeric_cols_upper]
    synthetic_non_numeric_df = synthetic_df[non_numeric_cols_upper].copy()

    # Reset index on both parts to ensure alignment before concatenation
    synthetic_non_numeric_df.reset_index(drop=True, inplace=True)
    generated_numeric_data.reset_index(drop=True, inplace=True)

    # Combine non-numeric part with newly generated numeric part
    final_df_upper = pd.concat([synthetic_non_numeric_df, generated_numeric_data], axis=1)

    # Ensure the column order matches the original synthetic file if possible
    original_order_upper = list(synthetic_original_case_map.keys())
    # Filter to only include columns present in the final df
    final_columns_ordered_upper = [col for col in original_order_upper if col in final_df_upper.columns]
    # Add any potentially new columns (shouldn't happen here, but good practice)
    final_columns_ordered_upper.extend([col for col in final_df_upper.columns if col not in final_columns_ordered_upper])

    final_df_upper = final_df_upper[final_columns_ordered_upper]

    # Map column names back to the original case from the synthetic file
    final_df_original_case = final_df_upper.rename(columns=synthetic_original_case_map)
    print("Columns replaced successfully.")

    # 10. Save Updated File
    print(f"Saving updated data to: {OUTPUT_FILE}")
    try:
        final_df_original_case.to_excel(OUTPUT_FILE, index=False)
        print("File saved successfully.")
    except Exception as e:
        print(f"Error saving updated data to {OUTPUT_FILE}: {e}")

    print("\n--- CTGAN Numeric Column Replacement Process Finished ---")