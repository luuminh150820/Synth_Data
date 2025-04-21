import pandas as pd
import json
import os
import numpy as np
from sdv.single_table import GaussianCopulaSynthesizer
# Corrected imports for constraints - We will now work with dictionaries directly
# from sdv.constraints.tabular import ScalarRange, Unique
from sdv.metadata import SingleTableMetadata
import random
from faker import Faker
import inspect
import warnings # Import warnings module

# Configuration
INPUT_CSV = "customer_data.csv"  # Input CSV file
OUTPUT_CSV = "synthetic_data.csv"  # Output CSV file for synthetic data
INPUT_SCHEMA_JSON = "enhanced_schema.json"  # Input JSON file for the enhanced schema
NUM_ROWS = 1000  # Number of synthetic rows to generate
CORRELATION_THRESHOLD = 0.7  # Define the correlation threshold

# Initialize Faker
fake = Faker(['vi_VN'])
Faker.seed(42)

# --- Helper Functions ---

def read_schema(json_file_path):
    """Reads the enhanced schema from a JSON file."""
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            schema = json.load(f)
        print(f"Successfully read schema from {json_file_path}")
        return schema
    except FileNotFoundError:
        print(f"Error: Schema file not found at {json_file_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {json_file_path}: {str(e)}")
        return None
    except Exception as e:
        print(f"Error reading schema: {str(e)}")
        return None

def detect_column_correlations(df):
    """Detects column correlations, including functional dependencies."""
    correlations = {}
    try:
        # Select only columns that are purely numeric (int or float)
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if len(numeric_cols) < 2:
            print("Not enough numeric columns to calculate correlations.")
            return correlations

        # Ensure all selected columns are indeed numeric before calculating correlation
        df_numeric = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        df_numeric = df_numeric.dropna(axis=1, how='all') # Drop columns that became all NaN
        numeric_cols = df_numeric.columns.tolist() # Update numeric_cols list

        if len(numeric_cols) < 2:
            print("Not enough valid numeric columns after conversion for correlation calculation.")
            return correlations

        corr_matrix = df_numeric.corr(method='pearson') # numeric_only=True is default

        # Pearson Correlation
        for i, col1 in enumerate(numeric_cols):
            for j, col2 in enumerate(numeric_cols):
                if i < j:
                    corr_value = corr_matrix.loc[col1, col2]
                    if abs(corr_value) >= CORRELATION_THRESHOLD and not pd.isna(corr_value):
                        if col1 not in correlations:
                            correlations[col1] = []
                        correlations[col1].append({
                            "column": col2,
                            "correlation": round(corr_value, 3),
                            "type": "positive" if corr_value > 0 else "negative"
                        })
                        # Add reciprocal correlation
                        if col2 not in correlations:
                            correlations[col2] = []
                        correlations[col2].append({
                            "column": col1,
                            "correlation": round(corr_value, 3),
                            "type": "positive" if corr_value > 0 else "negative"
                        })


        # Functional Dependency (Simple Check: one value maps to only one other value)
        # Note: This is a basic check and might not capture all functional dependencies.
        all_cols = df.columns.tolist() # Check dependency across all columns
        for col1 in all_cols:
            for col2 in all_cols:
                if col1 != col2:
                    try:
                        # Drop rows where either column is NaN for this check
                        df_filtered = df[[col1, col2]].dropna()
                        if not df_filtered.empty:
                            # Check if each value in col1 maps to at most one unique value in col2
                            # Ensure col1 has hashable types for grouping
                            # Check if column dtype is hashable before attempting iloc[0] access on potentially empty series
                            if pd.api.types.is_hashable(df_filtered[col1].dtype):
                                # Further check if the filtered series is not empty before accessing iloc[0]
                                if not df_filtered[col1].empty:
                                     # Now safe to check if the first element is hashable (covers lists/dicts if they sneak in)
                                     if pd.api.types.is_hashable(df_filtered[col1].iloc[0]):
                                        unique_values_per_group = df_filtered.groupby(col1)[col2].nunique()
                                        if (unique_values_per_group <= 1).all():
                                            if col1 not in correlations:
                                                correlations[col1] = []
                                            # Avoid adding duplicate dependency entries
                                            if not any(d.get("column") == col2 and d.get("type") == "functional_dependency" for d in correlations[col1]):
                                                correlations[col1].append({
                                                    "column": col2,
                                                    "correlation": 1.0, # Represent dependency
                                                    "type": "functional_dependency"
                                                })
                                                print(f"Detected potential functional dependency: {col1} -> {col2}")
                            # else: # Optional: Handle non-hashable types if necessary
                            #    print(f"Skipping functional dependency check for non-hashable column: {col1}")

                    except pd.errors.UnhashableTypeError:
                         print(f"Skipping functional dependency check for {col1} -> {col2} due to unhashable type in {col1}.")
                    except Exception as dep_e:
                        # Handle cases where grouping might fail (e.g., mixed types within col1)
                        print(f"Could not check functional dependency for {col1} -> {col2}: {dep_e}")


    except Exception as e:
        print(f"Error detecting correlations: {str(e)}")
        import traceback
        traceback.print_exc() # Print full traceback for correlation errors
    return correlations


def prepare_sdv_metadata_components(df, enhanced_schema):
    """
    Prepares the components needed to create SDV metadata:
    columns dictionary, constraints list (as dictionaries), and primary key.
    """
    columns_dict = {}
    constraint_dicts = []
    primary_key = None

    # First pass: Detect initial metadata and identify primary key from schema
    temp_metadata = SingleTableMetadata()
    temp_metadata.detect_from_dataframe(data=df)
    temp_metadata_dict = temp_metadata.to_dict() # Get initial detection as dict

    # Populate columns dictionary based on detection and schema overrides
    for col_name in df.columns:
         columns_dict[col_name] = temp_metadata_dict['columns'].get(col_name, {}) # Start with detected info

    # Second pass: Apply schema overrides and identify primary key
    for col_name, col_schema in enhanced_schema.items():
        if col_name not in columns_dict:
            print(f"Warning: Column '{col_name}' from schema not found in DataFrame. Skipping metadata update.")
            continue

        # Apply schema overrides for sdtype, pii, etc.
        data_type = col_schema.get("data_type", "").lower()
        if data_type in ["numerical", "integer", "float"]:
             columns_dict[col_name]['sdtype'] = 'numerical'
        elif data_type == "categorical":
             columns_dict[col_name]['sdtype'] = 'categorical'
        elif data_type == "datetime":
             columns_dict[col_name]['sdtype'] = 'datetime'
             if "datetime_format" in col_schema:
                 columns_dict[col_name]['datetime_format'] = col_schema["datetime_format"]
        elif data_type == "boolean":
             columns_dict[col_name]['sdtype'] = 'boolean'
        elif data_type == "id":
             columns_dict[col_name]['sdtype'] = 'id'

        if col_schema.get("is_pii", False):
             columns_dict[col_name]['pii'] = True

        # Identify primary key from schema
        if col_schema.get("key_type", "").lower() == "primary key":
            if primary_key is None:
                primary_key = col_name
                print(f"Identified primary key from schema: '{primary_key}'")
                # *** FIX: Ensure primary key column sdtype is 'id' ***
                columns_dict[col_name]['sdtype'] = 'id' # Explicitly set sdtype to 'id'
            else:
                print(f"Warning: Multiple primary keys defined in schema ('{primary_key}', '{col_name}'). Using the first one found: '{primary_key}'.")

    # Fallback to detected primary key if schema doesn't define one
    if primary_key is None and 'primary_key' in temp_metadata_dict:
         primary_key = temp_metadata_dict['primary_key']
         print(f"Using detected primary key: '{primary_key}' (not defined in schema).")
         # *** FIX: Ensure detected primary key column sdtype is 'id' ***
         if primary_key in columns_dict:
             columns_dict[primary_key]['sdtype'] = 'id' # Explicitly set detected PK sdtype to 'id'

    elif primary_key is None:
         print("Warning: No primary key defined in schema or detected from data.")


    # Third pass: Parse and collect constraints as dictionaries
    print("\nParsing constraints from enhanced schema...")
    # Need primary key info during constraint parsing, so pass it
    current_primary_key = primary_key # Use the identified primary key

    for col_name, col_schema in enhanced_schema.items():
        if col_name in df.columns and "sdv_constraints" in col_schema and col_schema["sdv_constraints"]:
            print(f"   Parsing constraints for column: '{col_name}'")
            col_constraint_dicts = parse_sdv_constraints_to_dict(
                col_schema["sdv_constraints"],
                col_name,
                current_primary_key, # Pass the identified primary key
                enhanced_schema
            )
            # Ensure col_constraint_dicts is a list before extending
            if isinstance(col_constraint_dicts, list):
                constraint_dicts.extend(col_constraint_dicts)
                if col_constraint_dicts:
                     print(f"   Added {len(col_constraint_dicts)} constraint dictionary(ies) for '{col_name}'.")
            else:
                print(f"   Warning: parse_sdv_constraints_to_dict for '{col_name}' did not return a list. Skipping constraints from this column.")

    print(f"\nTotal constraint dictionaries collected: {len(constraint_dicts)}")

    # Return the components needed for the metadata constructor
    return columns_dict, constraint_dicts, primary_key


def parse_sdv_constraints_to_dict(constraints_list, col_name, primary_key, enhanced_schema):
    """
    Parses SDV constraints from the schema definition for a specific column
    and returns them as a list of constraint dictionaries in the format
    expected by SDV metadata.

    Args:
        constraints_list (list): List of constraint definitions from the schema.
        col_name (str): The name of the column the constraints apply to.
        primary_key (str or None): The name of the primary key column, if any.
        enhanced_schema (dict): The full schema dictionary (used for context if needed).

    Returns:
        list: A list of SDV constraint dictionaries.
    """
    constraint_dicts = []

    for constraint_def in constraints_list:
        try:
            constraint_type = None
            params = {}

            # --- Handle Dictionary-based Constraints ---
            if isinstance(constraint_def, dict):
                # Infer constraint type if not explicitly provided but min/max are
                if "constraint_type" not in constraint_def and ("min_value" in constraint_def or "max_value" in constraint_def):
                    constraint_type = "scalar_range" # Assume range if bounds are given
                    print(f"Assuming 'scalar_range' constraint for {col_name} based on presence of min/max_value.")
                else:
                    constraint_type = constraint_def.get("constraint_type", "").lower()

                params = constraint_def # Use the dict directly for params

                # --- Handle ScalarRange (covers range, greater than, less than) ---
                if constraint_type in ["range", "scalar_range", "greater_than", "less_than", "min_only", "max_only"]:
                    min_val = params.get("min_value")
                    max_val = params.get("max_value")
                    strict_low = params.get("strict_low", False) # Default to inclusive
                    strict_high = params.get("strict_high", False) # Default to inclusive

                    if min_val is None and max_val is None:
                        print(f"Warning: ScalarRange constraint for '{col_name}' has neither min_value nor max_value. Skipping.")
                        continue

                    try:
                        min_val = float(min_val) if min_val is not None else None
                        max_val = float(max_val) if max_val is not None else None
                    except (ValueError, TypeError) as e:
                        print(f"Warning: Invalid min/max value for ScalarRange on '{col_name}': min={min_val}, max={max_val}. Error: {e}. Skipping.")
                        continue

                    constraint_dicts.append({
                        'constraint_class': 'ScalarRange',
                        'constraint_parameters': {
                            'column_name': col_name,
                            'low': min_val,
                            'high': max_val,
                            'strict_low': strict_low, # Include strictness params
                            'strict_high': strict_high
                        }
                    })
                    print(f"Parsed ScalarRange constraint dict for '{col_name}'.")


                # --- Handle Unique Constraint ---
                elif constraint_type == "unique":
                    # Check if column is already primary key - SDV handles PK uniqueness internally
                    if col_name == primary_key:
                        print(f"Skipping explicit Unique constraint for '{col_name}' as it is the primary key.")
                        continue
                    constraint_dicts.append({
                        'constraint_class': 'Unique',
                        'constraint_parameters': {
                            'column_names': [col_name] # Unique takes a list of columns
                        }
                    })
                    print(f"Parsed Unique constraint dict for '{col_name}'.")

                # --- Handle ColumnFormula Constraint (REMOVED) ---
                elif constraint_type == "column_formula":
                     print(f"Warning: Constraint type 'column_formula' for column '{col_name}' is not supported in this SDV setup and will be skipped.")
                     continue # Skip this constraint

                else:
                    print(f"Warning: Unknown constraint type '{constraint_type}' in dict definition for column '{col_name}'. Skipping.")


            # --- Handle String-based Constraints (Simple Cases) ---
            elif isinstance(constraint_def, str):
                constraint_str = constraint_def.strip().lower() # Added strip() for safety

                # --- Handle plain "Range" string specifically ---
                if constraint_str == "range":
                     print(f"Warning: Unsupported constraint string format 'Range' for column '{col_name}'. Expected format like 'range:min,max'. Skipping.")
                     continue # Skip this specific unsupported format

                if constraint_str == "unique":
                    # Check if column is already primary key
                    if col_name == primary_key:
                        print(f"Skipping explicit Unique constraint for '{col_name}' as it is the primary key.")
                        continue
                    constraint_dicts.append({
                        'constraint_class': 'Unique',
                        'constraint_parameters': {
                            'column_names': [col_name]
                        }
                    })
                    print(f"Parsed Unique constraint dict for '{col_name}' from string.")

                # Example for range string: "range:0,100", "scalar_range:10,None", "range:None,50"
                elif constraint_str.startswith("range:") or constraint_str.startswith("scalar_range:"):
                     try:
                         parts_str = constraint_str.split(':', 1)[1] # Split only once

                         min_val = None
                         max_val = None

                         if parts_str:
                             parts = parts_str.split(',')
                             if len(parts) == 2:
                                 min_str = parts[0].strip().lower()
                                 max_str = parts[1].strip().lower()
                                 min_val = float(min_str) if min_str != 'none' and min_str else None # Also check for empty string
                                 max_val = float(max_str) if max_str != 'none' and max_str else None
                             else:
                                 print(f"Warning: Range string '{constraint_def}' for '{col_name}' has unexpected format. Skipping.")
                                 continue

                         if min_val is None and max_val is None:
                             print(f"Warning: Range string '{constraint_def}' for '{col_name}' has no valid bounds. Skipping.")
                             continue

                         constraint_dicts.append({
                            'constraint_class': 'ScalarRange',
                            'constraint_parameters': {
                                'column_name': col_name,
                                'low': min_val,
                                'high': max_val,
                                # String format doesn't easily support strict bounds, defaulting to inclusive
                                'strict_low': False,
                                'strict_high': False
                            }
                        })
                         print(f"Parsed ScalarRange constraint dict for '{col_name}' from string.")

                     except Exception as parse_e:
                         print(f"Error parsing range string '{constraint_def}' for column '{col_name}': {parse_e}. Skipping.")

                else:
                    print(f"Warning: Unknown or unsupported constraint string format '{constraint_def}' for column '{col_name}'. Skipping.")
            else:
                 print(f"Warning: Invalid constraint format '{constraint_def}' (type: {type(constraint_def)}) for column '{col_name}'. Skipping.")


        except Exception as e:
            print(f"Error processing constraint definition '{constraint_def}' for column '{col_name}': {str(e)}")
            import traceback
            traceback.print_exc() # Print detailed traceback for parsing errors

    return constraint_dicts


def generate_weighted_random_element(categories):
    """Generates a random element from a list of categories with weights."""
    # Check if categories list is valid and contains required keys
    if not isinstance(categories, list) or not categories:
        # print("Warning: Invalid or empty categories list provided for weighted random choice.")
        return None # Or return a default value, or raise an error

    # Filter out items with missing 'value' or 'percentage' or invalid percentage
    valid_categories = []
    for item in categories:
        if isinstance(item, dict) and "value" in item and "percentage" in item:
            try:
                # Ensure percentage is a valid number >= 0
                percentage = float(item["percentage"])
                if percentage >= 0 and not np.isnan(percentage) and not np.isinf(percentage):
                     valid_categories.append(item)
                # else: # Optional: Warn about invalid percentages
                #    print(f"Warning: Invalid percentage '{item['percentage']}' for value '{item['value']}'. Skipping category.")
            except (ValueError, TypeError):
                 # print(f"Warning: Could not convert percentage '{item['percentage']}' to float for value '{item['value']}'. Skipping category.")
                 continue


    if not valid_categories:
        # print("Warning: No valid categories with 'value' and 'percentage' found.")
        return None

    values = [item["value"] for item in valid_categories]
    # Recalculate weights from the filtered list
    weights = [float(item["percentage"]) for item in valid_categories]

    try:
        total_weight = sum(weights)
        if total_weight <= 0:
            # If all weights are 0, choose uniformly from values
            if all(w == 0 for w in weights):
                 # print(f"Warning: All valid category weights are zero. Choosing uniformly.")
                 return random.choice(values) if values else None
            else:
                 # print(f"Warning: Total weight is zero or negative ({total_weight}). Cannot perform weighted choice. Returning None.")
                 return None # Avoid division by zero or negative weights if library doesn't handle it


        # random.choices handles normalization automatically if weights don't sum to 1
        return random.choices(values, weights=weights, k=1)[0]

    except (ValueError, TypeError) as e:
        print(f"Error during weighted choice: {e}. Values: {values}, Weights: {weights}")
        return None # Or handle error appropriately
    except Exception as e: # Catch any other potential errors from random.choices
         print(f"Unexpected error during weighted choice: {e}. Values: {values}, Weights: {weights}")
         return None


def generate_synthetic_data_with_sdv(df, enhanced_schema, num_rows, output_file):
    """Generates synthetic data using SDV, applying Faker providers and post-processing."""
    try:
        print("\n--- Starting SDV Synthetic Data Generation ---")

        print("1. Creating initial SingleTableMetadata object and detecting from dataframe...")
        # Create an empty metadata object and detect from the dataframe
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(data=df)
        print("   Initial metadata detected from dataframe.")
        print(f"   Detected primary key: {metadata.primary_key}")


        print("\n2. Applying schema overrides to metadata and identifying primary key...")
        # Apply schema overrides (sdtypes, pii) and identify primary key from schema
        primary_key_from_schema = None
        for col_name, col_schema in enhanced_schema.items():
            if col_name in metadata.columns:
                update_params = {}
                data_type = col_schema.get("data_type", "").lower()
                if data_type in ["numerical", "integer", "float"]:
                     update_params['sdtype'] = 'numerical'
                elif data_type == "categorical":
                     update_params['sdtype'] = 'categorical'
                elif data_type == "datetime":
                     update_params['sdtype'] = 'datetime'
                     if "datetime_format" in col_schema:
                         update_params['datetime_format'] = col_schema["datetime_format"]
                elif data_type == "boolean":
                     update_params['sdtype'] = 'boolean'
                elif data_type == "id":
                     update_params['sdtype'] = 'id'

                if col_schema.get("is_pii", False):
                     update_params['pii'] = True

                if update_params:
                    try:
                        metadata.update_column(column_name=col_name, **update_params)
                        # print(f"Updated metadata for column '{col_name}': {update_params}")
                    except Exception as update_e:
                        print(f"Error updating metadata for column '{col_name}' with params {update_params}: {update_e}")

                # Identify primary key from schema during this pass
                if col_schema.get("key_type", "").lower() == "primary key":
                    if primary_key_from_schema is None:
                        primary_key_from_schema = col_name
                        print(f"Identified primary key from schema: '{primary_key_from_schema}'")
                        # *** FIX: Ensure primary key column sdtype is 'id' ***
                        # We update the metadata object directly here after identifying the PK
                        try:
                            metadata.update_column(column_name=col_name, sdtype='id')
                            print(f"   Set sdtype of primary key '{col_name}' to 'id'.")
                        except Exception as sdtype_update_e:
                            print(f"   Error setting sdtype of primary key '{col_name}' to 'id': {sdtype_update_e}")


                    else:
                        print(f"Warning: Multiple primary keys defined in schema ('{primary_key_from_schema}', '{col_name}'). Using the first one found: '{primary_key_from_schema}'.")

        # Set primary key in the metadata object after iterating through schema
        if primary_key_from_schema:
             try:
                 metadata.set_primary_key(primary_key_from_schema)
                 print(f"   Set primary key in metadata object: {metadata.primary_key}")
             except Exception as set_pk_e:
                 print(f"   Error setting primary key '{primary_key_from_schema}' in metadata object: {set_pk_e}")
                 # The InvalidMetadataError might still occur here if sdtype wasn't set correctly

        elif metadata.primary_key:
             print(f"   Using detected primary key: {metadata.primary_key} (not defined in schema)")
             # Ensure detected primary key sdtype is 'id' if it wasn't already
             if metadata.primary_key in metadata.columns and metadata.columns[metadata.primary_key].get('sdtype') != 'id':
                 try:
                     metadata.update_column(column_name=metadata.primary_key, sdtype='id')
                     print(f"   Set sdtype of detected primary key '{metadata.primary_key}' to 'id'.")
                 except Exception as sdtype_update_e:
                      print(f"   Error setting sdtype of detected primary key '{metadata.primary_key}' to 'id': {sdtype_update_e}")
        else:
             print("   Warning: No primary key defined in schema or detected.")


        # After applying schema overrides and setting PK, validate metadata
        try:
            metadata.validate() # Validate metadata after schema overrides and PK setting
            print("\nMetadata validation successful after schema overrides and PK setting.")
        except Exception as validate_e:
             print(f"\nMetadata validation failed after schema overrides and PK setting: {validate_e}")
             # The InvalidMetadataError is likely caught here


        print("\n3. Parsing constraints from enhanced schema (to dictionaries)...") # Step number updated
        # Need primary key info during constraint parsing, so pass it from the metadata object
        current_primary_key = metadata.primary_key

        constraint_dicts = []
        for col_name, col_schema in enhanced_schema.items():
            if col_name in df.columns and "sdv_constraints" in col_schema and col_schema["sdv_constraints"]:
                print(f"   Parsing constraints for column: '{col_name}'")
                col_constraint_dicts = parse_sdv_constraints_to_dict(
                    col_schema["sdv_constraints"],
                    col_name,
                    current_primary_key, # Pass the primary key from the metadata object
                    enhanced_schema
                )
                # Ensure col_constraint_dicts is a list before extending
                if isinstance(col_constraint_dicts, list):
                    constraint_dicts.extend(col_constraint_dicts)
                    if col_constraint_dicts:
                         print(f"   Added {len(col_constraint_dicts)} constraint dictionary(ies) for '{col_name}'.")
                else:
                    print(f"   Warning: parse_sdv_constraints_to_dict for '{col_name}' did not return a list. Skipping constraints from this column.")

        print(f"\nTotal constraint dictionaries collected: {len(constraint_dicts)}")


        print("\n4. Adding constraints to the metadata object...") # Step number updated
        try:
            if constraint_dicts:
                # Iterate through constraint dictionaries and try adding them
                # Assuming metadata.add_constraint accepts a constraint dictionary in this version
                constraints_added_count = 0
                for constraint_dict in constraint_dicts:
                    try:
                        # Attempt to add constraint using the dictionary format
                        # This is the crucial point to test if add_constraint accepts dicts
                        metadata.add_constraint(constraint_dict)
                        constraints_added_count += 1
                        print(f"   Successfully added constraint: {constraint_dict.get('constraint_class')}")
                    except Exception as add_c_e:
                        print(f"   Error adding constraint {constraint_dict.get('constraint_class')}: {add_c_e}")
                        import traceback
                        traceback.print_exc()

                print(f"   Attempted to add {len(constraint_dicts)} constraints, {constraints_added_count} added successfully.")
            else:
                print("   No constraints to add to metadata.")

            print("\nFinal metadata after applying schema and constraints:")
            print(metadata.to_dict()) # Print final metadata for verification
            try:
                metadata.validate() # Validate the final metadata
                print("\nMetadata validation successful.")
            except Exception as validate_e:
                 print(f"\nMetadata validation failed after adding constraints: {validate_e}")
                 # Decide if you want to proceed with potentially invalid metadata
                 # return False # Or raise error


        except Exception as meta_constraint_e:
            print(f"   Error adding parsed constraints to metadata: {meta_constraint_e}")
            import traceback
            traceback.print_exc()
            print("   Proceeding without adding all parsed constraints due to error.")


        print("\n5. Setting up SDV synthesizer (GaussianCopulaSynthesizer)...") # Step number updated
        # Initialize synthesizer with the metadata object
        synthesizer = GaussianCopulaSynthesizer(metadata=metadata)


        print("\n6. Fitting synthesizer to the original data...") # Step number updated
        try:
             # Suppress warnings during fitting if needed
             with warnings.catch_warnings():
                # warnings.filterwarnings('ignore', category=UserWarning, module='sdv')
                # warnings.filterwarnings('ignore', category=RuntimeWarning) # e.g., for covariance issues
                synthesizer.fit(df)
             print("   Synthesizer fitting completed.")
        except Exception as fit_e:
            print(f"   Error fitting synthesizer: {fit_e}")
            import traceback
            traceback.print_exc()
            print("   Aborting due to fitting error.")
            return False


        print(f"\n7. Generating {num_rows} rows of synthetic data...") # Step number updated
        try:
            # Suppress warnings during sampling if needed
            with warnings.catch_warnings():
                # warnings.filterwarnings('ignore', category=UserWarning, module='sdv')
                synthetic_data = synthesizer.sample(num_rows=num_rows)
            print(f"   Successfully generated {len(synthetic_data)} raw synthetic rows.")
        except Exception as sample_e:
            print(f"   Error during sampling: {sample_e}")
            import traceback
            traceback.print_exc()
            print("   Aborting due to sampling error.")
            return False

        # --- Post-processing Steps ---
        print("\n8. Applying post-processing (Faker, categorical distributions, functions)...") # Step number updated

        # Handle categorical columns with preserved probability distribution FIRST
        print("   Applying preserved categorical distributions...")
        for col_name, col_schema in enhanced_schema.items():
             if col_name not in synthetic_data.columns: continue # Skip if column not generated

             if col_schema.get("data_type") == "categorical":
                 # Check for explicit instruction to preserve distribution
                 preserve_dist = col_schema.get("preserve_distribution", False) # Add this flag to your schema if needed
                 categories_data = None
                 if "stats" in col_schema:
                     categories_data = col_schema["stats"].get("categories") or col_schema["stats"].get("top_categories")

                 # Preserve distribution only if flag is set AND category data is valid
                 if preserve_dist and isinstance(categories_data, list) and categories_data:
                     # Further validation inside generate_weighted_random_element
                     print(f"     - Preserving value distribution for categorical column '{col_name}'")
                     # Use .loc to avoid SettingWithCopyWarning if synthetic_data is a slice
                     synthetic_data.loc[:, col_name] = synthetic_data[col_name].apply(
                         lambda _: generate_weighted_random_element(categories_data)
                     )
                     # Mark as processed to avoid Faker override later
                     col_schema['_processed_categorical'] = True # Add a temporary flag
                 elif preserve_dist:
                      print(f"     - Warning: 'preserve_distribution' is True for '{col_name}' but valid 'categories' data not found in schema stats. Skipping distribution preservation.")


        # Apply Faker providers (unless categorical distribution was already applied)
        print("   Applying Faker providers...")
        for col_name, col_schema in enhanced_schema.items():
            if col_name not in synthetic_data.columns: continue
            if col_schema.get('_processed_categorical'): continue # Skip if handled above

            faker_provider = col_schema.get("faker_provider")
            faker_args = col_schema.get("faker_args", {})
            if faker_provider:
                print(f"     - Applying Faker provider '{faker_provider}' to column '{col_name}'")

                # Handle random_element specifically if needed (though Faker has it)
                if faker_provider == "random_element" and "elements" in faker_args:
                    elements = faker_args.get("elements", [])
                    if elements and isinstance(elements, list):
                         # Use .loc to avoid SettingWithCopyWarning if synthetic_data is a slice
                         synthetic_data.loc[:, col_name] = synthetic_data[col_name].apply(
                             lambda _: random.choice(elements)
                         )
                    else:
                         print(f"       Warning: 'elements' list is empty or invalid for random_element on '{col_name}'. Setting to None.")
                         synthetic_data.loc[:, col_name] = None
                    continue # Move to next column

                # General Faker provider handling
                try:
                    faker_method = getattr(fake, faker_provider)
                    # Use .loc to avoid SettingWithCopyWarning if synthetic_data is a slice
                    synthetic_data.loc[:, col_name] = synthetic_data[col_name].apply(
                        lambda _: faker_method(**faker_args)
                    )
                except AttributeError:
                    print(f"       Warning: Faker provider '{faker_provider}' not found. Keeping generated values for '{col_name}'.")
                except Exception as e:
                    print(f"       Error applying Faker provider '{faker_provider}' to '{col_name}': {str(e)}")


        print("\n9. Comparing correlations between original and synthetic data...") # Step number updated
        # Ensure synthetic data columns used for correlation are numeric
        try:
            # Select columns that are likely numeric based on original data's numeric columns
            original_numeric_cols = df.select_dtypes(include=np.number).columns
            potential_numeric_synthetic = synthetic_data[[col for col in original_numeric_cols if col in synthetic_data.columns]]

            # Convert selected columns to numeric, coercing errors
            numeric_synthetic = potential_numeric_synthetic.apply(pd.to_numeric, errors='coerce')

            # Drop columns that are all NaN after conversion/selection
            numeric_synthetic = numeric_synthetic.dropna(axis=1, how='all')

            if not numeric_synthetic.empty and len(numeric_synthetic.columns) >= 2:
                original_correlations = detect_column_correlations(df) # Use original numeric detection
                synthetic_correlations = detect_column_correlations(numeric_synthetic) # Use filtered synthetic numeric

                correlation_report = {}
                all_numeric_cols = set(original_correlations.keys()) | set(synthetic_correlations.keys())

                for col in all_numeric_cols:
                     correlation_report[col] = {
                         "original": original_correlations.get(col, "N/A"),
                         "synthetic": synthetic_correlations.get(col, "N/A")
                     }

                correlation_report_file = f"{os.path.splitext(output_file)[0]}_correlation_report.json"
                try:
                    with open(correlation_report_file, 'w', encoding='utf-8') as f:
                        json.dump(correlation_report, f, indent=2, ensure_ascii=False)
                    print(f"   Correlation comparison saved to {correlation_report_file}")
                except Exception as json_e:
                    print(f"   Error saving correlation report: {json_e}")
            else:
                print("   Skipping correlation comparison: Not enough valid numeric columns found in synthetic data for comparison.")
        except Exception as corr_e:
            print(f"   Error during correlation comparison step: {corr_e}")


        print(f"\n10. Saving final synthetic data to {output_file}...") # Step number updated
        synthetic_data.to_csv(output_file, index=False, encoding='utf-8') # Specify encoding
        print(f"   Successfully generated and saved {len(synthetic_data)} rows of synthetic data!")
        print("\n--- Synthetic Data Generation Finished ---")
        return True

    except Exception as e:
        print(f"\n--- Error during Synthetic Data Generation ---")
        print(f"An unexpected error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function to orchestrate the synthetic data generation process."""
    print("--- Starting Script ---")
    # --- File Checks ---
    if not os.path.exists(INPUT_CSV):
        print(f"Error: Input data file '{INPUT_CSV}' not found.")
        return
    if not os.path.exists(INPUT_SCHEMA_JSON):
        print(f"Error: Input schema file '{INPUT_SCHEMA_JSON}' not found.")
        return

    # --- Read Schema ---
    print(f"\nReading enhanced schema from: {INPUT_SCHEMA_JSON}")
    enhanced_schema = read_schema(INPUT_SCHEMA_JSON)
    if not enhanced_schema:
        print("Failed to read or parse the enhanced schema. Exiting.")
        return

    # --- Load Original Data ---
    print(f"\nLoading original data from: {INPUT_CSV}")
    try:
        # Try detecting encoding, fall back to utf-8 or latin-1 if needed
        try:
            original_df = pd.read_csv(INPUT_CSV, encoding='utf-8')
        except UnicodeDecodeError:
            print("UTF-8 decoding failed, trying latin-1...")
            original_df = pd.read_csv(INPUT_CSV, encoding='latin-1')

        print(f"Loaded original data: {original_df.shape[0]} rows, {original_df.shape[1]} columns")
        # print("Original data sample:\n", original_df.head()) # Optional: print head
        # print("\nOriginal data types:\n", original_df.dtypes) # Optional: print dtypes

        # --- Data Cleaning/Preparation (Optional but Recommended) ---
        # Example: Convert columns intended to be numeric if they aren't already
        # for col_name, col_schema in enhanced_schema.items():
        #     if col_name in original_df.columns:
        #         data_type = col_schema.get("data_type", "").lower()
        #         if data_type in ["numerical", "integer", "float"]:
        #              if not pd.api.types.is_numeric_dtype(original_df[col_name]):
        #                  print(f"Attempting to convert column '{col_name}' to numeric...")
        #                  original_df[col_name] = pd.to_numeric(original_df[col_name], errors='coerce')
        #                  # Handle NaNs introduced by coercion if necessary
        #                  # original_df[col_name].fillna(0, inplace=True) # Example: fill with 0
        #         elif data_type == "datetime":
        #              if not pd.api.types.is_datetime64_any_dtype(original_df[col_name]):
        #                  print(f"Attempting to convert column '{col_name}' to datetime...")
        #                  date_format = col_schema.get("datetime_format") # Get format from schema if specified
        #                  original_df[col_name] = pd.to_datetime(original_df[col_name], format=date_format, errors='coerce')


    except FileNotFoundError:
         print(f"Error: Input data file '{INPUT_CSV}' not found during loading.")
         return
    except Exception as e:
        print(f"Error loading or preparing original data from '{INPUT_CSV}': {str(e)}")
        return

    # --- Generate Synthetic Data ---
    print(f"\nInitiating synthetic data generation for {NUM_ROWS} rows...")
    success = generate_synthetic_data_with_sdv(original_df, enhanced_schema, NUM_ROWS, OUTPUT_CSV)

    # --- Conclusion ---
    print("\n--- Script Finished ---")
    if success:
        print(f"Synthetic data generation process completed successfully.")
        print(f"Output saved to: {OUTPUT_CSV}")
        report_file = f"{os.path.splitext(OUTPUT_CSV)[0]}_correlation_report.json"
        if os.path.exists(report_file):
             print(f"Correlation report saved to: {report_file}")
    else:
        print("Synthetic data generation process encountered errors.")

if __name__ == "__main__":
    main()
