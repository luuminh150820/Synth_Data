import pandas as pd
import json
import os
import numpy as np
import random
from faker import Faker
import inspect
import warnings

# Import necessary components from YData-Synthetic 2.0.0 paths identified by introspection
try:
    # Import ModelParameters and a GAN model directly from ydata_synthetic.synthesizers
    from ydata_synthetic.synthesizers import ModelParameters
    from ydata_synthetic.synthesizers import VanillaGAN # Using VanillaGAN as identified

    print("Successfully imported ModelParameters and VanillaGAN from ydata_synthetic.synthesizers")

except ImportError as e:
    print(f"\n--- Final Import Error ---")
    print(f"Error importing core YData-Synthetic components from ydata_synthetic.synthesizers: {e}")
    print("Based on introspection, these components should be here in version 2.0.0.")
    print("There might still be an issue with your ydata-synthetic 2.0.0 installation.")
    print("Please verify the installation or consider using a different version.")
    print("--------------------------")
    # Re-raise the error as we cannot proceed without these imports
    raise


# Configuration
INPUT_CSV = "customer_data.csv"  # Input CSV file
OUTPUT_CSV = "synthetic_data.csv"  # Output CSV file for synthetic data
INPUT_SCHEMA_JSON = "enhanced_schema.json"  # Input JSON file for the enhanced schema
NUM_ROWS = 1000  # Number of synthetic rows to generate
CORRELATION_THRESHOLD = 0.7  # Define the correlation threshold

# Initialize Faker
fake = Faker(['vi_VN'])
Faker.seed(42)

# --- Helper Functions (Kept from previous code) ---

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


        # random.choices handles normalization automatically if weights doesn't sum to 1
        return random.choices(values, weights=weights, k=1)[0]

    except (ValueError, TypeError) as e:
        print(f"Error during weighted choice: {e}. Values: {values}, Weights: {weights}")
        return None # Or handle error appropriately
    except Exception as e: # Catch any other potential errors from random.choices
         print(f"Unexpected error during weighted choice: {e}. Values: {values}, Weights: {weights}")
         return None


# --- YData-Synthetic Specific Functions ---

def prepare_ydata_schema_info(df, enhanced_schema):
    """
    Prepares column type information (categorical, continuous) and constraint
    definitions for YData-Synthetic based on the enhanced schema.
    """
    categorical_cols = []
    continuous_cols = []
    # Store constraints for post-processing
    constraints_for_postprocessing = {}

    # YData-Synthetic doesn't have a direct 'id' type or built-in primary key concept like SDV
    # Uniqueness and range constraints will be handled in post-processing.
    # We'll collect the constraint definitions here.

    for col_name, col_schema in enhanced_schema.items():
        if col_name not in df.columns:
            print(f"Warning: Column '{col_name}' from schema not found in DataFrame. Skipping YData schema prep.")
            continue

        data_type = col_schema.get("data_type", "").lower()

        if data_type == "categorical":
            categorical_cols.append(col_name)
        elif data_type in ["numerical", "integer", "float", "id", "datetime"]:
             # YData-Synthetic regular synthesizers treat numerical, integer, float, id, and datetime
             # as continuous for synthesis purposes. We handle specific formatting/uniqueness later.
            continuous_cols.append(col_name)
        # Add other types if needed, mapping to YData-Synthetic's understanding

        # Parse constraints for post-processing
        if "sdv_constraints" in col_schema and col_schema["sdv_constraints"]:
             constraints_for_postprocessing[col_name] = col_schema["sdv_constraints"]
             print(f"Parsed constraints for post-processing for column: '{col_name}'")
             # Note: We are just storing the raw constraints from the schema here.
             # The logic to interpret and apply them will be in post_process_synthetic_data.


    print(f"\nIdentified categorical columns for YData-Synthetic: {categorical_cols}")
    print(f"Identified continuous columns for YData-Synthetic: {continuous_cols}")
    print(f"Constraints collected for post-processing: {list(constraints_for_postprocessing.keys())}")

    return categorical_cols, continuous_cols, constraints_for_postprocessing


def enforce_constraints_post_processing(synthetic_df, enhanced_schema, constraints_for_postprocessing):
    """
    Applies constraints (uniqueness, range) to the synthetic data after generation.
    This is a post-processing step as YData-Synthetic's regular synthesizers
    may not enforce these directly during generation.
    """
    print("\n--- Enforcing Constraints Post-Processing ---")
    processed_df = synthetic_df.copy() # Work on a copy

    for col_name, constraints_list in constraints_for_postprocessing.items():
        if col_name not in processed_df.columns:
            print(f"Warning: Column '{col_name}' with constraints not found in synthetic data. Skipping constraint enforcement.")
            continue

        print(f"  Enforcing constraints for column: '{col_name}'")

        for constraint_def in constraints_list:
            try:
                constraint_type = None
                params = {}

                if isinstance(constraint_def, dict):
                    constraint_type = constraint_def.get("constraint_type", "").lower()
                    params = constraint_def
                elif isinstance(constraint_def, str):
                    constraint_str = constraint_def.strip().lower()
                    if constraint_str == "unique":
                        constraint_type = "unique"
                    elif constraint_str.startswith("range:") or constraint_str.startswith("scalar_range:"):
                        constraint_type = "scalar_range" # Map string range to scalar_range type
                        # Parse min/max from string for params
                        try:
                             parts_str = constraint_str.split(':', 1)[1]
                             parts = parts_str.split(',')
                             if len(parts) == 2:
                                 min_str = parts[0].strip().lower()
                                 max_str = parts[1].strip().lower()
                                 params["min_value"] = float(min_str) if min_str != 'none' and min_str else None
                                 params["max_value"] = float(max_str) if max_str != 'none' and max_str else None
                             else:
                                 print(f"    Warning: Skipping invalid range string format '{constraint_def}' for '{col_name}' during enforcement.")
                                 continue # Skip this specific constraint
                        except Exception as parse_e:
                             print(f"    Error parsing range string '{constraint_def}' for '{col_name}' during enforcement: {parse_e}. Skipping.")
                             continue # Skip this specific constraint


                # --- Enforce Unique Constraint ---
                if constraint_type == "unique":
                    # Note: Enforcing strict uniqueness post-processing can be tricky.
                    # If duplicates exist, we can't easily regenerate a statistically valid unique value.
                    # A common approach is to identify duplicates and potentially replace them
                    # with unique values generated separately (e.g., using Faker or random).
                    # This simple implementation just warns about duplicates.
                    # For true uniqueness, consider if the column should be the primary key
                    # and if the synthesis model can better handle it (less common for regular GANs).
                    if processed_df[col_name].duplicated().any():
                        num_duplicates = processed_df[col_name].duplicated().sum()
                        print(f"    Warning: Unique constraint violation in column '{col_name}'. Found {num_duplicates} duplicates.")
                        # Optional: Implement logic to replace duplicates here if needed.
                        # Example: Replace duplicates with Faker data or random unique values.
                        # This requires careful consideration to not break other correlations.
                        # For now, we just warn.


                # --- Enforce ScalarRange Constraint ---
                elif constraint_type == "scalar_range":
                    min_val = params.get("min_value")
                    max_val = params.get("max_value")
                    strict_low = params.get("strict_low", False)
                    strict_high = params.get("strict_high", False)

                    # Ensure column is numeric for range checks
                    processed_df[col_name] = pd.to_numeric(processed_df[col_name], errors='coerce')

                    if min_val is not None:
                        if strict_low:
                            violations = processed_df[processed_df[col_name] <= min_val]
                        else:
                            violations = processed_df[processed_df[col_name] < min_val]
                        if not violations.empty:
                            print(f"    Warning: ScalarRange constraint violation (min) in column '{col_name}'. Found {len(violations)} values below {'or equal to' if strict_low else 'less than'} {min_val}.")
                            # Simple enforcement: Clip values to the bound
                            if strict_low:
                                processed_df.loc[processed_df[col_name] <= min_val, col_name] = min_val + (1e-9 if processed_df[col_name].dtype != object else 'min_clipped') # Add small epsilon for strict
                            else:
                                processed_df.loc[processed_df[col_name] < min_val, col_name] = min_val


                    if max_val is not None:
                        if strict_high:
                            violations = processed_df[processed_df[col_name] >= max_val]
                        else:
                            violations = processed_df[processed_df[col_name] > max_val]
                        if not violations.empty:
                            print(f"    Warning: ScalarRange constraint violation (max) in column '{col_name}'. Found {len(violations)} values above {'or equal to' if strict_high else 'greater than'} {max_val}.")
                            # Simple enforcement: Clip values to the bound
                            if strict_high:
                                processed_df.loc[processed_df[col_name] >= max_val, col_name] = max_val - (1e-9 if processed_df[col_name].dtype != object else 'max_clipped') # Subtract small epsilon for strict
                            else:
                                processed_df.loc[processed_df[col_name] > max_val, col_name] = max_val

                # Handle other constraint types if needed
                # elif constraint_type == "another_type":
                #     # Enforcement logic for another constraint type
                #     pass

                else:
                    # Handle unknown constraint types encountered during enforcement
                    if isinstance(constraint_def, str) and constraint_def.strip().lower() == "range":
                         print(f"    Warning: Skipping unsupported plain 'Range' string constraint for '{col_name}' during enforcement.")
                    else:
                         print(f"    Warning: Skipping unknown or unsupported constraint type '{constraint_type}' for '{col_name}' during enforcement.")


            except Exception as enforce_e:
                print(f"  Error enforcing constraint '{constraint_def}' for column '{col_name}': {enforce_e}")
                import traceback
                traceback.print_exc()

    print("--- Constraint Enforcement Post-Processing Finished ---")
    return processed_df


def generate_synthetic_data_with_ydata(df, enhanced_schema, num_rows, output_file):
    """Generates synthetic data using YData-Synthetic."""
    try:
        print("\n--- Starting YData-Synthetic Data Generation ---")

        print("1. Preparing schema information for YData-Synthetic...")
        categorical_cols, continuous_cols, constraints_for_postprocessing = prepare_ydata_schema_info(df, enhanced_schema)

        # Define YData-Synthetic model parameters as a dictionary
        # These are example parameters for a GAN model. You might need to tune these.
        # Parameters often include batch size, learning rate, noise dimensions, layers, etc.
        # Refer to YData-Synthetic documentation or examples for version 2.0.0 for required parameters.
        print("\n2. Defining YData-Synthetic model parameters as a dictionary (using example GAN parameters)...")
        model_parameters_dict = {
            'batch_size': 500,
            'lr': 1e-4,
            'noise_dim': 32,
            'layers_dim': [128, 128]
            # Add other parameters required by the specific GAN model constructor in 2.0.0
            # For example: 'log_details': True, 'l2scale': 1e-6, etc.
            # Based on introspection, ModelParameters class is available, but we are using dict for now.
            # If ModelParameters import worked, we would use:
            # model_parameters = ModelParameters(...)
            # model_parameters_dict = model_parameters.__dict__ # Or a similar way to get dict
        }

        # Define the model type (e.g., 'GAN', 'CGAN', 'VAE')
        # For YData-Synthetic 2.0.0, we instantiate the model class directly.
        model_type = 'VanillaGAN' # Keep the type name for logging
        print(f"   Using model type: {model_type}")

        print("\n3. Initializing YData-Synthetic model...")
        # Initialize the specific GAN model directly, passing the parameters dictionary
        # The constructor arguments might vary slightly by model and version.
        # Common arguments are model_parameters (or the dict itself), log_details, and column lists.
        # We pass the dictionary using **model_parameters_dict
        # Based on introspection, VanillaGAN is available directly under synthesizers
        synthesizer = VanillaGAN(
            # Pass parameters from the dictionary
            # Note: The constructor might expect a ModelParameters object, not a dict.
            # If the ModelParameters import worked, we would pass: model_parameters=model_parameters
            # Since it doesn't, we try passing the dict directly. This might fail if the constructor is strict.
            model_parameters=model_parameters_dict, # Try passing the dict as model_parameters
            continuous_columns=continuous_cols,
            categorical_columns=categorical_cols
            # Add other required arguments like log_details=True if needed by the constructor
        )
        print(f"   {model_type} model initialized.")

        print("\n4. Training YData-Synthetic model...")
        # Train the model on the original data
        # The number of epochs is a key parameter for training time and quality.
        # You might need to adjust this (e.g., 100, 500, 1000+ epochs).
        epochs = 100 # Example: Start with 100 epochs
        print(f"   Training for {epochs} epochs...")
        # Training arguments might vary by model and version.
        # The train method likely expects epochs and potentially other arguments.
        synthesizer.train(df, epochs=epochs) # Use .train() method for models
        print("   Model training completed.")

        print(f"\n5. Generating {num_rows} rows of synthetic data...")
        # Generate synthetic data
        synthetic_data = synthesizer.sample(num_rows)
        print(f"   Successfully generated {len(synthetic_data)} raw synthetic rows.")

        # --- Post-processing Steps ---
        print("\n6. Applying post-processing (Constraints, Faker, custom functions)...")

        # Enforce constraints (Uniqueness, Range)
        synthetic_data = enforce_constraints_post_processing(synthetic_data, enhanced_schema, constraints_for_postprocessing)


        # Apply Faker providers (unless categorical distribution was already applied - handled in old code)
        print("   Applying Faker providers and preserved categorical distributions...")
        for col_name, col_schema in enhanced_schema.items():
            if col_name not in synthetic_data.columns: continue
            # Check for temporary flag set by preserved categorical distribution logic
            if col_schema.get('_processed_categorical'):
                 # Clean up the temporary flag
                 del col_schema['_processed_categorical']
                 continue # Skip if handled below

            faker_provider = col_schema.get("faker_provider")
            faker_args = col_schema.get("faker_args", {})
            if faker_provider:
                print(f"     - Applying Faker provider '{faker_provider}' to column '{col_name}'")

                # Handle random_element specifically if needed (though Faker has it)
                if faker_provider == "random_element" and "elements" in faker_args:
                    elements = faker_args.get("elements", [])
                    if elements and isinstance(elements, list):
                         # Use .loc for assignment
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
                    # Use .loc for assignment
                    synthetic_data.loc[:, col_name] = synthetic_data[col_name].apply(
                        lambda _: faker_method(**faker_args)
                    )
                except AttributeError:
                    print(f"       Warning: Faker provider '{faker_provider}' not found. Keeping generated values for '{col_name}'.")
                except Exception as e:
                    print(f"       Error applying Faker provider '{faker_provider}' to '{col_name}': {str(e)}")


        # Apply other post-processing functions
        print("   Applying custom post-processing functions...")
        for col_name, col_schema in enhanced_schema.items():
            if col_name not in synthetic_data.columns: continue

            post_processing = col_schema.get("post_processing")
            if post_processing:
                print(f"     - Applying post-processing '{post_processing}' to column '{col_name}'")
                try:
                    # Use .loc for assignment
                    if post_processing == "format_as_currency":
                        # Ensure the column is numeric before formatting
                        synthetic_data[col_name] = pd.to_numeric(synthetic_data[col_name], errors='coerce')
                        synthetic_data.loc[:, col_name] = synthetic_data[col_name].apply(
                            lambda x: f"{float(x):,.2f} VND" if pd.notna(x) else None # Handle NaN
                        )
                    elif post_processing == "ensure_valid_id":
                         # Attempt to extract digits, then format. Generate random if fails.
                         def format_id(x):
                             if pd.isna(x):
                                 return f"ID{random.randint(10000000, 99999999):08d}"
                             try:
                                 # Try converting directly first if it's already numeric
                                 if isinstance(x, (int, float)):
                                     return f"ID{int(abs(x)):08d}"
                                 # If string, extract digits
                                 digits = ''.join(filter(str.isdigit, str(x)))
                                 if digits:
                                     return f"ID{int(digits):08d}"
                                 else: # If no digits found in string
                                     return f"ID{random.randint(10000000, 99999999):08d}"
                             except (ValueError, TypeError): # Catch potential errors during conversion
                                 return f"ID{random.randint(10000000, 99999999):08d}"
                         synthetic_data.loc[:, col_name] = synthetic_data[col_name].apply(format_id)

                    elif post_processing == "format_percentage":
                        synthetic_data[col_name] = pd.to_numeric(synthetic_data[col_name], errors='coerce')
                        synthetic_data.loc[:, col_name] = synthetic_data[col_name].apply(
                            lambda x: f"{float(x):.2f}%" if pd.notna(x) else None
                        )
                    elif post_processing == "format_date":
                        # Ensure column is datetime before formatting
                        synthetic_data[col_name] = pd.to_datetime(synthetic_data[col_name], errors='coerce')
                        date_format = col_schema.get("datetime_format", "%Y-%m-%d") # Use format from schema if available
                        synthetic_data.loc[:, col_name] = synthetic_data[col_name].apply(
                            lambda x: x.strftime(date_format) if pd.notna(x) else None
                        )
                    # Add more custom functions here
                    # elif post_processing == "some_other_function":
                    #     synthetic_data.loc[:, col_name] = synthetic_data[col_name].apply(my_custom_func)

                except Exception as post_e:
                    print(f"       Error during post-processing '{post_processing}' for '{col_name}': {post_e}")


        # Apply functional dependencies manually (if needed, use with caution)
        # YData-Synthetic models learn dependencies, manual enforcement might conflict.
        apply_manual_dependencies = False # Set to True to enable manual override
        if apply_manual_dependencies:
            print("   Applying functional dependencies manually (use with caution)...")
            original_correlations = detect_column_correlations(df) # Re-detect on original df if needed
            for source_col, dependencies in original_correlations.items():
                 if source_col not in synthetic_data.columns: continue
                 for dep in dependencies:
                     if dep.get("type") == "functional_dependency":
                         target_col = dep["column"]
                         if target_col not in synthetic_data.columns: continue

                         print(f"     - Applying functional dependency: {source_col} -> {target_col}")
                         # Create mapping from original data (handle NaN and potential duplicates)
                         # Keep the first occurrence in case of duplicates in source_col
                         mapping = df.dropna(subset=[source_col, target_col]).drop_duplicates(subset=[source_col], keep='first').set_index(source_col)[target_col].to_dict()

                         if mapping:
                             # Apply mapping to synthetic data, handle missing keys
                             synthetic_data.loc[:, target_col] = synthetic_data[source_col].map(mapping)
                             # Optionally, fill NaNs created by mapping (e.g., with mode or another strategy)
                             missing_count = synthetic_data[target_col].isna().sum()
                             if missing_count > 0:
                                 print(f"       {missing_count} values in '{target_col}' became NaN after mapping '{source_col}' -> '{target_col}'. Consider a fill strategy if needed.")
                                 # Example: Fill with mode from original target column
                                 # try:
                                 #     target_mode = df[target_col].mode()[0]
                                 #     synthetic_data[target_col].fillna(target_mode, inplace=True)
                                 # except IndexError: # Handle empty mode case
                                 #      print(f"       Could not determine mode for '{target_col}' to fill NaNs.")


                         else:
                             print(f"       Warning: Could not create mapping for dependency {source_col} -> {target_col}.")


        print("\n7. Comparing correlations between original and synthetic data...")
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


        print(f"\n8. Saving final synthetic data to {output_file}...")
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
    success = generate_synthetic_data_with_ydata(original_df, enhanced_schema, NUM_ROWS, OUTPUT_CSV)

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
