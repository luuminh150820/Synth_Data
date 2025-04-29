import pandas as pd
import json
import os
import numpy as np
from collections import defaultdict
import re
from datetime import datetime
import math 
from itertools import combinations, product 
import inspect 
import google.generativeai as genai 
import warnings 
import time 

# Configuration
INPUT_CSV = "customer_data.csv" 
OUTPUT_SCHEMA_JSON = "enhanced_schema.json" 
METADATA_JSON = "metadata.json" 
TEMP_RAW_RESPONSE_FILE = "temp_raw_gemini_response_v4.txt" 

# Thresholds for relationship detection
MIN_UNIQUE_VALUES_FOR_RELATIONSHIP_SOURCE = 15 # Minimum number of unique non-null values for a column to be considered a relationship source
MIN_UNIQUE_VALUE_RATIO_FOR_RELATIONSHIP_SOURCE = 0.05 # Minimum ratio of unique non-null values to total non-null values
RELATIONSHIP_DETECTION_THRESHOLD = 0.999 # Threshold for functional dependency and value relationship consistency

# Thresholds for type detection
DATETIME_CONVERSION_THRESHOLD = 0.80 # Minimum proportion of non-null values that must convert to datetime to be flagged as datetime
CATEGORICAL_UNIQUE_RATIO_THRESHOLD = 0.2 # Maximum ratio of unique values for a column to be considered categorical
CATEGORICAL_MAX_UNIQUE_VALUES = 20 # Maximum number of unique values for a column to be considered categorical

# API Call Delay
API_CALL_DELAY_SECONDS = 3 

# --- Gemini API Setup ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
gemini_model = None 

def load_data(csv_file_path):
    """Loads data from a CSV file."""
    if not os.path.exists(csv_file_path):
        print(f"Error: Input data file not found: {csv_file_path}")
        return None, None
    try:
        # Attempt to read with UTF-8, fallback to latin-1
        # Keep strings as objects initially to detect leading zeros etc.
        try:
            df = pd.read_csv(csv_file_path, encoding='utf-8', low_memory=False, dtype=str) # Read all as string initially
        except UnicodeDecodeError:
            print("UTF-8 decoding failed, trying latin-1...")
            df = pd.read_csv(csv_file_path, encoding='latin-1', low_memory=False, dtype=str) # Read all as string initially

        print(f"Successfully loaded data from {csv_file_path} as strings. Shape: {df.shape}")
        # Store original dtypes (will be 'object' now)
        original_dtypes = df.dtypes.to_dict()
        # Attempt to infer better types after loading as string
        df = df.infer_objects()
        print(f"Inferred object types. Example dtypes now: {df.dtypes.head()}")
        return df, original_dtypes # Return df with inferred types and original (string) dtypes
    except Exception as e:
        print(f"Error loading data from {csv_file_path}: {e}")
        return None, None

def load_external_metadata(json_file_path):
    """ key_type, description, ...."""
    if not os.path.exists(json_file_path):
        print(f"Error: Metadata file not found at {json_file_path}. Metadata is required.")
        return None
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        print(f"Successfully loaded external metadata from {json_file_path}.")

        column_metadata = {}
        table_key = next((key for key, value in metadata.items() if isinstance(value, dict) and "columns" in value), None)

        if table_key and "columns" in metadata[table_key]:
            for col_info in metadata[table_key].get("columns", []):
                 if isinstance(col_info, dict) and "Column_name" in col_info:
                      col_name_upper = col_info["Column_name"].upper()
                      column_metadata[col_name_upper] = col_info
            print(f"Extracted metadata for {len(column_metadata)} columns from table '{table_key}'.")
            return column_metadata
        else:
             print(f"Error: Could not find a valid table structure with 'columns' in {json_file_path}.")
             return None

    except Exception as e:
        print(f"Error loading or parsing external metadata from {json_file_path}: {e}")
        return None

def check_leading_zeros(series):
    """Checks a series for values that look numeric but have leading zeros."""
    if not pd.api.types.is_object_dtype(series.dtype) and not pd.api.types.is_string_dtype(series.dtype):
        return False # Only check string/object columns

    had_leading_zero = False
    for value in series.dropna():
        if isinstance(value, str):
            stripped_value = value.strip()
            # Ignore if empty string, purely "0", or starts with "0." (decimal)
            if not stripped_value or stripped_value == '0' or stripped_value.startswith('0.'):
                continue
            # Check if it looks like an integer and starts with '0'
            if stripped_value.isdigit() and stripped_value.startswith('0'):
                 # Further check: ensure it's not octal representation if needed (simple check here)
                 try:
                      # If int conversion works and string representation differs, it likely had leading zero
                      if str(int(stripped_value)) != stripped_value:
                           had_leading_zero = True
                           break # Found one, no need to check further
                 except ValueError:
                      continue # Not a simple integer string

    return had_leading_zero


def calculate_basic_stats(df, original_dtypes):
    """
    Calculates basic statistics for each column.
    Includes is_unique and had_leading_zero flags.
    """
    print("\nCalculating basic statistics...")
    stats = {}
    df_columns_upper = df.columns.tolist()
    original_case_map_from_dtypes = {name.upper(): name for name in original_dtypes.keys()}

    for col_upper in df_columns_upper:
        col_original_case_dtype_key = original_case_map_from_dtypes.get(col_upper, col_upper) # Key for original_dtypes dict
        col_series = df[col_upper] # Access df using uppercase name
        original_dtype_str = str(original_dtypes.get(col_original_case_dtype_key, 'unknown')) # Get original dtype string

        col_stats = {}
        col_stats["original_dtype"] = original_dtype_str
        col_stats["inferred_dtype"] = str(col_series.dtype) # Dtype after infer_objects()
        col_stats["null_count"] = col_series.isnull().sum()
        col_stats["total_count"] = len(df)
        col_stats["null_percentage"] = (col_stats["null_count"] / col_stats["total_count"]) * 100 if col_stats["total_count"] > 0 else 0
        col_stats["unique_count"] = col_series.nunique()

        non_null_series = col_series.dropna()
        num_non_null = len(non_null_series)

        # --- is_unique Flag ---
        col_stats["is_unique"] = (num_non_null > 0) and (col_stats["unique_count"] == num_non_null)

        col_stats["is_numeric"] = False
        col_stats["is_datetime"] = False
        col_stats["is_categorical"] = False
        col_stats["had_leading_zero"] = False # Initialize flag

        if num_non_null > 0:
            # --- Sample Values ---
            sample_size = min(20, num_non_null)
            col_stats["sample_values"] = non_null_series.sample(sample_size).tolist()

            # --- Leading Zero Check ---
            col_stats["had_leading_zero"] = check_leading_zeros(non_null_series)

            # --- Numeric Stats ---
            if not col_stats["had_leading_zero"] and \
               (pd.api.types.is_numeric_dtype(col_series.dtype) or pd.api.types.is_object_dtype(col_series.dtype)):
                # Use the original series for conversion attempt
                numeric_series = pd.to_numeric(non_null_series, errors='coerce').dropna()
                if not numeric_series.empty:
                     # Check if a significant portion could be converted if original was object
                     if pd.api.types.is_object_dtype(col_series.dtype) and len(numeric_series) / num_non_null < 0.5:
                          pass # Not primarily numeric
                     else:
                          col_stats["min"] = numeric_series.min()
                          col_stats["max"] = numeric_series.max()
                          col_stats["mean"] = numeric_series.mean()
                          col_stats["median"] = numeric_series.median()
                          col_stats["std_dev"] = numeric_series.std()
                          col_stats["is_numeric"] = True
            if not col_stats["is_numeric"]:
                 col_stats["min"] = None
                 col_stats["max"] = None
                 col_stats["mean"] = None
                 col_stats["median"] = None
                 col_stats["std_dev"] = None

            # --- Datetime Stats (Stricter Check) ---
            if not col_stats["is_numeric"] and \
               (pd.api.types.is_object_dtype(col_series.dtype) or pd.api.types.is_datetime64_any_dtype(col_series.dtype)):
                datetime_series = None
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", UserWarning)
                        temp_datetime_series = pd.to_datetime(non_null_series, errors='coerce')
                    num_converted = temp_datetime_series.notna().sum()
                    if num_non_null > 0 and (num_converted / num_non_null) >= DATETIME_CONVERSION_THRESHOLD:
                        datetime_series = temp_datetime_series.dropna()
                        if not datetime_series.empty:
                            col_stats["min_date"] = datetime_series.min().date().isoformat()
                            col_stats["max_date"] = datetime_series.max().date().isoformat()
                            col_stats["is_datetime"] = True
                except Exception: pass
            if not col_stats["is_datetime"]:
                col_stats["min_date"] = None
                col_stats["max_date"] = None

            # --- Categorical Stats (Refined Check) ---
            # Consider categorical if not numeric/datetime, specific dtypes, and meets unique thresholds
            unique_ratio = col_stats["unique_count"] / num_non_null if num_non_null > 0 else 0
            is_likely_type_for_cat = pd.api.types.is_object_dtype(col_series.dtype) or \
                                     pd.api.types.is_categorical_dtype(col_series.dtype) or \
                                     pd.api.types.is_bool_dtype(col_series.dtype)

            if not col_stats["is_numeric"] and not col_stats["is_datetime"] and is_likely_type_for_cat and \
               (unique_ratio < CATEGORICAL_UNIQUE_RATIO_THRESHOLD or col_stats["unique_count"] < CATEGORICAL_MAX_UNIQUE_VALUES) and \
               col_stats["unique_count"] > 0:
                 col_stats["is_categorical"] = True
                 value_counts = non_null_series.value_counts(normalize=True)
                 top_categories = value_counts.head(20)
                 col_stats["categories"] = [{"value": str(index), "percentage": round(value * 100, 2)} for index, value in top_categories.items()]
            else:
                 col_stats["categories"] = []

        # Store stats using uppercase column name as key
        stats[col_upper] = col_stats
        print(f"  '{col_upper}': Null%={col_stats['null_percentage']:.1f}, UniqueN={col_stats['unique_count']}, "
              f"IsUnique={col_stats['is_unique']}, LeadZero={col_stats['had_leading_zero']}, "
              f"Num={col_stats['is_numeric']}, Date={col_stats['is_datetime']}, Cat={col_stats['is_categorical']}")
    print("Basic statistics calculation complete.")
    return stats


def detect_key_types(df_columns, external_metadata):
    """Detects primary and foreign keys based *only* on the external metadata file."""
    print("\nDetecting key types from external metadata...")
    key_types = {}
    if not external_metadata:
        print("  Warning: External metadata not provided or empty. Cannot determine key types.")
        for col_upper in df_columns: key_types[col_upper] = "None"
        return key_types

    found_pk = False
    for col_upper in df_columns:
        col_meta = external_metadata.get(col_upper)
        if col_meta and "Key_type" in col_meta:
            key_type_str = str(col_meta["Key_type"]).strip().lower()
            if key_type_str == "primary key": key_types[col_upper] = "Primary Key"; print(f"  '{col_upper}': Primary Key (from metadata)"); found_pk = True
            elif key_type_str == "foreign key": key_types[col_upper] = "Foreign Key"; print(f"  '{col_upper}': Foreign Key (from metadata)")
            elif key_type_str in ["null", "none", ""]: key_types[col_upper] = "None"
            else: print(f"  Warning: Unknown Key_type '{col_meta['Key_type']}' for column '{col_upper}' in metadata. Treating as 'None'."); key_types[col_upper] = "None"
        else:
            key_types[col_upper] = "None"
            if col_upper not in external_metadata: print(f"  Warning: Column '{col_upper}' found in data but not in metadata. Key type set to 'None'.")
            elif col_meta and "Key_type" not in col_meta: print(f"  Warning: Metadata for column '{col_upper}' lacks 'Key_type'. Key type set to 'None'.")
            elif not col_meta: print(f"  Warning: Column '{col_upper}' found in data but not in metadata dict. Key type set to 'None'.")

    for col_upper in df_columns:
        if col_upper not in key_types: key_types[col_upper] = "None"
    if not found_pk: print("  Warning: No Primary Key explicitly defined in the provided metadata for the loaded columns.")
    print("Key type detection from metadata complete.")
    return key_types

def infer_column_info_with_llm(col_name_upper, stats, key_type, external_metadata_info=None):
    """Infers data type, Faker provider, and args using LLM with descriptive stats."""
    print(f"  Inferring info for column '{col_name_upper}' using LLM...")
    global gemini_model
    if gemini_model is None:
        print(f"  Skipping LLM inference for '{col_name_upper}': Gemini model not configured.")
        return {"data_type": "unknown", "faker_provider": None, "faker_args": {}, "data_domain": None, "constraints_description": None}

    # --- Clean stats for JSON ---
    cleaned_stats = {}
    for key, value in stats.items():
        if isinstance(value, (np.int64, np.int32)): cleaned_stats[key] = int(value)
        elif isinstance(value, (np.float64, np.float32)): cleaned_stats[key] = float(value) if not (np.isnan(value) or np.isinf(value)) else None
        elif isinstance(value, pd.Timestamp): cleaned_stats[key] = value.isoformat()
        elif isinstance(value, (np.bool_, bool)): cleaned_stats[key] = bool(value)
        elif isinstance(value, dict):
             cleaned_nested_dict = {}
             for k, v in value.items():
                  if isinstance(v, (np.int64, np.int32)): cleaned_nested_dict[k] = int(v)
                  elif isinstance(v, (np.float64, np.float32)): cleaned_nested_dict[k] = float(v) if not (np.isnan(v) or np.isinf(v)) else None
                  elif isinstance(v, pd.Timestamp): cleaned_nested_dict[k] = v.isoformat()
                  elif isinstance(v, (np.bool_, bool)): cleaned_nested_dict[k] = bool(v)
                  else: cleaned_nested_dict[k] = v
             cleaned_stats[key] = cleaned_nested_dict
        elif isinstance(value, list):
             cleaned_list = []
             for item in value:
                  if isinstance(item, (np.int64, np.int32)): cleaned_list.append(int(item))
                  elif isinstance(item, (np.float64, np.float32)): cleaned_list.append(float(item) if not (np.isnan(item) or np.isinf(item)) else None)
                  elif isinstance(item, pd.Timestamp): cleaned_list.append(item.isoformat())
                  elif isinstance(item, (np.bool_, bool)): cleaned_list.append(bool(item))
                  elif isinstance(item, dict):
                      cleaned_nested_dict_in_list = {}
                      for k_list, v_list in item.items():
                           if isinstance(v_list, (np.int64, np.int32)): cleaned_nested_dict_in_list[k_list] = int(v_list)
                           elif isinstance(v_list, (np.float64, np.float32)): cleaned_nested_dict_in_list[k_list] = float(v_list) if not (np.isnan(v_list) or np.isinf(v_list)) else None
                           elif isinstance(v_list, pd.Timestamp): cleaned_nested_dict_in_list[k_list] = v_list.isoformat()
                           elif isinstance(v_list, (np.bool_, bool)): cleaned_nested_dict_in_list[k_list] = bool(v_list)
                           else: cleaned_nested_dict_in_list[k_list] = v_list
                      cleaned_list.append(cleaned_nested_dict_in_list)
                  else: cleaned_list.append(str(item)) # Ensure samples are strings
             cleaned_stats[key] = cleaned_list
        else: cleaned_stats[key] = value
    # --- End Cleaning ---

    # Construct the prompt with descriptions
    prompt = f"""
Analyze the following information about a column named '{col_name_upper}' from a dataset.
Based on the column name, key type, statistics, sample values, and potentially external metadata, suggest:
1.  The most appropriate data type ('integer', 'float', 'numerical', 'datetime', 'boolean', 'categorical', 'text', 'unknown').
2.  A suitable Faker provider name (e.g., 'name', 'random_int', 'date', 'unique.random_int', 'regexify', 'company'). If spotted a simple pattern from the samples provided, suggest 'regexify'.
    *IMPORTANT*: If the Key Type is 'Primary Key':
        - If the data appears numeric (based on stats like min/max/mean), strongly prefer 'unique.random_int' or similar numeric unique provider. Define appropriate 'min'/'max' in faker_args.
        - If the data appears to be strings with a pattern (check 'had_leading_zero' flag and samples), suggest 'unique.regexify' with a pattern in faker_args.
        - Otherwise, suggest a suitable unique string provider like 'unique.pystr' or let Faker handle uniqueness via the generation script if a standard provider is chosen. Avoid 'uuid4'.
3.  Appropriate arguments (`faker_args`) for the suggested Faker provider, in a JSON object format (e.g., {{"min": 0, "max": 100}}, {{"elements": ["A", "B"]}}, {{"start_date": "{cleaned_stats.get('min_date', 'YYYY-MM-DD')}", "end_date": "{cleaned_stats.get('max_date', 'YYYY-MM-DD')}"}}, don't add time if the samples don't have time, {{"pattern": "regex_pattern"}}, provide a pattern that satisfy the majority, or empty {{}} if no args}}). Try to use min/max/date stats to inform args.
4.  (Optional) A general data domain category (e.g., 'Customer Info', 'Financial Transaction', 'Product Details', 'Date/Time', 'Identifier', 'Address', 'Geography', 'Internal ID'). Respond with null if unsure.
5.  (Optional) A brief, human-readable description of any constraints implied by the data (e.g., "Must be unique", "Date must be after 2020", "References another table"). Respond with null if none obvious.

Column Name: {col_name_upper}
Key Type: {key_type}
External Metadata: {json.dumps(external_metadata_info, indent=2, default=str)}
Statistics: {{
    "original_dtype": "{cleaned_stats.get('original_dtype', 'unknown')}", // Original pandas dtype when loaded
    "inferred_dtype": "{cleaned_stats.get('inferred_dtype', 'unknown')}", // Pandas dtype after infer_objects()
    "null_count": {cleaned_stats.get('null_count', 0)}, // Number of null values
    "total_count": {cleaned_stats.get('total_count', 0)}, // Total number of rows
    "null_percentage": {cleaned_stats.get('null_percentage', 0):.2f}, // Percentage of null values
    "unique_count": {cleaned_stats.get('unique_count', 0)}, // Number of unique non-null values
    "is_unique": {cleaned_stats.get('is_unique', False)}, // True if all non-null values are unique, faker provider should not be unique if this is False
    "is_numeric": {cleaned_stats.get('is_numeric', False)}, // True if column is likely numeric
    "is_datetime": {cleaned_stats.get('is_datetime', False)}, // True if column is likely datetime
    "is_categorical": {cleaned_stats.get('is_categorical', False)}, // True if column is likely categorical
    "had_leading_zero": {cleaned_stats.get('had_leading_zero', False)}, // True if numeric-like strings had leading zeros (e.g., "007")
    "min": {json.dumps(cleaned_stats.get('min'))}, // Minimum numeric value (if numeric)
    "max": {json.dumps(cleaned_stats.get('max'))}, // Maximum numeric value (if numeric)
    "mean": {json.dumps(cleaned_stats.get('mean'))}, // Mean numeric value (if numeric)
    "median": {json.dumps(cleaned_stats.get('median'))}, // Median numeric value (if numeric)
    "std_dev": {json.dumps(cleaned_stats.get('std_dev'))}, // Standard deviation (if numeric)
    "min_date": "{cleaned_stats.get('min_date')}", // Minimum date (if datetime)
    "max_date": "{cleaned_stats.get('max_date')}", // Maximum date (if datetime)
    "categories": {json.dumps(cleaned_stats.get('categories', []), default=str)}, // Top categories and percentages (if categorical)
    "sample_values": {json.dumps(cleaned_stats.get('sample_values', []), default=str)} // Random sample of non-null values
}}

Provide the response as a single JSON object with the keys 'data_type', 'faker_provider', 'faker_args', 'data_domain', 'constraints_description'.
Respond ONLY with the JSON object, no conversational text or explanations.
"""
    # print(f"--- LLM Prompt for {col_name_upper} ---\n{prompt}\n------------------------") # Debug prompts

    default_response = {"data_type": "unknown", "faker_provider": None, "faker_args": {}, "data_domain": None, "constraints_description": None}
    try:
        response = gemini_model.generate_content(prompt)
        llm_response_text = response.text
        try:
            with open(TEMP_RAW_RESPONSE_FILE, 'a', encoding='utf-8') as f:
                 f.write(f"\n--- Response for Column: {col_name_upper} ---\n{llm_response_text}\n-------------------------------------\n")
        except Exception as e: print(f"  Error saving raw response for '{col_name_upper}' to file: {e}")

        json_match = re.search(r'```json\s*(\{.*?\})\s*```|\{.*?\}', llm_response_text, re.DOTALL | re.IGNORECASE)
        if json_match:
            json_string = json_match.group(1) if json_match.group(1) else json_match.group(0)
            try:
                llm_response = json.loads(json_string)
                if isinstance(llm_response, dict) and 'data_type' in llm_response and 'faker_provider' in llm_response and 'faker_args' in llm_response:
                    llm_response.setdefault('data_domain', None)
                    llm_response.setdefault('constraints_description', None)
                    if key_type == "Primary Key" and isinstance(llm_response.get('faker_provider'), str) and not llm_response['faker_provider'].startswith('unique.'):
                           base_provider = llm_response['faker_provider']
                           if base_provider not in ['profile', 'json', 'file_path', 'image_url', 'password']:
                                print(f"  Note: Wrapping suggested provider '{base_provider}' with 'unique.' for Primary Key '{col_name_upper}'.")
                                llm_response['faker_provider'] = f"unique.{base_provider}"
                    print(f"  LLM inferred: Type='{llm_response['data_type']}', Provider='{llm_response['faker_provider']}', Args={llm_response['faker_args']}, Domain='{llm_response['data_domain']}'")
                    return llm_response
                else: print(f"  Warning: LLM returned valid JSON but incomplete structure for '{col_name_upper}'. Falling back.")
            except json.JSONDecodeError as json_err: print(f"  Warning: Could not decode JSON from LLM response for '{col_name_upper}'. Error: {json_err}. Falling back.")
        else: print(f"  Warning: Could not find JSON object pattern in LLM response for '{col_name_upper}'. Falling back.")
    except Exception as e: print(f"  Error calling LLM for inference on '{col_name_upper}': {e}. Falling back.")
    return default_response


def detect_column_correlations(df, stats, key_types):
    """Detects column relationships (FD, O2O, Value, Temporal)."""
    print("\nDetecting column relationships (FD, O2O, Value, Temporal)...")
    relationships = defaultdict(list)
    all_cols_upper = df.columns.tolist()
    numeric_cols_upper = [col for col in all_cols_upper if stats.get(col, {}).get("is_numeric")]
    datetime_cols_upper = [col for col in all_cols_upper if stats.get(col, {}).get("is_datetime")]
    print(f"  Identified {len(numeric_cols_upper)} numeric columns and {len(datetime_cols_upper)} datetime columns.")

    def meets_uniqueness_criteria(col_name_upper, stats):
        col_stats = stats.get(col_name_upper, {})
        unique_count = col_stats.get("unique_count", 0)
        total_non_null = col_stats.get("total_count", 0) - col_stats.get("null_count", 0)
        if total_non_null == 0: return False
        unique_ratio = unique_count / total_non_null if total_non_null > 0 else 0
        return unique_count >= MIN_UNIQUE_VALUES_FOR_RELATIONSHIP_SOURCE and unique_ratio >= MIN_UNIQUE_VALUE_RATIO_FOR_RELATIONSHIP_SOURCE

    print(f"  Applying uniqueness thresholds for sources: Min Unique Values = {MIN_UNIQUE_VALUES_FOR_RELATIONSHIP_SOURCE}, Min Unique Ratio = {MIN_UNIQUE_VALUE_RATIO_FOR_RELATIONSHIP_SOURCE}")

    # --- Functional Dependency (A -> B) ---
    print("  Checking for Functional Dependencies...")
    functional_dependencies_detected = defaultdict(list)
    for col1_upper in all_cols_upper:
        if not meets_uniqueness_criteria(col1_upper, stats): continue
        for col2_upper in all_cols_upper:
            if col1_upper == col2_upper: continue
            try:
                df_filtered = df[[col1_upper, col2_upper]].dropna()
                if df_filtered.empty or len(df_filtered) < MIN_UNIQUE_VALUES_FOR_RELATIONSHIP_SOURCE: continue
                hashable_col1 = False; col1_for_groupby = None
                try:
                    if pd.api.types.is_object_dtype(df_filtered[col1_upper].dtype):
                        temp_col1 = df_filtered[col1_upper].astype(str)
                        if not temp_col1.empty and pd.api.types.is_hashable(temp_col1.iloc[0]): hashable_col1 = True; col1_for_groupby = temp_col1
                    elif not df_filtered[col1_upper].empty and pd.api.types.is_hashable(df_filtered[col1_upper].iloc[0]): hashable_col1 = True; col1_for_groupby = df_filtered[col1_upper]
                except Exception: pass
                if hashable_col1 and col1_for_groupby is not None:
                    unique_values_per_group = df_filtered.groupby(col1_for_groupby)[col2_upper].nunique()
                    consistent_groups_ratio = (unique_values_per_group <= 1).sum() / len(unique_values_per_group) if len(unique_values_per_group) > 0 else 1.0
                    if consistent_groups_ratio >= RELATIONSHIP_DETECTION_THRESHOLD: functional_dependencies_detected[col1_upper].append(col2_upper)
            except Exception: pass

    # --- Prioritize One-to-One ---
    print("  Prioritizing One-to-One relationships...")
    one_to_one_pairs = set()
    for col1_upper, targets in functional_dependencies_detected.items():
        for col2_upper in targets:
            if col2_upper in functional_dependencies_detected and col1_upper in functional_dependencies_detected[col2_upper] and meets_uniqueness_criteria(col2_upper, stats):
                pair = tuple(sorted((col1_upper, col2_upper)))
                if pair not in one_to_one_pairs:
                    relationships[pair[0]].append({"column": pair[1], "type": "one_to_one"})
                    relationships[pair[1]].append({"column": pair[0], "type": "one_to_one"})
                    one_to_one_pairs.add(pair); print(f"    Detected One-to-One: {pair[0]} <-> {pair[1]}")
    # Add remaining FDs
    for col1_upper, targets in functional_dependencies_detected.items():
        for col2_upper in targets:
            pair = tuple(sorted((col1_upper, col2_upper)))
            if pair not in one_to_one_pairs:
                 if not any(r.get("column") == col2_upper and r.get("type") == "functional_dependency" for r in relationships[col1_upper]):
                      relationships[col1_upper].append({"column": col2_upper, "type": "functional_dependency"}); print(f"    Detected Functional Dependency: {col1_upper} -> {col2_upper}")

    # --- Value Relationships (Numeric) ---
    print("  Checking for Numeric Value Relationships (>, <, =)...")
    if len(numeric_cols_upper) >= 2:
         for col1_upper, col2_upper in product(numeric_cols_upper, repeat=2):
             if col1_upper == col2_upper or not meets_uniqueness_criteria(col1_upper, stats): continue
             try:
                 df_filtered = df[[col1_upper, col2_upper]].dropna()
                 if not df_filtered.empty:
                     series1 = pd.to_numeric(df_filtered[col1_upper], errors='coerce'); series2 = pd.to_numeric(df_filtered[col2_upper], errors='coerce')
                     valid_comparison = pd.concat([series1, series2], axis=1).dropna(); num_valid_rows = len(valid_comparison)
                     if num_valid_rows > 0:
                          s1, s2 = valid_comparison[col1_upper], valid_comparison[col2_upper]
                          if (s1 > s2).sum() / num_valid_rows >= RELATIONSHIP_DETECTION_THRESHOLD: 
                              rel = {"column": col2_upper, "relationship": "greater_than", "type": "value_relationship"};
                              if rel not in relationships[col1_upper]: relationships[col1_upper].append(rel); print(f"    Detected Value Relationship: {col1_upper} > {col2_upper}")
                          if (s1 < s2).sum() / num_valid_rows >= RELATIONSHIP_DETECTION_THRESHOLD: 
                              rel = {"column": col2_upper, "relationship": "less_than", "type": "value_relationship"}; 
                              if rel not in relationships[col1_upper]: 
                                   relationships[col1_upper].append(rel); print(f"    Detected Value Relationship: {col1_upper} < {col2_upper}")
                          if np.isclose(s1, s2).sum() / num_valid_rows >= RELATIONSHIP_DETECTION_THRESHOLD:
                              rel = {"column": col2_upper, "relationship": "equal_to", "type": "value_relationship"}
                              if rel not in relationships[col1_upper]:
                                   relationships[col1_upper].append(rel)
                                   print(f"    Detected Value Relationship: {col1_upper} == {col2_upper}")
             except Exception: pass

    # --- Temporal Relationships ---
    print("  Checking for Temporal Relationships (>, <, =)...")
    if len(datetime_cols_upper) >= 2:
         for col1_upper, col2_upper in product(datetime_cols_upper, repeat=2):
             if col1_upper == col2_upper or not meets_uniqueness_criteria(col1_upper, stats): continue
             try:
                 df_filtered = df[[col1_upper, col2_upper]].dropna()
                 if not df_filtered.empty:
                     series1 = pd.to_datetime(df_filtered[col1_upper], errors='coerce'); series2 = pd.to_datetime(df_filtered[col2_upper], errors='coerce')
                     valid_comparison = pd.concat([series1, series2], axis=1).dropna(); num_valid_rows = len(valid_comparison)
                     if num_valid_rows > 0:
                          s1, s2 = valid_comparison[col1_upper], valid_comparison[col2_upper]
                          if (s1 > s2).sum() / num_valid_rows >= RELATIONSHIP_DETECTION_THRESHOLD: 
                              rel = {"column": col2_upper, "relationship": "greater_than", "type": "temporal_relationship"}; 
                              if rel not in relationships[col1_upper]: relationships[col1_upper].append(rel); print(f"    Detected Temporal Relationship: {col1_upper} > {col2_upper}")
                          if (s1 < s2).sum() / num_valid_rows >= RELATIONSHIP_DETECTION_THRESHOLD: 
                              rel = {"column": col2_upper, "relationship": "less_than", "type": "temporal_relationship"}; 
                              if rel not in relationships[col1_upper]: relationships[col1_upper].append(rel); print(f"    Detected Temporal Relationship: {col1_upper} < {col2_upper}")
                          if (s1 == s2).sum() / num_valid_rows >= RELATIONSHIP_DETECTION_THRESHOLD: 
                              rel = {"column": col2_upper, "relationship": "equal_to", "type": "temporal_relationship"}; 
                              if rel not in relationships[col1_upper]: relationships[col1_upper].append(rel); print(f"    Detected Temporal Relationship: {col1_upper} == {col2_upper}")
             except Exception: pass

    print("Column relationship detection complete.")
    return dict(relationships)


def generate_enhanced_schema(df_columns, key_types, stats, relationships, column_inferences, external_metadata=None):
    """Generates the enhanced schema dictionary."""
    print("\nGenerating enhanced schema dictionary...")
    enhanced_schema = {
        "metadata": {
            "generated_at": datetime.now().isoformat(), "original_file": INPUT_CSV,
            "row_count": stats[df_columns[0]]["total_count"] if df_columns and df_columns[0] in stats else 0,
            "column_count": len(df_columns)
        },
        "columns": {}, "relationships_summary": relationships
    }
    for col_upper in df_columns:
        inferred = column_inferences.get(col_upper, {"data_type": "unknown", "faker_provider": None, "faker_args": {}, "data_domain": None, "constraints_description": None})
        col_stats = stats.get(col_upper, {})
        meta_info = external_metadata.get(col_upper, {}) if external_metadata else {}
        enhanced_schema["columns"][col_upper] = {
            "description": meta_info.get("Description", ""), "key_type": key_types.get(col_upper, "None"),
            "data_type": inferred.get("data_type"), "stats": col_stats,
            "null_count": col_stats.get("null_count", 0), "null_percentage": col_stats.get("null_percentage", 0),
            "total_count": col_stats.get("total_count", 0), "sample_values": col_stats.get("sample_values", []),
            "faker_provider": inferred.get("faker_provider"), "faker_args": inferred.get("faker_args", {}),
            "data_domain": inferred.get("data_domain"), "constraints_description": inferred.get("constraints_description"),
            "post_processing_rules": relationships.get(col_upper, []), "data_quality": {}
        }
    print("Enhanced schema dictionary generated.")
    return enhanced_schema

def save_schema(schema, json_file_path):
    """Saves the enhanced schema to a JSON file."""
    try:
        with open(json_file_path, 'w', encoding='utf-8') as f: json.dump(schema, f, indent=4, ensure_ascii=False, default=str)
        print(f"Enhanced schema successfully saved to {json_file_path}")
    except Exception as e: print(f"Error saving enhanced schema to {json_file_path}: {e}")

def main():
    """Main function to orchestrate the schema generation process."""
    print("--- Starting Enhanced Schema Generation Process ---")
    global gemini_model, GEMINI_API_KEY
    if not GEMINI_API_KEY: print("Warning: GEMINI_API_KEY environment variable not set. Proceeding without Gemini inference."); gemini_model = None
    else:
        try: genai.configure(api_key=GEMINI_API_KEY); gemini_model = genai.GenerativeModel("gemini-1.5-flash"); print("Gemini model configured successfully.")
        except Exception as e: print(f"Error configuring Gemini model: {str(e)}. Proceeding without Gemini inference."); gemini_model = None

    df, original_dtypes = load_data(INPUT_CSV)
    if df is None: print("Failed to load input data. Exiting."); return

    original_case_map = {col.upper(): col for col in df.columns}
    df.columns = df.columns.str.upper(); print("Converted DataFrame column names to uppercase for processing.")
    df_columns_upper = df.columns.tolist()

    external_metadata = load_external_metadata(METADATA_JSON)
    if external_metadata is None: print("Failed to load required external metadata. Exiting."); return

    missing_meta_cols = [col for col in df_columns_upper if col not in external_metadata]
    if missing_meta_cols:
        print(f"Warning: The following columns exist in the CSV but are missing from the metadata file: {missing_meta_cols}")
        for col in missing_meta_cols: external_metadata[col] = {"Column_name": original_case_map.get(col, col), "Key_type": "None", "Description": "N/A - Missing from metadata"}; print(f"  Added default metadata for missing column: {col}")

    stats = calculate_basic_stats(df, original_dtypes)
    key_types = detect_key_types(df_columns_upper, external_metadata)

    column_inferences = {}
    print("\nInferring column info using LLM...")
    if os.path.exists(TEMP_RAW_RESPONSE_FILE):
        try: os.remove(TEMP_RAW_RESPONSE_FILE); print(f"Cleared previous raw Gemini response file: {TEMP_RAW_RESPONSE_FILE}")
        except Exception as e: print(f"Warning: Could not clear raw Gemini response file: {e}")

    for i, col_upper in enumerate(df_columns_upper):
         col_stats = stats.get(col_upper, {})
         col_key_type = key_types.get(col_upper, "None")
         external_meta_info = external_metadata.get(col_upper, {})
         inferred_info = infer_column_info_with_llm(col_upper, col_stats, col_key_type, external_meta_info)
         column_inferences[col_upper] = inferred_info
         # --- Add Delay ---
         if gemini_model is not None and i < len(df_columns_upper) - 1: # Don't sleep after the last call
              print(f"    Waiting {API_CALL_DELAY_SECONDS}s before next API call...")
              time.sleep(API_CALL_DELAY_SECONDS)
    print("LLM inference for all columns complete.")

    relationships = detect_column_correlations(df, stats, key_types)
    enhanced_schema = generate_enhanced_schema(df_columns_upper, key_types, stats, relationships, column_inferences, external_metadata)
    save_schema(enhanced_schema, OUTPUT_SCHEMA_JSON)
    print("\n--- Enhanced schema generation process finished ---")

if __name__ == "__main__":
    main()
