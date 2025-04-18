import pandas as pd
import json
import os
import numpy as np
import google.generativeai as genai
from google.generativeai import GenerativeModel
import math
from faker import Faker
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.constraints import Constraint, Range, ScalarInequality, Unique
from sdv.metadata import SingleTableMetadata
import re
import random
from datetime import datetime, timedelta

INPUT_CSV = "customer_data.csv"
OUTPUT_CSV = "synthetic_data.csv"
METADATA_JSON = "FCT_ENT_TERM_DEPOSIT_metadata.json"
NUM_ROWS = 210
BATCH_SIZE = 5
CORRELATION_THRESHOLD = 0.5

os.environ["HTTP_PROXY"] = "http://dc2-proxyuat.seauat.com.vn:8080"
os.environ["HTTPS_PROXY"] = "http://dc2-proxyuat.seauat.com.vn:8080"

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set.")
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")

fake = Faker(['vi_VN'])
Faker.seed(42)

def read_metadata(json_file_path):
    try:
        if not os.path.exists(json_file_path):
            print(f"Metadata file {json_file_path} not found. Continuing without metadata.")
            return {}
        with open(json_file_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        column_metadata = {}
        if "fct_ent_term_deposit" in metadata and "columns" in metadata["fct_ent_term_deposit"]:
            for column in metadata["fct_ent_term_deposit"]["columns"]:
                column_name = column.get("Column_name", "").strip()
                if column_name:
                    column_metadata[column_name] = {
                        "description": column.get("Description", ""),
                        "key_type": column.get("Key_type", "")
                    }
        return column_metadata
    except Exception as e:
        print(f"Error reading metadata: {str(e)}")
        return {}

def detect_column_correlations(df):
    correlations = {}
    try:
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if len(numeric_cols) < 2:
            return correlations
        corr_matrix = df[numeric_cols].corr(method='pearson', numeric_only=True)
        for i, col1 in enumerate(numeric_cols):
            for j, col2 in enumerate(numeric_cols):
                if i < j:
                    corr_value = corr_matrix.iloc[i, j]
                    if abs(corr_value) >= CORRELATION_THRESHOLD and not pd.isna(corr_value):
                        if col1 not in correlations:
                            correlations[col1] = []
                        correlations[col1].append({
                            "column": col2,
                            "correlation": round(corr_value, 3),
                            "type": "positive" if corr_value > 0 else "negative"
                        })
        for col1 in numeric_cols:
            for col2 in numeric_cols:
                if col1 != col2:
                    df_grouped = df.groupby(col1)[col2]
                    unique_values = df_grouped.nunique()
                    if (unique_values <= 1).all():
                        if col1 not in correlations:
                            correlations[col1] = []
                        correlations[col1].append({
                            "column": col2,
                            "correlation": 1.0,
                            "type": "functional_dependency"
                        })
    except Exception as e:
        print(f"Error detecting correlations: {str(e)}")
    return correlations

def is_categorical(series, threshold=0.5):
    if pd.api.types.is_categorical_dtype(series):
        return True
    if series.dtype == object:
        if series.nunique() / series.count() <= threshold:
            return True
    if pd.api.types.is_numeric_dtype(series):
        unique_values = series.dropna().nunique()
        if unique_values <= 20 and unique_values / series.count() <= threshold:
            return True
    return False

def read_csv_and_generate_schema(csv_file_path, metadata=None):
    try:
        df = pd.read_csv(csv_file_path)
        if df.empty:
            print("Warning: CSV file is empty")
            return {}
        schema = {}
        correlations = detect_column_correlations(df)
        for column in df.columns:
            col_data = df[column]
            non_null_count = col_data.count()
            null_count = col_data.isna().sum()
            total_count = len(col_data)
            null_percentage = round((null_count / total_count) * 100, 2) if total_count > 0 else 0
            sample_values = []
            if non_null_count > 0:
                sample_size = min(5, non_null_count)
                sample_values = col_data.dropna().sample(sample_size).tolist() if sample_size > 0 else []
            schema[column] = {
                "description": metadata.get(column, {}).get("description", "") if metadata else "",
                "key_type": metadata.get(column, {}).get("key_type", "") if metadata else "",
                "null_count": int(null_count),
                "null_percentage": null_percentage,
                "total_count": int(total_count),
                "sample_values": sample_values,
                "faker_provider": None,
                "faker_args": {},
                "sdv_constraints": [],
                "post_processing": None,
            }
            if is_categorical(col_data):
                data_type = "categorical"
                stats = {"unique_values_count": int(col_data.nunique())}
                if non_null_count > 0:
                    value_counts = col_data.value_counts(dropna=True)
                    value_percentages = (value_counts / non_null_count * 100).round(2)
                    categories = []
                    for val, count in value_counts.items():
                        percentage = value_percentages[val]
                        categories.append({
                            "value": val,
                            "count": int(count),
                            "percentage": float(percentage)
                        })
                    if len(categories) > 10:
                        stats["top_categories"] = categories[:10]
                    else:
                        stats["categories"] = categories
                schema[column]["data_type"] = data_type
                schema[column]["stats"] = stats
            elif pd.api.types.is_numeric_dtype(col_data):
                non_null_data = col_data.dropna()
                data_type = "integer" if non_null_data.apply(lambda x: float(x).is_integer()).all() else "float"
                stats = {}
                if non_null_count > 0:
                    stats = {
                        "min": float(non_null_data.min()),
                        "max": float(non_null_data.max()),
                        "mean": float(non_null_data.mean()),
                        "median": float(non_null_data.median()),
                        "std_dev": float(non_null_data.std()),
                        "unique_values_count": int(col_data.nunique())
                    }
                schema[column]["data_type"] = data_type
                schema[column]["stats"] = stats
            elif pd.api.types.is_datetime64_dtype(col_data):
                data_type = "datetime"
                stats = {}
                if non_null_count > 0:
                    date_min = col_data.min()
                    date_max = col_data.max()
                    stats = {
                        "min": str(date_min),
                        "max": str(date_max),
                        "unique_values_count": int(col_data.nunique()),
                        "date_range_days": (date_max - date_min).days if not pd.isna(date_min) and not pd.isna(date_max) else 0
                    }
                schema[column]["data_type"] = data_type
                schema[column]["stats"] = stats
            else:
                data_type = "string"
                stats = {"unique_values_count": int(col_data.nunique())}
                if non_null_count > 0:
                    str_lengths = col_data.dropna().astype(str).str.len()
                    stats.update({
                        "max_length": int(str_lengths.max()),
                        "min_length": int(str_lengths.min()),
                        "mean_length": float(str_lengths.mean())
                    })
                schema[column]["data_type"] = data_type
                schema[column]["stats"] = stats
            if column in correlations:
                schema[column]["correlations"] = correlations[column]
        return schema
    except Exception as e:
        print(f"Error generating schema: {str(e)}")
        return None

def enhance_schema_batch_alternative(schema_batch):
    try:
        prompt = f"""
        Enhance the schema of these columns from a Vietnamese banking dataset:
        {json.dumps(schema_batch, indent=2)}
        
        For each column, add:
        1. A description field with a detailed description if not already present
        2. A domain field with realistic ranges for values
        3. A constraints_description field with patterns to maintain (human readable)
        4. A relationships field describing correlations to other columns
        5. A data_quality field with recommendations for validation
        
        Additionally, add these fields for integration with SDV and Faker:
        6. faker_provider: The appropriate Faker provider method to generate realistic values
        7. faker_args: A JSON object with parameters for the Faker provider
        8. sdv_constraints: Array of constraint types to apply using current SDV v1.20.0 constraints (Unique, Range, ScalarInequality)
        9. post_processing: Any special post-processing function needed

        Return your response as valid JSON matching the original structure with these additional fields.
        """
        response = model.generate_content(prompt)
        response_text = response.text
        if "```json" in response_text:
            json_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            json_text = response_text.split("```")[1].strip()
        else:
            json_text = response_text
        enhanced_batch = json.loads(json_text)
        return enhanced_batch
    except Exception as e:
        print(f"Error in alternative enhancement approach: {str(e)}")
        return schema_batch

def enhance_schema_batch(schema_batch):
    try:
        schema_properties = {}
        required_fields = []
        for column_name, column_info in schema_batch.items():
            stats_schema = {
                "type": "object",
                "properties": {
                    "unique_values_count": {"type": "integer"},
                    "min": {"type": ["number", "string"]},
                    "max": {"type": ["number", "string"]},
                    "mean": {"type": "number"},
                    "median": {"type": "number"},
                    "std_dev": {"type": "number"},
                    "categories": {"type": "array", "items": {"type": "object"}},
                    "top_categories": {"type": "array", "items": {"type": "object"}},
                    "max_length": {"type": "integer"},
                }
            }
            correlations_schema = {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "column": {"type": "string"},
                        "correlation": {"type": "number"},
                        "type": {"type": "string"}
                    }
                }
            }
            schema_properties[column_name] = {
                "type": "object",
                "properties": {
                    "data_type": {"type": "string"},
                    "description": {"type": "string", "description": "Detailed description of what this column represents"},
                    "key_type": {"type": "string"},
                    "stats": stats_schema,
                    "null_count": {"type": "integer"},
                    "null_percentage": {"type": "number"},
                    "total_count": {"type": "integer"},
                    "sample_values": {"type": "array", "items": {"type": ["string", "number", "null"]}},
                    "correlations": correlations_schema,
                    "domain": {"type": "string", "description": "Domain and realistic ranges for values"},
                    "constraints_description": {"type": "string", "description": "Human-readable patterns or constraints that should be maintained"},
                    "relationships": {"type": "string", "description": "Relationships or correlations to other columns"},
                    "data_quality": {"type": "string", "description": "Recommendations for data validation"},
                    "faker_provider": {"type": ["string", "null"], "description": "Faker provider to generate realistic values"},
                    "faker_args": {"type": "object", "description": "Arguments for the Faker provider"},
                    "sdv_constraints": {"type": "array", "items": {"type": "string"}, "description": "SDV constraints to apply"},
                    "post_processing": {"type": ["string", "null"], "description": "Post-processing function needed"}
                },
                "required": ["data_type", "stats", "null_count", "sample_values", "description", "domain", 
                             "constraints_description", "faker_provider", "faker_args", "sdv_constraints"]
            }
            required_fields.append(column_name)
        structured_schema = {
            "type": "object", 
            "properties": schema_properties,
            "required": required_fields
        }
        structured_model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            generation_config={
                "response_mime_type": "application/json",
            }
        )
        prompt = f"""
        Enhance the schema of these columns from a Vietnamese banking dataset:
        {json.dumps(schema_batch, indent=2)}
        
        For each column, add:
        1. A detailed description field if not already present 
        2. A domain field with realistic ranges for values
        3. A constraints_description field with human-readable patterns to maintain
        4. A relationships field describing correlations to other columns
        5. A data_quality field with recommendations for validation
        
        Additionally, add these fields for integration with SDV 1.20.0 and Faker:
        6. faker_provider: The appropriate Faker provider method to generate realistic values
           - Use null if SDV should handle the distribution 
           - For strings/categoricals that don't match a Faker provider, use "random_element" with predefined values
        7. faker_args: A JSON object with parameters for the Faker provider
        8. sdv_constraints: Array of constraints using ONLY these modern SDV 1.20.0 classes:
           - "unique" (Unique constraint)
           - "range:min,max" (Range constraint) 
           - "scalar_inequality:column_name,operation" (ScalarInequality constraint with operation being '>', '<', '>=', or '<=')
        9. post_processing: Any special post-processing function needed
        """
        try:
            response = structured_model.generate_content(
                prompt,
                generation_config={"response_schema": structured_schema}
            )
            enhanced_batch = response.candidates[0].content.parts[0].text
            enhanced_batch = json.loads(enhanced_batch)
            return enhanced_batch
        except Exception as e:
            print(f"Structured output failed: {str(e)}. Trying alternative approach...")
            return enhance_schema_batch_alternative(schema_batch)
    except Exception as e:
        print(f"Error enhancing schema batch: {str(e)}")
        return schema_batch

def enhance_schema_with_gemini(schema):
    try:
        column_names = list(schema.keys())
        num_columns = len(column_names)
        num_batches = math.ceil(num_columns / BATCH_SIZE)
        print(f"Processing {num_columns} columns in {num_batches} batches of {BATCH_SIZE}...")
        enhanced_schema = {}
        for i in range(num_batches):
            start_idx = i * BATCH_SIZE
            end_idx = min((i + 1) * BATCH_SIZE, num_columns)
            batch_columns = column_names[start_idx:end_idx]
            print(f"Processing batch {i+1}/{num_batches} with columns: {batch_columns}")
            schema_batch = {col: schema[col] for col in batch_columns}
            try:
                enhanced_batch = enhance_schema_batch(schema_batch)
                if not isinstance(enhanced_batch, dict):
                    print(f"Warning: Received non-dictionary result for batch {i+1}. Using original schema for this batch.")
                    enhanced_batch = schema_batch
                enhanced_schema.update(enhanced_batch)
            except Exception as batch_error:
                print(f"Error processing batch {i+1}: {str(batch_error)}")
                print("Using original schema for this batch.")
                enhanced_schema.update(schema_batch)
        return enhanced_schema
    except Exception as e:
        print(f"Error in batch enhancement process: {str(e)}")
        return schema

def create_sdv_metadata(df, enhanced_schema):
    try:
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(data=df)
        for col_name, col_schema in enhanced_schema.items():
            if col_schema.get("key_type", "").lower() == "primary key":
                metadata.update_column(column_name=col_name, sdtype='id')
            if col_schema.get("data_type") == "datetime":
                metadata.update_column(column_name=col_name, sdtype='datetime')
            if col_schema.get("data_type") == "categorical":
                metadata.update_column(column_name=col_name, sdtype='categorical')
        return metadata
    except Exception as e:
        print(f"Error creating SDV metadata: {str(e)}")
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(data=df)
        return metadata

def parse_sdv_constraints(constraints_list, col_name, enhanced_schema):
    constraints = []
    for constraint_str in constraints_list:
        try:
            if constraint_str.lower() == "unique":
                constraints.append(Unique(columns=[col_name]))
            elif constraint_str.startswith("range:"):
                values = constraint_str.split(":")[1].strip().split(",")
                min_val = float(values[0].strip())
                max_val = float(values[1].strip())
                constraints.append(Range(column_name=col_name, low=min_val, high=max_val))
            elif constraint_str.startswith("scalar_inequality:"):
                parts = constraint_str.split(":")[1].strip().split(",")
                other_col = parts[0].strip()
                operation = parts[1].strip()
                op_map = {">": ">", "<": "<", ">=": ">=", "<=": "<="}
                if operation in op_map:
                    constraints.append(ScalarInequality(column_names=[col_name, other_col], relation=op_map[operation]))
                else:
                    print(f"Unsupported operation '{operation}' in constraint '{constraint_str}'")
        except Exception as e:
            print(f"Error parsing constraint '{constraint_str}' for column '{col_name}': {str(e)}")
    return constraints

def generate_weighted_random_element(categories):
    values = [item["value"] for item in categories]
    weights = [item["percentage"] for item in categories]
    return random.choices(values, weights=weights, k=1)[0]

def generate_synthetic_data_with_sdv(df, enhanced_schema, num_rows, output_file):
    try:
        print("Creating SDV metadata...")
        metadata = create_sdv_metadata(df, enhanced_schema)
        print("Setting up SDV synthesizer...")
        synthesizer = GaussianCopulaSynthesizer(metadata)
        print("Fitting synthesizer to data...")
        synthesizer.fit(df)
        print("Parsing constraints from enhanced schema...")
        constraints = []
        for col_name, col_schema in enhanced_schema.items():
            if "sdv_constraints" in col_schema and col_schema["sdv_constraints"]:
                col_constraints = parse_sdv_constraints(col_schema["sdv_constraints"], col_name, enhanced_schema)
                constraints.extend(col_constraints)
        if constraints:
            print(f"Applying {len(constraints)} constraints to synthesizer...")
            for constraint in constraints:
                synthesizer.add_constraint(constraint)
        print(f"Generating {num_rows} rows of synthetic data...")
        synthetic_data = synthesizer.sample(num_rows=num_rows)
        
        # Handle categorical columns with preserved probability distribution
        print("Applying Faker providers and preserving categorical distributions...")
        for col_name, col_schema in enhanced_schema.items():
            if col_schema.get("data_type") == "categorical":
                if "stats" in col_schema and ("categories" in col_schema["stats"] or "top_categories" in col_schema["stats"]):
                    categories = col_schema["stats"].get("categories") or col_schema["stats"].get("top_categories")
                    if categories:
                        print(f"  - Preserving value distribution for categorical column '{col_name}'")
                        synthetic_data[col_name] = synthetic_data.apply(
                            lambda _: generate_weighted_random_element(categories), axis=1
                        )
                        continue
            
            faker_provider = col_schema.get("faker_provider")
            faker_args = col_schema.get("faker_args", {})
            if faker_provider:
                print(f"  - Applying Faker provider '{faker_provider}' to column '{col_name}'")
                if faker_provider == "random_element" and "elements" in faker_args:
                    elements = faker_args["elements"]
                    synthetic_data[col_name] = synthetic_data.apply(
                        lambda _: random.choice(elements), axis=1
                    )
                    continue
                try:
                    faker_method = getattr(fake, faker_provider)
                    synthetic_data[col_name] = synthetic_data.apply(
                        lambda _: faker_method(**faker_args), axis=1
                    )
                except AttributeError:
                    print(f"Warning: Faker provider '{faker_provider}' not found. Keeping original values.")
                except Exception as e:
                    print(f"Error applying Faker provider to '{col_name}': {str(e)}")
        
        print("Applying post-processing functions...")
        for col_name, col_schema in enhanced_schema.items():
            post_processing = col_schema.get("post_processing")
            if post_processing:
                print(f"  - Applying post-processing '{post_processing}' to column '{col_name}'")
                if post_processing == "format_as_currency":
                    synthetic_data[col_name] = synthetic_data[col_name].apply(
                        lambda x: f"{float(x):,.2f} VND" if not pd.isna(x) else x
                    )
                elif post_processing == "ensure_valid_id":
                    synthetic_data[col_name] = synthetic_data[col_name].apply(
                        lambda x: f"ID{int(abs(x)):08d}" if not pd.isna(x) else x
                    )
                elif post_processing == "format_percentage":
                    synthetic_data[col_name] = synthetic_data[col_name].apply(
                        lambda x: f"{float(x):.2f}%" if not pd.isna(x) else x
                    )
                elif post_processing == "format_date":
                    synthetic_data[col_name] = synthetic_data[col_name].apply(
                        lambda x: x.strftime("%Y-%m-%d") if not pd.isna(x) else x
                    )
        
        print("Applying functional dependencies from correlations...")
        for col_name, col_schema in enhanced_schema.items():
            if "correlations" in col_schema:
                for correlation in col_schema["correlations"]:
                    if correlation.get("type") == "functional_dependency":
                        target_col = correlation["column"]
                        print(f"  - Applying functional dependency: {col_name} -> {target_col}")
                        mapping = {}
                        for _, row in df.iterrows():
                            if not pd.isna(row[col_name]) and not pd.isna(row[target_col]):
                                mapping[row[col_name]] = row[target_col]
                        if mapping:
                            def find_matching_value(x):
                                if pd.isna(x): return x
                                if x in mapping: return mapping[x]
                                if isinstance(x, (int, float)):
                                    keys = list(mapping.keys())
                                    closest_key = min(keys, key=lambda k: abs(k - x))
                                    return mapping[closest_key]
                                return x
                            synthetic_data[target_col] = synthetic_data[col_name].apply(find_matching_value)
        
        # Compare correlations between original and synthetic data
        print("Checking correlations between original and synthetic data...")
        original_correlations = detect_column_correlations(df)
        synthetic_correlations = detect_column_correlations(synthetic_data)
        
        correlation_report = {}
        for col_name in original_correlations:
            if col_name in synthetic_correlations:
                correlation_report[col_name] = {
                    "original": original_correlations[col_name],
                    "synthetic": synthetic_correlations[col_name]
                }
        
        correlation_report_file = f"{os.path.splitext(output_file)[0]}_correlation_report.json"
        with open(correlation_report_file, 'w', encoding='utf-8') as f:
            json.dump(correlation_report, f, indent=2, ensure_ascii=False)
        print(f"Correlation comparison saved to {correlation_report_file}")
        
        print(f"Saving synthetic data to {output_file}...")
        synthetic_data.to_csv(output_file, index=False)
        print(f"Successfully generated {len(synthetic_data)} rows of synthetic data!")
        return True
    except Exception as e:
        print(f"Error generating synthetic data with SDV: {str(e)}")
        return False

def main():
    if not os.path.exists(INPUT_CSV):
        print(f"Error: Input file '{INPUT_CSV}' does not exist.")
        return
    
    print("Starting synthetic data generation process")
    metadata = read_metadata(METADATA_JSON)
    
    print(f"Generating schema from {INPUT_CSV}...")
    schema = read_csv_and_generate_schema(INPUT_CSV, metadata)
    if not schema:
        print("Failed to generate schema. Exiting.")
        return
    
    basic_schema_file = f"{os.path.splitext(OUTPUT_CSV)[0]}_basic_schema.json"
    with open(basic_schema_file, 'w', encoding='utf-8') as f:
        json.dump(schema, f, indent=2, ensure_ascii=False)
    print(f"Basic schema saved to {basic_schema_file}")
    
    print("Enhancing schema with Gemini...")
    try:
        enhanced_schema = enhance_schema_with_gemini(schema)
        enhanced_schema_file = f"{os.path.splitext(OUTPUT_CSV)[0]}_enhanced_schema.json"
        with open(enhanced_schema_file, 'w', encoding='utf-8') as f:
            json.dump(enhanced_schema, f, indent=2, ensure_ascii=False)
        print(f"Enhanced schema saved to {enhanced_schema_file}")
    except Exception as e:
        print(f"Error enhancing schema: {str(e)}")
        print("Using basic schema without enhancements...")
        enhanced_schema = schema
    
    try:
        original_df = pd.read_csv(INPUT_CSV)
        print(f"Loaded original data with {len(original_df)} rows and {len(original_df.columns)} columns")
    except Exception as e:
        print(f"Error loading original data: {str(e)}")
        return

    print(f"Generating {NUM_ROWS} rows of synthetic data...")
    success = generate_synthetic_data_with_sdv(original_df, enhanced_schema, NUM_ROWS, OUTPUT_CSV)
    
    if success:
        print(f"Synthetic data generation completed successfully! Output saved to {OUTPUT_CSV}")
    else:
        print("Synthetic data generation failed.")

if __name__ == "__main__":
    main()