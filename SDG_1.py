import pandas as pd
import json
import os
import numpy as np
import google.generativeai as genai
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.constraints import Constraint, Range, ScalarInequality, Unique
from sdv.metadata import SingleTableMetadata
import random
from faker import Faker

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
        
        for col1 in df.columns:
            for col2 in df.columns:
                if col1 != col2:
                    groups = df.groupby(col1)[col2].nunique()
                    if (groups <= 1).all():
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
                schema[column]["data_type"] = "categorical"
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
                        
                schema[column]["stats"] = stats
                
            elif pd.api.types.is_numeric_dtype(col_data):
                non_null_data = col_data.dropna()
                schema[column]["data_type"] = "integer" if non_null_data.apply(lambda x: float(x).is_integer()).all() else "float"
                
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
                    
                schema[column]["stats"] = stats
                
            elif pd.api.types.is_datetime64_dtype(col_data):
                schema[column]["data_type"] = "datetime"
                
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
                    
                schema[column]["stats"] = stats
                
            else:
                schema[column]["data_type"] = "string"
                
                stats = {"unique_values_count": int(col_data.nunique())}
                if non_null_count > 0:
                    str_lengths = col_data.dropna().astype(str).str.len()
                    stats.update({
                        "max_length": int(str_lengths.max()),
                        "min_length": int(str_lengths.min()),
                        "mean_length": float(str_lengths.mean())
                    })
                    
                schema[column]["stats"] = stats
            
            if column in correlations:
                schema[column]["correlations"] = correlations[column]
                
        return schema
    except Exception as e:
        print(f"Error generating schema: {str(e)}")
        return None

def enhance_schema_batch(schema_batch):
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
           For Range constraints use exact format: "Range:min,max" 
           For Unique constraints use format: "Unique"
           For ScalarInequality use format: "ScalarInequality:other_column,operation"
        9. post_processing: Any special post-processing function needed

        Return just the valid JSON matching the original structure, nothing else.
        """
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        if "```json" in response_text:
            json_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            json_text = response_text.split("```")[1].strip()
        else:
            json_text = response_text
            
        # Try to clean the JSON text if there are issues
        json_text = json_text.replace('\n', ' ').replace('\r', '')
        json_text = json_text.replace('```', '')
        
        try:
            enhanced_batch = json.loads(json_text)
            return enhanced_batch
        except json.JSONDecodeError as je:
            print(f"JSON decode error: {je}. Trying to clean the JSON text...")
            # Safer approach: process the original schema and just keep what we got
            for col in schema_batch:
                schema_batch[col]["description"] = schema_batch[col].get("description", "")
                schema_batch[col]["domain"] = "Unknown"
                schema_batch[col]["constraints_description"] = ""
                schema_batch[col]["relationships"] = ""
                schema_batch[col]["data_quality"] = ""
            return schema_batch
    except Exception as e:
        print(f"Error in schema enhancement: {str(e)}")
        return schema_batch

def enhance_schema_with_gemini(schema):
    column_names = list(schema.keys())
    num_columns = len(column_names)
    num_batches = (num_columns + BATCH_SIZE - 1) // BATCH_SIZE
    
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

def create_sdv_metadata(df, enhanced_schema):
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=df)
    
    # Fix for datetime_format warning
    for col_name in df.columns:
        if col_name in enhanced_schema and enhanced_schema[col_name].get("data_type") == "datetime":
            # Set a default format for all datetime columns
            metadata.update_column(column_name=col_name, sdtype='datetime', datetime_format="%Y-%m-%d")
            
        if col_name in enhanced_schema and enhanced_schema[col_name].get("key_type", "").lower() == "primary key":
            metadata.update_column(column_name=col_name, sdtype='id')
            
    return metadata

def generate_weighted_random_element(categories):
    values = [item["value"] for item in categories]
    weights = [item["percentage"] for item in categories]
    return random.choices(values, weights=weights, k=1)[0]

def get_dependency_mapping(df, source_col, target_col):
    mapping = {}
    for _, row in df.iterrows():
        source_val = row[source_col]
        target_val = row[target_col]
        if not pd.isna(source_val) and not pd.isna(target_val):
            mapping[source_val] = target_val
    return mapping

def generate_synthetic_data_with_sdv(df, enhanced_schema, num_rows, output_file):
    try:
        print("Creating SDV metadata...")
        metadata = create_sdv_metadata(df, enhanced_schema)
        
        print("Setting up SDV synthesizer...")
        synthesizer = GaussianCopulaSynthesizer(metadata)
        
        print("Fitting synthesizer to data...")
        synthesizer.fit(df)
        
        # Note: SDV constraint errors fixed - we won't try to add constraints via synthesizer
        # since the GaussianCopulaSynthesizer doesn't support add_constraint
        
        print(f"Generating {num_rows} rows of synthetic data...")
        synthetic_data = synthesizer.sample(num_rows=num_rows)
        
        # Find all dependency relationships
        dependencies = []
        for col_name, col_schema in enhanced_schema.items():
            if "correlations" in col_schema:
                for correlation in col_schema["correlations"]:
                    if correlation.get("type") == "functional_dependency":
                        dependencies.append((col_name, correlation["column"]))
        
        # Handle categorical columns with preserved probability distribution
        print("Applying Faker providers and preserving categorical distributions...")
        for col_name, col_schema in enhanced_schema.items():
            if col_schema.get("data_type") == "categorical":
                if "stats" in col_schema and ("categories" in col_schema["stats"] or "top_categories" in col_schema["stats"]):
                    categories = col_schema["stats"].get("categories") or col_schema["stats"].get("top_categories")
                    if categories:
                        print(f"  - Preserving value distribution for '{col_name}'")
                        synthetic_data[col_name] = synthetic_data.apply(
                            lambda _: generate_weighted_random_element(categories), axis=1
                        )
                        continue
            
            faker_provider = col_schema.get("faker_provider")
            faker_args = col_schema.get("faker_args", {})
            
            if faker_provider:
                print(f"  - Applying Faker provider '{faker_provider}' to '{col_name}'")
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
        
        # Apply functional dependencies
        print("Applying functional dependencies...")
        for source_col, target_col in dependencies:
            print(f"  - Applying dependency: {source_col} -> {target_col}")
            mapping = get_dependency_mapping(df, source_col, target_col)
            
            if not mapping:
                continue
                
            def apply_mapping(x):
                if pd.isna(x): 
                    return x
                    
                # Exact match
                if x in mapping:
                    return mapping[x]
                    
                # Find closest match for numeric columns
                if isinstance(x, (int, float)):
                    keys = [k for k in mapping.keys() if isinstance(k, (int, float))]
                    if keys:
                        closest_key = min(keys, key=lambda k: abs(k - x))
                        return mapping[closest_key]
                        
                # No match found
                return x
                
            synthetic_data[target_col] = synthetic_data[source_col].apply(apply_mapping)
        
        # Apply post-processing
        print("Applying post-processing functions...")
        for col_name, col_schema in enhanced_schema.items():
            post_processing = col_schema.get("post_processing")
            if post_processing:
                print(f"  - Post-processing '{col_name}' with '{post_processing}'")
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
        
        # Compare correlations
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
        
        # Save the final output
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