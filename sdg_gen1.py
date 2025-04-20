import pandas as pd
import json
import os
import numpy as np
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.constraints import ScalarRange, Unique
from sdv.metadata import SingleTableMetadata
import random
from faker import Faker
import inspect

# Configuration
INPUT_CSV = "customer_data.csv"  # Input CSV file
OUTPUT_CSV = "synthetic_data.csv"  # Output CSV file for synthetic data
INPUT_SCHEMA_JSON = "enhanced_schema.json"  # Input JSON file for the enhanced schema
NUM_ROWS = 210  # Number of synthetic rows to generate
CORRELATION_THRESHOLD = 0.7  # Define the correlation threshold that was missing

# Initialize Faker
fake = Faker(['vi_VN'])
Faker.seed(42)

def read_schema(json_file_path):
    """Reads the enhanced schema from a JSON file."""
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            schema = json.load(f)
        return schema
    except Exception as e:
        print(f"Error reading schema: {str(e)}")
        return None

def detect_column_correlations(df):
    """Detects column correlations, including functional dependencies."""
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

def create_sdv_metadata(df, enhanced_schema):
    """Creates SDV metadata from the DataFrame and enhanced schema."""
    try:
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(data=df)
        
        # Fix for datetime format warning - specify the format for datetime columns
        datetime_cols = [
            "DATE_LAST_CR_BANK", "DATE_LAST_DR_BANK", 
            "DATE_LAST_CR_AUTO", "DATE_LAST_DR_AUTO"
        ]
        
        for col_name, col_schema in enhanced_schema.items():
            if col_schema.get("key_type", "").lower() == "primary key":
                metadata.update_column(column_name=col_name, sdtype='id')
            if col_schema.get("data_type") == "datetime" or col_name in datetime_cols:
                # Add datetime_format parameter
                metadata.update_column(
                    column_name=col_name, 
                    sdtype='datetime',
                    datetime_format='%Y-%m-%d'  # Default format, adjust as needed
                )
            if col_schema.get("data_type") == "categorical":
                metadata.update_column(column_name=col_name, sdtype='categorical')
        return metadata
    except Exception as e:
        print(f"Error creating SDV metadata: {str(e)}")
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(data=df)
        return metadata

# Helper function to check the available parameters for a class
def get_param_names(cls):
    try:
        sig = inspect.signature(cls.__init__)
        return list(sig.parameters.keys())[1:]  # Skip 'self'
    except Exception as e:
        print(f"Could not get parameters for {cls.__name__}: {str(e)}")
        return []

def parse_sdv_constraints(constraints_list, col_name, enhanced_schema):
    """Parses SDV constraints from the schema."""
    # First, dynamically inspect ScalarRange parameters
    scalar_range_params = get_param_names(ScalarRange)
    print(f"ScalarRange parameters: {scalar_range_params}")
    
    # Map common parameter names to what might be in the actual class
    column_param = next((p for p in scalar_range_params if 'column' in p.lower()), None)
    min_param = next((p for p in scalar_range_params if 'min' in p.lower() or 'low' in p.lower()), None)
    max_param = next((p for p in scalar_range_params if 'max' in p.lower() or 'high' in p.lower() or 'upper' in p.lower()), None)
    
    print(f"Using column param: {column_param}, min param: {min_param}, max param: {max_param}")
    
    if not all([column_param, min_param, max_param]):
        print("WARNING: Could not identify all required ScalarRange parameters. Constraints will be skipped.")
        return []
        
    constraints = []
    for constraint in constraints_list:
        try:
            # Handle dictionary-based constraints
            if isinstance(constraint, dict):
                if constraint.get("constraint_type") == "Range":
                    min_val = constraint.get("min_value")
                    max_val = constraint.get("max_value")
                    # Create kwargs dictionary dynamically based on parameter names
                    kwargs = {
                        column_param: col_name,
                        min_param: min_val,
                        max_param: max_val
                    }
                    constraints.append(ScalarRange(**kwargs))
                elif constraint.get("constraint_type") == "Unique":
                    # Try to find the right parameter for Unique
                    unique_params = get_param_names(Unique)
                    column_list_param = next((p for p in unique_params if 'column' in p.lower() and ('names' in p.lower() or 's' in p.lower())), None)
                    if column_list_param:
                        constraints.append(Unique(**{column_list_param: [col_name]}))
                    else:
                        print(f"Warning: Cannot determine parameters for Unique constraint on {col_name}")
                continue
                
            # Handle string constraints
            if isinstance(constraint, str):
                constraint_str = constraint
                if constraint_str.lower() == "unique":
                    unique_params = get_param_names(Unique)
                    column_list_param = next((p for p in unique_params if 'column' in p.lower() and ('names' in p.lower() or 's' in p.lower())), None)
                    if column_list_param:
                        constraints.append(Unique(**{column_list_param: [col_name]}))
                    else:
                        print(f"Warning: Cannot determine parameters for Unique constraint on {col_name}")
                elif constraint_str.startswith("range:"):
                    values = constraint_str.split(":")[1].strip().split(",")
                    min_val = float(values[0].strip())
                    max_val = float(values[1].strip())
                    kwargs = {
                        column_param: col_name,
                        min_param: min_val,
                        max_param: max_val
                    }
                    constraints.append(ScalarRange(**kwargs))
                elif constraint_str.startswith("scalar_inequality:"):
                    print(f"Warning: ScalarInequality constraints are not supported in the current SDV version.")
        except Exception as e:
            print(f"Error parsing constraint '{constraint}' for column '{col_name}': {str(e)}")
            import traceback
            traceback.print_exc()
    return constraints

def generate_weighted_random_element(categories):
    """Generates a random element from a list of categories with weights."""
    values = [item["value"] for item in categories]
    weights = [item["percentage"] for item in categories]
    return random.choices(values, weights=weights, k=1)[0]

def generate_synthetic_data_with_sdv(df, enhanced_schema, num_rows, output_file):
    """Generates synthetic data using SDV, applying Faker providers and post-processing."""
    try:
        print("Creating SDV metadata...")
        metadata = create_sdv_metadata(df, enhanced_schema)
        
        # Check if the synthesizer accepts constraints in the constructor
        synthesizer_params = get_param_names(GaussianCopulaSynthesizer)
        print(f"GaussianCopulaSynthesizer parameters: {synthesizer_params}")
        
        print("Setting up SDV synthesizer...")
        # Create synthesizer with only required parameters
        synthesizer = GaussianCopulaSynthesizer(metadata=metadata)
        
        print("Parsing constraints from enhanced schema...")
        constraints = []
        for col_name, col_schema in enhanced_schema.items():
            if "sdv_constraints" in col_schema and col_schema["sdv_constraints"]:
                col_constraints = parse_sdv_constraints(col_schema["sdv_constraints"], col_name, enhanced_schema)
                constraints.extend(col_constraints)
        
        print(f"Found {len(constraints)} constraints that will be applied during sampling")
        
        print("Fitting synthesizer to data...")
        synthesizer.fit(df)
        
        # Check sample method parameters
        sample_params = inspect.signature(synthesizer.sample).parameters
        print(f"Sample method parameters: {list(sample_params.keys())}")
        
        print(f"Generating {num_rows} rows of synthetic data...")
        # Pass constraints directly to the sample method in SDV 1.20.0
        synthetic_data = synthesizer.sample(num_rows=num_rows, constraints=constraints)
        
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
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function to generate synthetic data from an enhanced schema."""
    if not os.path.exists(INPUT_CSV):
        print(f"Error: Input file '{INPUT_CSV}' does not exist.")
        return
    if not os.path.exists(INPUT_SCHEMA_JSON):
        print(f"Error: Input schema file '{INPUT_SCHEMA_JSON}' does not exist.")
        return

    print("Starting synthetic data generation process from schema...")
    enhanced_schema = read_schema(INPUT_SCHEMA_JSON)
    if not enhanced_schema:
        print("Failed to read enhanced schema. Exiting.")
        return

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