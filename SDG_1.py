import pandas as pd
import json
import os
import csv
import google.generativeai as genai
from google.generativeai import GenerativeModel

# Configuration - EDIT THESE VALUES
INPUT_CSV = "customer_data.csv"  # Path to your input CSV file
OUTPUT_CSV = "synthetic_data.csv"  # Path for the output synthetic data
NUM_ROWS = 100  # Number of synthetic rows to generate
details = "A vietnamese bank data of finance, clients and transaction"

# Proxy and API configuration
os.environ["HTTP_PROXY"] = "http://dc2-proxyuat.seauat.com.vn:8080"
os.environ["HTTPS_PROXY"] = "http://dc2-proxyuat.seauat.com.vn:8080"

# Configure Gemini API
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set.")
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")


def read_csv_and_generate_schema(csv_file_path):
    """
    Read a CSV file and generate a basic schema with data types and basic statistics.
    """
    try:
        # Read CSV file
        df = pd.read_csv(csv_file_path)
        
        # Check if DataFrame is empty
        if df.empty:
            print("Warning: CSV file is empty")
            return {}
        
        # Initialize schema dictionary
        schema = {}
        
        # For each column, extract data type and basic statistics
        for column in df.columns:
            col_data = df[column]
            
            # Calculate number of non-null values for sampling
            non_null_count = col_data.count()
            
            # Initialize sample values list
            sample_values = []
            
            # Only attempt to get sample values if there are non-null values
            if non_null_count > 0:
                # Get a sample of non-null values (up to 5)
                sample_size = min(5, non_null_count)
                sample_values = col_data.dropna().sample(sample_size).tolist() if sample_size > 0 else []
            
            # Determine the data type
            if pd.api.types.is_numeric_dtype(col_data):
                data_type = "integer" if col_data.dropna().apply(lambda x: float(x).is_integer()).all() else "float"
                
                # Get statistics for numeric columns (with null checks)
                stats = {}
                if non_null_count > 0:
                    stats = {
                        "min": float(col_data.min()),
                        "max": float(col_data.max()),
                        "mean": float(col_data.mean()),
                        "unique_values_count": int(col_data.nunique())
                    }
                
            elif pd.api.types.is_datetime64_dtype(col_data):
                data_type = "datetime"
                stats = {}
                if non_null_count > 0:
                    stats = {
                        "min": str(col_data.min()),
                        "max": str(col_data.max()),
                        "unique_values_count": int(col_data.nunique())
                    }
                
            elif pd.api.types.is_categorical_dtype(col_data) or (non_null_count > 0 and col_data.nunique() / non_null_count < 0.5):
                data_type = "categorical"
                stats = {"unique_values_count": int(col_data.nunique())}
                
                # Only attempt to get categories if there are non-null values
                if non_null_count > 0:
                    # Get top 10 categories if there are many
                    if col_data.nunique() > 10:
                        top_categories = col_data.value_counts().nlargest(10).index.tolist()
                        stats["top_categories"] = top_categories
                    else:
                        stats["categories"] = col_data.dropna().unique().tolist()
            else:
                data_type = "string"
                stats = {"unique_values_count": int(col_data.nunique())}
                if non_null_count > 0:
                    stats["max_length"] = int(col_data.str.len().max())
            
            # Add column info to schema
            schema[column] = {
                "data_type": data_type,
                "stats": stats,
                "null_count": int(col_data.isna().sum()),
                "sample_values": sample_values
            }
            
        return schema
    
    except Exception as e:
        print(f"Error generating schema: {str(e)}")
        return None

def enhance_schema_with_gemini(schema):
    """
    Use Gemini to enhance the schema with additional details.
    """
    try:
        # Prepare the prompt for Gemini which contains {details}
        prompt = f"""
        I have a dataset with the following schema:
        {json.dumps(schema, indent=2)}
        
        For each column in the schema, please enhance it with:
        1. A more detailed description of what the column represents
        2. The domain and realistic ranges for values
        3. Any patterns or constraints that should be maintained in this data
        4. Relationships or correlations to other columns if applicable
        5. Provide more complete statistical information where it makes sense
        
        Return the enhanced schema in valid JSON format that extends the original schema I provided.
        Only add new fields, don't modify the existing ones.
        """
        
        # Generate response from Gemini
        response = model.generate_content(prompt)
        
        # Extract the enhanced schema from the response
        enhanced_schema_text = response.text
        
        # Find JSON in the response
        json_start = enhanced_schema_text.find('{')
        json_end = enhanced_schema_text.rfind('}') + 1
        
        if json_start >= 0 and json_end > json_start:
            enhanced_schema_json = enhanced_schema_text[json_start:json_end]
            try:
                enhanced_schema = json.loads(enhanced_schema_json)
                return enhanced_schema
            except json.JSONDecodeError:
                print("Error decoding JSON from Gemini response")
                return schema
        else:
            print("No valid JSON found in Gemini response")
            return schema
            
    except Exception as e:
        print(f"Error enhancing schema with Gemini: {str(e)}")
        return schema

def generate_synthetic_data(enhanced_schema, num_rows, output_file):
    """
    Use Gemini to generate synthetic data based on the enhanced schema.
    """
    try:
        # Create a prompt for Gemini to generate synthetic data
        prompt = f"""
        I need you to generate {num_rows} rows of synthetic tabular data based on this schema:
        {json.dumps(enhanced_schema, indent=2)}
        
        Important guidelines:
        - Ensure the data follows all patterns, constraints and relationships specified in the schema
        - Make the data realistic and varied, not random
        - Respect the data types and statistical distributions
        - Ensure correlations between columns are maintained
        - Return ONLY valid CSV data without any explanations
        - Include a header row with column names
        - Do not include row numbers as a column
        - The data should be formatted properly as CSV with appropriate delimiters
        
        Generate {num_rows} rows of synthetic data in CSV format:
        """
        
        # Generate response from Gemini
        response = model.generate_content(prompt)
        
        # Extract the CSV data from the response
        csv_text = response.text
        
        # Clean up the response text to get only CSV data
        if "```csv" in csv_text:
            csv_text = csv_text.split("```csv")[1].split("```")[0].strip()
        elif "```" in csv_text:
            csv_text = csv_text.split("```")[1].strip()
        
        # Write the CSV data to the output file
        with open(output_file, 'w', newline='') as f:
            f.write(csv_text)
        
        print(f"Synthetic data generated and saved to {output_file}")
        return True
    
    except Exception as e:
        print(f"Error generating synthetic data: {str(e)}")
        return False

def main():
    # Verify input file exists
    if not os.path.exists(INPUT_CSV):
        print(f"Error: Input file '{INPUT_CSV}' does not exist.")
        return
    
    # Step 1: Generate schema from the CSV file
    print(f"Generating schema from {INPUT_CSV}...")
    schema = read_csv_and_generate_schema(INPUT_CSV)
    if not schema:
        print("Failed to generate schema. Exiting.")
        return
    
    # Step 2: Enhance the schema with Gemini
    print("Enhancing schema with Gemini...")
    enhanced_schema = enhance_schema_with_gemini(schema)
    
    # Save the enhanced schema for reference
    schema_file = f"{os.path.splitext(OUTPUT_CSV)[0]}_schema.json"
    with open(schema_file, 'w') as f:
        json.dump(enhanced_schema, f, indent=2)
    print(f"Enhanced schema saved to {schema_file}")
    
    # Step 3: Generate synthetic data with Gemini
    print(f"Generating {NUM_ROWS} rows of synthetic data...")
    success = generate_synthetic_data(enhanced_schema, NUM_ROWS, OUTPUT_CSV)
    
    if success:
        print("Process completed successfully!")
    else:
        print("Failed to generate synthetic data.")

if __name__ == "__main__":
    main()