import pandas as pd
import json
import os
import csv
import google.generativeai as genai
from google.generativeai import GenerativeModel
import math

# Configuration - EDIT THESE VALUES
INPUT_CSV = "customer_data.csv"  # input CSV file path
OUTPUT_CSV = "synthetic_data.csv"  # output synthetic data path
NUM_ROWS = 100  # Num rows to generate
BATCH_SIZE = 5  # Num columns to process in each API call

# Proxy and API config
os.environ["HTTP_PROXY"] = "http://dc2-proxyuat.seauat.com.vn:8080"
os.environ["HTTPS_PROXY"] = "http://dc2-proxyuat.seauat.com.vn:8080"

# Config Gemini API
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set.")
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")


def read_csv_and_generate_schema(csv_file_path):
    try:
        df = pd.read_csv(csv_file_path)
        
        if df.empty:
            print("Warning: CSV file is empty")
            return {}
        
        schema = {}
        
        # For each column, extract data type and basic statistics
        for column in df.columns:
            col_data = df[column]
            
            # Calculate number of non-null values for sampling
            non_null_count = col_data.count()
            
            sample_values = []
            
            # Only attempt to get sample values if there are non-null values
            if non_null_count > 0:
                # Get a sample of non-null values
                sample_size = min(5, non_null_count)
                sample_values = col_data.dropna().sample(sample_size).tolist() if sample_size > 0 else []
            

            if pd.api.types.is_numeric_dtype(col_data):
                data_type = "integer" if col_data.dropna().apply(lambda x: float(x).is_integer()).all() else "float"
                
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

def enhance_schema_batch_alternative(schema_batch):
    try:
        model = genai.GenerativeModel(model_name="gemini-2.0-flash")
        
        prompt = f"""
        Enhance the schema of these columns from a Vietnamese banking dataset:
        {json.dumps(schema_batch, indent=2)}
        
        Please enhance this schema with your knowledge of financial data patterns and typical constraints or relationships.
        
        For each column, add:
        1. A description field with a detailed description
        2. A domain field with realistic ranges for values
        3. A constraints field with patterns to maintain
        4. A relationships field describing correlations to other columns
        
        Return your response as valid JSON matching the original structure with these additional fields.
        Make sure the response is valid JSON and maintains the original structure exactly.
        """
        
        response = model.generate_content(prompt)
        
        # Parse the response to extract JSON
        response_text = response.text
        # Find JSON in the response
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
        # Return original schema batch if enhancement fails
        return schema_batch

def enhance_schema_batch(schema_batch):
    """
    Enhance a batch of columns from the schema using structured generation.
    """
    try:
        # Create schema property description for each column in this batch
        schema_properties = {}
        required_fields = []
        
        # Build the schema for structured output
        for column_name, column_info in schema_batch.items():
            # Define the stats schema based on the data_type of the column
            stats_schema = {
                "type": "object",
                "properties": {
                    "unique_values_count": {"type": "integer"},
                    "min": {"type": ["number", "string"]},
                    "max": {"type": ["number", "string"]},
                    "mean": {"type": "number"},
                    "categories": {"type": "array", "items": {"type": ["string", "number"]}},
                    "top_categories": {"type": "array", "items": {"type": ["string", "number"]}},
                    "max_length": {"type": "integer"}
                }
            }
            
            # Create structured output description for this column
            schema_properties[column_name] = {
                "type": "object",
                "properties": {
                    "data_type": {"type": "string"},
                    "stats": stats_schema,
                    "null_count": {"type": "integer"},
                    "sample_values": {"type": "array", "items": {"type": ["string", "number", "null"]}},
                    # Enhanced fields
                    "description": {"type": "string", "description": "Detailed description of what this column represents"},
                    "domain": {"type": "string", "description": "Domain and realistic ranges for values"},
                    "constraints": {"type": "string", "description": "Patterns or constraints that should be maintained"},
                    "relationships": {"type": "string", "description": "Relationships or correlations to other columns"}
                },
                "required": ["data_type", "stats", "null_count", "sample_values", "description", "domain", "constraints"]
            }
            required_fields.append(column_name)
        
        # Define the overall schema structure for the response
        structured_schema = {
            "type": "object", 
            "properties": schema_properties,
            "required": required_fields
        }
        
        # Create the model with structured output format
        structured_model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            generation_config={
                "response_mime_type": "application/json",
            }
        )
        
        # Prepare the prompt for Gemini - keep it concise
        prompt = f"""
        Enhance the schema of these columns from a Vietnamese banking dataset:
        {json.dumps(schema_batch, indent=2)}
        
        Please enhance this schema with your knowledge of financial data patterns and typical constraints or relationships.
        
        For each column, add:
        1. A description field with a detailed description
        2. A domain field with realistic ranges for values
        3. A constraints field with patterns to maintain
        4. A relationships field describing correlations to other columns
        """
        
        try:
            # Generate the structured response
            response = structured_model.generate_content(
                prompt,
                generation_config={"response_schema": structured_schema}
            )
            
            # Parse the response
            enhanced_batch = response.candidates[0].content.parts[0].text
            enhanced_batch = json.loads(enhanced_batch)
            
            return enhanced_batch
        except Exception as e:
            print(f"Structured output failed: {str(e)}. Trying alternative approach...")
            return enhance_schema_batch_alternative(schema_batch)
            
    except Exception as e:
        print(f"Error enhancing schema batch: {str(e)}")
        print("Detailed exception info:", str(e.__class__.__name__))
        # Return original schema batch if enhancement fails
        return schema_batch

def enhance_schema_with_gemini(schema):
    """
    Enhance the schema by processing it in batches.
    """
    try:
        # Calculate the number of batches needed
        column_names = list(schema.keys())
        num_columns = len(column_names)
        num_batches = math.ceil(num_columns / BATCH_SIZE)
        
        print(f"Processing {num_columns} columns in {num_batches} batches of {BATCH_SIZE}...")
        
        # Initialize the enhanced schema with a copy of the original
        enhanced_schema = {}
        
        # Process each batch
        for i in range(num_batches):
            start_idx = i * BATCH_SIZE
            end_idx = min((i + 1) * BATCH_SIZE, num_columns)
            batch_columns = column_names[start_idx:end_idx]
            
            print(f"Processing batch {i+1}/{num_batches} with columns: {batch_columns}")
            
            # Create a subset of the schema for this batch
            schema_batch = {col: schema[col] for col in batch_columns}
            
            # Enhance this batch
            try:
                enhanced_batch = enhance_schema_batch(schema_batch)
                
                # Verify that enhanced_batch is a dictionary before updating
                if not isinstance(enhanced_batch, dict):
                    print(f"Warning: Received non-dictionary result for batch {i+1}. Using original schema for this batch.")
                    enhanced_batch = schema_batch
                    
                # Add the enhanced batch to the complete schema
                enhanced_schema.update(enhanced_batch)
            except Exception as batch_error:
                print(f"Error processing batch {i+1}: {str(batch_error)}")
                print("Using original schema for this batch.")
                enhanced_schema.update(schema_batch)
        
        return enhanced_schema
    
    except Exception as e:
        print(f"Error in batch enhancement process: {str(e)}")
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
        - Make the data realistic and varied for a Vietnamese banking context
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
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
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
    
    # Save the basic schema for reference
    basic_schema_file = f"{os.path.splitext(OUTPUT_CSV)[0]}_basic_schema.json"
    with open(basic_schema_file, 'w') as f:
        json.dump(schema, f, indent=2)
    print(f"Basic schema saved to {basic_schema_file}")
    
    # Step 2: Enhance the schema with Gemini in batches
    print("Enhancing schema with Gemini (in batches)...")
    try:
        enhanced_schema = enhance_schema_with_gemini(schema)
        
        # Save the enhanced schema for reference
        enhanced_schema_file = f"{os.path.splitext(OUTPUT_CSV)[0]}_enhanced_schema.json"
        with open(enhanced_schema_file, 'w') as f:
            json.dump(enhanced_schema, f, indent=2)
        print(f"Enhanced schema saved to {enhanced_schema_file}")
    except Exception as e:
        print(f"Failed to enhance schema: {str(e)}")
        print("Using basic schema instead.")
        enhanced_schema = schema
    
    # Step 3: Generate synthetic data with Gemini
    print(f"Generating {NUM_ROWS} rows of synthetic data...")
    success = generate_synthetic_data(enhanced_schema, NUM_ROWS, OUTPUT_CSV)
    
    if success:
        print("Process completed successfully!")
    else:
        print("Failed to generate synthetic data.")

if __name__ == "__main__":
    main()