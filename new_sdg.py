import pandas as pd
import json
import numpy as np
from faker import Faker
import os
import re
import string
from collections import defaultdict
import random
from datetime import datetime, timedelta, date
import math
import time # for potential API rate limiting
import openpyxl

try:
    import rstr
    RSTR_AVAILABLE = True
except ImportError:
    RSTR_AVAILABLE = False
    print("Info: 'rstr' library not found. Will use custom regex generation only.")

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Info: 'google-generativeai' library not found. 'gemini' provider will not function.")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") 
gemini_model = None # Initialize gemini_model to None

if GEMINI_AVAILABLE:
    if GEMINI_API_KEY != "YOUR_GEMINI_API_KEY":
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            gemini_model = genai.GenerativeModel("gemini-1.5-flash") # Use the model specified by user
            print("Gemini API configured successfully.")
        except Exception as e:
            print(f"Warning: Error configuring Gemini API: {e}. 'gemini' provider disabled.")
            GEMINI_AVAILABLE = False # Disable Gemini if configuration fails
    else:
        print("Warning: Gemini API key not configured (using placeholder). 'gemini' provider disabled.")
        GEMINI_AVAILABLE = False


# --- User's Original Regex Generator (Kept Intact) ---
# Global state variables for the custom regex generator
last_element_type = None
last_char_set = []
last_literal_char = ''

def generate_from_regex_pattern(pattern):
    """
    Generates a string based on a regex pattern using custom logic. (fallback)
    """
    global last_element_type, last_char_set, last_literal_char # Use global state

    if not isinstance(pattern, str):
        return None

    # --- Handle Top-Level OR (|) Operator ---
    # (Code exactly as provided by user)
    alternatives = []
    current_alternative_start = 0
    in_char_class = False
    in_repetition_count = False
    i = 0
    while i < len(pattern):
        if pattern[i] == '\\' and i + 1 < len(pattern):
            # Skip escaped characters
            i += 2
            continue
        elif pattern[i] == '[':
            in_char_class = True
            i += 1
        elif pattern[i] == ']':
            in_char_class = False
            i += 1
        elif pattern[i] == '{':
            in_repetition_count = True
            i += 1
        elif pattern[i] == '}':
            in_repetition_count = False
            i += 1
        elif pattern[i] == '|' and not in_char_class and not in_repetition_count:
            # Found a top-level '|', add the current alternative
            alternatives.append(pattern[current_alternative_start:i])
            current_alternative_start = i + 1 # Start the next alternative after '|'
            i += 1
        else:
            i += 1
    alternatives.append(pattern[current_alternative_start:])

    if len(alternatives) > 1:
        chosen_pattern = random.choice(alternatives)
        # Reset state before processing the chosen alternative recursively
        last_element_type = None
        last_char_set = []
        last_literal_char = ''
        # Recursive call using the original function name
        return generate_from_regex_pattern(chosen_pattern)

    pattern_to_process = alternatives[0]

    # --- Reset Global State for the actual generation process ---
    last_element_type = None
    last_char_set = []
    last_literal_char = ''

    # --- Handle Specific Case (from user's code) ---
    if pattern_to_process == r'\d{10}':
        generated_chars = ''.join(random.choice(string.digits) for _ in range(10))
        last_element_type = 'escape_d'
        last_char_set = string.digits # Keep as string
        last_literal_char = ''
        return generated_chars

    # --- Linear Parsing and Generation (User's Original Logic) ---
    result = []
    i = 0
    while i < len(pattern_to_process):
        # --- Handle Escape Sequences ---
        if i + 1 < len(pattern_to_process) and pattern_to_process[i] == '\\':
            escape_char = pattern_to_process[i+1]
            generated_char = ''
            char_set = [] # Keep as list as per original
            element_type = None

            if escape_char == 'd':
                generated_char = random.choice(string.digits)
                char_set = list(string.digits) # Original used list here
                element_type = 'escape_d'
            elif escape_char == 'w':
                generated_char = random.choice(string.ascii_letters + string.digits + '_')
                char_set = list(string.ascii_letters + string.digits + '_') # Original used list
                element_type = 'escape_w'
            elif escape_char == 's':
                generated_char = random.choice(' \t\n\r\f\v')
                char_set = list(' \t\n\r\f\v') # Original used list
                element_type = 'escape_s'
            elif escape_char in '{}[]+*?.|()\\': # Common escaped metacharacters
                generated_char = escape_char
                char_set = []
                element_type = 'literal'
            else: # Unrecognized escape
                result.append('\\')
                result.append(escape_char)
                last_element_type = 'literal'
                last_char_set = []
                last_literal_char = escape_char
                i += 2
                continue

            result.append(generated_char)
            last_element_type = element_type
            last_char_set = char_set # Store the list
            last_literal_char = generated_char if element_type == 'literal' else ''
            i += 2
            continue

        # --- Handle Character Classes (User's Original Logic) ---
        elif pattern_to_process[i] == '[':
            end_bracket = pattern_to_process.find(']', i)
            if end_bracket == -1: # Malformed
                result.append('[')
                last_element_type = 'literal'
                last_char_set = []
                last_literal_char = '['
                i += 1
                continue

            char_class_content = pattern_to_process[i+1:end_bracket]
            chars_in_class = []
            j = 0
            while j < len(char_class_content):
                 # Handle ranges like a-z, 0-9
                 if j + 2 < len(char_class_content) and char_class_content[j+1] == '-':
                      start_char, end_char = char_class_content[j], char_class_content[j+2]
                      if len(start_char) == 1 and len(end_char) == 1 and ord(start_char) <= ord(end_char):
                           chars_in_class.extend([chr(x) for x in range(ord(start_char), ord(end_char) + 1)])
                           j += 3
                      else: # Invalid range
                           chars_in_class.append(char_class_content[j])
                           j += 1
                 else: # Literal character in class
                      chars_in_class.append(char_class_content[j])
                      j += 1

            # Check if followed by repetition {n}
            if end_bracket + 1 < len(pattern_to_process) and pattern_to_process[end_bracket + 1] == '{':
                 last_element_type = 'char_class'
                 last_char_set = chars_in_class
                 last_literal_char = ''
                 i = end_bracket + 1 # Move index to the '{'
                 continue # Handle '{' in next iteration

            # Generate one char if not followed by {n}
            if chars_in_class:
                generated_char = random.choice(chars_in_class)
                result.append(generated_char)
                last_element_type = 'char_class'
                last_char_set = chars_in_class
                last_literal_char = ''
            else: # Empty class []
                 last_element_type = None
                 last_char_set = []
                 last_literal_char = ''

            i = end_bracket + 1
            continue

        # --- Handle Repetition {n}, {n,}, {,m}, {n,m} (User's Original Logic) ---
        elif pattern_to_process[i] == '{':
            open_brace = i
            close_brace = pattern_to_process.find('}', open_brace)
            if close_brace == -1: # Malformed
                result.append('{')
                last_element_type = 'literal'
                last_char_set = []
                last_literal_char = '{'
                i += 1
                continue

            count_spec = pattern_to_process[open_brace+1:close_brace]
            repeat_count = 0
            try:
                if ',' in count_spec:
                    parts = count_spec.split(',')
                    min_count_str = parts[0].strip()
                    max_count_str = parts[1].strip()
                    min_count = int(min_count_str) if min_count_str else 0
                    max_count = int(max_count_str) if max_count_str else min_count + 5
                    if min_count > max_count: min_count, max_count = max_count, min_count # Swap if needed
                    repeat_count = random.randint(min_count, max_count)
                else: # Exact count {n}
                    repeat_count = int(count_spec)
                if repeat_count < 0: raise ValueError("Negative repeat count") # Add check
            except ValueError: # Invalid count spec
                result.append('{')
                result.append(count_spec)
                result.append('}')
                last_element_type = 'literal'
                last_char_set = []
                last_literal_char = '}'
                i = close_brace + 1
                continue

            # Determine what to repeat based on the previous element
            char_set_for_repetition = []
            is_literal_repetition = False
            element_to_repeat = ''

            if last_element_type in ['escape_d', 'escape_w', 'escape_s', 'char_class']:
                 char_set_for_repetition = last_char_set # Use the stored set (should be list)
                 is_literal_repetition = False
            elif last_element_type == 'literal':
                 if last_literal_char: # Ensure there was a stored literal
                      element_to_repeat = last_literal_char
                      is_literal_repetition = True
                 else: # No preceding element to repeat
                      result.append('{') # Treat { as literal
                      last_element_type = 'literal'
                      last_char_set = []
                      last_literal_char = '{'
                      i += 1
                      continue
            else: # No preceding element to repeat
                result.append('{') # Treat { as literal
                last_element_type = 'literal'
                last_char_set = []
                last_literal_char = '{'
                i += 1
                continue


            # Generate the repeated characters
            generated_chars = []
            if not is_literal_repetition:
                 if char_set_for_repetition: # Handle empty char set
                      for _ in range(repeat_count):
                           generated_chars.append(random.choice(char_set_for_repetition))
                 # else: generate nothing for empty set repetition
            elif is_literal_repetition:
                 generated_chars.append(element_to_repeat * repeat_count)

            result.extend(generated_chars)

            i = close_brace + 1
            continue

        # --- Handle other repetition operators: +, *, ? ---
        elif pattern_to_process[i] in '+*?':
             operator = pattern_to_process[i]
             if last_element_type is None: # Operator at start or after nothing generated
                 i += 1
                 continue # Ignore operator

             # Determine what to repeat
             char_set_for_repetition = []
             is_literal_repetition = False
             element_to_repeat = ''
             if last_element_type in ['escape_d', 'escape_w', 'escape_s', 'char_class']:
                  char_set_for_repetition = last_char_set # Use the stored set (list)
                  is_literal_repetition = False
             elif last_element_type == 'literal':
                  if last_literal_char:
                       element_to_repeat = last_literal_char
                       is_literal_repetition = True
                  else: # Should not happen
                       i += 1
                       continue

             # Apply the operator to the last generated element
             if operator == '?': # Zero or one
                 if random.random() < 0.5 and result:
                     if len(result) > 0: result.pop() # Remove the last character generated

             elif operator == '*': # Zero or more
                 repeats = random.randint(0, 5)
                 if not is_literal_repetition:
                      if char_set_for_repetition:
                           for _ in range(repeats):
                                result.append(random.choice(char_set_for_repetition))
                 elif is_literal_repetition:
                      result.extend([element_to_repeat] * repeats)

             elif operator == '+': # One or more
                 repeats = random.randint(0, 4)
                 if not is_literal_repetition:
                      if char_set_for_repetition:
                           for _ in range(repeats):
                                result.append(random.choice(char_set_for_repetition))
                 elif is_literal_repetition:
                      result.extend([element_to_repeat] * repeats)

             # Keep last_element_type etc. as they were
             i += 1
             continue

        # --- Handle Regular Character ---
        else:
            literal_char = pattern_to_process[i]
            result.append(literal_char)
            last_element_type = 'literal'
            last_char_set = []
            last_literal_char = literal_char # Store the literal
            i += 1
            continue

    return ''.join(result)


# --- Added: Combined Regex Generation Function ---
def generate_from_regex(pattern):
    """
    Generates a string from a regex pattern, trying rstr first and falling back to the original custom generator.
    """
    if not isinstance(pattern, str):
        print(f"Warning: Regex pattern must be a string, got {type(pattern)}")
        return None

    if RSTR_AVAILABLE:
        try:
            generated_value = rstr.xeger(pattern)
            # Basic validation (rstr might produce empty strings)
            if generated_value is not None: # Check for None explicitly
                 return generated_value
            else:
                 pass # Fall through to custom if rstr returns None or empty 
        except Exception as e:
            print(f"Info: rstr failed for pattern '{pattern}' ({e}). Falling back to custom generator.")

    try:
        # Call the original function directly
        return generate_from_regex_pattern(pattern)
    except Exception as e:
        print(f"Error in custom regex generator fallback for pattern '{pattern}': {e}")
        return f"regex_fallback_error_{random.randint(100, 999)}" # Return placeholder on error


class SyntheticDataGenerator:
    def __init__(self, schema_path, csv_path, num_rows=1000):
        """ Initialize the synthetic data generator. """
        self.schema_path = schema_path
        self.csv_path = csv_path
        self.NUM_ROWS = num_rows
        self.faker = Faker(['vi_VN']) # Using Vietnamese locale

        # Load schema and original data
        self.schema = self._load_json(schema_path)
        self.original_data = self._load_csv(csv_path)

        # Initialize variables 
        self.relationships = {}
        self.fd_relationships = {}
        self.o2o_relationships = {}
        self.value_relationships = {}
        self.temporal_relationships = {}
        self.column_unique_counts = {}
        self.synthetic_data = {}
        self.column_unique_values = {}

        # Extract column order from original data
        self.column_order = list(self.original_data.columns)

        # --- Added: Gemini specific initialization ---
        self.gemini_model = gemini_model # Use the globally configured model 
        self.GEMINI_BATCH_SIZE = 15 # Max items per Gemini API call

        print(f"Initialized generator with {len(self.schema['columns'])} columns")
        print(f"Original data shape: {self.original_data.shape}")
        print(f"Target synthetic rows: {self.NUM_ROWS}")
        # --- Added: Print status of optional libraries ---
        print(f"rstr available: {RSTR_AVAILABLE}")
        print(f"Gemini available: {GEMINI_AVAILABLE}")


    def _load_json(self, file_path):
        """Load and parse JSON file (Original)"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            raise Exception(f"Failed to load schema from {file_path}: {str(e)}")

    def _load_csv(self, file_path):
        """Load CSV or Excel file into DataFrame (Original)"""
        try:
            file_extension = os.path.splitext(file_path)[1].lower()
            print(f"Detected file format: {file_extension}") 

            if file_extension == '.xlsx' or file_extension == '.xls':
                return pd.read_excel(file_path)
            else: # Assumes CSV otherwise
                # Keep original simple read_csv
                return pd.read_csv(file_path)
        except Exception as e:
            raise Exception(f"Failed to load data from {file_path}: {str(e)}")

    def step0_map_relationships(self):
        """
        Step 0: Sort through relationships and create mappings.
        """
        print("\n--- STEP 0: Mapping Relationships ---")

        for col_name, col_info in self.schema['columns'].items():
            post_processing_rules = col_info.get('post_processing_rules', [])
            self.column_unique_counts[col_name] = col_info['stats'].get('unique_count', 0)

            for rule in post_processing_rules:
                related_col = rule.get('column')
                rel_type = rule.get('type')
                if not related_col or not rel_type: continue

                if rel_type == 'functional_dependency':
                    if col_name not in self.fd_relationships: self.fd_relationships[col_name] = []
                    self.fd_relationships[col_name].append(related_col)
                    print(f"  FD: {col_name} -> {related_col}") 

                elif rel_type == 'one_to_one':
                    if col_name not in self.o2o_relationships: self.o2o_relationships[col_name] = []
                    self.o2o_relationships[col_name].append(related_col)
                    print(f"  O2O: {col_name} <-> {related_col}") 

                elif rel_type == 'value_relationship':
                    relationship = rule.get('relationship')
                    if not relationship: continue
                    key = f"{col_name}_{related_col}"
                    self.value_relationships[key] = {'source': col_name, 'target': related_col, 'relationship': relationship}
                    print(f"  Value: {col_name} {relationship} {related_col}")

                elif rel_type == 'temporal_relationship':
                    relationship = rule.get('relationship')
                    if not relationship: continue
                    key = f"{col_name}_{related_col}"
                    self.temporal_relationships[key] = {'source': col_name, 'target': related_col, 'relationship': relationship}
                    print(f"  Temporal: {col_name} {relationship} {related_col}") 

        self.relationships = {
            'fd': self.fd_relationships, 'o2o': self.o2o_relationships,
            'value': self.value_relationships, 'temporal': self.temporal_relationships
        }
        print(f"  Found {len(self.fd_relationships)} columns with FD relationships")
        print(f"  Found {len(self.o2o_relationships)} columns with O2O relationships")
        print(f"  Found {len(self.value_relationships)} value relationships")
        print(f"  Found {len(self.temporal_relationships)} temporal relationships")
        return self.relationships

    # --- Custom Providers  ---
    def _create_date_provider(self, faker_args):
        """custom date provider that handles Faker's date method parameter"""
        date_start = faker_args.get('start_date', '-30y')
        date_end = faker_args.get('end_date', 'today')
        def date_provider():
            try:
                start = datetime.strptime(date_start, '%Y-%m-%d').date() if isinstance(date_start, str) and re.match(r'\d{4}-\d{2}-\d{2}', date_start) else date_start
                end = datetime.strptime(date_end, '%Y-%m-%d').date() if isinstance(date_end, str) and re.match(r'\d{4}-\d{2}-\d{2}', date_end) else date_end
                return self.faker.date_between(start_date=start, end_date=end)
            except Exception as e:
                print(f"Error in date provider: {str(e)}")
                return datetime.now().date() # Fallback
        return date_provider

    def _create_random_number_provider(self, faker_args):
        """Create a custom random number provider with correct parameter handling"""
        min_value = faker_args.get('min', 0)
        max_value = faker_args.get('max', 100)
        def number_provider():
            try:
                return self.faker.random_int(min=min_value, max=max_value) 
            except Exception as e:
                print(f"Error in random_number provider: {str(e)}")
                return random.randint(min_value, max_value) # Fallback
        return number_provider

    def _create_pyfloat_provider(self, faker_args):
        """Create a custom float provider with correct parameter handling (Original)"""
        min_value = faker_args.get('min', 0.0) 
        max_value = faker_args.get('max', 100.0)
        right_digits = faker_args.get('right_digits', 2)
        positive = faker_args.get('positive', None)
        MAX_TOTAL_DIGITS = 15 # Faker limit

        def float_provider():
            try:
                 # calculation of left_digits
                if max_value is not None:
                    if max_value > 0: left_digits = len(str(int(max_value)))
                    else: left_digits = 1
                else: left_digits = 5

                if left_digits + right_digits > MAX_TOTAL_DIGITS:
                    # fallback
                    return min_value + (random.random() * (max_value - min_value))

                # kwargs structure
                kwargs = { 'left_digits': left_digits, 'right_digits': right_digits, 'positive': positive }
                if min_value is not None: kwargs['min_value'] = min_value
                if max_value is not None: kwargs['max_value'] = max_value

                return self.faker.pyfloat(**kwargs)
            except Exception as e:
                print(f"Error in pyfloat provider: {str(e)}")
                # fallback
                return round(min_value + (random.random() * (max_value - min_value)), right_digits)
        return float_provider

    # --- Modified: _get_faker_provider ---
    def _get_faker_provider(self, column_name):
        """Get the appropriate Faker provider or special handler for a column"""
        col_info = self.schema['columns'].get(column_name, {})
        provider_name = col_info.get('faker_provider')
        faker_args = col_info.get('faker_args', {})

        if not provider_name:
            return lambda: None

        # --- Added: Handle 'gemini' provider ---
        if provider_name == 'gemini':
            if not GEMINI_AVAILABLE:
                print(f"Error: 'gemini' provider requested for {column_name}, but Gemini API is not available/configured.")
                return lambda: f"gemini_disabled_{random.randint(100, 999)}" # Return error placeholder
            # Return the marker for _generate_unique_values to handle
            return 'gemini_provider_marker'

        # --- Modified: Handle 'regexify' provider ---
        if provider_name == 'regexify':
            pattern = faker_args.get('pattern')
            if pattern is None:
                print(f"Warning: 'regexify' provider specified for {column_name} but no 'pattern' found in faker_args. Returning None provider.")
                return lambda: None
            print(f"  Using regex generator for {column_name} with pattern: {pattern}") 
            # Return a function that calls the combined regex generator (rstr -> fallback)
            return lambda: generate_from_regex(pattern) # Use the new combined function

        # --- Handle Custom Providers (Original Logic) ---
        if provider_name in ['date', 'date_between', 'date_between_dates', 'date_time']:
            print(f"Creating custom date provider for {column_name}")
            return self._create_date_provider(faker_args)

        if provider_name in ['random_int', 'random_number']:
            print(f"Creating custom random_number provider for {column_name}") 
            return self._create_random_number_provider(faker_args)

        if provider_name in ['pyfloat', 'random_float']:
            print(f"Creating custom pyfloat provider for {column_name}") 
            return self._create_pyfloat_provider(faker_args)

        if provider_name == 'date_time_between_dates': 
            start_date = faker_args.get('start_date', '-30y')
            end_date = faker_args.get('end_date', 'now')
            def datetime_provider():
                try:
                    return self.faker.date_time_between(start_date=start_date, end_date=end_date)
                except Exception as e:
                    print(f"Error in date_time_between provider: {str(e)}")
                    return datetime.now() # Fallback
            return datetime_provider

        # Handle unique providers 
        is_unique = False
        if provider_name.startswith('unique.'):
            is_unique = True
            provider_name = provider_name[7:] # Remove 'unique.' prefix

        # --- Default Faker Provider Handling  ---
        try:
            provider_obj = self.faker
            if '.' in provider_name:
                provider_parts = provider_name.split('.')
                for part in provider_parts:
                    provider_obj = getattr(provider_obj, part)
                provider_method = provider_obj
            else:
                provider_method = getattr(self.faker, provider_name)

            def provider_func():
                try:
                    # safe_args filtering
                    safe_args = {k: v for k, v in faker_args.items() if not k.startswith('_') and k not in ['self', 'cls', 'locale', 'weight']}
                    return provider_method(**safe_args)
                except Exception as e:
                    print(f"Error using provider {provider_name} for {column_name}: {str(e)}")
                    # fallback logic
                    data_type = col_info.get('data_type', 'text')
                    if data_type == 'integer': return random.randint(0, 1000)
                    elif data_type == 'float': return random.random() * 1000
                    elif data_type == 'datetime': return self.faker.date_time_this_decade()
                    else: return f"unknown_{column_name}_{random.randint(1, 1000)}"
            return provider_func

        except Exception as e:
            # error handling
            print(f"Failed to get provider for {column_name} ({provider_name}): {str(e)}")
            return lambda: f"error_{column_name}_{random.randint(1, 1000)}"


    # --- Added: Gemini Value Generation ---
    def _generate_gemini_values(self, column_name, target_count):
        """Generates unique values using the Gemini API in batches."""
        if not self.gemini_model: # Check if model was initialized
            print(f"Error: Gemini model not available. Cannot generate values for {column_name}.")
            # Return placeholders matching the target count
            return [f"gemini_error_disabled_{i}" for i in range(target_count)]

        col_info = self.schema['columns'].get(column_name, {})
        faker_args = col_info.get('faker_args', {})
        base_prompt = faker_args.get('prompt')
        samples = faker_args.get('sample_values')

        if not base_prompt:
            print(f"Error: 'gemini' provider for {column_name} requires a 'prompt' in 'faker_args'.")
            return [f"gemini_error_no_prompt_{i}" for i in range(target_count)]

        print(f"  Generating {target_count} unique values for {column_name} using Gemini...")

        unique_values = set()
        generated_count = 0
        max_api_attempts_per_batch = 3 # How many times to retry a failing API call
        total_batches = math.ceil(target_count / self.GEMINI_BATCH_SIZE)
        batch_num = 0

        while generated_count < target_count:
            batch_size = min(self.GEMINI_BATCH_SIZE, target_count - generated_count)
            if batch_size <= 0: break

            batch_num += 1
            print(f"    Batch {batch_num}/{total_batches}: Requesting {batch_size} values...")

            # Construct prompt for the batch
            # Keep it simple: ask for a list, one item per line.
            prompt = f"{base_prompt} based on {samples}. Generate {batch_size} unique examples of this. Provide the output as a simple list, with each item on a new line."
            # Optional: Add context of already generated values to encourage more uniqueness
            # if unique_values:
            #    context = list(unique_values)[-5:] # Show last 5 generated
            #    prompt += f"\nAvoid generating examples similar to these: {context}"

            api_attempts = 0
            success = False
            while api_attempts < max_api_attempts_per_batch and not success:
                api_attempts += 1
                try:
                    # Make the API call
                    response = self.gemini_model.generate_content(prompt)
                    raw_text = response.text.strip()

                    # Simple parsing: split by newline and strip whitespace/markers
                    batch_values = []
                    for line in raw_text.split('\n'):
                        cleaned_line = line.strip()
                        # Remove potential list markers (like '-', '*', '1.', etc.) at the start
                        cleaned_line = re.sub(r'^\s*[-\*\d]+\.?\s*', '', cleaned_line)
                        if cleaned_line: # Add if not empty after cleaning
                            batch_values.append(cleaned_line)

                    print(f"    Batch {batch_num} received {len(batch_values)} potential values.")

                    # Add successfully parsed values to the main set
                    newly_added_count = 0
                    for val in batch_values:
                        if val not in unique_values:
                            # Check if we still need more values overall
                            if generated_count < target_count:
                                unique_values.add(val)
                                generated_count += 1
                                newly_added_count += 1
                            else:
                                break # Stop adding if we've reached the global target

                    print(f"    Batch {batch_num} added {newly_added_count} new unique values (Total unique: {len(unique_values)}).")
                    success = True # Mark batch API call as successful
                    print("sleep 2s")
                    time.sleep(2)

                except Exception as e:
                    print(f"Error calling Gemini API (Attempt {api_attempts}/{max_api_attempts_per_batch}): {e}")
                    if api_attempts < max_api_attempts_per_batch:
                        wait_time = 2 ** api_attempts # Exponential backoff
                        print(f"      Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                    else:
                        print(f"Error: Max retries reached for Gemini batch {batch_num}. Skipping batch.")
                        # Add placeholder errors for this batch? Or just continue? Let's continue.
                        break # Stop trying for this batch

            if generated_count >= target_count:
                 print("    Target count reached.")
                 break # Exit outer loop if target reached

        if generated_count < target_count:
            print(f"  Warning: Gemini generation finished. Could only generate {generated_count}/{target_count} unique values for {column_name}")
            # Fill remaining with placeholders if needed? Keep original behavior (just return what was generated)

        return list(unique_values)


    # --- Modified: _generate_unique_values ---
    def _generate_unique_values(self, column_name, target_count):
        """Generate unique values for a column using appropriate provider (Faker/Regex/Gemini)."""

        provider = self._get_faker_provider(column_name)

        # --- Added: Handle Gemini Provider Marker ---
        if provider == 'gemini_provider_marker':
            # Call the dedicated Gemini generation function
            return self._generate_gemini_values(column_name, target_count)

        # --- Handle Standard Faker/Regex Providers ---
        if not callable(provider): # Check if provider is valid
             print(f"Error: Invalid provider obtained for {column_name}. Type: {type(provider)}")
             # Return placeholders 
             return [f"invalid_provider_err_{i}" for i in range(target_count)]

        # generating unique values with Faker/Regex
        unique_values = set()
        max_attempts = 1000
        attempts = 0

        while len(unique_values) < target_count and attempts < max_attempts:
            value = provider()
            # handling of non-hashable types
            if isinstance(value, (list, np.ndarray)):
                value = str(value)

            unique_values.add(value)
            attempts += 1

        if attempts >= max_attempts and len(unique_values) < target_count:
            print(f"  Warning: Could only generate {len(unique_values)}/{target_count} unique values for {column_name} after {attempts} attempts.")

        return list(unique_values)


    def step1_generate_initial_uniques(self):
        """
        Step 1: Generate initial unique values for each column. 
        """
        print("\n--- STEP 1: Generating Initial Unique Values ---")

        # scaling factor calculation
        scaling_factor = math.ceil(len(self.original_data) / self.NUM_ROWS)
        print("SCALING FACTOR__________________________________________________") 
        print(scaling_factor)
        print(len(self.original_data))
        print(self.NUM_ROWS)

        for column_name in self.schema['columns']:
            unique_count = self.column_unique_counts.get(column_name, 0)

            if unique_count == 0:
                self.column_unique_values[column_name] = []
                continue

            # target count logic
            is_categorical = unique_count <= 20
            if is_categorical:
                target_count = min(unique_count, self.NUM_ROWS)
            else:
                target_count = max(1, int(unique_count / scaling_factor))

            print(f"  Generating {target_count} unique values for {column_name} (original unique: {unique_count})") 

            # --- Modified Call ---
            # Generate unique values using the updated function that handles all providers
            unique_values = self._generate_unique_values(column_name, target_count)
            self.column_unique_values[column_name] = unique_values

            print(f"  Generated {len(unique_values)} unique values for {column_name}") 

        print(f"  Generated unique values for {len(self.column_unique_values)} columns") 
        return self.column_unique_values

    # ========================================================================
    # Steps 2, 3, 4, 5 and helper methods
    # ========================================================================

    def step2_populate_based_on_fd(self):
        """
        Step 2: Populate based on Functional Dependencies (FD).
        """
        print("\n--- STEP 2: Populating Based on Functional Dependencies ---")
        if not self.synthetic_data:
            self.synthetic_data = {col: [None] * self.NUM_ROWS for col in self.column_order}

        for col_name, unique_values in self.column_unique_values.items():
             for i, val in enumerate(unique_values):
                 if i < self.NUM_ROWS:
                     # Check if column exists in synthetic_data (might not if not in column_order)
                     if col_name in self.synthetic_data:
                          self.synthetic_data[col_name][i] = val


        sorted_columns = sorted(
            self.column_unique_counts.keys(),
            key=lambda col: self.column_unique_counts.get(col, 0),
            reverse=True
        )
        processed_columns = set()
        categorical_distributions = {}
        for col_name in self.column_order:
            if self.column_unique_counts.get(col_name, 0) <= 20:
                # Check if column exists before accessing
                if col_name in self.original_data.columns:
                    value_counts = self.original_data[col_name].value_counts(normalize=True).to_dict()
                    categorical_distributions[col_name] = value_counts
                    print(f"  Extracted distribution for categorical column {col_name}") 
                else:
                    print(f"  Warning: Categorical column {col_name} not found in original data for distribution.")


        fd_mappings = self._extract_fd_mappings()

        for source_col in sorted_columns:
            if source_col in processed_columns: continue
            # Check if source_col is actually in the final list of columns
            if source_col not in self.column_order: continue


            print(f"  Processing column {source_col} with {self.column_unique_counts.get(source_col, 0)} unique values")

            dependent_cols = []
            if source_col in self.fd_relationships:
                dependent_cols = self.fd_relationships[source_col]
                print(f"    Found {len(dependent_cols)} dependent columns via FD") 

            o2o_cols = []
            if source_col in self.o2o_relationships:
                o2o_cols = self.o2o_relationships[source_col]
                print(f"    Found {len(o2o_cols)} one-to-one related columns") 

            self._fill_column_blanks(source_col, categorical_distributions)
            processed_columns.add(source_col)

            for target_col in dependent_cols:
                 # Ensure target_col is in the final column list
                if target_col in processed_columns or target_col not in self.column_order: continue

                print(f"    Filling dependent column {target_col} based on FD from {source_col}") 
                self._fill_fd_dependent_column(source_col, target_col, fd_mappings, categorical_distributions)
                processed_columns.add(target_col)

                if target_col in self.o2o_relationships:
                    for o2o_col in self.o2o_relationships[target_col]:
                         # Ensure o2o_col is in the final list and not processed
                        if o2o_col not in processed_columns and o2o_col in self.column_order:
                            print(f"      Filling O2O column {o2o_col} related to {target_col}") 
                            self._fill_o2o_related_column(target_col, o2o_col, fd_mappings)
                            processed_columns.add(o2o_col)

            for o2o_col in o2o_cols:
                 # Ensure o2o_col is in the final list and not processed
                if o2o_col not in processed_columns and o2o_col in self.column_order:
                    print(f"    Filling O2O column {o2o_col} related to {source_col}") 
                    self._fill_o2o_related_column(source_col, o2o_col, fd_mappings)
                    processed_columns.add(o2o_col)

        filled_count = sum(1 for col in self.synthetic_data for val in self.synthetic_data[col] if val is not None)
        total_cells = len(self.synthetic_data) * self.NUM_ROWS if self.NUM_ROWS > 0 else 0 # Avoid division by zero
        fill_percentage = (filled_count / total_cells) * 100 if total_cells > 0 else 0

        print(f"  Step 2 completed: {filled_count}/{total_cells} cells filled ({fill_percentage:.2f}%)") 
        return self.synthetic_data

    def _extract_fd_mappings(self):
        """Extract functional dependency mappings from original data. """
        fd_mappings = {}
        for source_col, target_cols in self.fd_relationships.items():
            if source_col not in self.original_data.columns: continue
            for target_col in target_cols:
                if target_col not in self.original_data.columns: continue
                for _, row in self.original_data.iterrows():
                    source_val = row[source_col]
                    target_val = row[target_col]
                    if pd.isna(source_val) or pd.isna(target_val): continue
                    mapping_key = (source_col, source_val)
                    if mapping_key not in fd_mappings: fd_mappings[mapping_key] = {}
                    if target_col not in fd_mappings[mapping_key]:
                        fd_mappings[mapping_key][target_col] = target_val
        return fd_mappings

    def _fill_column_blanks(self, column, categorical_distributions):
        """Fill blanks in a column using its existing unique values. """
        if column not in self.synthetic_data: return
        unique_values = [val for val in self.synthetic_data[column] if val is not None]
        if not unique_values:
            print(f"    Warning: No unique values available for {column}") 
            # Try using Step 1 generated values as fallback
            unique_values = self.column_unique_values.get(column, [])
            if not unique_values:
                 # Try original data as last resort
                 if column in self.original_data.columns:
                      unique_values = list(self.original_data[column].dropna().unique())
                 if not unique_values:
                      print(f"    Error: Still no values for {column}. Cannot fill blanks.")
                      return # Cannot fill

        is_categorical = column in categorical_distributions
        weights = None
        if is_categorical:
            weights = []
            values_for_weighting = [] # Need to align weights with values
            distribution = categorical_distributions[column]
            total_weight = 0
            for val in unique_values:
                 weight = distribution.get(val, 0) # Use 0 if not in original dist
                 if weight > 0: # Only include values present in original distribution
                     values_for_weighting.append(val)
                     weights.append(weight)
                     total_weight += weight

            if total_weight > 0:
                weights = [w / total_weight for w in weights] # Normalize
                unique_values = values_for_weighting # Use the filtered list for sampling
            else:
                 weights = None # Fallback to uniform if no overlap

        # Fill blanks 
        indices_to_fill = [i for i, v in enumerate(self.synthetic_data[column]) if v is None]
        if not indices_to_fill: return

        if weights and unique_values: # Ensure unique_values is not empty
            try:
                 # Check length mismatch (can happen if unique_values got filtered)
                 if len(weights) == len(unique_values):
                      chosen_values = random.choices(unique_values, weights=weights, k=len(indices_to_fill))
                 else:
                      print(f"    Warning: Weight/value mismatch for {column}. Using uniform sampling.")
                      chosen_values = random.choices(unique_values, k=len(indices_to_fill))
            except ValueError as e: # Handle potential errors in random.choices
                 print(f"    Error during weighted sampling for {column}: {e}. Using uniform.")
                 chosen_values = random.choices(unique_values, k=len(indices_to_fill))
        elif unique_values: # Fallback to uniform random sampling
            chosen_values = random.choices(unique_values, k=len(indices_to_fill))
        else: # Should not happen if checks above work
             print(f"    Error: No values to sample from for {column} during blank filling.")
             return


        for i, idx in enumerate(indices_to_fill):
            self.synthetic_data[column][idx] = chosen_values[i]


    def _fill_fd_dependent_column(self, source_col, target_col, fd_mappings, categorical_distributions):
        """Fill a dependent column based on FD relationship with source column. (Original)"""
        if source_col not in self.synthetic_data or target_col not in self.synthetic_data: return

        # Original logic used synthetic data uniques first
        target_unique_values = [val for val in self.synthetic_data[target_col] if val is not None]
        if not target_unique_values:
            # Fallback to Step 1 uniques
            target_unique_values = self.column_unique_values.get(target_col, [])
            if not target_unique_values:
                 # Fallback to original data uniques
                 if target_col in self.original_data.columns:
                      target_unique_values = list(self.original_data[target_col].dropna().unique())
                 if not target_unique_values:
                      print(f"    Warning: No unique values for FD target {target_col}. Cannot fill.")
                      return


        is_categorical = target_col in categorical_distributions
        local_fd_mapping = {}
        for key, mapping in fd_mappings.items():
            if key[0] == source_col and target_col in mapping:
                local_fd_mapping[key[1]] = mapping[target_col]

        for i in range(self.NUM_ROWS):
            if self.synthetic_data[target_col][i] is not None: continue
            source_val = self.synthetic_data[source_col][i]
            if source_val is None: continue

            if source_val in local_fd_mapping:
                self.synthetic_data[target_col][i] = local_fd_mapping[source_val]
            else:
                if target_unique_values:
                    # Original weighted choice logic if categorical
                    if is_categorical and target_col in categorical_distributions:
                        weights = []
                        values_for_weighting = []
                        distribution = categorical_distributions[target_col]
                        total_weight = 0
                        for val in target_unique_values:
                             weight = distribution.get(val, 0)
                             if weight > 0:
                                 values_for_weighting.append(val)
                                 weights.append(weight)
                                 total_weight += weight

                        if total_weight > 0:
                            weights = [w / total_weight for w in weights]
                            try:
                                 if len(weights) == len(values_for_weighting):
                                      self.synthetic_data[target_col][i] = random.choices(values_for_weighting, weights=weights)[0]
                                 else:
                                      self.synthetic_data[target_col][i] = random.choice(target_unique_values) # Fallback uniform
                            except ValueError:
                                 self.synthetic_data[target_col][i] = random.choice(target_unique_values) # Fallback uniform
                        else:
                            self.synthetic_data[target_col][i] = random.choice(target_unique_values) # Uniform if no overlap
                    else: # Uniform random choice
                        self.synthetic_data[target_col][i] = random.choice(target_unique_values)


    def _fill_o2o_related_column(self, source_col, o2o_col, fd_mappings):
        """Fill a one-to-one related column based on source column. (Original)"""
        if source_col not in self.synthetic_data or o2o_col not in self.synthetic_data: return

        # Original logic used synthetic data uniques first
        o2o_unique_values = [val for val in self.synthetic_data[o2o_col] if val is not None]
        if not o2o_unique_values:
             # Fallback to Step 1 uniques
             o2o_unique_values = self.column_unique_values.get(o2o_col, [])
             if not o2o_unique_values:
                  # Fallback to original data uniques
                  if o2o_col in self.original_data.columns:
                       o2o_unique_values = list(self.original_data[o2o_col].dropna().unique())
                  if not o2o_unique_values:
                       print(f"    Warning: No unique values for O2O target {o2o_col}. Cannot fill.")
                       return


        local_o2o_mapping = {}
        # Original mapping extraction
        if source_col in self.original_data.columns and o2o_col in self.original_data.columns:
             for _, row in self.original_data.iterrows():
                 source_val = row[source_col]
                 o2o_val = row[o2o_col]
                 if pd.isna(source_val) or pd.isna(o2o_val): continue
                 local_o2o_mapping[source_val] = o2o_val # Original just took last mapping seen

        for i in range(self.NUM_ROWS):
            if self.synthetic_data[o2o_col][i] is not None: continue
            source_val = self.synthetic_data[source_col][i]
            if source_val is None: continue

            if source_val in local_o2o_mapping:
                self.synthetic_data[o2o_col][i] = local_o2o_mapping[source_val]
            elif o2o_unique_values: # Fallback to random choice
                self.synthetic_data[o2o_col][i] = random.choice(o2o_unique_values)


    def step3_fill_remaining_blanks(self):
        """Step 3: Fill remaining blanks using sampling and reverse FD relationships."""
        print("\n--- STEP 3: Filling Remaining Blanks ---")
        if not self.synthetic_data:
            # Original didn't have this check, but it's good practice
            print("  Error: Synthetic data not initialized.")
            return {}

        sorted_columns = sorted(
            self.column_unique_counts.keys(),
            key=lambda col: self.column_unique_counts.get(col, 0),
            reverse=True
        )
        reverse_fd = {}
        for source, targets in self.fd_relationships.items():
            for target in targets:
                if target not in reverse_fd: reverse_fd[target] = []
                reverse_fd[target].append(source)

        fd_targets = set()
        for source, targets in self.fd_relationships.items(): fd_targets.update(targets)

        reverse_fd_mappings = self._extract_reverse_fd_mappings()

        for column in sorted_columns:
            if column not in self.synthetic_data: continue

            print(f"  Processing column {column}") 
            blanks = sum(1 for val in self.synthetic_data[column] if val is None)
            if blanks == 0:
                print(f"    No blanks in {column}, skipping") 
                continue
            print(f"    Filling {blanks} remaining blanks in {column}") 

            existing_values = [val for val in self.synthetic_data[column] if val is not None]
            if not existing_values:
                print(f"    Warning: No existing values in {column} to sample from") 
                existing_values = self.column_unique_values.get(column, [])
                if not existing_values and column in self.original_data.columns:
                    existing_values = list(self.original_data[column].dropna().unique())
                if not existing_values:
                    print(f"    Error: Still no values for {column}. Cannot fill.")
                    continue # Skip column


            # Fill blanks 
            indices_to_fill = [i for i, v in enumerate(self.synthetic_data[column]) if v is None]
            chosen_values = random.choices(existing_values, k=len(indices_to_fill)) # Simple random choice

            for i, idx in enumerate(indices_to_fill):
                sampled_value = chosen_values[i]
                self.synthetic_data[column][idx] = sampled_value

                # Process reverse FD 
                if column in reverse_fd:
                    for source_col in reverse_fd[column]:
                         # Check source exists and is blank
                        if source_col in self.synthetic_data and self.synthetic_data[source_col][idx] is None:
                            key = (column, sampled_value)
                            if key in reverse_fd_mappings and source_col in reverse_fd_mappings[key]:
                                self.synthetic_data[source_col][idx] = reverse_fd_mappings[key][source_col]
                            else:
                                source_values = [val for val in self.synthetic_data[source_col] if val is not None]
                                if source_values:
                                    self.synthetic_data[source_col][idx] = random.choice(source_values)

                # Handle O2O 
                if column in self.o2o_relationships:
                    for o2o_col in self.o2o_relationships[column]:
                         # Check o2o exists and is blank
                        if o2o_col in self.synthetic_data and self.synthetic_data[o2o_col][idx] is None:
                            o2o_val = None
                            # Original searched original data here
                            if column in self.original_data.columns and o2o_col in self.original_data.columns:
                                 for _, row in self.original_data.iterrows():
                                     # Check for equality carefully, handling potential type issues
                                     try:
                                         if row[column] == sampled_value and not pd.isna(row[o2o_col]):
                                             o2o_val = row[o2o_col]
                                             break
                                     except TypeError: # Ignore comparison errors between types
                                         pass
                            if o2o_val is not None:
                                self.synthetic_data[o2o_col][idx] = o2o_val
                            else:
                                # Original fallback: sample from o2o column's existing values
                                o2o_values = [val for val in self.synthetic_data[o2o_col] if val is not None]
                                if o2o_values:
                                    self.synthetic_data[o2o_col][idx] = random.choice(o2o_values)


        filled_count = sum(1 for col in self.synthetic_data for val in self.synthetic_data[col] if val is not None)
        total_cells = len(self.synthetic_data) * self.NUM_ROWS if self.NUM_ROWS > 0 else 0
        fill_percentage = (filled_count / total_cells) * 100 if total_cells > 0 else 0
        print(f"  Step 3 completed: {filled_count}/{total_cells} cells filled ({fill_percentage:.2f}%)") 
        return self.synthetic_data

    def _extract_reverse_fd_mappings(self):
        """Extract reverse functional dependency mappings from original data. (Original)"""
        reverse_fd_mappings = {}
        for source_col, target_cols in self.fd_relationships.items():
            if source_col not in self.original_data.columns: continue
            for target_col in target_cols:
                if target_col not in self.original_data.columns: continue
                for _, row in self.original_data.iterrows():
                    source_val = row[source_col]
                    target_val = row[target_col]
                    if pd.isna(source_val) or pd.isna(target_val): continue
                    mapping_key = (target_col, target_val)
                    if mapping_key not in reverse_fd_mappings: reverse_fd_mappings[mapping_key] = {}
                    if source_col not in reverse_fd_mappings[mapping_key]: 
                        reverse_fd_mappings[mapping_key][source_col] = source_val
        return reverse_fd_mappings

    def step4_enforce_value_relationships(self):
        """Step 4: Enforce Value Relationships (VLs) and temporal relationships. """
        print("\n--- STEP 4: Enforcing Value Relationships ---")
        if not self.value_relationships and not self.temporal_relationships:
            print("  No value or temporal relationships defined, skipping step 4") 
            return self.synthetic_data

        all_relationships = {**self.value_relationships, **self.temporal_relationships}
        if not all_relationships: # Should be caught above, but safe check
             print("  No relationships to enforce, skipping step 4")
             return self.synthetic_data

        print(f"  Found {len(all_relationships)} relationships to enforce") 

        involved_columns = set()
        for rel_key, rel_info in all_relationships.items():
            involved_columns.add(rel_info['source'])
            involved_columns.add(rel_info['target'])

        sorted_columns = sorted(
            involved_columns,
            key=lambda col: self.column_unique_counts.get(col, 0)
        )
        print(f"  Processing {len(sorted_columns)} columns in ascending order of unique counts") 

        for column in sorted_columns:
            if column not in self.synthetic_data: continue
            source_relationships = { k: v for k, v in all_relationships.items() if v['source'] == column }
            if not source_relationships: continue

            print(f"  Processing column {column} with {len(source_relationships)} relationships") 

            # Use pd.Series unique for potentially faster unique extraction if column exists
            if column in self.synthetic_data:
                 unique_values = pd.Series(self.synthetic_data[column]).dropna().unique()
                 # Convert back to list/set if needed by later logic
                 unique_values = set(unique_values)
            else:
                 unique_values = set()


            processed_values = set()

            for unique_val in unique_values:
                if unique_val in processed_values or unique_val is None: continue

                rows_with_value = [i for i, v in enumerate(self.synthetic_data[column]) if v == unique_val]
                if not rows_with_value: continue

                new_bounds = {}
                for rel_key, rel_info in source_relationships.items():
                    target_col = rel_info['target']
                    relationship = rel_info['relationship']
                    if target_col not in self.synthetic_data: continue

                    has_fd = False # Original FD check logic
                    if column in self.fd_relationships and target_col in self.fd_relationships[column]: has_fd = True

                    target_values = [self.synthetic_data[target_col][i] for i in rows_with_value if self.synthetic_data[target_col][i] is not None]
                    if not target_values: continue

                    col_schema = self.schema['columns'].get(column, {})
                    col_stats = col_schema.get('stats', {})
                    schema_min = col_stats.get('min')
                    schema_max = col_stats.get('max')
                    min_bound = schema_min
                    max_bound = schema_max
                    data_type = col_schema.get('data_type', 'text')

                    if data_type not in ['integer', 'float', 'datetime', 'date']: continue

                    # bound adjustment logic
                    try:
                        if relationship == '<': max_bound = min(target_values) if target_values else schema_max
                        elif relationship == '>': min_bound = max(target_values) if target_values else schema_min
                        elif relationship == '=':
                            # Ensure all target values are the same for '=' bound
                            if len(set(target_values)) == 1: min_bound = max_bound = target_values[0]
                            else: continue # Cannot set bound if targets differ for '='
                        # Original didn't handle <=, >=, != here
                    except TypeError:
                         print(f"    Warning: Type error comparing bounds for {column} ({data_type}) and {target_col}. Skipping bound for this relationship.")
                         continue # Skip setting bounds if comparison fails


                    # Store bounds (only if they are valid, i.e., not None or based on schema defaults)
                    bound_changed = False
                    if min_bound != schema_min or schema_min is None: bound_changed = True
                    if max_bound != schema_max or schema_max is None: bound_changed = True

                    if bound_changed:
                        # Ensure bounds are comparable before storing
                        try:
                             if min_bound is not None and max_bound is not None and min_bound > max_bound:
                                  # If bounds are invalid, don't store them for this target
                                  print(f"    Warning: Invalid calculated bounds ({min_bound} > {max_bound}) for {column} based on {target_col}. Ignoring.")
                             else:
                                  new_bounds[target_col] = (min_bound, max_bound)
                        except TypeError:
                             print(f"    Warning: Type error comparing final bounds for {column}. Ignoring bounds from {target_col}.")


                if not new_bounds:
                    processed_values.add(unique_val)
                    continue

                # final bound calculation
                final_min = None
                final_max = None
                for _, (min_val, max_val) in new_bounds.items():
                     try:
                         if min_val is not None: final_min = min_val if final_min is None else max(final_min, min_val)
                         if max_val is not None: final_max = max_val if final_max is None else min(final_max, max_val)
                     except TypeError:
                          print(f"    Warning: Type error combining bounds for {column}. Bound may be inaccurate.")


                # check for invalid final bounds
                try:
                    if final_min is not None and final_max is not None and final_min > final_max:
                        print(f"    Warning: Invalid final bounds for {column} value {unique_val}, skipping regeneration")
                        processed_values.add(unique_val)
                        continue
                except TypeError:
                    print(f"    Warning: Type error comparing final min/max bounds for {column}. Proceeding with potentially invalid bounds.")


                new_val = self._generate_within_bounds(column, final_min, final_max)
                if new_val is None:
                    processed_values.add(unique_val)
                    continue

                if new_val in unique_values and new_val != unique_val:
                    print(f"    Warning: Generated value {new_val} already exists in {column}, skipping regeneration")
                    processed_values.add(unique_val)
                    continue

                for i in rows_with_value: self.synthetic_data[column][i] = new_val
                processed_values.add(unique_val)
                if new_val != unique_val:
                    # Need to handle unique_values update carefully if it's a set
                    if unique_val in unique_values: unique_values.remove(unique_val)
                    unique_values.add(new_val)

                print(f"    Regenerated value {unique_val} -> {new_val} in {column}") 

        self._verify_relationships(all_relationships)
        return self.synthetic_data

    def _generate_within_bounds(self, column, min_val, max_val):
        """Generate a value within the specified bounds for a column. (Original)"""
        col_schema = self.schema['columns'].get(column, {})
        data_type = col_schema.get('data_type', 'text')

        if data_type == 'integer':
            # default bounds
            low = int(min_val if min_val is not None else -10000)
            high = int(max_val if max_val is not None else 10000)
            if low > high: return None 
            # Handle case where low == high
            if low == high: return low
            return random.randint(low, high)

        elif data_type == 'float':
            low = float(min_val if min_val is not None else -10000.0)
            high = float(max_val if max_val is not None else 10000.0)
            if low > high: return None 
            # Handle case where low == high
            if low == high: return low
            precision = col_schema.get('faker_args', {}).get('right_digits', 2)
            return round(low + (high - low) * random.random(), precision)

        elif data_type in ['date', 'datetime']:
            try:
                # date conversion logic (simplified)
                def to_datetime_original(val, default):
                     if isinstance(val, str):
                          try: return datetime.strptime(val, '%Y-%m-%d') 
                          except: return default
                     if isinstance(val, (datetime, date)): return val
                     return default

                start_dt = to_datetime_original(min_val, datetime.now() - timedelta(days=365))
                end_dt = to_datetime_original(max_val, datetime.now())

                if isinstance(start_dt, type(date(1,1,1))): start_dt = datetime.combine(start_dt, datetime.min.time())
                if isinstance(end_dt, type(date(1,1,1))): end_dt = datetime.combine(end_dt, datetime.max.time().replace(microsecond=0))


                if start_dt > end_dt: return None 

                delta_seconds = int((end_dt - start_dt).total_seconds())
                if delta_seconds <= 0: return start_dt # Return start if delta is 0 or negative

                random_seconds = random.randint(0, delta_seconds)
                generated_date = start_dt + timedelta(seconds=random_seconds)

                if data_type == 'date': return generated_date.date()
                return generated_date
            except Exception as e:
                print(f"    Error generating date/datetime value: {str(e)}") 
                return None
        else:
            return None 

    def _verify_relationships(self, relationships):
        """Verify that all relationships are satisfied after regeneration. (Original)"""
        print("  Verifying relationships...") 
        violations = 0
        for rel_key, rel_info in relationships.items():
            source_col = rel_info['source']
            target_col = rel_info['target']
            relationship = rel_info['relationship']
            if source_col not in self.synthetic_data or target_col not in self.synthetic_data: continue

            for i in range(self.NUM_ROWS):
                source_val = self.synthetic_data[source_col][i]
                target_val = self.synthetic_data[target_col][i]
                if source_val is None or target_val is None: continue

                violated = False
                try: # Add try-except for comparison errors
                    if relationship == '<': violated = not (source_val < target_val)
                    elif relationship == '>': violated = not (source_val > target_val)
                    elif relationship == '=': violated = not (source_val == target_val)
                except TypeError:
                     # Ignore comparison errors between incompatible types
                     violated = False # Treat as not violated if comparison fails

                if violated: violations += 1

        if violations > 0:
            print(f"  Found {violations} relationship violations after regeneration") 
        else:
            print("  All relationships satisfied") 

    def step5_add_nulls(self):
        """Step 5: Add nulls based on the null percentage in the original data."""
        print("\n--- STEP 5: Adding Nulls ---")
        if not self.synthetic_data:
            print("  No synthetic data to add nulls to, skipping step 5") 
            return self.synthetic_data

        total_nulls_added = 0
        for column in self.column_order:
            if column not in self.synthetic_data: continue
            col_schema = self.schema['columns'].get(column, {})
            col_stats = col_schema.get('stats', {})
            null_percentage = col_stats.get('null_percentage', 0)
            if null_percentage <= 0: continue

            print(f"  Adding nulls to {column} (null percentage: {null_percentage:.2f}%)")

            target_null_count = int(self.NUM_ROWS * (null_percentage / 100))
            # null counting logic
            existing_nulls = sum(1 for val in self.synthetic_data[column] if val is None)
            additional_nulls = max(0, target_null_count - existing_nulls)

            if additional_nulls <= 0:
                print(f"    Already have enough nulls in {column}, skipping") 
                continue

            print(f"    Adding {additional_nulls} nulls to {column}") 

            non_null_indices = [i for i, v in enumerate(self.synthetic_data[column]) if v is not None]
            if not non_null_indices: continue

            if additional_nulls > len(non_null_indices):
                additional_nulls = len(non_null_indices)

            null_indices = random.sample(non_null_indices, additional_nulls)
            for i in null_indices:
                self.synthetic_data[column][i] = None
            total_nulls_added += additional_nulls

        print(f"  Added {total_nulls_added} nulls across all columns") 
        return self.synthetic_data

    def save_synthetic_data(self, output_path):
        """
        Save synthetic data to CSV or Excel file, ensuring date format.
        """
        # Create a DataFrame from the synthetic data
        result_df = pd.DataFrame(self.synthetic_data)

        # Reorder columns to match original data
        ordered_cols = [col for col in self.column_order if col in result_df.columns]
        result_df = result_df[ordered_cols]

        # --- Add this section to format date columns ---
        for col_name in result_df.columns:
            # Check schema to see if the column is intended to be a date
            col_info = self.schema['columns'].get(col_name, {})
            data_type = col_info.get('data_type')

            # If it's a date or datetime column, format it
            if data_type in ['date', 'datetime']:
                # Ensure the column contains datetime-like objects before formatting
                # This prevents errors if the column somehow ended up with non-date data
                if pd.api.types.is_datetime64_any_dtype(result_df[col_name]) or pd.api.types.is_object_dtype(result_df[col_name]):
                     try:
                         # Attempt to convert to datetime and then format to string
                         result_df[col_name] = pd.to_datetime(result_df[col_name]).dt.strftime('%Y-%m-%d')
                     except Exception as e:
                         print(f"Warning: Could not format date column {col_name}: {e}")
        # -----------------------------------------------

        try:
            # Determine output file type based on extension
            file_extension = os.path.splitext(output_path)[1].lower()

            if file_extension == '.xlsx' or file_extension == '.xls':
                 result_df.to_excel(output_path, index=False)
                 print(f"\nSaved synthetic data to Excel: {output_path}")
            else: # Default to CSV for other extensions
                 # Ensure the directory exists
                 os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
                 result_df.to_csv(output_path, index=False, encoding='utf-8') # Use utf-8 for CSV
                 print(f"\nSaved synthetic data to CSV: {output_path}")

            print(f"Shape: {result_df.shape}")
            return result_df

        except Exception as e:
             print(f"\nError saving data to {output_path}: {e}")
             return None

# --- Main Function ---
def main():
    """Main function to run the synthetic data generator"""

    schema = 'enhanced_schema2.json'
    #csv = 'customer_data2.xlsx'
    csv = 'dim_retail_casa_sample_data.csv'
    rows = 1000
    output = 'synth3.xlsx' 

    # Initialize generator
    print("lets start") 
    try: # Add basic try/except around initialization
        generator = SyntheticDataGenerator(schema, csv, rows)
    except Exception as e:
        print(f"Error initializing generator: {e}")
        return

    # Run steps 
    try: 
        generator.step0_map_relationships()
        generator.step1_generate_initial_uniques()
        generator.step2_populate_based_on_fd()
        generator.step3_fill_remaining_blanks()
        generator.step4_enforce_value_relationships()
        generator.step5_add_nulls()

        # Save results
        generator.save_synthetic_data(output)
        print("\nSynthetic data generation process completed.") 
    except Exception as e:
         print(f"\nAn error occurred during data generation steps: {e}")
         import traceback
         traceback.print_exc()


if __name__ == "__main__":
    main()
