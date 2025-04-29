import pandas as pd
import json
import numpy as np
from faker import Faker
import os
import re
import string
from collections import defaultdict
import random
from datetime import datetime, timedelta
import rstr

class SyntheticDataGenerator:
    def __init__(self, schema_path, csv_path, num_rows=1000):
        """ Initialize the synthetic data generator. """
        self.schema_path = schema_path
        self.csv_path = csv_path
        self.NUM_ROWS = num_rows
        self.faker = Faker()
        
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
        
        print(f"Initialized generator with {len(self.schema['columns'])} columns")
        print(f"Original data shape: {self.original_data.shape}")
        print(f"Target synthetic rows: {self.NUM_ROWS}")
        
    def _load_json(self, file_path):
        """Load and parse JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            raise Exception(f"Failed to load schema from {file_path}: {str(e)}")
            
    def _load_csv(self, file_path):
        """Load CSV file into DataFrame"""
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise Exception(f"Failed to load data from {file_path}: {str(e)}")
            
    def step0_map_relationships(self):
        """
        Step 0: Sort through relationships and create mappings.  
        Returns:
            dict: Organized relationships by type
        """
        print("\n--- STEP 0: Mapping Relationships ---")
        
        # Process relationships from schema
        for col_name, col_info in self.schema['columns'].items():
            post_processing_rules = col_info.get('post_processing_rules', [])
            
            # Store unique counts for later use
            self.column_unique_counts[col_name] = col_info['stats'].get('unique_count', 0)
            
            for rule in post_processing_rules:
                related_col = rule.get('column')
                rel_type = rule.get('type')
                
                # Skip if missing information
                if not related_col or not rel_type:
                    continue
                
                # Store in appropriate relationship container
                if rel_type == 'functional_dependency':
                    if col_name not in self.fd_relationships:
                        self.fd_relationships[col_name] = []
                    self.fd_relationships[col_name].append(related_col)
                    print(f"  FD: {col_name} -> {related_col}")
                    
                elif rel_type == 'one_to_one':
                    if col_name not in self.o2o_relationships:
                        self.o2o_relationships[col_name] = []
                    self.o2o_relationships[col_name].append(related_col)
                    print(f"  O2O: {col_name} <-> {related_col}")
                    
                elif rel_type == 'value_relationship':
                    relationship = rule.get('relationship')
                    if not relationship:
                        continue
                    
                    key = f"{col_name}_{related_col}"
                    self.value_relationships[key] = {
                        'source': col_name,
                        'target': related_col,
                        'relationship': relationship
                    }
                    print(f"  Value: {col_name} {relationship} {related_col}")
                    
                elif rel_type == 'temporal_relationship':
                    relationship = rule.get('relationship')
                    if not relationship:
                        continue
                    
                    key = f"{col_name}_{related_col}"
                    self.temporal_relationships[key] = {
                        'source': col_name,
                        'target': related_col,
                        'relationship': relationship
                    }
                    print(f"  Temporal: {col_name} {relationship} {related_col}")
        
        # Combine all relationships
        self.relationships = {
            'fd': self.fd_relationships,
            'o2o': self.o2o_relationships,
            'value': self.value_relationships,
            'temporal': self.temporal_relationships
        }
        
        # Print summary
        print(f"  Found {len(self.fd_relationships)} columns with FD relationships")
        print(f"  Found {len(self.o2o_relationships)} columns with O2O relationships")
        print(f"  Found {len(self.value_relationships)} value relationships")
        print(f"  Found {len(self.temporal_relationships)} temporal relationships")
        
        return self.relationships
    
    def _create_date_provider(self, faker_args):
        """Create a custom date provider that handles Faker's date method parameters correctly"""
        # Extract date parameters with appropriate defaults
        date_start = faker_args.get('start_date', '-30y')  # Default: 30 years ago
        date_end = faker_args.get('end_date', 'today')     # Default: today
        
        def date_provider():
            try:
                # Convert string dates to datetime objects if they're in ISO format
                if isinstance(date_start, str) and re.match(r'\d{4}-\d{2}-\d{2}', date_start):
                    start = datetime.strptime(date_start, '%Y-%m-%d').date()
                else:
                    start = date_start
                
                if isinstance(date_end, str) and re.match(r'\d{4}-\d{2}-\d{2}', date_end):
                    end = datetime.strptime(date_end, '%Y-%m-%d').date()
                else:
                    end = date_end
                
                # Generate a date between start and end
                return self.faker.date_between(start_date=start, end_date=end)
            except Exception as e:
                print(f"Error in date provider: {str(e)}")
                # Fallback: return today's date
            return datetime.now().date()
                
        return date_provider
        
    def _create_random_number_provider(self, faker_args):
        """Create a custom random number provider with correct parameter handling"""
        # Extract parameters with sensible defaults
        min_value = faker_args.get('min', 0)  
        max_value = faker_args.get('max', 100)
        
        def number_provider():
            try:
                return self.faker.random_int(min=min_value, max=max_value)
            except Exception as e:
                print(f"Error in random_number provider: {str(e)}")
                # Fallback: random number within range
                return random.randint(min_value, max_value)
                
        return number_provider
        
    def _create_pyfloat_provider(self, faker_args):
        """Create a custom float provider with correct parameter handling"""
        # Extract parameters with sensible defaults
        min_value = faker_args.get('min', 0.0)
        max_value = faker_args.get('max', 100.0)
        right_digits = faker_args.get('right_digits', 2)
        positive = faker_args.get('positive', None)
        MAX_TOTAL_DIGITS = 15
        
        def float_provider():
            try:
                # Calculate required left_digits based on the max_value
                if max_value is not None:
                    # Calculate how many digits are needed for the integer part
                    if max_value > 0:
                        left_digits = len(str(int(max_value)))
                    else:
                        left_digits = 1  # For values between -1 and 0
                else:
                    left_digits = 5  # Default
                
                if left_digits + right_digits > MAX_TOTAL_DIGITS:
                    return min_value + (random.random() * (max_value - min_value))
                
                kwargs = {
                    'left_digits': left_digits,  # Default
                    'right_digits': right_digits,
                    'positive': positive
                }
                
                # Handle min/max values
                if min_value is not None:
                    kwargs['min_value'] = min_value
                if max_value is not None:
                    kwargs['max_value'] = max_value
                    
                return self.faker.pyfloat(**kwargs)
            except Exception as e:
                print(f"Error in pyfloat provider: {str(e)}")
                # Fallback: random float within range
                return round(min_value + (random.random() * (max_value - min_value)), right_digits)
                
        return float_provider

    def _get_faker_provider(self, column_name):
        """Get the appropriate Faker provider for a column"""
        col_info = self.schema['columns'].get(column_name, {})
        provider_name = col_info.get('faker_provider')
        faker_args = col_info.get('faker_args', {})
        
        if not provider_name:
            return lambda: None

        # Handle specific providers with custom implementations
        if provider_name in ['date', 'date_between', 'date_between_dates']:
            print(f"Creating custom date provider for {column_name}")
            return self._create_date_provider(faker_args)
            
        if provider_name in ['random_int', 'random_number']:
            print(f"Creating custom random_number provider for {column_name}")
            return self._create_random_number_provider(faker_args)
            
        if provider_name in ['pyfloat', 'random_float']:
            print(f"Creating custom pyfloat provider for {column_name}")
            return self._create_pyfloat_provider(faker_args)
        
        # Handle date_time_between_dates with custom implementation
        if provider_name == 'date_time_between_dates':
            start_date = faker_args.get('start_date', '-30y')
            end_date = faker_args.get('end_date', 'now')
            
            def datetime_provider():
                try:
                    return self.faker.date_time_between(start_date=start_date, end_date=end_date)
                except Exception as e:
                    print(f"Error in date_time_between provider: {str(e)}")
                    return datetime.now()
                    
            return datetime_provider
            
        # Handle unique providers
        is_unique = False
        if provider_name.startswith('unique.'):
            is_unique = True
            provider_name = provider_name[7:]  # Remove 'unique.' prefix
            
        if provider_name == 'regexify':
            pattern = faker_args.get('pattern')
            if pattern is None:
                print(f"Warning: 'regexify' provider specified for {column_name} but no 'pattern' found in faker_args. Returning None provider.")
                return lambda: None # Return a provider that gives None if no pattern
            print(f"  Using regex generator for {column_name} with pattern: {pattern}")
            return lambda: generate_from_regex_pattern(pattern) # Return function that calls the regex generator

        try:
            # Get the faker method
            if '.' in provider_name:
                # Handle nested providers like 'person.name'
                provider_parts = provider_name.split('.')
                provider_obj = self.faker
                for part in provider_parts:
                    provider_obj = getattr(provider_obj, part)
                provider_method = provider_obj
            else:
                provider_method = getattr(self.faker, provider_name)
                
            # Create a function that calls the provider with the right args
            def provider_func():
                try:
                    # For safety, filter out any args that might cause issues
                    safe_args = {k: v for k, v in faker_args.items() 
                                if not k.startswith('_') and k not in ['self', 'cls']}
                    return provider_method(**safe_args)
                except Exception as e:
                    print(f"Error using provider {provider_name} for {column_name}: {str(e)}")
                    # Fallback based on data type
                    data_type = col_info.get('data_type', 'text')
                    if data_type == 'integer':
                        return random.randint(0, 1000)
                    elif data_type == 'float':
                        return random.random() * 1000
                    elif data_type == 'datetime':
                        return self.faker.date_time_this_decade()
                    else:
                        return f"unknown_{column_name}_{random.randint(1, 1000)}"
            
            return provider_func
            
        except Exception as e:
            print(f"Failed to get provider for {column_name} ({provider_name}): {str(e)}")
            # Return a simple default provider
            return lambda: f"error_{column_name}_{random.randint(1, 1000)}"
            
    def _generate_unique_values(self, column_name, target_count):
        """Generate unique values for a column"""
        provider = self._get_faker_provider(column_name)
        unique_values = set()
        
        # Cap at 10,000 attempts to prevent infinite loops
        max_attempts = 10000
        attempts = 0
        
        while len(unique_values) < target_count and attempts < max_attempts:
            value = provider()
            # Handle non-hashable types like lists or numpy arrays
            if isinstance(value, (list, np.ndarray)):
                value = str(value)
            unique_values.add(value)
            attempts += 1
            
        if attempts >= max_attempts and len(unique_values) < target_count:
            print(f"  Warning: Could only generate {len(unique_values)}/{target_count} unique values for {column_name}")
            
        return list(unique_values)
    
    def step1_generate_initial_uniques(self):
        """
        Step 1: Generate initial unique values for each column.
        Returns:
            dict: Column unique values
        """
        print("\n--- STEP 1: Generating Initial Unique Values ---")
        
        # Calculate scaling factor for columns with >20 unique values
        scaling_factor = len(self.original_data) / self.NUM_ROWS
        print("SCALING FACTOR__________________________________________________")
        print(scaling_factor)

        for column_name in self.schema['columns']:
            # Get unique count from schema
            unique_count = self.column_unique_counts.get(column_name, 0)
            
            # Skip if no unique values needed
            if unique_count == 0:
                self.column_unique_values[column_name] = []
                continue
                
            # Determine target unique count
            is_categorical = unique_count <= 20
            if is_categorical:
                target_count = min(unique_count, self.NUM_ROWS)
            else:
                # Scale unique count based on target row count
                target_count = max(1, int(unique_count / scaling_factor))
                
            print(f"  Generating {target_count} unique values for {column_name} (original unique: {unique_count})")
            
            # Generate unique values
            unique_values = self._generate_unique_values(column_name, target_count)
            self.column_unique_values[column_name] = unique_values
            
            print(f"  Generated {len(unique_values)} unique values for {column_name}")
            
        # Summary
        print(f"  Generated unique values for {len(self.column_unique_values)} columns")
        return self.column_unique_values
    
    def step2_populate_based_on_fd(self):
        """
        Step 2: Populate based on Functional Dependencies (FD).
        Start with columns having highest unique count and fill related columns.
        """
        print("\n--- STEP 2: Populating Based on Functional Dependencies ---")
    
        # Initialize synthetic data structure if not already done
        if not self.synthetic_data:
            self.synthetic_data = {col: [None] * self.NUM_ROWS for col in self.column_order}
    
        # Fill in initial unique values generated in step 1
        for col_name, unique_values in self.column_unique_values.items():
            for i, val in enumerate(unique_values):
                if i < self.NUM_ROWS:
                    self.synthetic_data[col_name][i] = val
    
        # Get columns sorted by unique count (descending)
        sorted_columns = sorted(
            self.column_unique_counts.keys(),
            key=lambda col: self.column_unique_counts.get(col, 0),
            reverse=True
        )
    
        # Track which columns have been processed
        processed_columns = set()
    
        # Extract value distributions from original data for categorical columns
        categorical_distributions = {}
        for col_name in self.column_order:
            if self.column_unique_counts.get(col_name, 0) <= 20:  # Categorical threshold
                # Calculate value distribution
                value_counts = self.original_data[col_name].value_counts(normalize=True).to_dict()
                categorical_distributions[col_name] = value_counts
                print(f"  Extracted distribution for categorical column {col_name}")
    
        # Track FD mappings observed in original data
        fd_mappings = self._extract_fd_mappings()
    
        # Process columns by unique count (highest first)
        for source_col in sorted_columns:
            # Skip if already processed
            if source_col in processed_columns:
                continue
        
            print(f"  Processing column {source_col} with {self.column_unique_counts.get(source_col, 0)} unique values")
        
            # Get columns that depend on this column via FD
            dependent_cols = []
            if source_col in self.fd_relationships:
                dependent_cols = self.fd_relationships[source_col]
                print(f"    Found {len(dependent_cols)} dependent columns via FD")
        
            # Get O2O related columns
            o2o_cols = []
            if source_col in self.o2o_relationships:
                o2o_cols = self.o2o_relationships[source_col]
                print(f"    Found {len(o2o_cols)} one-to-one related columns")
        
            # Step 2.1: Fill source column blanks using its existing unique values
            self._fill_column_blanks(source_col, categorical_distributions)
            processed_columns.add(source_col)
        
            # Step 2.2: Fill FD dependent columns using FD mappings
            for target_col in dependent_cols:
                if target_col in processed_columns:
                    continue
                
                print(f"    Filling dependent column {target_col} based on FD from {source_col}")
                self._fill_fd_dependent_column(source_col, target_col, fd_mappings, categorical_distributions)
                processed_columns.add(target_col)
            
                # Also handle O2O relationships of this dependent column
                if target_col in self.o2o_relationships:
                    for o2o_col in self.o2o_relationships[target_col]:
                        if o2o_col not in processed_columns:
                            print(f"      Filling O2O column {o2o_col} related to {target_col}")
                            self._fill_o2o_related_column(target_col, o2o_col, fd_mappings)
                            processed_columns.add(o2o_col)
        
            # Step 2.3: Fill O2O related columns
            for o2o_col in o2o_cols:
                if o2o_col not in processed_columns:
                    print(f"    Filling O2O column {o2o_col} related to {source_col}")
                    self._fill_o2o_related_column(source_col, o2o_col, fd_mappings)
                    processed_columns.add(o2o_col)
    
        # Print summary
        filled_count = sum(1 for col in self.synthetic_data for val in self.synthetic_data[col] if val is not None)
        total_cells = len(self.synthetic_data) * self.NUM_ROWS
        fill_percentage = (filled_count / total_cells) * 100 if total_cells > 0 else 0
    
        print(f"  Step 2 completed: {filled_count}/{total_cells} cells filled ({fill_percentage:.2f}%)")
        return self.synthetic_data

    def _extract_fd_mappings(self):
        """
        Extract functional dependency mappings from original data.
        """
        fd_mappings = {}
    
        # Process each FD relationship
        for source_col, target_cols in self.fd_relationships.items():
            if source_col not in self.original_data.columns:
                continue
            
            for target_col in target_cols:
                if target_col not in self.original_data.columns:
                    continue
                
                # Extract mappings from original data
                for _, row in self.original_data.iterrows():
                    source_val = row[source_col]
                    target_val = row[target_col]
                
                    # Skip if either value is NaN
                    if pd.isna(source_val) or pd.isna(target_val):
                        continue
                
                    # Create mapping key
                    mapping_key = (source_col, source_val)
                
                    # Initialize if key doesn't exist
                    if mapping_key not in fd_mappings:
                        fd_mappings[mapping_key] = {}
                
                    # Add or update target mapping
                    if target_col not in fd_mappings[mapping_key]:
                        fd_mappings[mapping_key][target_col] = target_val
    
        return fd_mappings

    def _fill_column_blanks(self, column, categorical_distributions):
        """
        Fill blanks in a column using its existing unique values.
        If categorical, use the distribution from original data.
        """
        # Skip if column not in synthetic data
        if column not in self.synthetic_data:
            return
        
        # Get existing unique values for this column
        unique_values = [val for val in self.synthetic_data[column] if val is not None]
    
        if not unique_values:
            print(f"    Warning: No unique values available for {column}")
            return
    
        # Determine if categorical and get weights if applicable
        is_categorical = column in categorical_distributions
        weights = None
        if is_categorical:
            # Create weights list aligned with unique values
            weights = []
            for val in unique_values:
                weights.append(categorical_distributions[column].get(val, 1.0))
        
            # Normalize weights if any are available
            if weights and sum(weights) > 0:
                weights = [w/sum(weights) for w in weights]
            else:
                weights = None
    
        # Fill blanks
        for i in range(self.NUM_ROWS):
            if self.synthetic_data[column][i] is None:
                if weights:
                    # Sample based on distribution
                    self.synthetic_data[column][i] = random.choices(unique_values, weights=weights)[0]
                else:
                    # Random sampling
                    self.synthetic_data[column][i] = random.choice(unique_values)

    def _fill_fd_dependent_column(self, source_col, target_col, fd_mappings, categorical_distributions):
        """
    Fill a dependent column based on FD relationship with source column.
    """
        # Skip if either column not in synthetic data
        if source_col not in self.synthetic_data or target_col not in self.synthetic_data:
            return
    
        # Get unique values in target column
        target_unique_values = [val for val in self.synthetic_data[target_col] if val is not None]
    
        # Determine if target is categorical
        is_categorical = target_col in categorical_distributions
    
        # Create local mapping from source values to target values seen in the data
        local_fd_mapping = {}
        for key, mapping in fd_mappings.items():
            if key[0] == source_col and target_col in mapping:
                local_fd_mapping[key[1]] = mapping[target_col]
    
        # Fill target column based on source column values
        for i in range(self.NUM_ROWS):
            # Skip if target already has value
            if self.synthetic_data[target_col][i] is not None:
                continue
            
            source_val = self.synthetic_data[source_col][i]
        
            # Skip if source is None
            if source_val is None:
                continue
            
            # Try to use FD mapping if available
            if source_val in local_fd_mapping:
                self.synthetic_data[target_col][i] = local_fd_mapping[source_val]
            else:
                # No mapping found, use random selection from target unique values
                if target_unique_values:
                    if is_categorical and categorical_distributions[target_col]:
                        # Create weights aligned with unique values
                        weights = []
                        for val in target_unique_values:
                            weights.append(categorical_distributions[target_col].get(val, 1.0))
                    
                        # Normalize weights
                        if sum(weights) > 0:
                            weights = [w/sum(weights) for w in weights]
                            self.synthetic_data[target_col][i] = random.choices(
                                target_unique_values, weights=weights)[0]
                        else:
                            self.synthetic_data[target_col][i] = random.choice(target_unique_values)
                    else:
                        self.synthetic_data[target_col][i] = random.choice(target_unique_values)

    def _fill_o2o_related_column(self, source_col, o2o_col, fd_mappings):
        """
    Fill a one-to-one related column based on source column.
    """
        # Skip if either column not in synthetic data
        if source_col not in self.synthetic_data or o2o_col not in self.synthetic_data:
            return
    
        # Get unique values in o2o column
        o2o_unique_values = [val for val in self.synthetic_data[o2o_col] if val is not None]
    
        # Create local mapping from source values to o2o values seen in the data
        local_o2o_mapping = {}
    
        # Extract mappings from original data
        for _, row in self.original_data.iterrows():
            source_val = row[source_col]
            o2o_val = row[o2o_col]
        
            # Skip if either value is NaN
            if pd.isna(source_val) or pd.isna(o2o_val):
                continue
            
            local_o2o_mapping[source_val] = o2o_val
    
        # Fill o2o column based on source column values
        for i in range(self.NUM_ROWS):
            # Skip if o2o already has value
            if self.synthetic_data[o2o_col][i] is not None:
                continue
            
            source_val = self.synthetic_data[source_col][i]
        
                # Skip if source is None
            if source_val is None:
                continue
            
            # Try to use O2O mapping if available
            if source_val in local_o2o_mapping:
                self.synthetic_data[o2o_col][i] = local_o2o_mapping[source_val]
            elif o2o_unique_values:
                # No mapping found, use random selection from o2o unique values
                self.synthetic_data[o2o_col][i] = random.choice(o2o_unique_values)

    def step3_fill_remaining_blanks(self):
        """
    Step 3: Fill remaining blanks using sampling and reverse FD relationships.
    """
        print("\n--- STEP 3: Filling Remaining Blanks ---")
    
        # Ensure synthetic data structure is initialized
        if not self.synthetic_data:
            self.synthetic_data = {col: [None] * self.NUM_ROWS for col in self.column_order}
    
        # Get columns sorted by original unique count (descending)
        sorted_columns = sorted(
            self.column_unique_counts.keys(),
            key=lambda col: self.column_unique_counts.get(col, 0),
            reverse=True
        )
    
        # Build reverse FD mapping
        reverse_fd = {}
        for source, targets in self.fd_relationships.items():
            for target in targets:
                if target not in reverse_fd:
                    reverse_fd[target] = []
                reverse_fd[target].append(source)
    
        # Identify which columns that are targets of FDs
        fd_targets = set()
        for source, targets in self.fd_relationships.items():
            fd_targets.update(targets)
    
        # Create mappings for reverse FD relationships (target -> source values)
        reverse_fd_mappings = self._extract_reverse_fd_mappings()
    
        # Process columns by unique count (highest first)
        for column in sorted_columns:
            # Skip if column doesn't exist in synthetic data
            if column not in self.synthetic_data:
                continue
            
            print(f"  Processing column {column}")
        
            # Count remaining blanks in this column
            blanks = sum(1 for val in self.synthetic_data[column] if val is None)
        
            if blanks == 0:
                print(f"    No blanks in {column}, skipping")
                continue
            
            print(f"    Filling {blanks} remaining blanks in {column}")
        
            # Get non-null values in this column for sampling
            existing_values = [val for val in self.synthetic_data[column] if val is not None]
        
            if not existing_values:
                print(f"    Warning: No existing values in {column} to sample from")
                continue
        
            # Step 3.1: Fill remaining blanks by sampling from existing values
            for i in range(self.NUM_ROWS):
                if self.synthetic_data[column][i] is None:
                    # Sample a random value
                    sampled_value = random.choice(existing_values)
                    self.synthetic_data[column][i] = sampled_value
                
                    # Step 3.2: Process reverse FD relationships
                    if column in reverse_fd:
                        # This column is a target of some FD relationships
                        for source_col in reverse_fd[column]:
                            # Skip if source column is already filled at this position
                            if self.synthetic_data[source_col][i] is not None:
                                continue
                        
                            # Find source values that map to this target value
                            key = (column, sampled_value)
                            if key in reverse_fd_mappings and source_col in reverse_fd_mappings[key]:
                                # Use mapped source value
                                self.synthetic_data[source_col][i] = reverse_fd_mappings[key][source_col]
                            else:
                                # No mapping found, sample from source column's existing values
                                source_values = [val for val in self.synthetic_data[source_col] if val is not None]
                                if source_values:
                                    self.synthetic_data[source_col][i] = random.choice(source_values)
                
                    # Step 3.3: Handle O2O related columns
                    if column in self.o2o_relationships:
                        for o2o_col in self.o2o_relationships[column]:
                            # Skip if o2o column is already filled at this position
                            if self.synthetic_data[o2o_col][i] is not None:
                                continue
                        
                            # Find corresponding o2o value in original data
                            o2o_val = None
                            for _, row in self.original_data.iterrows():
                                if row[column] == sampled_value and not pd.isna(row[o2o_col]):
                                    o2o_val = row[o2o_col]
                                    break
                        
                            if o2o_val is not None:
                                self.synthetic_data[o2o_col][i] = o2o_val
                            else:
                                # No mapping found, sample from o2o column's existing values
                                o2o_values = [val for val in self.synthetic_data[o2o_col] if val is not None]
                                if o2o_values:
                                    self.synthetic_data[o2o_col][i] = random.choice(o2o_values)
    
        # Print summary
        filled_count = sum(1 for col in self.synthetic_data for val in self.synthetic_data[col] if val is not None)
        total_cells = len(self.synthetic_data) * self.NUM_ROWS
        fill_percentage = (filled_count / total_cells) * 100 if total_cells > 0 else 0
    
        print(f"  Step 3 completed: {filled_count}/{total_cells} cells filled ({fill_percentage:.2f}%)")
        return self.synthetic_data

    def _extract_reverse_fd_mappings(self):
        """
    Extract reverse functional dependency mappings from original data.
    """
        reverse_fd_mappings = {}
    
        # Process each FD relationship
        for source_col, target_cols in self.fd_relationships.items():
            if source_col not in self.original_data.columns:
                continue
            
            for target_col in target_cols:
                if target_col not in self.original_data.columns:
                    continue
                
                # Extract mappings from original data
                for _, row in self.original_data.iterrows():
                    source_val = row[source_col]
                    target_val = row[target_col]
                
                    # Skip if either value is NaN
                    if pd.isna(source_val) or pd.isna(target_val):
                        continue
                
                    # Create reverse mapping key
                    mapping_key = (target_col, target_val)
                
                    # Initialize if key doesn't exist
                    if mapping_key not in reverse_fd_mappings:
                        reverse_fd_mappings[mapping_key] = {}
                
                    # Add or update source mapping
                    if source_col not in reverse_fd_mappings[mapping_key]:
                        reverse_fd_mappings[mapping_key][source_col] = source_val
    
        return reverse_fd_mappings
    
    def step4_enforce_value_relationships(self):
        """
    Step 4: Enforce Value Relationships (VLs) and temporal relationships.
    """
        print("\n--- STEP 4: Enforcing Value Relationships ---")
    
        # Skip if no value relationships or temporal relationships are defined
        if not self.value_relationships and not self.temporal_relationships:
            print("  No value or temporal relationships defined, skipping step 4")
            return self.synthetic_data
    
        # Combine value and temporal relationships for processing
        all_relationships = {**self.value_relationships, **self.temporal_relationships}
    
        if not all_relationships:
            print("  No relationships to enforce, skipping step 4")
            return self.synthetic_data
    
        print(f"  Found {len(all_relationships)} relationships to enforce")
    
        # Get all columns involved in relationships
        involved_columns = set()
        for rel_key, rel_info in all_relationships.items():
            involved_columns.add(rel_info['source'])
            involved_columns.add(rel_info['target'])
    
        # Sort columns by unique count (ascending)
        sorted_columns = sorted(
            involved_columns,
            key=lambda col: self.column_unique_counts.get(col, 0)
        )
    
        print(f"  Processing {len(sorted_columns)} columns in ascending order of unique counts")
    
        # Process each column
        for column in sorted_columns:
            # Skip if column not in synthetic data
            if column not in self.synthetic_data:
                continue
        
            # Get relationships where this column is the source
            source_relationships = {
                k: v for k, v in all_relationships.items() 
                if v['source'] == column
            }
        
            # Skip if no relationships where this column is the source
            if not source_relationships:
                continue
        
            print(f"  Processing column {column} with {len(source_relationships)} relationships")
        
            # Get all unique values in this column
            unique_values = set(self.synthetic_data[column])
            if None in unique_values:
                unique_values.remove(None)
        
            # Track which values have been processed to avoid redundant work
            processed_values = set()
        
            # Process each unique value in the column
            for unique_val in unique_values:
                # Skip if already processed or null
                if unique_val in processed_values or unique_val is None:
                    continue
            
                # Find rows where this value appears
                rows_with_value = [
                    i for i in range(self.NUM_ROWS)
                    if self.synthetic_data[column][i] == unique_val
                ]
            
                if not rows_with_value:
                    continue
            
                # Determine value regeneration bounds for each relationship
                new_bounds = {}  # Store bounds for each target column
            
                for rel_key, rel_info in source_relationships.items():
                    target_col = rel_info['target']
                    relationship = rel_info['relationship']
                
                    # Skip if target column not in synthetic data
                    if target_col not in self.synthetic_data:
                        continue
                
                    # Check if there's an FD from source to target
                    has_fd = False
                    if column in self.fd_relationships and target_col in self.fd_relationships[column]:
                        has_fd = True
                
                    # Collect all target values in rows where source value appears
                    target_values = [
                        self.synthetic_data[target_col][i] 
                        for i in rows_with_value 
                        if self.synthetic_data[target_col][i] is not None
                    ]
                
                    if not target_values:
                        continue
                
                    # Get schema min/max for column (default to None if not available)
                    col_schema = self.schema['columns'].get(column, {})
                    col_stats = col_schema.get('stats', {})
                    schema_min = col_stats.get('min')
                    schema_max = col_stats.get('max')
                
                    # Initialize bounds
                    min_bound = schema_min
                    max_bound = schema_max
                
                    # Get data type
                    data_type = col_schema.get('data_type', 'text')
                
                    # Skip if not numeric or datetime
                    if data_type not in ['integer', 'float', 'datetime', 'date']:
                        continue
                
                    # Adjust bounds based on relationship
                    if relationship == '<':
                        # Source should be less than target
                        min_bound = schema_min
                        max_bound = min(target_values) if min(target_values) is not None else schema_max
                    elif relationship == '>':
                        # Source should be greater than target
                        min_bound = max(target_values) if max(target_values) is not None else schema_min
                        max_bound = schema_max
                    elif relationship == '=':
                        # Source should equal target (only one possible value)
                        if len(set(target_values)) == 1:
                            min_bound = max_bound = target_values[0]
                
                    # Store bounds for this target column
                    if min_bound is not None or max_bound is not None:
                        new_bounds[target_col] = (min_bound, max_bound)
            
                # Skip regeneration if no valid bounds
                if not new_bounds:
                    processed_values.add(unique_val)
                    continue
            
                # Determine the most restrictive bounds across all relationships
                final_min = None
                final_max = None
            
                for _, (min_val, max_val) in new_bounds.items():
                    if min_val is not None:
                        final_min = min_val if final_min is None else max(final_min, min_val)
                    if max_val is not None:
                        final_max = max_val if final_max is None else min(final_max, max_val)
            
                #   Skip if bounds are invalid
                if final_min is not None and final_max is not None and final_min > final_max:
                    print(f"    Warning: Invalid bounds for {column} value {unique_val}, skipping regeneration")
                    processed_values.add(unique_val)
                    continue
            
                # Generate new value within bounds
                new_val = self._generate_within_bounds(column, final_min, final_max)
            
                # Skip if generation failed
                if new_val is None:
                    processed_values.add(unique_val)
                    continue
            
                # Check if new value already exists in the column
                if new_val in unique_values and new_val != unique_val:
                    print(f"    Warning: Generated value {new_val} already exists in {column}, skipping regeneration")
                    processed_values.add(unique_val)
                    continue
            
                # Replace old value with new value in all rows
                for i in rows_with_value:
                    self.synthetic_data[column][i] = new_val
            
                # Update processed values
                processed_values.add(unique_val)
                if new_val != unique_val:
                    unique_values.remove(unique_val)
                    unique_values.add(new_val)
            
                print(f"    Regenerated value {unique_val} -> {new_val} in {column}")
    
        # Check if relationships are satisfied
        self._verify_relationships(all_relationships)
    
        return self.synthetic_data

    def _generate_within_bounds(self, column, min_val, max_val):
        """
    Generate a value within the specified bounds for a column.
    """
        # Get column data type
        col_schema = self.schema['columns'].get(column, {})
        data_type = col_schema.get('data_type', 'text')
    
        # Handle different data types
        if data_type == 'integer':
            # Handle None bounds
            if min_val is None:
                min_val = -10000  # Default min
            if max_val is None:
                max_val = 10000   # Default max
            
            # Ensure min <= max
            if min_val > max_val:
                return None
            
            return random.randint(int(min_val), int(max_val))
        
        elif data_type == 'float':
            # Handle None bounds
            if min_val is None:
                min_val = -10000.0  # Default min
            if max_val is None:
                max_val = 10000.0   # Default max
            
            # Ensure min <= max
            if min_val > max_val:
                return None
            
            # Generate random float within bounds
            precision = col_schema.get('faker_args', {}).get('right_digits', 2)
            return round(min_val + (max_val - min_val) * random.random(), precision)
        
        elif data_type in ['date', 'datetime']:
            # Handle date generation
            try:
                # Convert string dates to datetime objects if necessary
                if isinstance(min_val, str):
                    try:
                        min_val = datetime.strptime(min_val, '%Y-%m-%d')
                    except:
                        min_val = None
                    
                if isinstance(max_val, str):
                    try:
                        max_val = datetime.strptime(max_val, '%Y-%m-%d')
                    except:
                        max_val = None
            
                # Handle None bounds for dates
                if min_val is None:
                    min_val = datetime.now() - timedelta(days=365)  # Default: 1 year ago
                if max_val is None:
                    max_val = datetime.now()  # Default: today
                
                # Ensure min <= max
                if min_val > max_val:
                    return None
                
                # Calculate delta in seconds
                delta_seconds = int((max_val - min_val).total_seconds())
                if delta_seconds <= 0:
                    return min_val  # Return min if delta is 0 or negative
                
                # Generate random datetime within bounds
                random_seconds = random.randint(0, delta_seconds)
                generated_date = min_val + timedelta(seconds=random_seconds)
            
                # Return date only if data_type is 'date'
                if data_type == 'date':
                    return generated_date.date()
                
                return generated_date
        
            except Exception as e:
                print(f"    Error generating date/datetime value: {str(e)}")
                return None
            
        else:
            # For non-numeric, non-date columns, return None
            return None

    def _verify_relationships(self, relationships):
        """
    Verify that all relationships are satisfied after regeneration.
    """
        print("  Verifying relationships...")
        violations = 0
    
        for rel_key, rel_info in relationships.items():
            source_col = rel_info['source']
            target_col = rel_info['target']
            relationship = rel_info['relationship']
        
            # Skip if either column not in synthetic data
            if source_col not in self.synthetic_data or target_col not in self.synthetic_data:
                continue
            
            # Check each row
            for i in range(self.NUM_ROWS):
                source_val = self.synthetic_data[source_col][i]
                target_val = self.synthetic_data[target_col][i]
            
                # Skip if either value is None
                if source_val is None or target_val is None:
                    continue
                
                # Check relationship
                violated = False
                if relationship == '<':
                    violated = not (source_val < target_val)
                elif relationship == '>':
                    violated = not (source_val > target_val)
                elif relationship == '=':
                    violated = not (source_val == target_val)
                
                # Count violation
                if violated:
                    violations += 1
                
        # Print summary
        if violations > 0:
            print(f"  Found {violations} relationship violations after regeneration")
        else:
            print("  All relationships satisfied")

    def step5_add_nulls(self):
        """
    Step 5: Add nulls based on the null percentage in the original data.
    """
        print("\n--- STEP 5: Adding Nulls ---")
    
        # Skip if no synthetic data
        if not self.synthetic_data:
            print("  No synthetic data to add nulls to, skipping step 5")
            return self.synthetic_data
        
        total_nulls_added = 0
    
        # Process each column
        for column in self.column_order:
            # Skip if column not in synthetic data
            if column not in self.synthetic_data:
                continue
            
            # Get column schema
            col_schema = self.schema['columns'].get(column, {})
            col_stats = col_schema.get('stats', {})
        
            # Get null percentage from schema (default to 0 if not available)
            null_percentage = col_stats.get('null_percentage', 0)
        
            # Skip if no nulls in original data
            if null_percentage <= 0:
                continue
            
            print(f"  Adding nulls to {column} (null percentage: {null_percentage:.2f}%)")
        
            # Calculate number of nulls to add
            target_null_count = int(self.NUM_ROWS * (null_percentage / 100))
        
            # Count existing nulls
            existing_nulls = sum(1 for val in self.synthetic_data[column] if val is None)
        
            # Calculate additional nulls needed
            additional_nulls = max(0, target_null_count - existing_nulls)
        
            if additional_nulls <= 0:
                print(f"    Already have enough nulls in {column}, skipping")
                continue
            
            print(f"    Adding {additional_nulls} nulls to {column}")
        
            # Get non-null indices
            non_null_indices = [
                i for i in range(self.NUM_ROWS)
                if self.synthetic_data[column][i] is not None
            ]
        
            # Skip if no non-null values to replace
            if not non_null_indices:
                continue
            
            # Randomly select indices to set to None
            if additional_nulls > len(non_null_indices):
                additional_nulls = len(non_null_indices)
            
            null_indices = random.sample(non_null_indices, additional_nulls)
        
            # Set values to None
            for i in null_indices:
                self.synthetic_data[column][i] = None
            
            total_nulls_added += additional_nulls
        
        print(f"  Added {total_nulls_added} nulls across all columns")
    
        return self.synthetic_data


    def save_synthetic_data(self, output_path):
        """
        Save synthetic data to CSV file.
        """
        # Create a DataFrame from the synthetic data
        result_df = pd.DataFrame(self.synthetic_data)
    
        # Reorder columns to match original data
        ordered_cols = [col for col in self.column_order if col in result_df.columns]
        result_df = result_df[ordered_cols]
    
        # Save to CSV
        result_df.to_csv(output_path, index=False)
        print(f"\nSaved synthetic data to {output_path}")
        print(f"Shape: {result_df.shape}")
    
        return result_df

# Improved regex pattern generator

def generate_from_regex_pattern(pattern):
    """
    Generate a string that matches the given regex pattern using the rstr library.
    Handles standard regex patterns and falls back to custom handling when needed.
    """
    if not isinstance(pattern, str):
        return None
        
    # Dictionary of patterns requiring special handling
    custom_handlers = {
        # Add custom handlers for any patterns that need special treatment
        # Format: 'regex_pattern': handler_function
    }
    
    # Check if pattern matches any of our custom handlers
    for pattern_regex, handler in custom_handlers.items():
        if re.fullmatch(pattern_regex, pattern):
            return handler()
    
    # Special case for exact pattern matches
    if pattern in custom_handlers:
        return custom_handlers[pattern]()
    
    # Use rstr for standard pattern generation
    try:
        return rstr.xeger(pattern)
    except Exception as e:
        # If rstr fails, try to recover with a fallback approach
        try:
            # Fallback for very large repetition counts
            large_repetition = re.search(r'(.+)\{(\d+)(,\d+)?\}', pattern)
            if large_repetition:
                base_element = large_repetition.group(1)
                min_count = int(large_repetition.group(2))
                if min_count > 1000:  # Adjust threshold as needed
                    # Handle the pattern manually for large counts
                    subpattern = pattern[:large_repetition.start(2)] + "1000" + pattern[large_repetition.end(2):]
                    partial_result = rstr.xeger(subpattern)
                    return partial_result
        
            # Log the error or notify but don't expose in returned string
            print(f"Error generating string for pattern '{pattern}': {e}")
            
            # Last resort: return a placeholder indicating failure but not exposing the error
            return "PATTERN_GENERATION_FAILED"
        except:
            # If all recovery attempts fail
            return "PATTERN_GENERATION_FAILED"
        
def main():
    """Main function to run the synthetic data generator"""

    schema = 'enhanced_schema.json'
    csv = 'customer_data.csv'
    rows = 500
    output = 'synth.csv'
    
    # Initialize generator
    print("lets start")
    generator = SyntheticDataGenerator(schema, csv, rows)
    
    # Run steps
    generator.step0_map_relationships()
    generator.step1_generate_initial_uniques()
    generator.step2_populate_based_on_fd()
    generator.step3_fill_remaining_blanks()
    generator.step4_enforce_value_relationships()
    generator.step5_add_nulls()
    
    # Save results
    generator.save_synthetic_data(output)
    
if __name__ == "__main__":
    main()