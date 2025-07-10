import os
import json
import csv
import re
from pathlib import Path


def parse_nist_dat_file(file_path):
    """
    Parse a NIST .dat file and extract metadata and data points.
    
    Args:
        file_path (str): Path to the .dat file
    
    Returns:
        tuple: (metadata_dict, data_points_list)
    """
    with open(file_path, 'r') as file:
        content = file.read()
    
    lines = content.split('\n')
    
    # Initialize variables
    metadata = {}
    data_points = []
    
    # Extract dataset name from filename
    dataset_name = Path(file_path).stem
    metadata['dataset_name'] = dataset_name
    
    # Find difficulty level
    if 'Higher Level of Difficulty' in content:
        metadata['difficulty'] = 'Higher'
    elif 'Lower Level of Difficulty' in content:
        metadata['difficulty'] = 'Lower'
    elif 'Average Level of Difficulty' in content:
        metadata['difficulty'] = 'Average'
    else:
        metadata['difficulty'] = 'Unknown'
    
    # Extract number of observations
    obs_match = re.search(r'(\d+)\s+Observations', content)
    if obs_match:
        metadata['observations_no'] = int(obs_match.group(1))
    
    # Extract number of parameters
    param_match = re.search(r'(\d+)\s+Parameters', content)
    if param_match:
        metadata['parameters_no'] = int(param_match.group(1))
    
    # Extract model equation
    model_match = re.search(r'y\s*=\s*(.+?)(?:\+\s*e\s*$|\s*$)', content, re.MULTILINE)
    if model_match:
        model_eq = model_match.group(1).strip()
        metadata['model_function'] = convert_equation_to_lambda(model_eq)
    
    # Parse starting values and certified values using a more robust approach
    # Find the section between "Starting values" and "Residual Sum of Squares"
    start_values_section = re.search(
        r'Starting values.*?Certified Values.*?\n(.*?)(?=Residual Sum of Squares)',
        content, re.DOTALL | re.IGNORECASE
    )
    
    if start_values_section:
        section_text = start_values_section.group(1)
        
        start_1 = []
        start_2 = []
        certified = []
        
        # Split into lines and process each line
        section_lines = section_text.strip().split('\n')
        
        for line in section_lines:
            line = line.strip()
            if line and line.startswith('b') and '=' in line:
                # Parse lines like: "b1 =   -2000       -1500        -2.5235058043E+03  2.9715175411E+02"
                # Remove the parameter name and equals sign
                values_part = re.sub(r'^b\d+\s*=\s*', '', line)
                
                # Split by whitespace and extract numeric values
                parts = values_part.split()
                
                if len(parts) >= 3:
                    try:
                        # First value is Start 1
                        start_1_val = float(parts[0])
                        start_1.append(start_1_val)
                        
                        # Second value is Start 2
                        start_2_val = float(parts[1])
                        start_2.append(start_2_val)
                        
                        # Third value is Certified (handle scientific notation)
                        certified_str = parts[2]
                        # Handle scientific notation like -2.5235058043E+03
                        certified_val = float(certified_str)
                        certified.append(certified_val)
                        
                    except (ValueError, IndexError) as e:
                        print(f"Warning: Could not parse parameter line: {line}")
                        continue
        
        metadata['start_1'] = start_1
        metadata['start_2'] = start_2
        metadata['certified'] = certified
    
    # Extract residual sum of squares
    rss_match = re.search(r'Residual Sum of Squares:\s*([+-]?\d+(?:\.\d+)?(?:E[+-]?\d+)?)', content)
    if rss_match:
        metadata['residual_sum_squares'] = float(rss_match.group(1))
    
    # Extract data points
    # Look for data section starting with "Data:   y               x"
    data_section_match = re.search(r'Data:\s+y\s+x\s*\n(.*?)(?:\n\s*$|\Z)', content, re.DOTALL)
    
    if data_section_match:
        data_text = data_section_match.group(1)
        
        # Parse each data line
        data_lines = data_text.strip().split('\n')
        for line in data_lines:
            line = line.strip()
            if line and not line.startswith('Data:'):
                # Handle scientific notation with E0, E+, E-
                line = re.sub(r'E0\b', '', line)  # Remove E0
                
                # Split the line and try to extract y and x values
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        y_val = float(parts[0])
                        x_val = float(parts[1])
                        data_points.append([y_val, x_val])
                    except (ValueError, IndexError):
                        continue
    
    return metadata, data_points


def convert_equation_to_lambda(equation):
    """
    Convert a mathematical equation to a lambda function string.
    
    Args:
        equation (str): Mathematical equation string
    
    Returns:
        str: Lambda function string
    """
    # Remove common prefixes and suffixes
    eq = equation.strip()
    eq = re.sub(r'^\s*y\s*=\s*', '', eq)  # Remove y =
    eq = re.sub(r'\s*\+\s*e\s*$', '', eq)  # Remove + e at end
    eq = re.sub(r'\s*\+\s*$', '', eq)     # Remove trailing +
    
    # Fix common mathematical notations
    eq = eq.replace('exp(', 'np.exp(')
    eq = eq.replace('exp[', 'np.exp(')
    eq = eq.replace(']', ')')
    eq = eq.replace('arctan[', 'np.arctan(')
    eq = eq.replace('cos(', 'np.cos(')
    eq = eq.replace('sin(', 'np.sin(')
    eq = eq.replace('pi', 'np.pi')
    
    # Fix exponentiation
    eq = re.sub(r'\*\*\s*\(-1/(\w+)\)', r'**(-1/\1)', eq)
    
    # Determine parameters based on content
    param_nums = re.findall(r'b(\d+)', eq)
    if param_nums:
        max_param = max(int(p) for p in param_nums)
        params = ['x'] + [f'b{i}' for i in range(1, max_param + 1)]
        params_str = ', '.join(params)
    else:
        params_str = 'x'
    
    return f"lambda {params_str}: {eq}"


def debug_parse_starting_values(file_path):
    """
    Debug function to help understand the format of starting values section.
    """
    with open(file_path, 'r') as file:
        content = file.read()
    
    # Find the starting values section
    start_idx = content.find('Starting values')
    end_idx = content.find('Residual Sum of Squares')
    
    if start_idx != -1 and end_idx != -1:
        section = content[start_idx:end_idx]
        print("Starting values section:")
        print("=" * 50)
        print(section)
        print("=" * 50)
        
        # Find parameter lines
        lines = section.split('\n')
        for i, line in enumerate(lines):
            if 'b1' in line or 'b2' in line or 'b3' in line:
                print(f"Line {i}: '{line}'")


def extract_all_nist_data(raw_dir, processed_dir, models_dir):
    """
    Extract data from all .dat files in the raw directory.
    
    Args:
        raw_dir (str): Path to raw data directory
        processed_dir (str): Path to processed data directory
        models_dir (str): Path to models directory
    """
    # Create output directories if they don't exist
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    # Initialize models dictionary
    models_dict = {}
    
    # Process all .dat files
    raw_path = Path(raw_dir)
    for dat_file in raw_path.glob('*.dat'):
        print(f"Processing {dat_file.name}...")
        
        try:
            # Parse the .dat file
            metadata, data_points = parse_nist_dat_file(dat_file)
            
            if not data_points:
                print(f"Warning: No data points found in {dat_file.name}")
                continue
            
            # Create CSV file
            csv_filename = dat_file.stem + '.csv'
            csv_path = Path(processed_dir) / csv_filename
            
            with open(csv_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['y', 'x'])  # Header
                for y, x in data_points:
                    writer.writerow([y, x])
            
            # Add to models dictionary
            dataset_name = metadata.get('dataset_name', dat_file.stem)
            models_dict[dataset_name] = {
                'difficulty': metadata.get('difficulty', 'Unknown'),
                'observations_no': metadata.get('observations_no', len(data_points)),
                'parameters_no': metadata.get('parameters_no', 0),
                'start_1': metadata.get('start_1', []),
                'start_2': metadata.get('start_2', []),
                'certified': metadata.get('certified', []),
                'residual_sum_squares': metadata.get('residual_sum_squares', 0.0),
                'model_function': metadata.get('model_function', '')
            }
            
            print(f"Successfully processed {dat_file.name}")
            print(f"  - Found {len(data_points)} data points")
            print(f"  - Parameters: {metadata.get('parameters_no', 0)}")
            print(f"  - Start 1: {metadata.get('start_1', [])}")
            print(f"  - Start 2: {metadata.get('start_2', [])}")
            print(f"  - Certified: {metadata.get('certified', [])}")
            print()
            
        except Exception as e:
            print(f"Error processing {dat_file.name}: {str(e)}")
            # For debugging, you can uncomment the next line
            # debug_parse_starting_values(dat_file)
            continue
    
    # Save models JSON file
    models_json_path = Path(models_dir) / 'nist_models.json'
    with open(models_json_path, 'w') as jsonfile:
        json.dump(models_dict, jsonfile, indent=2)
    
    print(f"\nProcessing complete!")
    print(f"Processed {len(models_dict)} datasets")
    print(f"CSV files saved to: {processed_dir}")
    print(f"Models JSON saved to: {models_json_path}")


def main():
    """
    Main function to run the NIST data extraction.
    """
    # Define directory paths
    raw_dir = "../data/raw"
    processed_dir = "../data/processed"
    models_dir = "../data/models"
    
    # Check if raw directory exists
    if not os.path.exists(raw_dir):
        print(f"Error: Raw data directory '{raw_dir}' does not exist.")
        print("Please ensure the directory exists and contains .dat files.")
        return
    
    # Extract all NIST data
    extract_all_nist_data(raw_dir, processed_dir, models_dir)


if __name__ == "__main__":
    main()