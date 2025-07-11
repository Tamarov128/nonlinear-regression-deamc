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
    
    # Extract model equation - improved version
    model_equation = extract_model_equation(content)
    if model_equation:
        metadata['model_function'] = convert_equation_to_lambda(model_equation)
    
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


def extract_model_equation(content):
    """
    Extract the model equation from the NIST file content.
    Handles both single-line and multi-line equations.
    
    Args:
        content (str): File content
    
    Returns:
        str: Model equation or None if not found
    """
    # Find the model section - everything between "Model:" and "Starting values"
    model_section_match = re.search(
        r'Model:\s*.*?\n(.*?)(?=Starting values)',
        content, re.DOTALL | re.IGNORECASE
    )
    
    if not model_section_match:
        return None
    
    model_section = model_section_match.group(1).strip()
    
    # Split into lines and look for equation
    lines = model_section.split('\n')
    
    # Find the line that starts with "y =" or contains "y ="
    equation_lines = []
    found_equation_start = False
    
    for line in lines:
        line = line.strip()
        
        # Skip empty lines and lines with just whitespace
        if not line:
            continue
            
        # Skip lines that are clearly not equations (like parameter counts)
        if re.match(r'^\d+\s+Parameters', line):
            continue
        if 'Class' in line:
            continue
        if line.startswith('(') and line.endswith(')'):
            continue
            
        # Look for the start of the equation
        if (line.startswith('y =') or line.startswith('y=')) and not found_equation_start:
            found_equation_start = True
            equation_lines.append(line)
            continue
        
        # If we found the start, continue collecting lines that are part of the equation
        if found_equation_start:
            # Check if this line is part of the equation
            # Lines that start with operators, contain mathematical expressions, or end with + e
            if (line.startswith('+') or line.startswith('-') or 
                line.startswith('*') or line.startswith('/') or
                'exp(' in line or 'cos(' in line or 'sin(' in line or
                'arctan(' in line or 'log(' in line or
                line.endswith('+ e') or line.endswith('+e') or
                '**' in line or '*' in line or 
                ('(' in line and ')' in line and ('b' in line or 'x' in line))):
                equation_lines.append(line)
            else:
                # This line doesn't seem to be part of the equation, stop collecting
                break
    
    if not equation_lines:
        return None
    
    # Join all equation lines and clean up
    equation = ' '.join(equation_lines)
    
    # Remove "y =" from the beginning
    equation = re.sub(r'^\s*y\s*=\s*', '', equation)
    
    # Remove trailing "+ e" or "+e"
    equation = re.sub(r'\s*\+\s*e\s*$', '', equation)
    
    # Clean up extra whitespace
    equation = re.sub(r'\s+', ' ', equation)
    
    # Remove any remaining trailing operators
    equation = re.sub(r'\s*[+\-]\s*$', '', equation)
    
    return equation.strip() if equation.strip() else None


def convert_equation_to_lambda(equation):
    """
    Convert a mathematical equation to a lambda function string.
    
    Args:
        equation (str): Mathematical equation string
    
    Returns:
        str: Lambda function string
    """
    if not equation:
        return ""
    
    # Clean up the equation
    eq = equation.strip()
    
    # Fix common mathematical notations
    eq = eq.replace('exp(', 'np.exp(')
    eq = eq.replace('exp[', 'np.exp(')
    eq = eq.replace(']', ')')
    eq = eq.replace('arctan[', 'np.arctan(')
    eq = eq.replace('arctan(', 'np.arctan(')
    eq = eq.replace('cos(', 'np.cos(')
    eq = eq.replace('sin(', 'np.sin(')
    eq = eq.replace('log(', 'np.log(')
    eq = eq.replace('log[', 'np.log(')
    eq = eq.replace('pi', 'np.pi')
    
    # Handle power operations
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


def debug_model_equation(file_path):
    """
    Debug function to help understand the model equation extraction.
    """
    with open(file_path, 'r') as file:
        content = file.read()
    
    print(f"Debugging model equation for: {file_path}")
    print("=" * 50)
    
    # Show the model section
    model_section_match = re.search(
        r'Model:\s*.*?\n(.*?)(?=Starting values)',
        content, re.DOTALL | re.IGNORECASE
    )
    
    if model_section_match:
        model_section = model_section_match.group(1)
        print("Model section found:")
        print(repr(model_section))
        print("Model section content:")
        print(model_section)
        print("-" * 30)
        
        # Show line by line analysis
        lines = model_section.split('\n')
        for i, line in enumerate(lines):
            print(f"Line {i}: '{line.strip()}'")
        print("-" * 30)
    
    # Try to extract equation
    equation = extract_model_equation(content)
    print(f"Extracted equation: '{equation}'")
    
    if equation:
        lambda_func = convert_equation_to_lambda(equation)
        print(f"Lambda function: {lambda_func}")
    
    print("=" * 50)


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
            print(f"  - Model function: {metadata.get('model_function', 'Not found')}")
            print(f"  - Start 1: {metadata.get('start_1', [])}")
            print(f"  - Start 2: {metadata.get('start_2', [])}")
            print(f"  - Certified: {metadata.get('certified', [])}")
            print()
            
        except Exception as e:
            print(f"Error processing {dat_file.name}: {str(e)}")
            # For debugging, you can uncomment the next line
            # debug_model_equation(dat_file)
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