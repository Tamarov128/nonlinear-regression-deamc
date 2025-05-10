import re
import pandas as pd
from pathlib import Path

def parse_nist_dat(file_path):
    """Parse NIST .dat files and extract data"""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Extract data section
    data_section = content.split('Data:   y               x\n')[-1]
    data = []
    for line in data_section.split('\n'):
        if not line.strip():
            continue
        # Handle both scientific and decimal notation
        numbers = re.findall(r'[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?', line)
        if len(numbers) >= 2:
            y, x = numbers[:2]
            data.append((float(y), float(x)))
    
    df = pd.DataFrame(data, columns=['y', 'x'])
    
    return df

def process_directory(input_dir, output_dir):
    """Process all .dat files in a directory"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    (output_path / 'processed').mkdir(exist_ok=True)
    
    for dat_file in input_path.glob('*.dat'):
        try:
            # Process data
            df = parse_nist_dat(dat_file)
            
            # Save processed data
            csv_path = output_path / 'processed' / f'{dat_file.stem}.csv'
            df.to_csv(csv_path, index=False)
            
            print(f"Processed {dat_file.name}")
            
        except Exception as e:
            print(f"Error processing {dat_file.name}: {str(e)}")

if __name__ == "__main__":
    import sys
    input_dir = sys.argv[1] if len(sys.argv) > 1 else 'nist_models/data/raw'
    output_dir = sys.argv[2] if len(sys.argv) > 2 else 'nist_models/data'

    process_directory(input_dir, output_dir)