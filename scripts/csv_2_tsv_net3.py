#!/usr/bin/env python3
import sys
import os
import re
import csv

def get_technique_from_filename(filename):
    """
    Extract the technique from the input filename based on the specified rules.
    """
    # Rule 1: Check for '.*final*-data*' pattern
    if re.search(r'.*final.*data.*', filename, re.IGNORECASE):
        return "DimReduction"
    
    # Rule 2: Check for GRN patterns
    grn_genie3_match = re.search(r'GRN_(GENIE3)_([^_]+).*\.', filename, re.IGNORECASE)
    if grn_genie3_match:
        return f"{grn_genie3_match.group(1)}_{grn_genie3_match.group(2)}"
    
    # Rule 3: General GRN pattern
    grn_match = re.search(r'GRN_([^_]+).*\.', filename, re.IGNORECASE)
    if grn_match:
        return grn_match.group(1)
    
    return "Unknown"

def convert_csv_to_tsv(input_file, output_file):
    """
    Convert CSV file to TSV format.
    """
    try:
        with open(input_file, 'r', newline='', encoding='utf-8') as csv_in:
            with open(output_file, 'w', newline='', encoding='utf-8') as tsv_out:
                # Read CSV and write TSV
                csv_reader = csv.reader(csv_in)
                tsv_writer = csv.writer(tsv_out, delimiter='\t')
                
                for row in csv_reader:
                    tsv_writer.writerow(row)
        
        print(f"Successfully converted {input_file} to {output_file}")
        
    except Exception as e:
        print(f"Error during conversion: {e}")
        sys.exit(1)

def main():
    if len(sys.argv) != 2:
        print("Usage: python convert_csv_to_tsv.py <input_file.csv>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    folder_output = sys.argv[2]
    
    # Validate input file exists
    if not os.path.isfile(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        sys.exit(1)
    
    # Extract technique from filename
    filename = os.path.basename(input_file)
    technique = get_technique_from_filename(filename)
    
    # Generate output filename
    output_file = f"{folder_output}/DREAM5_NetworkInference_{technique}_Network3.tsv"
    
    # Convert CSV to TSV
    convert_csv_to_tsv(input_file, output_file)

if __name__ == "__main__":
    main()