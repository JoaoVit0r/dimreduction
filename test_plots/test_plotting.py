#!/usr/bin/env python3

import os
import sys

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.plot_dstat import read_dstat_csv, create_plots

def main():
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define input and output paths
    dstat_file = os.path.join(script_dir, "test_dstat_output.csv")
    markers_file = os.path.join(script_dir, "test_markers.txt")
    output_dir = os.path.join(script_dir, "plots")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        print("\n===============================================")
        print("Starting Test Plot Generation")
        print("===============================================")
        print(f"Input file: {dstat_file}")
        print(f"Output directory: {output_dir}")
        print(f"Markers file: {markers_file}")
        print("===============================================")
        
        # Read the dstat data
        df = read_dstat_csv(dstat_file)
        
        # Create plots
        create_plots(df, "test_plot", output_dir, markers_file, "split_by:hour")
        
        print("\n===============================================")
        print("Test Plot Generation Completed Successfully")
        print("===============================================")
        print(f"Output directory: {output_dir}")
        
    except Exception as e:
        print(f"\nError during test plotting: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 