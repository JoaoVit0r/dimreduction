#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import sys
from datetime import datetime
import os

def read_dstat_csv(file_path):
    # Skip the first rows that contain header information
    df = pd.read_csv(file_path, skiprows=5)
    
    # Rename columns to remove spaces and percentage signs
    df.columns = [col.strip().replace('%', 'pct').replace(' ', '_').lower() for col in df.columns]
    return df

def create_plots(df, output_prefix, output_folder):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Set style to a default one
    plt.style.use('default')
    
    # Create timestamp for unique filenames
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # CPU Usage Plot
    plt.figure(figsize=(12, 6))
    cpu_cols = [col for col in df.columns if col.startswith('usr') or col.startswith('sys') or col.startswith('idl')]
    df[cpu_cols].plot(title='CPU Usage Over Time')
    plt.xlabel('Time (seconds)')
    plt.ylabel('CPU Usage (%)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'{output_prefix}_cpu_{timestamp}.svg'))
    plt.close()
    
    # Memory Usage Plot
    plt.figure(figsize=(12, 6))
    mem_cols = [col for col in df.columns if col.startswith('used') or col.startswith('free') or col.startswith('buff')]
    df[mem_cols].plot(title='Memory Usage Over Time')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Memory (bytes)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'{output_prefix}_memory_{timestamp}.svg'))
    plt.close()
    
    # Disk I/O Plot
    plt.figure(figsize=(12, 6))
    disk_cols = [col for col in df.columns if col.startswith('read') or col.startswith('writ')]
    df[disk_cols].plot(title='Disk I/O Over Time')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Bytes/s')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'{output_prefix}_disk_{timestamp}.svg'))
    plt.close()
    
    # Network I/O Plot
    plt.figure(figsize=(12, 6))
    net_cols = [col for col in df.columns if col.startswith('recv') or col.startswith('send')]
    df[net_cols].plot(title='Network I/O Over Time')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Bytes/s')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'{output_prefix}_network_{timestamp}.svg'))
    plt.close()

def main():
    if len(sys.argv) != 3:
        print("Usage: python plot_dstat.py <dstat_csv_file> <output_folder>")
        sys.exit(1)
        
    csv_file = sys.argv[1]
    output_folder = sys.argv[2]
    output_prefix = 'dstat_plot'
    
    try:
        df = read_dstat_csv(csv_file)
        create_plots(df, output_prefix, output_folder)
        print(f"Plots have been generated in folder '{output_folder}' with prefix '{output_prefix}'")
    except Exception as e:
        print(f"Error processing the CSV file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()