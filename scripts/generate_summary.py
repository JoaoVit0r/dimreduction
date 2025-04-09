#!/usr/bin/env python3

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import re
import matplotlib.dates as mdates
import glob

def read_env_file(env_path):
    """Read and parse the .env file."""
    env_vars = {}
    with open(env_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                if '=' in line:
                    # Split at the first '=' and remove any trailing comments
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.split('#')[0].strip()  # Remove any comments after the value
                    env_vars[key] = value
    return env_vars

def analyze_input_file(input_path):
    """Analyze the input file to get information about genes."""
    try:
        # Read the file with tab separator
        df = pd.read_csv(input_path, sep='\t')
        
        # Get the actual column names
        columns = list(df.columns)
        
        # Count genes (assuming first column is not a gene if it's descriptive)
        num_genes = len(columns)
        if 'ARE_COLUMNS_DESCRIPTIVE' in os.environ and os.environ['ARE_COLUMNS_DESCRIPTIVE'].lower() == 'true':
            num_genes -= 1
        
        return {
            'file_path': input_path,
            'file_size': os.path.getsize(input_path),
            'num_genes': num_genes,
            'num_samples': len(df),
            'columns': columns
        }
    except Exception as e:
        return {
            'error': str(e),
            'file_path': input_path
        }

def read_dstat_data(dstat_file):
    """Read and analyze dstat output file."""
    try:
        df = pd.read_csv(dstat_file, skiprows=5)
        df.columns = [col.strip().replace('%', 'pct').replace(' ', '_').lower() for col in df.columns]
        
        # Calculate statistics
        cpu_cols = [col for col in df.columns if col.startswith('usr') or col.startswith('sys')]
        mem_cols = ['used', 'free', 'cach', 'avai']
        load_cols = [col for col in df.columns if col.startswith('1m') or col.startswith('5m') or col.startswith('15m')]
        
        return {
            'cpu_stats': {
                'avg': df[cpu_cols].mean().to_dict(),
                'max': df[cpu_cols].max().to_dict(),
                'min': df[cpu_cols].min().to_dict()
            },
            'memory_stats': {
                'avg': df[mem_cols].mean().to_dict(),
                'max': df[mem_cols].max().to_dict(),
                'min': df[mem_cols].min().to_dict()
            },
            'load_stats': {
                'avg': df[load_cols].mean().to_dict(),
                'max': df[load_cols].max().to_dict(),
                'min': df[load_cols].min().to_dict()
            }
        }
    except Exception as e:
        return {'error': str(e)}

def compare_quantization_results(base_dir):
    """Compare quantization results between different runs."""
    comparisons = {}
    try:
        # Find all result files in monitoring_plots directory
        quantized_files = []
        final_data_files = []
        monitoring_plots_dir = os.path.join(base_dir, 'monitoring_plots')
        if os.path.exists(monitoring_plots_dir):
            for root, _, files in os.walk(monitoring_plots_dir):
                for file in files:
                    if file.endswith('-quantized_data.txt'):
                        quantized_files.append(os.path.join(root, file))
                    elif file.endswith('-final_data.txt'):
                        final_data_files.append(os.path.join(root, file))
        
        # Compare quantized files
        if len(quantized_files) >= 2:
            comparisons['quantized_data'] = compare_files(quantized_files, 'quantized')
        else:
            comparisons['quantized_data'] = {
                'error': 'Not enough quantized files to compare',
                'searched_directory': monitoring_plots_dir,
                'found_files': quantized_files
            }
        
        # Compare final data files
        if len(final_data_files) >= 2:
            comparisons['final_data'] = compare_files(final_data_files, 'final')
        else:
            comparisons['final_data'] = {
                'error': 'Not enough final data files to compare',
                'searched_directory': monitoring_plots_dir,
                'found_files': final_data_files
            }
        
        return comparisons
    except Exception as e:
        return {'error': str(e)}

def read_markers_file(markers_file):
    """Read a file with timestamp markers and return as dictionary."""
    markers = []  # Change to list of tuples (timestamp, label)
    if not os.path.exists(markers_file):
        return markers
        
    print(f"\nReading execution markers from: {markers_file}")
    month_map = {
        'jan': 'Jan', 'fev': 'Feb', 'mar': 'Mar', 'abr': 'Apr', 'mai': 'May', 'jun': 'Jun',
        'jul': 'Jul', 'ago': 'Aug', 'set': 'Sep', 'out': 'Oct', 'nov': 'Nov', 'dez': 'Dec'
    }
    
    with open(markers_file, 'r') as f:
        for line in f:
            lineInfos = list(filter(lambda x: x != '', line.strip().split(' ')))
            if len(lineInfos) < 3:
                print(f"Warning: Could not parse timestamp in markers file: {line.strip()}")
                continue
            
            # Replace Portuguese month with English month
            month = lineInfos[0].lower()
            if month in month_map:
                lineInfos[0] = month_map[month]
            
            lineDateTime = ' '.join(lineInfos[:2])
            lineLabel = ' '.join(lineInfos[2:])
            try:
                timestamp = pd.to_datetime(lineDateTime, format='%Y-%m-%d %H:%M:%S')
                if lineLabel:
                    # Store as tuple (timestamp, label) in a list
                    markers.append((timestamp, lineLabel))
            except ValueError as e:
                print(f"Warning: Could not parse timestamp in markers file: {line.strip()}")
    
    # Sort markers by timestamp
    markers.sort(key=lambda x: x[0])
    return markers

def compare_files(files, file_type):
    """Compare a list of files of the same type."""
    comparisons = {}
    for i in range(len(files)):
        for j in range(i+1, len(files)):
            file1 = files[i]
            file2 = files[j]
            
            try:
                df1 = pd.read_csv(file1, sep='\t')
                df2 = pd.read_csv(file2, sep='\t')
                
                # Convert numpy types to Python native types for JSON serialization
                total_diff = int((df1 != df2).sum().sum())
                percentage_diff = float(((df1 != df2).sum().sum() / df1.size) * 100)
                
                comparison = {
                    'files': [file1, file2],
                    'shape_difference': df1.shape != df2.shape,
                    'value_differences': {
                        'total_differences': total_diff,
                        'percentage_different': percentage_diff
                    }
                }
                
                key = f"{os.path.basename(file1)}_vs_{os.path.basename(file2)}"
                comparisons[key] = comparison
            except Exception as e:
                comparisons[f"{os.path.basename(file1)}_vs_{os.path.basename(file2)}"] = {'error': str(e)}
    
    return comparisons

def calculate_execution_durations(markers_file):
    """Calculate duration of each execution from markers file."""
    durations = []
    if not os.path.exists(markers_file):
        return {'error': 'Markers file not found'}
    
    print(f"Processing markers file: {markers_file}")
    markers = read_markers_file(markers_file)
    if not markers:
        return {'error': 'No markers found in file'}
    
    # Look for start/end pairs
    # Typically markers will contain labels like "START" and "END" or similar patterns
    for i in range(len(markers) - 1):
        start_time, start_label = markers[i]
        end_time, end_label = markers[i+1]
        
        # Check if this is a start/end pair
        if any(keyword in start_label.lower() for keyword in ['start', 'begin', 'inicio']) and \
           any(keyword in end_label.lower() for keyword in ['end', 'finish', 'fim']):
            
            # Convert to seconds
            duration_seconds = (end_time - start_time).total_seconds()
            
            if duration_seconds > 0:  # Valid duration
                durations.append({
                    'execution_number': len(durations) + 1,
                    'start_time': start_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'end_time': end_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'duration_seconds': float(duration_seconds),
                    'start_label': start_label,
                    'end_label': end_label
                })
    
    # If no clear start/end pairs were found, try sequential pairs
    if not durations and len(markers) >= 2:
        print("No clear start/end pairs found. Using sequential marker pairs.")
        for i in range(0, len(markers)-1, 2):
            if i+1 >= len(markers):
                break
                
            start_time, start_label = markers[i]
            end_time, end_label = markers[i+1]
            
            # Convert to seconds
            duration_seconds = (end_time - start_time).total_seconds()
            
            if duration_seconds > 0:  # Valid duration
                durations.append({
                    'execution_number': i//2 + 1,
                    'start_time': start_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'end_time': end_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'duration_seconds': float(duration_seconds),
                    'start_label': start_label,
                    'end_label': end_label
                })
    
    if not durations:
        # Debug output
        print(f"Found {len(markers)} markers but could not identify execution pairs.")
        for i, (ts, label) in enumerate(markers):
            print(f"  {i}: {ts.strftime('%Y-%m-%d %H:%M:%S')} - {label}")
        return {'error': 'Could not identify execution start/end pairs', 'markers_found': len(markers)}
    
    # Calculate statistics
    durations_seconds = [d['duration_seconds'] for d in durations]
    return {
        'executions': durations,
        'statistics': {
            'mean_duration_seconds': float(np.mean(durations_seconds)),
            'min_duration_seconds': float(np.min(durations_seconds)),
            'max_duration_seconds': float(np.max(durations_seconds)),
            'std_duration_seconds': float(np.std(durations_seconds))
        }
    }

def find_env_file(base_dir):
    """Find the environment variables file."""
    monitoring_plots_dir = None
    
    # First, check if base_dir itself contains 'monitoring_plots'
    potential_monitoring_dir = os.path.join(base_dir, 'monitoring_plots')
    if os.path.isdir(potential_monitoring_dir):
        monitoring_plots_dir = potential_monitoring_dir
    else:
        # If not found directly, search within subdirectories (implementation/script structure)
        for root, dirs, _ in os.walk(base_dir):
            if 'monitoring_plots' in dirs:
                monitoring_plots_dir = os.path.join(root, 'monitoring_plots')
                break  # Found the first monitoring_plots dir
        
    if monitoring_plots_dir and os.path.exists(monitoring_plots_dir):
        for file in os.listdir(monitoring_plots_dir):
            if file.startswith('env_variables_') and file.endswith('.txt'):
                return os.path.join(monitoring_plots_dir, file)
                
    # Fallback or if not found in expected structure
    env_in_base = os.path.join(base_dir, '.env')
    if os.path.exists(env_in_base):
        return env_in_base
        
    return None # Indicate not found

def find_markers_files(base_dir):
    """Find all execution markers files in the directory structure."""
    markers_files = []
    
    # Search for markers files in the entire directory structure
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.startswith('execution_markers_') and file.endswith('.txt'):
                markers_files.append(os.path.join(root, file))
                
    return markers_files

def generate_summary_files(base_dir):
    """Generate summary files for the monitoring results."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    summary_dir = os.path.join(base_dir, f'summary_{timestamp}')
    os.makedirs(summary_dir, exist_ok=True)
    
    # 1. Input Analysis
    env_file = find_env_file(base_dir)
    if not env_file:
        print("Warning: Could not find env_variables file or .env file in base directory. Cannot perform input analysis.")
        input_analysis = {'error': 'Environment file not found'}
    else:
        print(f"Using environment file: {env_file}")
        env_vars = read_env_file(env_file)
        input_file_path = env_vars.get('INPUT_FILE_PATH')
        if input_file_path:
            input_analysis = analyze_input_file(input_file_path)
        else:
            input_analysis = {'error': 'INPUT_FILE_PATH not found in environment file.'}
    
    with open(os.path.join(summary_dir, 'input_analysis.json'), 'w') as f:
        json.dump(input_analysis, f, indent=4)
    
    # 2. Execution Summary
    execution_summary = {
        'timestamp': timestamp,
        'environment_file': env_file if env_file else 'Not Found',
        'environment': env_vars if env_file else {},
        'input_analysis': input_analysis,
        'implementations': {}
    }
    
    # Process each implementation directory
    for impl_dir in os.listdir(base_dir):
        impl_path = os.path.join(base_dir, impl_dir)
        if not os.path.isdir(impl_path):
            continue
            
        execution_summary['implementations'][impl_dir] = {}
        
        # Process each script directory
        for script_dir in os.listdir(impl_path):
            script_path = os.path.join(impl_path, script_dir)
            if not os.path.isdir(script_path):
                continue
                
            # Find monitoring_plots directory - either directly or recursively
            monitoring_dir = os.path.join(script_path, 'monitoring_plots')
            if not os.path.exists(monitoring_dir):
                # Try to find monitoring_plots directory within script_path
                monitoring_dirs = []
                for root, dirs, _ in os.walk(script_path):
                    if 'monitoring_plots' in dirs:
                        monitoring_dirs.append(os.path.join(root, 'monitoring_plots'))
                
                if monitoring_dirs:
                    monitoring_dir = monitoring_dirs[0]  # Use the first one found
                else:
                    # No monitoring_plots directory found, skip this script
                    continue
            
            # Find markers files
            markers_files = []
            for root, _, files in os.walk(monitoring_dir):
                for file in files:
                    if file.startswith('execution_markers_') and file.endswith('.txt'):
                        markers_files.append(os.path.join(root, file))
            
            # Calculate execution durations if markers files exist
            execution_durations = {}
            if markers_files:
                # Use the first markers file found
                execution_durations = calculate_execution_durations(markers_files[0])
            else:
                execution_durations = {'error': 'Markers files not found in monitoring directory.'}
            
            # Read dstat data
            dstat_file_pattern = os.path.join(monitoring_dir, 'dstat_output_*.csv')
            dstat_files = glob.glob(dstat_file_pattern)
            dstat_data = {}
            if dstat_files:
                dstat_data = read_dstat_data(dstat_files[0])
            
            execution_summary['implementations'][impl_dir][script_dir] = {
                'resource_usage': dstat_data,
                'execution_durations': execution_durations
            }
    
    with open(os.path.join(summary_dir, 'execution_summary.json'), 'w') as f:
        json.dump(execution_summary, f, indent=4)
    
    # 3. Results Comparison
    results_comparison = compare_quantization_results(base_dir)
    
    with open(os.path.join(summary_dir, 'results_comparison.json'), 'w') as f:
        json.dump(results_comparison, f, indent=4)
    
    return summary_dir

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python generate_summary.py <monitoring_base_directory>")
        sys.exit(1)
    
    base_dir = sys.argv[1]
    summary_dir = generate_summary_files(base_dir)
    print(f"Summary files generated in: {summary_dir}") 