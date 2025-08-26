#!/usr/bin/env python3
"""
Script to process network inference results, evaluate them using GENECI,
and export results in a table format.
"""

import os
import re
import glob
import argparse
import subprocess
from datetime import datetime
import pandas as pd
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='Process and evaluate network inference results')
    parser.add_argument('--monitoring-dir', default='monitoring_plots',
                        help='Directory containing monitoring results')
    parser.add_argument('--external-projects', default='../external_projects',
                        help='Path to external projects directory')
    parser.add_argument('--output', default='evaluation_results.csv',
                        help='Output CSV file name')
    parser.add_argument('--threshold', type=float, default=0.7,
                        help='Threshold value for binarization (default: 0.7)')
    parser.add_argument('--skip-binarize', action='store_true',
                        help='Skip the binarization step and use continuous scores for evaluation')
    return parser.parse_args()

def find_network_files(monitoring_dir):
    """Find all network output files in the monitoring directory"""
    network_files = []
    
    # Find Java DimReduction files
    java_files = glob.glob(f"{monitoring_dir}/**/*-final_weight_data.txt", recursive=True)
    for file in java_files:
        network_files.append({
            'type': 'java',
            'path': file,
            'technique': 'DimReduction'
        })
    
    # Find GENECI files
    geneci_files = glob.glob(f"{monitoring_dir}/**/GRN_*.csv", recursive=True)
    for file in geneci_files:
        technique = Path(file).stem.replace('GRN_', '')
        network_files.append({
            'type': 'geneci',
            'path': file,
            'technique': technique
        })
    
    return network_files

def extract_metadata_from_path(file_path):
    """Extract metadata from file path"""
    path_parts = Path(file_path).parts
    metadata = {}
    
    # Extract dataset information
    dataset_dir = next((part for part in path_parts if part.startswith('dataset_')), None)
    if dataset_dir:
        metadata['dataset'] = dataset_dir.replace('dataset_', '')
        # Extract network ID (e.g., 100_1 from dream4_100_01_exp)
        match = re.search(r'dream4_(\d+)_(\d+)_exp', metadata['dataset'])
        if match:
            metadata['network_id'] = f"{match.group(1)}_{int(match.group(2))}"
    
    # Extract execution parameters
    metadata['threads'] = next((part.replace('threads_', '') 
                              for part in path_parts if part.startswith('threads_')), '1')
    
    return metadata


def get_execution_time(markers_file):
    """Parse execution time from markers file"""
    try:
        with open(markers_file, 'r') as f:
            lines = f.readlines()
            for line in reversed(lines):  # Look at last lines first
                if 'Duration:' in line:
                    # Match patterns like: 1h 2m 3s, 2m 3s, or 3s
                    match = re.search(r'Duration:\s*(?:(\d+)h\s*)?(?:(\d+)m\s*)?(?:(\d+)s)?', line)
                    if match:
                        hours = int(match.group(1) or 0)
                        minutes = int(match.group(2) or 0)
                        seconds = int(match.group(3) or 0)
                        return hours * 3600 + minutes * 60 + seconds
    except:
        return None
    return None

def process_java_network(file_path, output_dir, threshold, skip_binarize):
    """Process Java network file using converter and optionally binarizer"""
    base_name = Path(file_path).stem
    converted_file = os.path.join(output_dir, f"{base_name}_converted.txt")
    
    # Convert to DREAM5 format
    subprocess.run([
        'python', 'scripts/dream5_converter.py',
        file_path, converted_file,
    ], check=True)
    
    # Binarize predictions if not skipped
    if not skip_binarize:
        sufix = str(datetime.now().timestamp()).replace(".", "_")
        binary_file = os.path.join(output_dir, f"{base_name}_binary_{sufix}.txt")
        subprocess.run([
            'python', 'scripts/dream5_binarizer.py',
            converted_file, binary_file,
            '--method', 'threshold',
            '--threshold', str(threshold),
            '--sep', ',',
        ], check=True)
        return binary_file
    else:
        return converted_file

def process_geneci_network(file_path, output_dir, threshold, skip_binarize):
    """Process geneci network file using optionally binarizer"""
    if not skip_binarize:
        base_name = Path(file_path).stem
        sufix = str(datetime.now().timestamp()).replace(".", "_")
        binary_file = os.path.join(output_dir, f"{base_name}_binary_{sufix}.txt")
        
        # Binarize predictions
        subprocess.run([
            'python', 'scripts/dream5_binarizer.py',
            file_path, binary_file,
            '--method', 'threshold',
            '--threshold', str(threshold),
            '--sep', ',',
        ], check=True)
        
        return binary_file
    else:
        return file_path

def evaluate_network(file_path, metadata, external_projects_dir):
    """Evaluate network using GENECI evaluate command"""
    synapse_file = os.path.join(
        external_projects_dir,
        'input_data',
        'geneci',
        'DREAM4',
        'EVAL',
        f"pdf_size{metadata['network_id']}.mat"
    )
    
    try:
        result = subprocess.run([
            'geneci', 'evaluate', 'dream-prediction', 'dream-list-of-links',
            '--challenge', 'D4C2',
            '--network-id', metadata['network_id'],
            '--synapse-file', synapse_file,
            '--confidence-list', file_path
        ], capture_output=True, text=True, check=True)
        
        print(
            'geneci', 'evaluate', 'dream-prediction', 'dream-list-of-links',
            '--challenge', 'D4C2',
            '--network-id', metadata['network_id'],
            '--synapse-file', synapse_file,
            '--confidence-list', file_path
        )
        
        # Parse evaluation results
        metrics = {}
        for line in result.stdout.split('\n'):
            if 'AUPR:' in line:
                metrics['aupr'] = float(line.split(':')[1].strip())
            elif 'AUROC:' in line:
                metrics['auroc'] = float(line.split(':')[1].strip())
        
        
        return metrics
    except subprocess.CalledProcessError as e:
        print(f"Evaluation failed for {file_path}: {e}")
        return None

def main():
    args = parse_args()
    results = []
    
    network_files = find_network_files(args.monitoring_dir)
    
    for network_file in network_files:
        print(f"Processing {network_file['path']}...")
        
        # Extract metadata from path
        metadata = extract_metadata_from_path(network_file['path'])
        metadata.update(network_file)
        
        # Get execution time
        markers_pattern = network_file['path'].replace(
            Path(network_file['path']).name, 'execution_markers_*.txt'
        )
        markers_files = glob.glob(markers_pattern)
        if markers_files:
            metadata['execution_time'] = get_execution_time(markers_files[0])
        
        # Process file based on type and options
        if network_file['type'] == 'java':
            processed_file = process_java_network(
                network_file['path'], 
                os.path.dirname(network_file['path']),
                args.threshold,
                args.skip_binarize
            )
        elif network_file['type'] == 'geneci':
            processed_file = process_geneci_network(
                network_file['path'], 
                os.path.dirname(network_file['path']),
                args.threshold,
                args.skip_binarize
            )
        else:
            processed_file = network_file['path']
        
        # Evaluate network
        evaluation = evaluate_network(
            processed_file, 
            metadata,
            args.external_projects
        )
        
        if evaluation:
            metadata.update(evaluation)
            results.append(metadata)
    
    # Create results dataframe and save
    if results:
        df = pd.DataFrame(results)
        df.to_csv(args.output, index=False)
        print(f"Results saved to {args.output}")
    else:
        print("No results to save")

if __name__ == '__main__':
    main()