#!/usr/bin/env python3
"""
DREAM5 Network Inference Challenge - Adjacency Matrix Converter

This script converts an adjacency matrix (tab-separated, no header) into the 
DREAM5 submission format for network inference predictions.

Input: Adjacency matrix file (customizable delimiter, optional header)
Output: DREAM5 format file with columns: TF_gene, target_gene, confidence_score

Usage:
    python dream5_converter.py input_matrix.txt output_predictions.txt [options]
"""

import numpy as np
import pandas as pd
import argparse
import sys
from pathlib import Path


def load_adjacency_matrix(matrix_file, delimiter='\t', has_header=False):
    """Load adjacency matrix from delimited file with optional header."""
    try:
        if has_header:
            # Use pandas to handle headers more robustly
            df = pd.read_csv(matrix_file, sep=delimiter, index_col=0)
            matrix = df.values
            print(f"Loaded adjacency matrix of shape: {matrix.shape} (with header)")
            return matrix
        else:
            matrix = np.loadtxt(matrix_file, delimiter=delimiter)
            print(f"Loaded adjacency matrix of shape: {matrix.shape} (no header)")
            return matrix
    except Exception as e:
        print(f"Error loading adjacency matrix: {e}")
        print(f"Make sure the file uses '{delimiter}' as delimiter and has_header={has_header}")
        sys.exit(1)


def load_gene_labels(labels_file, n_genes):
    """Load gene labels or generate default labels (G1, G2, etc.)."""
    if labels_file and Path(labels_file).exists():
        try:
            with open(labels_file, 'r') as f:
                labels = [line.strip() for line in f if line.strip()]
            if len(labels) != n_genes:
                print(f"Warning: Number of labels ({len(labels)}) doesn't match matrix size ({n_genes})")
                print("Using default gene labels instead.")
                return [f"G{i+1}" for i in range(n_genes)]
            return labels
        except Exception as e:
            print(f"Error loading gene labels: {e}")
            print("Using default gene labels instead.")
    
    # Generate default labels
    return [f"G{i+1}" for i in range(n_genes)]


def load_tf_list(tf_file, all_genes):
    """Load transcription factor list or use all genes as potential TFs."""
    if tf_file and Path(tf_file).exists():
        try:
            with open(tf_file, 'r') as f:
                tf_genes = [line.strip() for line in f if line.strip()]
            # Filter to only include TFs that are in our gene list
            valid_tfs = [tf for tf in tf_genes if tf in all_genes]
            print(f"Loaded {len(valid_tfs)} transcription factors")
            return valid_tfs
        except Exception as e:
            print(f"Error loading TF list: {e}")
            print("Using all genes as potential transcription factors.")
    
    # Use all genes as potential TFs
    print("Using all genes as potential transcription factors.")
    return all_genes


def adjacency_to_dream5(matrix, gene_labels, tf_list, max_predictions=100000):
    """
    Convert adjacency matrix to DREAM5 submission format.
    
    Args:
        matrix: numpy array representing the adjacency matrix
        gene_labels: list of gene names/labels
        tf_list: list of transcription factor genes
        max_predictions: maximum number of predictions to output
    
    Returns:
        pandas DataFrame with columns: tf_gene, target_gene, confidence_score
    """
    predictions = []
    n_genes = len(gene_labels)
    
    # Create mapping from gene names to indices
    gene_to_idx = {gene: idx for idx, gene in enumerate(gene_labels)}
    
    # Get indices of all transcription factors
    tf_indices = [gene_to_idx[tf] for tf in tf_list if tf in gene_to_idx]
    
    print(f"Processing {len(tf_indices)} transcription factors...")
    
    # Extract all non-zero regulatory relationships
    for tf_idx in tf_indices:
        tf_gene = gene_labels[tf_idx]
        
        for target_idx in range(n_genes):
            # Skip self-interactions (auto-regulatory loops)
            if tf_idx == target_idx:
                continue
                
            confidence = matrix[tf_idx, target_idx]
            
            # Only include non-zero predictions
            if confidence != 0:
                target_gene = gene_labels[target_idx]
                predictions.append({
                    'tf_gene': tf_gene,
                    'target_gene': target_gene,
                    'confidence_score': confidence
                })
    
    # Convert to DataFrame and sort by confidence (descending)
    df = pd.DataFrame(predictions)
    
    if len(df) == 0:
        print("Warning: No non-zero predictions found in the adjacency matrix!")
        return df
    
    # Sort by confidence score in descending order (most confident first)
    df = df.sort_values('confidence_score', ascending=False)
    
    # Limit to maximum number of predictions
    if len(df) > max_predictions:
        print(f"Limiting output to top {max_predictions} predictions (from {len(df)} total)")
        df = df.head(max_predictions)
    
    return df


def save_dream5_format(df, output_file):
    """Save predictions in DREAM5 format."""
    try:
        # Save as comma-separated file without index and header
        df.to_csv(output_file, sep=',', index=False, header=False)
        print(f"Saved {len(df)} predictions to: {output_file}")
    except Exception as e:
        print(f"Error saving output file: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Convert adjacency matrix to DREAM5 submission format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage with default settings (tab-separated, no header)
    python dream5_converter.py adjacency_matrix.txt predictions.txt
    
    # Comma-separated file with header
    python dream5_converter.py matrix.csv predictions.txt --separator "," --header
    
    # Space-separated file without header
    python dream5_converter.py matrix.txt predictions.txt --sep " "
    
    # With custom gene labels and TF list
    python dream5_converter.py matrix.txt predictions.txt --gene-labels gene_labels.txt --tf-list tf_list.txt --header
        """
    )
    
    parser.add_argument('matrix_file', help='Input adjacency matrix file')
    parser.add_argument('output_file', help='Output predictions file in DREAM5 format')
    parser.add_argument('--gene-labels', help='Optional: file with gene labels (one per line)')
    parser.add_argument('--tf-list', help='Optional: file with transcription factor list (one per line)')
    
    parser.add_argument('--separator', '--sep', default='\t', 
                        help='Column separator in matrix file (default: tab)')
    parser.add_argument('--header', action='store_true',
                        help='Matrix file has header row (default: no header)')
    parser.add_argument('--max-predictions', type=int, default=100000,
                        help='Maximum number of predictions to output (default: 100000)')
    
    args = parser.parse_args()
    
    # Load adjacency matrix
    print("Loading adjacency matrix...")
    matrix = load_adjacency_matrix(args.matrix_file, delimiter=args.separator, has_header=args.header)
    n_genes = matrix.shape[0]
    
    if matrix.shape[0] != matrix.shape[1]:
        print(f"Error: Adjacency matrix must be square! Got shape: {matrix.shape}")
        sys.exit(1)
    
    # Load or generate gene labels
    print("Loading gene labels...")
    gene_labels = load_gene_labels(args.gene_labels, n_genes)
    
    # Load transcription factor list
    print("Loading transcription factors...")
    tf_list = load_tf_list(args.tf_list, gene_labels)
    
    # Convert to DREAM5 format
    print("Converting to DREAM5 format...")
    predictions_df = adjacency_to_dream5(matrix, gene_labels, tf_list, args.max_predictions)
    
    if len(predictions_df) == 0:
        print("No predictions generated. Check your input matrix.")
        sys.exit(1)
    
    # Save output
    save_dream5_format(predictions_df, args.output_file)
    
    print(f"\nConversion completed successfully!")
    print(f"Generated {len(predictions_df)} regulatory link predictions")
    print(f"Confidence scores range: {predictions_df['confidence_score'].min():.4f} - {predictions_df['confidence_score'].max():.4f}")


if __name__ == "__main__":
    main()