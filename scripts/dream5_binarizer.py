#!/usr/bin/env python3
"""
DREAM5 Network Inference Challenge - Predictions Binarizer

This script converts DREAM5 format predictions with continuous confidence scores
to binary predictions (0 or 1) based on various thresholding methods.

Input: DREAM5 format file with columns: TF_gene, target_gene, confidence_score
Output: DREAM5 format file with binary scores (0 or 1)

Usage:
    python dream5_binarizer.py input_predictions.txt output_binary.txt [options]
"""

import pandas as pd
import numpy as np
import argparse
import sys
from pathlib import Path


def load_dream5_predictions(input_file, sep='\t'):
    """Load DREAM5 format predictions file."""
    try:
        # Read tab-separated file without header
        df = pd.read_csv(input_file, sep=sep, header=None, 
                        names=['tf_gene', 'target_gene', 'confidence_score'])
        
        print(f"Loaded {len(df)} predictions from {input_file}")
        print(f"Confidence score range: {df['confidence_score'].min():.4f} - {df['confidence_score'].max():.4f}")
        
        return df
    except Exception as e:
        print(f"Error loading predictions file: {e}")
        sys.exit(1)


def binarize_by_threshold(df, threshold=0.5):
    """Convert predictions to binary using a fixed threshold."""
    binary_df = df.copy()
    binary_df['confidence_score'] = (df['confidence_score'] >= threshold).astype(int)
    
    n_ones = (binary_df['confidence_score'] == 1).sum()
    n_zeros = (binary_df['confidence_score'] == 0).sum()
    
    print(f"Threshold method (threshold={threshold}):")
    print(f"  - Predictions with score 1: {n_ones}")
    print(f"  - Predictions with score 0: {n_zeros}")
    
    return binary_df


def binarize_by_percentile(df, percentile=50):
    """Convert predictions to binary using percentile cutoff."""
    threshold = np.percentile(df['confidence_score'], percentile)
    
    binary_df = df.copy()
    binary_df['confidence_score'] = (df['confidence_score'] >= threshold).astype(int)
    
    n_ones = (binary_df['confidence_score'] == 1).sum()
    n_zeros = (binary_df['confidence_score'] == 0).sum()
    
    print(f"Percentile method (percentile={percentile}, threshold={threshold:.4f}):")
    print(f"  - Predictions with score 1: {n_ones}")
    print(f"  - Predictions with score 0: {n_zeros}")
    
    return binary_df


def binarize_by_top_k(df, k=1000):
    """Convert predictions to binary by keeping top K predictions as 1."""
    binary_df = df.copy()
    binary_df['confidence_score'] = 0  # Initialize all to 0
    
    # Sort by confidence score and keep top k as 1
    top_k = min(k, len(df))
    sorted_df = df.sort_values('confidence_score', ascending=False)
    top_indices = sorted_df.head(top_k).index
    
    binary_df.loc[top_indices, 'confidence_score'] = 1
    
    n_ones = (binary_df['confidence_score'] == 1).sum()
    n_zeros = (binary_df['confidence_score'] == 0).sum()
    
    print(f"Top-K method (K={k}):")
    print(f"  - Predictions with score 1: {n_ones}")
    print(f"  - Predictions with score 0: {n_zeros}")
    
    return binary_df


def binarize_by_mean(df):
    """Convert predictions to binary using mean as threshold."""
    threshold = df['confidence_score'].mean()
    
    binary_df = df.copy()
    binary_df['confidence_score'] = (df['confidence_score'] >= threshold).astype(int)
    
    n_ones = (binary_df['confidence_score'] == 1).sum()
    n_zeros = (binary_df['confidence_score'] == 0).sum()
    
    print(f"Mean method (threshold={threshold:.4f}):")
    print(f"  - Predictions with score 1: {n_ones}")
    print(f"  - Predictions with score 0: {n_zeros}")
    
    return binary_df


def binarize_by_median(df):
    """Convert predictions to binary using median as threshold."""
    threshold = df['confidence_score'].median()
    
    binary_df = df.copy()
    binary_df['confidence_score'] = (df['confidence_score'] >= threshold).astype(int)
    
    n_ones = (binary_df['confidence_score'] == 1).sum()
    n_zeros = (binary_df['confidence_score'] == 0).sum()
    
    print(f"Median method (threshold={threshold:.4f}):")
    print(f"  - Predictions with score 1: {n_ones}")
    print(f"  - Predictions with score 0: {n_zeros}")
    
    return binary_df


def binarize_by_otsu(df):
    """Convert predictions to binary using Otsu's method (optimal threshold)."""
    scores = df['confidence_score'].values
    
    # Try different thresholds and find the one that minimizes within-class variance
    thresholds = np.linspace(scores.min(), scores.max(), 100)
    best_threshold = 0.5
    best_variance = float('inf')
    
    for threshold in thresholds:
        class_0 = scores[scores < threshold]
        class_1 = scores[scores >= threshold]
        
        if len(class_0) == 0 or len(class_1) == 0:
            continue
            
        # Calculate weighted within-class variance
        w0 = len(class_0) / len(scores)
        w1 = len(class_1) / len(scores)
        var0 = np.var(class_0) if len(class_0) > 1 else 0
        var1 = np.var(class_1) if len(class_1) > 1 else 0
        
        within_class_variance = w0 * var0 + w1 * var1
        
        if within_class_variance < best_variance:
            best_variance = within_class_variance
            best_threshold = threshold
    
    binary_df = df.copy()
    binary_df['confidence_score'] = (df['confidence_score'] >= best_threshold).astype(int)
    
    n_ones = (binary_df['confidence_score'] == 1).sum()
    n_zeros = (binary_df['confidence_score'] == 0).sum()
    
    print(f"Otsu method (optimal threshold={best_threshold:.4f}):")
    print(f"  - Predictions with score 1: {n_ones}")
    print(f"  - Predictions with score 0: {n_zeros}")
    
    return binary_df


def save_binary_predictions(df, output_file, keep_zeros=True):
    """Save binary predictions in DREAM5 format."""
    
    # Option to remove zero predictions to reduce file size
    if not keep_zeros:
        df = df[df['confidence_score'] == 1]
        print(f"Keeping only predictions with score 1: {len(df)} predictions")
    
    try:
        # Save as tab-separated file without index and header
        df.to_csv(output_file, sep=',', index=False, header=False)
        print(f"Saved {len(df)} binary predictions to: {output_file}")
    except Exception as e:
        print(f"Error saving output file: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Convert DREAM5 predictions to binary (0 or 1) format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Binarization Methods:
    threshold   : Use fixed threshold (default: 0.5)
    percentile  : Use percentile cutoff (default: 50th percentile)
    top-k       : Keep top K predictions as 1, rest as 0 (default: K=1000)
    mean        : Use mean confidence score as threshold
    median      : Use median confidence score as threshold  
    otsu        : Use Otsu's optimal threshold method

Examples:
    # Fixed threshold of 0.7
    python dream5_binarizer.py predictions.txt binary.txt --method threshold --threshold 0.7
    
    # Keep top 500 predictions
    python dream5_binarizer.py predictions.txt binary.txt --method top-k --top-k 500
    
    # Use 75th percentile
    python dream5_binarizer.py predictions.txt binary.txt --method percentile --percentile 75
    
    # Remove zero predictions from output
    python dream5_binarizer.py predictions.txt binary.txt --remove-zeros
        """
    )
    
    parser.add_argument('input_file', help='Input DREAM5 predictions file')
    parser.add_argument('output_file', help='Output binary predictions file')
    
    parser.add_argument('--method', choices=['threshold', 'percentile', 'top-k', 'mean', 'median', 'otsu'],
                        default='threshold', help='Binarization method (default: threshold)')
    
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold value for threshold method (default: 0.5)')
    
    parser.add_argument('--percentile', type=float, default=50,
                        help='Percentile cutoff for percentile method (default: 50)')
    
    parser.add_argument('--top-k', type=int, default=1000,
                        help='Number of top predictions to keep for top-k method (default: 1000)')
    
    parser.add_argument('--remove-zeros', action='store_true',
                        help='Remove predictions with score 0 from output')
    
    parser.add_argument('--sep', type=str, default="\t",
                        help='Custom separated file (default: \\t)')
    
    args = parser.parse_args()
    
    # Load predictions
    print("Loading DREAM5 predictions...")
    df = load_dream5_predictions(args.input_file, args.sep)
    
    # Apply binarization method
    print(f"\nApplying {args.method} binarization...")
    
    if args.method == 'threshold':
        binary_df = binarize_by_threshold(df, args.threshold)
    elif args.method == 'percentile':
        binary_df = binarize_by_percentile(df, args.percentile)
    elif args.method == 'top-k':
        binary_df = binarize_by_top_k(df, args.top_k)
    elif args.method == 'mean':
        binary_df = binarize_by_mean(df)
    elif args.method == 'median':
        binary_df = binarize_by_median(df)
    elif args.method == 'otsu':
        binary_df = binarize_by_otsu(df)
    
    # Save binary predictions
    print(f"\nSaving binary predictions...")
    save_binary_predictions(binary_df, args.output_file, keep_zeros=not args.remove_zeros)
    
    print(f"\nBinarization completed successfully!")


if __name__ == "__main__":
    main()
