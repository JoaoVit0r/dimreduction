#!/usr/bin/env python3
"""
Script to calculate additional evaluation metrics (TP, TN, FP, FN) by comparing
a predicted network against a gold standard reference, following DREAM challenge methodology.
"""

import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='Calculate TP, TN, FP, FN metrics for network predictions following DREAM challenge methodology')
    parser.add_argument('--prediction', required=True, help='Path to prediction file (matrix or list format)')
    parser.add_argument('--gold-standard', required=True, help='Path to gold standard file (matrix or list format)')
    parser.add_argument('--pred-format', choices=['matrix', 'list'], required=True, 
                        help='Format of prediction file')
    parser.add_argument('--gold-format', choices=['matrix', 'list'], required=True,
                        help='Format of gold standard file')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold for binarizing predictions (default: 0.5)')
    parser.add_argument('--output', default='evaluation_metrics.csv',
                        help='Output CSV file name')
    parser.add_argument('--pred-has-header', action='store_true',
                        help='Prediction file has header row')
    parser.add_argument('--gold-has-header', action='store_true',
                        help='Gold standard file has header row')
    parser.add_argument('--pred-sep', default=',', help='Delimiter for prediction file (default: comma)')
    parser.add_argument('--gold-sep', default=',', help='Delimiter for gold standard files (default: comma)')
    parser.add_argument('--max-edges', type=int, default=100000,
                        help='Maximum number of edges to consider (DREAM challenge uses 100,000)')
    parser.add_argument('--gene-prefix', default='G', help='Prefix for gene names when matrix has no header (default: G)')
    parser.add_argument('--transcription-factors', help='Path to transcription factors file (one gene per line)')
    return parser.parse_args()

def read_tfs_file(tfs_file_path):
    """Read transcription factors file and return as set"""
    tfs = set()
    with open(tfs_file_path, 'r') as f:
        for line in f:
            tfs.add(line.strip())
    return tfs

def read_matrix_file(file_path, has_header=False, sep=',', gene_prefix='G'):
    """Read a matrix file and return as DataFrame"""
    if sep == "tab":
        sep = "\t"

    if has_header:
        df = pd.read_csv(file_path, sep=sep, index_col=0)
    else:
        df = pd.read_csv(file_path, sep=sep, header=None)
        # Create gene names if no header
        n_genes = df.shape[0]
        gene_names = [f"{gene_prefix}{i+1}" for i in range(n_genes)]
        df.index = gene_names
        df.columns = gene_names
    
    return df

def read_list_file(file_path, has_header=False, sep=','):
    """Read a list file and return as DataFrame"""
    if sep == "tab":
        sep = "\t"

    if has_header:
        return pd.read_csv(file_path, sep=sep)
    else:
        return pd.read_csv(file_path, header=None, names=['regulator', 'target', 'weight'], sep=sep)

def matrix_to_edges(matrix_df, threshold=0.5, tfs_set=None):
    """Convert a matrix to a set of edges with weights, optionally filtering by TFs"""
    edges = set()
    nodes = matrix_df.index
    
    for i, regulator in enumerate(nodes):
        # Skip if regulator is not in TFs set (if provided)
        if tfs_set is not None and regulator not in tfs_set:
            continue
            
        for j, target in enumerate(nodes):
            if i != j:  # Skip self-loops
                weight = matrix_df.iloc[i, j]
                if abs(weight) >= threshold:
                    edges.add((regulator, target, weight))
    
    return edges

def list_to_edges(list_df, threshold=0.5, tfs_set=None):
    """Convert a list to a set of edges with weights, optionally filtering by TFs"""
    edges = set()
    
    for regulator, target, weight in zip(list_df.iloc[:, 0].astype(str), list_df.iloc[:, 1].astype(str), list_df.iloc[:, 2].astype(float)):
        
        # Skip if regulator is not in TFs set (if provided)
        if tfs_set is not None and regulator not in tfs_set:
            continue
            
        if regulator != target and abs(weight) >= threshold:  # Skip self-loops
            edges.add((regulator, target, weight))
    
    return edges

def get_gold_standard_genes(gold_df, gold_format):
    """Extract all genes present in the gold standard"""
    regulator = set()
    target = set()
    
    if gold_format == 'matrix':
        # For matrix format, genes are row and column names
        regulator.update(gold_df.index.tolist())
        target.update(gold_df.columns.tolist())
    else:
        # For list format, genes are in regulator and target columns
        regulator.update(gold_df['regulator'].tolist())
        target.update(gold_df['target'].tolist())
        
    genes = {
        "regulator": (regulator),
        "target": (target)
    }
    
    return genes

def filter_and_truncate_predictions(pred_edges, gold_genes, max_edges=100000):
    """Filter predictions to include only genes in gold standard and truncate to max_edges"""
    # Filter out predictions with genes not in gold standard
    filtered_edges = [edge for edge in pred_edges if edge[0] in gold_genes['regulator'] and edge[1] in gold_genes['target']]
    
    # Sort by absolute weight in descending order
    filtered_edges.sort(key=lambda x: abs(x[2]), reverse=True)
    
    # Truncate to max_edges
    truncated_edges = filtered_edges[:max_edges]
    
    return truncated_edges

def calculate_metrics_dream(pred_edges, gold_edges, gold_genes, max_edges=100000):
    """
    Calculate TP, TN, FP, FN metrics following DREAM challenge methodology
    """
    # Filter and truncate predictions
    pred_edges_filtered = filter_and_truncate_predictions(pred_edges, gold_genes, max_edges)
    
    # Convert to sets for faster lookup
    pred_set = {(edge[0], edge[1]) for edge in pred_edges_filtered}
    gold_set = {(edge[0], edge[1]) for edge in gold_edges}
    
    # Get all possible edges between gold standard genes (excluding self-loops)
    all_possible_edges = set()
    for regulator in gold_genes['regulator']:
        for target in gold_genes['target']:
            if regulator != target:
                all_possible_edges.add((regulator, target))
    
    # Initialize counts
    tp = tn = fp = fn = 0
    
    # Count true positives and false positives
    for edge in pred_set:
        if edge in gold_set:
            tp += 1
        else:
            fp += 1
    
    # Count false negatives (edges in gold standard but not in predictions)
    for edge in gold_set:
        if edge not in pred_set:
            fn += 1
    
    # Count true negatives (edges not in gold standard and not in predictions)
    tn = len(all_possible_edges) - tp - fp - fn
    
    # Calculate precision, recall, etc.
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    
    return tp, tn, fp, fn, precision, recall, f1_score, accuracy, len(all_possible_edges)

def main():
    args = parse_args()
    
    # Read transcription factors if provided
    tfs_set = None
    if args.transcription_factors:
        tfs_set = read_tfs_file(args.transcription_factors)
        print(f"Loaded {len(tfs_set)} transcription factors")
    
    # Read gold standard first to get gene names
    if args.gold_format == 'matrix':
        gold_df = read_matrix_file(args.gold_standard, args.gold_has_header, args.gold_sep, args.gene_prefix)
        gold_edges = matrix_to_edges(gold_df, 0.5, tfs_set)  # Gold standard is typically binary
    else:
        gold_df = read_list_file(args.gold_standard, args.gold_has_header, args.gold_sep)
        gold_edges = list_to_edges(gold_df, 0.5, tfs_set)  # Gold standard is typically binary
    
    # Get all genes in gold standard
    gold_genes = get_gold_standard_genes(gold_df, args.gold_format)
    
    # Read prediction file
    if args.pred_format == 'matrix':
        # Use the same gene prefix as gold standard if available
        pred_df = read_matrix_file(args.prediction, args.pred_has_header, args.pred_sep, args.gene_prefix)
        pred_edges = matrix_to_edges(pred_df, args.threshold, tfs_set)
    else:
        pred_df = read_list_file(args.prediction, args.pred_has_header, args.pred_sep)
        pred_edges = list_to_edges(pred_df, args.threshold, tfs_set)
    
    # Calculate metrics following DREAM methodology
    tp, tn, fp, fn, precision, recall, f1_score, accuracy, total_possible = calculate_metrics_dream(
        list(pred_edges), list(gold_edges), gold_genes, args.max_edges
    )
    
    # Create results dictionary
    results = {
        'True Positives (TP)': tp,
        'True Negatives (TN)': tn,
        'False Positives (FP)': fp,
        'False Negatives (FN)': fn,
        'Precision': precision,
        'Recall (Sensitivity)': recall,
        'F1 Score': f1_score,
        'Accuracy': accuracy,
        'Total Possible Edges': total_possible,
        'Edges in Prediction': tp + fp,
        'Edges in Gold Standard': tp + fn
    }
    
    # Save results to CSV
    results_df = pd.DataFrame([results])
    results_df.to_csv(args.output, index=False)
    print(f"Evaluation metrics saved to {args.output}")
    
    # Print summary
    print("\nEvaluation Metrics Summary (DREAM Challenge Methodology):")
    print("=" * 60)
    for metric, value in results.items():
        if isinstance(value, float):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")
    
    print(f"\nNote: Predictions were truncated to {args.max_edges} edges and")
    print("edges involving genes not in the gold standard were ignored.")
    
    if args.transcription_factors:
        print("Only edges with regulators from the transcription factors list were considered.")

if __name__ == '__main__':
    main()