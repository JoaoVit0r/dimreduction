#!/usr/bin/env python3
"""
Script to calculate additional evaluation metrics (TP, TN, FP, FN) by comparing
a predicted network against a gold standard reference.
"""

import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='Calculate TP, TN, FP, FN metrics for network predictions')
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
    return parser.parse_args()

def read_matrix_file(file_path, has_header=False, sep=','):
    """Read a matrix file and return as DataFrame"""
    if sep == "tab":
        sep = "\t"

    if has_header:
        return pd.read_csv(file_path, sep=sep, index_col=0)
    else:
        return pd.read_csv(file_path, sep=sep, header=None)

def read_list_file(file_path, has_header=False, sep=','):
    """Read a list file and return as DataFrame"""
    if sep == "tab":
        sep = "\t"

    if has_header:
        return pd.read_csv(file_path, sep=sep)
    else:
        return pd.read_csv(file_path, header=None, names=['regulator', 'target', 'weight'], sep=sep)

def matrix_to_edges(matrix_df, threshold=0.5):
    """Convert a matrix to a set of edges with weights"""
    edges = set()
    nodes = matrix_df.index if matrix_df.index.name else range(matrix_df.shape[0])
    
    for i, regulator in enumerate(nodes):
        for j, target in enumerate(nodes):
            if i != j:  # Skip self-loops
                weight = matrix_df.iloc[i, j]
                if abs(weight) >= threshold:
                    edges.add((regulator, target, weight))
    
    return edges

def list_to_edges(list_df, threshold=0.5):
    """Convert a list to a set of edges with weights"""
    edges = set()
    
    for _, row in list_df.iterrows():
        regulator = row['regulator']
        target = row['target']
        weight = row['weight']
        
        if regulator != target and abs(weight) >= threshold:  # Skip self-loops
            edges.add((regulator, target, weight))
    
    return edges

def calculate_metrics(pred_edges, gold_edges):
    """Calculate TP, TN, FP, FN metrics"""
    # Get all possible edges from both sets
    all_edges = set()
    for edge in pred_edges | gold_edges:
        all_edges.add((edge[0], edge[1]))  # Just regulator, target pairs
    
    # Initialize counts
    tp = tn = fp = fn = 0
    missing_in_pred = []
    missing_in_gold = []
    
    for edge in all_edges:
        regulator, target = edge
        
        # Check if edge exists in prediction and gold standard
        pred_exists = any(e[0] == regulator and e[1] == target for e in pred_edges)
        gold_exists = any(e[0] == regulator and e[1] == target for e in gold_edges)
        
        if pred_exists and gold_exists:
            tp += 1
        elif not pred_exists and not gold_exists:
            tn += 1
        elif pred_exists and not gold_exists:
            fp += 1
            missing_in_gold.append((regulator, target))
        else:  # not pred_exists and gold_exists
            fn += 1
            missing_in_pred.append((regulator, target))
    
    return tp, tn, fp, fn, missing_in_pred, missing_in_gold

def main():
    args = parse_args()
    
    # Read prediction file
    if args.pred_format == 'matrix':
        pred_df = read_matrix_file(args.prediction, args.pred_has_header, args.pred_sep)
        pred_edges = matrix_to_edges(pred_df, args.threshold)
    else:
        pred_df = read_list_file(args.prediction, args.pred_has_header, args.pred_sep)
        pred_edges = list_to_edges(pred_df, args.threshold)
    
    # Read gold standard file
    if args.gold_format == 'matrix':
        gold_df = read_matrix_file(args.gold_standard, args.gold_has_header, args.gold_sep)
        gold_edges = matrix_to_edges(gold_df, 0.5)  # Gold standard is typically binary
    else:
        gold_df = read_list_file(args.gold_standard, args.gold_has_header, args.gold_sep)
        gold_edges = list_to_edges(gold_df, 0.5)  # Gold standard is typically binary
    
    # Calculate metrics
    tp, tn, fp, fn, missing_in_pred, missing_in_gold = calculate_metrics(pred_edges, gold_edges)
    
    # Calculate additional metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    
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
        'Total Possible Edges': tp + tn + fp + fn,
        'Edges Missing in Prediction': len(missing_in_pred),
        'Edges Missing in Gold Standard': len(missing_in_gold)
    }
    
    # Save results to CSV
    results_df = pd.DataFrame([results])
    results_df.to_csv(args.output, index=False)
    print(f"Evaluation metrics saved to {args.output}")
    
    # Print summary
    print("\nEvaluation Metrics Summary:")
    print("=" * 40)
    for metric, value in results.items():
        if isinstance(value, float):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")
    
    # Print information about missing edges if any
    if missing_in_pred:
        print(f"\nEdges present in gold standard but missing in prediction: {len(missing_in_pred)}")
        if len(missing_in_pred) <= 10:  # Show up to 10 examples
            for i, edge in enumerate(missing_in_pred[:10]):
                print(f"  {i+1}. {edge[0]} -> {edge[1]}")
        else:
            print("  (Too many to display, see detailed output if needed)")
    
    if missing_in_gold:
        print(f"\nEdges present in prediction but missing in gold standard: {len(missing_in_gold)}")
        if len(missing_in_gold) <= 10:  # Show up to 10 examples
            for i, edge in enumerate(missing_in_gold[:10]):
                print(f"  {i+1}. {edge[0]} -> {edge[1]}")
        else:
            print("  (Too many to display, see detailed output if needed)")

if __name__ == '__main__':
    main()