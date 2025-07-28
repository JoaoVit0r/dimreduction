import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def load_matrix(path, sep=None):
    """
    Load adjacency matrix from CSV/TSV with row and column labels.
    """
    return pd.read_csv(path, sep=sep or r"\s+|,", index_col=0, engine='python')


def matrix_to_list(mat, threshold=0.0):
    """
    Convert adjacency matrix to edge list. Returns DataFrame with columns ['A','B','score'].
    Only entries with score > threshold.
    """
    edges = []
    for src in mat.index:
        for tgt in mat.columns:
            score = mat.at[src, tgt]
            if src != tgt and score > threshold:
                edges.append((src, tgt, score))
    return pd.DataFrame(edges, columns=['A', 'B', 'score'])


def list_to_matrix(edge_list, genes=None):
    """
    Convert edge list DataFrame with ['A','B','score'] to adjacency matrix.
    If genes is provided, use that ordering; else infer.
    """
    if genes is None:
        genes = sorted(set(edge_list['A']).union(edge_list['B']))
    mat = pd.DataFrame(0, index=genes, columns=genes, dtype=float)
    for _, row in edge_list.iterrows():
        mat.at[row['A'], row['B']] = row['score']
    return mat


def evaluate(pred_list, gold_list, genes=None):
    """
    Evaluate predictions against gold standard. Returns dict of metrics.
    """
    all_genes = genes or sorted(set(gold_list['A']).union(gold_list['B']).union(pred_list['A']).union(pred_list['B']))
    # Create binary matrices
    gold_mat = list_to_matrix(gold_list, genes=all_genes)
    pred_mat = list_to_matrix(pred_list, genes=all_genes)
    # Flatten excluding self
    y_true = []
    y_pred = []
    for i in all_genes:
        for j in all_genes:
            if i == j:
                continue
            y_true.append(1 if gold_mat.at[i, j] > 0 else 0)
            y_pred.append(1 if pred_mat.at[i, j] > 0 else 0)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    metrics = {
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'accuracy': accuracy_score(y_true, y_pred),
        'true_positives': int((y_true & y_pred).sum()),
        'false_positives': int((~y_true & y_pred).sum()),
        'false_negatives': int((y_true & ~y_pred).sum()),
        'true_negatives': int((~y_true & ~y_pred).sum()),
    }
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate network predictions against a gold standard.')
    parser.add_argument('--pred-matrix', help='Path to predicted adjacency matrix (CSV/TSV).',
                        required=False)
    parser.add_argument('--pred-list', help='Path to predicted edge list TSV.', required=False)
    parser.add_argument('--gold-matrix', help='Path to gold adjacency matrix CSV/TSV.', required=False)
    parser.add_argument('--gold-list', help='Path to gold edge list TSV.', required=False)
    parser.add_argument('--threshold', type=float, default=0.0,
                        help='Threshold to binarize matrix entries (default=0.0).')
    parser.add_argument('--out-prefix', default='output',
                        help='Prefix for output files.')
    args = parser.parse_args()

    # Load predictions
    if args.pred_matrix:
        pred_mat = load_matrix(args.pred_matrix)
        pred_list = matrix_to_list(pred_mat, threshold=args.threshold)
        pred_list.to_csv(f"{args.out_prefix}_pred_list.tsv", sep='\t', index=False)
        print(f"Predicted list written to {args.out_prefix}_pred_list.tsv")
    elif args.pred_list:
        pred_list = pd.read_csv(args.pred_list, sep='\t')
    else:
        parser.error('Provide either --pred-matrix or --pred-list')

    # Load gold
    if args.gold_matrix:
        gold_mat = load_matrix(args.gold_matrix)
        gold_list = matrix_to_list(gold_mat, threshold=args.threshold)
        gold_list.to_csv(f"{args.out_prefix}_gold_list.tsv", sep='\t', index=False)
        print(f"Gold list written to {args.out_prefix}_gold_list.tsv")
    elif args.gold_list:
        gold_list = pd.read_csv(args.gold_list, sep='\t')
    else:
        parser.error('Provide either --gold-matrix or --gold-list')

    # Evaluate
    metrics = evaluate(pred_list, gold_list)
    print("\nEvaluation metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v}")

if __name__ == '__main__':
    main()
