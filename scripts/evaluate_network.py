import argparse
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

def load_adjacency_matrix(path):
    """
    Load adjacency matrix from a whitespace- or comma-separated file without header.
    """
    return np.loadtxt(path)


def load_gold_list(path, n_genes):
    """
    Load gold standard list of edges (A, B, score) into a binary adjacency matrix.
    Genes labeled G1...Gn in order.
    """
    gold = np.zeros((n_genes, n_genes), dtype=int)
    with open(path) as f:
        for line in f:
            parts = line.strip().split()  # tab or whitespace
            if len(parts) < 2:
                continue
            a_label, b_label = parts[0], parts[1]
            # Convert G1->0 index
            i = int(a_label.lstrip('G')) - 1
            j = int(b_label.lstrip('G')) - 1
            if 0 <= i < n_genes and 0 <= j < n_genes and i != j:
                gold[i, j] = 1
    return gold


def matrix_to_list(matrix, output_path, threshold=None):
    """
    Convert adjacency matrix to list of triples (A, B, score) and save to file.
    If threshold is set, include only entries >= threshold.
    """
    n = matrix.shape[0]
    with open(output_path, 'w') as out:
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                score = matrix[i, j]
                if threshold is None or score >= threshold:
                    out.write(f"G{i+1}\tG{j+1}\t{score}\n")
    print(f"List written to {output_path}")


def list_to_matrix(list_path, n_genes, output_path):
    """
    Convert gold list to a binary adjacency matrix and save as whitespace file.
    """
    mat = load_gold_list(list_path, n_genes)
    np.savetxt(output_path, mat, fmt='%d')
    print(f"Matrix written to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate predicted network against gold standard list.")
    parser.add_argument('--pred-matrix', required=True, help='Path to predicted adjacency matrix (no header)')
    parser.add_argument('--gold-list', required=True, help='Path to gold standard list (A TAB B TAB score)')
    parser.add_argument('--output-prefix', default='output', help='Prefix for optional outputs')
    parser.add_argument('--threshold', type=float, help='Threshold for binarizing predicted matrix (optional)')
    parser.add_argument('--export-matrix2list', action='store_true', help='Export predicted matrix to list')
    parser.add_argument('--export-list2matrix', action='store_true', help='Export gold list to matrix')
    args = parser.parse_args()

    # Load and binarize predictions
    pred_mat = load_adjacency_matrix(args.pred_matrix)
    n = pred_mat.shape[0]
    if args.threshold is not None:
        pred_bin = (pred_mat >= args.threshold).astype(int)
    else:
        pred_bin = pred_mat.astype(int)

    # Load gold standard
    gold_bin = load_gold_list(args.gold_list, n)

    # Flatten ignoring diagonal
    mask = ~np.eye(n, dtype=bool)
    y_true = gold_bin[mask].ravel()
    y_pred = pred_bin[mask].ravel()

    # Compute metrics
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)

    # Print results
    print("Confusion Matrix:")
    print(f"TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")

    # Optional exports
    if args.export_matrix2list:
        matrix_to_list(pred_mat, f"{args.output_prefix}_pred_list.tsv", args.threshold)
    if args.export_list2matrix:
        list_to_matrix(args.gold_list, n, f"{args.output_prefix}_gold_matrix.txt")

if __name__ == '__main__':
    main()
