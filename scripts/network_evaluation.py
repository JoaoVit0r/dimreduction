import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, precision_recall_curve, roc_curve, auc,
    precision_score, recall_score, f1_score, average_precision_score
)
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings('ignore')

PLOT_SHOW = False;

class NetworkEvaluator:
    def __init__(self):
        self.gold_standard_matrix = None
        self.predicted_matrix = None
        self.gene_labels = None
        
    def load_gold_standard(self, file_path, num_genes_B=None):
        """
        Load gold standard from tab-separated file format.
        
        Args:
            file_path: Path to the gold standard file (tab-separated: A \t B \t X)
            num_genes_B: Total number of genes B in the full network (if not provided, will be inferred)
        """
        # Read the tab-separated file
        df = pd.read_csv(file_path, sep='\t', header=None, names=['Gene_A', 'Gene_B', 'Score'])
        
        # Get unique genes from the gold standard
        unique_A = sorted(df['Gene_A'].unique())
        unique_B = sorted(df['Gene_B'].unique())
        
        # The gene labels should be the unique_B genes that appear in gold standard
        self.gene_labels = unique_B.copy()
        
        # If num_genes_B is provided, it means the full network has more genes than in gold standard
        # We need to create a mapping to the full gene space
        if num_genes_B is not None:
            # Create full gene space (G1, G2, ..., G_num_genes_B)
            full_gene_space = [f'G{i+1}' for i in range(num_genes_B)]
            
            # Create mapping from gold standard genes to full gene space indices
            gene_B_to_full_idx = {}
            for gene in unique_B:
                if gene in full_gene_space:
                    gene_B_to_full_idx[gene] = full_gene_space.index(gene)
            
            # Initialize gold standard matrix (SizeA x num_genes_B)
            self.gold_standard_matrix = np.zeros((len(unique_A), num_genes_B))
            
            # Create mapping for Gene_A to matrix row index
            gene_A_to_idx = {gene: idx for idx, gene in enumerate(unique_A)}
            
            # Fill the matrix
            for _, row in df.iterrows():
                gene_a = row['Gene_A']
                gene_b = row['Gene_B']
                score = row['Score']
                
                if gene_a in gene_A_to_idx and gene_b in gene_B_to_full_idx:
                    a_idx = gene_A_to_idx[gene_a]
                    b_idx = gene_B_to_full_idx[gene_b]
                    self.gold_standard_matrix[a_idx, b_idx] = score
            
            # Update gene_labels to reflect the full gene space
            self.gene_labels = full_gene_space
            
        else:
            # Use only genes present in gold standard
            # Initialize gold standard matrix (SizeA x SizeB)
            self.gold_standard_matrix = np.zeros((len(unique_A), len(unique_B)))
            
            # Create mappings
            gene_A_to_idx = {gene: idx for idx, gene in enumerate(unique_A)}
            gene_B_to_idx = {gene: idx for idx, gene in enumerate(unique_B)}
            
            # Fill the matrix
            for _, row in df.iterrows():
                gene_a = row['Gene_A']
                gene_b = row['Gene_B']
                score = row['Score']
                
                a_idx = gene_A_to_idx[gene_a]
                b_idx = gene_B_to_idx[gene_b]
                self.gold_standard_matrix[a_idx, b_idx] = score
        
        print(f"Gold standard matrix shape: {self.gold_standard_matrix.shape}")
        print(f"Unique TFs (Gene A): {len(unique_A)} - {unique_A[:5]}{'...' if len(unique_A) > 5 else ''}")
        print(f"Unique target genes (Gene B): {len(unique_B)} - {unique_B[:5]}{'...' if len(unique_B) > 5 else ''}")
        print(f"Gene labels length: {len(self.gene_labels)}")
        
        # Store the TF labels for reference
        self.tf_labels = unique_A
        
        return self.gold_standard_matrix
    
    def get_gene_mapping_info(self):
        """
        Get information about gene mapping for debugging and verification.
        """
        if self.gold_standard_matrix is None:
            print("Gold standard not loaded yet.")
            return
        
        info = {
            'gold_standard_shape': self.gold_standard_matrix.shape,
            'num_tf_genes': len(self.tf_labels),
            'num_target_genes': len(self.gene_labels),
            'tf_genes': self.tf_labels,
            'target_genes': self.gene_labels,
            'non_zero_entries': np.count_nonzero(self.gold_standard_matrix),
            'total_possible_entries': self.gold_standard_matrix.size,
            'sparsity': 1 - (np.count_nonzero(self.gold_standard_matrix) / self.gold_standard_matrix.size)
        }
        
        print("Gene Mapping Information:")
        print("-" * 30)
        print(f"Gold standard matrix shape: {info['gold_standard_shape']}")
        print(f"Number of TF genes (rows): {info['num_tf_genes']}")
        print(f"Number of target genes (columns): {info['num_target_genes']}")
        print(f"Non-zero entries: {info['non_zero_entries']}")
        print(f"Total possible entries: {info['total_possible_entries']}")
        print(f"Sparsity: {info['sparsity']:.4f}")
        
        if len(self.tf_labels) <= 20:
            print(f"TF genes: {self.tf_labels}")
        else:
            print(f"TF genes (first 10): {self.tf_labels[:10]}...")
            
        if len(self.gene_labels) <= 20:
            print(f"Target genes: {self.gene_labels}")
        else:
            print(f"Target genes (first 10): {self.gene_labels[:10]}...")
        
        return info
    
    def debug_gene_mapping(self):
        """
        Debug method to check gene name to index mapping.
        """
        if self.gene_labels is None:
            print("Gene labels not loaded yet.")
            return
        
        print("Gene Mapping Debug Information:")
        print("-" * 40)
        
        print("\nTarget genes (first 10):")
        for i, gene in enumerate(self.gene_labels[:10]):
            if gene.startswith('G'):
                try:
                    gene_idx = int(gene[1:]) - 1
                    print(f"  {gene} -> predicted matrix index {gene_idx}")
                except ValueError:
                    print(f"  {gene} -> ERROR: cannot parse index")
        
        if hasattr(self, 'tf_labels') and self.tf_labels is not None:
            print("\nTF genes (first 10):")
            for i, tf in enumerate(self.tf_labels[:10]):
                if tf.startswith('G'):
                    try:
                        tf_idx = int(tf[1:]) - 1
                        print(f"  {tf} -> predicted matrix index {tf_idx}")
                    except ValueError:
                        print(f"  {tf} -> ERROR: cannot parse index")
        
        if self.predicted_matrix is not None:
            print(f"\nPredicted matrix dimensions: {self.predicted_matrix.shape}")
            max_gene_idx = max([int(gene[1:]) - 1 for gene in self.gene_labels if gene.startswith('G')])
            if hasattr(self, 'tf_labels'):
                max_tf_idx = max([int(tf[1:]) - 1 for tf in self.tf_labels if tf.startswith('G')])
                print(f"Max TF index needed: {max_tf_idx}")
            print(f"Max target gene index needed: {max_gene_idx}")
            print(f"Predicted matrix can accommodate: {self.predicted_matrix.shape[0] - 1} (0-indexed)")
    
    def get_evaluation_subset_info(self):
        """
        Get information about which subset of the predicted matrix will be used for evaluation.
        """
        if self.gold_standard_matrix is None:
            print("Gold standard not loaded yet.")
            return
        
        # Count how many TF->gene pairs will be evaluated
        evaluable_pairs = 0
        total_gold_pairs = 0
        
        # Create mappings
        gene_name_to_pred_idx = {}
        for gene in self.gene_labels:
            if gene.startswith('G'):
                try:
                    gene_idx = int(gene[1:]) - 1
                    if self.predicted_matrix is not None and 0 <= gene_idx < self.predicted_matrix.shape[0]:
                        gene_name_to_pred_idx[gene] = gene_idx
                except ValueError:
                    pass
        
        tf_name_to_pred_idx = {}
        if hasattr(self, 'tf_labels'):
            for tf in self.tf_labels:
                if tf.startswith('G'):
                    try:
                        tf_idx = int(tf[1:]) - 1
                        if self.predicted_matrix is not None and 0 <= tf_idx < self.predicted_matrix.shape[0]:
                            tf_name_to_pred_idx[tf] = tf_idx
                    except ValueError:
                        pass
        
        # Count evaluable pairs
        for tf_idx, tf_gene in enumerate(self.tf_labels):
            for target_idx, target_gene in enumerate(self.gene_labels):
                total_gold_pairs += 1
                if tf_gene in tf_name_to_pred_idx and target_gene in gene_name_to_pred_idx:
                    evaluable_pairs += 1
        
        print("Evaluation Subset Information:")
        print("-" * 30)
        print(f"Total gold standard entries: {total_gold_pairs}")
        print(f"Evaluable pairs (present in predicted matrix): {evaluable_pairs}")
        print(f"Coverage: {evaluable_pairs/total_gold_pairs*100:.2f}%")
        print(f"Predicted matrix total size: {self.predicted_matrix.size if self.predicted_matrix is not None else 'Not loaded'}")
        print(f"Evaluation will use: {evaluable_pairs} entries")
        
        return {
            'total_gold_pairs': total_gold_pairs,
            'evaluable_pairs': evaluable_pairs,
            'coverage_percent': evaluable_pairs/total_gold_pairs*100 if total_gold_pairs > 0 else 0
        }
    
    def load_predicted_matrix(self, matrix_path, has_variance_path=None):
        """
        Load predicted adjacency matrix and adjust for missing genes if needed.
        
        Args:
            matrix_path: Path to the adjacency matrix file
            has_variance_path: Path to the has_variance file (optional)
        """
        # Load the predicted matrix
        self.predicted_matrix = np.loadtxt(matrix_path)
        
        print(f"Original predicted matrix shape: {self.predicted_matrix.shape}")
        
        # If has_variance file is provided, adjust the matrix
        if has_variance_path is not None:
            has_variance = np.loadtxt(has_variance_path, dtype=bool)
            self.predicted_matrix = self._adjust_matrix_for_variance(self.predicted_matrix, has_variance)
            print(f"Adjusted predicted matrix shape: {self.predicted_matrix.shape}")
        
        return self.predicted_matrix
    
    def _adjust_matrix_for_variance(self, matrix, has_variance):
        """
        Adjust matrix by inserting zero rows/columns for genes with no variance.
        """
        original_size = matrix.shape[0]
        target_size = len(has_variance)
        
        if original_size == target_size:
            return matrix
        
        # Create new matrix with target size
        new_matrix = np.zeros((target_size, target_size))
        
        # Map indices from original matrix to new matrix
        valid_indices = np.where(has_variance)[0]
        
        for i, orig_i in enumerate(valid_indices):
            for j, orig_j in enumerate(valid_indices):
                if i < original_size and j < original_size:
                    new_matrix[orig_i, orig_j] = matrix[i, j]
        
        return new_matrix
    
    def prepare_for_evaluation(self):
        """
        Prepare matrices for evaluation by filtering predicted matrix to match gold standard structure.
        Only compares TF->gene pairs that exist in the gold standard.
        """
        if self.gold_standard_matrix is None or self.predicted_matrix is None:
            raise ValueError("Both gold standard and predicted matrices must be loaded first")
        
        gold_shape = self.gold_standard_matrix.shape
        pred_shape = self.predicted_matrix.shape
        
        print(f"Gold standard shape: {gold_shape}")
        print(f"Predicted matrix shape: {pred_shape}")
        
        # Create mappings from gene names to matrix indices
        # Assuming gene_labels are in format G1, G2, etc. and correspond to matrix indices
        gene_name_to_pred_idx = {}
        for i, gene in enumerate(self.gene_labels):
            if gene.startswith('G'):
                try:
                    # Extract gene number (G1 -> 0, G2 -> 1, etc.)
                    gene_idx = int(gene[1:]) - 1
                    if 0 <= gene_idx < pred_shape[0]:  # Ensure valid index
                        gene_name_to_pred_idx[gene] = gene_idx
                except ValueError:
                    print(f"Warning: Could not parse gene index from {gene}")
        
        # Create mapping for TF genes to predicted matrix indices
        tf_name_to_pred_idx = {}
        for tf in self.tf_labels:
            if tf.startswith('G'):
                try:
                    tf_idx = int(tf[1:]) - 1
                    if 0 <= tf_idx < pred_shape[0]:
                        tf_name_to_pred_idx[tf] = tf_idx
                except ValueError:
                    print(f"Warning: Could not parse TF index from {tf}")
        
        print(f"Mapped {len(gene_name_to_pred_idx)} target genes to predicted matrix")
        print(f"Mapped {len(tf_name_to_pred_idx)} TF genes to predicted matrix")
        
        # Extract only the relevant entries from both matrices
        gold_values = []
        pred_values = []
        
        # Iterate through gold standard matrix and extract corresponding predicted values
        for tf_idx, tf_gene in enumerate(self.tf_labels):
            for target_idx, target_gene in enumerate(self.gene_labels):
                gold_value = self.gold_standard_matrix[tf_idx, target_idx]
                
                # Get predicted value if both genes are mappable
                if tf_gene in tf_name_to_pred_idx and target_gene in gene_name_to_pred_idx:
                    pred_tf_idx = tf_name_to_pred_idx[tf_gene]
                    pred_target_idx = gene_name_to_pred_idx[target_gene]
                    pred_value = self.predicted_matrix[pred_tf_idx, pred_target_idx]
                    
                    gold_values.append(gold_value)
                    pred_values.append(pred_value)
        
        gold_values = np.array(gold_values)
        pred_values = np.array(pred_values)
        
        print(f"Extracted {len(gold_values)} matching TF->gene pairs")
        print(f"Gold standard range: [{gold_values.min():.4f}, {gold_values.max():.4f}]")
        print(f"Predicted values range: [{pred_values.min():.4f}, {pred_values.max():.4f}]")
        print(f"Non-zero gold standard entries: {np.count_nonzero(gold_values)}")
        
        return gold_values, pred_values
    
    def plot_confusion_matrix_heatmap(self, threshold=0.5, save_path='confusion_matrix.png'):
        """
        Generate confusion matrix heatmap.
        """
        y_true, y_pred_cont = self.prepare_for_evaluation()
        
        # Binarize predictions based on threshold
        y_pred = (y_pred_cont >= threshold).astype(int)
        y_true_bin = (y_true >= threshold).astype(int)
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true_bin, y_pred)
        
        # Plot
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Predicted 0', 'Predicted 1'],
                   yticklabels=['Actual 0', 'Actual 1'])
        plt.title(f'Confusion Matrix (Threshold = {threshold})')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if PLOT_SHOW:
            plt.show()
        
        return cm
    
    def plot_precision_recall_curve(self, save_path='precision_recall_curve.png'):
        """
        Generate Precision-Recall curve.
        """
        y_true, y_pred = self.prepare_for_evaluation()
        
        # Compute precision-recall curve
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
        avg_precision = average_precision_score(y_true, y_pred)
        
        # Plot
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, linewidth=2, label=f'AP = {avg_precision:.3f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if PLOT_SHOW:
            plt.show()
        
        return precision, recall, thresholds, avg_precision
    
    def plot_roc_curve(self, save_path='roc_curve.png'):
        """
        Generate ROC curve.
        """
        y_true, y_pred = self.prepare_for_evaluation()
        
        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        
        # Plot
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, linewidth=2, label=f'ROC AUC = {roc_auc:.3f}')
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if PLOT_SHOW:
            plt.show()
        
        return fpr, tpr, thresholds, roc_auc
    
    def plot_fscore_vs_threshold(self, save_path='fscore_vs_threshold.png'):
        """
        Generate F-score vs Threshold plot.
        """
        y_true, y_pred = self.prepare_for_evaluation()
        
        # Define threshold range
        thresholds = np.linspace(0, 1, 100)
        f_scores = []
        
        for threshold in thresholds:
            y_pred_bin = (y_pred >= threshold).astype(int)
            f_score = f1_score(y_true, y_pred_bin, zero_division=0)
            f_scores.append(f_score)
        
        # Plot
        plt.figure(figsize=(8, 6))
        plt.plot(thresholds, f_scores, linewidth=2)
        plt.xlabel('Threshold')
        plt.ylabel('F-Score')
        plt.title('F-Score vs Threshold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if PLOT_SHOW:
            plt.show()
        
        # Find optimal threshold
        optimal_idx = np.argmax(f_scores)
        optimal_threshold = thresholds[optimal_idx]
        optimal_fscore = f_scores[optimal_idx]
        
        print(f"Optimal threshold: {optimal_threshold:.3f} (F-Score: {optimal_fscore:.3f})")
        
        return thresholds, f_scores, optimal_threshold
    
    def plot_precision_recall_vs_threshold(self, save_path='precision_recall_vs_threshold.png'):
        """
        Generate Precision and Recall vs Threshold plot.
        """
        y_true, y_pred = self.prepare_for_evaluation()
        
        # Define threshold range
        thresholds = np.linspace(0, 1, 100)
        precisions = []
        recalls = []
        
        for threshold in thresholds:
            y_pred_bin = (y_pred >= threshold).astype(int)
            precision = precision_score(y_true, y_pred_bin, zero_division=0)
            recall = recall_score(y_true, y_pred_bin, zero_division=0)
            precisions.append(precision)
            recalls.append(recall)
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, precisions, linewidth=2, label='Precision')
        plt.plot(thresholds, recalls, linewidth=2, label='Recall')
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.title('Precision and Recall vs Threshold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if PLOT_SHOW:
            plt.show()
        
        return thresholds, precisions, recalls
    
    def generate_evaluation_report(self, output_path='evaluation_report.txt'):
        """
        Generate a comprehensive evaluation report.
        """
        y_true, y_pred = self.prepare_for_evaluation()
        
        # Calculate metrics at different thresholds
        thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
        
        with open(output_path, 'w') as f:
            f.write("Network Adjacency Matrix Evaluation Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Gold Standard Matrix Shape: {self.gold_standard_matrix.shape}\n")
            f.write(f"Predicted Matrix Shape: {self.predicted_matrix.shape}\n")
            f.write(f"Total Comparisons: {len(y_true)}\n\n")
            
            # ROC AUC
            fpr, tpr, _, roc_auc = self.plot_roc_curve('temp_roc.png')
            f.write(f"ROC AUC: {roc_auc:.4f}\n")
            
            # Average Precision
            _, _, _, avg_precision = self.plot_precision_recall_curve('temp_pr.png')
            f.write(f"Average Precision: {avg_precision:.4f}\n\n")
            
            # Metrics at different thresholds
            f.write("Metrics at Different Thresholds:\n")
            f.write("-" * 30 + "\n")
            for threshold in thresholds:
                y_pred_bin = (y_pred >= threshold).astype(int)
                precision = precision_score(y_true, y_pred_bin, zero_division=0)
                recall = recall_score(y_true, y_pred_bin, zero_division=0)
                f1 = f1_score(y_true, y_pred_bin, zero_division=0)
                
                f.write(f"Threshold {threshold:.1f}: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}\n")
        
        print(f"Evaluation report saved to: {output_path}")

# Example usage function
def run_evaluation_example():
    """
    Example of how to use the NetworkEvaluator class.
    """
    # Initialize evaluator
    evaluator = NetworkEvaluator()
    
    # Example: Load gold standard (you would use your actual file path)
    # Option 1: Let the system infer the number of genes from gold standard
    # evaluator.load_gold_standard('gold_standard.txt')
    
    # Option 2: Specify total number of genes if gold standard omits some genes
    # evaluator.load_gold_standard('gold_standard.txt', num_genes_B=100)
    
    # Check gene mapping information
    # evaluator.get_gene_mapping_info()
    
    # Example: Load predicted matrix (you would use your actual file paths)
    # evaluator.load_predicted_matrix('predicted_matrix.txt', 'has_variance.txt')
    
    # Generate all evaluations
    print("Generating evaluation plots...")
    
    # evaluator.plot_confusion_matrix_heatmap(threshold=0.5)
    # evaluator.plot_precision_recall_curve()
    # evaluator.plot_roc_curve()
    # evaluator.plot_fscore_vs_threshold()
    # evaluator.plot_precision_recall_vs_threshold()
    # evaluator.generate_evaluation_report()
    
    print("Evaluation complete!")

if __name__ == "__main__":
    # Create sample data for demonstration
    print("Network Adjacency Matrix Evaluator")
    print("To use this script:")
    print("1. Create a NetworkEvaluator instance")
    print("2. Load your gold standard using load_gold_standard()")
    print("3. Check gene mapping with get_gene_mapping_info() and debug_gene_mapping()")
    print("4. Load your predicted matrix using load_predicted_matrix()")
    print("5. Check evaluation subset with get_evaluation_subset_info()")
    print("6. Run the evaluation methods")
    print("\nExample:")
    print("evaluator = NetworkEvaluator()")
    print("evaluator.load_gold_standard('gold_standard.txt', num_genes_B=100)")
    print("evaluator.get_gene_mapping_info()  # Check gene mapping")
    print("evaluator.load_predicted_matrix('predicted.txt', 'has_variance.txt')")
    print("evaluator.debug_gene_mapping()  # Debug gene->index mapping")
    print("evaluator.get_evaluation_subset_info()  # Check evaluation coverage")
    print("evaluator.plot_confusion_matrix_heatmap()")
    print("# ... run other evaluation methods")
    
    evaluator = NetworkEvaluator()

    evaluator.load_gold_standard('data/DREAM5_NetworkInference_GoldStandard_Network3.tsv')
    evaluator.get_gene_mapping_info()

    evaluator.load_predicted_matrix('../final-delivery/dimreduction/references/final_data/full-gui-final_data.txt')
    evaluator.debug_gene_mapping()
    evaluator.get_evaluation_subset_info()
    
    name="full-gui-final_data"
    evaluator.plot_confusion_matrix_heatmap(threshold=0.7, save_path=f"confusion_matrix_{name}.png")
    evaluator.plot_precision_recall_curve(save_path = f'precision_recall_curve_{name}.png')
    evaluator.plot_roc_curve(save_path = f'roc_curve_{name}.png')
    evaluator.plot_precision_recall_vs_threshold(save_path = f'precision_recall_vs_threshold_{name}.png')
    evaluator.plot_fscore_vs_threshold(save_path = f'fscore_vs_threshold_{name}.png')
