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
            num_genes_B: Number of genes B (if not provided, will be inferred)
        """
        # Read the tab-separated file
        df = pd.read_csv(file_path, sep='\t', header=None, names=['Gene_A', 'Gene_B', 'Score'])
        
        # Get unique genes
        unique_A = df['Gene_A'].unique()
        unique_B = df['Gene_B'].unique()
        
        if num_genes_B is None:
            num_genes_B = len(unique_B)
            
        # Create gene labels (G1, G2, etc.)
        self.gene_labels = unique_B
        
        # Initialize gold standard matrix (SizeA x SizeB)
        self.gold_standard_matrix = np.zeros((len(unique_A), num_genes_B))
        
        # Fill the matrix
        for _, row in df.iterrows():
            # Assuming Gene_A and Gene_B are in format G1, G2, etc.
            if row['Gene_A'].startswith('G') and row['Gene_B'].startswith('G'):
                a_idx = int(row['Gene_A'][1:]) - 1  # Convert G1 -> 0, G2 -> 1, etc.
                b_idx = int(row['Gene_B'][1:]) - 1
                
                if a_idx < len(unique_A) and b_idx < num_genes_B:
                    # Find the index of Gene_A in unique_A list
                    a_matrix_idx = np.where(unique_A == row['Gene_A'])[0][0]
                    self.gold_standard_matrix[a_matrix_idx, b_idx] = row['Score']
            else:
                raise "Gold Standard invalid format"                    
        
        print(f"Gold standard matrix shape: {self.gold_standard_matrix.shape}")
        return self.gold_standard_matrix
    
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
        Prepare matrices for evaluation by ensuring compatible dimensions.
        """
        if self.gold_standard_matrix is None or self.predicted_matrix is None:
            raise ValueError("Both gold standard and predicted matrices must be loaded first")
        
        # For evaluation, we need to compare at the same granularity
        # If gold standard is SizeA x SizeB and predicted is SizeB x SizeB,
        # we extract the relevant part of the predicted matrix
        
        gold_shape = self.gold_standard_matrix.shape
        pred_shape = self.predicted_matrix.shape
        
        print(f"Gold standard shape: {gold_shape}")
        print(f"Predicted matrix shape: {pred_shape}")
        
        # If the predicted matrix is square and larger, extract the relevant submatrix
        if len(gold_shape) == 2 and len(pred_shape) == 2:
            if pred_shape[0] == pred_shape[1] and gold_shape[1] == pred_shape[1]:
                # Extract the first SizeA rows to match gold standard
                self.predicted_matrix_eval = self.predicted_matrix[:gold_shape[0], :]
            else:
                self.predicted_matrix_eval = self.predicted_matrix
        else:
            self.predicted_matrix_eval = self.predicted_matrix
        
        print(f"Evaluation matrices - Gold: {self.gold_standard_matrix.shape}, Pred: {self.predicted_matrix_eval.shape}")
        
        return self.gold_standard_matrix.flatten(), self.predicted_matrix_eval.flatten()
    
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
    # evaluator.load_gold_standard('gold_standard.txt', num_genes_B=100)
    
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
    print("3. Load your predicted matrix using load_predicted_matrix()")
    print("4. Run the evaluation methods")
    print("\nExample:")
    print("evaluator = NetworkEvaluator()")
    print("evaluator.load_gold_standard('gold_standard.txt')")
    print("evaluator.load_predicted_matrix('predicted.txt', 'has_variance.txt')")
    print("evaluator.plot_confusion_matrix_heatmap()")
    print("# ... run other evaluation methods")
    
    evaluator = NetworkEvaluator()
    evaluator.load_gold_standard('data/DREAM5_NetworkInference_GoldStandard_Network3.tsv')
