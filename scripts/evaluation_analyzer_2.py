import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import glob
import argparse
import sys
from pathlib import Path

class EvaluationAnalyzer:
    def __init__(self):
        self.time_data = None
        self.summary_data = None
        self.curve_data = None
        self.combined_data = None
        self.technique_colors = {}
        
    def setup_plotting(self):
        """Set up consistent plotting style and color scheme"""
        plt.style.use('default')
        sns.set_palette("husl")
        # Create a color palette with at least 12 distinct colors
        self.color_palette = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
            '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#aec7e8', '#ffbb78',
            '#98df8a', '#ff9896', '#c5b0d5', '#c49c94', '#f7b6d2', '#c7c7c7'
        ]
        
    def load_time_data(self, file_path, num_threads):
        """
        Load and preprocess time data from CSV file
        
        Args:
            file_path (str): Path to the time data CSV file
            num_threads (int): Number of threads to filter by
            
        Returns:
            pandas.DataFrame: Processed time data
        """
        try:
            print(f"Loading time data from: {file_path}")
            df = pd.read_csv(file_path)
            
            # Check required columns
            required_cols = ['technique', 'execution_time', 'threads']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            df.rename(columns={'threads': 'num_threads'}, inplace=True)
            
            # Filter by threads
            original_count = len(df)
            filtered_count = len(df[df['num_threads'] == num_threads])
            
            if filtered_count == 0:
                raise ValueError(f"No data found for {num_threads} threads")
                
            print(f"Filtered data: {filtered_count} rows (from {original_count} total)")
            
            # Handle missing execution times
            missing_time = df['execution_time'].isna().sum()
            if missing_time > 0:
                print(f"Warning: Ignoring {missing_time} entries with missing execution time")
                df = df.dropna(subset=['execution_time'])
            
            # Convert execution time to minutes
            df['execution_time_minutes'] = df['execution_time'] / 60.0
            
            # Calculate average execution time per technique
            time_avg = df.groupby(['technique', 'num_threads'])['execution_time_minutes'].mean().reset_index()
            time_avg.rename(columns={'execution_time_minutes': 'avg_execution_time_minutes'}, inplace=True)
            
            self.time_data = df
            print(f"Successfully loaded time data for {len(df)} entries")
            return time_avg
            
        except Exception as e:
            print(f"Error loading time data: {e}")
            return None
    
    def load_summary_data(self, folder_path, num_threads):
        """
        Load and preprocess summary data from CSV files
        
        Args:
            folder_path (str): Path to folder containing summary CSV files
            num_threads (int): Number of threads to filter by
            
        Returns:
            pandas.DataFrame: Processed summary data
        """
        try:
            print(f"Loading summary data from: {folder_path}")
            pattern = os.path.join(folder_path, f"summary_*_{num_threads}_threads.csv")
            files = glob.glob(pattern)
            
            if not files:
                raise ValueError(f"No summary files found for {num_threads} threads")
            
            dfs = []
            for file_path in files:
                try:
                    # Extract technique name from filename
                    filename = os.path.basename(file_path)
                    technique = filename.split('_')[1]  # Assuming format: summary_<technique>_<threads>_threads.csv
                    if technique == 'GENIE3':
                        technique = f'{technique}_{filename.split("_")[2]}'
                    
                    df = pd.read_csv(file_path)
                    df['technique'] = technique
                    dfs.append(df)
                    
                except Exception as e:
                    print(f"Warning: Could not load {file_path}: {e}")
            
            if not dfs:
                raise ValueError("No valid summary files could be loaded")
            
            df_combined = pd.concat(dfs, ignore_index=True)
            
            # Handle missing performance metrics
            missing_auroc = df_combined['auroc'].isna().sum()
            missing_aupr = df_combined['aupr'].isna().sum()
            
            if missing_auroc > 0 or missing_aupr > 0:
                print(f"Warning: Ignoring {max(missing_auroc, missing_aupr)} entries with missing AUROC/AUPR")
                df_combined = df_combined.dropna(subset=['auroc', 'aupr'])
            
            # Calculate average metrics per technique
            summary_avg = df_combined.groupby(['technique', 'num_threads']).agg({
                'auroc': 'mean',
                'aupr': 'mean'
            }).reset_index()
            
            self.summary_data = df_combined
            print(f"Successfully loaded summary data for {len(df_combined)} entries")
            return summary_avg
            
        except Exception as e:
            print(f"Error loading summary data: {e}")
            return None
    
    def load_curve_data(self, folder_path, num_threads):
        """
        Load and preprocess curve data from CSV files
        
        Args:
            folder_path (str): Path to folder containing curve CSV files
            num_threads (int): Number of threads to filter by
            
        Returns:
            pandas.DataFrame: Processed curve data
        """
        try:
            print(f"Loading curve data from: {folder_path}")
            pattern = os.path.join(folder_path, f"curve_data_*_{num_threads}_threads.csv")
            files = glob.glob(pattern)
            
            if not files:
                raise ValueError(f"No curve data files found for {num_threads} threads")
            
            dfs = []
            for file_path in files:
                try:
                    # Extract technique name from filename
                    filename = os.path.basename(file_path)
                    technique = filename.split('_')[2]  # Assuming format: curve_data_<technique>_<threads>_threads.csv
                    if technique == 'GENIE3':
                        technique = f'{technique}_{filename.split("_")[3]}'
                    
                    df = pd.read_csv(file_path)
                    df['technique'] = technique
                    dfs.append(df)
                    
                except Exception as e:
                    print(f"Warning: Could not load {file_path}: {e}")
            
            if not dfs:
                raise ValueError("No valid curve data files could be loaded")
            
            df_combined = pd.concat(dfs, ignore_index=True)
            
            # Handle missing values in curve data
            required_curve_cols = ['rank', 'fpr', 'tpr', 'recall', 'precision']
            available_cols = [col for col in required_curve_cols if col in df_combined.columns]
            
            missing_count = df_combined[available_cols].isna().sum().sum()
            if missing_count > 0:
                print(f"Warning: Ignoring {missing_count} missing values in curve data")
                df_combined = df_combined.dropna(subset=available_cols)
            
            self.curve_data = df_combined
            print(f"Successfully loaded curve data for {len(df_combined)} entries")
            return df_combined
            
        except Exception as e:
            print(f"Error loading curve data: {e}")
            return None
    
    def combine_data(self, time_avg, summary_avg):
        """
        Combine time and summary data into a single DataFrame
        
        Args:
            time_avg (pandas.DataFrame): Average time data per technique
            summary_avg (pandas.DataFrame): Average summary data per technique
            
        Returns:
            pandas.DataFrame: Combined data
        """
        try:
            if time_avg is None or summary_avg is None:
                raise ValueError("Time or summary data not available")
            
            # Merge time and summary data on technique
            combined = pd.merge(time_avg, summary_avg, on=['technique', 'num_threads'], how='inner')
            
            self.combined_data = combined
            print(f"Combined data created with {len(combined)} techniques")
            return combined
            
        except Exception as e:
            print(f"Error combining data: {e}")
            return None
    
    def plot_execution_time(self, output_dir=None, show=False):
        """Generate execution time comparison bar chart"""
        if self.time_data is None:
            print("No time data available for plotting")
            return
        
        try:
            plt.figure(figsize=(12, 8))
            
            # Calculate average execution time per technique
            time_avg = self.time_data.groupby('technique')['execution_time_minutes'].mean().sort_values()
            
            # Create color mapping
            techniques = time_avg.index.tolist()
            colors = [self.color_palette[i % len(self.color_palette)] for i in range(len(techniques))]
            self.technique_colors.update(dict(zip(techniques, colors)))
            
            # Create bar plot
            bars = plt.bar(range(len(techniques)), time_avg.values, color=colors, alpha=0.7)
            
            plt.xlabel('Technique', fontsize=12)
            plt.ylabel('Average Execution Time (minutes)', fontsize=12)
            plt.title('Average Execution Time by Technique', fontsize=14, fontweight='bold')
            plt.xticks(range(len(techniques)), techniques, rotation=45, ha='right')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}',
                        ha='center', va='bottom')
            
            plt.tight_layout()
            
            if output_dir:
                output_path = os.path.join(output_dir, 'execution_time_comparison.png')
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                print(f"Execution time plot saved to: {output_path}")
            
            if show:
                plt.show()
            else:
                plt.close()
                
        except Exception as e:
            print(f"Error creating execution time plot: {e}")
    
    def plot_roc_curve(self, output_dir=None, show=False):
        """Generate ROC curve visualization"""
        if self.curve_data is None:
            print("No curve data available for plotting ROC curve")
            return
        
        try:
            plt.figure(figsize=(10, 8))
            
            # Get unique techniques
            techniques = self.curve_data['technique'].unique()
            
            # Plot ROC curve for each technique
            for i, technique in enumerate(techniques):
                tech_data = self.curve_data[self.curve_data['technique'] == technique]
                
                # Calculate average FPR and TPR per rank (if multiple runs)
                if len(tech_data) > len(tech_data['rank'].unique()):
                    # Multiple runs - average them
                    avg_curve = tech_data.groupby('rank')[['fpr', 'tpr']].mean().reset_index()
                    fpr = avg_curve['fpr']
                    tpr = avg_curve['tpr']
                else:
                    # Single run
                    fpr = tech_data['fpr']
                    tpr = tech_data['tpr']
                
                auc = np.trapezoid(tpr, fpr)
                color = self.color_palette[i % len(self.color_palette)]
                label = f'{technique} (AUC: {auc:.3f})'
                plt.plot(fpr, tpr, label=label, color=color, linewidth=2)
            
            # Plot diagonal reference line
            plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
            
            plt.xlabel('False Positive Rate', fontsize=12)
            plt.ylabel('True Positive Rate', fontsize=12)
            plt.title('ROC Curve', fontsize=14, fontweight='bold')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.tight_layout()
            
            if output_dir:
                output_path = os.path.join(output_dir, 'roc_curve.png')
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                print(f"ROC curve plot saved to: {output_path}")
            
            if show:
                plt.show()
            else:
                plt.close()
                
        except Exception as e:
            print(f"Error creating ROC curve plot: {e}")
    
    def plot_pr_curve(self, output_dir=None, show=False):
        """Generate Precision-Recall curve visualization"""
        if self.curve_data is None:
            print("No curve data available for plotting PR curve")
            return
        
        try:
            plt.figure(figsize=(10, 8))
            
            # Get unique techniques
            techniques = self.curve_data['technique'].unique()
            
            # Plot PR curve for each technique
            for i, technique in enumerate(techniques):
                tech_data = self.curve_data[self.curve_data['technique'] == technique]
                
                # Calculate average recall and precision per rank (if multiple runs)
                if len(tech_data) > len(tech_data['rank'].unique()):
                    # Multiple runs - average them
                    avg_curve = tech_data.groupby('rank')[['recall', 'precision']].mean().reset_index()
                    recall = avg_curve['recall']
                    precision = avg_curve['precision']
                else:
                    # Single run
                    recall = tech_data['recall']
                    precision = tech_data['precision']
                
                auc = np.trapezoid(precision, recall)
                color = self.color_palette[i % len(self.color_palette)]
                label = f'{technique} (AUC: {auc:.3f})'
                plt.plot(recall, precision, label=label, color=color, linewidth=2)
            
            plt.xlabel('Recall', fontsize=12)
            plt.ylabel('Precision', fontsize=12)
            plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.tight_layout()
            
            if output_dir:
                output_path = os.path.join(output_dir, 'precision_recall_curve.png')
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                print(f"Precision-Recall curve plot saved to: {output_path}")
            
            if show:
                plt.show()
            else:
                plt.close()
                
        except Exception as e:
            print(f"Error creating Precision-Recall curve plot: {e}")
    def plot_tradeoff_analysis(self, output_dir=None, show=False, metric='auroc'):
        """
        Generate trade-off analysis between performance metric and execution speed
        
        Args:
            output_dir (str): Directory to save the plot
            show (bool): Whether to display the plot
            metric (str): Performance metric to use ('auroc' or 'aupr')
        """
        if self.combined_data is None:
            print("No combined data available for trade-off analysis")
            return
        
        try:
            # Get unique thread counts
            thread_counts = self.combined_data['num_threads'].unique()
            
            for thread_count in thread_counts:
                # Filter data for current thread count
                thread_data = self.combined_data[self.combined_data['num_threads'] == thread_count].copy()
                
                if len(thread_data) == 0:
                    print(f"No data available for {thread_count} threads")
                    continue
                
                # Calculate worst execution time
                worst_time = thread_data['avg_execution_time_minutes'].max()
                
                # Calculate relative time score
                thread_data['time_relative'] = (worst_time - thread_data['avg_execution_time_minutes']) / worst_time
                
                # Generate alpha values from 0 to 1
                alpha_values = np.arange(0, 1.01, 0.01)
                
                plt.figure(figsize=(12, 8))
                
                # Store AUC values for each technique
                auc_values = {}
                
                # Calculate and plot score for each technique
                for i, (_, row) in enumerate(thread_data.iterrows()):
                    technique = row['technique']
                    metric_value = row[metric]
                    time_relative = row['time_relative']
                    
                    # Calculate score for each alpha
                    scores = [alpha * metric_value + (1 - alpha) * time_relative for alpha in alpha_values]
                    
                    # Calculate AUC using trapezoidal rule
                    auc = np.trapezoid(scores, alpha_values)
                    auc_values[technique] = auc
                    
                    # Get color for technique
                    color = self.color_palette[i % len(self.color_palette)]
                    
                    # Create label with AUC information
                    label = f'{technique} (AUC: {auc:.3f})'
                    
                    # Plot line
                    plt.plot(alpha_values, scores, label=label, color=color, linewidth=2)
                
                plt.xlabel('α (Weight of Metric)', fontsize=12)
                plt.ylabel('Score', fontsize=12)
                plt.title(f'Trade-off Analysis: {metric.upper()} vs Speed (Thread = {thread_count})', 
                        fontsize=14, fontweight='bold')
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.grid(True, alpha=0.3)
                plt.xlim([0, 1])
                plt.ylim([0, 1])
                plt.tight_layout()
                
                # Print AUC summary to console
                print(f"\nTrade-off Analysis AUC Summary (Thread = {thread_count}, Metric = {metric.upper()}):")
                print("-" * 60)
                sorted_auc = sorted(auc_values.items(), key=lambda x: x[1], reverse=True)
                for technique, auc in sorted_auc:
                    print(f"  {technique}: {auc:.4f}")
                
                if output_dir:
                    output_path = os.path.join(output_dir, f'tradeoff_analysis_{metric}_thread_{thread_count}.png')
                    plt.savefig(output_path, dpi=300, bbox_inches='tight')
                    print(f"Trade-off analysis plot saved to: {output_path}")
                
                if show:
                    plt.show()
                else:
                    plt.close()
                    
        except Exception as e:
            print(f"Error creating trade-off analysis plot: {e}")
                
    def export_tradeoff_rank_tables(self, output_dir=None):
        """
        Export CSV tables showing ordered rank of technique trade-off performance
        at fixed weight values for both AUROC and AUPR metrics.
        """
        if self.combined_data is None:
            print("No combined data available for trade-off rank tables")
            return
        
        try:
            # Get unique thread counts
            thread_counts = self.combined_data['num_threads'].unique()
            
            for thread_count in thread_counts:
                # Filter data for current thread count
                thread_data = self.combined_data[self.combined_data['num_threads'] == thread_count].copy()
                
                if len(thread_data) == 0:
                    continue
                
                # Calculate worst execution time
                worst_time = thread_data['avg_execution_time_minutes'].max()
                thread_data['time_relative'] = (worst_time - thread_data['avg_execution_time_minutes']) / worst_time
                
                # Fixed weight values (converted to decimal)
                fixed_weights_auroc = [0.0, 0.25, 0.5, 0.75, 1.0]
                fixed_weights_aupr = [0.0, 0.25, 0.5, 0.75, 1.0]
                
                # Prepare data for AUROC trade-off ranking
                auroc_rank_data = []
                for weight in fixed_weights_auroc:
                    # Calculate scores for this weight
                    scores = []
                    for _, row in thread_data.iterrows():
                        score = weight * row['auroc'] + (1 - weight) * row['time_relative']
                        scores.append((row['technique'], score))
                    
                    # Sort by score descending
                    scores.sort(key=lambda x: x[1], reverse=True)
                    
                    # Add to rank data
                    rank_entry = {'Weight': f"{int(weight*100)}"}
                    for rank, (technique, score) in enumerate(scores, 1):
                        rank_entry[f'Rank_{rank}'] = f"{score:.3f} {technique} (threads={thread_count})"
                    auroc_rank_data.append(rank_entry)
                
                # Prepare data for AUPR trade-off ranking
                aupr_rank_data = []
                for weight in fixed_weights_aupr:
                    # Calculate scores for this weight
                    scores = []
                    for _, row in thread_data.iterrows():
                        score = weight * row['aupr'] + (1 - weight) * row['time_relative']
                        scores.append((row['technique'], score))
                    
                    # Sort by score descending
                    scores.sort(key=lambda x: x[1], reverse=True)
                    
                    # Add to rank data
                    rank_entry = {'Weight': f"{int(weight*100)}"}
                    for rank, (technique, score) in enumerate(scores, 1):
                        rank_entry[f'Rank_{rank}'] = f"{score:.3f} {technique} (threads={thread_count})"
                    aupr_rank_data.append(rank_entry)
                
                # Convert to DataFrames
                df_auroc_ranks = pd.DataFrame(auroc_rank_data).T
                df_aupr_ranks = pd.DataFrame(aupr_rank_data).T
                
                # Save to CSV files
                if output_dir:
                    auroc_filename = f"tradeoff_auroc_ranks_thread_{thread_count}.csv"
                    aupr_filename = f"tradeoff_aupr_ranks_thread_{thread_count}.csv"
                    
                    auroc_path = os.path.join(output_dir, auroc_filename)
                    aupr_path = os.path.join(output_dir, aupr_filename)
                    
                    df_auroc_ranks.to_csv(auroc_path, index=False, header=False)
                    df_aupr_ranks.to_csv(aupr_path, index=False, header=False)
                    
                    print(f"AUROC trade-off ranks saved to: {auroc_path}")
                    print(f"AUPR trade-off ranks saved to: {aupr_path}")
            
        except Exception as e:
            print(f"Error exporting trade-off rank tables: {e}")

    def export_auc_rank_tables(self, output_dir=None):
        """
        Export CSV tables showing ordered rank of technique AUC values
        for trade-off performance of both AUROC and AUPR.
        """
        if self.combined_data is None:
            print("No combined data available for AUC rank tables")
            return
        
        try:
            # Get unique thread counts
            thread_counts = self.combined_data['num_threads'].unique()
            
            for thread_count in thread_counts:
                # Filter data for current thread count
                thread_data = self.combined_data[self.combined_data['num_threads'] == thread_count].copy()
                
                if len(thread_data) == 0:
                    continue
                
                # Calculate worst execution time
                worst_time = thread_data['avg_execution_time_minutes'].max()
                thread_data['time_relative'] = (worst_time - thread_data['avg_execution_time_minutes']) / worst_time
                
                # Generate alpha values from 0 to 1
                alpha_values = np.arange(0, 1.01, 0.01)
                
                # Calculate AUC values for AUROC trade-off
                auroc_auc_data = []
                for _, row in thread_data.iterrows():
                    technique = row['technique']
                    scores = [alpha * row['auroc'] + (1 - alpha) * row['time_relative'] for alpha in alpha_values]
                    auc = np.trapezoid(scores, alpha_values)
                    auroc_auc_data.append((technique, auc))
                
                # Calculate AUC values for AUPR trade-off
                aupr_auc_data = []
                for _, row in thread_data.iterrows():
                    technique = row['technique']
                    scores = [alpha * row['aupr'] + (1 - alpha) * row['time_relative'] for alpha in alpha_values]
                    auc = np.trapezoid(scores, alpha_values)
                    aupr_auc_data.append((technique, auc))
                
                # Sort by AUC descending
                auroc_auc_data.sort(key=lambda x: x[1], reverse=True)
                aupr_auc_data.sort(key=lambda x: x[1], reverse=True)
                
                # Prepare rank tables
                auroc_rank_table = []
                aupr_rank_table = []
                
                # AUROC AUC ranks
                for rank, (technique, auc) in enumerate(auroc_auc_data, 1):
                    auroc_rank_table.append({
                        'Rank': rank,
                        'Technique_AUC': f"{auc:.3f} {technique} (threads={thread_count})"
                    })
                
                # AUPR AUC ranks
                for rank, (technique, auc) in enumerate(aupr_auc_data, 1):
                    aupr_rank_table.append({
                        'Rank': rank,
                        'Technique_AUC': f"{auc:.3f} {technique} (threads={thread_count})"
                    })
                
                # Convert to DataFrames
                df_auroc_auc_ranks = pd.DataFrame(auroc_rank_table)
                df_aupr_auc_ranks = pd.DataFrame(aupr_rank_table)
                
                # Save to CSV files
                if output_dir:
                    auroc_filename = f"auc_tradeoff_auroc_ranks_thread_{thread_count}.csv"
                    aupr_filename = f"auc_tradeoff_aupr_ranks_thread_{thread_count}.csv"
                    
                    auroc_path = os.path.join(output_dir, auroc_filename)
                    aupr_path = os.path.join(output_dir, aupr_filename)
                    
                    df_auroc_auc_ranks.to_csv(auroc_path, index=False)
                    df_aupr_auc_ranks.to_csv(aupr_path, index=False)
                    
                    print(f"AUROC AUC trade-off ranks saved to: {auroc_path}")
                    print(f"AUPR AUC trade-off ranks saved to: {aupr_path}")
            
        except Exception as e:
            print(f"Error exporting AUC rank tables: {e}")

def main():
    """Main function to run the evaluation analysis"""
    parser = argparse.ArgumentParser(description='Evaluate and visualize performance metrics')
    parser.add_argument('--time_file', type=str, help='Path to time data CSV file')
    parser.add_argument('--data_folder', type=str, help='Path to folder containing summary and curve data')
    parser.add_argument('--threads', type=int, help='Number of threads to analyze')
    parser.add_argument('--output_dir', type=str, help='Output directory for saving graphs')
    parser.add_argument('--show_plots', action='store_true', help='Show plots interactively')
    
    args = parser.parse_args()
    
    # Get user input if not provided via command line
    if not args.time_file:
        args.time_file = input("Enter path to time data CSV file: ").strip()
    
    if not args.data_folder:
        args.data_folder = input("Enter path to data folder (summary and curve data): ").strip()
    
    if not args.threads:
        try:
            args.threads = int(input("Enter number of threads to analyze: ").strip())
        except ValueError:
            print("Invalid thread count. Using default: 1")
            args.threads = 1
    
    if not args.output_dir:
        args.output_dir = input("Enter output directory for graphs (press Enter for current directory): ").strip()
        if not args.output_dir:
            args.output_dir = "."
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize analyzer
    analyzer = EvaluationAnalyzer()
    analyzer.setup_plotting()
    
    print("\n" + "="*50)
    print("Starting Evaluation Analysis")
    print("="*50)
    
    # Load data
    time_avg = analyzer.load_time_data(args.time_file, args.threads)
    summary_avg = analyzer.load_summary_data(args.data_folder, args.threads)
    curve_data = analyzer.load_curve_data(args.data_folder, args.threads)
    
    # Combine data
    if time_avg is not None and summary_avg is not None:
        combined_data = analyzer.combine_data(time_avg, summary_avg)
        
        # Generate visualizations
        print("\nGenerating visualizations...")
        
        analyzer.plot_execution_time(output_dir=args.output_dir, show=args.show_plots)
        analyzer.plot_roc_curve(output_dir=args.output_dir, show=args.show_plots)
        analyzer.plot_pr_curve(output_dir=args.output_dir, show=args.show_plots)
        
        # Generate trade-off analysis for both metrics
        analyzer.plot_tradeoff_analysis(output_dir=args.output_dir, show=args.show_plots, metric='auroc')
        analyzer.plot_tradeoff_analysis(output_dir=args.output_dir, show=args.show_plots, metric='aupr')
        
        # Generate trade-off rank tables
        print("\nGenerating trade-off rank tables...")
        analyzer.export_tradeoff_rank_tables(output_dir=args.output_dir)
        analyzer.export_auc_rank_tables(output_dir=args.output_dir)
        
        print("\n" + "="*50)
        print("Analysis Complete!")
        print(f"Graphs saved to: {os.path.abspath(args.output_dir)}")
        print("="*50)
        
    else:
        print("Failed to load required data. Analysis cannot proceed.")

if __name__ == "__main__":
    main()