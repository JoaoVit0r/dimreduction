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
    def __init__(self, figure_width=12.0, transpose_plots=False):
        self.time_data = None
        self.summary_data = None
        self.curve_data = None
        self.combined_data = None
        self.technique_colors = {}
        self.figure_width = figure_width
        self.transpose_plots = transpose_plots
        
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
    
    def get_figure_size(self, base_width=None, aspect_ratio=0.67):
        """
        Calculate figure size based on configured width and aspect ratio
        
        Args:
            base_width (float): Base width for calculation (uses self.figure_width if None)
            aspect_ratio (float): Height to width ratio
            
        Returns:
            tuple: (width, height) figure size
        """
        if base_width is None:
            base_width = self.figure_width
        height = base_width * aspect_ratio
        return (base_width, height)
    
    def save_plot_transposed(self, plt, output_path, transpose_suffix="_transpose"):
        """
        Save both original and transposed versions of the plot if requested
        
        Args:
            plt: matplotlib pyplot object
            output_path (str): Original output file path
            transpose_suffix (str): Suffix for transposed version
        """
        # Save original plot
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
        
        # Save transposed version if requested
        if self.transpose_plots:
            # Extract directory and filename
            output_dir = os.path.dirname(output_path)
            filename = os.path.basename(output_path)
            name, ext = os.path.splitext(filename)
            
            # Create transposed filename
            transposed_filename = f"{name}{transpose_suffix}{ext}"
            transposed_path = os.path.join(output_dir, transposed_filename)
            
            # Get current axes and transpose
            ax = plt.gca()
            self.transpose_current_plot(ax)
            
            # Save transposed version
            plt.savefig(transposed_path, dpi=300, bbox_inches='tight')
            print(f"Transposed plot saved to: {transposed_path}")
    
    def transpose_current_plot(self, ax):
        """
        Transpose the current plot by swapping data and labels
        
        Args:
            ax: matplotlib axes object
        """
        # Get all lines and collections
        lines = ax.get_lines()
        collections = ax.collections
        
        # For line plots, swap x and y data
        for line in lines:
            xdata = line.get_xdata()
            ydata = line.get_ydata()
            line.set_xdata(ydata)
            line.set_ydata(xdata)
        
        # For bar charts and other collections, we need to handle differently
        # This is a simplified approach - complex plots may need special handling
        for collection in collections:
            # Try to get offsets (for scatter plots and some bar charts)
            try:
                offsets = collection.get_offsets()
                if offsets.size > 0:
                    # Swap x and y coordinates
                    new_offsets = np.column_stack([offsets[:, 1], offsets[:, 0]])
                    collection.set_offsets(new_offsets)
            except:
                pass
        
        # Swap axis labels
        xlabel = ax.get_xlabel()
        ylabel = ax.get_ylabel()
        ax.set_xlabel(ylabel)
        ax.set_ylabel(xlabel)
        
        # Reset limits to fit new data
        ax.relim()
        ax.autoscale_view()
    
    def transpose_bar_plot(self, bars, xlabels, xlabel, ylabel, title):
        """
        Create a transposed version of a bar plot (horizontal bars)
        
        Returns:
            tuple: (fig, ax) for the transposed plot
        """
        # Calculate appropriate size for horizontal plot
        width, height = self.get_figure_size(aspect_ratio=0.8)
        fig, ax = plt.subplots(figsize=(height, width))  # Swap dimensions
        
        # Get bar values and labels for horizontal bars
        values = [bar.get_height() for bar in bars]
        labels = [label.get_text() for label in xlabels]
        
        # Create horizontal bar plot
        y_pos = np.arange(len(values))
        bars_transposed = ax.barh(y_pos, values, color=[bar.get_facecolor() for bar in bars])
        
        # Set labels and title
        ax.set_xlabel(ylabel)  # Original y-label becomes x-label
        ax.set_ylabel(xlabel)  # Original x-label becomes y-label
        ax.set_title(f"{title} (Transposed)")
        
        # Set y-tick labels
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        
        # Add value labels on bars
        for bar in bars_transposed:
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2., 
                   f'{width:.2f}', ha='left', va='center')
        
        plt.tight_layout()
        return fig, ax

    def load_time_data(self, file_paths, num_threads_list):
        """
        Load and preprocess time data from multiple CSV files
        
        Args:
            file_paths (list): List of paths to time data CSV files
            num_threads_list (list): List of thread counts corresponding to each file
            
        Returns:
            pandas.DataFrame: Processed time data
        """
        try:
            if len(file_paths) != len(num_threads_list):
                raise ValueError("Number of time files must match number of thread counts")
            
            print("Loading time data from multiple files:")
            dfs = []
            
            for file_path, num_threads in zip(file_paths, num_threads_list):
                print(f"  - {file_path} (threads: {num_threads})")
                df = pd.read_csv(file_path)
                
                # Check required columns
                required_cols = ['technique', 'execution_time', 'threads']
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    raise ValueError(f"Missing required columns in {file_path}: {missing_cols}")
                
                df.rename(columns={'threads': 'num_threads'}, inplace=True)
                
                # Filter by threads
                original_count = len(df)
                filtered_df = df[df['num_threads'] == num_threads]
                
                if len(filtered_df) == 0:
                    print(f"Warning: No data found for {num_threads} threads in {file_path}")
                    continue
                
                print(f"    Filtered data: {len(filtered_df)} rows (from {original_count} total)")
                
                # Handle missing execution times
                missing_time = filtered_df['execution_time'].isna().sum()
                if missing_time > 0:
                    print(f"    Warning: Ignoring {missing_time} entries with missing execution time")
                    filtered_df = filtered_df.dropna(subset=['execution_time'])
                
                # Convert execution time to minutes
                filtered_df['execution_time_minutes'] = filtered_df['execution_time'] / 60.0
                dfs.append(filtered_df)
            
            if not dfs:
                raise ValueError("No valid time data could be loaded from any file")
            
            df_combined = pd.concat(dfs, ignore_index=True)
            
            # Calculate average execution time per technique and thread count
            time_avg = df_combined.groupby(['technique', 'num_threads'])['execution_time_minutes'].mean().reset_index()
            time_avg.rename(columns={'execution_time_minutes': 'avg_execution_time_minutes'}, inplace=True)
            
            self.time_data = df_combined
            print(f"Successfully loaded time data for {len(df_combined)} entries across {len(time_avg)} technique-thread combinations")
            return time_avg
            
        except Exception as e:
            print(f"Error loading time data: {e}")
            return None
    
    def load_summary_data(self, folder_paths, num_threads_list):
        """
        Load and preprocess summary data from multiple folders
        
        Args:
            folder_paths (list): List of paths to folders containing summary CSV files
            num_threads_list (list): List of thread counts corresponding to each folder
            
        Returns:
            pandas.DataFrame: Processed summary data
        """
        try:
            if len(folder_paths) != len(num_threads_list):
                raise ValueError("Number of data folders must match number of thread counts")
            
            print("Loading summary data from multiple folders:")
            dfs = []
            
            for folder_path, num_threads in zip(folder_paths, num_threads_list):
                print(f"  - {folder_path} (threads: {num_threads})")
                pattern = os.path.join(folder_path, f"summary_*_{num_threads}_threads.csv")
                files = glob.glob(pattern)
                
                if not files:
                    print(f"    Warning: No summary files found for {num_threads} threads in {folder_path}")
                    continue
                
                for file_path in files:
                    try:
                        # Extract technique name from filename
                        filename = os.path.basename(file_path)
                        technique = filename.split('_')[1]  # Assuming format: summary_<technique>_<threads>_threads.csv
                        if technique == 'GENIE3':
                            technique = f'{technique}_{filename.split("_")[2]}'
                        
                        df = pd.read_csv(file_path)
                        df['technique'] = technique
                        df['num_threads'] = num_threads
                        dfs.append(df)
                        
                    except Exception as e:
                        print(f"    Warning: Could not load {file_path}: {e}")
            
            if not dfs:
                raise ValueError("No valid summary files could be loaded from any folder")
            
            df_combined = pd.concat(dfs, ignore_index=True)
            
            # Handle missing performance metrics
            missing_auroc = df_combined['auroc'].isna().sum()
            missing_aupr = df_combined['aupr'].isna().sum()
            
            if missing_auroc > 0 or missing_aupr > 0:
                print(f"Warning: Ignoring {max(missing_auroc, missing_aupr)} entries with missing AUROC/AUPR")
                df_combined = df_combined.dropna(subset=['auroc', 'aupr'])
            
            # Calculate average metrics per technique and thread count
            summary_avg = df_combined.groupby(['technique', 'num_threads']).agg({
                'auroc': 'mean',
                'aupr': 'mean'
            }).reset_index()
            
            self.summary_data = df_combined
            print(f"Successfully loaded summary data for {len(df_combined)} entries across {len(summary_avg)} technique-thread combinations")
            return summary_avg
            
        except Exception as e:
            print(f"Error loading summary data: {e}")
            return None
    
    def load_curve_data(self, folder_paths, num_threads_list):
        """
        Load and preprocess curve data from multiple folders
        
        Args:
            folder_paths (list): List of paths to folders containing curve CSV files
            num_threads_list (list): List of thread counts corresponding to each folder
            
        Returns:
            pandas.DataFrame: Processed curve data
        """
        try:
            if len(folder_paths) != len(num_threads_list):
                raise ValueError("Number of data folders must match number of thread counts")
            
            print("Loading curve data from multiple folders:")
            dfs = []
            
            for folder_path, num_threads in zip(folder_paths, num_threads_list):
                print(f"  - {folder_path} (threads: {num_threads})")
                pattern = os.path.join(folder_path, f"curve_data_*_{num_threads}_threads.csv")
                files = glob.glob(pattern)
                
                if not files:
                    print(f"    Warning: No curve data files found for {num_threads} threads in {folder_path}")
                    continue
                
                for file_path in files:
                    try:
                        # Extract technique name from filename
                        filename = os.path.basename(file_path)
                        technique = filename.split('_')[2]  # Assuming format: curve_data_<technique>_<threads>_threads.csv
                        if technique == 'GENIE3':
                            technique = f'{technique}_{filename.split("_")[3]}'
                        
                        df = pd.read_csv(file_path)
                        df['technique'] = technique
                        df['num_threads'] = num_threads
                        dfs.append(df)
                        
                    except Exception as e:
                        print(f"    Warning: Could not load {file_path}: {e}")
            
            if not dfs:
                raise ValueError("No valid curve data files could be loaded from any folder")
            
            df_combined = pd.concat(dfs, ignore_index=True)
            
            # Handle missing values in curve data
            required_curve_cols = ['rank', 'fpr', 'tpr', 'recall', 'precision']
            available_cols = [col for col in required_curve_cols if col in df_combined.columns]
            
            missing_count = df_combined[available_cols].isna().sum().sum()
            if missing_count > 0:
                print(f"Warning: Ignoring {missing_count} missing values in curve data")
                df_combined = df_combined.dropna(subset=available_cols)
            
            self.curve_data = df_combined
            print(f"Successfully loaded curve data for {len(df_combined)} entries across thread counts: {sorted(df_combined['num_threads'].unique())}")
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
            
            # Merge time and summary data on technique and thread count
            combined = pd.merge(time_avg, summary_avg, on=['technique', 'num_threads'], how='inner')
            
            self.combined_data = combined
            print(f"Combined data created with {len(combined)} technique-thread combinations")
            print(f"Thread counts in combined data: {sorted(combined['num_threads'].unique())}")
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
            fig, ax = plt.subplots(figsize=self.get_figure_size(10))
            
            # Calculate average execution time per technique across all thread counts
            time_avg = self.time_data.groupby(['technique', 'num_threads'])['execution_time_minutes'].mean().sort_values()
            
            # Create color mapping with improved labels
            techniques = [f'{tech} (threads={num_thread})' for tech, num_thread in time_avg.index.tolist()]
            colors = [self.color_palette[i % len(self.color_palette)] for i in range(len(techniques))]
            self.technique_colors.update(dict(zip(techniques, colors)))
            
            # Create bar plot
            bars = ax.bar(range(len(techniques)), time_avg.values, color=colors, alpha=0.7)
            
            ax.set_xlabel('Technique', fontsize=12)
            ax.set_ylabel('Average Execution Time (minutes)', fontsize=12)
            ax.set_title('Average Execution Time by Technique (All Thread Counts)', fontsize=14, fontweight='bold')
            ax.set_xticks(range(len(techniques)))
            ax.set_xticklabels(techniques, rotation=45, ha='right')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}',
                        ha='center', va='bottom')
            
            plt.tight_layout()
            
            if output_dir:
                output_path = os.path.join(output_dir, 'execution_time_comparison.png')
                self.save_plot_transposed(plt, output_path)
            
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
            fig, ax = plt.subplots(figsize=self.get_figure_size(10, 0.8))
            
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
                ax.plot(fpr, tpr, label=label, color=color, linewidth=2)
            
            # Plot diagonal reference line
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
            
            ax.set_xlabel('False Positive Rate', fontsize=12)
            ax.set_ylabel('True Positive Rate', fontsize=12)
            ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            plt.tight_layout()
            
            if output_dir:
                output_path = os.path.join(output_dir, 'roc_curve.png')
                self.save_plot_transposed(plt, output_path)
            
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
            fig, ax = plt.subplots(figsize=self.get_figure_size(10, 0.8))
            
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
                ax.plot(recall, precision, label=label, color=color, linewidth=2)
            
            ax.set_xlabel('Recall', fontsize=12)
            ax.set_ylabel('Precision', fontsize=12)
            ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            plt.tight_layout()
            
            if output_dir:
                output_path = os.path.join(output_dir, 'precision_recall_curve.png')
                self.save_plot_transposed(plt, output_path)
            
            if show:
                plt.show()
            else:
                plt.close()
                
        except Exception as e:
            print(f"Error creating Precision-Recall curve plot: {e}")

    def generate_confusion_matrix_tables(self, output_dir=None, show=False):
        """Generate confusion matrix tables for each technique and save as CSV."""
        if self.curve_data is None:
            print("No curve data available for generating confusion matrix tables.")
            return

        try:
            all_cm_data = []
            
            # Average curve data across runs for each technique and rank
            avg_curve_data = self.curve_data.groupby(['technique', 'rank']).mean().reset_index()

            techniques = avg_curve_data['technique'].unique()

            for technique in techniques:
                tech_data = avg_curve_data[avg_curve_data['technique'] == technique]
                
                if 'P' not in tech_data.columns or 'N' not in tech_data.columns:
                    print(f"Warning: P and N values not found for technique {technique}. Skipping confusion matrix.")
                    continue

                P = tech_data['P'].iloc[0]
                N = tech_data['N'].iloc[0]
                T = P + N
                
                ranks_to_evaluate = [round(T/4), round(T/2), round(3*T/4), T]
                
                for k in ranks_to_evaluate:
                    # Find the closest rank in the data
                    rank_data = tech_data.iloc[(tech_data['rank'] - k).abs().argsort()[:1]]
                    
                    if rank_data.empty:
                        continue

                    tpr = rank_data['tpr'].iloc[0]
                    fpr = rank_data['fpr'].iloc[0]

                    TP = tpr * P
                    FP = fpr * N
                    FN = P - TP
                    TN = N - FP
                    
                    all_cm_data.append({
                        'technique': technique,
                        'rank_k': k,
                        'TP': TP,
                        'FP': FP,
                        'FN': FN,
                        'TN': TN
                    })

            if all_cm_data:
                cm_df = pd.DataFrame(all_cm_data)
                if output_dir:
                    output_path = os.path.join(output_dir, 'confusion_matrix_analysis.csv')
                    cm_df.to_csv(output_path, index=False)
                    print(f"Confusion matrix analysis saved to: {output_path}")

        except Exception as e:
            print(f"Error generating confusion matrix tables: {e}")

    def plot_tp_vs_rank(self, output_dir=None, show=False):
        """Generate a line graph of True Positives (TP) vs Rank K for each technique."""
        if self.curve_data is None:
            print("No curve data available for plotting TP vs Rank.")
            return

        try:
            fig, ax = plt.subplots(figsize=self.get_figure_size(10))
            
            # Average curve data across runs for each technique and rank
            avg_curve_data = self.curve_data.groupby(['technique', 'rank']).mean().reset_index()
            
            techniques = avg_curve_data['technique'].unique()

            for i, technique in enumerate(techniques):
                tech_data = avg_curve_data[avg_curve_data['technique'] == technique]

                if 'P' not in tech_data.columns or 'N' not in tech_data.columns or 'L' not in tech_data.columns:
                    print(f"Warning: P, N, or L values not found for technique {technique}. Skipping TP plot.")
                    continue

                P = tech_data['P'].iloc[0]
                N = tech_data['N'].iloc[0]
                L = tech_data['L'].iloc[0]
                T = P + N

                ranks_to_plot = sorted(list(set([L] + [round(T/4), round(T/2), round(3*T/4), T])))
                
                tps = []
                rank_labels = sorted(list(set([0] + [round(T/4), round(T/2), round(3*T/4), T])))
                
                for k in ranks_to_plot:
                    # Find the closest rank in the data
                    rank_data = tech_data.iloc[(tech_data['rank'] - k).abs().argsort()[:1]]
                    
                    if rank_data.empty:
                        tps.append(np.nan)
                        rank_labels.append(str(k))
                        continue

                    tpr = rank_data['tpr'].iloc[0]
                    TP = tpr * P
                    tps.append(TP)

                color = self.color_palette[i % len(self.color_palette)]
                
                # Plot the line
                line = ax.plot(ranks_to_plot, tps, marker='', linestyle='-', color=color, label=f'{technique}', linewidth=2)
                
                # Plot markers for all points
                for j, (rank, tp) in enumerate(zip(ranks_to_plot, tps)):
                    if not np.isnan(tp):
                        # Use different marker for L point
                        if rank == L:
                            marker = 'D'  # Diamond for L point
                            markersize = 10
                            markeredgewidth = 2
                            markeredgecolor = color
                        else:
                            marker = 'o'  # Circle for other points
                            markersize = 8
                            markeredgewidth = 1
                            markeredgecolor = color
                        
                        ax.plot(rank, tp, marker=marker, color=color, markersize=markersize,
                                markeredgewidth=markeredgewidth, markeredgecolor=markeredgecolor)

            ax.set_xlabel('Rank (k)', fontsize=12)
            ax.set_ylabel('True Positives (TP)', fontsize=12)
            ax.set_title('True Positives (TP) vs. Rank', fontsize=14, fontweight='bold')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            
            # Set x-axis labels
            ax.set_xticks(rank_labels)
            ax.set_xticklabels(rank_labels, rotation=0)
            
            plt.tight_layout()

            if output_dir:
                output_path = os.path.join(output_dir, 'tp_vs_rank.png')
                self.save_plot_transposed(plt, output_path)

            if show:
                plt.show()
            else:
                plt.close()

        except Exception as e:
            print(f"Error creating TP vs Rank plot: {e}")

    def plot_fp_vs_rank(self, output_dir=None, show=False):
        """Generate a line graph of False Positives (FP) vs Rank K for each technique."""
        if self.curve_data is None:
            print("No curve data available for plotting FP vs Rank.")
            return

        try:
            fig, ax = plt.subplots(figsize=self.get_figure_size(10))
            
            # Average curve data across runs for each technique and rank
            avg_curve_data = self.curve_data.groupby(['technique', 'rank']).mean().reset_index()
            
            techniques = avg_curve_data['technique'].unique()

            for i, technique in enumerate(techniques):
                tech_data = avg_curve_data[avg_curve_data['technique'] == technique]

                if 'P' not in tech_data.columns or 'N' not in tech_data.columns or 'L' not in tech_data.columns:
                    print(f"Warning: P, N, or L values not found for technique {technique}. Skipping FP plot.")
                    continue

                P = tech_data['P'].iloc[0]
                N = tech_data['N'].iloc[0]
                L = tech_data['L'].iloc[0]
                T = P + N

                ranks_to_plot = sorted(list(set([L] + [round(T/4), round(T/2), round(3*T/4), T])))
                
                fps = []
                rank_labels = sorted(list(set([0] + [round(T/4), round(T/2), round(3*T/4), T])))
                
                for k in ranks_to_plot:
                    # Find the closest rank in the data
                    rank_data = tech_data.iloc[(tech_data['rank'] - k).abs().argsort()[:1]]
                    
                    if rank_data.empty:
                        fps.append(np.nan)
                        continue

                    fpr = rank_data['fpr'].iloc[0]
                    FP = fpr * N
                    fps.append(FP)

                color = self.color_palette[i % len(self.color_palette)]
                
                # Plot the line
                line = ax.plot(ranks_to_plot, fps, marker='', linestyle='--', color=color, label=f'{technique}', linewidth=2)
                
                # Plot markers for all points
                for j, (rank, fp) in enumerate(zip(ranks_to_plot, fps)):
                    if not np.isnan(fp):
                        # Use different marker for L point
                        if rank == L:
                            marker = 'D'  # Diamond for L point
                            markersize = 10
                            markeredgewidth = 2
                            markeredgecolor = color
                        else:
                            marker = 'o'  # Circle for other points
                            markersize = 8
                            markeredgewidth = 1
                            markeredgecolor = color
                        
                        ax.plot(rank, fp, marker=marker, color=color, markersize=markersize,
                                markeredgewidth=markeredgewidth, markeredgecolor=markeredgecolor)

            ax.set_xlabel('Rank (k)', fontsize=12)
            ax.set_ylabel('False Positives (FP)', fontsize=12)
            ax.set_title('False Positives (FP) vs. Rank', fontsize=14, fontweight='bold')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            
            # Set x-axis labels
            ax.set_xticks(rank_labels)
            ax.set_xticklabels(rank_labels, rotation=0)
            
            plt.tight_layout()

            if output_dir:
                output_path = os.path.join(output_dir, 'fp_vs_rank.png')
                self.save_plot_transposed(plt, output_path)

            if show:
                plt.show()
            else:
                plt.close()

        except Exception as e:
            print(f"Error creating FP vs Rank plot: {e}")

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
                
                fig, ax = plt.subplots(figsize=self.get_figure_size(10, 0.8))
                
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
                    ax.plot(alpha_values, scores, label=label, color=color, linewidth=2)
                
                ax.set_xlabel('α (Weight of Metric)', fontsize=12)
                ax.set_ylabel('Score', fontsize=12)
                ax.set_title(f'Trade-off Analysis: {metric.upper()} vs Speed (Thread = {thread_count})', 
                        fontsize=14, fontweight='bold')
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                ax.grid(True, alpha=0.3)
                ax.set_xlim([0, 1])
                ax.set_ylim([0, 1])
                plt.tight_layout()
                
                # Print AUC summary to console
                print(f"\nTrade-off Analysis AUC Summary (Thread = {thread_count}, Metric = {metric.upper()}):")
                print("-" * 60)
                sorted_auc = sorted(auc_values.items(), key=lambda x: x[1], reverse=True)
                for technique, auc in sorted_auc:
                    print(f"  {technique}: {auc:.4f}")
                
                if output_dir:
                    output_path = os.path.join(output_dir, f'tradeoff_analysis_{metric}_thread_{thread_count}.png')
                    self.save_plot_transposed(plt, output_path)
                
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

    def export_performance_rank_tables(self, output_dir=None):
        """
        Export CSV tables showing ordered rank of technique performance
        for both AUROC and AUPR metrics based on summary data only.
        """
        if self.summary_data is None:
            print("No summary data available for performance rank tables")
            return
        
        try:
            # Get unique thread counts
            thread_counts = self.summary_data['num_threads'].unique()
            
            for thread_count in thread_counts:
                # Filter data for current thread count
                thread_data = self.summary_data[self.summary_data['num_threads'] == thread_count].copy()
                
                if len(thread_data) == 0:
                    continue
                
                # Calculate average performance metrics per technique
                performance_avg = thread_data.groupby('technique').agg({
                    'auroc': 'mean',
                    'aupr': 'mean'
                }).reset_index()
                
                # Sort by AUROC descending
                auroc_ranked = performance_avg.sort_values('auroc', ascending=False)
                
                # Sort by AUPR descending
                aupr_ranked = performance_avg.sort_values('aupr', ascending=False)
                
                # Prepare rank tables
                auroc_rank_table = []
                aupr_rank_table = []
                
                # AUROC ranks
                for rank, (_, row) in enumerate(auroc_ranked.iterrows(), 1):
                    auroc_rank_table.append({
                        'Rank': rank,
                        'Technique_AUROC': f"{row['auroc']:.3f} {row['technique']} (threads={thread_count})"
                    })
                
                # AUPR ranks
                for rank, (_, row) in enumerate(aupr_ranked.iterrows(), 1):
                    aupr_rank_table.append({
                        'Rank': rank,
                        'Technique_AUPR': f"{row['aupr']:.3f} {row['technique']} (threads={thread_count})"
                    })
                
                # Convert to DataFrames
                df_auroc_ranks = pd.DataFrame(auroc_rank_table)
                df_aupr_ranks = pd.DataFrame(aupr_rank_table)
                
                # Save to CSV files
                if output_dir:
                    auroc_filename = f"auroc_ranks_thread_{thread_count}.csv"
                    aupr_filename = f"aupr_ranks_thread_{thread_count}.csv"
                    
                    auroc_path = os.path.join(output_dir, auroc_filename)
                    aupr_path = os.path.join(output_dir, aupr_filename)
                    
                    df_auroc_ranks.to_csv(auroc_path, index=False)
                    df_aupr_ranks.to_csv(aupr_path, index=False)
                    
                    print(f"AUROC performance ranks saved to: {auroc_path}")
                    print(f"AUPR performance ranks saved to: {aupr_path}")
            
        except Exception as e:
            print(f"Error exporting performance rank tables: {e}")

    def export_performance_rank_tables_all_threads(self, output_dir=None):
        """
        Export CSV tables showing ordered rank of technique performance
        for both AUROC and AUPR metrics including all thread configurations.
        """
        if self.summary_data is None:
            print("No summary data available for performance rank tables")
            return
        
        try:
            # Calculate average performance metrics per technique and thread count
            performance_avg = self.summary_data.groupby(['technique', 'num_threads']).agg({
                'auroc': 'mean',
                'aupr': 'mean'
            }).reset_index()
            
            # Sort by AUROC descending (all thread counts together)
            auroc_ranked = performance_avg.sort_values('auroc', ascending=False)
            
            # Sort by AUPR descending (all thread counts together)
            aupr_ranked = performance_avg.sort_values('aupr', ascending=False)
            
            # Prepare rank tables
            auroc_rank_table = []
            aupr_rank_table = []
            
            # AUROC ranks (all thread counts together)
            for rank, (_, row) in enumerate(auroc_ranked.iterrows(), 1):
                auroc_rank_table.append({
                    'Rank': rank,
                    'Technique_AUROC': f"{row['auroc']:.3f} {row['technique']} (threads={row['num_threads']})"
                })
            
            # AUPR ranks (all thread counts together)
            for rank, (_, row) in enumerate(aupr_ranked.iterrows(), 1):
                aupr_rank_table.append({
                    'Rank': rank,
                    'Technique_AUPR': f"{row['aupr']:.3f} {row['technique']} (threads={row['num_threads']})"
                })
            
            # Convert to DataFrames
            df_auroc_ranks = pd.DataFrame(auroc_rank_table)
            df_aupr_ranks = pd.DataFrame(aupr_rank_table)
            
            # Save to CSV files
            if output_dir:
                auroc_filename = "auroc_ranks_all_threads.csv"
                aupr_filename = "aupr_ranks_all_threads.csv"
                
                auroc_path = os.path.join(output_dir, auroc_filename)
                aupr_path = os.path.join(output_dir, aupr_filename)
                
                df_auroc_ranks.to_csv(auroc_path, index=False)
                df_aupr_ranks.to_csv(aupr_path, index=False)
                
                print(f"AUROC performance ranks (all threads) saved to: {auroc_path}")
                print(f"AUPR performance ranks (all threads) saved to: {aupr_path}")
        
        except Exception as e:
            print(f"Error exporting performance rank tables for all threads: {e}")

    def plot_tradeoff_analysis_all_threads(self, output_dir=None, show=False, metric='auroc'):
        """
        Generate trade-off analysis between performance metric and execution speed
        with all thread configurations shown together
        
        Args:
            output_dir (str): Directory to save the plot
            show (bool): Whether to display the plot
            metric (str): Performance metric to use ('auroc' or 'aupr')
        """
        if self.combined_data is None:
            print("No combined data available for trade-off analysis")
            return
        
        try:
            # Calculate worst execution time across all data
            worst_time = self.combined_data['avg_execution_time_minutes'].max()
            
            # Calculate relative time score for all data
            combined_data = self.combined_data.copy()
            combined_data['time_relative'] = (worst_time - combined_data['avg_execution_time_minutes']) / worst_time
            
            # Generate alpha values from 0 to 1
            alpha_values = np.arange(0, 1.01, 0.01)
            
            fig, ax = plt.subplots(figsize=self.get_figure_size(10, 0.8))
            
            # Store AUC values for each technique-thread combination
            auc_values = {}
            
            # Get unique techniques and thread counts
            techniques = sorted(combined_data['technique'].unique())
            thread_counts = sorted(combined_data['num_threads'].unique())
            
            # Define linestyles for different thread counts
            linestyles = ['-', '--', '-.', ':']
            markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']
            
            # Plot score for each technique-thread combination
            for i, technique in enumerate(techniques):
                tech_data = combined_data[combined_data['technique'] == technique]
                
                # Get color for technique (consistent across thread counts)
                color = self.color_palette[i % len(self.color_palette)]
                
                for j, thread_count in enumerate(thread_counts):
                    thread_data = tech_data[tech_data['num_threads'] == thread_count]
                    
                    if len(thread_data) == 0:
                        continue
                    
                    row = thread_data.iloc[0]  # Get the row for this technique-thread combination
                    metric_value = row[metric]
                    time_relative = row['time_relative']
                    
                    # Calculate score for each alpha
                    scores = [alpha * metric_value + (1 - alpha) * time_relative for alpha in alpha_values]
                    
                    # Calculate AUC using trapezoidal rule
                    auc = np.trapezoid(scores, alpha_values)
                    key = f"{technique}_threads_{thread_count}"
                    auc_values[key] = auc
                    
                    # Choose linestyle and marker based on thread count
                    linestyle = linestyles[j % len(linestyles)]
                    marker = markers[j % len(markers)] if len(alpha_values) <= 20 else None
                    marker_freq = max(1, len(alpha_values) // 10)  # Show markers at regular intervals
                    
                    # Create label with technique and thread information
                    label = f'{technique} (threads={thread_count})'
                    
                    # Plot line with optional markers
                    if marker and len(alpha_values) <= 20:
                        ax.plot(alpha_values, scores, label=label, color=color, 
                                linestyle=linestyle, marker=marker, markersize=6, 
                                linewidth=2, markevery=marker_freq)
                    else:
                        ax.plot(alpha_values, scores, label=label, color=color, 
                                linestyle=linestyle, linewidth=2)
            
            ax.set_xlabel('α (Weight of Metric)', fontsize=12)
            ax.set_ylabel('Score', fontsize=12)
            ax.set_title(f'Trade-off Analysis: {metric.upper()} vs Speed (All Thread Configurations)', 
                    fontsize=14, fontweight='bold')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            plt.tight_layout()
            
            # Print AUC summary to console
            print(f"\nTrade-off Analysis AUC Summary (All Threads, Metric = {metric.upper()}):")
            print("-" * 70)
            sorted_auc = sorted(auc_values.items(), key=lambda x: x[1], reverse=True)
            for technique_thread, auc in sorted_auc:
                print(f"  {technique_thread}: {auc:.4f}")
            
            if output_dir:
                output_path = os.path.join(output_dir, f'tradeoff_analysis_{metric}_all_threads.png')
                self.save_plot_transposed(plt, output_path)
            
            if show:
                plt.show()
            else:
                plt.close()
                
        except Exception as e:
            print(f"Error creating trade-off analysis plot: {e}")

    def export_tradeoff_rank_tables_all_threads(self, output_dir=None):
        """
        Export CSV tables showing ordered rank of technique trade-off performance
        at fixed weight values for both AUROC and AUPR metrics, including all thread configurations.
        """
        if self.combined_data is None:
            print("No combined data available for trade-off rank tables")
            return
        
        try:
            # Calculate worst execution time across all data
            worst_time = self.combined_data['avg_execution_time_minutes'].max()
            combined_data = self.combined_data.copy()
            combined_data['time_relative'] = (worst_time - combined_data['avg_execution_time_minutes']) / worst_time
            
            # Fixed weight values (converted to decimal)
            fixed_weights_auroc = [0.0, 0.25, 0.5, 0.75, 1.0]
            fixed_weights_aupr = [0.0, 0.25, 0.5, 0.75, 1.0]
            
            # Prepare data for AUROC trade-off ranking (all thread counts together)
            auroc_rank_data = []
            for weight in fixed_weights_auroc:
                # Calculate scores for this weight for all technique-thread combinations
                scores = []
                for _, row in combined_data.iterrows():
                    score = weight * row['auroc'] + (1 - weight) * row['time_relative']
                    scores.append((row['technique'], row['num_threads'], score))
                
                # Sort by score descending
                scores.sort(key=lambda x: x[2], reverse=True)
                
                # Add to rank data
                rank_entry = {'Weight': f"{int(weight*100)}"}
                for rank, (technique, thread_count, score) in enumerate(scores, 1):
                    rank_entry[f'Rank_{rank}'] = f"{score:.3f} {technique} (threads={thread_count})"
                auroc_rank_data.append(rank_entry)
            
            # Prepare data for AUPR trade-off ranking (all thread counts together)
            aupr_rank_data = []
            for weight in fixed_weights_aupr:
                # Calculate scores for this weight for all technique-thread combinations
                scores = []
                for _, row in combined_data.iterrows():
                    score = weight * row['aupr'] + (1 - weight) * row['time_relative']
                    scores.append((row['technique'], row['num_threads'], score))
                
                # Sort by score descending
                scores.sort(key=lambda x: x[2], reverse=True)
                
                # Add to rank data
                rank_entry = {'Weight': f"{int(weight*100)}"}
                for rank, (technique, thread_count, score) in enumerate(scores, 1):
                    rank_entry[f'Rank_{rank}'] = f"{score:.3f} {technique} (threads={thread_count})"
                aupr_rank_data.append(rank_entry)
            
            # Convert to DataFrames
            df_auroc_ranks = pd.DataFrame(auroc_rank_data).T
            df_aupr_ranks = pd.DataFrame(aupr_rank_data).T
            
            # Save to CSV files
            if output_dir:
                auroc_filename = "tradeoff_auroc_ranks_all_threads.csv"
                aupr_filename = "tradeoff_aupr_ranks_all_threads.csv"
                
                auroc_path = os.path.join(output_dir, auroc_filename)
                aupr_path = os.path.join(output_dir, aupr_filename)
                
                df_auroc_ranks.to_csv(auroc_path, index=False, header=False)
                df_aupr_ranks.to_csv(aupr_path, index=False, header=False)
                
                print(f"AUROC trade-off ranks saved to: {auroc_path}")
                print(f"AUPR trade-off ranks saved to: {aupr_path}")
        
        except Exception as e:
            print(f"Error exporting trade-off rank tables: {e}")

    def export_auc_rank_tables_all_threads(self, output_dir=None):
        """
        Export CSV tables showing ordered rank of technique AUC values
        for trade-off performance of both AUROC and AUPR, including all thread configurations.
        """
        if self.combined_data is None:
            print("No combined data available for AUC rank tables")
            return
        
        try:
            # Calculate worst execution time across all data
            worst_time = self.combined_data['avg_execution_time_minutes'].max()
            combined_data = self.combined_data.copy()
            combined_data['time_relative'] = (worst_time - combined_data['avg_execution_time_minutes']) / worst_time
            
            # Generate alpha values from 0 to 1
            alpha_values = np.arange(0, 1.01, 0.01)
            
            # Calculate AUC values for AUROC trade-off (all technique-thread combinations)
            auroc_auc_data = []
            for _, row in combined_data.iterrows():
                technique = row['technique']
                thread_count = row['num_threads']
                scores = [alpha * row['auroc'] + (1 - alpha) * row['time_relative'] for alpha in alpha_values]
                auc = np.trapezoid(scores, alpha_values)
                auroc_auc_data.append((technique, thread_count, auc))
            
            # Calculate AUC values for AUPR trade-off (all technique-thread combinations)
            aupr_auc_data = []
            for _, row in combined_data.iterrows():
                technique = row['technique']
                thread_count = row['num_threads']
                scores = [alpha * row['aupr'] + (1 - alpha) * row['time_relative'] for alpha in alpha_values]
                auc = np.trapezoid(scores, alpha_values)
                aupr_auc_data.append((technique, thread_count, auc))
            
            # Sort by AUC descending
            auroc_auc_data.sort(key=lambda x: x[2], reverse=True)
            aupr_auc_data.sort(key=lambda x: x[2], reverse=True)
            
            # Prepare rank tables
            auroc_rank_table = []
            aupr_rank_table = []
            
            # AUROC AUC ranks (all thread counts together)
            for rank, (technique, thread_count, auc) in enumerate(auroc_auc_data, 1):
                auroc_rank_table.append({
                    'Rank': rank,
                    'Technique_AUC': f"{auc:.3f} {technique} (threads={thread_count})"
                })
            
            # AUPR AUC ranks (all thread counts together)
            for rank, (technique, thread_count, auc) in enumerate(aupr_auc_data, 1):
                aupr_rank_table.append({
                    'Rank': rank,
                    'Technique_AUC': f"{auc:.3f} {technique} (threads={thread_count})"
                })
            
            # Convert to DataFrames
            df_auroc_auc_ranks = pd.DataFrame(auroc_rank_table)
            df_aupr_auc_ranks = pd.DataFrame(aupr_rank_table)
            
            # Save to CSV files
            if output_dir:
                auroc_filename = "auc_tradeoff_auroc_ranks_all_threads.csv"
                aupr_filename = "auc_tradeoff_aupr_ranks_all_threads.csv"
                
                auroc_path = os.path.join(output_dir, auroc_filename)
                aupr_path = os.path.join(output_dir, aupr_filename)
                
                df_auroc_auc_ranks.to_csv(auroc_path, index=False)
                df_aupr_auc_ranks.to_csv(aupr_path, index=False)
                
                print(f"AUROC AUC trade-off ranks saved to: {auroc_path}")
                print(f"AUPR AUC trade-off ranks saved to: {aupr_path}")
        
        except Exception as e:
            print(f"Error exporting AUC rank tables: {e}")

    def plot_performance_metrics_bar(self, output_dir=None, show=False):
        """
        Generate grouped bar charts for performance metrics: 
        - AUROC and AUPR together
        - AUC of Trade-off for AUROC and AUPR together (per thread and all threads)
        
        Args:
            output_dir (str): Directory to save the plots
            show (bool): Whether to display the plots
        """
        if self.combined_data is None or self.summary_data is None:
            print("No combined or summary data available for performance metrics bar chart")
            return
        
        try:
            # Get unique thread counts for trade-off AUC
            thread_counts = sorted(self.combined_data['num_threads'].unique())
            
            # Calculate average AUROC and AUPR per technique (mean across all threads)
            performance_avg = self.summary_data.groupby('technique').agg({
                'auroc': 'mean',
                'aupr': 'mean'
            }).reset_index()
            
            # Sort techniques alphabetically for consistent ordering
            performance_avg = performance_avg.sort_values('technique')
            techniques = performance_avg['technique'].tolist()
            
            # Define colors for AUROC and AUPR
            auroc_color = '#1f77b4'  # Blue
            aupr_color = '#ff7f0e'   # Orange
            
            # Plot 1: Combined AUROC-AUPR Performance Bar Chart
            fig, ax = plt.subplots(figsize=self.get_figure_size(10))
            
            x_pos = np.arange(len(techniques))
            bar_width = 0.35
            
            # Create bars for AUROC and AUPR
            bars_auroc = ax.bar(x_pos - bar_width/2, performance_avg['auroc'], 
                                bar_width, color=auroc_color, alpha=0.7, label='AUROC')
            bars_aupr = ax.bar(x_pos + bar_width/2, performance_avg['aupr'], 
                               bar_width, color=aupr_color, alpha=0.7, label='AUPR')
            
            ax.set_xlabel('Technique', fontsize=12)
            ax.set_ylabel('Performance Score', fontsize=12)
            ax.set_title('AUROC and AUPR Performance by Technique (Mean Across All Threads)', fontsize=14, fontweight='bold')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(techniques, rotation=45, ha='right')
            ax.set_ylim(0, 1)
            ax.legend()
            
            # Add value labels on bars
            for bars in [bars_auroc, bars_aupr]:
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.3f}',
                            ha='center', va='bottom', fontsize=9)
            
            plt.tight_layout()
            
            if output_dir:
                output_path = os.path.join(output_dir, 'auroc-aupr_performance_bar.png')
                self.save_plot_transposed(plt, output_path)
            
            if show:
                plt.show()
            else:
                plt.close()
            
            # Plot 2: Trade-off AUC for AUROC and AUPR (per thread count)
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
                
                # Calculate AUC values for AUROC and AUPR trade-off
                tradeoff_data = []
                for _, row in thread_data.iterrows():
                    technique = row['technique']
                    
                    # AUROC trade-off AUC
                    scores_auroc = [alpha * row['auroc'] + (1 - alpha) * row['time_relative'] for alpha in alpha_values]
                    auc_auroc = np.trapezoid(scores_auroc, alpha_values)
                    
                    # AUPR trade-off AUC
                    scores_aupr = [alpha * row['aupr'] + (1 - alpha) * row['time_relative'] for alpha in alpha_values]
                    auc_aupr = np.trapezoid(scores_aupr, alpha_values)
                    
                    tradeoff_data.append({
                        'technique': technique,
                        'auroc_tradeoff_auc': auc_auroc,
                        'aupr_tradeoff_auc': auc_aupr
                    })
                
                # Create DataFrame and sort by technique name
                tradeoff_df = pd.DataFrame(tradeoff_data)
                tradeoff_df = tradeoff_df.sort_values('technique')
                
                if len(tradeoff_df) == 0:
                    continue
                
                fig, ax = plt.subplots(figsize=self.get_figure_size(10))
                
                x_pos = np.arange(len(tradeoff_df))
                bar_width = 0.35
                
                # Create bars for AUROC and AUPR trade-off AUC
                bars_auroc_tradeoff = ax.bar(x_pos - bar_width/2, tradeoff_df['auroc_tradeoff_auc'], 
                                            bar_width, color=auroc_color, alpha=0.7, label='AUROC Trade-off AUC')
                bars_aupr_tradeoff = ax.bar(x_pos + bar_width/2, tradeoff_df['aupr_tradeoff_auc'], 
                                           bar_width, color=aupr_color, alpha=0.7, label='AUPR Trade-off AUC')
                
                ax.set_xlabel('Technique', fontsize=12)
                ax.set_ylabel('Trade-off AUC', fontsize=12)
                ax.set_title(f'Trade-off AUC for AUROC and AUPR by Technique (Thread = {thread_count})', fontsize=14, fontweight='bold')
                ax.set_xticks(x_pos)
                ax.set_xticklabels(tradeoff_df['technique'], rotation=45, ha='right')
                ax.set_ylim(0, 1)
                ax.legend()
                
                # Add value labels on bars
                for bars in [bars_auroc_tradeoff, bars_aupr_tradeoff]:
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                                f'{height:.3f}',
                                ha='center', va='bottom', fontsize=9)
                
                plt.tight_layout()
                
                if output_dir:
                    output_path = os.path.join(output_dir, f'auc_tradeoff_auroc-aupr_bar_thread_{thread_count}.png')
                    self.save_plot_transposed(plt, output_path)
                
                if show:
                    plt.show()
                else:
                    plt.close()
            
            # Plot 3: Trade-off AUC for AUROC and AUPR (all threads - mean)
            # Calculate worst execution time across all data
            worst_time = self.combined_data['avg_execution_time_minutes'].max()
            combined_data = self.combined_data.copy()
            combined_data['time_relative'] = (worst_time - combined_data['avg_execution_time_minutes']) / worst_time
            
            # Generate alpha values from 0 to 1
            alpha_values = np.arange(0, 1.01, 0.01)
            
            # Calculate AUC values for AUROC and AUPR trade-off (all technique-thread combinations)
            all_tradeoff_data = []
            for _, row in combined_data.iterrows():
                technique = row['technique']
                num_threads = row['num_threads']
                
                # AUROC trade-off AUC
                scores_auroc = [alpha * row['auroc'] + (1 - alpha) * row['time_relative'] for alpha in alpha_values]
                auc_auroc = np.trapezoid(scores_auroc, alpha_values)
                
                # AUPR trade-off AUC
                scores_aupr = [alpha * row['aupr'] + (1 - alpha) * row['time_relative'] for alpha in alpha_values]
                auc_aupr = np.trapezoid(scores_aupr, alpha_values)
                
                all_tradeoff_data.append({
                    'technique': technique,
                    'num_threads': num_threads,
                    'auroc_tradeoff_auc': auc_auroc,
                    'aupr_tradeoff_auc': auc_aupr
                })
            
            # Create DataFrame and calculate mean per technique
            all_tradeoff_df = pd.DataFrame(all_tradeoff_data)
            all_tradeoff_avg = all_tradeoff_df.groupby(['technique', 'num_threads']).agg({
                'auroc_tradeoff_auc': 'mean',
                'aupr_tradeoff_auc': 'mean'
            }).reset_index()
            
            # Sort by technique name
            all_tradeoff_avg = all_tradeoff_avg.sort_values('technique')
            
            if len(all_tradeoff_avg) > 0:
                fig, ax = plt.subplots(figsize=self.get_figure_size(10))
                
                x_pos = np.arange(len(all_tradeoff_avg))
                bar_width = 0.35
                
                # Create bars for AUROC and AUPR trade-off AUC
                bars_auroc_tradeoff = ax.bar(x_pos - bar_width/2, all_tradeoff_avg['auroc_tradeoff_auc'], 
                                            bar_width, color=auroc_color, alpha=0.7, label='AUROC Trade-off AUC')
                bars_aupr_tradeoff = ax.bar(x_pos + bar_width/2, all_tradeoff_avg['aupr_tradeoff_auc'], 
                                           bar_width, color=aupr_color, alpha=0.7, label='AUPR Trade-off AUC')
                
                label = (all_tradeoff_avg['technique'] + ' (threads=' + all_tradeoff_avg['num_threads'].astype(str) + ')').tolist()
                ax.set_xlabel('Technique', fontsize=12)
                ax.set_ylabel('Trade-off AUC', fontsize=12)
                ax.set_title('Trade-off AUC for AUROC and AUPR by Technique (All Threads - Mean)', fontsize=14, fontweight='bold')
                ax.set_xticks(x_pos)
                ax.set_xticklabels(label, rotation=45, ha='right')
                ax.set_ylim(0, 1)
                ax.legend()
                
                # Add value labels on bars
                for bars in [bars_auroc_tradeoff, bars_aupr_tradeoff]:
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                                f'{height:.3f}',
                                ha='center', va='bottom', fontsize=9)
                
                plt.tight_layout()
                
                if output_dir:
                    output_path = os.path.join(output_dir, 'auc_tradeoff_auroc-aupr_bar_all_threads.png')
                    self.save_plot_transposed(plt, output_path)
                
                if show:
                    plt.show()
                else:
                    plt.close()
                    
        except Exception as e:
            print(f"Error creating performance metrics bar charts: {e}")

def main():
    """Main function to run the evaluation analysis"""
    parser = argparse.ArgumentParser(description='Evaluate and visualize performance metrics')
    parser.add_argument('--time_file', type=str, nargs='+', required=True, help='Paths to time data CSV files')
    parser.add_argument('--data_folder', type=str, nargs='+', required=True, help='Paths to folders containing summary and curve data')
    parser.add_argument('--threads', type=int, nargs='+', required=True, help='Thread counts corresponding to each time file and data folder')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for saving graphs')
    parser.add_argument('--show_plots', action='store_true', help='Show plots interactively')
    parser.add_argument('--generate_report', action='store_true', help='Generate comprehensive report')
    parser.add_argument('--figure_width', type=float, default=12.0, help='Base width for figures (default: 12.0)')
    parser.add_argument('--transpose_plots', action='store_true', help='Generate transposed versions of plots')
    
    args = parser.parse_args()
    
    # Validate input lengths
    if len(args.time_file) != len(args.threads) or len(args.data_folder) != len(args.threads):
        print("Error: Number of time files, data folders, and thread counts must be the same")
        print(f"  Time files: {len(args.time_file)}")
        print(f"  Data folders: {len(args.data_folder)}")
        print(f"  Thread counts: {len(args.threads)}")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize analyzer with new parameters
    analyzer = EvaluationAnalyzer(figure_width=args.figure_width, transpose_plots=args.transpose_plots)
    analyzer.setup_plotting()
    
    print("\n" + "="*50)
    print("Starting Evaluation Analysis")
    print("="*50)
    print(f"Time files: {args.time_file}")
    print(f"Data folders: {args.data_folder}")
    print(f"Thread counts: {args.threads}")
    print(f"Output directory: {args.output_dir}")
    print(f"Figure width: {args.figure_width}")
    print(f"Transpose plots: {args.transpose_plots}")
    
    # Load data for all specified thread configurations
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

        print("\nGenerating confusion matrix and TP/FP plots...")
        analyzer.generate_confusion_matrix_tables(output_dir=args.output_dir)
        analyzer.plot_tp_vs_rank(output_dir=args.output_dir, show=args.show_plots)
        analyzer.plot_fp_vs_rank(output_dir=args.output_dir, show=args.show_plots)
        
        # Generate performance metrics bar charts
        analyzer.plot_performance_metrics_bar(output_dir=args.output_dir, show=args.show_plots)
        
        # Generate trade-off analysis for both metrics (individual thread plots)
        analyzer.plot_tradeoff_analysis(output_dir=args.output_dir, show=args.show_plots, metric='auroc')
        analyzer.plot_tradeoff_analysis(output_dir=args.output_dir, show=args.show_plots, metric='aupr')
        
        # Generate trade-off analysis for both metrics (all threads together)
        analyzer.plot_tradeoff_analysis_all_threads(output_dir=args.output_dir, show=args.show_plots, metric='auroc')
        analyzer.plot_tradeoff_analysis_all_threads(output_dir=args.output_dir, show=args.show_plots, metric='aupr')
        
        # Generate trade-off rank tables
        print("\nGenerating trade-off rank tables...")
        analyzer.export_tradeoff_rank_tables(output_dir=args.output_dir)
        analyzer.export_auc_rank_tables(output_dir=args.output_dir)
        analyzer.export_tradeoff_rank_tables_all_threads(output_dir=args.output_dir)
        analyzer.export_auc_rank_tables_all_threads(output_dir=args.output_dir)
        
        # Generate performance rank tables
        print("\nGenerating performance rank tables...")
        analyzer.export_performance_rank_tables(output_dir=args.output_dir)
        analyzer.export_performance_rank_tables_all_threads(output_dir=args.output_dir)
        
        print("\n" + "="*50)
        print("Analysis Complete!")
        print(f"Graphs saved to: {os.path.abspath(args.output_dir)}")
        print("="*50)
        
    else:
        print("Failed to load required data. Analysis cannot proceed.")

if __name__ == "__main__":
    main()