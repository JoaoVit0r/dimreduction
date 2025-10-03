#!/usr/bin/env python3
"""
Evaluation Analysis and Visualization Script

This script analyzes evaluation results from a CSV file and generates
multiple types of graphs to visualize performance metrics.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
import os
import sys
from pathlib import Path
from scipy.integrate import simpson
import warnings

# Set consistent styling
plt.style.use('seaborn-v0_8')

class EvaluationAnalyzer:
    """Main class for analyzing and visualizing evaluation results."""
    
    def __init__(self, csv_path, threads, output_dir):
        """
        Initialize the analyzer.
        
        Args:
            csv_path (str): Path to the CSV file
            threads (int): Number of threads to filter by
            output_dir (str): Directory to save output graphs
        """
        self.csv_path = csv_path
        self.threads = threads
        self.output_dir = output_dir
        self.df = None
        self.filtered_df = None
        self.auc_results = None
        
        # Extended color palette with at least 12 distinct colors
        self.color_palette = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
            '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#aec7e8', '#ffbb78',
            '#98df8a', '#ff9896', '#c5b0d5', '#c49c94', '#f7b6d2', '#c7c7c7',
            '#dbdb8d', '#9edae5', '#393b79', '#637939', '#8c6d31', '#843c39',
            '#7b4173', '#5254a3', '#6b6ecf', '#9c9ede', '#31a354', '#74c476'
        ]
        
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
    def get_color(self, index):
        """Get a color from the extended palette, cycling if necessary."""
        return self.color_palette[index % len(self.color_palette)]
    
    def load_and_preprocess_data(self):
        """Load CSV data and perform preprocessing."""
        try:
            # Load CSV file
            self.df = pd.read_csv(self.csv_path)
            print(f"✅ Loaded data from {self.csv_path}")
            print(f"📊 Original dataset shape: {self.df.shape}")
            
            # Display available columns
            print(f"📋 Available columns: {list(self.df.columns)}")
            
            # Check required columns
            required_columns = ['threshold', 'technique', 'execution_time', 
                              'F1 Score', 'Precision', 'Recall (Sensitivity)', 'threads']
            missing_columns = [col for col in required_columns if col not in self.df.columns]
            
            if missing_columns:
                print(f"❌ Missing required columns: {missing_columns}")
                return False
            
            # Filter by threads
            self.filtered_df = self.df[self.df['threads'] == self.threads].copy()
            print(f"🔍 Filtered to {self.threads} thread(s): {self.filtered_df.shape[0]} rows")
            
            # Check for missing execution times
            missing_time = self.filtered_df['execution_time'].isna().sum()
            if missing_time > 0:
                print(f"⚠️  Warning: Ignoring {missing_time} entries with missing execution time")
                self.filtered_df = self.filtered_df.dropna(subset=['execution_time'])
            
            # Convert execution time to minutes
            self.filtered_df['execution_time_min'] = self.filtered_df['execution_time'] / 60.0
            
            print(f"✅ Preprocessing complete. Final dataset shape: {self.filtered_df.shape}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def calculate_auc(self, x_values, y_values):
        """Calculate Area Under Curve using Simpson's rule."""
        try:
            # Ensure we have valid data
            if len(x_values) < 2 or len(y_values) < 2:
                return 0.0
            
            # Use Simpson's rule for numerical integration
            auc = simpson(y_values, x_values)
            return max(auc, 0)  # AUC should be non-negative
        except:
            # Fallback to trapezoidal rule if Simpson fails
            return np.trapz(y_values, x_values)
    
    def create_execution_time_chart(self, show_plot=False):
        """Create bar chart comparing execution times by technique."""
        try:
            # Use only threshold=0 data for execution time comparison
            threshold_zero_data = self.filtered_df[self.filtered_df['threshold'] == 0]
            
            if threshold_zero_data.empty:
                print("⚠️  No data with threshold=0 found. Using all data for execution time comparison.")
                threshold_zero_data = self.filtered_df
            
            # Calculate average execution time per technique
            avg_times = threshold_zero_data.groupby('technique')['execution_time_min'].mean().sort_values()
            
            plt.figure(figsize=(12, 6))
            
            # Create bars with colors from extended palette
            bars = plt.bar(range(len(avg_times)), avg_times.values, 
                          color=[self.get_color(i) for i in range(len(avg_times))])
            
            # Add value labels on bars
            for i, bar in enumerate(bars):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f} min', ha='center', va='bottom')
            
            plt.title('Average Execution Time by Technique', fontsize=14, fontweight='bold')
            plt.xlabel('Technique', fontsize=12)
            plt.ylabel('Execution Time (minutes)', fontsize=12)
            plt.xticks(range(len(avg_times)), avg_times.index, rotation=45, ha='right')
            plt.tight_layout()
            
            # Save the plot
            filename = os.path.join(self.output_dir, 'execution_time_comparison.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"💾 Saved execution time chart to {filename}")
            
            if show_plot:
                plt.show()
            else:
                plt.close()
                
        except Exception as e:
            print(f"❌ Error creating execution time chart: {e}")
    
    def create_metric_vs_threshold_linechart(self, show_plot=False):
        """Create multi-line chart showing metrics vs threshold by technique."""
        try:
            # Aggregate data by technique and threshold
            aggregated = self.filtered_df.groupby(['technique', 'threshold']).agg({
                'F1 Score': 'mean',
                'Precision': 'mean', 
                'Recall (Sensitivity)': 'mean'
            }).reset_index()
            
            metrics = ['F1 Score', 'Precision', 'Recall (Sensitivity)']
            techniques = aggregated['technique'].unique()
            
            # Create subplots for each metric
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            for i, metric in enumerate(metrics):
                ax = axes[i]
                
                for j, technique in enumerate(techniques):
                    tech_data = aggregated[aggregated['technique'] == technique]
                    ax.plot(tech_data['threshold'], tech_data[metric], 
                           marker='o', linewidth=2, label=technique,
                           color=self.get_color(j))
                
                ax.set_title(f'{metric} vs Threshold', fontweight='bold')
                ax.set_xlabel('Threshold')
                ax.set_ylabel(metric)
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                ax.grid(True, alpha=0.3)
            
            plt.suptitle('Metric Performance vs Threshold by Technique', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            # Save the plot
            filename = os.path.join(self.output_dir, 'metric_vs_threshold_line.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"💾 Saved metric vs threshold line chart to {filename}")
            
            if show_plot:
                plt.show()
            else:
                plt.close()
                
        except Exception as e:
            print(f"❌ Error creating metric vs threshold line chart: {e}")
    
    def create_metric_vs_threshold_barchart(self, show_plot=False):
        """Create grouped bar chart showing metrics by technique and threshold."""
        try:
            # Aggregate data
            aggregated = self.filtered_df.groupby(['technique', 'threshold']).agg({
                'F1 Score': 'mean',
                'Precision': 'mean',
                'Recall (Sensitivity)': 'mean'
            }).reset_index()
            
            metrics = ['F1 Score', 'Precision', 'Recall (Sensitivity)']
            thresholds = sorted(aggregated['threshold'].unique())
            techniques = aggregated['technique'].unique()
            
            # Create subplots for each metric
            fig, axes = plt.subplots(1, 3, figsize=(20, 6))
            
            for i, metric in enumerate(metrics):
                ax = axes[i]
                
                # Prepare data for grouped bar chart
                bar_width = 0.8 / len(thresholds)
                x_pos = np.arange(len(techniques))
                
                for j, threshold in enumerate(thresholds):
                    threshold_data = aggregated[aggregated['threshold'] == threshold]
                    values = []
                    for tech in techniques:
                        tech_data = threshold_data[threshold_data['technique'] == tech]
                        if not tech_data.empty:
                            values.append(tech_data[metric].values[0])
                        else:
                            values.append(0)  # or np.nan if you prefer
                    
                    positions = x_pos + j * bar_width - (len(thresholds) - 1) * bar_width / 2
                    bars = ax.bar(positions, values, bar_width, 
                                 label=f'Thresh={threshold}',
                                 color=self.get_color(j))
                    
                    # Add value labels on bars
                    for k, bar in enumerate(bars):
                        height = bar.get_height()
                        if height > 0:  # Only label non-zero bars
                            ax.text(bar.get_x() + bar.get_width()/2., height,
                                   f'{height:.3f}', ha='center', va='bottom', fontsize=8)
                
                ax.set_title(f'{metric} by Technique and Threshold', fontweight='bold')
                ax.set_xlabel('Technique')
                ax.set_ylabel(metric)
                ax.set_xticks(x_pos)
                ax.set_xticklabels(techniques, rotation=45, ha='right')
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                ax.grid(True, alpha=0.3, axis='y')
            
            plt.suptitle('Metric Performance by Technique and Threshold', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            # Save the plot
            filename = os.path.join(self.output_dir, 'metric_vs_threshold_bar.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"💾 Saved metric vs threshold bar chart to {filename}")
            
            if show_plot:
                plt.show()
            else:
                plt.close()
                
        except Exception as e:
            print(f"❌ Error creating metric vs threshold bar chart: {e}")
    
    def create_tradeoff_analysis(self, show_plot=False):
        """Create trade-off analysis charts for each threshold and each metric."""
        try:
            # Aggregate data by technique and threshold
            aggregated = self.filtered_df.groupby(['technique', 'threshold']).agg({
                'F1 Score': 'mean',
                'Precision': 'mean',
                'Recall (Sensitivity)': 'mean',
                'execution_time_min': 'mean'
            }).reset_index()
            
            thresholds = sorted(aggregated['threshold'].unique())
            techniques = aggregated['technique'].unique()
            alpha_values = np.arange(0, 1.01, 0.01)
            
            metrics = [
                ('F1 Score', 'F1 Score'),
                ('Precision', 'Precision'), 
                ('Recall (Sensitivity)', 'Recall')
            ]
            
            # Initialize AUC results storage
            self.auc_results = []
            
            # Create separate plots for each threshold and each metric
            for threshold in thresholds:
                threshold_data = aggregated[aggregated['threshold'] == threshold]
                
                if threshold_data.empty:
                    continue
                
                # Create subplots for all metrics
                fig, axes = plt.subplots(1, 3, figsize=(18, 5))
                
                for metric_idx, (metric_col, metric_name) in enumerate(metrics):
                    ax = axes[metric_idx]
                    
                    # Calculate min and max values for normalization
                    metric_min = threshold_data[metric_col].min()
                    metric_max = threshold_data[metric_col].max()
                    time_min = threshold_data['execution_time_min'].min()
                    time_max = threshold_data['execution_time_min'].max()
                    
                    for tech_idx, technique in enumerate(techniques):
                        tech_data = threshold_data[threshold_data['technique'] == technique]
                        
                        if tech_data.empty:
                            continue
                        
                        metric_value = tech_data[metric_col].values[0]
                        time_value = tech_data['execution_time_min'].values[0]
                        
                        # Calculate normalized scores
                        if metric_max != metric_min:
                            metric_normalized = (metric_value - metric_min) / (metric_max - metric_min)
                        else:
                            metric_normalized = 0.5
                            
                        if time_max != time_min:
                            time_normalized = (time_max - time_value) / (time_max - time_min)
                        else:
                            time_normalized = 0.5
                        
                        # Calculate score for each alpha
                        scores = [alpha * metric_normalized + (1 - alpha) * time_normalized 
                                 for alpha in alpha_values]
                        
                        # Calculate AUC for this trade-off curve
                        auc = self.calculate_auc(alpha_values, scores)
                        self.auc_results.append({
                            'threshold': threshold,
                            'technique': technique,
                            'metric': metric_name,
                            'auc': auc,
                            'metric_value': metric_value,
                            'time_value': time_value
                        })
                        
                        # Add AUC to legend label
                        label = f'{technique} (AUC: {auc:.3f})'
                        ax.plot(alpha_values, scores, linewidth=2, label=label,
                               color=self.get_color(tech_idx))
                    
                    ax.set_title(f'{metric_name} vs Speed Trade-off\n(Threshold = {threshold})', 
                               fontweight='bold')
                    ax.set_xlabel('α (Weight of Metric)', fontsize=10)
                    ax.set_ylabel('Combined Score', fontsize=10)
                    ax.grid(True, alpha=0.3)
                    ax.set_xlim(0, 1)
                    
                    # Only show legend on the first subplot to avoid repetition
                    if metric_idx == 0:
                        ax.legend(bbox_to_anchor=(0, 1.02, 3, 0.2), loc="lower left", 
                                 mode="expand", borderaxespad=0, ncol=2, fontsize=8)
                
                # Add explanatory text below the subplots
                plt.figtext(0.02, 0.02, 
                           f'Score = α × Metric_normalized + (1-α) × Speed_normalized | '
                           f'AUC = Area Under Curve (higher is better)',
                           fontsize=9, alpha=0.7)
                
                plt.tight_layout(rect=[0, 0.05, 1, 0.95])
                
                # Save the plot
                filename = os.path.join(self.output_dir, f'tradeoff_analysis_threshold_{threshold}.png')
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                print(f"💾 Saved trade-off analysis for threshold {threshold} to {filename}")
                
                if show_plot:
                    plt.show()
                else:
                    plt.close()
                    
                # Also create individual metric trade-off plots for higher detail
                self.create_individual_tradeoff_plots(threshold_data, threshold, techniques, 
                                                     alpha_values, metrics, show_plot)
            
            # Create AUC summary table and visualization
            self.create_auc_summary(show_plot)
                    
        except Exception as e:
            print(f"❌ Error creating trade-off analysis: {e}")
    
    def create_individual_tradeoff_plots(self, threshold_data, threshold, techniques, 
                                       alpha_values, metrics, show_plot=False):
        """Create individual trade-off plots for each metric with larger size."""
        try:
            for metric_col, metric_name in metrics:
                plt.figure(figsize=(12, 8))
                
                # Calculate min and max values for normalization
                metric_min = threshold_data[metric_col].min()
                metric_max = threshold_data[metric_col].max()
                time_min = threshold_data['execution_time_min'].min()
                time_max = threshold_data['execution_time_min'].max()
                
                for tech_idx, technique in enumerate(techniques):
                    tech_data = threshold_data[threshold_data['technique'] == technique]
                    
                    if tech_data.empty:
                        continue
                    
                    metric_value = tech_data[metric_col].values[0]
                    time_value = tech_data['execution_time_min'].values[0]
                    
                    # Calculate normalized scores
                    if metric_max != metric_min:
                        metric_normalized = (metric_value - metric_min) / (metric_max - metric_min)
                    else:
                        metric_normalized = 0.5
                        
                    if time_max != time_min:
                        time_normalized = (time_max - time_value) / (time_max - time_min)
                    else:
                        time_normalized = 0.5
                    
                    # Calculate score for each alpha
                    scores = [alpha * metric_normalized + (1 - alpha) * time_normalized 
                             for alpha in alpha_values]
                    
                    # Calculate AUC for this trade-off curve
                    auc = self.calculate_auc(alpha_values, scores)
                    
                    # Add AUC to the legend
                    label = f'{technique} (AUC: {auc:.3f})'
                    plt.plot(alpha_values, scores, linewidth=3, label=label,
                            color=self.get_color(tech_idx), marker='', markersize=2)
                
                plt.title(f'Trade-off Analysis: {metric_name} vs Speed (Threshold = {threshold})', 
                         fontsize=14, fontweight='bold')
                plt.xlabel('α (Weight of Metric)', fontsize=12)
                plt.ylabel('Combined Score', fontsize=12)
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.grid(True, alpha=0.3)
                plt.xlim(0, 1)
                
                # Add explanatory text
                plt.figtext(0.02, 0.02, 
                           f'Score = α × Metric_normalized + (1-α) × Speed_normalized\n'
                           f'AUC = Area Under Curve (higher values indicate better overall trade-off performance)',
                           fontsize=10, alpha=0.7)
                
                plt.tight_layout()
                
                # Save the individual plot
                filename = os.path.join(self.output_dir, 
                                      f'tradeoff_{metric_name.replace(" ", "_").lower()}_threshold_{threshold}.png')
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                print(f"💾 Saved individual {metric_name} trade-off plot to {filename}")
                
                if show_plot:
                    plt.show()
                else:
                    plt.close()
                    
        except Exception as e:
            print(f"❌ Error creating individual trade-off plots: {e}")
    
    def create_auc_summary(self, show_plot=False):
        """Create AUC summary table and visualization."""
        if not self.auc_results:
            print("⚠️  No AUC results to summarize")
            return
            
        try:
            # Create DataFrame from AUC results
            auc_df = pd.DataFrame(self.auc_results)
            
            # Save AUC results to CSV
            auc_csv_path = os.path.join(self.output_dir, 'auc_summary.csv')
            auc_df.to_csv(auc_csv_path, index=False)
            print(f"💾 Saved AUC summary table to {auc_csv_path}")
            
            # Display AUC summary in console
            print("\n" + "="*60)
            print("📊 AUC Summary (Area Under Trade-off Curve)")
            print("="*60)
            print("Higher AUC values indicate better overall trade-off performance")
            print("-"*60)
            
            # Group by metric and threshold for better display
            for metric in auc_df['metric'].unique():
                print(f"\n🔍 {metric} Trade-off AUC Results:")
                metric_data = auc_df[auc_df['metric'] == metric]
                
                for threshold in metric_data['threshold'].unique():
                    thresh_data = metric_data[metric_data['threshold'] == threshold]
                    print(f"   Threshold {threshold}:")
                    
                    for _, row in thresh_data.sort_values('auc', ascending=False).iterrows():
                        print(f"     {row['technique']:20} AUC: {row['auc']:.3f}")
            
            # Create AUC visualization
            self.create_auc_visualization(auc_df, show_plot)
            
        except Exception as e:
            print(f"❌ Error creating AUC summary: {e}")
    
    def create_auc_visualization(self, auc_df, show_plot=False):
        """Create visualizations for AUC results."""
        try:
            # Create a figure with subplots for different visualizations
            # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
            fig, (ax1) = plt.subplots(1, 1, figsize=(15, 12))
            ax2 = ax1
            
            techniques = auc_df['technique'].unique()
            metrics = auc_df['metric'].unique()
            thresholds = auc_df['threshold'].unique()
            # # Plot 1: AUC by technique and metric (grouped bar chart)
            # # Create grouped bar chart for the first threshold (or average if multiple thresholds)
            # if len(thresholds) == 1:
            #     threshold = thresholds[0]
            #     plot_data = auc_df[auc_df['threshold'] == threshold]
                
            #     # Pivot data for grouped bar chart
            #     pivot_data = plot_data.pivot(index='technique', columns='metric', values='auc')
                
            #     x_pos = np.arange(len(techniques))
            #     bar_width = 0.25
                
            #     for i, metric in enumerate(metrics):
            #         metric_data = pivot_data[metric].reindex(techniques)
            #         ax1.bar(x_pos + i * bar_width, metric_data, bar_width, 
            #                label=metric, color=self.get_color(i))
                
            #     ax1.set_xlabel('Technique')
            #     ax1.set_ylabel('AUC Value')
            #     ax1.set_title(f'AUC by Technique and Metric (Threshold = {threshold})', 
            #                 fontweight='bold')
            #     ax1.set_xticks(x_pos + bar_width)
            #     ax1.set_xticklabels(techniques, rotation=45, ha='right')
            #     ax1.legend()
            #     ax1.grid(True, alpha=0.3, axis='y')
                
            #     # Add value labels on bars
            #     for i, technique in enumerate(techniques):
            #         for j, metric in enumerate(metrics):
            #             value = pivot_data.loc[technique, metric] if technique in pivot_data.index else 0
            #             if not np.isnan(value):
            #                 ax1.text(i + j * bar_width, value + 0.01, f'{value:.3f}', 
            #                        ha='center', va='bottom', fontsize=8)
            
            # Plot 2: AUC heatmap across thresholds and techniques for the first metric
            if len(metrics) > 0:
                primary_metric = metrics[0]
                heatmap_data = auc_df[auc_df['metric'] == primary_metric]
                
                if len(thresholds) > 1:
                    # Create pivot table for heatmap
                    pivot_heatmap = heatmap_data.pivot(index='technique', columns='threshold', values='auc')
                    
                    # Create heatmap
                    im = ax2.imshow(pivot_heatmap.values, cmap='YlOrRd', aspect='auto')
                    
                    # Set labels
                    ax2.set_xticks(range(len(pivot_heatmap.columns)))
                    ax2.set_xticklabels([f'Thresh {t}' for t in pivot_heatmap.columns])
                    ax2.set_yticks(range(len(pivot_heatmap.index)))
                    ax2.set_yticklabels(pivot_heatmap.index)
                    
                    # Add value annotations
                    for i in range(len(pivot_heatmap.index)):
                        for j in range(len(pivot_heatmap.columns)):
                            ax2.text(j, i, f'{pivot_heatmap.iloc[i, j]:.3f}',
                                   ha="center", va="center", color="black", fontsize=8)
                    
                    ax2.set_title(f'AUC Heatmap: {primary_metric} across Thresholds', 
                                fontweight='bold')
                    plt.colorbar(im, ax=ax2, label='AUC Value')
                else:
                    # If only one threshold, create a simple bar chart
                    single_thresh_data = heatmap_data.sort_values('auc', ascending=False)
                    bars = ax2.bar(range(len(single_thresh_data)), single_thresh_data['auc'],
                                 color=[self.get_color(i) for i in range(len(single_thresh_data))])
                    
                    ax2.set_xlabel('Technique')
                    ax2.set_ylabel('AUC Value')
                    ax2.set_title(f'AUC for {primary_metric} (Threshold = {thresholds[0]})', 
                                fontweight='bold')
                    ax2.set_xticks(range(len(single_thresh_data)))
                    ax2.set_xticklabels(single_thresh_data['technique'], rotation=45, ha='right')
                    ax2.grid(True, alpha=0.3, axis='y')
                    
                    # Add value labels
                    for bar in bars:
                        height = bar.get_height()
                        ax2.text(bar.get_x() + bar.get_width()/2., height,
                               f'{height:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            
            # Save AUC visualization
            filename = os.path.join(self.output_dir, 'auc_analysis.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"💾 Saved AUC analysis visualization to {filename}")
            
            if show_plot:
                plt.show()
            else:
                plt.close()
                
        except Exception as e:
            print(f"❌ Error creating AUC visualization: {e}")
            # Create a simpler visualization if the complex one fails
            self.create_simple_auc_plot(auc_df, show_plot)
    
    def create_simple_auc_plot(self, auc_df, show_plot=False):
        """Create a simpler AUC plot as fallback."""
        try:
            plt.figure(figsize=(12, 8))
            
            # Group by technique and calculate average AUC across metrics and thresholds
            avg_auc = auc_df.groupby('technique')['auc'].mean().sort_values(ascending=False)
            
            bars = plt.bar(range(len(avg_auc)), avg_auc.values,
                         color=[self.get_color(i) for i in range(len(avg_auc))])
            
            plt.title('Average AUC Across All Metrics and Thresholds', fontweight='bold')
            plt.xlabel('Technique')
            plt.ylabel('Average AUC Value')
            plt.xticks(range(len(avg_auc)), avg_auc.index, rotation=45, ha='right')
            plt.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            
            filename = os.path.join(self.output_dir, 'auc_simple_summary.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"💾 Saved simple AUC summary to {filename}")
            
            if show_plot:
                plt.show()
            else:
                plt.close()
                
        except Exception as e:
            print(f"❌ Error creating simple AUC plot: {e}")
    
    def generate_all_visualizations(self, show_plots=False):
        """Generate all visualizations."""
        print("\n" + "="*60)
        print("📈 Generating Visualizations")
        print("="*60)
        
        self.create_execution_time_chart(show_plots)
        self.create_metric_vs_threshold_linechart(show_plots)
        self.create_metric_vs_threshold_barchart(show_plots)
        self.create_tradeoff_analysis(show_plots)
        
        print("\n✅ All visualizations completed!")


def main():
    """Main function to handle user input and execute analysis."""
    parser = argparse.ArgumentParser(description='Evaluation Analysis and Visualization Script')
    parser.add_argument('--csv', type=str, help='Path to CSV file')
    parser.add_argument('--threads', type=int, help='Number of threads to filter by')
    parser.add_argument('--output', type=str, default='output_graphs', 
                       help='Output directory for graphs (default: output_graphs)')
    parser.add_argument('--show', action='store_true', 
                       help='Show plots interactively (default: save only)')
    
    args = parser.parse_args()
    
    # Get CSV file path
    if args.csv:
        csv_path = args.csv
    else:
        csv_path = input("📁 Enter the path to your CSV file: ").strip()
    
    if not os.path.exists(csv_path):
        print(f"❌ File not found: {csv_path}")
        sys.exit(1)
    
    # Get number of threads
    if args.threads is not None:
        threads = args.threads
    else:
        try:
            threads = int(input("🔢 Enter the number of threads to analyze: "))
        except ValueError:
            print("❌ Please enter a valid integer for threads")
            sys.exit(1)
    
    # Get output directory
    output_dir = args.output
    
    # Create analyzer instance
    analyzer = EvaluationAnalyzer(csv_path, threads, output_dir)
    
    # Load and preprocess data
    if not analyzer.load_and_preprocess_data():
        sys.exit(1)
    
    # Generate visualizations
    analyzer.generate_all_visualizations(show_plots=args.show)
    
    print(f"\n🎉 Analysis complete! All graphs saved to: {output_dir}")


if __name__ == "__main__":
    main()