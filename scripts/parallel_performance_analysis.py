import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from pathlib import Path
from scipy.interpolate import make_interp_spline

def parse_duration(duration_str):
    """Convert duration string (HH:MM:SS) to seconds"""
    parts = list(map(int, duration_str.split(':')))
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    elif len(parts) == 2:
        return parts[0] * 60 + parts[1]
    else:
        return parts[0]

def load_and_process_data(time_file, portion_file):
    """Load and process the input data files"""
    
    # Load execution times
    time_df = pd.read_csv(time_file)
    
    # Parse duration to seconds
    time_df['duration_seconds'] = time_df['duration'].apply(parse_duration)
    
    # Calculate average execution time per thread count and distribution type
    avg_times = time_df.groupby(['num_threads_total', 'distribution_type'])['duration_seconds'].mean().reset_index()
    
    # Load parallelizable portions
    portion_df = pd.read_csv(portion_file)
    
    # Map distribution types to aliases
    distribution_map = {
        'demain': 'Dynamic Assignment',
        'sequential': 'Slice Groups', 
        'spaced': 'Stride Groups'
    }
    
    avg_times['distribution_alias'] = avg_times['distribution_type'].map(distribution_map)
    portion_df['distribution_alias'] = portion_df.iloc[:, 0].map(distribution_map)
    
    return avg_times, portion_df

def calculate_metrics(avg_times, portion_df):
    """Calculate all performance metrics"""
    
    results = []
    
    for distribution in avg_times['distribution_alias'].unique():
        dist_data = avg_times[avg_times['distribution_alias'] == distribution].copy()
        portion_data = portion_df[portion_df['distribution_alias'] == distribution]
        
        if len(portion_data) == 0:
            print(f"Warning: No parallel portion data found for {distribution}")
            continue
            
        # Get parallel fraction (using recoverednetwork_time_percentage)
        parallel_fraction = portion_data['recoverednetwork_time_percentage'].values[0]
        
        # Get single-thread execution time (baseline)
        single_thread_time = dist_data[dist_data['num_threads_total'] == 1]['duration_seconds'].values[0]
        
        for _, row in dist_data.iterrows():
            threads = row['num_threads_total']
            exec_time = row['duration_seconds']
            
            # Real Speedup
            real_speedup = single_thread_time / exec_time
            
            # Ideal Efficiency
            parallel_efficiency = real_speedup / threads
            
            # Amdahl's Law Speedup
            amdahl_speedup = 1 / ((1 - parallel_fraction) + (parallel_fraction / threads))
            
            # Amdahl's Law Efficiency
            amdahl_efficiency = amdahl_speedup / threads
            
            
            speedup_efficiency = parallel_efficiency / amdahl_efficiency
            
            results.append({
                'distribution': distribution,
                'threads': threads,
                'execution_time': exec_time,
                'real_speedup': real_speedup,
                'parallel_efficiency': parallel_efficiency,
                'amdahl_speedup': amdahl_speedup,
                'amdahl_efficiency': amdahl_efficiency,
                'speedup_efficiency': speedup_efficiency,
                'parallel_fraction': parallel_fraction
            })
    
    return pd.DataFrame(results)

def create_plots(results_df, output_dir, figure_width=10, suffix=''):
    """Create the four performance graphs"""
    
    # Set up plotting style
    plt.style.use('default')
    
    # Define colors for each distribution type
    colors = {
        'Dynamic Assignment': '#1f77b4',
        'Slice Groups': '#ff7f0e', 
        'Stride Groups': '#2ca02c'
    }
    
    # Define markers for each distribution type
    markers = {
        'Dynamic Assignment': 'o',
        'Slice Groups': 's',
        'Stride Groups': '^'
    }
    
    # Calculate figure size (width x height)
    fig_width = figure_width
    fig_height = fig_width * 0.8  # Aspect ratio
    
    # 1. Execution Time vs. Threads
    fig1, ax1 = plt.subplots(figsize=(fig_width, fig_height))
    
    for distribution in results_df['distribution'].unique():
        dist_data = results_df[results_df['distribution'] == distribution].sort_values('threads')
        color = colors[distribution]
        marker = markers[distribution]
        
        ax1.plot(dist_data['threads'], dist_data['execution_time']/60, 
                label=distribution, color=color, marker=marker, linewidth=2, markersize=8)
    
    ax1.set_xlabel('Number of Threads', fontsize=12)
    ax1.set_ylabel('Execution Time (minutes)', fontsize=12)
    ax1.set_title('Execution Time vs. Threads', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log', base=2)
    ax1.set_xticks(results_df['threads'].unique())
    ax1.legend(bbox_to_anchor=(0.5, -0.1), loc='upper center', ncol=3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'execution_time_vs_threads{suffix}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Real Speedup vs. Threads
    fig2, ax2 = plt.subplots(figsize=(fig_width, fig_height))
    
    # Ideal speedup line (linear)
    ideal_threads = np.sort(results_df['threads'].unique())
    ax2.plot(ideal_threads, ideal_threads, 'k--', label='Ideal Speedup', alpha=0.7)
    
    for distribution in results_df['distribution'].unique():
        dist_data = results_df[results_df['distribution'] == distribution].sort_values('threads')
        color = colors[distribution]
        marker = markers[distribution]
        
        x = dist_data['threads']
        y = dist_data['real_speedup']

        if len(x) > 1:
            spline = make_interp_spline(x, y, k=min(2, len(x)-1))
            x_smooth = np.linspace(x.min(), x.max(), 80)
            y_smooth = spline(x_smooth)
            ax2.plot(x_smooth, y_smooth, label=distribution, color=color, linewidth=2)

        ax2.scatter(x, y, color=color, marker=marker, s=80, label=distribution if len(x) <= 1 else '')

    ax2.set_xlabel('Number of Threads', fontsize=12)
    ax2.set_ylabel('Speedup', fontsize=12)
    ax2.set_title('Speedup vs. Threads', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log', base=2)
    ax2.set_xticks(results_df['threads'].unique())
    ax2.legend(bbox_to_anchor=(0.5, -0.1), loc='upper center', ncol=4)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'speedup_vs_threads{suffix}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Parallel Efficiency vs. Threads
    fig3, ax3 = plt.subplots(figsize=(fig_width, fig_height))
    
    for distribution in results_df['distribution'].unique():
        dist_data = results_df[results_df['distribution'] == distribution].sort_values('threads')
        color = colors[distribution]
        marker = markers[distribution]
        
        x = dist_data['threads']
        y = dist_data['parallel_efficiency']

        if len(x) > 1:
            spline = make_interp_spline(x, y, k=min(2, len(x)-1))
            x_smooth = np.linspace(x.min(), x.max(), 80)
            y_smooth = spline(x_smooth)
            ax3.plot(x_smooth, y_smooth, label=distribution, color=color, linewidth=2)

        ax3.scatter(x, y, color=color, marker=marker, s=80, label=distribution if len(x) <= 1 else '')

    ax3.set_xlabel('Number of Threads', fontsize=12)
    ax3.set_ylabel('Efficiency', fontsize=12)
    ax3.set_title('Parallel Efficiency vs. Threads', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_xscale('log', base=2)
    ax3.set_xticks(results_df['threads'].unique())
    ax3.legend(bbox_to_anchor=(0.5, -0.1), loc='upper center', ncol=3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'parallel_efficiency_vs_threads{suffix}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Speedup Efficiency vs. Threads
    fig4, ax4 = plt.subplots(figsize=(fig_width, fig_height))
    
    for distribution in results_df['distribution'].unique():
        dist_data = results_df[results_df['distribution'] == distribution].sort_values('threads')
        color = colors[distribution]
        marker = markers[distribution]
        
        x = dist_data['threads']
        y = dist_data['speedup_efficiency']

        if len(x) > 1:
            spline = make_interp_spline(x, y, k=min(2, len(x)-1))
            x_smooth = np.linspace(x.min(), x.max(), 80)
            y_smooth = spline(x_smooth)
            ax4.plot(x_smooth, y_smooth, label=distribution, color=color, linewidth=2)

        ax4.scatter(x, y, color=color, marker=marker, s=80, label=distribution if len(x) <= 1 else '')

    ax4.set_xlabel('Number of Threads', fontsize=12)
    ax4.set_ylabel('Efficiency', fontsize=12)
    ax4.set_title('Speedup Efficiency vs. Threads', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.set_xscale('log', base=2)
    ax4.set_xticks(results_df['threads'].unique())
    ax4.legend(bbox_to_anchor=(0.5, -0.1), loc='upper center', ncol=3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'speedup_efficiency_vs_threads{suffix}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    
    # 5. Real Speedup vs. Threads
    fig5, ax5 = plt.subplots(figsize=(fig_width, fig_height))
    
    # Ideal speedup line (linear)
    ideal_threads = np.sort(results_df['threads'].unique())
    ax5.plot(ideal_threads, ideal_threads, 'k--', label='Ideal Speedup', alpha=0.7)
    
    for distribution in results_df['distribution'].unique():
        dist_data = results_df[results_df['distribution'] == distribution].sort_values('threads')
        color = colors[distribution]
        marker = markers[distribution]
        
        x = dist_data['threads']
        y = dist_data['real_speedup']

        if len(x) > 1:
            spline = make_interp_spline(x, y, k=min(2, len(x)-1))
            x_smooth = np.linspace(x.min(), x.max(), 80)
            y_smooth = spline(x_smooth)
            ax5.plot(x_smooth, y_smooth, label=distribution, color=color, linewidth=2)

        ax5.scatter(x, y, color=color, marker=marker, s=80, label=distribution if len(x) <= 1 else '')

    ax5.set_xlabel('Number of Threads', fontsize=12)
    ax5.set_ylabel('Speedup', fontsize=12)
    ax5.set_title('Speedup vs. Threads', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.set_xticks(results_df['threads'].unique())
    ax5.legend(bbox_to_anchor=(0.5, -0.1), loc='upper center', ncol=4)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'speedup_vs_threads_noLog{suffix}.png'), dpi=300, bbox_inches='tight')
    plt.close()

def generate_summary_report(results_df, output_dir, suffix=''):
    """Generate a summary report with key findings"""
    
    report_path = os.path.join(output_dir, f'performance_analysis_report{suffix}.txt')
    
    with open(report_path, 'w') as f:
        f.write("PARALLEL PERFORMANCE ANALYSIS REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        # Overall summary
        f.write("EXECUTION OVERVIEW:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total thread configurations analyzed: {len(results_df)}\n")
        f.write(f"Thread counts tested: {sorted(results_df['threads'].unique())}\n")
        f.write(f"Scheduling strategies: {list(results_df['distribution'].unique())}\n\n")
        
        # Best performing configurations
        f.write("BEST PERFORMANCE BY METRIC:\n")
        f.write("-" * 30 + "\n")
        
        # Fastest execution
        fastest = results_df.loc[results_df['execution_time'].idxmin()]
        f.write(f"Fastest execution: {fastest['distribution']} with {fastest['threads']} threads "
                f"({fastest['execution_time']/60:.2f} minutes)\n")
        
        # Highest speedup
        highest_speedup = results_df.loc[results_df['real_speedup'].idxmax()]
        f.write(f"Highest speedup: {highest_speedup['distribution']} with {highest_speedup['threads']} threads "
                f"(Speedup: {highest_speedup['real_speedup']:.2f}x)\n")
        
        # Most efficient (ideal)
        most_efficient_ideal = results_df.loc[results_df['parallel_efficiency'].idxmax()]
        f.write(f"Most efficient (ideal): {most_efficient_ideal['distribution']} with {most_efficient_ideal['threads']} threads "
                f"(Efficiency: {most_efficient_ideal['parallel_efficiency']:.2%})\n")
        
        # Most efficient (Amdahl)
        most_efficient_amdahl = results_df.loc[results_df['amdahl_efficiency'].idxmax()]
        f.write(f"Most efficient (Amdahl): {most_efficient_amdahl['distribution']} with {most_efficient_amdahl['threads']} threads "
                f"(Efficiency: {most_efficient_amdahl['amdahl_efficiency']:.2%})\n\n")
        
        # Parallel fraction analysis
        f.write("PARALLEL FRACTION ANALYSIS:\n")
        f.write("-" * 25 + "\n")
        for dist in results_df['distribution'].unique():
            dist_data = results_df[results_df['distribution'] == dist]
            parallel_frac = dist_data['parallel_fraction'].iloc[0]
            f.write(f"{dist}: {parallel_frac:.4f} ({parallel_frac:.2%} parallelizable)\n")
        
        f.write(f"\nMaximum theoretical speedup (Amdahl's Law): {1/(1-parallel_frac):.2f}x\n")
        
        # Scalability insights
        f.write("\nSCALABILITY INSIGHTS:\n")
        f.write("-" * 20 + "\n")
        
        for dist in results_df['distribution'].unique():
            dist_data = results_df[results_df['distribution'] == dist].sort_values('threads')
            single_thread_time = dist_data[dist_data['threads'] == 1]['execution_time'].iloc[0]
            max_threads_time = dist_data['execution_time'].min()
            actual_speedup = single_thread_time / max_threads_time
            
            f.write(f"\n{dist}:\n")
            f.write(f"  - Single-thread time: {single_thread_time/60:.2f} min\n")
            f.write(f"  - Best multi-thread time: {max_threads_time/60:.2f} min\n")
            f.write(f"  - Actual speedup achieved: {actual_speedup:.2f}x\n")
            f.write(f"  - Efficiency at max threads: {dist_data['parallel_efficiency'].min():.2%}\n")
    
    print(f"Summary report saved to: {report_path}")

def main():
    """Main function to run the analysis"""
    parser = argparse.ArgumentParser(description='Generate parallel performance analysis graphs')
    parser.add_argument('--time_file', type=str, required=True, 
                       help='Path to execution times CSV file')
    parser.add_argument('--portion_file', type=str, required=True,
                       help='Path to parallel portion CSV file')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for saving graphs')
    parser.add_argument('--figure_width', type=float, default=10.0,
                       help='Base width for figures (default: 10.0)')
    parser.add_argument('--generate_report', action='store_true',
                       help='Generate a summary report')
    parser.add_argument('--suffix', type=str, default='',
                       help='Suffix to add to output filenames (e.g., _v1)')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Loading and processing data...")
    avg_times, portion_df = load_and_process_data(args.time_file, args.portion_file)
    
    print("Calculating performance metrics...")
    results_df = calculate_metrics(avg_times, portion_df)
    
    print("Generating graphs...")
    create_plots(results_df, args.output_dir, args.figure_width, args.suffix)
    
    if args.generate_report:
        print("Generating summary report...")
        generate_summary_report(results_df, args.output_dir, args.suffix)
    
    print(f"\nAnalysis complete! Graphs saved to: {os.path.abspath(args.output_dir)}")
    print("Generated graphs:")
    print("  - execution_time_vs_threads.png")
    print("  - real_speedup_vs_threads.png")
    print("  - parallel_efficiency_vs_threads.png")
    print("  - amdahl_efficiency_vs_threads.png")

if __name__ == "__main__":
    main()