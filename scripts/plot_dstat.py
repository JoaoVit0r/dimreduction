#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import sys
import os
from datetime import datetime

CURRENT_YEAR = datetime.now().year
TIME_COLUMN = 'time'

# Suppress the figure limit warning
plt.rcParams['figure.max_open_warning'] = 0

def ensure_dir_exists(directory):
    """Create directory if it doesn't exist."""
    os.makedirs(directory, exist_ok=True)

def read_dstat_csv(file_path):
    print(f"\nReading data from: {file_path}")
    # Skip the first rows that contain header information
    df = pd.read_csv(file_path, skiprows=5)
    
    # Rename columns to remove spaces and percentage signs
    df.columns = [col.strip().replace('%', 'pct').replace(' ', '_').lower() for col in df.columns]
    
    # Check if timestamp column exists (from dstat -t option)
    time_cols = [col for col in df.columns if TIME_COLUMN in col]
    if len(time_cols):
        time_col = time_cols[0]
        df[time_col] = pd.to_datetime(df[TIME_COLUMN], format='%b-%d %H:%M:%S')
        df.set_index(time_col, inplace=True)
        
    df.index = mdates.date2num(df.index)
    
    return df

def read_markers_file(markers_file):
    """Read a file with timestamp markers and return as dictionary."""
    markers = {}
    if not os.path.exists(markers_file):
        return markers
        
    print(f"\nReading execution markers from: {markers_file}")
    with open(markers_file, 'r') as f:
        for line in f:
            lineInfos = list(filter(lambda x: x != '',line.strip().split(' ')))
            if len(lineInfos) < 3:
                print(f"Warning: Could not parse timestamp in markers file: {line.strip()}")
                continue
            
            lineDateTime = ' '.join(lineInfos[:2])
            lineLabel = ' '.join(lineInfos[2:])
            try:
                timestamp = pd.to_datetime(lineDateTime, format='%b-%d %H:%M:%S')
                if lineLabel:
                    markers[mdates.date2num(timestamp)] = lineLabel
            except ValueError as e:
                print(f"Warning: Could not parse timestamp in markers file: {line.strip()}")
    return markers

def add_markers_to_plot(ax, markers, y_min, y_max):
    """Add vertical lines and annotations for markers."""
    for timestamp, label in markers.items():
        # Convert timestamp to date2num if it's not already
        if not isinstance(timestamp, float):
            timestamp = mdates.date2num(timestamp)
        ax.axvline(x=timestamp, color='r', linestyle='--', alpha=0.7)
        ax.text(timestamp, y_max * 0.95, label, rotation=90, verticalalignment='top')

def create_plots(df, output_prefix, output_folder, markers_file=None, split_by_time=None):
    # Ensure the output folder exists
    ensure_dir_exists(output_folder)
    ensure_dir_exists(os.path.join(output_folder, 'output'))
    
    # Set style to a default one
    plt.style.use('default')
    
    # Create timestamp for unique filenames
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Load markers if provided
    markers = {}
    if markers_file:
        markers = read_markers_file(markers_file)
    
    # Check if we're using a datetime index from dstat -T
    has_time_index = isinstance(df.index, pd.DatetimeIndex)
    
    # Split dataframe by execution markers if provided
    dfs_to_plot = []
    if markers:
        # Sort markers by timestamp
        sorted_markers = sorted(markers.items())
        
        # Create execution periods
        for i in range(0, len(sorted_markers)-1, 2):
            if i+1 >= len(sorted_markers):
                break
                
            start_time, start_label = sorted_markers[i]
            end_time, end_label = sorted_markers[i+1]
            
            # Filter data for this execution period
            period_df = df[(df.index >= start_time) & (df.index <= end_time)]
            if not period_df.empty:
                period_name = f"Execution_{i//2 + 1}"
                dfs_to_plot.append((period_name, period_df))
        
        # Add the full data plot with all markers
        dfs_to_plot.insert(0, ('All Data', df))

    elif split_by_time and has_time_index:
        # Fallback to time-based splitting if no markers
        if split_by_time == 'hour':
            grouped = df.groupby(pd.Grouper(freq='H'))
        elif split_by_time == 'day':
            grouped = df.groupby(pd.Grouper(freq='D'))
        else:
            grouped = df.groupby(pd.Grouper(freq='H'))
        
        for period_start, group in grouped:
            if not group.empty:
                period_end = period_start + pd.Timedelta('1 hour') if split_by_time == 'hour' else period_start + pd.Timedelta('1 day')
                period_name = f"{period_start.strftime('%Y-%m-%d %H:%M')} to {period_end.strftime('%H:%M')}"
                dfs_to_plot.append((period_name, group))
    else:
        dfs_to_plot = [('All Data', df)]
    
    print("\nGenerating plots...")
    for period_name, period_df in dfs_to_plot:
        print(f"\nProcessing period: {period_name}")
        suffix = f"_{period_name.lower().replace(' ', '_').replace(':', '').replace('-', '')}" if period_name != "All Data" else ""
        
        # CPU Usage Plot
        print("  - CPU Usage plot")
        fig = plt.figure(figsize=(12, 6))
        cpu_cols = [col for col in period_df.columns if col.startswith('usr') or col.startswith('sys') or col.startswith('idl')]
        ax = period_df[cpu_cols].plot(title=f'CPU Usage Over Time - {period_name}')
        
        if has_time_index:
            plt.xlabel('Time')
            # Format x-axis with appropriate date format
            if len(period_df) > 0:
                time_range = (period_df.index.max() - period_df.index.min()).total_seconds()
                if time_range < 3600:  # Less than an hour
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
                elif time_range < 86400:  # Less than a day
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                else:  # More than a day
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
            plt.xticks(rotation=45)
        else:
            plt.xlabel('Time (seconds)')
            
        plt.ylabel('CPU Usage (%)')
        plt.grid(True)
        
        if markers and period_name == "All Data":
            y_min, y_max = ax.get_ylim()
            add_markers_to_plot(ax, markers, y_min, y_max)
            
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f'{output_prefix}_cpu{suffix}_{timestamp}.svg'))
        plt.close(fig)
        
        # Memory Usage Plot
        print("  - Memory Usage plot")
        fig = plt.figure(figsize=(12, 6))
        mem_cols = ['used', 'free', 'cach', 'avai']
        
        # Calculate total memory and percentages
        total_memory = period_df[mem_cols].sum(axis=1)
        mem_percentages = period_df[mem_cols].div(total_memory, axis=0) * 100
        
        # Plot memory percentages
        ax = mem_percentages.plot(title=f'Memory Usage Over Time - {period_name}')
        
        if has_time_index:
            plt.xlabel('Time')
            if len(period_df) > 0:
                time_range = (period_df.index.max() - period_df.index.min()).total_seconds()
                if time_range < 3600:
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
                elif time_range < 86400:
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                else:
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
            plt.xticks(rotation=45)
        else:
            plt.xlabel('Time (seconds)')
            
        plt.ylabel('Memory Usage (%)')
        plt.grid(True)
        
        if markers and period_name == "All Data":
            y_min, y_max = ax.get_ylim()
            add_markers_to_plot(ax, markers, y_min, y_max)
            
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f'{output_prefix}_memory{suffix}_{timestamp}.svg'))
        plt.close(fig)
        
        # Load Average Plot
        print("  - Load Average plot")
        fig = plt.figure(figsize=(12, 6))
        load_cols = [col for col in period_df.columns if col.startswith('1m') or col.startswith('5m') or col.startswith('15m')]
        ax = period_df[load_cols].plot(title=f'System Load Average Over Time - {period_name}')
        
        if has_time_index:
            plt.xlabel('Time')
            if len(period_df) > 0:
                time_range = (period_df.index.max() - period_df.index.min()).total_seconds()
                if time_range < 3600:
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
                elif time_range < 86400:
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                else:
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
            plt.xticks(rotation=45)
        else:
            plt.xlabel('Time (seconds)')
            
        plt.ylabel('Load Average')
        plt.grid(True)
        
        if markers and period_name == "All Data":
            y_min, y_max = ax.get_ylim()
            add_markers_to_plot(ax, markers, y_min, y_max)
            
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f'{output_prefix}_load{suffix}_{timestamp}.svg'))
        plt.close(fig)

def main():
    if len(sys.argv) < 3:
        print("Usage: python plot_dstat.py <dstat_csv_file> <output_folder> [markers_file] [split_by:hour|day]")
        sys.exit(1)
        
    csv_file = sys.argv[1]
    output_folder = sys.argv[2]
    output_prefix = 'dstat_plot'
    
    # Optional parameters
    markers_file = sys.argv[3] if len(sys.argv) > 3 else None
    split_by_time = None
    if len(sys.argv) > 4 and sys.argv[4].startswith('split_by:'):
        split_by_time = sys.argv[4].split(':')[1]
    
    try:
        print("\n===============================================")
        print("Starting Plot Generation")
        print("===============================================")
        print(f"Input file: {csv_file}")
        print(f"Output directory: {output_folder}")
        if markers_file:
            print(f"Markers file: {markers_file}")
        if split_by_time:
            print(f"Split by: {split_by_time}")
        print("===============================================")
        
        df = read_dstat_csv(csv_file)
        create_plots(df, output_prefix, output_folder, markers_file, split_by_time)
        
        print("\n===============================================")
        print("Plot Generation Completed Successfully")
        print("===============================================")
        print(f"Output directory: {output_folder}")
    except Exception as e:
        print(f"\nError processing the CSV file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()