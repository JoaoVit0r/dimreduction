#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import sys
import os
from datetime import datetime
import signal

# ignore SIGUSR1 signal
signal.signal(signal.SIGUSR1, signal.SIG_IGN)

CURRENT_YEAR = datetime.now().year
TIME_COLUMN = 'time'

# Suppress the figure limit warning
plt.rcParams['figure.max_open_warning'] = 0

def ensure_dir_exists(directory):
    """Create directory if it doesn't exist."""
    os.makedirs(directory, exist_ok=True)

def add_xaxis_padding(ax, padding_ratio=0.05):
    """Add padding to x-axis limits."""
    x_min, x_max = ax.get_xlim()
    x_range = x_max - x_min
    padding = x_range * padding_ratio
    ax.set_xlim(x_min - padding, x_max + padding)

def read_dstat_csv(file_path):
    print(f"\nReading data from: {file_path}")
    
    # First read the column names from the 6th row (0-based index 5)
    column_names = None
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if i == 5:  # This is the header row
                # Get column names and remove any empty ones
                column_names = [col.strip('"') for col in line.strip().split(',') if col.strip()]
                break
    
    print("\nColumn names:", column_names)
    
    # Read the raw data lines to handle the time values correctly
    data_lines = []
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if i > 5:  # Skip header rows
                # Split the line and remove empty values
                parts = [p.strip() for p in line.strip().split(',') if p.strip()]
                if len(parts) >= 2:
                    data_lines.append(parts)
    
    # Create DataFrame from the processed data
    df = pd.DataFrame(data_lines, columns=column_names)
    
    print("\nData after reading CSV:")
    print(df.head())
    print("\nColumns:", df.columns.tolist())
    
    # Rename columns to remove spaces and percentage signs, but preserve the time column
    new_columns = []
    for col in df.columns:
        if col.strip().lower() == 'time':
            new_columns.append('time')
        else:
            new_columns.append(col.strip().replace('%', 'pct').replace(' ', '_').lower())
    df.columns = new_columns
    
    print("\nData after column renaming:")
    print(df.head())
    print("\nColumns:", df.columns.tolist())
    
    # Convert numeric columns to float
    numeric_cols = df.columns.difference(['time'])
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    
    # Check if timestamp column exists (from dstat -t option)
    time_cols = [col for col in df.columns if col == 'time']
    if len(time_cols):
        time_col = time_cols[0]
        # Add current year to the date string before parsing
        df[time_col] = df[time_col].apply(lambda x: f"{x} {CURRENT_YEAR}")
        df[time_col] = pd.to_datetime(df[time_col], format='%b-%d %H:%M:%S %Y')
        df.set_index(time_col, inplace=True)
        
    df.index = mdates.date2num(df.index)
    
    print("\nFinal data:")
    print(df.head())
    print("\nColumns:", df.columns.tolist())
    
    return df

def read_markers_file(markers_file):
    """Read a file with timestamp markers and return as dictionary."""
    markers = {}
    if not os.path.exists(markers_file):
        return markers
        
    print(f"\nReading execution markers from: {markers_file}")
    
    with open(markers_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            # Split by space and get the parts
            parts = line.split()
            if len(parts) < 3:
                print(f"Warning: Could not parse line in markers file: {line}")
                continue
            
            # First two parts are date and time (e.g., 'Apr-02' and '21:21:31')
            date_part = parts[0]
            time_part = parts[1]
            label = ' '.join(parts[2:])
            
            # Combine date and time and add current year
            datetime_str = f"{date_part} {time_part} {CURRENT_YEAR}"
            try:
                timestamp = pd.to_datetime(datetime_str, format='%b-%d %H:%M:%S %Y')
                if label:
                    markers[timestamp] = label
                    print(f"  - Parsed marker: {timestamp} -> {label}")
            except ValueError as e:
                print(f"Warning: Could not parse timestamp '{datetime_str}' in markers file: {e}")
    
    print(f"Total markers parsed: {len(markers)}")
    return markers

def add_markers_to_plot(ax, markers, y_min, y_max):
    """Add vertical lines and annotations for markers with alternating positions."""
    sorted_markers = sorted(markers.items())
    
    # Calculate a small offset based on the x-axis range to prevent text from overlapping the line
    x_min, x_max = ax.get_xlim()
    x_offset = (x_max - x_min) * 0.005  # 0.5% of the total range

    for i, (timestamp, label) in enumerate(sorted_markers):
        # Use different colors for start and end markers
        if 'start' in label.lower():
            color = 'green'
            line_style = '--'
        elif 'end' in label.lower():
            color = 'red'
            line_style = '--'
        else:
            color = 'blue'
            line_style = '--'
        
        ax.axvline(x=timestamp, color=color, linestyle=line_style, alpha=0.7)
        
        # Alternate vertical position to avoid overlapping
        vertical_positions = [0.7,0.95]  # Different Y positions as fractions of y-range
        position_index = i % len(vertical_positions)
        y_pos = y_min + (y_max - y_min) * vertical_positions[position_index]
        
        # Shorten label for better readability
        short_label = label.replace('Execution_', 'Execution_').replace('_Start', '').replace('_End', '_(End)').replace('_', ' ')
        
        # Alternate horizontal position (left/right of the line)
        if i % 2 == 1:  # Even index, place text on the right
            x_pos = timestamp + x_offset
            ha = 'left'
        else:  # Odd index, place text on the left
            x_pos = timestamp - x_offset
            ha = 'right'

        ax.text(x_pos, y_pos, short_label, rotation=90, verticalalignment='top', horizontalalignment=ha,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.1))

def create_plots(df, output_prefix, output_folder, markers_file=None, split_by_time=None):
    # Ensure the output folder exists
    ensure_dir_exists(output_folder)
    
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
    
    print(f"DataFrame index type: {type(df.index)}")
    print(f"Number of markers found: {len(markers)}")
    
    # Split dataframe by execution markers if provided
    dfs_to_plot = []
    if markers and len(markers) >= 2:
        print("Splitting data by execution markers...")
        
        # Sort markers by timestamp
        sorted_markers = sorted(markers.items())
        print(f"Sorted markers: {[(ts, label) for ts, label in sorted_markers]}")
        
        # Create execution periods - pair start and end markers
        execution_count = 0
        i = 0
        while i < len(sorted_markers) - 1:
            start_time, start_label = sorted_markers[i]
            end_time, end_label = sorted_markers[i + 1]
            
            # Check if this is a valid start-end pair
            if "start" in start_label.lower() and "end" in end_label.lower():
                print(f"Creating execution period {execution_count + 1}: {start_time} to {end_time}")
                
                # Filter data for this execution period
                period_mask = (df.index >= mdates.date2num(start_time)) & (df.index <= mdates.date2num(end_time))
                period_df = df[period_mask].copy()
                
                if not period_df.empty:
                    # Convert to relative time (seconds from start) for individual execution plots
                    period_df_rel = period_df.copy()
                    start_timestamp = period_df_rel.index[0]
                    period_df_rel.index = (period_df_rel.index - start_timestamp) * 24 * 3600  # Convert from days to seconds
                    
                    period_name = f"Execution_{execution_count + 1}"
                    dfs_to_plot.append((period_name, period_df_rel, True))  # True indicates relative time
                    execution_count += 1
                    print(f"  - Found {len(period_df)} data points for {period_name}")
                else:
                    print(f"  - Warning: No data found between {start_time} and {end_time}")
                
                i += 2  # Move to next pair
            else:
                i += 1  # Skip this marker
        
        # Add the full data plot with all markers (use absolute time)
        if execution_count > 0:
            dfs_to_plot.insert(0, ('All Data', df, False))  # False indicates absolute time
        else:
            dfs_to_plot = [('All Data', df, False)]
            
    elif split_by_time and has_time_index:
        # Fallback to time-based splitting if no markers
        print("Falling back to time-based splitting...")
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
                dfs_to_plot.append((period_name, group, False))
    else:
        print("No markers found, using full data only")
        dfs_to_plot = [('All Data', df, False)]
    
    print(f"Plots to generate: {[name for name, _, _ in dfs_to_plot]}")
    
    print("\nGenerating plots...")
    for period_name, period_df, use_relative_time in dfs_to_plot:
        print(f"\nProcessing period: {period_name}")
        suffix = f"_{period_name.lower().replace(' ', '_').replace(':', '').replace('-', '')}" if period_name != "All Data" else ""
        
        # CPU Usage Plot
        print("  - CPU Usage plot")
        fig = plt.figure(figsize=(12, 6))
        # Remove 'idl' (idle) and keep usr, sys, wai (wait I/O)
        cpu_cols = [col for col in period_df.columns if col.startswith('usr') or col.startswith('sys') or col.startswith('wai')]
        
        ax = period_df[cpu_cols].plot()
        ax.set_title(f'CPU Usage Over Time - {period_name}', fontsize=14, fontweight='bold')
        
        # Rename legend for better clarity
        if 'usr' in cpu_cols and 'sys' in cpu_cols and 'wai' in cpu_cols:
            ax.legend(['User CPU', 'System CPU', 'I/O Wait'], loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=3)
        
        if use_relative_time:
            if len(period_df) > 0:
                max_time = period_df.index.max()
                if max_time < 10 * 60:  # If duration is less than 3 minutes, use seconds
                    ax.set_xlabel('Time (seconds)', fontsize=12)
                    ax.set_xlim(0, max_time)
                else:  # If duration is 3 minutes or more, use minutes
                    def minutes_formatter(value, tick_number):
                        minutes = value / 60
                        return f'{int(round(minutes))}'
                    ax.xaxis.set_major_formatter(plt.FuncFormatter(minutes_formatter))
                    ax.set_xlabel('Time (minutes)', fontsize=12)
                    ax.set_xlim(0, max_time)
        else:
            ax.set_xlabel('Time', fontsize=12)
            if len(period_df) > 0:
                time_nums = period_df.index
                if len(time_nums) > 0:
                    dates = mdates.num2date(time_nums)
                    ax.set_xlim(time_nums[0], time_nums[-1])
                    time_range = (dates[-1] - dates[0]).total_seconds()
                    if time_range < 3600:
                        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
                    elif time_range < 86400:
                        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                    else:
                        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
                    plt.xticks(rotation=45)
            
        ax.set_ylabel('CPU Usage (%)', fontsize=12)
        plt.grid(True)
        
        add_xaxis_padding(ax)
        
        if markers and period_name == "All Data" and not use_relative_time:
            y_min, y_max = ax.get_ylim()
            numeric_markers = {mdates.date2num(ts): label for ts, label in markers.items()}
            add_markers_to_plot(ax, numeric_markers, y_min, y_max)
        
        if period_name == "All Data" and not use_relative_time and not period_df.empty:
            start_time_num = period_df.index[0]
            def minutes_formatter(value, tick_number):
                minutes = ((value - start_time_num) * 24 * 3600) / 60
                return f'{int(round(minutes))}'
            ax.xaxis.set_major_formatter(plt.FuncFormatter(minutes_formatter))
            ax.set_xlabel('Time (minutes)', fontsize=12)
            plt.setp(ax.get_xticklabels(), rotation=0, ha='center')
            
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f'{output_prefix}_cpu{suffix}_{timestamp}.svg'))
        plt.close(fig)
        
        # Memory Usage Plot
        print("  - Memory Usage plot")
        fig = plt.figure(figsize=(12, 6))
        mem_cols = ['used', 'cach']
        
        if all(col in period_df.columns for col in mem_cols):
            total_memory = period_df[['used', 'free', 'cach', 'avai']].sum(axis=1)
            mem_percentages = period_df[mem_cols].div(total_memory, axis=0) * 100
            ax = mem_percentages.plot()
            ax.set_title(f'Memory Usage Over Time - {period_name}', fontsize=14, fontweight='bold')
            ax.legend(['Used Memory', 'Cache Memory'], loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=3)
        else:
            print("  - Warning: Memory columns not found, skipping memory plot")
            continue
        
        if use_relative_time:
            if len(period_df) > 0:
                max_time = period_df.index.max()
                if max_time < 10 * 60:  # If duration is less than 3 minutes, use seconds
                    ax.set_xlabel('Time (seconds)', fontsize=12)
                    ax.set_xlim(0, max_time)
                else:  # If duration is 3 minutes or more, use minutes
                    def minutes_formatter(value, tick_number):
                        minutes = value / 60
                        return f'{int(round(minutes))}'
                    ax.xaxis.set_major_formatter(plt.FuncFormatter(minutes_formatter))
                    ax.set_xlabel('Time (minutes)', fontsize=12)
                    ax.set_xlim(0, max_time)
        else:
            ax.set_xlabel('Time', fontsize=12)
            if len(period_df) > 0:
                time_nums = period_df.index
                if len(time_nums) > 0:
                    dates = mdates.num2date(time_nums)
                    ax.set_xlim(time_nums[0], time_nums[-1])
                    time_range = (dates[-1] - dates[0]).total_seconds()
                    if time_range < 3600:
                        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
                    elif time_range < 86400:
                        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                    else:
                        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
                    plt.xticks(rotation=45)
            
        ax.set_ylabel('Memory Usage (%)', fontsize=12)
        plt.grid(True)
        
        add_xaxis_padding(ax)
        
        if markers and period_name == "All Data" and not use_relative_time:
            y_min, y_max = ax.get_ylim()
            numeric_markers = {mdates.date2num(ts): label for ts, label in markers.items()}
            add_markers_to_plot(ax, numeric_markers, y_min, y_max)
        
        if period_name == "All Data" and not use_relative_time and not period_df.empty:
            start_time_num = period_df.index[0]
            def minutes_formatter(value, tick_number):
                minutes = ((value - start_time_num) * 24 * 3600) / 60
                return f'{int(round(minutes))}'
            ax.xaxis.set_major_formatter(plt.FuncFormatter(minutes_formatter))
            ax.set_xlabel('Time (minutes)', fontsize=12)
            plt.setp(ax.get_xticklabels(), rotation=0, ha='center')
            
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f'{output_prefix}_memory{suffix}_{timestamp}.svg'))
        plt.close(fig)
        
        # Load Average Plot
        print("  - Load Average plot")
        fig = plt.figure(figsize=(12, 6))
        load_cols = [col for col in period_df.columns if col in ['1m', '5m', '15m']]
        
        if any(col in period_df.columns for col in load_cols):
            ax = period_df[load_cols].plot()
            ax.set_title(f'System Load Average Over Time - {period_name}', fontsize=14, fontweight='bold')
            ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=3)
        else:
            print("  - Warning: Load average columns not found, skipping load plot")
            continue
        
        if use_relative_time:
            if len(period_df) > 0:
                max_time = period_df.index.max()
                if max_time < 10 * 60:  # If duration is less than 3 minutes, use seconds
                    ax.set_xlabel('Time (seconds)', fontsize=12)
                    ax.set_xlim(0, max_time)
                else:  # If duration is 3 minutes or more, use minutes
                    def minutes_formatter(value, tick_number):
                        minutes = value / 60
                        return f'{int(round(minutes))}'
                    ax.xaxis.set_major_formatter(plt.FuncFormatter(minutes_formatter))
                    ax.set_xlabel('Time (minutes)', fontsize=12)
                    ax.set_xlim(0, max_time)
        else:
            ax.set_xlabel('Time', fontsize=12)
            if len(period_df) > 0:
                time_nums = period_df.index
                if len(time_nums) > 0:
                    dates = mdates.num2date(time_nums)
                    ax.set_xlim(time_nums[0], time_nums[-1])
                    time_range = (dates[-1] - dates[0]).total_seconds()
                    if time_range < 3600:
                        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
                    elif time_range < 86400:
                        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                    else:
                        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
                    plt.xticks(rotation=45)
            
        ax.set_ylabel('Load Average', fontsize=12)
        plt.grid(True)
        
        add_xaxis_padding(ax)
        
        if markers and period_name == "All Data" and not use_relative_time:
            y_min, y_max = ax.get_ylim()
            numeric_markers = {mdates.date2num(ts): label for ts, label in markers.items()}
            add_markers_to_plot(ax, numeric_markers, y_min, y_max)
            
        if period_name == "All Data" and not use_relative_time and not period_df.empty:
            start_time_num = period_df.index[0]
            def minutes_formatter(value, tick_number):
                minutes = ((value - start_time_num) * 24 * 3600) / 60
                return f'{int(round(minutes))}'
            ax.xaxis.set_major_formatter(plt.FuncFormatter(minutes_formatter))
            ax.set_xlabel('Time (minutes)', fontsize=12)
            plt.setp(ax.get_xticklabels(), rotation=0, ha='center')

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
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

