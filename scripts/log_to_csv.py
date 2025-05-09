import re
import csv
from datetime import datetime

def log_to_csv(log_file_path, output_csv_path):
    # Regular expression to extract information from log lines
    pattern = r'monitoring_plots/\d+_\d+/(?P<environment>venv_\w+)/(?P<script>[^/]+)/distribution_(?P<distribution_type>[^/]+)/threads_(?P<thread_count>\d+)/.*\.txt:(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) Execution_(?P<execution_num>\d+)_(?P<event_type>Start|End) \w+_\d+( \(Duration: (?P<duration>\d+m \d+s)\))?'
    
    # Map environment names to program names as in the CSV
    env_to_program = {
        'venv_v12': 'python_v12',
        'venv_v13': 'python_v13',
        'venv_v13t': 'python_v13_no_gil'  # Based on timing patterns
    }
    
    # Storage for paired Start/End events
    executions = {}
    
    # Read the log file
    with open(log_file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:  # Skip empty lines
                continue
            
            match = re.search(pattern, line)
            if match:
                environment = match.group('environment')
                script = match.group('script')
                distribution_type = match.group('distribution_type')
                thread_count = match.group('thread_count')
                timestamp = match.group('timestamp')
                execution_num = match.group('execution_num')
                event_type = match.group('event_type')
                duration_str = match.group('duration')
                
                # Create a unique key for this execution
                key = f"{environment}_{script}_{distribution_type}_{thread_count}_{execution_num}"
                
                if key not in executions:
                    executions[key] = {
                        'program': env_to_program.get(environment, environment),
                        'script': script,
                        'num_threads_total': thread_count,
                        'num_features': '400',  # Fixed based on original CSV
                        'distribution_type': distribution_type
                    }
                
                if event_type == 'Start':
                    executions[key]['start'] = timestamp
                else:  # End event
                    executions[key]['end'] = timestamp
                    if duration_str:
                        # Convert "XYm ZZs" format to H:MM:SS
                        duration_parts = re.match(r'(\d+)m (\d+)s', duration_str)
                        if duration_parts:
                            minutes = int(duration_parts.group(1))
                            seconds = int(duration_parts.group(2))
                            total_seconds = minutes * 60 + seconds
                            hours = total_seconds // 3600
                            minutes = (total_seconds % 3600) // 60
                            seconds = total_seconds % 60
                            executions[key]['duration'] = f"{hours}:{minutes:02d}:{seconds:02d}"
    
    # Write to CSV
    with open(output_csv_path, 'w', newline='') as csvfile:
        fieldnames = ['program', 'script', 'num_threads_total', 'num_features', 'distribution_type', 'duration', 'start', 'end']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=';')
        
        writer.writeheader()
        for execution in executions.values():
            # Only write complete entries
            if 'start' in execution and 'end' in execution and 'duration' in execution:
                writer.writerow(execution)
    
    print(f"CSV file created: {output_csv_path}")

# Usage
log_to_csv('Untitled-1.log', 'execution_results.csv')