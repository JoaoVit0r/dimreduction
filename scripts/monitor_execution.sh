#!/bin/bash

SLEEP_TIME=5

# Parse arguments
OUTPUT_DIR=""
REPOSITORY_PYTHON=""
COMMAND=""
NUMBER_OF_EXECUTIONS=3
CUSTOM_INPUT_FILE_PATH=""
THREADS=1  # Default to 1 thread
THREAD_DISTRIBUTION="spaced"  # Default to spaced distribution
SKIP_MONITORING=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --repository-python)
            REPOSITORY_PYTHON="$2"
            shift 2
            ;;
        --sleep-time)
            SLEEP_TIME="$2"
            shift 2
            ;;
        --number-of-executions)
            NUMBER_OF_EXECUTIONS="$2"
            shift 2
            ;;
        --custom-input-file)
            CUSTOM_INPUT_FILE_PATH="$2"
            shift 2
            ;;
        --threads)
            THREADS="$2"
            shift 2
            ;;
        --thread-distribution)
            THREAD_DISTRIBUTION="$2"
            shift 2
            ;;
        --skip-monitoring)
            SKIP_MONITORING=true
            shift
            ;;
        --help)
            cat << 'EOF'
Usage: ./monitor_execution.sh [OPTIONS] COMMAND

Monitor system performance metrics while executing a command.

Required Arguments:
  COMMAND                           The command to execute and monitor

Required Options:
  --output-dir <directory>          Directory to store monitoring results
  --repository-python <path>        Path to Python repository for scripts

Optional Settings:
  --sleep-time <seconds>            Time between executions (default: 5)
  --number-of-executions <number>   Times to run the command (default: 3)
  --custom-input-file <path>        Path to custom input dataset
  --skip-monitoring                 Skip monitoring and just execute the command
  --threads <count>                 Number of threads to use (default: 1)
  --thread-distribution <types>     Thread distribution types: values like 
                                    "spaced,sequential" (default: spaced)

Output Files Generated:
  - monitoring_plots/               Directory containing all monitoring data
    ├── dstat_output_*.csv          Raw monitoring data
    ├── execution_markers_*.txt     Execution timestamps
    ├── env_variables_*.txt         Environment variables snapshot
    ├── plots/                      Generated performance plots
    ├── logs.log                    Execution logs
    ├── timers.log                  Timing information
    └── thread_execution.log        Thread execution details

Examples:
  # Basic monitoring:
  ./monitor_execution.sh --output-dir ./output --repository-python . "python script.py"

  # Custom executions and timing:
  ./monitor_execution.sh --output-dir ./output --repository-python . \
    --sleep-time 10 --number-of-executions 5 "python script.py"

  # With custom input file and threads:
  ./monitor_execution.sh --output-dir ./output --repository-python . \
    --custom-input-file data.csv --threads 4 "python script.py"
    
  # With multiple thread distribution types:
  ./monitor_execution.sh --output-dir ./output --repository-python . \
    --thread-distribution spaced "python script.py"
EOF
            exit 0
            ;;
        *)
            if [ -z "$COMMAND" ]; then
                COMMAND="$@"
                break
            fi
            ;;
    esac
done

# Check if required commands are available
for cmd in dool python3; do
    if [ "$SKIP_MONITORING" = true ]; then
        break
    fi
    if ! command -v $cmd &> /dev/null; then
        echo "Error: $cmd is required but not installed."
        exit 1
    fi
done

if [ "$SKIP_MONITORING" = false ]; then
    # Check if required Python packages are available
    python3 -c "import pandas, matplotlib" # 2>/dev/null
    if [ $? -ne 0 ]; then
        echo "Error: Required Python packages (pandas, matplotlib) are not installed."
        echo "Please install them using: pip install pandas matplotlib"
        exit 1
    fi
fi

# Check if output directory and command were provided
if [ -z "$OUTPUT_DIR" ] || [ -z "$COMMAND" ]; then
    echo "Usage: $0 --output-dir <output_directory> --repository-python <python_repository> --sleep-time <sleep_time> <command_to_execute>"
    exit 1
fi

# Create output directory for plots
MONITOR_DIR="$OUTPUT_DIR/monitoring_plots"
mkdir -p "$MONITOR_DIR"

# Create a unique temporary file for dstat output
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DSTAT_OUTPUT_File="$MONITOR_DIR/dstat_output_$TIMESTAMP.csv"
DSTAT_MARKERS_FILE="$MONITOR_DIR/execution_markers_$TIMESTAMP.txt"
ENV_FILE="$MONITOR_DIR/env_variables_$TIMESTAMP.txt"

# Create a directory for plots
PLOTS_DIR="$MONITOR_DIR/plots"
mkdir -p "$PLOTS_DIR"

if [ -L .env ]; then
    DEFAULT_ENV_FILE=$(realpath .env)
else
    DEFAULT_ENV_FILE=".env"
fi

# Always backup the environment file before making any changes
echo "-----------------------------------------------"
echo "Backing up Environment Variables"
echo "-----------------------------------------------"
cp "$DEFAULT_ENV_FILE" "$DEFAULT_ENV_FILE.bak"

# Make all necessary changes to the environment file
echo "-----------------------------------------------"
echo "Updating Environment Variables"
echo "-----------------------------------------------"

# Update custom input file path if specified
if [ -n "$CUSTOM_INPUT_FILE_PATH" ]; then
    if [ ! -f "$CUSTOM_INPUT_FILE_PATH" ]; then
        echo "Error: Custom input file does not exist."
        rm "$DEFAULT_ENV_FILE.bak"
        exit 1
    fi
    
    echo "Setting INPUT_FILE_PATH=$(realpath "$CUSTOM_INPUT_FILE_PATH")"
    sed -i -E "s|^INPUT_FILE_PATH=.*|INPUT_FILE_PATH=$(realpath "$CUSTOM_INPUT_FILE_PATH" | sed 's/\//\\\//g')|" "$DEFAULT_ENV_FILE"
fi


# Update the NUMBER_OF_THREADS environment variable
echo "Setting NUMBER_OF_THREADS=$THREADS"
if grep -q "^NUMBER_OF_THREADS=" "$DEFAULT_ENV_FILE"; then
    sed -i -E "s/^NUMBER_OF_THREADS=.*/NUMBER_OF_THREADS=$THREADS/" "$DEFAULT_ENV_FILE"
else
    echo "" >> "$DEFAULT_ENV_FILE"
    echo "NUMBER_OF_THREADS=$THREADS" >> "$DEFAULT_ENV_FILE"
fi

# Update the THREAD_DISTRIBUTION environment variable
echo "Setting THREAD_DISTRIBUTION=$THREAD_DISTRIBUTION"
if grep -q "^THREAD_DISTRIBUTION=" "$DEFAULT_ENV_FILE"; then
    sed -i -E "s/^THREAD_DISTRIBUTION=.*/THREAD_DISTRIBUTION=$THREAD_DISTRIBUTION/" "$DEFAULT_ENV_FILE"
else
    echo "" >> "$DEFAULT_ENV_FILE"
    echo "THREAD_DISTRIBUTION=$THREAD_DISTRIBUTION" >> "$DEFAULT_ENV_FILE"
fi

echo "-----------------------------------------------"
echo "Saving Environment Variables"
echo "-----------------------------------------------"
cp "$DEFAULT_ENV_FILE" "$ENV_FILE"

echo "==============================================="
echo "Starting System Monitoring"
echo "==============================================="
echo "Command to monitor: $COMMAND"
echo "Output directory: $OUTPUT_DIR"
echo "Thread count: $THREADS"
echo "==============================================="

if [ "$SKIP_MONITORING" = false ]; then
    # Start dstat(dool) in the background (time, cpu, load, mem)
    dool -tclm --output "$DSTAT_OUTPUT_File" 1 &
    DSTAT_PID=$!
fi

# Wait for dstat to initialize
sleep 2

echo -e "\nStarting command execution monitoring..."
echo "Will execute: $COMMAND"

# Execute the command 3 times with intervals
for ((i=1; i <= "$NUMBER_OF_EXECUTIONS"; i++)); do
    echo -e "\n-----------------------------------------------"
    echo "Execution $i of $NUMBER_OF_EXECUTIONS (Threads: $THREADS)"
    echo "-----------------------------------------------"
    
    # Record start time in seconds since epoch
    start_time=$(date +%s)
    start_datetime=$(date +'%Y-%m-%d %H:%M:%S')
    echo "$start_datetime Execution_${i}_Start Threads_${THREADS}" >> "$DSTAT_MARKERS_FILE"
    
    # Export NUMBER_OF_THREADS directly for this execution
    export NUMBER_OF_THREADS=$THREADS
    
    eval "$COMMAND"
    
    # Record end time and calculate duration
    end_time=$(date +%s)
    end_datetime=$(date +'%Y-%m-%d %H:%M:%S')
    duration=$((end_time - start_time))
    
    # Format duration into hours, minutes and seconds
    hours=$((duration / 3600))
    minutes=$(( (duration % 3600) / 60 ))
    seconds=$((duration % 60))
    
    # Create duration string
    duration_str=""
    if [ $hours -gt 0 ]; then
        duration_str="${hours}h "
    fi
    if [ $minutes -gt 0 ] || [ $hours -gt 0 ]; then
        duration_str="${duration_str}${minutes}m "
    fi
    duration_str="${duration_str}${seconds}s"
    
    echo "$end_datetime Execution_${i}_End Threads_${THREADS} (Duration: $duration_str)" >> "$DSTAT_MARKERS_FILE"
    
    # Wait for SLEEP_TIME between executions if not the last one
    if [ $i -lt "$NUMBER_OF_EXECUTIONS" ]; then
        echo -e "\nWaiting between executions..."
        if [ $SLEEP_TIME -ge 60 ]; then
            echo "Next execution in $(($SLEEP_TIME / 60)) minutes"
        else
            echo "Next execution in $SLEEP_TIME seconds"
        fi
        sleep "$SLEEP_TIME"
    fi
done

# Continue monitoring for $SLEEP_TIME more seconds after last execution
echo -e "\n-----------------------------------------------"
echo "All executions completed"
echo "-----------------------------------------------"
if [ $SLEEP_TIME -ge 60 ]; then
    echo "Monitoring for $(($SLEEP_TIME / 60)) minutes..."
else
    echo "Monitoring for $SLEEP_TIME seconds..."
fi
sleep "$SLEEP_TIME"

if [ "$SKIP_MONITORING" = false ]; then
    # Gracefully stop dstat
    kill $DSTAT_PID
fi

echo -e "\n==============================================="
echo "Generating Performance Plots"
echo "==============================================="
echo "Input data: $DSTAT_OUTPUT_File"
echo "Output directory: $PLOTS_DIR"

if [ "$SKIP_MONITORING" = false ]; then
    # Generate plots using the Python script
    python3 "$REPOSITORY_PYTHON"/scripts/plot_dstat.py "$DSTAT_OUTPUT_File" "$PLOTS_DIR" "$DSTAT_MARKERS_FILE" "split_by:hour"
fi

# Restore the original environment file
echo "-----------------------------------------------"
echo "Restoring Original Environment Variables"
echo "-----------------------------------------------"
if [ -f "$DEFAULT_ENV_FILE.bak" ]; then
    mv "$DEFAULT_ENV_FILE.bak" "$DEFAULT_ENV_FILE"
else
    echo "Warning: Backup environment file not found!"
fi

# Move monitoring files to output directory using move_monitoring_files.sh
echo -e "\n-----------------------------------------------"
echo "Moving monitoring files"
echo "-----------------------------------------------"
"$REPOSITORY_PYTHON"/scripts/move_monitoring_files.sh "$MONITOR_DIR"

echo -e "\n==============================================="
echo "Monitoring Completed Successfully"
echo "==============================================="

if [ "$SKIP_MONITORING" = false ]; then
    # Generate summary files
    echo -e "\nGenerating summary files..."
    python3 "$REPOSITORY_PYTHON"/scripts/generate_summary.py "$OUTPUT_DIR"

    echo -e "\n==============================================="
    echo "Summary Generation Completed"
    echo "==============================================="
fi