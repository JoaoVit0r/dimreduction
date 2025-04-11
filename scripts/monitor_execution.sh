#!/bin/bash

SLEEP_TIME=5

# Parse arguments
OUTPUT_DIR=""
REPOSITORY_PYTHON=""
COMMAND=""
NUMBER_OF_EXECUTIONS=3
CUSTOM_INPUT_FILE_PATH=""

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
        --help)
            echo "Usage: $0 --output-dir <output_directory> --repository-python <python_repository> --sleep-time <sleep_time> [--number-of-executions <number>] <command_to_execute>"
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
    if ! command -v $cmd &> /dev/null; then
        echo "Error: $cmd is required but not installed."
        exit 1
    fi
done

# Check if required Python packages are available
python3 -c "import pandas, matplotlib" # 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Error: Required Python packages (pandas, matplotlib) are not installed."
    echo "Please install them using: pip install pandas matplotlib"
    exit 1
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

if [ -n "$CUSTOM_INPUT_FILE_PATH" ]; then
    if [ ! -f "$CUSTOM_INPUT_FILE_PATH" ]; then
        echo "Error: Custom input file does not exist."
        exit 1
    fi
    # Save the DEFAULT_ENV_FILE to restore it later
    cp "$DEFAULT_ENV_FILE" "$DEFAULT_ENV_FILE.bak"

    echo "-----------------------------------------------"
    echo "Updating Environment Variables"
    echo "-----------------------------------------------"
    sed -i -E "s/^INPUT_FILE_PATH=.*/INPUT_FILE_PATH=$(realpath "$CUSTOM_INPUT_FILE_PATH" | sed 's/\//\\\//g')/" "$DEFAULT_ENV_FILE"
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
echo "==============================================="

# Start dstat(dool) in the background (time, cpu, load, mem)
dool -tclm --output "$DSTAT_OUTPUT_File" 1 &
DSTAT_PID=$!

# Wait for dstat to initialize
sleep 2

echo -e "\nStarting command execution monitoring..."
echo "Will execute: $COMMAND"

# Execute the command 3 times with intervals
for ((i=1; i <= "$NUMBER_OF_EXECUTIONS"; i++)); do
    echo -e "\n-----------------------------------------------"
    echo "Execution $i of $NUMBER_OF_EXECUTIONS"
    echo "-----------------------------------------------"
    echo "$(date +'%Y-%m-%d %H:%M:%S') Execution_${i}_Start" >> "$DSTAT_MARKERS_FILE"
    eval "$COMMAND"
    echo "$(date +'%Y-%m-%d %H:%M:%S') Execution_${i}_End" >> "$DSTAT_MARKERS_FILE"
    
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

# Gracefully stop dstat
kill $DSTAT_PID

echo -e "\n==============================================="
echo "Generating Performance Plots"
echo "==============================================="
echo "Input data: $DSTAT_OUTPUT_File"
echo "Output directory: $PLOTS_DIR"

# Generate plots using the Python script
python3 "$REPOSITORY_PYTHON"/scripts/plot_dstat.py "$DSTAT_OUTPUT_File" "$PLOTS_DIR" "$DSTAT_MARKERS_FILE" "split_by:hour"

# Restore the DEFAULT_ENV_FILE
if [ -f "$DEFAULT_ENV_FILE.bak" ]; then
    echo "-----------------------------------------------"
    echo "Restoring Environment Variables"
    echo "-----------------------------------------------"
    mv "$DEFAULT_ENV_FILE.bak" "$DEFAULT_ENV_FILE"
fi

# Move command results to the output directory
if [ -f "logs/logs.log" ]; then
    mv "logs/logs.log" "$MONITOR_DIR"
fi
if [ -f "timing/timers.log" ]; then
    mv "timing/timers.log" "$MONITOR_DIR"
fi
if [ -f "timing/thread_execution.log" ]; then
    mv "timing/thread_execution.log" "$MONITOR_DIR"
fi
if [ -d "results" ]; then
    mv results/*/*.txt "$MONITOR_DIR"
fi
if [ -f "perf.data" ]; then
    mv "perf.data" "$MONITOR_DIR"
fi

echo -e "\n==============================================="
echo "Monitoring Completed Successfully"
echo "==============================================="

# Generate summary files
echo -e "\nGenerating summary files..."
python3 "$REPOSITORY_PYTHON"/scripts/generate_summary.py "$OUTPUT_DIR"

echo -e "\n==============================================="
echo "Summary Generation Completed"
echo "==============================================="