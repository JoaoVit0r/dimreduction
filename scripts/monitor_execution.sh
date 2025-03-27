#!/bin/bash

SLEEP_TIME=5

# Parse arguments
OUTPUT_DIR=""
REPOSITORY_PYTHON=""
COMMAND=""

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
        *)
            if [ -z "$COMMAND" ]; then
                COMMAND="$@"
                break
            fi
            ;;
    esac
done

# Check if required commands are available
for cmd in dstat python3; do
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
PLOTS_DIR="$OUTPUT_DIR/monitoring_plots"
mkdir -p "$PLOTS_DIR"

# Create a unique temporary file for dstat output
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DSTAT_OUTPUT="$PLOTS_DIR/dstat_output_$TIMESTAMP.csv"
DSTAT_MARKERS="$PLOTS_DIR/execution_markers_$TIMESTAMP.txt"

echo "==============================================="
echo "Starting System Monitoring"
echo "==============================================="
echo "Command to monitor: $COMMAND"
echo "Output directory: $OUTPUT_DIR"
echo "==============================================="

# Start dstat(dool) in the background (time, cpu, load, mem)
dool -tclm --output "$DSTAT_OUTPUT" 1 &
DSTAT_PID=$!

# Wait for dstat to initialize
sleep 2

echo -e "\nStarting command execution monitoring..."
echo "Will execute: $COMMAND"

# Execute the command 3 times with intervals
for i in {1..3}; do
    echo -e "\n-----------------------------------------------"
    echo "Execution $i of 3"
    echo "-----------------------------------------------"
    echo "$(date +'%b-%d %H:%M:%S') Execution_${i}_Start" >> "$DSTAT_MARKERS"
    eval "$COMMAND"
    echo "$(date +'%b-%d %H:%M:%S') Execution_${i}_End" >> "$DSTAT_MARKERS"
    
    # Wait for SLEEP_TIME between executions if not the last one
    if [ $i -lt 3 ]; then
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
echo "Input data: $DSTAT_OUTPUT"
echo "Output directory: $PLOTS_DIR/output"

# Generate plots using the Python script
python3 "$REPOSITORY_PYTHON"/scripts/plot_dstat.py "$DSTAT_OUTPUT" "$PLOTS_DIR/output" "$DSTAT_MARKERS" "split_by:hour"

echo -e "\n==============================================="
echo "Monitoring Completed Successfully"
echo "==============================================="