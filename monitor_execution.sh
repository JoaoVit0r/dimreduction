#!/bin/bash

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

# Check if a command was provided as an argument
if [ $# -eq 0 ]; then
    echo "Usage: $0 <command_to_execute>"
    exit 1
fi

# Store the command to execute
COMMAND="$@"

# Create output directory for plots
PLOTS_DIR="monitoring_plots"
mkdir -p "$PLOTS_DIR"

# Create a unique temporary file for dstat output
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DSTAT_OUTPUT="$PLOTS_DIR/dstat_output_$TIMESTAMP.csv"

# Start dstat in the background (CPU, memory, disk, network)
dool --cpu --mem --disk --net --output "$DSTAT_OUTPUT" 1 &
DSTAT_PID=$!

# Wait for dstat to initialize
sleep 2

echo "Starting command execution monitoring..."
echo "Will execute: $COMMAND"

# Execute the command 3 times with intervals
for i in {1..3}; do
    echo -e "\nExecution $i starting at $(date)"
    eval "$COMMAND"
    echo "Execution $i completed at $(date)"
    
    # Wait for 5 seconds between executions if not the last one
    if [ $i -lt 3 ]; then
        echo "Waiting 5 seconds before next execution..."
        sleep 5
    fi
done

# Continue monitoring for 5 more seconds after last execution
echo -e "\nAll executions completed. Monitoring for 5 more seconds..."
sleep 5

# Gracefully stop dstat
kill $DSTAT_PID

echo -e "\nMonitoring completed. Dstat output saved to: $DSTAT_OUTPUT"

# Generate plots using the Python script
echo "Generating performance plots..."
python3 plot_dstat.py "$DSTAT_OUTPUT" "$PLOTS_DIR"

echo -e "\nPlots have been generated in the $PLOTS_DIR directory"
echo "You can find:"
echo "- Raw data: $DSTAT_OUTPUT"
echo "- Performance plots: $PLOTS_DIR/dstat_plot_*.svg" 