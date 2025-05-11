#!/bin/bash

# Check if output directory is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <output_directory>"
    echo "Example: $0 ./output/monitoring_plots"
    exit 1
fi

OUTPUT_DIR="$1"

# Check if output directory exists
if [ ! -d "$OUTPUT_DIR" ]; then
    echo "Error: Output directory '$OUTPUT_DIR' does not exist."
    exit 1
fi

echo "Moving monitoring files to $OUTPUT_DIR..."

# Move logs.log if it exists
if [ -f "logs/logs.log" ]; then
    echo "Moving logs.log..."
    mv --backup="numbered" "logs/logs.log" "$OUTPUT_DIR"
fi

# Move timing log files if directory exists
if [ -d "timing" ]; then
    echo "Moving timing log files..."
    find timing -name "*.log" -exec mv --backup="numbered" {} "$OUTPUT_DIR" \;
fi

# Move results text files if directory exists
if [ -d "results" ]; then
    echo "Moving results text files..."
    mv --backup="numbered" results/*/*.txt "$OUTPUT_DIR" 2>/dev/null || true
fi

# Move perf.data if it exists
if [ -f "perf.data" ]; then
    echo "Moving perf.data..."
    mv --backup="numbered" "perf.data" "$OUTPUT_DIR"
fi

# Move perf*.data if they exist
if ls perf*.data 1> /dev/null 2>&1; then
    echo "Moving perf*.data files..."
    mv --backup="numbered" perf*.data "$OUTPUT_DIR"
fi

echo "File moving completed successfully!" 