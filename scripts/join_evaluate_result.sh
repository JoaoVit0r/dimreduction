#!/bin/bash

cd test_evaluation_vm2_20250905_multi || exit 1

output_file="evaluation_results_threshold_results.csv"
first_file=true

# Process files in sorted order by their threshold value
for file in $(ls evaluation_results_threshold_*percent.csv | sort -V); do
    # Extract threshold number from filename
    threshold=$(echo "$file" | grep -o '[0-9]\+percent' | cut -d'p' -f1)
    
    if [ "$first_file" = true ]; then
        # For the first file, include header with threshold as first column
        awk -v th="$threshold" 'NR==1 {print "threshold," $0} NR>1 {print th "," $0}' "$file" > "$output_file"
        first_file=false
    else
        # For subsequent files, skip header and add threshold as first column
        awk -v th="$threshold" 'NR>1 {print th "," $0}' "$file" >> "$output_file"
    fi
done

cd - || exit 1
