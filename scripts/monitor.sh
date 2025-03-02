#!/bin/bash

# Define the output file
OUTPUT_FILE="monitor/monitoring_data.log"

# Function to get memory consumption
get_memory_consumption() {
    # free -m | awk 'NR==2{printf "Memory Usage: %s/%sMB (%.2f%%)\n", $3,$2,$3*100/$2 }'
    free -m | awk 'NR==2{printf "%.2f%\n", $3*100/$2 }'
}

# Function to get CPU usage
get_cpu_usage() {
    top -bn1 | grep "Cpu(s)" | \
    grep -P -o "([0-9,]*)(?= id)" | \
    awk '{printf "%.2f", 100 - $1}'
}

# Function to get CPU load
get_cpu_load() {
    # uptime | awk -F'[a-z]:' '{ print "CPU Load (1m, 5m, 15m):" $2 }'
    # split the cpu load into 3 parts separated by comma
    # load=$(uptime | awk -F'[a-z]:' '{ print $2 }')
    # echo "CPU Load (1m, 5m, 15m): $load"
    # echo "$(uptime | awk -F'[a-z]:' '{ print $2 }' | sed 's/,/./g')"
    uptime | awk -F'load average: ' '{ print $2 }' | awk -F', ' '{ printf "%s;%s;%s\n", $1, $2, $3 }'
}

# # Write data to the output file
# {
#     echo "Timestamp: $(date)"
#     get_memory_consumption
#     get_cpu_usage
#     get_cpu_load
#     echo "-----------------------------------"
# } >> "$OUTPUT_FILE"

echo "Monitoring data written to $OUTPUT_FILE"

# Write in csv format
{
    echo "$(date);$(get_memory_consumption);$(get_cpu_usage);$(get_cpu_load)"
} >> "$OUTPUT_FILE"


echo "Monitoring data written to $OUTPUT_FILE finished"