#!/bin/bash

SLEEP_TIME=900
MONITOR_SLEEP_TIME=900  # Sleep time for the monitor script between executions

REPOSITORY_PYTHON="${1:-.}"
REPOSITORY_JAVA="${2:-../dimreduction-java}"

echo "==============================================="
echo "Starting Monitoring Session"
echo "==============================================="
echo "Python Repository: $REPOSITORY_PYTHON"
echo "Java Repository: $REPOSITORY_JAVA"
echo "Sleep Time: $(($SLEEP_TIME / 60)) minutes"
echo "==============================================="

# Convert paths to absolute paths
REPOSITORY_PYTHON=$(realpath "$REPOSITORY_PYTHON")
REPOSITORY_JAVA=$(realpath "$REPOSITORY_JAVA")

. "$REPOSITORY_PYTHON"/venv/bin/activate
echo -e "\nVirtual Environment: $VIRTUAL_ENV"

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

MONITOR_SCRIPT="$REPOSITORY_PYTHON/scripts/monitor_execution.sh"
JAVA_CLASSPATH="./lib/*:./out/production/java-dimreduction:./lib/jgraph.jar:./lib/jgraphlayout.jar:./lib/prefuse.jar:./lib/jfreechart-1.0.9.jar:./lib/jcommon-1.0.12.jar:./lib/dotenv-java-3.0.2.jar"

# Get current timestamp for directory organization
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Create base directory for this run inside the Python repository
BASE_DIR="$REPOSITORY_PYTHON/monitoring_plots/${TIMESTAMP}"
mkdir -p "${BASE_DIR}"

# Function to move files
move_files() {
    local output_dir="$1"
    local source_dir="${2:-monitoring_plots}"
    
    # Create output directory
    mkdir -p "${output_dir}"
    
    # Move monitoring files
    mv "${source_dir}"/dstat_*.csv "${output_dir}/" 2>/dev/null || true
    mv "${source_dir}"/execution_markers*.txt "${output_dir}/" 2>/dev/null || true
    mv "${source_dir}"/output/dstat_plot_*.svg "${output_dir}/" 2>/dev/null || true
}

# Function to wait between executions with proper message
wait_between_executions() {
    echo -e "\n-----------------------------------------------"
    echo "Waiting between executions..."
    if [ $SLEEP_TIME -ge 60 ]; then
        echo "Next execution in $(($SLEEP_TIME / 60)) minutes"
    else
        echo "Next execution in $SLEEP_TIME seconds"
    fi
    echo "-----------------------------------------------"
    sleep "$SLEEP_TIME"
}

# Function to run monitoring and move files
run_monitoring_python() {
    local venv=$1
    local script=$2

    local script_name
    script_name="$(basename "${script}")"
    local output_dir="${BASE_DIR}/${venv}/${script_name%.py}"
    mkdir -p "${output_dir}"
    
    echo -e "\n==============================================="
    echo "Running Python Monitoring"
    echo "==============================================="
    echo "Environment: ${venv}"
    echo "Script: ${script_name}"
    echo "Output Directory: ${output_dir}"
    echo "==============================================="
    
    cd "$REPOSITORY_PYTHON" || exit
    $MONITOR_SCRIPT --output-dir "${output_dir}" --repository-python "$REPOSITORY_PYTHON" --sleep-time "$MONITOR_SLEEP_TIME" "${venv}/bin/python" "${script}"
    cd - || exit
    
    echo -e "\n-----------------------------------------------"
    echo "Completed ${venv} with ${script_name}"
    echo "-----------------------------------------------"
}

run_monitoring_java() {
    local script="MainCLI"

    local output_dir="${BASE_DIR}/java/${script}"
    mkdir -p "${output_dir}"
    
    echo -e "\n==============================================="
    echo "Running Java Monitoring"
    echo "==============================================="
    echo "Script: ${script}"
    echo "Output Directory: ${output_dir}"
    echo "==============================================="
    
    cd "$REPOSITORY_JAVA" || exit
    $MONITOR_SCRIPT --output-dir "${output_dir}" --repository-python "$REPOSITORY_PYTHON" --sleep-time "$MONITOR_SLEEP_TIME" "java" -Dfile.encoding=UTF-8 -Dsun.stdout.encoding=UTF-8 -Dsun.stderr.encoding=UTF-8 -classpath $JAVA_CLASSPATH -Xmx16384m -XX:+UnlockDiagnosticVMOptions -XX:+DumpPerfMapAtExit fs."${script}"
    cd - || exit
    
    echo -e "\n-----------------------------------------------"
    echo "Completed Java with ${script}"
    echo "-----------------------------------------------"
}

# Run all combinations in sequence
run_monitoring_java "$REPOSITORY_JAVA/src/fs/MainCLI.java"

wait_between_executions
run_monitoring_python "venv_v12" "$REPOSITORY_PYTHON/main_from_cli_no_performing.py"

wait_between_executions
run_monitoring_python "venv_v12" "$REPOSITORY_PYTHON/main_from_cli_performing.py"

wait_between_executions
run_monitoring_python "venv_v13" "$REPOSITORY_PYTHON/main_from_cli_no_performing.py"

wait_between_executions
run_monitoring_python "venv_v13" "$REPOSITORY_PYTHON/main_from_cli_performing.py"

wait_between_executions
run_monitoring_python "venv_v13-nogil" "$REPOSITORY_PYTHON/main_from_cli_no_performing.py"

wait_between_executions
run_monitoring_python "venv_v13-nogil" "$REPOSITORY_PYTHON/main_from_cli_performing.py"

echo -e "\n==============================================="
echo "All Executions Completed"
echo "==============================================="
echo "Results are organized in: ${BASE_DIR}"
echo "==============================================="
date
