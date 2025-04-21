#!/bin/bash

SLEEP_TIME=900
MONITOR_SLEEP_TIME=900  # Sleep time for the monitor script between executions

REPOSITORY_PYTHON="."
REPOSITORY_JAVA="../dimreduction-java"
COMMANDS=("java" "venv_v12" "venv_v13" "venv_v13-nogil")
CUSTOM_INPUT_FILE_PATH="../writing/output/processed_dataset_dream5_40.csv"
NUMBER_OF_EXECUTIONS=3
PYTHON_FILES=("main_from_cli.py" "main_from_cli_ThreadPoolExecutor.py" "main_from_cli_no_performing.py")
THREADS="1,2,4,8"  # Default thread counts

while [[ $# -gt 0 ]]; do
    case $1 in
        --sleep-time)
            SLEEP_TIME="$2"
	        MONITOR_SLEEP_TIME="$2"
            shift 2
            ;;
        --repository-python)
            REPOSITORY_PYTHON="${2:-.}"
            shift 2
            ;;
    	--repository-java)
            REPOSITORY_JAVA="${2:-../dimreduction-java}"
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
        --python-files)
            # Convert comma-separated string to array
            IFS=',' read -r -a PYTHON_FILES <<< "$2"
            shift 2
            ;;
        --threads)
            THREADS="$2"
            shift 2
            ;;
        --help)
            cat << 'EOF'
Usage: ./run_all_monitoring.sh [OPTIONS] [COMMANDS]

Monitor system performance while executing Python and Java programs.

Commands:
  java              Execute Java implementation
  venv_v12          Execute using Python 3.12 virtual environment
  venv_v13          Execute using Python 3.13 virtual environment
  venv_v13-nogil    Execute using Python 3.13 no-GIL virtual environment

Options:
  Basic Configuration:
    --sleep-time <seconds>           Time to wait between executions (default: 900)
    --number-of-executions <number>  Number of times to run each test (default: 3)
    --threads <numbers>              Comma-separated list of thread counts (default: 1,2,4,8)

  File and Directory Settings:
    --repository-python <path>       Path to Python repository (default: .)
    --repository-java <path>         Path to Java repository (default: ../dimreduction-java)
    --custom-input-file <path>       Path to custom input dataset
    --python-files <files>           Comma-separated list of Python files to execute
                                    (default: main_from_cli.py,main_from_cli_ThreadPoolExecutor.py,main_from_cli_no_performing.py)

  Help:
    --help                          Show this help message

Examples:
  # Run all Python files with venv_v12:
  ./run_all_monitoring.sh venv_v12

  # Run specific Python files with custom sleep time:
  ./run_all_monitoring.sh --sleep-time 300 --python-files main_from_cli.py,main_from_cli_ThreadPoolExecutor.py venv_v12

  # Run with specific thread counts:
  ./run_all_monitoring.sh --threads 1,4,16 venv_v12

  # Run Java implementation with custom input file:
  ./run_all_monitoring.sh --custom-input-file path/to/dataset.csv java
EOF
            exit 0
            ;;
        *)
            COMMANDS=("$@")
            break
            ;;
    esac
done

echo "==============================================="
echo "Starting Monitoring Session"
echo "==============================================="
echo "Python Repository: $REPOSITORY_PYTHON"
echo "Java Repository: $REPOSITORY_JAVA"
if [ $SLEEP_TIME -ge 60 ]; then
    echo "Sleep Time: $(($SLEEP_TIME / 60)) minutes"
else
    echo "Sleep Time: $SLEEP_TIME seconds"
fi
echo "Commands: ${COMMANDS[*]}"
echo "Python Files: ${PYTHON_FILES[*]}"
echo "Custom Input File: $CUSTOM_INPUT_FILE_PATH"
echo "Number of Executions: $NUMBER_OF_EXECUTIONS"
echo "Thread Counts: $THREADS"
echo "==============================================="

# Convert paths to absolute paths
REPOSITORY_PYTHON=$(realpath "$REPOSITORY_PYTHON")
REPOSITORY_JAVA=$(realpath "$REPOSITORY_JAVA")

if [ -f "$REPOSITORY_PYTHON"/venv/bin/activate ]; then
    . "$REPOSITORY_PYTHON"/venv/bin/activate
fi
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

# Function to wait between executions with proper message
wait_between_executions() {
    echo -e "\n-----------------------------------------------"
    echo "Waiting between Monitor script executions..."
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
    local python_bin=$1
    local script=$2
    local thread_count=$3

    local script_name
    script_name="$(basename "${script}")"
    local output_dir="${BASE_DIR}/${python_bin%/bin/python}/${script_name%.py}/threads_${thread_count}"
    mkdir -p "${output_dir}"
    
    echo -e "\n================================================================="
    echo "Running Python Monitoring"
    echo "================================================================="
    echo "Environment: ${python_bin}"
    echo "Script: ${script_name}"
    echo "Thread Count: ${thread_count}"
    echo "Output Directory: ${output_dir}"
    echo "==============================================="
    
    cd "$REPOSITORY_PYTHON" || exit
    $MONITOR_SCRIPT --output-dir "${output_dir}" --repository-python "$REPOSITORY_PYTHON" --sleep-time "$MONITOR_SLEEP_TIME" --number-of-executions "$NUMBER_OF_EXECUTIONS" --custom-input-file "$CUSTOM_INPUT_FILE_PATH" --threads "$thread_count" "${python_bin}" "${script}"
    cd - || exit
    
    echo -e "\n-----------------------------------------------"
    echo "Completed ${python_bin} with ${script_name} using ${thread_count} threads"
    echo "-----------------------------------------------"
}

run_monitoring_java() {
    local script="MainCLI"
    local thread_count=$1

    local output_dir="${BASE_DIR}/java/${script}_threads${thread_count}"
    mkdir -p "${output_dir}"
    
    echo -e "\n================================================================="
    echo "Running Java Monitoring"
    echo "================================================================="
    echo "Script: ${script}"
    echo "Thread Count: ${thread_count}"
    echo "Output Directory: ${output_dir}"
    echo "==============================================="
    
    cd "$REPOSITORY_JAVA" || exit
    # cp -r src/img out/production/java-dimreduction/
    # javac -d out/production/java-dimreduction -cp $JAVA_CLASSPATH src/**/*.java
    $MONITOR_SCRIPT --output-dir "${output_dir}" --repository-python "$REPOSITORY_PYTHON" --sleep-time "$MONITOR_SLEEP_TIME" --number-of-executions "$NUMBER_OF_EXECUTIONS" --custom-input-file "$CUSTOM_INPUT_FILE_PATH" --threads "$thread_count" "java" -cp "${JAVA_CLASSPATH}" -Xmx16384m -XX:+UnlockDiagnosticVMOptions -XX:+DumpPerfMapAtExit fs."${script}"
    cd - || exit
    
    echo -e "\n-----------------------------------------------"
    echo "Completed Java with ${script} using ${thread_count} threads"
    echo "-----------------------------------------------"
}

# Convert thread counts string to array
IFS=',' read -r -a THREAD_COUNTS <<< "$THREADS"

# Run all combinations in sequence
for cmd in "${COMMANDS[@]}"; do
    if [ "$cmd" == "java" ]; then
        for thread_count in "${THREAD_COUNTS[@]}"; do
            run_monitoring_java "$thread_count"
            wait_between_executions
        done
    else
        if [ -f "$REPOSITORY_PYTHON/$cmd/bin/python" ]; then
            for python_file in "${PYTHON_FILES[@]}"; do
                for thread_count in "${THREAD_COUNTS[@]}"; do
                    run_monitoring_python "$cmd/bin/python" "$REPOSITORY_PYTHON/$python_file" "$thread_count"
                    wait_between_executions
                done
            done
        else
            for python_file in "${PYTHON_FILES[@]}"; do
                for thread_count in "${THREAD_COUNTS[@]}"; do
                    run_monitoring_python "$cmd" "$REPOSITORY_PYTHON/$python_file" "$thread_count"
                    wait_between_executions
                done
            done
        fi
    fi
done

echo -e "\n================================================================="
echo "All Executions Completed"
echo "================================================================="
echo "Results are organized in: ${BASE_DIR}"
echo "==============================================="
date
