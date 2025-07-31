#!/bin/bash

# Add signal handling
cleanup() {
    local exit_code=$?
    echo -e "\n==============================================="
    echo "Received termination signal - Finishing current execution..."
    echo "==============================================="
    
    # The child processes (monitor_execution.sh) will handle their own cleanup
    exit $exit_code
}

# Set up trap for SIGINT and SIGTERM
trap cleanup SIGINT SIGTERM

SLEEP_TIME=900
MONITOR_SLEEP_TIME=900  # Sleep time for the monitor script between internal executions

REPOSITORY_PYTHON="."
REPOSITORY_JAVA="../dimreduction-java"
COMMANDS=("java" "venv_v12" "venv_v13" "venv_v13-nogil")
CUSTOM_INPUT_FILE_PATH="../writing/output/processed_dataset_dream5_40.csv"
CUSTOM_QUANTIZATION_INPUT_FILE_PATH=""
NUMBER_OF_EXECUTIONS=3
PYTHON_FILES=("main_from_cli.py")
R_FILES=("run_clr.R")
THREADS="1,2,4,8"  # Default thread counts
THREAD_DISTRIBUTION="spaced"  # Default thread distribution
SKIP_MONITORING=false
ENABLE_PERF=false # Flag to enable perf profiling
ENABLE_MANUAL_GC=false # Flag to enable manual garbage collection for Java
BREAK_LOOP=false

break_loop_handler() {
    echo -e "\nReceived SIGUSR1: run_all_monitoring.sh will break after current monitoring run. Command: $BASH_COMMAND"
    BREAK_LOOP=true
}
trap break_loop_handler SIGUSR1

while [[ $# -gt 0 ]]; do
    case $1 in
        --sleep-time)
            SLEEP_TIME="$2"
            shift 2
            ;;
        --sleep-time-monitor)
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
        --repository-r)
            REPOSITORY_R="${2:-../dimreduction_external_comparisons}"
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
        --custom-quantization-input-file)
            CUSTOM_QUANTIZATION_INPUT_FILE_PATH="$2"
            shift 2
            ;;
        --python-files)
            # Convert comma-separated string to array
            IFS=',' read -r -a PYTHON_FILES <<< "$2"
            shift 2
            ;;
        --r-files)
            # Convert comma-separated string to array
            IFS=',' read -r -a R_FILES <<< "$2"
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
        --enable-perf)
            ENABLE_PERF=true
            shift
            ;;
        --enable-manual-gc)
            ENABLE_MANUAL_GC=true
            shift
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
    --sleep-time-monitor <seconds>   Time to wait between monitor script internal executions (default: 900)
    --number-of-executions <number>  Number of times to run each test (default: 3)
    --threads <numbers>              Comma-separated list of thread counts (default: 1,2,4,8)
    --thread-distribution <types>    Comma-separated list of thread distribution types 
                                    (default: spaced,sequential)

  File and Directory Settings:
    --repository-python <path>       Path to Python repository (default: .)
    --repository-java <path>         Path to Java repository (default: ../dimreduction-java)
    --repository-r <path>            Path to R repository (default: ../dimreduction_external_comparisons)
    --custom-input-file <path>       Path to custom input dataset
    --python-files <files>           Comma-separated list of Python files to execute
                                    (default: main_from_cli.py)
    --r-files <files>                Comma-separated list of R files to execute
                                    (default: run_clr.R)

  Performance Settings:
    --enable-perf                    Enable perf profiling
    --enable-manual-gc               Enable manual garbage collection for Java
    --skip-monitoring                Skip monitoring and just execute commands

  Help:
    --help                          Show this help message

Examples:
  # Run all Python files with venv_v12:
  ./run_all_monitoring.sh venv_v12

  # Run specific Python files with custom sleep time:
  ./run_all_monitoring.sh --sleep-time 300 --python-files main_from_cli.py,main_from_cli_ThreadPoolExecutor.py venv_v12

  # Run with specific thread counts:
  ./run_all_monitoring.sh --threads 1,4,16 venv_v12

  # Run Java implementation with custom input file and manual GC:
  ./run_all_monitoring.sh --custom-input-file path/to/dataset.csv --enable-manual-gc java
EOF
            exit 0
            ;;
        *)
            COMMANDS=("$@")
            break
            ;;
    esac
done

DATASET_NAME=$(basename "$CUSTOM_INPUT_FILE_PATH")
DATASET_NAME="${DATASET_NAME%.csv}"
DATASET_NAME="${DATASET_NAME#processed_dataset_}"
DATASET_NAME="${DATASET_NAME#dream5_}"

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
echo "Dataset: $DATASET_NAME"
echo "Number of Executions: $NUMBER_OF_EXECUTIONS"
echo "Thread Counts: $THREADS"
echo "==============================================="

# Convert paths to absolute paths
REPOSITORY_PYTHON=$(realpath "$REPOSITORY_PYTHON")
REPOSITORY_JAVA=$(realpath "$REPOSITORY_JAVA")
REPOSITORY_R=$(realpath "$REPOSITORY_R")

if [ -f "$REPOSITORY_PYTHON"/venv/bin/activate ]; then
    . "$REPOSITORY_PYTHON"/venv/bin/activate
fi
echo -e "\nVirtual Environment: $VIRTUAL_ENV"

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

MONITOR_SCRIPT="$REPOSITORY_PYTHON/scripts/monitor_execution.sh"
JAVA_CLASSPATH="./lib/*:./out/production/java-dimreduction"

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
    # Check if break was requested after sleep
    if [ "$BREAK_LOOP" = true ]; then
        echo "Exiting after wait due to SIGUSR1."
        exit 0
    fi
}

# Function to run monitoring and move files
run_monitoring_python() {
    local python_bin=$1
    local script=$2
    local thread_count=$3
    local thread_distribution=$4

    local script_name
    script_name="$(basename "${script}")"
    local output_dir="${BASE_DIR}/dataset_${DATASET_NAME}/${python_bin%/bin/python}/${script_name%.py}/distribution_${thread_distribution}/threads_${thread_count}"
    mkdir -p "${output_dir}"
    
    echo -e "\n================================================================="
    echo "Running Python Monitoring"
    echo "================================================================="
    echo "Environment: ${python_bin}"
    echo "Script: ${script_name}"
    echo "Thread Count: ${thread_count}"
    echo "Output Directory: ${output_dir}"
    if [ "$ENABLE_PERF" = true ]; then
        echo "Perf Profiling: Enabled"
    fi
    echo "==============================================="

    # Build command with conditional perf option
    local monitor_cmd="$MONITOR_SCRIPT --output-dir \"${output_dir}\" --repository-python \"$REPOSITORY_PYTHON\" \
      --sleep-time \"$MONITOR_SLEEP_TIME\" --number-of-executions \"$NUMBER_OF_EXECUTIONS\" \
      --custom-quantization-input-file \"$CUSTOM_QUANTIZATION_INPUT_FILE_PATH\" \
      --custom-input-file \"$CUSTOM_INPUT_FILE_PATH\" --threads \"$thread_count\" \
      --thread-distribution \"$THREAD_DISTRIBUTION\""

    # Add perf option if enabled
    if [ "$ENABLE_PERF" = true ]; then
        monitor_cmd+=" \"perf record -o ${output_dir}/perf_${script_name%.py}_${python_bin%/bin/python}_dataset_${DATASET_NAME}.data --call-graph dwarf --aio --sample-cpu --mmap-pages 16M\""
        monitor_cmd+=" \"${python_bin}\" -X perf \"${script}\""
    else
        monitor_cmd+=" \"${python_bin}\" \"${script}\""
    fi
    
    cd "$REPOSITORY_PYTHON" || exit
    eval "$monitor_cmd"
    cd - || exit
    
    echo -e "\n-----------------------------------------------"
    echo "Completed ${python_bin} with ${script_name} using ${thread_count} threads"
    echo "-----------------------------------------------"
}

run_monitoring_java() {
    local script="MainCLI"
    local thread_count=$1
    local thread_distribution=$2

    local output_dir="${BASE_DIR}/dataset_${DATASET_NAME}/java/${script}/distribution_${thread_distribution}/threads_${thread_count}"
    mkdir -p "${output_dir}"
    
    echo -e "\n================================================================="
    echo "Running Java Monitoring"
    echo "================================================================="
    echo "Script: ${script}"
    echo "Thread Count: ${thread_count}"
    echo "Thread Distribution: ${thread_distribution}"
    echo "Output Directory: ${output_dir}"
    if [ "$ENABLE_PERF" = true ]; then
        echo "Perf Profiling: Enabled"
    fi
    if [ "$ENABLE_MANUAL_GC" = true ]; then
        echo "Manual GC: Enabled"
    fi
    echo "==============================================="
    
    # Build command with conditional perf option
    local monitor_cmd="$MONITOR_SCRIPT --output-dir \"${output_dir}\" --repository-python \"$REPOSITORY_PYTHON\" \
      --sleep-time \"$MONITOR_SLEEP_TIME\" --number-of-executions \"$NUMBER_OF_EXECUTIONS\" \
      --custom-quantization-input-file \"$CUSTOM_QUANTIZATION_INPUT_FILE_PATH\" \
      --custom-input-file \"$CUSTOM_INPUT_FILE_PATH\" --threads \"$thread_count\" \
      --thread-distribution \"${thread_distribution}\""
    # Add manual GC option if enabled
    if [ "$ENABLE_MANUAL_GC" = true ]; then
        monitor_cmd+=" --enable-manual-gc"
    fi

    # Add perf option if enabled
    if [ "$ENABLE_PERF" = true ]; then
        monitor_cmd+=" \"perf record -o ${output_dir}/perf_${script}_dataset_${DATASET_NAME}.data --call-graph dwarf --aio --sample-cpu --mmap-pages 16M\""
        monitor_cmd+=" \"java\" -cp \"${JAVA_CLASSPATH}\" -XX:+UnlockDiagnosticVMOptions -XX:+DumpPerfMapAtExit -XX:+PreserveFramePointer fs.${script}"
    else
        monitor_cmd+=" \"java\" -cp \"${JAVA_CLASSPATH}\" fs.${script}"
    fi

    cd "$REPOSITORY_JAVA" || exit
    javac -d out/production/java-dimreduction -cp $JAVA_CLASSPATH src/**/*.java;
    cp -r src/img out/production/java-dimreduction/;
    eval "$monitor_cmd";
    cd - || exit
    
    echo -e "\n-----------------------------------------------"
    echo "Completed Java with ${script} using ${thread_count} threads"
    echo "-----------------------------------------------"
}

# Function to run monitoring and move files
run_monitoring_r() {
    local r_bin=$1
    local script=$2
    local thread_count=$3
    local thread_distribution="none"

    local script_name
    script_name="$(basename "${script}")"
    local output_dir="${BASE_DIR}/dataset_${DATASET_NAME}/${r_bin}/${script_name%.R}/distribution_${thread_distribution}/threads_${thread_count}"
    mkdir -p "${output_dir}"
    
    echo -e "\n================================================================="
    echo "Running R Monitoring"
    echo "================================================================="
    echo "R repository: ${REPOSITORY_R}"
    echo "Environment: ${r_bin}"
    echo "Script: ${script_name}"
    echo "Thread Count: ${thread_count}"
    echo "Output Directory: ${output_dir}"
    if [ "$ENABLE_PERF" = true ]; then
        echo "Perf Profiling: Enabled"
    fi
    echo "==============================================="

    # Build command with conditional perf option
    local monitor_cmd="$MONITOR_SCRIPT --output-dir \"${output_dir}\" --repository-python \"$REPOSITORY_PYTHON\" --repository-r \"$REPOSITORY_R\" \
      --sleep-time \"$MONITOR_SLEEP_TIME\" --number-of-executions \"$NUMBER_OF_EXECUTIONS\" \
      --custom-quantization-input-file \"$CUSTOM_QUANTIZATION_INPUT_FILE_PATH\" \
      --custom-input-file \"$CUSTOM_INPUT_FILE_PATH\" --threads \"$thread_count\" \
      --thread-distribution \"$THREAD_DISTRIBUTION\""

    # Add perf option if enabled
    if [ "$ENABLE_PERF" = true ]; then
        echo "Perf not supported to R executions"
    fi
    monitor_cmd+=" \"${r_bin}\" \"${script}\""
    
    echo "Running: $monitor_cmd"
    cd "$REPOSITORY_R" || exit
    eval "$monitor_cmd"
    cd - || exit
    
    echo -e "\n-----------------------------------------------"
    echo "Completed ${r_bin} with ${script_name} using ${thread_count} threads"
    echo "-----------------------------------------------"
}

# Convert thread counts string to array
IFS=',' read -r -a THREAD_COUNTS <<< "$THREADS"
# Convert thread distribution string to array
IFS=',' read -r -a THREAD_DISTRIBUTION <<< "$THREAD_DISTRIBUTION"

# Run all combinations in sequence
for cmd in "${COMMANDS[@]}"; do
    if [ "$cmd" == "java" ]; then
        for thread_distribution in "${THREAD_DISTRIBUTION[@]}"; do
            for thread_count in "${THREAD_COUNTS[@]}"; do
                run_monitoring_java "$thread_count" "$thread_distribution"
                wait_between_executions
            done
        done
    elif [ "$cmd" == "Rscript" ]; then
        for r_file in "${R_FILES[@]}"; do
            for thread_distribution in "${THREAD_DISTRIBUTION[@]}"; do
                for thread_count in "${THREAD_COUNTS[@]}"; do
                    run_monitoring_r "$cmd" "$REPOSITORY_R/$r_file" "$thread_count"
                    wait_between_executions
                done
            done
        done
    else
        if [ -f "$REPOSITORY_PYTHON/$cmd/bin/python" ]; then
            for python_file in "${PYTHON_FILES[@]}"; do
                for thread_distribution in "${THREAD_DISTRIBUTION[@]}"; do
                    for thread_count in "${THREAD_COUNTS[@]}"; do
                        run_monitoring_python "$cmd/bin/python" "$REPOSITORY_PYTHON/$python_file" "$thread_count" "$thread_distribution"
                        wait_between_executions
                    done
                done
            done
        else
            for python_file in "${PYTHON_FILES[@]}"; do
                for thread_distribution in "${THREAD_DISTRIBUTION[@]}"; do
                    for thread_count in "${THREAD_COUNTS[@]}"; do
                        run_monitoring_python "$cmd" "$REPOSITORY_PYTHON/$python_file" "$thread_count" "$thread_distribution"
                        wait_between_executions
                    done
                done
            done
        fi
    fi

    # At the end of each main loop iteration, check if break was requested
    if [ "$BREAK_LOOP" = true ]; then
        echo "Breaking main loop after command '$cmd' due to SIGUSR1."
        break
    fi
done

echo -e "\n================================================================="
echo "All Executions Completed"
echo "================================================================="
echo "Results are organized in: ${BASE_DIR}"
echo "==============================================="
date
