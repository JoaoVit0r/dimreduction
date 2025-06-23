#!/bin/bash

# to run with perf on PC
##
#./run_all_monitoring.sh --sleep-time 5 --sleep-time-monitor 5 --number-of-executions 1 --thread-distribution none --threads 1 --custom-input-file ../writing/output/processed_dataset_dream5_40.csv --enable-perf --python-files main_from_cli_no_performing.py,main_from_cli_no_performing_with_GC.py venv_v12
#./run_all_monitoring.sh --sleep-time 5 --sleep-time-monitor 5 --number-of-executions 1 --thread-distribution none --threads 1 --custom-input-file ../writing/output/processed_dataset_dream5_400.csv --enable-perf --python-files main_from_cli_no_performing.py,main_from_cli_no_performing_with_GC.py venv_v12

# -----------------------------------------------
# to run with perf on VM1
## 
#./run_all_monitoring.sh --sleep-time 5 --sleep-time-monitor 5 --number-of-executions 1 --thread-distribution none --threads 1 --custom-input-file ../writing/output/processed_dataset_dream5_40.csv --python-files main_from_cli.py venv12
#./run_all_monitoring.sh --sleep-time 5 --sleep-time-monitor 5 --number-of-executions 1 --thread-distribution none --threads 1 --custom-input-file ../writing/output/processed_dataset_dream5_40.csv --python-files main_from_cli.py venv_14
#./run_all_monitoring.sh --sleep-time 5 --sleep-time-monitor 5 --number-of-executions 1 --thread-distribution none --threads 1 --custom-input-file ../writing/output/processed_dataset_dream5_40.csv --python-files main_from_cli.py venv_pypy


# #./run_all_monitoring.sh --sleep-time 5 --number-of-executions 2 --thread-distribution demain,spaced,sequencial --threads 64,32,16,8,4 --custom-input-file ../writing/output/processed_dataset_dream5_full.csv --python-files main_from_cli.py venv_pypy
# ./run_all_monitoring.sh --sleep-time 5 --number-of-executions 2 --thread-distribution demain,spaced,sequencial --threads 64,32,16,8,4 --custom-input-file ../writing/output/processed_dataset_dream5_full.csv --python-files main_from_cli.py venv_14t
# ./run_all_monitoring.sh --sleep-time 5 --number-of-executions 2 --thread-distribution demain,spaced,sequencial --threads 2,1 --custom-input-file ../writing/output/processed_dataset_dream5_full.csv --python-files main_from_cli.py venv_14t
# ./run_all_monitoring.sh --sleep-time 5 --number-of-executions 2 --thread-distribution demain,spaced,sequencial --threads 2,1 --custom-input-file ../writing/output/processed_dataset_dream5_full.csv --python-files main_from_cli.py venv_pypy

# ./run_all_monitoring.sh --sleep-time 5 --number-of-executions 2 --thread-distribution demain,spaced,sequencial --threads 64,32 --custom-input-file ../writing/output/processed_dataset_dream5_full.csv --python-files main_from_cli.py venv_pypy
# ./run_all_monitoring.sh --sleep-time 5 --number-of-executions 2 --thread-distribution demain,spaced,sequencial --threads 16,8,4 --custom-input-file ../writing/output/processed_dataset_dream5_full.csv --python-files main_from_cli.py venv_pypy


#./run_all_monitoring.sh --sleep-time 5 --sleep-time-monitor 5 --number-of-executions 1 --thread-distribution none --threads 1 --custom-input-file ../writing/output/processed_dataset_dream5_40.csv --enable-perf java

# -----------------------------------------------
# to run with perf on VM2
##
#./run_all_monitoring.sh --sleep-time 5 --number-of-executions 2 --thread-distribution demain,spaced,sequential --threads 64 --custom-input-file ../writing/output/processed_dataset_dream5_full.csv java

#./run_all_monitoring.sh --sleep-time 5 --sleep-time-monitor 5 --number-of-executions 1 --thread-distribution none --threads 1 --custom-input-file ../writing/output/processed_dataset_dream5_400.csv java
# ./run_all_monitoring.sh --sleep-time 5 --number-of-executions 1 --thread-distribution demain,spaced,sequential --threads 1 --custom-input-file ../writing/output/processed_dataset_dream5_full.csv java

./run_all_monitoring.sh --sleep-time 5 --sleep-time-monitor 5 --number-of-executions 1 --thread-distribution none --threads 1 --custom-input-file ../writing/output/processed_dataset_dream5_40.csv --python-files main_from_cli.py venv_v12 venv_14t venv_pypy venv_13t

./run_all_monitoring.sh --sleep-time 5 --number-of-executions 2 --thread-distribution demain,spaced,sequencial --threads 64 --custom-input-file ../writing/output/processed_dataset_dream5_full.csv --python-files main_from_cli.py venv_14t venv_13t venv_pypy

./run_all_monitoring.sh --sleep-time 5 --number-of-executions 2 --thread-distribution demain,spaced,sequencial --threads 32,16,8,4,2,1 --custom-input-file ../writing/output/processed_dataset_dream5_full.csv --python-files main_from_cli.py venv_14t
./run_all_monitoring.sh --sleep-time 5 --number-of-executions 2 --thread-distribution demain,spaced,sequencial --threads 1 --custom-input-file ../writing/output/processed_dataset_dream5_full.csv --python-files main_from_cli.py venv_pypy
./run_all_monitoring.sh --sleep-time 5 --number-of-executions 2 --thread-distribution demain,spaced,sequencial --threads 1 --custom-input-file ../writing/output/processed_dataset_dream5_full.csv --python-files main_from_cli.py venv_13t
