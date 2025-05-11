#!/bin/bash

# to run with perf on PC
## 
./run_all_monitoring.sh --sleep-time 5 --sleep-time-monitor 5 --number-of-executions 1 --thread-distribution none --threads 1 --custom-input-file ../writing/output/processed_dataset_dream5_40.csv --enable-perf --python-files main_from_cli_no_performing.py,main_from_cli_no_performing_with_GC.py venv_v12
./run_all_monitoring.sh --sleep-time 5 --sleep-time-monitor 5 --number-of-executions 1 --thread-distribution none --threads 1 --custom-input-file ../writing/output/processed_dataset_dream5_400.csv --enable-perf --python-files main_from_cli_no_performing.py,main_from_cli_no_performing_with_GC.py venv_v12

./run_all_monitoring.sh --sleep-time 5 --sleep-time-monitor 5 --number-of-executions 1 --thread-distribution none --threads 1 --custom-input-file ../writing/output/processed_dataset_dream5_40.csv --enable-perf java
