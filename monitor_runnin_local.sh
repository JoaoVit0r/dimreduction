#!/bin/bash

# java 40genes
rm test_to_report_monitor_vm1/40_genes_java_cli_withGC/plots/*
python scripts/plot_dstat.py test_to_report_monitor_vm1/40_400genes/*40genes/java/MainCLI/monitoring_plots/dstat_output_*.csv  test_to_report_monitor_vm1/40_genes_java_cli_withGC/plots  test_to_report_monitor_vm1/40_400genes/*40genes/java/MainCLI/monitoring_plots/execution_markers_*.txt split_by:hour

# java 400genes
rm test_to_report_monitor_vm1/400_genes_java_cli_withGC/plots/*
python scripts/plot_dstat.py test_to_report_monitor_vm1/40_400genes/*400genes/java/MainCLI/monitoring_plots/dstat_output_*.csv  test_to_report_monitor_vm1/400_genes_java_cli_withGC/plots  test_to_report_monitor_vm1/40_400genes/*400genes/java/MainCLI/monitoring_plots/execution_markers_*.txt split_by:hour

# python 40genes
rm test_to_report_monitor_vm1/40_genes_python_cli_withGIL_withGC/plots/*
python scripts/plot_dstat.py test_to_report_monitor_vm1/40_400genes/*40genes/venv12/main_from_cli_no_performing/monitoring_plots/dstat_output_*.csv test_to_report_monitor_vm1/40_genes_python_cli_withGIL_withGC/plots test_to_report_monitor_vm1/40_400genes/*40genes/venv12/main_from_cli_no_performing/monitoring_plots/execution_markers_*.txt 

# python 400genes
rm test_to_report_monitor_vm1/400_genes_python_cli_withGIL_withGC/plots/*
python scripts/plot_dstat.py test_to_report_monitor_vm1/40_400genes/*400genes/venv12/main_from_cli_no_performing/monitoring_plots/dstat_output_*.csv test_to_report_monitor_vm1/400_genes_python_cli_withGIL_withGC/plots test_to_report_monitor_vm1/40_400genes/*400genes/venv12/main_from_cli_no_performing/monitoring_plots/execution_markers_*.txt 
