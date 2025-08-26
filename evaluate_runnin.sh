#!/bin/bash

set -e

# # Run Local Geneci
# python scripts/geneci_evaluate.py \
#     --monitoring-dir "monitoring_plots/20250824_142959" \
#     --external-projects ../test_external_code/try_minet \
#     --threshold 0.4 \
#     --output evaluation_results_40percent.csv
    
# python scripts/geneci_evaluate.py \
#     --monitoring-dir "monitoring_plots/20250824_142959" \
#     --external-projects ../test_external_code/try_minet \
#     --threshold 0.7 \
#     --output evaluation_results_70percent.csv

# python scripts/geneci_evaluate.py \
#     --monitoring-dir "monitoring_plots/20250824_142959" \
#     --external-projects ../test_external_code/try_minet \
#     --threshold 0 \
#     --output evaluation_results_0percent.csv

# python scripts/geneci_evaluate.py \
#     --monitoring-dir "monitoring_plots/20250824_142959" \
#     --external-projects ../test_external_code/try_minet \
#     --skip-binarize \
#     --output evaluation_results_weight.csv


# Run VM2 Multi
python scripts/geneci_evaluate.py \
    --monitoring-dir "monitoring_plots/20250825_multi" \
    --external-projects ../dimreduction_external_comparisons \
    --threshold 1 \
    --output monitoring_plots/20250825_multi/evaluation_results_threshold_100percent.csv

python scripts/geneci_evaluate.py \
    --monitoring-dir "monitoring_plots/20250825_multi" \
    --external-projects ../dimreduction_external_comparisons \
    --threshold 0.3 \
    --output monitoring_plots/20250825_multi/evaluation_results_threshold_30percent.csv

python scripts/geneci_evaluate.py \
    --monitoring-dir "monitoring_plots/20250825_multi" \
    --external-projects ../dimreduction_external_comparisons \
    --threshold 0.4 \
    --output monitoring_plots/20250825_multi/evaluation_results_threshold_40percent.csv

python scripts/geneci_evaluate.py \
    --monitoring-dir "monitoring_plots/20250825_multi" \
    --external-projects ../dimreduction_external_comparisons \
    --threshold 0.7 \
    --output monitoring_plots/20250825_multi/evaluation_results_threshold_70percent.csv

python scripts/geneci_evaluate.py \
    --monitoring-dir "monitoring_plots/20250825_multi" \
    --external-projects ../dimreduction_external_comparisons \
    --skip-binarize \
    --output monitoring_plots/20250825_multi/evaluation_results_weight.csv

    