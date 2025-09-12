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


# # # Run VM2 Multi
# # python scripts/geneci_evaluate.py \
# #     --monitoring-dir "monitoring_plots/20250825_multi" \
# #     --external-projects ../dimreduction_external_comparisons \
# #     --threshold 1 \
# #     --output monitoring_plots/20250825_multi/evaluation_results_threshold_100percent.csv

# # python scripts/geneci_evaluate.py \
# #     --monitoring-dir "monitoring_plots/20250825_multi" \
# #     --external-projects ../dimreduction_external_comparisons \
# #     --threshold 0.3 \
# #     --output monitoring_plots/20250825_multi/evaluation_results_threshold_30percent.csv

# # python scripts/geneci_evaluate.py \
# #     --monitoring-dir "monitoring_plots/20250825_multi" \
# #     --external-projects ../dimreduction_external_comparisons \
# #     --threshold 0.4 \
# #     --output monitoring_plots/20250825_multi/evaluation_results_threshold_40percent.csv

# # python scripts/geneci_evaluate.py \
# #     --monitoring-dir "monitoring_plots/20250825_multi" \
# #     --external-projects ../dimreduction_external_comparisons \
# #     --threshold 0.7 \
# #     --output monitoring_plots/20250825_multi/evaluation_results_threshold_70percent.csv

# # python scripts/geneci_evaluate.py \
# #     --monitoring-dir "monitoring_plots/20250825_multi" \
# #     --external-projects ../dimreduction_external_comparisons \
# #     --skip-binarize \
# #     --output monitoring_plots/20250825_multi/evaluation_results_weight.csv


# Testing local evaluate_metrics
# python ./scripts/evaluate_metrics.py \
#         --prediction temp-final_data.txt \
#         --pred-sep tab \
#         --pred-format matrix \
#         --threshold 0.001 \
#         --gold-standard ../test_external_code/try_minet/input_data/geneci/DREAM4/GS/dream4_100_01_gs.csv \
#         --gold-has-header \
#         --gold-format matrix

# python ./scripts/evaluate_metrics.py \
#         --prediction temp_ARACNE.csv \
#         --pred-format list \
#         --threshold 0.001 \
#         --gold-standard ../test_external_code/try_minet/input_data/geneci/DREAM4/GS/dream4_100_01_gs.csv \
#         --gold-has-header \
#         --gold-format matrix

# Run Local Geneci (+ metrics)
python scripts/geneci_evaluate.py \
    --monitoring-dir "test_evaluation_vm2_20250909_multi/20250909*" \
    --external-projects $THESIS_HOME/test_external_code/try_minet \
    --threshold 0.4 \
    --output test_evaluation_vm2_20250909_multi/evaluation_results_confident-in-40.csv \
    --output-dir test_evaluation_vm2_20250909_multi/

python scripts/geneci_evaluate.py \
    --monitoring-dir "test_evaluation_vm2_20250909_multi/20250909*" \
    --external-projects $THESIS_HOME/test_external_code/try_minet \
    --threshold 0.7 \
    --output test_evaluation_vm2_20250909_multi/evaluation_results_confident-in-70.csv \
    --output-dir test_evaluation_vm2_20250909_multi/

python scripts/geneci_evaluate.py \
    --monitoring-dir "test_evaluation_vm2_20250909_multi/20250909*" \
    --external-projects $THESIS_HOME/test_external_code/try_minet \
    --threshold 0.0001 \
    --output test_evaluation_vm2_20250909_multi/evaluation_results_confident-in-0.csv \
    --output-dir test_evaluation_vm2_20250909_multi/

python scripts/geneci_evaluate.py \
    --monitoring-dir "test_evaluation_vm2_20250909_multi/20250909*" \
    --external-projects $THESIS_HOME/test_external_code/try_minet \
    --threshold 0.3 \
    --output test_evaluation_vm2_20250909_multi/evaluation_results_confident-in-30.csv \
    --output-dir test_evaluation_vm2_20250909_multi/

python scripts/geneci_evaluate.py \
    --monitoring-dir "test_evaluation_vm2_20250909_multi/20250909*" \
    --external-projects $THESIS_HOME/test_external_code/try_minet \
    --threshold 1 \
    --output test_evaluation_vm2_20250909_multi/evaluation_results_confident-in-100.csv \
    --output-dir test_evaluation_vm2_20250909_multi/


# # # Run VM2 Geneci (+ metrics)
# # mkdir -p monitoring_plots/20250905_multi
# # cp -t monitoring_plots/20250905_multi monitoring_plots/20250905_064504 monitoring_plots/20250905_071657 monitoring_plots/20250905_074625 monitoring_plots/20250905_081259 monitoring_plots/20250905_083940 monitoring_plots/20250905_134037

# python scripts/geneci_evaluate.py \
#     --monitoring-dir "monitoring_plots/20250905_multi" \
#     --external-projects $HOME/workspace/dimreduction_external_comparisons \
#     --threshold 0 \
#     --output monitoring_plots/20250905_multi/evaluation_results_threshold_0percent.csv

# python scripts/geneci_evaluate.py \
#     --monitoring-dir "monitoring_plots/20250905_multi" \
#     --external-projects $HOME/workspace/dimreduction_external_comparisons \
#     --threshold 0.7 \
#     --output monitoring_plots/20250905_multi/evaluation_results_threshold_70percent.csv

# python scripts/geneci_evaluate.py \
#     --monitoring-dir "monitoring_plots/20250905_multi" \
#     --external-projects $HOME/workspace/dimreduction_external_comparisons \
#     --threshold 0.4 \
#     --output monitoring_plots/20250905_multi/evaluation_results_threshold_40percent.csv

# python scripts/geneci_evaluate.py \
#     --monitoring-dir "monitoring_plots/20250905_multi" \
#     --external-projects $HOME/workspace/dimreduction_external_comparisons \
#     --threshold 0.3 \
#     --output monitoring_plots/20250905_multi/evaluation_results_threshold_30percent.csv

# python scripts/geneci_evaluate.py \
#     --monitoring-dir "monitoring_plots/20250905_multi" \
#     --external-projects $HOME/workspace/dimreduction_external_comparisons \
#     --threshold 1.0 \
#     --output monitoring_plots/20250905_multi/evaluation_results_threshold_100percent.csv