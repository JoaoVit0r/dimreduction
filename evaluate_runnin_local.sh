#!/bin/bash

set -e

THESIS_HOME_LOCAL="$THESIS_HOME"

# DIMREDUCTION_JAVA_FOLDER="$THESIS_HOME_LOCAL/virt_machine/java-dimreduction"
# DIMREDUCTION_PYTHON_FOLDER="$THESIS_HOME_LOCAL/dimreduction"
GENECI_FOLDER="$THESIS_HOME_LOCAL/test_external_code/try_minet"


# Run LOCAL Geneci (+ metrics)
MONITORING_FOLDER="monitoring_plots/20251016_191003"
# mkdir -p $MONITORING_FOLDER
# cp -r -t $MONITORING_FOLDER monitoring_plots/20250912_190348 monitoring_plots/20250912_200940
EVALUATION_FOLDER="$HOME/Downloads/Evaluation scripts/Evaluation scripts"

# python scripts/geneci_evaluate.py \
#     --monitoring-dir "$MONITORING_FOLDER" \
#     --external-projects "$GENECI_FOLDER" \
#     --threshold 0.000001 \
#     --output $MONITORING_FOLDER/evaluation_results_confident-in-0.csv \
#     --output-dir $MONITORING_FOLDER/

# printf -v first_file '%s' $MONITORING_FOLDER/*-final_weight_data_converted.txt || exit 1;
# python scripts/csv_2_tsv_net3.py \
#     "$first_file" \
#     $MONITORING_FOLDER/

MATLAB_PREDICTIONS_FULL_PATH_VM1_64threads=$(realpath $MONITORING_FOLDER/)
# mkdir -p $MATLAB_PREDICTIONS_FULL_PATH_VM1_64threads/results

# cd "$EVALUATION_FOLDER/matlab" || exit 1;
# sudo matlab -nodesktop -nosplash -r "go_my('DimReduction', 1, '${MATLAB_PREDICTIONS_FULL_PATH_VM1_64threads}'); exit;" && \
#     cp results/* $MATLAB_PREDICTIONS_FULL_PATH_VM1_64threads/results
# cd - || exit 1;


MONITORING_FOLDER_VM2_1thread="test_to_report_mix_vm1_vm2/vm2"
EVALUATION_ANALYZER_FOLDER=$MONITORING_FOLDER/graphs
mkdir -p "$EVALUATION_ANALYZER_FOLDER"
python scripts/evaluation_analyzer_2.py \
    --time_file $MONITORING_FOLDER/evaluation_results_confident-in-0.csv $MONITORING_FOLDER_VM2_1thread/evaluation_results_confident-in-0.csv \
    --data_folder $MONITORING_FOLDER/results $MONITORING_FOLDER_VM2_1thread/../matlab/results \
    --threads 1 1 \
    --output_dir "$EVALUATION_ANALYZER_FOLDER"