#!/bin/bash

set -e

THESIS_HOME_VM2="$HOME/workspace"

DIMREDUCTION_JAVA_FOLDER="$THESIS_HOME_VM2/dimreduction-java"
DIMREDUCTION_PYTHON_FOLDER="$THESIS_HOME_VM2/dimreduction-python"
GENECI_FOLDER="$THESIS_HOME_VM2/dimreduction_external_comparisons"

# VM2 get last updates
cd "$DIMREDUCTION_JAVA_FOLDER"
git pull
git status

cd "$GENECI_FOLDER"
git pull
git status

cd "$DIMREDUCTION_PYTHON_FOLDER"
git pull
git status


# Run VM1 Geneci (+ metrics)
MONITORING_FOLDER="monitoring_plots/20250914_mutiples_runs_VM1"
mkdir -p $MONITORING_FOLDER/matlab/results

# # already executed - START
# python scripts/geneci_evaluate.py \
#     --monitoring-dir "$MONITORING_FOLDER" \
#     --external-projects "$GENECI_FOLDER" \
#     --threshold 0.000000001 \
#     --output $MONITORING_FOLDER/evaluation_results_confident-in-0.csv \
#     --output-dir $MONITORING_FOLDER/ \
#     --skip-binarize

# ref_file=$(find $MONITORING_FOLDER -name "*-final_weight_data.txt" -type f | head -1)
# # Check if file was found
# if [[ -z "$ref_file" ]]; then
#     echo "Error: No file found matching the pattern"
#     exit 1
# fi
# dir_path=$(dirname "$ref_file")
# base_name=$(basename "$ref_file" ".txt")
# DimReduction_list_file="$dir_path/${base_name}_list_with_nonTFpred.txt"
# python scripts/dream5_converter.py \
#     "$ref_file" \
#     "$DimReduction_list_file" \
#     --max-predictions 21000000

# python scripts/csv_2_tsv_net3.py \
#     "$DimReduction_list_file" \
#     $MONITORING_FOLDER/matlab/
# # already executed - END

echo manual edit the go_my to use this path to predictions!!!
exit;

sudo matlab -nodesktop -nosplash -r "go_my('DimReduction', 64); exit;" && \

# List of methods
methods=("GENIE3_ET" "GENIE3_RF" "KBOOST")

for method in "${methods[@]}"; do
    # method_lower=$(echo "$method" | tr '[:upper:]' '[:lower:]')
    python scripts/csv_2_tsv_net3.py \
        $MONITORING_FOLDER/GRN_$method.csv \
        "$MONITORING_FOLDER/matlab/"
done

for method in "${methods[@]}"; do
    sudo matlab -nodesktop -nosplash -r "go_my('${method}', 64); exit;"
done

cp results/* $MONITORING_FOLDER/matlab/results

mkdir -p $MONITORING_FOLDER/graphs
python scripts/evaluation_analyzer_2.py \
    --time_file $MONITORING_FOLDER/evaluation_results_confident-in-0.csv \
    --data_folder $MONITORING_FOLDER/matlab/results \
    --threads 1 \
    --output_dir $MONITORING_FOLDER/graphs
    