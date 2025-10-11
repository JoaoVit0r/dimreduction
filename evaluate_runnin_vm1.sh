#!/bin/bash

set -e

THESIS_HOME_VM1="$HOME/virt_machine"

DIMREDUCTION_JAVA_FOLDER="$THESIS_HOME_VM1/dimreduction-java"
DIMREDUCTION_PYTHON_FOLDER="$THESIS_HOME_VM1/dimreduction-python"
GENECI_FOLDER="$THESIS_HOME_VM1/dimreduction_external_comparisons"
EVALUATION_FOLDER="$THESIS_HOME_VM1/dimreduction_evaluation"

# VM1 get last updates
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
MONITORING_FOLDER="monitoring_plots/20250914_mutiples_runs"
# mkdir -p $MONITORING_FOLDER
# cp -r -t $MONITORING_FOLDER monitoring_plots/20250912_190348 monitoring_plots/20250912_200940
mkdir -p $MONITORING_FOLDER/matlab/results

python scripts/geneci_evaluate.py \
    --monitoring-dir "$MONITORING_FOLDER" \
    --external-projects "$GENECI_FOLDER" \
    --threshold 0.000000001 \
    --output $MONITORING_FOLDER/evaluation_results_confident-in-0.csv \
    --output-dir $MONITORING_FOLDER/ \
    --skip-binarize

# CSV list to TSV list (DimReduction)
printf -v first_file '%s' $MONITORING_FOLDER/*-final_weight_data_converted.txt || exit 1;
python scripts/csv_2_tsv_net3.py \
    "$first_file" \
    $MONITORING_FOLDER/matlab/
# already executed - END


# CSV list to TSV list (Geneci)
# List of methods
methods=("GENIE3_ET" "GENIE3_RF" "KBOOST")

for method in "${methods[@]}"; do
    # method_lower=$(echo "$method" | tr '[:upper:]' '[:lower:]')
    printf -v first_file '%s' $MONITORING_FOLDER/GRN_$method* || exit 1;
    python scripts/csv_2_tsv_net3.py \
        "$first_file" \
        "$MONITORING_FOLDER/matlab/"
done


MATLAB_PREDICTIONS_FULL_PATH=$(realpath $MONITORING_FOLDER/matlab/)

cd $EVALUATION_FOLDER/matlab || exit 1;

sudo matlab -nodesktop -nosplash -r "go_my('DimReduction', 64, '${MATLAB_PREDICTIONS_FULL_PATH}'); exit;" && \

for method in "${methods[@]}"; do
    sudo matlab -nodesktop -nosplash -r "go_my('${method}', 64, '${MATLAB_PREDICTIONS_FULL_PATH}'); exit;"
done

cp results/* $MATLAB_PREDICTIONS_FULL_PATH/results

cd - || exit 1;

mkdir -p $MONITORING_FOLDER/graphs
python scripts/evaluation_analyzer_2.py \
    --time_file $MONITORING_FOLDER/evaluation_results_confident-in-0.csv \
    --data_folder $MONITORING_FOLDER/matlab/results \
    --threads 1 \
    --output_dir $MONITORING_FOLDER/graphs