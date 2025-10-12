#!/bin/bash

set -e

THESIS_HOME_VM1="$HOME/virt_machine"

DIMREDUCTION_JAVA_FOLDER="$THESIS_HOME_VM1/dimreduction-java"
DIMREDUCTION_PYTHON_FOLDER="$THESIS_HOME_VM1/dimreduction-python"
GENECI_FOLDER="$THESIS_HOME_VM1/dimreduction_external_comparisons"
EVALUATION_FOLDER="$THESIS_HOME_VM1/dimreduction_evaluation"

# Initialize skip flags
SKIP_GIT_PULL=false
SKIP_GENECI_EVALUATE_VM1=false
SKIP_GENECI_EVALUATE_VM2=false
SKIP_CSV_2_TSV_DIMREDUCTION_VM1=false
SKIP_CSV_2_TSV_GENECI_VM1=false
SKIP_CSV_2_TSV_DIMREDUCTION_VM2=false
SKIP_CSV_2_TSV_GENECI_VM2=false
SKIP_MATLAB_VM1=false
SKIP_MATLAB_VM2=false
SKIP_EVALUATION_ANALYZER=false

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-git-pull)
            SKIP_GIT_PULL=true
            shift
            ;;
        --skip-geneci_evaluate-vm1)
            SKIP_GENECI_EVALUATE_VM1=true
            shift
            ;;
        --skip-geneci_evaluate-vm2)
            SKIP_GENECI_EVALUATE_VM2=true
            shift
            ;;
        --skip-csv_2_tsv_dimreduction-vm1)
            SKIP_CSV_2_TSV_DIMREDUCTION_VM1=true
            shift
            ;;
        --skip-csv_2_tsv_geneci-vm1)
            SKIP_CSV_2_TSV_GENECI_VM1=true
            shift
            ;;
        --skip-csv_2_tsv_dimreduction-vm2)
            SKIP_CSV_2_TSV_DIMREDUCTION_VM2=true
            shift
            ;;
        --skip-csv_2_tsv_geneci-vm2)
            SKIP_CSV_2_TSV_GENECI_VM2=true
            shift
            ;;
        --skip-matlab-vm1)
            SKIP_MATLAB_VM1=true
            shift
            ;;
        --skip-matlab-vm2)
            SKIP_MATLAB_VM2=true
            shift
            ;;
        --skip-evaluation_analyzer)
            SKIP_EVALUATION_ANALYZER=true
            shift
            ;;
        --output-dir-evaluation_analyzer)
            EVALUATION_ANALYZER=$2
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

# VM1 get last updates
if [ "$SKIP_GIT_PULL" = false ]; then
    cd "$DIMREDUCTION_JAVA_FOLDER"
    git pull
    git status

    cd "$GENECI_FOLDER"
    git pull
    git status

    cd "$DIMREDUCTION_PYTHON_FOLDER"
    git pull
    git status
fi

# Run VM1 Geneci (+ metrics)
MONITORING_FOLDER_VM2_1thread="monitoring_plots/20250912_mutiples_runs_From_VM2"
MONITORING_FOLDER_VM1_64threads="monitoring_plots/20250914_mutiples_runs"
mkdir -p $MONITORING_FOLDER_VM1_64threads/matlab/results
EVALUATION_ANALYZER=${EVALUATION_ANALYZER:-$MONITORING_FOLDER_VM1_64threads/graphs}

if [ "$SKIP_GENECI_EVALUATE_VM1" = false ]; then
    python scripts/geneci_evaluate.py \
        --monitoring-dir "$MONITORING_FOLDER_VM1_64threads" \
        --external-projects "$GENECI_FOLDER" \
        --threshold 0.000000001 \
        --output $MONITORING_FOLDER_VM1_64threads/evaluation_results_confident-in-0.csv \
        --output-dir $MONITORING_FOLDER_VM1_64threads/ \
        --skip-binarize
fi

if [ "$SKIP_GENECI_EVALUATE_VM2" = false ]; then
    python scripts/geneci_evaluate.py \
        --monitoring-dir "$MONITORING_FOLDER_VM2_1thread" \
        --external-projects "$GENECI_FOLDER" \
        --threshold 0.000000001 \
        --output $MONITORING_FOLDER_VM2_1thread/evaluation_results_confident-in-0.csv \
        --output-dir $MONITORING_FOLDER_VM2_1thread/ \
        --skip-binarize
fi

# CSV list to TSV list (DimReduction)
if [ "$SKIP_CSV_2_TSV_DIMREDUCTION_VM1" = false ]; then
    printf -v first_file '%s' $MONITORING_FOLDER_VM1_64threads/*-final_weight_data_converted.txt || exit 1;
    python scripts/csv_2_tsv_net3.py \
        "$first_file" \
        $MONITORING_FOLDER_VM1_64threads/matlab/
fi

# CSV list to TSV list (Geneci)
if [ "$SKIP_CSV_2_TSV_GENECI_VM1" = false ]; then
    methods=("GENIE3_ET" "GENIE3_RF" "KBOOST")
    for method in "${methods[@]}"; do
        printf -v first_file '%s' $MONITORING_FOLDER_VM1_64threads/GRN_$method* || exit 1;
        python scripts/csv_2_tsv_net3.py \
            "$first_file" \
            "$MONITORING_FOLDER_VM1_64threads/matlab/"
    done
fi

# CSV list to TSV list (DimReduction)
if [ "$SKIP_CSV_2_TSV_DIMREDUCTION_VM2" = false ]; then
    printf -v first_file '%s' $MONITORING_FOLDER_VM2_1thread/*-final_weight_data_converted.txt || exit 1;
    python scripts/csv_2_tsv_net3.py \
        "$first_file" \
        $MONITORING_FOLDER_VM2_1thread/matlab/
fi

# CSV list to TSV list (Geneci)
if [ "$SKIP_CSV_2_TSV_GENECI_VM2" = false ]; then
    methods=("ARACNE" "CLR" "GENIE3_ET" "GENIE3_RF" "BC3NET" "C3NET" "KBOOST" "MRNETB" "MRNET" "PCIT")
    for method in "${methods[@]}"; do
        printf -v first_file '%s' $MONITORING_FOLDER_VM2_1thread/GRN_$method* || exit 1;
        python scripts/csv_2_tsv_net3.py \
            "$first_file" \
            "$MONITORING_FOLDER_VM2_1thread/matlab/"
    done
fi

if [ "$SKIP_MATLAB_VM1" = false ]; then
    MATLAB_PREDICTIONS_FULL_PATH_VM1_64threads=$(realpath $MONITORING_FOLDER_VM1_64threads/matlab/)
    cd $EVALUATION_FOLDER/matlab || exit 1;
    sudo matlab -nodesktop -nosplash -r "go_my('DimReduction', 64, '${MATLAB_PREDICTIONS_FULL_PATH_VM1_64threads}'); exit;" && \
    for method in "${methods[@]}"; do
        sudo matlab -nodesktop -nosplash -r "go_my('${method}', 64, '${MATLAB_PREDICTIONS_FULL_PATH_VM1_64threads}'); exit;"
    done
    cp results/* $MATLAB_PREDICTIONS_FULL_PATH_VM1_64threads/results
    cd - || exit 1;
fi

if [ "$SKIP_MATLAB_VM2" = false ]; then
    MATLAB_PREDICTIONS_FULL_PATH_VM2_1thread=$(realpath $MONITORING_FOLDER_VM2_1thread/matlab/)
    cd $EVALUATION_FOLDER/matlab || exit 1;
    sudo matlab -nodesktop -nosplash -r "go_my('DimReduction', 1, '${MATLAB_PREDICTIONS_FULL_PATH_VM2_1thread}'); exit;" && \
    for method in "${methods[@]}"; do
        sudo matlab -nodesktop -nosplash -r "go_my('${method}', 1, '${MATLAB_PREDICTIONS_FULL_PATH_VM2_1thread}'); exit;"
    done
    cp results/* $MATLAB_PREDICTIONS_FULL_PATH_VM2_1thread/results
    cd - || exit 1;
fi

if [ "$SKIP_EVALUATION_ANALYZER" = false ]; then
    mkdir -p "$EVALUATION_ANALYZER"
    python scripts/evaluation_analyzer_2.py \
        --time_file $MONITORING_FOLDER_VM2_1thread/evaluation_results_confident-in-0.csv MONITORING_FOLDER_VM1_64threadsevaluation_results_confident-in-0.csv \
        --data_folder $MONITORING_FOLDER_VM2_1thread/matlab/results $MONITORING_FOLDER_VM1_64threads/matlab/results \
        --threads 1 64 \
        --output_dir "$EVALUATION_ANALYZER"
fi