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


# Run VM2 Geneci (+ metrics)
MONITORING_FOLDER="monitoring_plots/20250912_mutiples_runs"
# mkdir -p $MONITORING_FOLDER
# cp -r -t $MONITORING_FOLDER monitoring_plots/20250912_190348 monitoring_plots/20250912_200940
mkdir -p $MONITORING_FOLDER/matlab/results

python scripts/geneci_evaluate.py \
    --monitoring-dir "$MONITORING_FOLDER/2025*" \
    --external-projects "$GENECI_FOLDER" \
    --threshold 0.000000001 \
    --output $MONITORING_FOLDER/evaluation_results_confident-in-0.csv \
    --output-dir $MONITORING_FOLDER/ \
    --skip-binarize

# python scripts/geneci_evaluate.py \
#     --monitoring-dir "$MONITORING_FOLDER" \
#     --external-projects "$GENECI_FOLDER" \
#     --threshold 0.7 \
#     --output $MONITORING_FOLDER/evaluation_results_confident-in-70.csv \
#     --output-dir $MONITORING_FOLDER/

# python scripts/geneci_evaluate.py \
#     --monitoring-dir "$MONITORING_FOLDER" \
#     --external-projects "$GENECI_FOLDER" \
#     --threshold 0.4 \
#     --output $MONITORING_FOLDER/evaluation_results_confident-in-40.csv \
#     --output-dir $MONITORING_FOLDER/

# python scripts/geneci_evaluate.py \
#     --monitoring-dir "$MONITORING_FOLDER" \
#     --external-projects "$GENECI_FOLDER" \
#     --threshold 0.3 \
#     --output $MONITORING_FOLDER/evaluation_results_confident-in-30.csv \
#     --output-dir $MONITORING_FOLDER/

# python scripts/geneci_evaluate.py \
#     --monitoring-dir "$MONITORING_FOLDER" \
#     --external-projects "$GENECI_FOLDER" \
#     --threshold 1.0 \
#     --output $MONITORING_FOLDER/evaluation_results_confident-in-100.csv \
#     --output-dir $MONITORING_FOLDER/


ref_file=$(find $MONITORING_FOLDER/202509*/dataset_net3_exp/java/MainCLI/distribution_demain/threads_*/monitoring_plots -name "*-final_weight_data.txt" -type f | head -1)

# Check if file was found
if [[ -z "$ref_file" ]]; then
    echo "Error: No file found matching the pattern"
    exit 1
fi
dir_path=$(dirname "$ref_file")
base_name=$(basename "$ref_file" "-final_weight_data.txt")
DimReduction_list_file="$dir_path/${base_name}_list_with_nonTFpred.txt"
python scripts/dream5_converter.py \
    "$ref_file" \
    "$DimReduction_list_file" \
    --max-predictions 21000000

python scripts/csv_2_tsv_net3.py \
    "$DimReduction_list_file" \
    $MONITORING_FOLDER/matlab/

sudo matlab -nodesktop -nosplash -r "go_my('DimReduction'); exit;" && \

# python scripts/csv_2_tsv_net3.py \
#     $MONITORING_FOLDER/202509*/dataset_net3_exp/geneci/run_geneci_aracne/distribution_none/threads_1/monitoring_plots/GRN_ARACNE.csv \
#     $MONITORING_FOLDER/matlab/
# python scripts/csv_2_tsv_net3.py \
#     $MONITORING_FOLDER/202509*/dataset_net3_exp/geneci/run_geneci_clr/distribution_none/threads_1/monitoring_plots/GRN_CLR.csv \
#     $MONITORING_FOLDER/matlab/
# python scripts/csv_2_tsv_net3.py \
#     $MONITORING_FOLDER/202509*/dataset_net3_exp/geneci/run_geneci_genie3-et/distribution_none/threads_1/monitoring_plots/GRN_GENIE3_ET.csv \
#     $MONITORING_FOLDER/matlab/
# python scripts/csv_2_tsv_net3.py \
#     $MONITORING_FOLDER/202509*/dataset_net3_exp/geneci/run_geneci_genie3-rf/distribution_none/threads_1/monitoring_plots/GRN_GENIE3_RF.csv \
#     $MONITORING_FOLDER/matlab/
# python scripts/csv_2_tsv_net3.py \
#     $MONITORING_FOLDER/202509*/dataset_net3_exp/geneci/run_geneci_bc3net/distribution_none/threads_1/monitoring_plots/GRN_BC3NET.csv \
#     $MONITORING_FOLDER/matlab/
# python scripts/csv_2_tsv_net3.py \
#     $MONITORING_FOLDER/202509*/dataset_net3_exp/geneci/run_geneci_c3net/distribution_none/threads_1/monitoring_plots/GRN_C3NET.csv \
#     $MONITORING_FOLDER/matlab/
# python scripts/csv_2_tsv_net3.py \
#     $MONITORING_FOLDER/202509*/dataset_net3_exp/geneci/run_geneci_kboost/distribution_none/threads_1/monitoring_plots/GRN_KBOOST.csv \
#     $MONITORING_FOLDER/matlab/
# python scripts/csv_2_tsv_net3.py \
#     $MONITORING_FOLDER/202509*/dataset_net3_exp/geneci/run_geneci_mrnetb/distribution_none/threads_1/monitoring_plots/GRN_MRNETB.csv \
#     $MONITORING_FOLDER/matlab/
# python scripts/csv_2_tsv_net3.py \
#     $MONITORING_FOLDER/202509*/dataset_net3_exp/geneci/run_geneci_mrnet/distribution_none/threads_1/monitoring_plots/GRN_MRNET.csv \
#     $MONITORING_FOLDER/matlab/
# python scripts/csv_2_tsv_net3.py \
#     $MONITORING_FOLDER/202509*/dataset_net3_exp/geneci/run_geneci_pcit/distribution_none/threads_1/monitoring_plots/GRN_PCIT.csv \
#     $MONITORING_FOLDER/matlab/

# sudo matlab -nodesktop -nosplash -r "go_my('ARACNE'); exit;" && \
# sudo matlab -nodesktop -nosplash -r "go_my('GRN_CLR'); exit;" && \
# sudo matlab -nodesktop -nosplash -r "go_my('GRN_GENIE3_ET'); exit;" && \
# sudo matlab -nodesktop -nosplash -r "go_my('GRN_GENIE3_RF'); exit;" && \
# sudo matlab -nodesktop -nosplash -r "go_my('GRN_BC3NET'); exit;" && \
# sudo matlab -nodesktop -nosplash -r "go_my('GRN_C3NET'); exit;" && \
# sudo matlab -nodesktop -nosplash -r "go_my('GRN_KBOOST'); exit;" && \
# sudo matlab -nodesktop -nosplash -r "go_my('GRN_MRNETB'); exit;" && \
# sudo matlab -nodesktop -nosplash -r "go_my('GRN_MRNET'); exit;" && \
# sudo matlab -nodesktop -nosplash -r "go_my('GRN_PCIT'); exit;"

# List of methods
methods=("ARACNE" "CLR" "GENIE3_ET" "GENIE3_RF" "BC3NET" "C3NET" "KBOOST" "MRNETB" "MRNET" "PCIT")

for method in "${methods[@]}"; do
    method_lower=$(echo "$method" | tr '[:upper:]' '[:lower:]')
    python scripts/csv_2_tsv_net3.py \
        $MONITORING_FOLDER/202509*/dataset_net3_exp/geneci/run_geneci_$method_lower/distribution_none/threads_1/monitoring_plots/GRN_$method.csv \
        "$MONITORING_FOLDER/matlab/"
done

for method in "${methods[@]}"; do
    sudo matlab -nodesktop -nosplash -r "go_my('${method}'); exit;"
done

cp results/* $MONITORING_FOLDER/matlab/results

mkdir -p $MONITORING_FOLDER/graphs
python scripts/evaluation_analyzer_2.py \
    --time_file $MONITORING_FOLDER/evaluation_results_confident-in-0.csv \
    --data_folder $MONITORING_FOLDER/matlab/results \
    --threads 1 \
    --output_dir $MONITORING_FOLDER/graphs
    