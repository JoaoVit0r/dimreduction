#!/bin/bash

set -e

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

# ./run_all_monitoring.sh --sleep-time 5 --sleep-time-monitor 5 --number-of-executions 1 --thread-distribution none --threads 1 --custom-input-file ../writing/output/processed_dataset_dream5_40.csv --python-files main_from_cli.py venv_v12 venv_14t venv_pypy venv_13t



# Run VM2 R
# sed -i "s/^INPUT_FILE=.*/INPUT_FILE=..\/..\/..\/final-delivery\/dimreduction\/inputs_files\/quantized_data\/40-gui-quantized_data.txt/g" ../test_external_code/try_minet/.env;
# ./run_all_monitoring.sh --sleep-time 5 --sleep-time-monitor 5 --number-of-executions 1 --thread-distribution none --threads 4 --custom-input-file ../../writing/output/processed_dataset_dream5_40.csv --r-files run_genie3.R --repository-r ../test_external_code/try_minet Rscript
# ./run_all_monitoring.sh --sleep-time 5 --sleep-time-monitor 5 --number-of-executions 1 \
#     --thread-distribution none --threads 1 \
#     --repository-r ../dimreduction_external_comparisons \
#     --custom-quantization-input-file ../../saving_files/reference/quantized_data/40-gui-quantized_data.txt \
#     --custom-input-file ../../writing/output/processed_dataset_dream5_40.csv \
#     --r-files run_clr.R,run_aracne.R,run_genie3.R \
#     Rscript


# Run Local R
# ./run_all_monitoring.sh --sleep-time 5 --sleep-time-monitor 5 --number-of-executions 1 \
#     --thread-distribution none --threads 4 \
#     --repository-r ../test_external_code/try_minet \
#     --custom-quantization-input-file ../../final-delivery/dimreduction/inputs_files/quantized_data/40-gui-quantized_data.txt \
#     --custom-input-file ../../writing/output/processed_dataset_dream5_40.csv \
#     --r-files run_genie3.R \
#     Rscript

# ./run_all_monitoring.sh --sleep-time 5 --sleep-time-monitor 5 --number-of-executions 1 \
#     --thread-distribution none --threads 4 \
#     --repository-r ../test_external_code/try_minet \
#     --custom-quantization-input-file ../../final-delivery/dimreduction/inputs_files/quantized_data/40-gui-quantized_data.txt \
#     --custom-input-file ../../writing/output/processed_dataset_dream5_40.csv \
#     --r-files run_clr.R,run_aracne.R,run_genie3.R \
#     Rscript


# Run Local Geneci
# ./run_all_monitoring.sh --sleep-time 5 --sleep-time-monitor 5 --number-of-executions 1 \
#     --thread-distribution none --threads 1 \
#     --repository-geneci ../test_external_code/try_minet \
#     --custom-input-file input_data/geneci/DREAM4/EXP/dream4_100_01_exp.csv \
#     --geneci-files run_geneci_aracne.sh,run_geneci_clr.sh,run_geneci_genie3-et.sh,run_geneci_genie3-rf.sh \
#     geneci;

# Run VM2 GeneciGeneci
# ./run_all_monitoring.sh --sleep-time 5 --number-of-executions 1 \
#     --thread-distribution none --threads 1 \
#     --repository-geneci ../dimreduction_external_comparisons \
#     --custom-input-file input_data/geneci/DREAM4/EXP/dream4_100_01_exp.csv \
#     --geneci-files run_geneci_aracne.sh,run_geneci_clr.sh,run_geneci_genie3-et.sh,run_geneci_genie3-rf.sh \
#     geneci;

# # run again, clr script was not pushed
# ./run_all_monitoring.sh --sleep-time 5 --number-of-executions 1 \
#     --thread-distribution none --threads 1 \
#     --repository-geneci ../dimreduction_external_comparisons \
#     --custom-input-file input_data/geneci/DREAM4/EXP/dream4_100_01_exp.csv \
#     --geneci-files run_geneci_clr.sh \
#     geneci;

# # Run Local FIX ENV TO DREAM4
# cd $THESIS_HOME/virt_machine/java-dimreduction
# sed -i "s/^ARE_COLUMNS_DESCRIPTIVE=.*/ARE_COLUMNS_DESCRIPTIVE=false/g" .env;
# sed -i "s/^ARE_TITLES_ON_FIRST_COLUMN=.*/ARE_TITLES_ON_FIRST_COLUMN=true/g" .env;
# sed -i "s/^TRANSPOSE_MATRIX=.*/TRANSPOSE_MATRIX=false/g" .env;
# sed -i "s/^SAVE_FINAL_DATA=.*/SAVE_FINAL_DATA=true/g" .env;
# sed -i "s/^SAVE_FINAL_WEIGHT_DATA=.*/SAVE_FINAL_WEIGHT_DATA=true/g" .env;
# sed -i "s/^THRESHOLD=.*/THRESHOLD=1.0/g" .env;

# cd $THESIS_HOME/dimreduction
# sed -i "s/^ARE_COLUMNS_DESCRIPTIVE=.*/ARE_COLUMNS_DESCRIPTIVE=false/g" .env;
# sed -i "s/^ARE_TITLES_ON_FIRST_COLUMN=.*/ARE_TITLES_ON_FIRST_COLUMN=true/g" .env;
# sed -i "s/^TRANSPOSE_MATRIX=.*/TRANSPOSE_MATRIX=false/g" .env;

# FORMATTED_INPUT_FILE=$THESIS_HOME/test_external_code/try_minet/input_data/geneci/DREAM4/EXP_DimReduction/dream4_100_01_exp.csv

# files=($(ls -la "$THESIS_HOME/test_external_code/try_minet/input_data/geneci/DREAM4/EXP/" | grep "dream4_100" | awk '{print $9}'))
# for ORIGINAL_INPUT_FILE in "${files[@]}"; do
# # for ORIGINAL_INPUT_FILE in "$THESIS_HOME/test_external_code/try_minet/input_data/geneci/DREAM4/EXP/dream4_100"*; do
#     ORIGINAL_INPUT_FILE="$THESIS_HOME/test_external_code/try_minet/input_data/geneci/DREAM4/EXP/"$(basename "$ORIGINAL_INPUT_FILE")
#     FORMATTED_INPUT_FILE="$THESIS_HOME/test_external_code/try_minet/input_data/geneci/DREAM4/EXP_DimReduction/"$(basename "$ORIGINAL_INPUT_FILE")

#     echo FORMATTED_INPUT_FILE: $FORMATTED_INPUT_FILE
#     echo ORIGINAL_INPUT_FILE: $ORIGINAL_INPUT_FILE

#     # Run Local JAVA DREAM4
#     ./run_all_monitoring.sh \
#         --sleep-time 5 \
#         --sleep-time-monitor 5 \
#         --number-of-executions 1 \
#         --thread-distribution none \
#         --threads 1 \
#         --repository-java $THESIS_HOME/virt_machine/java-dimreduction \
#         --custom-input-file $FORMATTED_INPUT_FILE java

#     # Run Local Geneci
#     ./run_all_monitoring.sh --sleep-time 5 --number-of-executions 1 \
#         --sleep-time-monitor 5 \
#         --thread-distribution none --threads 1 \
#         --repository-geneci $THESIS_HOME/test_external_code/try_minet \
#         --custom-input-file $ORIGINAL_INPUT_FILE \
#         --geneci-files run_geneci_aracne.sh,run_geneci_clr.sh \
#         geneci;
#     # # Run Local Python DREAM4
#     # ./run_all_monitoring.sh --sleep-time 5 --sleep-time-monitor 5 --number-of-executions 1 --thread-distribution none --threads 1 --custom-input-file $FORMATTED_INPUT_FILE --python-files main_from_cli.py venv_v12 venv_v13-nogil
# done


# # Run VM2 FIX ENV TO DREAM4
# cd ~/workspace/dimreduction-java
# sed -i "s/^ARE_COLUMNS_DESCRIPTIVE=.*/ARE_COLUMNS_DESCRIPTIVE=false/g" .env;
# sed -i "s/^ARE_TITLES_ON_FIRST_COLUMN=.*/ARE_TITLES_ON_FIRST_COLUMN=true/g" .env;
# sed -i "s/^TRANSPOSE_MATRIX=.*/TRANSPOSE_MATRIX=false/g" .env;
# sed -i "s/^SAVE_FINAL_DATA=.*/SAVE_FINAL_DATA=true/g" .env;
# sed -i "s/^SAVE_FINAL_WEIGHT_DATA=.*/SAVE_FINAL_WEIGHT_DATA=true/g" .env;
# sed -i "s/^THRESHOLD=.*/THRESHOLD=1.0/g" .env;
# sed -i "s/^VERBOSITY_LEVEL=.*/VERBOSITY_LEVEL=0/g" .env;

# cd ~/workspace/dimreduction-python
# sed -i "s/^ARE_COLUMNS_DESCRIPTIVE=.*/ARE_COLUMNS_DESCRIPTIVE=false/g" .env;
# sed -i "s/^ARE_TITLES_ON_FIRST_COLUMN=.*/ARE_TITLES_ON_FIRST_COLUMN=true/g" .env;
# sed -i "s/^TRANSPOSE_MATRIX=.*/TRANSPOSE_MATRIX=false/g" .env;

# THESIS_HOME="$HOME/workspace"
# FORMATTED_INPUT_FILE=$THESIS_HOME/dimreduction_external_comparisons/input_data/geneci/DREAM4/EXP_DimReduction/dream4_100_01_exp.csv

# files=($(ls -la "$THESIS_HOME/dimreduction_external_comparisons/input_data/geneci/DREAM4/EXP/" | grep "dream4_100" | awk '{print $9}'))
# for ORIGINAL_INPUT_FILE in "${files[@]}"; do
# # for ORIGINAL_INPUT_FILE in "$THESIS_HOME/dimreduction_external_comparisons/input_data/geneci/DREAM4/EXP/dream4_100"*; do
#     ORIGINAL_INPUT_FILE="$THESIS_HOME/dimreduction_external_comparisons/input_data/geneci/DREAM4/EXP/"$(basename "$ORIGINAL_INPUT_FILE")
#     FORMATTED_INPUT_FILE="$THESIS_HOME/dimreduction_external_comparisons/input_data/geneci/DREAM4/EXP_DimReduction/"$(basename "$ORIGINAL_INPUT_FILE")

#     echo FORMATTED_INPUT_FILE: $FORMATTED_INPUT_FILE
#     echo ORIGINAL_INPUT_FILE: $ORIGINAL_INPUT_FILE
#     # Run VM2 JAVA DREAM4
#     ./run_all_monitoring.sh --sleep-time 5 --number-of-executions 1 \
#         --sleep-time-monitor 5 \
#         --thread-distribution none --threads 1 \
#         --repository-python $HOME/workspace/dimreduction-python \
#         --repository-java $HOME/workspace/dimreduction-java \
#         --custom-input-file $FORMATTED_INPUT_FILE \
#         java;

#     # Run VM2 GeneciGeneci
#     ./run_all_monitoring.sh --sleep-time 5 --number-of-executions 1 \
#         --sleep-time-monitor 5 \
#         --thread-distribution none --threads 1 \
#         --repository-python $HOME/workspace/dimreduction-python \
#         --repository-geneci $HOME/workspace/dimreduction_external_comparisons \
#         --custom-input-file $ORIGINAL_INPUT_FILE \
#         --geneci-files run_geneci_aracne.sh,run_geneci_clr.sh,run_geneci_genie3-et.sh,run_geneci_genie3-rf.sh \
#         geneci;

#     # # Run VM2 Python DREAM4
#     # ./run_all_monitoring.sh --sleep-time 5 --number-of-executions 1 --thread-distribution none --threads 1 --custom-input-file $FORMATTED_INPUT_FILE --python-files main_from_cli.py venv_pypy venv_v12 venv_14t venv_13t
# done



# # Run Local FIX ENV TO DREAM5-from-geneci-data
# cd $THESIS_HOME/virt_machine/java-dimreduction
# sed -i "s/^ARE_COLUMNS_DESCRIPTIVE=.*/ARE_COLUMNS_DESCRIPTIVE=false/g" .env;
# sed -i "s/^ARE_TITLES_ON_FIRST_COLUMN=.*/ARE_TITLES_ON_FIRST_COLUMN=true/g" .env;
# sed -i "s/^TRANSPOSE_MATRIX=.*/TRANSPOSE_MATRIX=false/g" .env;
# sed -i "s/^SAVE_FINAL_DATA=.*/SAVE_FINAL_DATA=true/g" .env;
# sed -i "s/^SAVE_FINAL_WEIGHT_DATA=.*/SAVE_FINAL_WEIGHT_DATA=true/g" .env;
# sed -i "s/^THRESHOLD=.*/THRESHOLD=1.0/g" .env;
# sed -i "s/^VERBOSITY_LEVEL=.*/VERBOSITY_LEVEL=0/g" .env;

# cd $THESIS_HOME/dimreduction
# sed -i "s/^ARE_COLUMNS_DESCRIPTIVE=.*/ARE_COLUMNS_DESCRIPTIVE=false/g" .env;
# sed -i "s/^ARE_TITLES_ON_FIRST_COLUMN=.*/ARE_TITLES_ON_FIRST_COLUMN=true/g" .env;
# sed -i "s/^TRANSPOSE_MATRIX=.*/TRANSPOSE_MATRIX=false/g" .env;

# FORMATTED_INPUT_FILE=$THESIS_HOME/test_external_code/try_minet/input_data/geneci/DREAM5/EXP_DimReduction/net3_exp.csv

# files=($(ls -la "$THESIS_HOME/test_external_code/try_minet/input_data/geneci/DREAM5/EXP/" | grep "net3" | awk '{print $9}'))
# for ORIGINAL_INPUT_FILE in "${files[@]}"; do
# # for ORIGINAL_INPUT_FILE in "$THESIS_HOME/test_external_code/try_minet/input_data/geneci/DREAM5/EXP/dream5_100"*; do
#     ORIGINAL_INPUT_FILE="$THESIS_HOME/test_external_code/try_minet/input_data/geneci/DREAM5/EXP/"$(basename "$ORIGINAL_INPUT_FILE")
#     FORMATTED_INPUT_FILE="$THESIS_HOME/test_external_code/try_minet/input_data/geneci/DREAM5/EXP_DimReduction/"$(basename "$ORIGINAL_INPUT_FILE")

#     echo FORMATTED_INPUT_FILE: $FORMATTED_INPUT_FILE
#     echo ORIGINAL_INPUT_FILE: $ORIGINAL_INPUT_FILE

#     # # Run Local JAVA DREAM5-from-geneci-data
#     # ./run_all_monitoring.sh \
#     #     --sleep-time 5 \
#     #     --sleep-time-monitor 5 \
#     #     --number-of-executions 1 \
#     #     --thread-distribution none \
#     #     --threads 1 \
#     #     --repository-java $THESIS_HOME/virt_machine/java-dimreduction \
#     #     --custom-input-file $FORMATTED_INPUT_FILE \
#     #     java

#     # Run Local Geneci DREAM5-from-geneci-data
#     ./run_all_monitoring.sh --sleep-time 5 \
#         --number-of-executions 1 \
#         --sleep-time-monitor 5 \
#         --thread-distribution none \
#         --threads 1 \
#         --repository-geneci $THESIS_HOME/test_external_code/try_minet \
#         --custom-input-file $ORIGINAL_INPUT_FILE \
#         --geneci-files dream5_scripts/run_geneci_aracne.sh,dream5_scripts/run_geneci_clr.sh \
#         geneci;
#     # # Run Local Python DREAM4
#     # ./run_all_monitoring.sh --sleep-time 5 --sleep-time-monitor 5 --number-of-executions 1 --thread-distribution none --threads 1 --custom-input-file $FORMATTED_INPUT_FILE --python-files main_from_cli.py venv_v12 venv_v13-nogil
# done



# Run VM2 FIX ENV TO DREAM5-from-geneci-data
cd ~/workspace/dimreduction-java
sed -i "s/^ARE_COLUMNS_DESCRIPTIVE=.*/ARE_COLUMNS_DESCRIPTIVE=false/g" .env;
sed -i "s/^ARE_TITLES_ON_FIRST_COLUMN=.*/ARE_TITLES_ON_FIRST_COLUMN=true/g" .env;
sed -i "s/^TRANSPOSE_MATRIX=.*/TRANSPOSE_MATRIX=false/g" .env;
sed -i "s/^SAVE_FINAL_DATA=.*/SAVE_FINAL_DATA=true/g" .env;
sed -i "s/^SAVE_FINAL_WEIGHT_DATA=.*/SAVE_FINAL_WEIGHT_DATA=true/g" .env;
sed -i "s/^THRESHOLD=.*/THRESHOLD=1.0/g" .env;
sed -i "s/^VERBOSITY_LEVEL=.*/VERBOSITY_LEVEL=0/g" .env;

cd ~/workspace/dimreduction-python
sed -i "s/^ARE_COLUMNS_DESCRIPTIVE=.*/ARE_COLUMNS_DESCRIPTIVE=false/g" .env;
sed -i "s/^ARE_TITLES_ON_FIRST_COLUMN=.*/ARE_TITLES_ON_FIRST_COLUMN=true/g" .env;
sed -i "s/^TRANSPOSE_MATRIX=.*/TRANSPOSE_MATRIX=false/g" .env;

THESIS_HOME="$HOME/workspace"
FORMATTED_INPUT_FILE=$THESIS_HOME/dimreduction_external_comparisons/input_data/geneci/DREAM5/EXP_DimReduction/net3_exp.csv

files=($(ls -la "$THESIS_HOME/dimreduction_external_comparisons/input_data/geneci/DREAM5/EXP/" | grep "net3" | awk '{print $9}'))
for ORIGINAL_INPUT_FILE in "${files[@]}"; do
# for ORIGINAL_INPUT_FILE in "$THESIS_HOME/dimreduction_external_comparisons/input_data/geneci/DREAM5/EXP/net3"*; do
    ORIGINAL_INPUT_FILE="$THESIS_HOME/dimreduction_external_comparisons/input_data/geneci/DREAM5/EXP/"$(basename "$ORIGINAL_INPUT_FILE")
    FORMATTED_INPUT_FILE="$THESIS_HOME/dimreduction_external_comparisons/input_data/geneci/DREAM5/EXP_DimReduction/"$(basename "$ORIGINAL_INPUT_FILE")

    echo FORMATTED_INPUT_FILE: $FORMATTED_INPUT_FILE
    echo ORIGINAL_INPUT_FILE: $ORIGINAL_INPUT_FILE
    # Run VM2 JAVA DREAM4
    ./run_all_monitoring.sh --sleep-time 5 \
        --number-of-executions 1 \
        --sleep-time-monitor 5 \
        --thread-distribution none \
        --threads 1 \
        --repository-python $HOME/workspace/dimreduction-python \
        --repository-java $HOME/workspace/dimreduction-java \
        --custom-input-file $FORMATTED_INPUT_FILE \
        java;

    # Run VM2 GeneciGeneci
    ./run_all_monitoring.sh --sleep-time 5 \
        --number-of-executions 1 \
        --sleep-time-monitor 5 \
        --thread-distribution none \
        --threads 1 \
        --repository-python $HOME/workspace/dimreduction-python \
        --repository-geneci $HOME/workspace/dimreduction_external_comparisons \
        --custom-input-file $ORIGINAL_INPUT_FILE \
        --geneci-files dream5_scripts/run_geneci_aracne.sh,dream5_scripts/run_geneci_clr.sh,dream5_scripts/run_geneci_genie3-et.sh,dream5_scripts/run_geneci_genie3-rf.sh \
        geneci;

    # more
    ./run_all_monitoring.sh --sleep-time 5 \
        --number-of-executions 1 \
        --sleep-time-monitor 5 \
        --thread-distribution none \
        --threads 1 \
        --repository-python $HOME/workspace/dimreduction-python \
        --repository-geneci $HOME/workspace/dimreduction_external_comparisons \
        --custom-input-file $ORIGINAL_INPUT_FILE \
        --geneci-files dream5_scripts/run_geneci_jump3.sh,dream5_scripts/run_geneci_mrnet.sh,dream5_scripts/run_geneci_plsnet.sh,dream5_scripts/run_geneci_kboost.sh,dream5_scripts/run_geneci_narromi.sh,dream5_scripts/run_geneci_puc.sh,dream5_scripts/run_geneci_bc3net.sh,dream5_scripts/run_geneci_leap.sh,dream5_scripts/run_geneci_nonlinearodes.sh,dream5_scripts/run_geneci_rsnet.sh,dream5_scripts/run_geneci_c3net.sh,dream5_scripts/run_geneci_grnboost2.sh,dream5_scripts/run_geneci_locpcacmi.sh,dream5_scripts/run_geneci_pcacmi.sh,dream5_scripts/run_geneci_tigress.sh,dream5_scripts/run_geneci_grnvbem.sh,dream5_scripts/run_geneci_meomi.sh,dream5_scripts/run_geneci_pcit.sh,dream5_scripts/run_geneci_cmi2ni.sh,dream5_scripts/run_geneci_inferelator.sh,dream5_scripts/run_geneci_mrnetb.sh,dream5_scripts/run_geneci_pidc.sh \
        geneci;

    # # Run VM2 Python DREAM4
    # ./run_all_monitoring.sh --sleep-time 5 --number-of-executions 1 --thread-distribution none --threads 1 --custom-input-file $FORMATTED_INPUT_FILE --python-files main_from_cli.py venv_pypy venv_v12 venv_14t venv_13t
done