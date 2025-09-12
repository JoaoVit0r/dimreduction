#!/bin/bash

set -e

THESIS_HOME_VM1="$HOME/virt_machine"

DIMREDUCTION_JAVA_FOLDER="$THESIS_HOME_VM1/dimreduction-java"
DIMREDUCTION_PYTHON_FOLDER="$THESIS_HOME_VM1/dimreduction-python"
GENECI_FOLDER="$THESIS_HOME_VM1/dimreduction_external_comparisons"

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

# Run VM1 FIX ENV TO DREAM5-from-geneci-data
cd "$DIMREDUCTION_JAVA_FOLDER"
sed -i "s/^ARE_COLUMNS_DESCRIPTIVE=.*/ARE_COLUMNS_DESCRIPTIVE=false/g" .env;
sed -i "s/^ARE_TITLES_ON_FIRST_COLUMN=.*/ARE_TITLES_ON_FIRST_COLUMN=true/g" .env;
sed -i "s/^TRANSPOSE_MATRIX=.*/TRANSPOSE_MATRIX=false/g" .env;
sed -i "s/^SAVE_FINAL_DATA=.*/SAVE_FINAL_DATA=true/g" .env;
sed -i "s/^SAVE_FINAL_WEIGHT_DATA=.*/SAVE_FINAL_WEIGHT_DATA=true/g" .env;
sed -i "s/^THRESHOLD=.*/THRESHOLD=1.0/g" .env;
sed -i "s/^VERBOSITY_LEVEL=.*/VERBOSITY_LEVEL=0/g" .env;

cd "$GENECI_FOLDER"
sed -i "s/^QUANTIZATION_INPUT_FILE_PATH=.*/QUANTIZATION_INPUT_FILE_PATH=/g" .env;

cd "$DIMREDUCTION_PYTHON_FOLDER"
sed -i "s/^ARE_COLUMNS_DESCRIPTIVE=.*/ARE_COLUMNS_DESCRIPTIVE=false/g" .env;
sed -i "s/^ARE_TITLES_ON_FIRST_COLUMN=.*/ARE_TITLES_ON_FIRST_COLUMN=true/g" .env;
sed -i "s/^TRANSPOSE_MATRIX=.*/TRANSPOSE_MATRIX=false/g" .env;

FORMATTED_INPUT_FILE=$GENECI_FOLDER/input_data/geneci/DREAM5/EXP_DimReduction/net3_exp.csv

files=($(ls -la "$GENECI_FOLDER/input_data/geneci/DREAM5/EXP/" | grep "net3" | awk '{print $9}'))
for ORIGINAL_INPUT_FILE in "${files[@]}"; do
    ORIGINAL_INPUT_FILE="$GENECI_FOLDER/input_data/geneci/DREAM5/EXP/"$(basename "$ORIGINAL_INPUT_FILE")
    FORMATTED_INPUT_FILE="$GENECI_FOLDER/input_data/geneci/DREAM5/EXP_DimReduction/"$(basename "$ORIGINAL_INPUT_FILE")

    echo FORMATTED_INPUT_FILE: "$FORMATTED_INPUT_FILE"
    echo ORIGINAL_INPUT_FILE: "$ORIGINAL_INPUT_FILE"
    # Run VM1 DimReduction JAVA (DREAM5 from GENECI)
    ./run_all_monitoring.sh --sleep-time 5 \
        --number-of-executions 1 \
        --thread-distribution demain \
        --threads 1 \
        --repository-python "$DIMREDUCTION_PYTHON_FOLDER" \
        --repository-java "$DIMREDUCTION_JAVA_FOLDER" \
        --custom-input-file "$FORMATTED_INPUT_FILE" \
        java;

    # Run VM1 GENECI (DREAM5 from GENECI)
    ./run_all_monitoring.sh --sleep-time 5 \
        --number-of-executions 1 \
        --thread-distribution none \
        --threads 1 \
        --repository-python "$DIMREDUCTION_PYTHON_FOLDER" \
        --repository-geneci "$GENECI_FOLDER" \
        --custom-input-file "$ORIGINAL_INPUT_FILE" \
        --geneci-files dream5_scripts/run_geneci_aracne.sh,dream5_scripts/run_geneci_clr.sh,dream5_scripts/run_geneci_genie3-et.sh,dream5_scripts/run_geneci_genie3-rf.sh \
        geneci;

    # more
    ./run_all_monitoring.sh --sleep-time 5 \
        --number-of-executions 1 \
        --thread-distribution none \
        --threads 1 \
        --repository-python "$DIMREDUCTION_PYTHON_FOLDER" \
        --repository-geneci "$GENECI_FOLDER" \
        --custom-input-file "$ORIGINAL_INPUT_FILE" \
        --geneci-files dream5_scripts/run_geneci_tigress.sh,dream5_scripts/run_geneci_mrnet.sh,dream5_scripts/run_geneci_bc3net.sh,dream5_scripts/run_geneci_c3net.sh,dream5_scripts/run_geneci_kboost.sh,dream5_scripts/run_geneci_mrnetb.sh,dream5_scripts/run_geneci_pcit.sh \
        geneci;

done
