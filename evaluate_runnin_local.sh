#!/bin/bash

set -e

THESIS_HOME_LOCAL="$THESIS_HOME"

# DIMREDUCTION_JAVA_FOLDER="$THESIS_HOME_LOCAL/virt_machine/java-dimreduction"
# DIMREDUCTION_PYTHON_FOLDER="$THESIS_HOME_LOCAL/dimreduction"
GENECI_FOLDER="$THESIS_HOME_LOCAL/test_external_code/try_minet"


# Run LOCAL Geneci (+ metrics)
MONITORING_FOLDER="test_evaluation_multi_mix_local_vm2"
# mkdir -p $MONITORING_FOLDER
# cp -r -t $MONITORING_FOLDER monitoring_plots/20250912_190348 monitoring_plots/20250912_200940

python scripts/geneci_evaluate.py \
    --monitoring-dir "$MONITORING_FOLDER" \
    --external-projects "$GENECI_FOLDER" \
    --threshold 0.000001 \
    --output $MONITORING_FOLDER/evaluation_results_confident-in-0.csv \
    --output-dir $MONITORING_FOLDER/

python scripts/geneci_evaluate.py \
    --monitoring-dir "$MONITORING_FOLDER" \
    --external-projects "$GENECI_FOLDER" \
    --threshold 0.7 \
    --output $MONITORING_FOLDER/evaluation_results_confident-in-70.csv \
    --output-dir $MONITORING_FOLDER/

python scripts/geneci_evaluate.py \
    --monitoring-dir "$MONITORING_FOLDER" \
    --external-projects "$GENECI_FOLDER" \
    --threshold 0.4 \
    --output $MONITORING_FOLDER/evaluation_results_confident-in-40.csv \
    --output-dir $MONITORING_FOLDER/

python scripts/geneci_evaluate.py \
    --monitoring-dir "$MONITORING_FOLDER" \
    --external-projects "$GENECI_FOLDER" \
    --threshold 0.3 \
    --output $MONITORING_FOLDER/evaluation_results_confident-in-30.csv \
    --output-dir $MONITORING_FOLDER/

python scripts/geneci_evaluate.py \
    --monitoring-dir "$MONITORING_FOLDER" \
    --external-projects "$GENECI_FOLDER" \
    --threshold 1.0 \
    --output $MONITORING_FOLDER/evaluation_results_confident-in-100.csv \
    --output-dir $MONITORING_FOLDER/
