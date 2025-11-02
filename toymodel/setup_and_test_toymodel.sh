#!/bin/sh

# Script to set up the toy model environment, run the project, and validate results

set -e

# Ask user about results folder handling
echo "Choose what to do with the results/ folder before running the test:"
echo "1) Clear results/"
echo "2) Move current results/ to results/old/"
echo "3) Do nothing"
read "user_choice?Enter 1, 2, or 3: "

case $user_choice in
  1)
    echo "[Toymodel] Clearing results/..."
    rm -rf results/final_data/*.txt results/quantized_data/*.txt
    ;;
  2)
    echo "[Toymodel] Moving results/ to results/old/..."
    mkdir -p results/old/final_data results/old/quantized_data
    mv results/final_data/*.txt results/old/final_data/ 2>/dev/null || true
    mv results/quantized_data/*.txt results/old/quantized_data/ 2>/dev/null || true
    ;;
  3)
    echo "[Toymodel] Leaving results/ unchanged."
    ;;
  *)
    echo "[Toymodel] Invalid option. Leaving results/ unchanged."
    ;;
esac

# 1. Copy the example .env file from toymodel to dimreduction-java as .env (overwrite if exists)
cp -b toymodel/.env.toymodel.example dimreduction-java/.env

echo "[Toymodel] .env file copied from toymodel/.env.toymodel.example to dimreduction-java/.env."

# 2. Ensure toy dataset is present in toymodel/
if [ ! -f toymodel/processed_dataset_dream5_40.csv ]; then
  cp inputs_files/datasets/processed_dataset_dream5_40.csv toymodel/
  echo "[Toymodel] Dataset copied to toymodel/ folder."
else
  echo "[Toymodel] Dataset already present."
fi

# 3. Run the project (assumes run.sh uses .env and DATASET_PATH)
cd dimreduction-java
bash run.sh
run_status=$?
cd ..

if [ $run_status -ne 0 ]; then
  echo "[ERROR] Project run failed. Aborting validation."
  exit 1
fi

echo "[Toymodel] Project run complete."

# 4. Compare outputs to references
# Find the most recent output files
last_final=$(ls -t results/final_data/*.txt 2>/dev/null | head -1)
last_quant=$(ls -t results/quantized_data/*.txt 2>/dev/null | head -1)

if [ ! -f "$last_final" ]; then
  echo "[ERROR] No final_data output file found. Validation aborted."
  exit 1
fi

if [ ! -f "$last_quant" ]; then
  echo "[ERROR] No quantized_data output file found. Validation aborted."
  exit 1
fi

# Compare final data
echo "\n[Validation] Comparing final data..."
diff "$last_final" toymodel/40-gui-final_data.txt && echo "[OK] Final data matches reference." || echo "[FAIL] Final data does not match reference."

# Compare quantized data
echo "\n[Validation] Comparing quantized data..."
diff "$last_quant" toymodel/40-gui-quantized_data.txt && echo "[OK] Quantized data matches reference." || echo "[FAIL] Quantized data does not match reference."

echo "\n[Toymodel] Setup, run, and validation complete."
