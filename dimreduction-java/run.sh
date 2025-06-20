#!/bin/bash
# run.sh - Script to execute the dimreduction JAR

# Ensure we're in the project directory
cd "$(dirname "$0")"

# Run the JAR
java -jar out/jar/dimreduction.jar