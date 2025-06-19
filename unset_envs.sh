#!/bin/bash

# Change this to your .env file if needed
ENV_FILE=".env"

echo "To unsetting environment variables defined in $ENV_FILE, run the following commands:"

# Unset all variables defined in the .env file
grep -v '^\s*#' "$ENV_FILE" | grep -E '^[A-Za-z_][A-Za-z0-9_]*=' | cut -d= -f1 | while read var; do
    echo "unset $var"
done
