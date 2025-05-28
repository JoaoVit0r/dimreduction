#!/bin/bash

NORMALIZE=false
HIGHLIGHT=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --normalize)
            NORMALIZE=true
            shift
            ;;
        --highlight)
            HIGHLIGHT=true
            shift
            ;;
        *)
            if [ -z "$COMMAND" ]; then
                COMMAND="$@"
                break
            fi
            ;;
    esac
done

if [ -z "$COMMAND" ]; then
    echo "Usage: $0 [--normalize] [--highlight] <command>"
    exit 1
fi

# Pattern to filter diff output if highlight is enabled
HIGHLIGHT_PATTERN="(/main_\\w+/|are identical|differ|full|threads_)"

# Helper function to normalize a file (convert all numbers to float with 6 decimals)
normalize_file() {
    awk '{for(i=1;i<=NF;i++){if($i ~ /^-?[0-9]+(\\.[0-9]+)?$/){printf "%.6f ", $i}else{printf $i" "}} print ""}' "$1"
}

# Parse positional arguments (expecting two patterns)
if [ -z "$COMMAND" ]; then
    if [ $# -lt 2 ]; then
        echo "Usage: $0 [--normalize] [--highlight] <from-file-pattern> <to-file-pattern>"
        exit 1
    fi
    FROM_PATTERN="$1"
    TO_PATTERN="$2"
else
    set -- $COMMAND
    FROM_PATTERN="$1"
    TO_PATTERN="$2"
fi

# Build diff command
if $NORMALIZE; then
    # Use process substitution to normalize files on the fly
    diff_cmd="diff --from-file <(normalize_file $FROM_PATTERN) <(normalize_file $TO_PATTERN) -sq"
else
    diff_cmd="diff --from-file $FROM_PATTERN $TO_PATTERN -sq"
fi

# Run diff and optionally filter output
if $HIGHLIGHT; then
    eval $diff_cmd | grep -P "$HIGHLIGHT_PATTERN"
else
    eval $diff_cmd
fi