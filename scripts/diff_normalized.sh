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

# Helper function to normalize a file (remove .0 from numbers)
normalize_file_sed() {
    local infile="$1"
    local outfile="$2"
    sed 's/\([0-9]\)\.0\b/\1/g' "$infile" > "$outfile"
}

# Function to get a unique normalized filename
get_normalized_name() {
    local orig="$1"
    local base="${orig%.*}"
    local ext="${orig##*.}"
    local n=1
    local candidate="${base}_normalized.${ext}"
    while [ -e "$candidate" ]; do
        candidate="${base}_normalized${n}.${ext}"
        n=$((n+1))
    done
    echo "$candidate"
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
    # set -- $COMMAND
    FROM_PATTERN="$1"
    TO_PATTERN="$2"
fi

# Build diff command
if $NORMALIZE; then
    TMP_FROM=()
    TMP_TO=()
    for f in $FROM_PATTERN; do
        if [[ -f "$f" && "${f}" != *_normalized.* ]]; then
            normf="${f%.*}_normalized.${f##*.}"
            if [ ! -e "$normf" ]; then
                normalize_file_sed "$f" "$normf"
            fi
            TMP_FROM+=("$normf")
        fi
    done
    for f in $TO_PATTERN; do
        if [[ -f "$f" && "${f}" != *_normalized.* ]]; then
            normf="${f%.*}_normalized.${f##*.}"
            if [ ! -e "$normf" ]; then
                normalize_file_sed "$f" "$normf"
            fi
            TMP_TO+=("$normf")
        fi
    done
    diff_cmd="diff --from-file ${TMP_FROM[*]} ${TMP_TO[*]} -sq"
else
    diff_cmd="diff --from-file $FROM_PATTERN $TO_PATTERN -sq"
fi

# Run diff and optionally filter output
if $HIGHLIGHT; then
    eval $diff_cmd | grep --color=always -P "$HIGHLIGHT_PATTERN"
else
    eval $diff_cmd
fi