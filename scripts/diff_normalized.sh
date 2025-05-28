#!/bin/bash

NORMALIZE=false
HIGHLIGHT=false
COUNT_DIFF=false

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
        --count-diff)
            COUNT_DIFF=true
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
    echo "Usage: $0 [--normalize] [--highlight] [--count-diff] <command>"
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

# Function to count matrix differences between two files using diff_matriz.py
count_matrix_diff() {
    local file1="$1"
    local file2="$2"
    local script_dir="$(dirname "$0")"
    local venv_python="$(which python3)"
    local diff_output
    diff_output=$($venv_python "$script_dir/diff_matriz.py" "$file1" "$file2" 2>/dev/null)
    echo "$diff_output" | grep '\[DIFF-COUNT\]' | awk -F': ' '{print $2}' | awk '{print $1}'
}

# Run diff and optionally filter output
if $HIGHLIGHT; then
    eval $diff_cmd | perl -pe '
        s{(/main_\w+/)}{\e[1;33m$1\e[0m}g;      # yellow
        s{(are identical)}{\e[1;32m$1\e[0m}g;    # green
        s{(differ)}{\e[1;31m$1\e[0m}g;           # red
        s{(full)}{\e[1;35m$1\e[0m}g;             # magenta
        s{(threads_)}{\e[1;36m$1\e[0m}g;         # cyan
    '
else
    eval $diff_cmd
fi

if $COUNT_DIFF && $NORMALIZE; then
    for i in "${!TMP_FROM[@]}"; do
        f1="${TMP_FROM[$i]}"
        for j in "${!TMP_TO[@]}"; do
            f2="${TMP_TO[$j]}"
            if [[ -f "$f1" && -f "$f2" ]]; then
                diff_count=$(count_matrix_diff "$f1" "$f2")
                echo "[DIFF-COUNT] $f1 vs $f2: $diff_count differing elements"
            fi
        done
    done
fi