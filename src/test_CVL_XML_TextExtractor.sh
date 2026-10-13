#!/usr/bin/env bash
# Run CVL_XML_TextExtractor.py on every CVL xml file and report pass / empty / error.
#
# Usage:
#   ./test_CVL_XML_TextExtractor.sh [cvl_root]
#   cvl_root defaults to cvl-database-1-1 next to this script
#
# PASS  = extracted non-empty text
# EMPTY = no handwriting found (expected for a few blank pages)
# ERROR = the extractor crashed (bug or bad file) -> exit code 1
#
# Paths resolve from this script's own folder, so it runs from anywhere.
# Keep CVL_XML_TextExtractor.py in the same folder as this script.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CVL_ROOT="${1:-$SCRIPT_DIR/cvl-database-1-1}"
EXTRACTOR="$SCRIPT_DIR/CVL_XML_TextExtractor.py"

for path in "$EXTRACTOR" "$CVL_ROOT"; do
    [ -e "$path" ] || { echo "Not found: $path"; exit 1; }
done

pass=0; empty=0; error=0
empty_files=(); error_files=()

mapfile -t files < <(find "$CVL_ROOT" -path "*/xml/*.xml" | sort)
total=${#files[@]}
if [ "$total" -eq 0 ]; then
    echo "No xml files found under $CVL_ROOT"; exit 1
fi
echo "Testing $total xml files under $CVL_ROOT ..."

for i in "${!files[@]}"; do
    f="${files[$i]}"
    if out=$(python3 "$EXTRACTOR" "$f" 2>&1); then
        if [ -n "$(echo "$out" | tr -d '[:space:]')" ]; then
            ((pass++))
        else
            ((empty++)); empty_files+=("$f")
        fi
    else
        ((error++)); error_files+=("$f :: $(echo "$out" | tail -1)")
    fi
    # progress every 100 files
    (( (i + 1) % 100 == 0 )) && echo "  $((i + 1))/$total"
done

echo
echo "===== RESULTS ====="
echo "Total : $total"
echo "PASS  : $pass"
echo "EMPTY : $empty"
echo "ERROR : $error"

if [ "$empty" -gt 0 ]; then
    echo; echo "Empty (no handwriting):"
    printf '  %s\n' "${empty_files[@]}"
fi
if [ "$error" -gt 0 ]; then
    echo; echo "Errors:"
    printf '  %s\n' "${error_files[@]}"
    exit 1
fi
exit 
