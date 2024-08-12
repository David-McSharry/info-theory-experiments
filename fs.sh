#!/bin/bash

# Additional files or folders to ignore (add your own here)
ADDITIONAL_IGNORES=(
    "activations"
    "activations2"
    "econ_texts"
    "models"
    "RNNs"
)

# Function to check if a file is ignored
is_ignored() {
    local path="$1"
    
    # Check if it's .git directory or its contents
    if [[ "$path" == */.git || "$path" == */.git/* ]]; then
        return 0
    fi
    
    # Check if it's in the additional ignores list
    for ignore in "${ADDITIONAL_IGNORES[@]}"; do
        if [[ "$path" == *"$ignore"* ]]; then
            return 0
        fi
    done
    
    # Check if it's ignored by git
    git check-ignore -q "$path"
    return $?
}

print_tree() {
    local dir="${1:-.}"
    local prefix="${2}"
    local last_dir="${3}"

    # List files and directories
    local items=($(ls -A1 "$dir" | sort))

    local count=${#items[@]}
    local index=0

    for item in "${items[@]}"; do
        local full_path="$dir/$item"
        
        # Skip if the item is ignored
        if is_ignored "$full_path"; then
            continue
        fi

        index=$((index + 1))
        
        if [ $index -eq $count ]; then
            echo "${prefix}└── $item"
            if [ -d "$full_path" ]; then
                print_tree "$full_path" "${prefix}    " "last"
            fi
        else
            echo "${prefix}├── $item"
            if [ -d "$full_path" ]; then
                print_tree "$full_path" "${prefix}│   " ""
            fi
        fi
    done
}

# Check if we're in a git repository
if [ ! -d .git ]; then
    echo "Error: This script should be run from the root of a Git repository."
    exit 1
fi

echo "Project structure:"
print_tree "."
