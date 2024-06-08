#!/bin/bash

# Function to recursively rename files
rename_files() {
    local dir=$1
    local file

    # Iterate through files in the directory
    for file in "$dir"/*; do
        if [[ -d "$file" ]]; then
            # If the file is a directory, recursively call the function
            rename_files "$file"
        elif [[ -f "$file" ]]; then
            # If the file is a regular file, rename it
            rename_file "$file"
        fi
    done
}

# Function to rename a single file
rename_file() {
    local file=$1
    local filename=$(basename -- "$file")
    local new_filename

    # Rename black pieces
    if [[ $filename == b* ]]; then
		new_filename=$(sed 's/^\(.\)\(.\)/\1\u\2/' <<< "$filename")
        mv "$file" "${file/$filename/$new_filename}"
    fi

    # Rename white pieces
    if [[ $filename == w* ]]; then
		new_filename=$(sed 's/^\(.\)\(.\)/\1\u\2/' <<< "$filename")
        mv "$file" "${file/$filename/$new_filename}"
    fi
}

# Start renaming from the current directory
rename_files .

echo "Renaming completed."
