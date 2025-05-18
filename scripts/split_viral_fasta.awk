#!/usr/bin/env -S awk -f

BEGIN {
    OUTPUT_DIR="/mnt/data/data/inphared/viral/pep/"
}

/^>/ {
    record_id=$1
    gsub(">",  "", record_id)
    split(record_id, record_arr, "_")
    viral_id=record_arr[1]
    file=OUTPUT_DIR "" viral_id ".faa"
    print $0 > file
}

/^[^>]/ {
    print $0 > file    
}