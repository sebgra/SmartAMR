#!/usr/bin/env -S awk -f
# Usage: awk -f extract_viral_proteins.awk

BEGIN {
    OUTPUT_DIR="/mnt/data/data/inphared/viral/pep/"
}

NR == FNR {
    protein_names[$0] = 1
    next
}

/^>/ {
    keep=0
    accession_number=$1
    split("_", arr, accession_number)
    virus_id=arr[1]
    for (name in protein_names) {
        if ($0 ~ name) {
            keep = 1
            file = OUTPUT_DIR virus_id
            print $0 > file
        }
    }
}

/^[^>]/ && keep=1 {
    print $0 > file
}