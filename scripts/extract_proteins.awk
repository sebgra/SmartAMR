#!/usr/bin/env awk -f
# Usage: awk -f extract_proteins.awk PROTEIN_NAMES.list PROTEOME.fasta
BEGIN {
    FS="  "    
}

NR == FNR {
    protein_names[$0] = 1
    next
}

/^>/ {
    keep=0
    for (name in protein_names) {
        if ($2 ~ name) {
            keep = 1
            print $0
            break
        }
    }
}

/^[^>]/ && keep == 1 {
    print $0
}
