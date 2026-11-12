#!/usr/bin/env awk -f
# Usage: awk -f extract_proteins.awk PROTEIN_NAMES.list PROTEOME.fasta

BEGIN {
    # Pas de FS strict pour s'adapter aux différents formats d'en-têtes UniProt/NCBI
}

# Phase 1 : Lecture du premier fichier (la liste des protéines à garder)
NR == FNR {
    # On nettoie les éventuels retours chariot (\r) Windows
    gsub(/\r/, "", $0)
    if ($0 != "") {
        protein_names[$0] = 1
    }
    next
}

# Phase 2 : Lecture du fichier FASTA
/^>/ {
    keep = 0
    # On vérifie si l'un des identifiants recherchés est présent dans la ligne d'en-tête entière ($0)
    for (name in protein_names) {
        if ($0 ~ name) {
            keep = 1
            break
        }
    }
}

# Impression de l'en-tête et de la séquence si le flag 'keep' est à 1
keep == 1 {
    print $0
}