#!/usr/bin/env bash

# Définition des chemins
PROTEIN_LIST="/mnt/d/Sebastien/SmartAMR/refactored/selected_bacterian_proteins.txt"
AWK_SCRIPT="/mnt/d/Sebastien/SmartAMR/refactored/scripts/extract_proteins.awk"
INPUT_DIR="/mnt/d/Sebastien/SmartAMR/refactored/uniprot_proteomes_output"

for file in "${INPUT_DIR}"/*.fasta; do
    # Éviter l'erreur si aucun fichier ne correspond
    [ -e "$file" ] || continue
    
    output_file="${file/.fasta/.selected.fasta}"
    echo "Traitement de : $(basename "$file") -> $(basename "$output_file")"
    
    # Appel d'Awk avec la liste en premier argument, puis le fichier FASTA
    awk -f "$AWK_SCRIPT" "$PROTEIN_LIST" "$file" > "$output_file"
done

echo "Extraction terminée."