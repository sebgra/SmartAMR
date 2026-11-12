#!/usr/bin/env bash
for file in ../../pep/*.fasta; do
    awk -f scripts/extract_proteins.awk ./config/selected_bacterian_proteins.txt "${file}" > "${file/.faa/.selected.faa}"
done