#!/usr/bin/env bash
#
# Extract only .faa (proteins of phages proteom) from inphared complet dataset

SOURCE_DIR="/mnt/d/Sebastien/SmartAMR/GenomesDB"
TARGET_DIR="/mnt/d/Sebastien/SmartAMR/GenomesDB_flat"

# Création du dossier cible s'il n'existe pas
mkdir -p "$TARGET_DIR"

# Recherche et copie sécurisée
find "$SOURCE_DIR" -type f -name "*.faa" -print0 | xargs -0 -I {} cp --backup=numbered "{}" "$TARGET_DIR/"

echo "Copie vers $TARGET_DIR terminée avec succès !"
