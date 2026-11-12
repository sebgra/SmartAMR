import pandas as pd
import numpy as np
import random

random.seed(42)
N_PHAGES = 100

# 1. Charger les métadonnées INPHARED
metadata = pd.read_csv("14Apr2025_data_excluding_refseq.tsv", sep="\t")

# 2. Définir le mapping direct entre le nom de l'hôte (dans INPHARED) et son Taxon ID
bacteria_taxon_map = {
    "Escherichia coli": 511145,
    "Klebsiella pneumoniae": 1125630,
    "Pseudomonas aeruginosa": 287,
    "Ralstonia solanacearum": 305,
    "Lactococcus lactis": 1360,
    "Staphylococcus aureus": 93061,
    "Salmonella": 28901,  # ou Salmonella enterica selon l'écriture exacte
    "Acinetobacter baumannii": 470,
    "Streptococcus thermophilus": 1308
}

host_col = "Isolation Host (beware inconsistent and nonsense values)"
selected_pairs = []
selected_viruses_set = set()

for bacteria_name, taxon_id in bacteria_taxon_map.items():
    # Filtrer les phages dont l'hôte correspond au nom de la bactérie
    host_filter = metadata[host_col].fillna("").str.contains(bacteria_name, case=False, na=False)
    matched_phages = metadata[host_filter]["Accession"].dropna().unique()
    
    print(f"-> {bacteria_name} (Taxon ID: {taxon_id}) : {len(matched_phages)} phages trouvés.")
    
    if len(matched_phages) == 0:
        continue
        
    # Échantillonner jusqu'à N_PHAGES sans remplacement
    k = min(N_PHAGES, len(matched_phages))
    chosen_phages = random.sample(list(matched_phages), k)
    
    for phage_id in chosen_phages:
        selected_pairs.append({
            "phage_id": phage_id,
            "bacteria": bacteria_name,
            "taxon_id": taxon_id
        })
        selected_viruses_set.add(phage_id)

# 3. Sauvegarder les couples validés
pairs_df = pd.DataFrame(selected_pairs)
pairs_df.to_csv("phage_bacteria_pairs.csv", index=False)

# Sauvegarder la liste unique des phages pour flatten_inphared.sh
viruses_df = pd.DataFrame({"phage_id": list(selected_viruses_set)})
viruses_df.to_csv("selected_virus.csv", index=False)

print(f"\nSélection terminée : {len(selected_viruses_set)} phages uniques sélectionnés.")