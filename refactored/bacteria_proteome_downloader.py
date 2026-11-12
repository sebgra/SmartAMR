import os
import requests

# Liste nettoyée des Taxon IDs (partie entière des identifiants BV-BRC)
taxon_ids = [
    511145,  # E. coli K-12 MG1655
    1125630, # K. pneumoniae HS11286
    287,     # P. aeruginosa
    305,     # R. solanacearum
    1360,    # L. lactis
    93061,   # S. aureus NCTC 8325
    28901,   # Salmonella enterica (ou adapter selon la souche précise)
    470,     # A. baumannii
    1308     # S. thermophilus
]

# Vous pouvez ensuite réutiliser la boucle de téléchargement direct via l'API UniProt présentée précédemment.

output_dir = "uniprot_proteomes_output"
os.makedirs(output_dir, exist_ok=True)

for taxon_id in taxon_ids:
  print(f"Téléchargement du protéome pour le Taxon ID : {taxon_id}...")

  # Utilisation directe de l'endpoint stream d'UniProtKB avec organism_id
  fasta_url = "https://rest.uniprot.org/uniprotkb/stream"
  params = {
      "query": f"organism_id:{taxon_id}",
      "format": "fasta",
      "compressed": "false"
  }

  response = requests.get(fasta_url, params=params, stream=True)

  if response.status_code == 200:
    # Vérification rapide si le contenu renvoyé n'est pas vide
    output_path = os.path.join(output_dir, f"proteome_taxon_{taxon_id}.fasta")
    
    downloaded = False
    with open(output_path, "wb") as f:
      for chunk in response.iter_content(chunk_size=8192):
        if chunk:
          f.write(chunk)
          downloaded = True
          
    if downloaded and os.path.getsize(output_path) > 100:  # S'assure qu'on a un vrai fichier FASTA
      print(f" -> Sauvegardé avec succès dans {output_path}")
    else:
      print(f" -> Aucun résultat ou fichier vide pour le Taxon ID {taxon_id}")
      if os.path.exists(output_path):
        os.remove(output_path)
  else:
    print(f" -> Erreur HTTP {response.status_code} pour le Taxon ID {taxon_id}")

print("Téléchargements terminés.")