
# %% [markdown]
# # Import necessary libraries

# %%
import glob
import os
import itertools
from typing import Sequence, Dict, List

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from Bio import SeqIO

from esm.sdk import client
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig, LogitsOutput, ProteinType

# %% [markdown]
# # Define utilitary functions

# %%
EMBEDDING_CONFIG = LogitsConfig(
    sequence=True, return_embeddings=True, return_hidden_states=True, ith_hidden_layer=100
)

def embed_sequence(model, sequence: str) -> LogitsOutput:
    protein = ESMProtein(sequence=sequence)
    protein_tensor = model.encode(protein)
    output = model.logits(protein_tensor, EMBEDDING_CONFIG)
    return output

def pad_arrays_to_max_shape(list_of_arrays, pad_value=0):
    """
    Pads a list of 2D NumPy arrays to the shape of the largest array in the list.
    """
    max_rows = max(arr.shape[0] for arr in list_of_arrays)
    padded_arrays = []
    for arr in list_of_arrays:
        rows_to_pad = max_rows - arr.shape[0]
        padded_arr = np.pad(arr,
                            pad_width=((0, rows_to_pad)),
                            mode='constant',
                            constant_values=pad_value)
        padded_arrays.append(padded_arr)
    return padded_arrays

# %% [markdown]
# # Load ESMC model

# %%
client = ESMC.from_pretrained("esmc_300m").to("cpu") # or "cuda"

# %% [markdown]
# # Embed all filtered phage proteomes from GenomesDB_flat (.faa)

# %%
phages_dir = "/mnt/d/Sebastien/SmartAMR/GenomesDB_flat/"
fasta_files = glob.glob(os.path.join(phages_dir, "*.faa"))

embeddings_df_columns = {
    "phage_id": [],
    "embeddings": []
}

for file in fasta_files:
    filename = os.path.basename(file)
    phage_id = filename.split('.')[0]
    
    print(f"Embedding du phage : {phage_id}")
    
    embeddings_dict = {}
    fasta_sequences = SeqIO.parse(file, 'fasta')
    
    for protein in fasta_sequences:
        protein_id = protein.id
        sequence = str(protein.seq)
        if len(sequence) > 0:
            emb = np.mean(embed_sequence(client, sequence).embeddings.cpu().squeeze(0).numpy(), axis=1)
            embeddings_dict[protein_id] = emb

    if embeddings_dict:
        embeddings_list = list(embeddings_dict.values())
        padded_embeddings = pad_arrays_to_max_shape(embeddings_list)
        mean_embedding = np.mean(padded_embeddings, axis=0)
        
        embeddings_df_columns["phage_id"].append(phage_id)
        embeddings_df_columns["embeddings"].append(mean_embedding)

embeddings_df = pd.DataFrame(embeddings_df_columns)
os.makedirs("../tmp", exist_ok=True)
embeddings_df.to_hdf("../tmp/phage_embeddings.h5", "phage_embeddings", mode='w')

# %% [markdown]
# ## Embed bacteria proteins using Taxon IDs

# %%
taxon_ids = [511145, 1125630, 287, 305, 1360, 93061, 28901, 470, 1308]

bacteria_embeddings_df_columns = {
    "taxon_id": [],
    "embeddings": []
}

for taxon_id in taxon_ids:
    proteome_path = f"uniprot_proteomes_output/proteome_taxon_{taxon_id}.fasta"
    
    if not os.path.exists(proteome_path):
        print(f"Attention : Fichier introuvable pour le Taxon ID {taxon_id}")
        continue
        
    print(f"Calcul de l'embedding pour le Taxon ID : {taxon_id}")
    
    embeddings_dict = {}
    fasta_sequences = SeqIO.parse(proteome_path, 'fasta')
    
    for protein in fasta_sequences:
        protein_id = protein.id
        sequence = str(protein.seq)
        if len(sequence) > 0:
            emb = np.mean(embed_sequence(client, sequence).embeddings.cpu().squeeze(0).numpy(), axis=1)
            embeddings_dict[protein_id] = emb

    if embeddings_dict:
        embeddings_list = list(embeddings_dict.values())
        padded_embeddings = pad_arrays_to_max_shape(embeddings_list)
        mean_embedding = np.mean(padded_embeddings, axis=0)
        
        bacteria_embeddings_df_columns["taxon_id"].append(str(taxon_id))
        bacteria_embeddings_df_columns["embeddings"].append(mean_embedding)

bacteria_embeddings_df = pd.DataFrame(bacteria_embeddings_df_columns)
bacteria_embeddings_df.to_hdf("../tmp/bacteria_embeddings.h5", key="bacteria_embeddings", mode='w')

# %% [markdown]
# ## Preparation of concatenated vectors

# %%
embeddings_df = pd.read_hdf("../tmp/phage_embeddings.h5", key="phage_embeddings")
bacteria_embeddings_df = pd.read_hdf("../tmp/bacteria_embeddings.h5", key="bacteria_embeddings")

metadata = pd.read_csv("14Apr2025_data_excluding_refseq.tsv", sep="\t")
pairs_df = pd.read_csv("phage_bacteria_pairs.csv", dtype=str)

# %%
embeddings_df["padded_embeddings"] = pad_arrays_to_max_shape(embeddings_df["embeddings"].values)
bacteria_embeddings_df["padded_embeddings"] = pad_arrays_to_max_shape(bacteria_embeddings_df["embeddings"].values)

embeddings_df["flat_embeddings"] = [embedding.flatten() for embedding in embeddings_df["padded_embeddings"].values]
bacteria_embeddings_df["flat_embeddings"] = [embedding.flatten() for embedding in bacteria_embeddings_df["padded_embeddings"].values]

# %%
def find_label(phage_id, bacteria_name):
    phage_metadata = metadata[metadata["Accession"] == phage_id]
    host_col = "Isolation Host (beware inconsistent and nonsense values)"
    pair_metadata = phage_metadata[phage_metadata[host_col].fillna("").str.contains(bacteria_name, case=False, na=False)]
    return pair_metadata.shape[0] > 0

# %%
features_df_columns = {
    "phage_id": [],
    "taxon_id": [],
    "embedding": [],
    "label": []
}

for _, row in pairs_df.iterrows():
    phage_id = row["phage_id"]
    bacteria_name = row["bacteria"]
    taxon_id = str(row["taxon_id"])

    # Vérifier si les embeddings existent pour ce phage et ce taxon
    v_row = embeddings_df[embeddings_df["phage_id"] == phage_id]
    b_row = bacteria_embeddings_df[bacteria_embeddings_df["taxon_id"] == taxon_id]
    
    if v_row.empty or b_row.empty:
        continue

    viral_embedding = v_row["flat_embeddings"].values[0]
    bacteria_embedding = b_row["flat_embeddings"].values[0]
    
    combined_embedding = np.append(viral_embedding, bacteria_embedding)
    label = find_label(phage_id, bacteria_name)

    features_df_columns["phage_id"].append(phage_id)
    features_df_columns["taxon_id"].append(taxon_id)
    features_df_columns["embedding"].append(combined_embedding)
    features_df_columns["label"].append(label)

features_df = pd.DataFrame(features_df_columns)
print("Distribution des labels :")
print(features_df["label"].value_counts())

# %%
# Export des features finales
exploded_df = pd.DataFrame(features_df['embedding'].tolist())
new_df = pd.concat([features_df.drop('embedding', axis=1), exploded_df], axis=1)
new_df.to_csv("final_features.csv", index=False)
print("Export terminé vers final_features.csv")