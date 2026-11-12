# %% [markdown]
# # Download Bacteria proteomes from BV-BRC database

# %%
from SmartAMR.scrapper import GenomeProteinSeqDownloader
import pandas as pd

# %%
bacterias = pd.read_csv("config/bacterias.csv", dtype={'bvbrc_id': object})
bacterias

# %%
downloader = GenomeProteinSeqDownloader("tmp/pep/")

# %%
for bvbrc_id in bacterias["bvbrc_id"]:
    downloader.download(bvbrc_id)


