# Smart-AMR

Circumvent antibioresistance with bacteriophages selection.


We use a large language model (LLM) finetuned on protein sequences to extract an embedding of a phage proteome and a bacteria proteome and predict phage targetting a bacteria.

# Environment setting 

To setup a fresh environment for the project:

```bash
mamba env create -f environment.yaml
mamba activate smartamrX
```

Then install the Python package to access utility functions:
```bash
pip install -e .
```

# Pre-written functions

Some functions were written to help in the process of data scrapping and relevent dataset generation. Functions are split in 3 modules:

- scrapper: to scrap information from the BVBRC database. Such contains different information distributed across different files as: 
    -  PATRIC_genomes_AMR.txt: containing all the metainformation about the database
    - .fna: FASTA contig sequences
    - .faa: FASTA protein sequence file
    - .features.tab: All genomic features and related information in tab-delimited format
    - .ffn: FASTA nucleotide sequences for genomic features, i.e. genes, RNAs, and other misc features
    - .frn: FASTA nucleotide sequences for RNAs
    - .gff: Genome annotations in GFF file format
    - .pathway.tab: Metabolic pathway assignments in tab-delimited format
    - .spgene.tab: Specialty gene assignements (i.e. AMR genes, virulance factors, essential genes, etc) in tab-delimited format
    - .subsystem.tab: Subsystem assignments in tab-delimited format
- filter: to filter the metadata, such extracting relevant set of (sub-)species depending on user defined criterium e.g. valid antibiotic sensibility.
- utils: to get information on a specie directly such as antibiotic resistance gene names, genomic coordinates of such genes, and related sequences.

## Source data

The phage-host relationship and phage proteome data comes from the [inphared dataset](https://doi.org/10.1089/phage.2021.0007) (Cook R et al., 2021).
We extracted the reference bacterian proteomes from the [BV-BRC database](https://doi.org/10.1093/nar/gkac1003) (Olson et al. 2023).


## Steps

1. Download the selected bacterian proteome from BV-BRC database with [./notebooks/bacteria_dataset.ipynb](./notebooks/bacteria_dataset.ipynb)
3. Subset the bacterian proteome on a selection of proteins expected to be involved in viral resistance (`scripts/extract_bacterian_proteomes.sh`)
2. Select the viruses targetting the selected bacteria with [./notebooks/virus_dataset.ipynb](./notebooks/virus_dataset.ipynb)
3. Embed the bacterian and viral proteomes and merge the embeddings with ESM-C LLM in [./notebooks/ESMC_embedding.ipynb](./notebooks/ESMC_embedding.ipynb).
4. Train prediction models
    - [./notebooks/binary_classification.ipynb](./notebooks/binary_classification.ipynb): classify phage-bacteria host relationship
    - [./notebooks/multilabel_classification.ipynb](./notebooks/multilabel_classification.ipynb): predict phage bacterian host
5. (optional): plot the phage proteome embeddings with [./notebooks/embedding_plot.ipynb](./notebooks/embedding_plot.ipynb).

