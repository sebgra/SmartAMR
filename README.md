# Smart-AMR

Circumvent antibioresistance with bacteriophages selection.


We use a large language model (LLM) finetuned on protein sequences to extract an embedding of a phage proteome and a bacteria proteome and predict phage targetting a bacteria.

# Quick start

The whole dataset and both models rebuild from the raw inputs with one command,
and the result is browsable in a web app:

```bash
pip install -e ".[all]"

smartamr reproduce     # select, fetch, filter, embed, pair, train
smartamr app           # browse the dataset, embeddings, models and predictions
```

Every run is capped at **half the machine** — RAM, GPU and cores — so it stays
usable while the embedding stage runs. Change the share with `--fraction 0.25`
or `SMARTAMR_RESOURCE_FRACTION`.

Individual stages, for when only part of it needs rebuilding:

```bash
smartamr reproduce --stages catalog          # choose the phages
smartamr reproduce --stages fetch filter     # bacterial proteomes
smartamr reproduce --stages embed            # the GPU stage, resumable
smartamr reproduce --stages features train   # pair up and fit
```

[`docs/REPRODUCE.md`](./docs/REPRODUCE.md) documents the stages, the resource
caps and — importantly — the places where this pipeline deliberately differs
from the original notebooks and `refactored/` scripts, including a taxon-id
correction for E. coli that materially changes the data.

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

# Package layout

```
SmartAMR/
  config.py       every path the pipeline uses, overridable by environment
  resources.py    the share of the machine a run may use (RAM, GPU, threads)
  data/           downloads, metadata filtering, phage and protein selection
  embedding/      ESM-C embeddings and their projections
  models/         training, evaluation, prediction, screening
  genomics/       AMR gene lookups and codon tables
  reproduce/      rebuild the dataset and the models end to end
  webapp/         the Streamlit web app (one front end, six pages)
  viz/            chart builders and colour tokens, shared with the figure export
  cli.py          the `smartamr` command
```

Nothing heavy is imported at package level: `torch`, `esm`, `scikit-learn` and
`streamlit` are imported inside the functions that use them, so `smartamr
--help` is fast and the package works with only some extras installed.

The app has four pages. **Analyze** is the one with an input: upload a
proteome — a phage *or* a bacterium — and it embeds it with ESM-C and predicts
the other side.

| Upload | You get |
|---|---|
| a **phage** | its score against all nine bacterial hosts, a second opinion from the host-prediction model, its closest known phage relatives, and where it lands in the embedding projection |
| a **bacterium** | every embedded phage ranked as a candidate, the spread of those scores, what the recommended phages normally infect, and which reference proteome it most resembles |

The other three explain what is behind that answer: **Dataset** (selection,
class balance, genome statistics, the feature table and the embedding space),
**Models** (both tasks, four estimators each) and **Pipeline** (what was built,
when, and under which resource caps).

A bacterial upload is filtered to the proteins in
`config/selected_bacterian_proteins.txt` before embedding, because the
bacterial side of the training data was. Embedding a whole ~5,000-protein
proteome would put the vector somewhere the models have never seen.

There used to be two Streamlit front ends, and then six pages across one. Both
simplifications are done: `SmartAMR/dashboard/` was merged in (its chart
library is now `SmartAMR/viz/`), and the Predict and Finder pages — which asked
the same question from two directions — became the single Analyze page, while
Embeddings folded into Dataset. `smartamr dashboard` is a deprecated alias for
`smartamr app`. Interactive model training was not carried over: it is
long-running GPU and CPU work that belongs in
`smartamr reproduce --stages train`, and the app stays read-only over what is
on disk.

`SmartAMR/scrapper.py`, `filter.py` and `utils.py` are **deprecated shims**.
Their contents moved to `SmartAMR/data/` and `SmartAMR/genomics/` when the
package was split up; the shims re-export from the new locations so the older
notebooks keep working, and they emit a `DeprecationWarning`.

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

These are the original, notebook-driven steps. `smartamr reproduce` runs the
same pipeline non-interactively and is the maintained path; the notebooks are
kept as the record of how the analysis was developed.

1. Download the selected bacterian proteome from BV-BRC database with [./notebooks/bacteria_dataset.ipynb](./notebooks/bacteria_dataset.ipynb)
3. Subset the bacterian proteome on a selection of proteins expected to be involved in viral resistance (`scripts/extract_bacterian_proteomes.sh`)
2. Select the viruses targetting the selected bacteria with [./notebooks/virus_dataset.ipynb](./notebooks/virus_dataset.ipynb)
3. Embed the bacterian and viral proteomes and merge the embeddings with ESM-C LLM in [./notebooks/ESMC_embedding.ipynb](./notebooks/ESMC_embedding.ipynb).
4. Train prediction models
    - [./notebooks/binary_classification.ipynb](./notebooks/binary_classification.ipynb): classify phage-bacteria host relationship
    - [./notebooks/multilabel_classification.ipynb](./notebooks/multilabel_classification.ipynb): predict phage bacterian host
5. (optional): plot the phage proteome embeddings with [./notebooks/embedding_plot.ipynb](./notebooks/embedding_plot.ipynb).

## Web application 

A prototype of a web application in VueJS is in [./web_interface](./web_interface/).