# Rebuilding the dataset and the models

This is the runnable version of what `refactored/` described as a sequence of
hand-edited scripts. Those scripts hardcoded `/mnt/d/Sebastien/SmartAMR/...`
paths from a Windows mount that no longer exists; every path here comes from
`SmartAMR.config.Settings` and can be redirected with environment variables.

```bash
smartamr reproduce                      # every stage, in order
smartamr reproduce --stages embed train # just these
smartamr app                            # browse the result
```

`smartamr app` is the only front end. The former `smartamr dashboard` was
merged into it and now forwards there with a deprecation notice.

## The stages

| Stage | Needs | Produces |
|---|---|---|
| `catalog` | INPHARED metadata, phage proteomes | `output/catalog/selected_phages.csv`, `phage_bacteria_pairs.csv` |
| `fetch` | network | `data/uniprot_proteomes/` |
| `filter` | `fetch` + `config/selected_bacterian_proteins.txt` | `data/uniprot_proteomes_filtered/` |
| `embed` | both proteome sets, GPU | `output/embeddings/*.npz` |
| `features` | `embed` + `catalog` | `output/features/reproduced_features.parquet` |
| `train` | `features` | `output/models/*.joblib`, `metrics.json` |

Each stage writes to disk before the next reads it, so any stage can be re-run
on its own. `embed` is resumable: it skips organisms already in its store, so an
interrupted run continues rather than restarting.

## Resource caps

Every run is limited to a share of the machine — half of it by default:

```bash
smartamr reproduce                       # 50% of RAM, GPU and cores
smartamr reproduce --fraction 0.25       # a quarter
SMARTAMR_RESOURCE_FRACTION=0.8 smartamr reproduce
```

`SmartAMR.resources` applies the share three ways, because one mechanism does
not cover all three resources:

- **GPU** — `torch.cuda.set_per_process_memory_fraction`, so the CUDA allocator
  refuses an allocation that would cross the share. The cap is a fraction of the
  card's **total** memory, not of what is free: "cap the GPU at 50%" is a
  statement about the card, and a budget that moved whenever another process
  started would be unpredictable.
- **RAM** — enforced cooperatively. `RLIMIT_AS` counts reserved address space,
  and both PyTorch and the CUDA driver reserve huge unbacked ranges, so a hard
  limit fires on a perfectly healthy process. Instead each stage calls
  `memory_guard()` between units of work, which reads the real RSS and raises
  `MemoryBudgetExceeded` with an actionable message.
- **Threads** — `OMP_NUM_THREADS` and friends, plus `n_jobs` on the estimators
  that take it. BLAS otherwise starts one thread per core and oversubscribes
  badly under scikit-learn's own parallelism.

None of this is a security boundary — it is a guard rail against this project's
own appetite.

## Decisions that differ from the original scripts

These change the numbers, so they are listed rather than buried.

### E. coli's taxon id

`refactored/bacteria_proteome_downloader.py` listed E. coli as UniProt taxon
`511145` (K-12 substr. MG1655). UniProt has **8** entries under that organism
id, so the download produced a 3.7 KB file where every other host got megabytes,
and E. coli — the largest host group in the dataset — contributed almost no
bacterial signal. Taxon `83333` (E. coli K-12) has 4,567 entries and is what
UniProt actually annotates that organism under.

The original id is kept in `Host.legacy_uniprot_taxon` and reported by
`smartamr reproduce`, so the change is visible rather than silent.

### Phage selection is restricted to what is on disk

The original sampled 100 accessions per host from the metadata, ignoring whether
each phage's proteome existed. Of the 900 it picked, **62** had a `.faa` file
locally; the rest were dropped during feature building, after the expensive
embedding stage had already run.

Selection now draws only from accessions with a proteome on disk, so every
selected phage is embeddable. With the local INPHARED database that yields 456
phages across the nine hosts.

### Negative pairs come from a cross product

`ESMC_embedding.py` wrote only the matched pairs and then computed each label by
re-checking that same match — which can only ever return `True`. The labels had
to come from somewhere else, and the shape of the project's own feature table
(360 phages × 9 hosts = 3,240 rows, 361 positive) shows what it was: the cross
product of the selected phages with all nine hosts, positive where the phage's
metadata lists that host.

That is what `build_pairs` does, giving 456 × 9 = 4,104 pairs with 456 positives
(11.1%), the same ratio as the original.

### Pooling: `residue` by default, `legacy` available

ESM-C returns one vector per residue, shape `(L, 960)`. Collapsing that to one
vector per protein can be done two ways:

- **`residue`** (default) averages over the residue axis → 960 numbers per
  protein. Every protein lands in the same space whatever its length, so
  proteins and organisms are comparable, and the feature space is stable across
  runs.
- **`legacy`** is what the original did: `np.mean(embeddings, axis=1)` averages
  over the *feature* axis, giving one number per residue position, zero-padded
  to the longest protein in the dataset. Feature *i* then means "mean activation
  at position *i*", and the width of the feature space changes whenever the
  dataset does — which is where the project's magic 3863 and 4199 column counts
  came from.

Pass `--pooling legacy` to reproduce the original feature space.

### Train/test splits are grouped by phage

The binary notebook downsampled and then split row-wise. A phage's embedding is
identical across its nine rows, so a row-wise split puts the same phage on both
sides and lets the model memorise the phage rather than learn the interaction.
Splits are grouped by phage by default; `--rowwise-split` restores the
notebooks' behaviour for comparison.

### The SVM is scored from `decision_function`, not from probabilities

This one was found by the pipeline's own below-chance-AUC check, and it is worth
knowing because it would quietly corrupt any comparison that included the SVM.

Every phage in this dataset contributes a positive row and a negative row that
differ **only in the bacteria half of the vector**. Both of scikit-learn's ways
of getting probabilities out of an SVC — the deprecated `SVC(probability=True)`
and the recommended `CalibratedClassifierCV` — fit their sigmoid on an internal
cross-validation that is *not* group-aware, so it puts a phage's two rows in
different folds. The sigmoid then learns a **negative** slope, and the
probabilities come out perfectly anti-correlated (r = −1.0) with the SVM's own
`decision_function`:

| scoring route | accuracy | ROC AUC |
|---|---|---|
| `predict_proba` (either calibration) | 0.332 | 0.342 |
| `decision_function` (used now) | 0.674 | 0.658 |

ROC only needs a ranking, so an uncalibrated margin is a valid score — and here
a far more trustworthy one. `evaluate()` falls back to `decision_function` for
any estimator without `predict_proba`, which is what the notebooks'
`RocCurveDisplay` was doing for the SVM all along.

The app keeps a visible warning for any estimator that scores below 0.5 AUC, so
a recurrence is reported rather than averaged into a comparison.

### The bacterial block is excluded from the host-prediction task

For "which host does this phage infect?", the host embedding is constant per
host, so including it would hand the model its own target. Only the phage block
is used — as the multilabel notebook also did.

## The awk filter

`refactored/scripts/extract_proteins.awk` is **correct** — its `keep == 1` rule
re-runs on each sequence line, so whole records are emitted, not bare headers.
`SmartAMR.reproduce.proteomes` reimplements it only to drop the hardcoded paths
and to change two behaviours on purpose:

- awk's `$0 ~ name` treats each protein name as a **regular expression**, so a
  name containing `.`, `(` or `+` would match more than intended. Matching here
  is plain substring containment.
- awk's match is case-sensitive, so `OmpA` would miss a header spelling it
  `ompA`. Matching here is case-insensitive.

## Where things go

```
data/uniprot_proteomes/           raw bacterial proteomes       (gitignored)
data/uniprot_proteomes_filtered/  subset to resistance proteins (gitignored)
GenomesDB/<accession>/*.faa       INPHARED phage proteomes      (gitignored)
output/catalog/                   selection and pair list
output/embeddings/                *.npz stores + .meta.json
output/features/                  reproduced_features.parquet
output/models/                    *.joblib + metrics.json
output/run_manifest.json          what the last run did
```

Redirect any of it:

```bash
export SMARTAMR_ROOT=/data/smartamr
export SMARTAMR_DATA_DIR=/scratch/smartamr/data
export SMARTAMR_OUTPUT_DIR=/scratch/smartamr/output
export SMARTAMR_GENOMES_DIR=/refdata/GenomesDB
```
