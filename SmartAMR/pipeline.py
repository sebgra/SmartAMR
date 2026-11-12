"""The project's pipeline, as the original README laid it out.

One definition of the steps, shared by ``smartamr pipeline`` and the
dashboard's Reproduce page, so the two always agree on what is done, what is
ready to run, and what is still blocked.

The steps follow the hand-written README:

1. download the selected bacterial proteomes from BV-BRC
2. subset them on the proteins expected to be involved in viral resistance
3. select the viruses targeting the selected bacteria
4. build one proteome per phage from the inphared protein FASTA
5. embed both sides with ESM-C and merge the embeddings
6. train the two prediction models:
   - host-phage relationship classification (binary)
   - host prediction (multilabel)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List

from SmartAMR.config import Settings

DONE = "done"
READY = "ready"
BLOCKED = "blocked"


@dataclass
class Step:
    """One documented pipeline step."""

    name: str
    title: str
    description: str
    command: str
    produces: Path
    needs: List[Path] = field(default_factory=list)
    heavy: str = ""
    run: Callable[[Settings], None] | None = None

    def status(self) -> str:
        """``done`` if its output exists, ``ready`` if it can run, else ``blocked``."""
        if self._exists(self.produces):
            return DONE
        if all(self._exists(path) for path in self.needs):
            return READY
        return BLOCKED

    def missing(self) -> List[Path]:
        """Prerequisites that are not on disk."""
        return [path for path in self.needs if not self._exists(path)]

    @staticmethod
    def _exists(path: Path) -> bool:
        path = Path(path)
        if path.is_dir():
            return any(path.iterdir())
        return path.exists()


# --------------------------------------------------------------------------
# Step bodies
# --------------------------------------------------------------------------
def _fetch_bacteria(settings: Settings) -> None:
    from SmartAMR.data.scrapper import download_genomes
    from SmartAMR.data.viruses import read_bacteria_table

    table = read_bacteria_table(settings.bacteria_table)
    settings.ensure(settings.bacteria_proteomes)
    download_genomes(
        table["bvbrc_id"].astype(str).tolist(),
        outdir=settings.bacteria_proteomes,
        file_type="protein",
    )


def _extract_proteins(settings: Settings) -> None:
    from SmartAMR.data.proteins import extract_proteins_in_dir

    results = extract_proteins_in_dir(settings.bacteria_proteomes, settings.selected_proteins)
    for path, kept in results:
        print(f"  {path.name}: {kept} record(s)")


def _select_viruses(settings: Settings) -> None:
    from SmartAMR.data.viruses import (
        read_bacteria_table,
        read_inphared_metadata,
        select_viruses,
        write_selection,
    )

    metadata = read_inphared_metadata(settings.inphared_metadata)
    bacteria = read_bacteria_table(settings.bacteria_table)["bacteria"].tolist()
    selected = select_viruses(metadata, bacteria)
    write_selection(selected, settings.selected_viruses, settings.virus_list, overwrite=True)


def _split_phages(settings: Settings) -> None:
    from SmartAMR.data.phages import split_phage_proteomes
    from SmartAMR.stats import selected_accessions

    selection = selected_accessions(settings.selected_viruses)
    accessions = sorted({a for values in selection.values() for a in values})

    counts = split_phage_proteomes(
        settings.phage_protein_fasta,
        outdir=settings.phage_proteomes,
        accessions=accessions,
    )
    print(f"  {len(counts)} phage proteome(s) written")


def _embed_bacteria(settings: Settings) -> None:
    from SmartAMR.embedding.esmc import embed_proteomes, load_model, save_embeddings

    files = sorted(settings.bacteria_proteomes.glob("*.selected.faa"))
    if not files:
        raise FileNotFoundError(
            f"No *.selected.faa in {settings.bacteria_proteomes}. Run the subset step first."
        )

    model = load_model()
    frame = embed_proteomes(files, model=model, id_column="bacteria_id")
    save_embeddings(frame, settings.bacteria_embeddings, key="bacteria_embeddings")


def _embed_phages(settings: Settings) -> None:
    from SmartAMR.embedding.esmc import embed_proteomes, load_model, save_embeddings

    files = sorted(settings.phage_proteomes.rglob("*.faa")) or sorted(
        settings.phage_proteomes.rglob("*.csv")
    )
    if not files:
        raise FileNotFoundError(f"No phage proteome in {settings.phage_proteomes}")

    model = load_model()
    frame = embed_proteomes(files, model=model, id_column="phage_id")
    save_embeddings(frame, settings.phage_embeddings, key="phage_embeddings")


def _build_features(settings: Settings) -> None:
    from SmartAMR.data.viruses import read_inphared_metadata
    from SmartAMR.embedding.esmc import load_embeddings
    from SmartAMR.embedding.features import build_features, save_features

    phages = load_embeddings(settings.phage_embeddings, key="phage_embeddings")
    bacterias = load_embeddings(settings.bacteria_embeddings, key="bacteria_embeddings")
    metadata = read_inphared_metadata(settings.inphared_metadata)

    features = build_features(phages, bacterias, metadata)
    save_features(features, settings.features)
    print(f"  {len(features):,} pairs, {int(features['label'].sum()):,} interacting")


def _train_relationship(settings: Settings) -> None:
    _train(settings, task="binary")


def _train_host(settings: Settings) -> None:
    _train(settings, task="multilabel")


def _train(settings: Settings, task: str, model_name: str = "xgboost") -> None:
    from SmartAMR.embedding.features import load_features
    from SmartAMR.models.classifiers import save_model, train

    features = load_features(settings.features)
    model = train(features, model_name=model_name, task=task)
    path = save_model(model, settings.model_path(f"{task}-{model_name}"))

    print(f"  accuracy {model.metrics['accuracy']:.3f} on {model.metrics['n_test']} pairs -> {path.name}")


# --------------------------------------------------------------------------
# The pipeline
# --------------------------------------------------------------------------
def pipeline_steps(settings: Settings, model_name: str = "xgboost") -> List[Step]:
    """The documented steps, in order, bound to these settings."""

    return [
        Step(
            name="fetch-bacteria",
            title="Download the bacterial proteomes",
            description="One protein FASTA per genome from the BV-BRC database.",
            command="smartamr fetch-bacteria",
            needs=[settings.bacteria_table],
            produces=settings.bacteria_proteomes,
            heavy="downloads from BV-BRC",
            run=_fetch_bacteria,
        ),
        Step(
            name="extract-proteins",
            title="Subset on the resistance proteins",
            description="Keep the proteins expected to be involved in viral resistance.",
            command="smartamr extract-proteins",
            needs=[settings.bacteria_proteomes, settings.selected_proteins],
            produces=settings.bacteria_proteomes / "_selected",  # checked below
            run=_extract_proteins,
        ),
        Step(
            name="select-viruses",
            title="Select the viruses targeting the bacteria",
            description="Draw phages per bacteria from the inphared metadata.",
            command="smartamr select-viruses",
            needs=[settings.inphared_metadata, settings.bacteria_table],
            produces=settings.selected_viruses,
            run=_select_viruses,
        ),
        Step(
            name="split-phages",
            title="Build one proteome per phage",
            description="Split the inphared protein FASTA on the accession prefix.",
            command="smartamr split-phages",
            needs=[settings.phage_protein_fasta, settings.selected_viruses],
            produces=settings.phage_proteomes,
            heavy="reads the multi-GB inphared protein FASTA",
            run=_split_phages,
        ),
        Step(
            name="embed-bacteria",
            title="Embed the bacterial proteomes",
            description="ESM-C over every selected protein, averaged per proteome.",
            command="smartamr embed --kind bacteria",
            needs=[settings.bacteria_proteomes],
            produces=settings.bacteria_embeddings,
            heavy="needs the embedding extra and a GPU",
            run=_embed_bacteria,
        ),
        Step(
            name="embed-phages",
            title="Embed the phage proteomes",
            description="Same pass over the phage side.",
            command="smartamr embed --kind phage",
            needs=[settings.phage_proteomes],
            produces=settings.phage_embeddings,
            heavy="needs the embedding extra and a GPU",
            run=_embed_phages,
        ),
        Step(
            name="build-features",
            title="Merge the embeddings into the feature table",
            description="Pair every phage with every bacteria and label each pair.",
            command="smartamr build-features",
            needs=[settings.phage_embeddings, settings.bacteria_embeddings],
            produces=settings.features,
            run=_build_features,
        ),
        Step(
            name="train-relationship",
            title="Train: host-phage relationship classification",
            description="Does this phage infect this bacteria? (binary)",
            command=f"smartamr train --task binary --model {model_name}",
            needs=[settings.features],
            produces=settings.model_path(f"binary-{model_name}"),
            run=_train_relationship,
        ),
        Step(
            name="train-host",
            title="Train: host prediction",
            description="Which bacteria does this phage infect? (multilabel)",
            command=f"smartamr train --task multilabel --model {model_name}",
            needs=[settings.features],
            produces=settings.model_path(f"multilabel-{model_name}"),
            run=_train_host,
        ),
    ]


def _selected_faa_exists(settings: Settings) -> bool:
    return any(settings.bacteria_proteomes.glob("*.selected.faa"))


def step_status(step: Step, settings: Settings) -> str:
    """Status of a step, with the special cases the plain check cannot see."""
    if step.name == "extract-proteins":
        if _selected_faa_exists(settings):
            return DONE
        return READY if Step._exists(settings.bacteria_proteomes) else BLOCKED
    return step.status()


def run_pipeline(
    settings: Settings,
    steps: List[Step] | None = None,
    only: List[str] | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> Dict[str, str]:
    """Run the steps in order, skipping those already done.

    Returns the outcome per step: ``done``, ``skipped``, ``blocked`` or an
    error message.
    """

    steps = steps or pipeline_steps(settings)
    outcomes: Dict[str, str] = {}

    for step in steps:
        if only and step.name not in only:
            continue

        status = step_status(step, settings)

        if status == DONE and not force:
            print(f"[skip]  {step.title} - already done")
            outcomes[step.name] = "skipped"
            continue

        if status == BLOCKED:
            missing = ", ".join(str(path) for path in step.missing())
            print(f"[block] {step.title} - missing: {missing}")
            outcomes[step.name] = "blocked"
            continue

        if dry_run:
            print(f"[run]   {step.title}   ({step.command})")
            outcomes[step.name] = "would run"
            continue

        print(f"[run]   {step.title}")
        try:
            if step.run is None:  # pragma: no cover - every step has one
                raise NotImplementedError(step.name)
            step.run(settings)
            outcomes[step.name] = "done"
        except Exception as error:  # noqa: BLE001 - report and carry on
            print(f"[fail]  {step.title}: {error}")
            outcomes[step.name] = f"failed: {error}"
            break

    return outcomes
