"""The ``smartamr`` command line interface.

One subcommand per pipeline step, in the order of the README:

1. ``fetch-metadata`` / ``fetch-bacteria``  download from BV-BRC
2. ``filter-metadata``                      keep the rows with a valid phenotype
3. ``extract-proteins``                     subset proteomes on resistance proteins
4. ``select-viruses``                       pick the phages targeting each bacteria
5. ``embed``                                ESM-C embeddings of the proteomes
6. ``build-features``                       pair the embeddings and label them
7. ``train`` / ``evaluate``                 fit and score a classifier
8. ``predict``                              score a phage against a bacteria

Every command reads its paths from :class:`SmartAMR.config.Settings`, so the
defaults work from a clone and the environment variables (or the global flags)
cover everything else.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Sequence

from SmartAMR import __version__
from SmartAMR.config import Settings


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _settings_from_args(args: argparse.Namespace) -> Settings:
    """Build the Settings object from the global flags."""
    return Settings(
        root=args.root or Path.cwd(),
        data_dir=args.data_dir,
        config_dir=args.config_dir,
        output_dir=args.output_dir,
    )


def _collect_inputs(inputs: Sequence[str], pattern: str) -> List[Path]:
    """Expand the given files and directories into a sorted list of files."""
    files: List[Path] = []

    for item in inputs:
        path = Path(item)
        if path.is_dir():
            files.extend(sorted(path.rglob(pattern)))
        elif path.exists():
            files.append(path)
        else:
            files.extend(sorted(Path().glob(item)))

    if not files:
        raise SystemExit(f"No input file found in: {', '.join(inputs)}")

    return files


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------
def cmd_check_env(args: argparse.Namespace, settings: Settings) -> int:
    """Report the installed versions of the scientific dependencies."""
    from SmartAMR.environment import check_modules, format_report

    report = check_modules(include_optional=not args.required_only)

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(format_report(report))

    missing = [name for name, info in report.items() if info["version"] is None and info["required"] == "yes"]
    return 1 if missing else 0


def cmd_fetch_metadata(args: argparse.Namespace, settings: Settings) -> int:
    """Download the global PATRIC AMR metadata table."""
    from SmartAMR.data.scrapper import MetadataUpdater

    outdir = Path(args.outdir) if args.outdir else settings.data_dir
    settings.ensure(outdir)

    MetadataUpdater(outdir=outdir, wait=args.wait).download()
    print(f"Metadata written to {outdir}")
    return 0


def cmd_fetch_bacteria(args: argparse.Namespace, settings: Settings) -> int:
    """Download genome files for the bacteria listed in the config table."""
    from SmartAMR.data.scrapper import download_genomes
    from SmartAMR.data.viruses import read_bacteria_table

    outdir = Path(args.outdir) if args.outdir else settings.bacteria_proteomes
    settings.ensure(outdir)

    if args.genome_id:
        genome_ids = list(args.genome_id)
    else:
        table_path = Path(args.bacteria_table) if args.bacteria_table else settings.bacteria_table
        table = read_bacteria_table(table_path)
        genome_ids = table["bvbrc_id"].astype(str).tolist()
        print(f"{len(genome_ids)} genome(s) from {table_path}")

    count = download_genomes(genome_ids, outdir=outdir, file_type=args.file_type, wait=args.wait)
    print(f"Downloaded {args.file_type} files for {count} genome(s) into {outdir}")
    return 0


def cmd_filter_metadata(args: argparse.Namespace, settings: Settings) -> int:
    """Drop the metadata rows missing the filtering field."""
    from SmartAMR.data.filter import filter_dataset

    source = Path(args.input) if args.input else settings.amr_metadata
    df = filter_dataset(data=source, filter_field=args.field)

    print(f"{len(df)} row(s) with a valid {args.field}")
    print(f"{df['genome_id'].nunique()} distinct genome(s)")

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output, sep="\t", index=False)
        print(f"Written to {output}")

    return 0


def cmd_extract_proteins(args: argparse.Namespace, settings: Settings) -> int:
    """Subset proteomes on the proteins involved in viral resistance."""
    from SmartAMR.data.proteins import extract_proteins, read_protein_names

    names_path = Path(args.protein_names) if args.protein_names else settings.selected_proteins
    names = read_protein_names(names_path)
    print(f"{len(names)} protein name(s) from {names_path}")

    inputs = args.input or [str(settings.bacteria_proteomes)]
    proteomes = _collect_inputs(inputs, args.pattern)

    total = 0
    for proteome in proteomes:
        output = None
        if args.outdir:
            settings.ensure(Path(args.outdir))
            output = Path(args.outdir) / proteome.name.replace(".faa", ".selected.faa")
        out_path, kept = extract_proteins(proteome, names, output)
        print(f"{proteome.name}: {kept} record(s) -> {out_path}")
        total += kept

    print(f"{total} record(s) kept across {len(proteomes)} proteome(s)")
    return 0


def cmd_select_viruses(args: argparse.Namespace, settings: Settings) -> int:
    """Pick the phages that target each bacteria of the config table."""
    from SmartAMR.data.viruses import (
        read_bacteria_table,
        read_inphared_metadata,
        select_viruses,
        write_selection,
    )

    metadata_path = Path(args.metadata) if args.metadata else settings.inphared_metadata
    table_path = Path(args.bacteria_table) if args.bacteria_table else settings.bacteria_table

    metadata = read_inphared_metadata(metadata_path)
    bacteria = read_bacteria_table(table_path)["bacteria"].tolist()

    selected = select_viruses(metadata, bacteria, n_phages=args.n_phages, seed=args.seed)

    output = Path(args.output) if args.output else settings.selected_viruses
    list_output = None if args.no_list else (Path(args.list_output) if args.list_output else settings.virus_list)

    write_selection(selected, output, list_output, overwrite=args.force)

    print(f"{args.n_phages} phage(s) selected for {len(selected)} bacteria -> {output}")
    if list_output:
        print(f"Flat accession list -> {list_output}")

    return 0


def cmd_amr_genes(args: argparse.Namespace, settings: Settings) -> int:
    """List the AMR genes annotated for a genome."""
    from SmartAMR.genomics.amr import extract_AMR_genes_names

    directory = Path(args.directory) if args.directory else settings.bacteria_proteomes
    genes = extract_AMR_genes_names(file_path=directory, gene_type=args.gene_type, genome_id=args.genome_id)

    print(f"{len(genes)} gene(s) of type {args.gene_type!r} for {args.genome_id}")
    for gene in genes:
        print(f"  {gene}")

    return 0


def cmd_embed(args: argparse.Namespace, settings: Settings) -> int:
    """Embed proteomes with ESM-C and save the result as HDF5."""
    from SmartAMR.embedding.esmc import embed_proteomes, embedding_config, load_model, save_embeddings

    is_phage = args.kind == "phage"
    id_column = "phage_id" if is_phage else "bacteria_id"

    default_input = settings.phage_proteomes if is_phage else settings.bacteria_proteomes
    inputs = args.input or [str(default_input)]
    pattern = args.pattern or ("*.csv" if is_phage else "*.selected.faa")
    proteomes = _collect_inputs(inputs, pattern)
    print(f"{len(proteomes)} {args.kind} proteome(s) to embed")

    model = load_model(args.esm_model, device=args.device)
    config = embedding_config(hidden_layer=args.hidden_layer)

    df = embed_proteomes(proteomes, model=model, id_column=id_column, config=config)

    output = Path(args.output) if args.output else (
        settings.phage_embeddings if is_phage else settings.bacteria_embeddings
    )
    key = args.key or ("phage_embeddings" if is_phage else "bacteria_embeddings")
    save_embeddings(df, output, key=key)

    print(f"{len(df)} embedding(s) written to {output} (key: {key})")
    return 0


def cmd_build_features(args: argparse.Namespace, settings: Settings) -> int:
    """Pair the phage and bacteria embeddings and label every pair."""
    from SmartAMR.data.viruses import read_inphared_metadata
    from SmartAMR.embedding.esmc import load_embeddings
    from SmartAMR.embedding.features import build_features, save_features

    phage_path = Path(args.phage_embeddings) if args.phage_embeddings else settings.phage_embeddings
    bacteria_path = Path(args.bacteria_embeddings) if args.bacteria_embeddings else settings.bacteria_embeddings
    metadata_path = Path(args.metadata) if args.metadata else settings.inphared_metadata

    phages = load_embeddings(phage_path, key=args.phage_key)
    bacterias = load_embeddings(bacteria_path, key=args.bacteria_key)
    metadata = read_inphared_metadata(metadata_path)

    features = build_features(phages, bacterias, metadata)
    positives = int(features["label"].sum())
    print(f"{len(features)} pair(s): {positives} interacting, {len(features) - positives} not")

    output = Path(args.output) if args.output else settings.features
    save_features(features, output)
    print(f"Feature table written to {output}")

    return 0


def cmd_train(args: argparse.Namespace, settings: Settings) -> int:
    """Fit a classifier on the feature table and save it."""
    from SmartAMR.embedding.features import load_features
    from SmartAMR.models.classifiers import save_model, train

    features_path = Path(args.features) if args.features else settings.features
    features = load_features(features_path)

    model = train(
        features,
        model_name=args.model,
        task=args.task,
        test_size=args.test_size,
        seed=args.seed,
        balance=not args.no_balance,
    )

    print(f"\nAccuracy on {model.metrics['n_test']} held-out pair(s): {model.metrics['accuracy']:.3f}")
    print(model.metrics["report_text"])

    output = Path(args.output) if args.output else settings.model_path(f"{args.task}-{args.model}")
    save_model(model, output)
    print(f"Model saved to {output}")

    return 0


def cmd_evaluate(args: argparse.Namespace, settings: Settings) -> int:
    """Show the metrics of a saved model, or score it on a feature table."""
    from SmartAMR.models.classifiers import BINARY, load_model, prepare_binary, prepare_multilabel

    model = load_model(args.model)
    print(f"{model.model_name} ({model.task}), {len(model.feature_columns)} feature(s)")

    if args.features:
        from SmartAMR.embedding.features import load_features
        from SmartAMR.models.classifiers import evaluate

        features = load_features(args.features)
        if model.task == BINARY:
            X, y = prepare_binary(features)
        else:
            X, y = prepare_multilabel(features)
        metrics = evaluate(model, X, y)
    else:
        metrics = model.metrics
        if not metrics:
            raise SystemExit("This model carries no metrics; pass --features to score it.")

    print(f"\nAccuracy on {metrics['n_test']} pair(s): {metrics['accuracy']:.3f}")
    print(metrics["report_text"])

    if args.json:
        print(json.dumps({k: v for k, v in metrics.items() if k != "report_text"}, indent=2))

    return 0



def cmd_split_phages(args: argparse.Namespace, settings: Settings) -> int:
    """Build one proteome per phage from the inphared protein FASTA."""
    from SmartAMR.data.phages import proteomes_to_csv, split_phage_proteomes
    from SmartAMR.stats import selected_accessions

    source = Path(args.protein_fasta) if args.protein_fasta else settings.phage_protein_fasta
    outdir = Path(args.outdir) if args.outdir else settings.phage_proteomes

    accessions = None
    if not args.all:
        selection = selected_accessions(
            Path(args.selection) if args.selection else settings.selected_viruses
        )
        if not selection:
            raise FileNotFoundError(
                "No phage selection found. Run 'smartamr select-viruses', or pass --all."
            )
        accessions = sorted({a for values in selection.values() for a in values})
        print(f"{len(accessions)} selected accession(s)")

    names = None
    if args.protein_names:
        from SmartAMR.data.proteins import read_protein_names

        names = read_protein_names(args.protein_names)

    counts = split_phage_proteomes(source, outdir, accessions=accessions, protein_names=names)
    print(f"{len(counts)} phage proteome(s) written to {outdir}")

    if counts:
        total = sum(counts.values())
        print(f"{total:,} protein(s), {total / len(counts):.0f} per phage on average")

    if args.csv:
        written = proteomes_to_csv(outdir)
        print(f"{len(written)} CSV copy(ies) written")

    return 0


def cmd_pipeline(args: argparse.Namespace, settings: Settings) -> int:
    """Run the pipeline as the README documents it."""
    from SmartAMR.pipeline import pipeline_steps, run_pipeline, step_status

    steps = pipeline_steps(settings, model_name=args.model)

    if args.status:
        width = max(len(step.title) for step in steps)
        print(f"{'STEP':<{width}}  STATUS   COMMAND")
        for step in steps:
            print(f"{step.title:<{width}}  {step_status(step, settings):<8} {step.command}")
        return 0

    only = args.only or None
    if args.start:
        names = [step.name for step in steps]
        if args.start not in names:
            raise ValueError(f"Unknown step {args.start!r}. Choose from: {', '.join(names)}")
        only = names[names.index(args.start):]

    outcomes = run_pipeline(settings, steps=steps, only=only, force=args.force, dry_run=args.dry_run)

    failed = [name for name, outcome in outcomes.items() if outcome.startswith("failed")]
    blocked = [name for name, outcome in outcomes.items() if outcome == "blocked"]

    print()
    if failed:
        print(f"Stopped at: {', '.join(failed)}")
    if blocked:
        print(f"Blocked (inputs missing): {', '.join(blocked)}")
    if not failed and not blocked:
        print("Every step is done.")

    return 1 if failed else 0


def cmd_demo(args: argparse.Namespace, settings: Settings) -> int:
    """Generate a demo feature table so the dashboard can be explored."""
    from SmartAMR.demo import build_demo_features, clear_demo

    if args.clear:
        removed = clear_demo(settings)
        if not removed:
            print("Nothing to clear.")
        for path in removed:
            print(f"Removed {path}")
        return 0

    path = build_demo_features(
        settings,
        dim=args.dim,
        phages_per_bacteria=args.phages,
        seed=args.seed,
        separation=args.separation,
        confusion_rate=args.confusion,
    )

    print(f"\nWritten to {path}")
    print("The labels are real; the embeddings are synthetic, so no number here is a result.")
    print("Explore it with: smartamr app    ·    undo with: smartamr demo --clear")

    return 0


def cmd_figures(args: argparse.Namespace, settings: Settings) -> int:
    """Write the project's graphs to image files."""
    from SmartAMR.viz.theme import DARK, LIGHT
    from SmartAMR.figures import save_figures

    formats = ["png", "svg"] if args.format == "both" else [args.format]
    outdir = Path(args.outdir) if args.outdir else settings.output_dir / "figures"

    written, skipped = save_figures(
        settings,
        outdir=outdir,
        formats=formats,
        theme=DARK if args.dark else LIGHT,
        scale=args.scale,
        include_projection=not args.no_projection,
        projection_method=args.projection,
        include_experiments=args.experiments,
    )

    for note in skipped:
        print(f"Skipped {note}")

    if not written:
        print("No figure could be built - see the notes above.")
        return 1

    print(f"\n{len(written)} file(s) written to {outdir}")
    for path in written:
        print(f"  {path.name}")

    return 0


def cmd_screen(args: argparse.Namespace, settings: Settings) -> int:
    """Screen one or more new phage proteomes against the project's bacteria."""
    from SmartAMR.models.screen import results_to_frame, screen_proteomes

    proteomes = _collect_inputs(args.proteome, args.pattern)
    print(f"{len(proteomes)} proteome(s) to screen")

    esm_model = None
    if args.model and not args.no_embed:
        from SmartAMR.embedding.esmc import load_model as load_esm

        esm_model = load_esm(args.esm_model, device=args.device)

    bacteria_embeddings = None
    path = Path(args.bacteria_embeddings) if args.bacteria_embeddings else settings.bacteria_embeddings
    if path.exists():
        from SmartAMR.embedding.esmc import load_embeddings

        bacteria_embeddings = load_embeddings(path, key=args.bacteria_key)

    settings.ensure(settings.embeddings_dir / "uploaded")
    results = screen_proteomes(
        proteomes,
        model=args.model,
        esm_model=esm_model,
        bacteria_embeddings=bacteria_embeddings,
        cache_dir=settings.embeddings_dir / "uploaded",
    )

    frame = results_to_frame(results, top_n=args.top)

    if any(result.is_placeholder for result in results):
        from SmartAMR.models.predict import PLACEHOLDER_WARNING

        print(f"WARNING: {PLACEHOLDER_WARNING}\n")

    if args.json:
        print(frame.to_json(orient="records", indent=2))
    else:
        print(frame.to_string(index=False))

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output, index=False)
        print(f"\nWritten to {output}")

    return 0


def cmd_dashboard(args: argparse.Namespace, settings: Settings) -> int:
    """Deprecated alias of ``smartamr app``.

    There used to be two Streamlit front ends: this dashboard over the demo
    pipeline, and the app over the reproduced one. They have been merged into
    :mod:`SmartAMR.webapp`, so this launches that and says so.
    """

    print(
        "Note: 'smartamr dashboard' is deprecated - the dashboard and the app are "
        "now one front end. Launching 'smartamr app' instead.",
        file=sys.stderr,
    )
    args.fraction = getattr(args, "fraction", None)
    return cmd_app(args, settings)


def _streamlit_env(settings: Settings) -> dict:
    """Environment for a Streamlit subprocess, carrying the resolved paths.

    The app resolves its own ``Settings`` in a separate process, so the global
    path flags have to travel as environment variables or the app would read a
    different project than the CLI was pointed at.
    """

    env = dict(os.environ)
    env["SMARTAMR_ROOT"] = str(settings.root)
    env["SMARTAMR_DATA_DIR"] = str(settings.data_dir)
    env["SMARTAMR_CONFIG_DIR"] = str(settings.config_dir)
    env["SMARTAMR_OUTPUT_DIR"] = str(settings.output_dir)
    env["SMARTAMR_GENOMES_DIR"] = str(settings.phage_genomes)
    return env


def cmd_app(args: argparse.Namespace, settings: Settings) -> int:
    """Launch the SmartAMR web app."""
    import subprocess

    try:
        import streamlit  # noqa: F401
    except ImportError as error:
        raise ImportError(
            "Streamlit is required for the app. Install it with: pip install streamlit"
        ) from error

    app = Path(__file__).resolve().parent / "webapp" / "main.py"
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app),
        "--server.port",
        str(args.port),
    ]
    if args.headless:
        command += ["--server.headless", "true"]

    env = _streamlit_env(settings)
    if args.fraction is not None:
        env["SMARTAMR_RESOURCE_FRACTION"] = str(args.fraction)

    print(f"Starting the SmartAMR app on http://localhost:{args.port}")
    return subprocess.call(command, env=env)


def cmd_reproduce(args: argparse.Namespace, settings: Settings) -> int:
    """Rebuild the dataset, embeddings and models from the raw inputs."""
    from SmartAMR.reproduce.runner import STAGES, run

    stages = tuple(args.stages) if args.stages else STAGES
    record = run(
        settings=settings,
        stages=stages,
        pooling=args.pooling,
        per_host=args.per_host,
        use_filtered=not args.whole_proteomes,
        group_by_phage=not args.rowwise_split,
        fraction=args.fraction,
        force=args.force,
    )

    print("\n== summary")
    for key, value in record.counts.items():
        print(f"  {key}: {value}")
    for note in record.notes:
        print(f"  note: {note}")

    if record.failures:
        print(f"\n{len(record.failures)} failure(s):")
        for failure in record.failures:
            print(f"  {failure}")
        return 1
    return 0


def cmd_predict(args: argparse.Namespace, settings: Settings) -> int:
    """Score a phage against a bacteria, or against a whole database."""
    from SmartAMR.fasta import iter_fasta, retrieve_sequence
    from SmartAMR.models.predict import (
        PLACEHOLDER_WARNING,
        evaluate_sequences,
        predictions_to_frame,
        screen_database,
    )

    def resolve_sequence(sequence: str | None, fasta: str | None, name: str | None) -> tuple[str, str]:
        """Return ``(name, sequence)`` from the flags given."""
        if sequence:
            return name or "query", sequence
        if fasta:
            records = list(iter_fasta(fasta))
            if name:
                found = retrieve_sequence(fasta, name)
                if found is None:
                    raise SystemExit(f"{name!r} not found in {fasta}")
                return name, found
            if not records:
                raise SystemExit(f"No record in {fasta}")
            return records[0]
        return name or "", ""

    phage_name, phage_sequence = resolve_sequence(args.phage_sequence, args.phage_fasta, args.phage)
    bacteria_name, bacteria_sequence = resolve_sequence(
        args.bacteria_sequence, args.bacteria_fasta, args.bacteria
    )

    if not phage_sequence and not bacteria_sequence:
        raise SystemExit("Provide at least a phage or a bacteria sequence.")

    esm_model = None
    if args.model and args.embed:
        from SmartAMR.embedding.esmc import load_model as load_esm

        esm_model = load_esm(args.esm_model, device=args.device)

    if phage_sequence and bacteria_sequence:
        predictions = [
            evaluate_sequences(
                bacteria_sequence=bacteria_sequence,
                phage_sequence=phage_sequence,
                model=args.model,
                esm_model=esm_model,
                phage=phage_name,
                bacteria=bacteria_name,
            )
        ]
    else:
        query_is_phage = bool(phage_sequence)
        database = args.database
        if database is None:
            raise SystemExit("Only one sequence was given; pass --database to screen against it.")
        predictions = screen_database(
            query_name=phage_name if query_is_phage else bacteria_name,
            query_sequence=phage_sequence or bacteria_sequence,
            database=database,
            query_is_phage=query_is_phage,
            model=args.model,
            esm_model=esm_model,
        )

    frame = predictions_to_frame(predictions)

    if any(prediction.is_placeholder for prediction in predictions):
        print(f"WARNING: {PLACEHOLDER_WARNING}\n")

    if args.json:
        print(json.dumps([prediction.as_dict() for prediction in predictions], indent=2))
    else:
        print(frame.to_string(index=False))

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output, index=False)
        print(f"\nWritten to {output}")

    return 0


# --------------------------------------------------------------------------
# Parser
# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """Build the top level parser and every subcommand."""

    parser = argparse.ArgumentParser(
        prog="smartamr",
        description="Circumvent antibioresistance with bacteriophage selection.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"smartamr {__version__}")
    parser.add_argument("--root", type=Path, default=None, help="Project root (default: current directory)")
    parser.add_argument("--data-dir", type=Path, default=None, help="Where the downloaded data lives")
    parser.add_argument("--config-dir", type=Path, default=None, help="Where the reference tables live")
    parser.add_argument("--output-dir", type=Path, default=None, help="Where embeddings and models are written")

    sub = parser.add_subparsers(dest="command", metavar="<command>")

    # check-env
    p = sub.add_parser("check-env", help="Report installed dependency versions")
    p.add_argument("--required-only", action="store_true", help="Skip the optional packages")
    p.add_argument("--json", action="store_true", help="Print the report as JSON")
    p.set_defaults(func=cmd_check_env)

    # fetch-metadata
    p = sub.add_parser("fetch-metadata", help="Download the PATRIC AMR metadata table")
    p.add_argument("--outdir", help="Destination directory")
    p.add_argument("--wait", type=int, default=1, help="Seconds between requests")
    p.set_defaults(func=cmd_fetch_metadata)

    # fetch-bacteria
    p = sub.add_parser("fetch-bacteria", help="Download genome files from BV-BRC")
    p.add_argument("--genome-id", action="append", help="BV-BRC id (repeatable); defaults to the config table")
    p.add_argument("--bacteria-table", help="CSV of bacteria and their BV-BRC id")
    p.add_argument(
        "--file-type",
        default="protein",
        choices=[
            "contig",
            "protein",
            "features",
            "feature-seq",
            "rna",
            "annotation",
            "pathway",
            "speciality-gene",
            "subsystem",
            "all",
        ],
        help="Which per-genome file to fetch (default: protein)",
    )
    p.add_argument("--outdir", help="Destination directory")
    p.add_argument("--wait", type=int, default=1, help="Seconds between requests")
    p.set_defaults(func=cmd_fetch_bacteria)

    # filter-metadata
    p = sub.add_parser("filter-metadata", help="Keep the metadata rows with a valid phenotype")
    p.add_argument("--input", help="Metadata TSV to filter")
    p.add_argument("--field", default="resistant_phenotype", help="Column whose missing values are dropped")
    p.add_argument("--output", help="Write the filtered table here")
    p.set_defaults(func=cmd_filter_metadata)

    # extract-proteins
    p = sub.add_parser("extract-proteins", help="Subset proteomes on resistance proteins")
    p.add_argument("input", nargs="*", help="Proteome files or directories")
    p.add_argument("--protein-names", help="File listing the protein names to keep")
    p.add_argument("--pattern", default="*.PATRIC.faa", help="Glob used when an input is a directory")
    p.add_argument("--outdir", help="Write the subsets here instead of beside the inputs")
    p.set_defaults(func=cmd_extract_proteins)

    # select-viruses
    p = sub.add_parser("select-viruses", help="Pick the phages targeting each bacteria")
    p.add_argument("--metadata", help="inphared metadata TSV")
    p.add_argument("--bacteria-table", help="CSV of bacteria and their BV-BRC id")
    p.add_argument("--n-phages", type=int, default=100, help="Phages drawn per bacteria (default: 100)")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    p.add_argument("--output", help="Destination CSV")
    p.add_argument("--list-output", help="Destination of the flat accession list")
    p.add_argument("--no-list", action="store_true", help="Skip the flat accession list")
    p.add_argument("--force", action="store_true", help="Replace an existing selection")
    p.set_defaults(func=cmd_select_viruses)

    # amr-genes
    p = sub.add_parser("amr-genes", help="List the AMR genes annotated for a genome")
    p.add_argument("genome_id", help="BV-BRC genome id")
    p.add_argument("--directory", help="Directory holding the genome files")
    p.add_argument("--gene-type", default="Antibiotic Resistance", help="Value of the 'property' column")
    p.set_defaults(func=cmd_amr_genes)

    # embed
    p = sub.add_parser("embed", help="Embed proteomes with ESM-C")
    p.add_argument("--kind", choices=["phage", "bacteria"], required=True, help="Which side to embed")
    p.add_argument("input", nargs="*", help="Proteome files or directories")
    p.add_argument("--pattern", default=None, help="Glob used when an input is a directory")
    p.add_argument("--esm-model", default="esmc_300m", help="Pretrained ESM-C model (default: esmc_300m)")
    p.add_argument("--device", default="cuda", help="Torch device (default: cuda, falls back to cpu)")
    p.add_argument("--hidden-layer", type=int, default=100, help="Hidden layer to request")
    p.add_argument("--output", help="Destination HDF5 file")
    p.add_argument("--key", default=None, help="HDF5 key")
    p.set_defaults(func=cmd_embed)

    # build-features
    p = sub.add_parser("build-features", help="Pair the embeddings and label every pair")
    p.add_argument("--phage-embeddings", help="HDF5 of phage embeddings")
    p.add_argument("--bacteria-embeddings", help="HDF5 of bacteria embeddings")
    p.add_argument("--phage-key", default="phage_embeddings", help="HDF5 key of the phage frame")
    p.add_argument("--bacteria-key", default="bacteria_embeddings", help="HDF5 key of the bacteria frame")
    p.add_argument("--metadata", help="inphared metadata TSV, used for the labels")
    p.add_argument("--output", help="Destination CSV")
    p.set_defaults(func=cmd_build_features)

    # train
    p = sub.add_parser("train", help="Fit a classifier and save it")
    p.add_argument("--features", help="Feature table CSV")
    p.add_argument(
        "--model",
        default="random-forest",
        choices=["random-forest", "svm", "xgboost", "adaboost"],
        help="Estimator (default: random-forest)",
    )
    p.add_argument(
        "--task",
        default="binary",
        choices=["binary", "multilabel"],
        help="binary: does this phage infect this bacteria; multilabel: which host (default: binary)",
    )
    p.add_argument("--test-size", type=float, default=0.2, help="Held-out fraction (default: 0.2)")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    p.add_argument("--no-balance", action="store_true", help="Skip the majority class downsampling")
    p.add_argument("--output", help="Destination model file")
    p.set_defaults(func=cmd_train)

    # evaluate
    p = sub.add_parser("evaluate", help="Show or recompute the metrics of a saved model")
    p.add_argument("model", help="Saved model file")
    p.add_argument("--features", help="Score the model on this feature table instead")
    p.add_argument("--json", action="store_true", help="Also print the metrics as JSON")
    p.set_defaults(func=cmd_evaluate)

    # predict
    p = sub.add_parser("predict", help="Score a phage against a bacteria")
    p.add_argument("--phage", help="Phage name (looked up in --phage-fasta)")
    p.add_argument("--phage-sequence", help="Phage protein sequence")
    p.add_argument("--phage-fasta", help="FASTA holding the phage sequence")
    p.add_argument("--bacteria", help="Bacteria name (looked up in --bacteria-fasta)")
    p.add_argument("--bacteria-sequence", help="Bacteria protein sequence")
    p.add_argument("--bacteria-fasta", help="FASTA holding the bacteria sequence")
    p.add_argument("--database", help="Screen the given sequence against this FASTA")
    p.add_argument("--model", help="Trained model; without it the result is a random placeholder")
    p.add_argument("--embed", action="store_true", help="Embed the sequences with ESM-C before predicting")
    p.add_argument("--esm-model", default="esmc_300m", help="Pretrained ESM-C model")
    p.add_argument("--device", default="cuda", help="Torch device")
    p.add_argument("--output", help="Write the results to this CSV")
    p.add_argument("--json", action="store_true", help="Print the results as JSON")
    p.set_defaults(func=cmd_predict)

    # screen
    p = sub.add_parser("screen", help="Screen new phage proteomes against the bacteria")
    p.add_argument("proteome", nargs="+", help="Phage proteome files or directories (FASTA or CSV)")
    p.add_argument("--pattern", default="*.faa", help="Glob used when an input is a directory")
    p.add_argument("--model", help="Trained model; a multilabel one ranks the hosts directly")
    p.add_argument("--bacteria-embeddings", help="HDF5 of bacteria embeddings (binary models need it)")
    p.add_argument("--bacteria-key", default="bacteria_embeddings", help="HDF5 key of the bacteria frame")
    p.add_argument("--no-embed", action="store_true", help="Skip ESM-C; returns placeholders")
    p.add_argument("--esm-model", default="esmc_300m", help="Pretrained ESM-C model")
    p.add_argument("--device", default="cuda", help="Torch device")
    p.add_argument("--top", type=int, default=None, help="Keep only the N best hosts per phage")
    p.add_argument("--output", help="Write the ranking to this CSV")
    p.add_argument("--json", action="store_true", help="Print the ranking as JSON")
    p.set_defaults(func=cmd_screen)

    # split-phages
    p = sub.add_parser("split-phages", help="Build one proteome per phage from the inphared FASTA")
    p.add_argument("--protein-fasta", help="The inphared protein FASTA")
    p.add_argument("--selection", help="CSV of selected accessions")
    p.add_argument("--outdir", help="Destination directory")
    p.add_argument("--protein-names", help="Keep only proteins matching these names")
    p.add_argument("--all", action="store_true", help="Split every phage, not just the selected ones")
    p.add_argument("--csv", action="store_true", help="Also write the CSV form of each proteome")
    p.set_defaults(func=cmd_split_phages)

    # pipeline
    p = sub.add_parser("pipeline", help="Run the whole pipeline as the README documents it")
    p.add_argument("--status", action="store_true", help="Show what is done, ready and blocked")
    p.add_argument("--dry-run", action="store_true", help="Say what would run, run nothing")
    p.add_argument("--force", action="store_true", help="Re-run steps whose output already exists")
    p.add_argument("--start", help="Begin at this step and run the rest")
    p.add_argument("--only", action="append", help="Run only this step (repeatable)")
    p.add_argument("--model", default="xgboost", help="Estimator for the two training steps")
    p.set_defaults(func=cmd_pipeline)

    # demo
    p = sub.add_parser("demo", help="Generate a demo feature table (real labels, synthetic embeddings)")
    p.add_argument("--dim", type=int, default=64, help="Width of each synthetic embedding")
    p.add_argument("--phages", type=int, default=40, help="Phages kept per bacteria")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument(
        "--separation",
        type=float,
        default=1.0,
        help="How far apart the per-host centres sit; lower makes the task harder",
    )
    p.add_argument(
        "--confusion",
        type=float,
        default=0.35,
        help="Share of phages pulled toward a second host, so the classes overlap",
    )
    p.add_argument("--clear", action="store_true", help="Delete the demo table and its marker")
    p.set_defaults(func=cmd_demo)

    # figures
    p = sub.add_parser("figures", help="Write the project's graphs to image files")
    p.add_argument("--outdir", help="Destination directory (default: <output>/figures)")
    p.add_argument("--format", default="png", choices=["png", "svg", "both"], help="Image format")
    p.add_argument("--scale", type=int, default=2, help="PNG scale factor (default: 2)")
    p.add_argument("--dark", action="store_true", help="Render with the dark palette")
    p.add_argument("--projection", default="pca", choices=["pca", "tsne", "umap"], help="Projection method")
    p.add_argument("--no-projection", action="store_true", help="Skip the embedding projection")
    p.add_argument(
        "--experiments",
        action="store_true",
        help="Also fit every estimator and export the ROC, DET, score and silhouette figures",
    )
    p.set_defaults(func=cmd_figures)

    # dashboard (deprecated alias of `app`)
    p = sub.add_parser(
        "dashboard",
        help="Deprecated: launches 'smartamr app', which the dashboard merged into",
    )
    p.add_argument("--port", type=int, default=8501, help="Port to serve on (default: 8501)")
    p.add_argument("--headless", action="store_true", help="Do not open a browser")
    p.set_defaults(func=cmd_dashboard)

    # app
    p = sub.add_parser("app", help="Launch the SmartAMR web app")
    p.add_argument("--port", type=int, default=8501, help="Port to serve on (default: 8501)")
    p.add_argument("--headless", action="store_true", help="Do not open a browser")
    p.add_argument(
        "--fraction",
        type=float,
        default=None,
        help="Share of RAM/GPU/cores the app may use (default: 0.5)",
    )
    p.set_defaults(func=cmd_app)

    # reproduce
    p = sub.add_parser(
        "reproduce",
        help="Rebuild dataset, embeddings and models from the raw inputs",
        description=(
            "Runs the stages of SmartAMR.reproduce in order: select phages, fetch "
            "and filter the bacterial proteomes, embed both sides with ESM-C, pair "
            "them into features, train the classifiers. Each stage writes to disk "
            "and can be run on its own."
        ),
    )
    p.add_argument(
        "--stages",
        nargs="+",
        choices=("catalog", "fetch", "filter", "embed", "features", "train"),
        default=None,
        help="Stages to run (default: all of them, in order)",
    )
    p.add_argument(
        "--per-host",
        type=int,
        default=100,
        help="Phages to sample per bacterial host (default: 100)",
    )
    p.add_argument(
        "--pooling",
        choices=("residue", "legacy"),
        default="residue",
        help=(
            "How a protein's per-residue matrix becomes one vector. 'residue' "
            "averages over residues (960 dims, comparable across runs); 'legacy' "
            "reproduces the original scripts' feature-axis mean (default: residue)"
        ),
    )
    p.add_argument(
        "--whole-proteomes",
        action="store_true",
        help="Embed the full bacterial proteomes instead of the resistance-protein subset",
    )
    p.add_argument(
        "--rowwise-split",
        action="store_true",
        help=(
            "Split train/test by row as the notebooks did, instead of grouping by "
            "phage. Lets the same phage appear on both sides of the split"
        ),
    )
    p.add_argument(
        "--fraction",
        type=float,
        default=None,
        help="Share of RAM/GPU/cores to use (default: 0.5, or SMARTAMR_RESOURCE_FRACTION)",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-download and re-embed instead of reusing what is on disk",
    )
    p.set_defaults(func=cmd_reproduce)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point of the ``smartamr`` command."""

    parser = build_parser()
    args = parser.parse_args(argv)

    if not getattr(args, "func", None):
        parser.print_help()
        return 1

    settings = _settings_from_args(args)

    try:
        return args.func(args, settings)
    except KeyboardInterrupt:  # pragma: no cover
        print("\nInterrupted")
        return 130
    except (FileNotFoundError, FileExistsError, ValueError, ImportError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
