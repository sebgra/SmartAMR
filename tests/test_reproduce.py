"""Tests for the reproduction stages that need neither a GPU nor the network."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from SmartAMR.config import INPHARED_ACCESSION_COLUMN, INPHARED_HOST_COLUMN, Settings
from SmartAMR.reproduce import catalog, embed, features, hosts, proteomes

# ---------------------------------------------------------------------------
# Fixtures: a miniature project on disk
# ---------------------------------------------------------------------------
METADATA_ROWS = [
    # accession, isolation host as free text
    ("PH0001", "Escherichia coli K-12"),
    ("PH0002", "Escherichia coli"),
    ("PH0003", "escherichia coli C"),  # lower case: must still match
    ("PH0004", "Salmonella enterica"),
    ("PH0005", "Klebsiella pneumoniae"),
    ("PH0006", "Homo sapiens"),  # not one of the nine hosts
    ("PH0007", ""),  # empty host, the column warns these exist
]


@pytest.fixture()
def project(tmp_path):
    """A Settings pointing at a temporary project with a metadata file."""
    (tmp_path / "data").mkdir()
    (tmp_path / "config").mkdir()
    genomes = tmp_path / "GenomesDB"
    genomes.mkdir()

    frame = pd.DataFrame(
        METADATA_ROWS, columns=[INPHARED_ACCESSION_COLUMN, INPHARED_HOST_COLUMN]
    )
    frame.to_csv(
        tmp_path / "data" / "14Apr2025_data_excluding_refseq.tsv", sep="\t", index=False
    )

    # Proteomes for a subset only: selection must notice the difference.
    for accession in ("PH0001", "PH0002", "PH0004", "PH0005"):
        directory = genomes / accession
        directory.mkdir()
        (directory / f"{accession}.faa").write_text(
            f">{accession}_prot1 OmpA outer membrane\nMKKTAIAIAVALAG\n", encoding="utf-8"
        )

    settings = Settings(root=tmp_path)
    monkey_genomes = genomes
    # phage_genomes reads an env var, so point it at the fixture.
    import os

    os.environ["SMARTAMR_GENOMES_DIR"] = str(monkey_genomes)
    yield settings
    os.environ.pop("SMARTAMR_GENOMES_DIR", None)


# ---------------------------------------------------------------------------
# Hosts
# ---------------------------------------------------------------------------
def test_there_are_nine_hosts():
    assert len(hosts.HOSTS) == 9


def test_ecoli_uses_the_corrected_taxon():
    # 511145 returns 8 proteins from UniProt, which is not a proteome; 83333 is
    # the id that organism is actually annotated under.
    ecoli = hosts.host_by_name("Escherichia coli")
    assert ecoli.uniprot_taxon == 83333
    assert ecoli.legacy_uniprot_taxon == 511145
    assert ecoli.corrected


def test_the_correction_is_discoverable():
    corrected = hosts.corrections()
    assert [host.inphared_name for host in corrected] == ["Escherichia coli"]
    assert corrected[0].note


def test_every_other_host_keeps_its_original_taxon():
    for host in hosts.HOSTS:
        if host.inphared_name != "Escherichia coli":
            assert not host.corrected


def test_hosts_are_unique_by_taxon_and_name():
    assert len({host.uniprot_taxon for host in hosts.HOSTS}) == 9
    assert len({host.inphared_name for host in hosts.HOSTS}) == 9


def test_an_unknown_host_is_an_error():
    with pytest.raises(KeyError):
        hosts.host_by_name("Bacillus subtilis")


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------
def test_host_matching_is_case_insensitive_substring(project):
    frame = catalog.read_metadata(project)
    mask = catalog.host_matches(frame, hosts.host_by_name("Escherichia coli"))
    matched = set(frame.loc[mask, INPHARED_ACCESSION_COLUMN])
    # PH0003 is spelled "escherichia coli C" and must still match.
    assert matched == {"PH0001", "PH0002", "PH0003"}


def test_an_empty_host_cell_matches_nothing(project):
    frame = catalog.read_metadata(project)
    for host in hosts.HOSTS:
        assert "PH0007" not in set(
            frame.loc[catalog.host_matches(frame, host), INPHARED_ACCESSION_COLUMN]
        )


def test_available_proteomes_finds_the_per_accession_layout(project):
    found = catalog.available_proteomes(project.phage_genomes)
    assert set(found) == {"PH0001", "PH0002", "PH0004", "PH0005"}


def test_available_proteomes_also_finds_a_flat_directory(tmp_path):
    (tmp_path / "AB0001.faa").write_text(">x\nMK\n", encoding="utf-8")
    assert set(catalog.available_proteomes(tmp_path)) == {"AB0001"}


def test_available_proteomes_of_a_missing_directory_is_empty(tmp_path):
    assert catalog.available_proteomes(tmp_path / "nope") == {}


def test_selection_only_picks_phages_with_a_proteome(project):
    selection, reports = catalog.select_phages(project, per_host=100)

    # PH0003 matches E. coli in the metadata but has no proteome on disk.
    assert set(selection["phage_id"]) == {"PH0001", "PH0002", "PH0004", "PH0005"}
    assert all(row.proteome_path for row in selection.itertuples())

    ecoli = next(report for report in reports if report.host == "Escherichia coli")
    assert ecoli.in_metadata == 3
    assert ecoli.with_proteome == 2
    assert ecoli.selected == 2


def test_selection_without_the_proteome_requirement_takes_everything(project):
    selection, _ = catalog.select_phages(project, per_host=100, require_proteome=False)
    assert "PH0003" in set(selection["phage_id"])


def test_selection_respects_the_per_host_cap(project):
    selection, _ = catalog.select_phages(project, per_host=1)
    by_host = selection.groupby("host").size()
    assert by_host.max() == 1


def test_selection_is_reproducible(project):
    first, _ = catalog.select_phages(project, per_host=1, seed=7)
    second, _ = catalog.select_phages(project, per_host=1, seed=7)
    pd.testing.assert_frame_equal(first, second)


def test_pairs_are_the_cross_product_with_real_labels(project):
    selection, _ = catalog.select_phages(project, per_host=100)
    pairs = catalog.build_pairs(project, selection)

    # 4 phages x 9 hosts, every phage against every host.
    assert len(pairs) == 4 * 9
    assert pairs["bacteria_id"].nunique() == 9

    # Positives are exactly the metadata's own host assignments.
    positives = {
        (row.phage_id, row.bacteria_id) for row in pairs.itertuples() if row.label
    }
    assert positives == {
        ("PH0001", "Escherichia coli"),
        ("PH0002", "Escherichia coli"),
        ("PH0004", "Salmonella"),
        ("PH0005", "Klebsiella pneumoniae"),
    }


def test_pairs_have_negatives(project):
    """The original scripts' label check could only return True; this cannot."""
    selection, _ = catalog.select_phages(project, per_host=100)
    pairs = catalog.build_pairs(project, selection)
    assert (~pairs["label"]).sum() == len(pairs) - 4


def test_write_catalog_puts_both_tables_on_disk(project):
    selection, pairs, _ = catalog.write_catalog(project, per_host=100)
    assert project.selected_phages.exists()
    assert project.phage_pairs.exists()
    assert len(pd.read_csv(project.selected_phages)) == len(selection)
    assert len(pd.read_csv(project.phage_pairs)) == len(pairs)


def test_missing_metadata_is_an_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="INPHARED metadata"):
        catalog.read_metadata(Settings(root=tmp_path))


# ---------------------------------------------------------------------------
# FASTA parsing and filtering
# ---------------------------------------------------------------------------
def test_iter_fasta_reads_whole_records(tmp_path):
    path = tmp_path / "p.faa"
    path.write_text(">a desc\nMKK\nTAI\n>b\nQQQ\n", encoding="utf-8")
    records = list(proteomes.iter_fasta(path))
    assert records == [(">a desc", ["MKK", "TAI"]), (">b", ["QQQ"])]


def test_iter_fasta_tolerates_a_leading_blank_line(tmp_path):
    path = tmp_path / "p.faa"
    path.write_text("\n>a\nMK\n", encoding="utf-8")
    assert list(proteomes.iter_fasta(path)) == [(">a", ["MK"])]


def test_the_filter_keeps_the_sequence_not_just_the_header(tmp_path):
    """The point of reimplementing the awk script: whole records come through."""
    source = tmp_path / "in.fasta"
    source.write_text(
        ">sp|P1|OmpA outer membrane\nMKKTAIA\nIAVALAG\n"
        ">sp|P2|Ribosomal protein\nQQQQQ\n"
        ">sp|P3|Cas9 nuclease\nWWWW\n",
        encoding="utf-8",
    )
    destination = tmp_path / "out.fasta"
    kept = proteomes.filter_proteome(source, destination, ["OmpA", "Cas9"])

    assert kept == 2
    text = destination.read_text(encoding="utf-8")
    assert "MKKTAIA" in text and "IAVALAG" in text  # sequence lines survive
    assert "QQQQQ" not in text


def test_the_filter_is_case_insensitive(tmp_path):
    source = tmp_path / "in.fasta"
    source.write_text(">sp|P1|ompA lower case\nMK\n", encoding="utf-8")
    assert proteomes.filter_proteome(source, tmp_path / "out.fasta", ["OmpA"]) == 1


def test_the_filter_treats_names_literally_not_as_regex(tmp_path):
    """awk's ``$0 ~ name`` would let "O.pA" match "OmpA"; substring must not."""
    source = tmp_path / "in.fasta"
    source.write_text(">sp|P1|OmpA\nMK\n", encoding="utf-8")
    assert proteomes.filter_proteome(source, tmp_path / "out.fasta", ["O.pA"]) == 0


def test_a_record_with_no_sequence_is_dropped(tmp_path):
    source = tmp_path / "in.fasta"
    source.write_text(">sp|P1|OmpA\n>sp|P2|OmpA real\nMK\n", encoding="utf-8")
    assert proteomes.filter_proteome(source, tmp_path / "out.fasta", ["OmpA"]) == 1


def test_count_records_counts_headers(tmp_path):
    path = tmp_path / "p.fasta"
    path.write_text(">a\nMK\n>b\nQQ\n", encoding="utf-8")
    assert proteomes.count_records(path) == 2
    assert proteomes.count_records(tmp_path / "missing.fasta") == 0


def test_protein_names_ignore_blanks_and_comments(tmp_path):
    path = tmp_path / "names.txt"
    path.write_text("OmpA\n\n# a comment\n  TolC  \n", encoding="utf-8")
    assert proteomes.read_protein_names(path) == ["OmpA", "TolC"]


def test_a_filtered_subset_is_never_flagged_as_suspicious():
    """A small record count is the goal of the filter, not a warning sign."""
    report = proteomes.ProteomeReport(
        host="x", taxon=1, path=None, records=20, is_subset=True
    )
    assert report.ok
    assert not report.suspicious


def test_a_small_download_is_flagged_as_suspicious():
    report = proteomes.ProteomeReport(host="x", taxon=1, path=None, records=8)
    assert report.suspicious


# ---------------------------------------------------------------------------
# Pooling
# ---------------------------------------------------------------------------
def test_residue_pooling_averages_over_residues():
    matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert embed.pool_protein(matrix, "residue").tolist() == [2.0, 3.0]


def test_legacy_pooling_averages_over_features():
    """The original's axis=1 mean: one number per residue position."""
    matrix = np.array([[1.0, 3.0], [5.0, 9.0]])
    assert embed.pool_protein(matrix, "legacy").tolist() == [2.0, 7.0]


def test_an_unknown_pooling_is_rejected():
    with pytest.raises(ValueError, match="unknown pooling"):
        embed.pool_protein(np.zeros((2, 2)), "mean-ish")


def test_residue_pooling_of_an_organism_is_a_plain_mean():
    vectors = [np.array([1.0, 1.0]), np.array([3.0, 5.0])]
    assert embed.pool_organism(vectors, "residue").tolist() == [2.0, 3.0]


def test_legacy_pooling_of_an_organism_pads_to_the_longest():
    # Lengths differ because legacy vectors are as long as their protein.
    vectors = [np.array([2.0, 2.0, 2.0]), np.array([4.0])]
    pooled = embed.pool_organism(vectors, "legacy")
    assert pooled.tolist() == [3.0, 1.0, 1.0]


def test_pooling_an_empty_organism_is_an_error():
    with pytest.raises(ValueError):
        embed.pool_organism([], "residue")


# ---------------------------------------------------------------------------
# The embedding store
# ---------------------------------------------------------------------------
def test_the_store_round_trips(tmp_path):
    store = embed.EmbeddingStore(tmp_path / "e.npz")
    store.add("PH0001", np.arange(4, dtype=np.float32))
    store.meta = {"pooling": "residue"}
    store.save()

    reloaded = embed.EmbeddingStore(tmp_path / "e.npz")
    assert "PH0001" in reloaded
    assert reloaded["PH0001"].tolist() == [0.0, 1.0, 2.0, 3.0]
    assert reloaded.meta["pooling"] == "residue"
    assert reloaded.width() == 4


def test_saving_leaves_no_partial_file(tmp_path):
    """np.savez appends .npz to a path; the store must not be fooled by that."""
    store = embed.EmbeddingStore(tmp_path / "e.npz")
    store.add("a", np.zeros(2, dtype=np.float32))
    store.save()

    assert (tmp_path / "e.npz").exists()
    assert not list(tmp_path.glob("*.partial*"))


def test_an_absent_store_loads_empty(tmp_path):
    store = embed.EmbeddingStore(tmp_path / "missing.npz")
    assert len(store) == 0
    assert store.width() == 0


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
def _store(tmp_path, name, vectors, pooling="residue"):
    store = embed.EmbeddingStore(tmp_path / name)
    for key, vector in vectors.items():
        store.add(key, np.asarray(vector, dtype=np.float32))
    store.meta = {"pooling": pooling}
    store.save()
    return store


def test_features_concatenate_both_blocks(tmp_path):
    pairs = pd.DataFrame(
        [
            {"phage_id": "P1", "bacteria_id": "E. coli", "taxon_id": "83333", "label": True},
            {"phage_id": "P1", "bacteria_id": "Other", "taxon_id": "287", "label": False},
        ]
    )
    phage = _store(tmp_path, "p.npz", {"P1": [1.0, 2.0]})
    bacteria = _store(tmp_path, "b.npz", {"83333": [10.0], "287": [20.0]})

    frame, report = features.build_features(
        Settings(root=tmp_path), pairs=pairs, phage_store=phage, bacteria_store=bacteria
    )

    assert report.rows == 2
    assert report.positives == 1
    assert report.phage_dim == 2
    assert report.bacteria_dim == 1
    assert features.phage_columns(frame) == ["p0", "p1"]
    assert features.bacteria_columns(frame) == ["b0"]
    assert frame.loc[0, ["p0", "p1", "b0"]].tolist() == [1.0, 2.0, 10.0]
    assert frame.loc[1, "b0"] == 20.0


def test_pairs_without_an_embedding_are_dropped_and_counted(tmp_path):
    pairs = pd.DataFrame(
        [
            {"phage_id": "P1", "bacteria_id": "A", "taxon_id": "1", "label": True},
            {"phage_id": "GHOST", "bacteria_id": "A", "taxon_id": "1", "label": False},
            {"phage_id": "P1", "bacteria_id": "B", "taxon_id": "999", "label": False},
        ]
    )
    phage = _store(tmp_path, "p.npz", {"P1": [1.0]})
    bacteria = _store(tmp_path, "b.npz", {"1": [2.0]})

    frame, report = features.build_features(
        Settings(root=tmp_path), pairs=pairs, phage_store=phage, bacteria_store=bacteria
    )

    assert report.rows == 1
    assert report.dropped_missing_phage == 1
    assert report.dropped_missing_bacteria == 1
    assert len(frame) == 1


def test_label_stays_boolean(tmp_path):
    pairs = pd.DataFrame(
        [{"phage_id": "P1", "bacteria_id": "A", "taxon_id": "1", "label": True}]
    )
    frame, _ = features.build_features(
        Settings(root=tmp_path),
        pairs=pairs,
        phage_store=_store(tmp_path, "p.npz", {"P1": [1.0]}),
        bacteria_store=_store(tmp_path, "b.npz", {"1": [2.0]}),
    )
    assert frame["label"].dtype == bool


def test_legacy_pooling_squares_off_ragged_vectors(tmp_path):
    pairs = pd.DataFrame(
        [
            {"phage_id": "P1", "bacteria_id": "A", "taxon_id": "1", "label": True},
            {"phage_id": "P2", "bacteria_id": "A", "taxon_id": "1", "label": False},
        ]
    )
    # Legacy vectors differ in length between organisms.
    phage = _store(tmp_path, "p.npz", {"P1": [1.0, 2.0, 3.0], "P2": [4.0]}, pooling="legacy")
    bacteria = _store(tmp_path, "b.npz", {"1": [9.0]}, pooling="legacy")

    frame, report = features.build_features(
        Settings(root=tmp_path), pairs=pairs, phage_store=phage, bacteria_store=bacteria
    )

    assert report.pooling == "legacy"
    assert report.phage_dim == 3
    assert frame.loc[1, ["p0", "p1", "p2"]].tolist() == [4.0, 0.0, 0.0]


def test_block_columns_are_not_confused_with_metadata(tmp_path):
    frame = pd.DataFrame(
        {"phage_id": ["x"], "bacteria_id": ["y"], "p0": [1.0], "b0": [2.0], "phage_note": ["z"]}
    )
    # "phage_id" and "phage_note" start with "p" but are not p<digits>.
    assert features.phage_columns(frame) == ["p0"]
    assert features.bacteria_columns(frame) == ["b0"]
    assert features.embedding_columns(frame) == ["p0", "b0"]


def test_building_features_without_embeddings_is_an_error(tmp_path):
    empty = embed.EmbeddingStore(tmp_path / "empty.npz")
    with pytest.raises(FileNotFoundError, match="No phage embeddings"):
        features.build_features(
            Settings(root=tmp_path), pairs=pd.DataFrame(), phage_store=empty, bacteria_store=empty
        )


def test_disjoint_keys_are_an_explicit_error(tmp_path):
    pairs = pd.DataFrame(
        [{"phage_id": "NOPE", "bacteria_id": "A", "taxon_id": "404", "label": False}]
    )
    with pytest.raises(ValueError, match="No pair had both"):
        features.build_features(
            Settings(root=tmp_path),
            pairs=pairs,
            phage_store=_store(tmp_path, "p.npz", {"P1": [1.0]}),
            bacteria_store=_store(tmp_path, "b.npz", {"1": [2.0]}),
        )
