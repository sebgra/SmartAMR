"""The nine bacterial hosts, and the three different ids each one has.

Every stage of the pipeline needs a different identifier for the same organism,
which is exactly the kind of thing that goes wrong when it is spelled out in
each script:

``inphared_name``
    The substring matched against the INPHARED metadata's isolation-host
    column. That column is free text -- the column name itself warns "beware
    inconsistent and nonsense values" -- so matching is by substring, which is
    what ``refactored/phages_selection.py`` did.

``uniprot_taxon``
    The organism id used to download the proteome from UniProt.

``bvbrc_id``
    The BV-BRC/PATRIC genome id, kept because ``config/bacterias.csv`` and the
    older scrapper path use it.

A correction is recorded here, with its reasoning, because it changes results:
``refactored/bacteria_proteome_downloader.py`` listed E. coli as taxon
``511145`` (strain K-12 substr. MG1655). UniProt has only **8** entries under
that organism id -- it is a strain id that almost nothing is annotated against
directly -- so the download produced a 4 KB file where every other host got
megabytes, and E. coli, the largest host group in the dataset, contributed
almost no bacterial signal. Taxon ``83333`` (E. coli K-12) has 4,567 entries
and is the id UniProt actually annotates that organism under. The original id
is kept in ``legacy_uniprot_taxon`` so the discrepancy stays visible.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Host:
    """One bacterial host and the ids it is known by in each data source."""

    inphared_name: str
    uniprot_taxon: int
    bvbrc_id: str
    #: Set when :attr:`uniprot_taxon` deliberately differs from the id the
    #: original scripts used. ``None`` when the two agree.
    legacy_uniprot_taxon: int | None = None
    note: str = ""

    @property
    def label(self) -> str:
        """Display name; the INPHARED name is already human-readable."""
        return self.inphared_name

    @property
    def taxon_key(self) -> str:
        """The taxon id as the string used for DataFrame keys and filenames."""
        return str(self.uniprot_taxon)

    @property
    def corrected(self) -> bool:
        return self.legacy_uniprot_taxon is not None


#: The nine hosts, in the order ``refactored/phages_selection.py`` listed them.
HOSTS: tuple[Host, ...] = (
    Host(
        inphared_name="Escherichia coli",
        uniprot_taxon=83333,
        bvbrc_id="511145.12",
        legacy_uniprot_taxon=511145,
        note=(
            "UniProt has 8 entries for organism_id:511145 (K-12 substr. MG1655) "
            "but 4567 for 83333 (K-12); the original id yielded an empty proteome."
        ),
    ),
    Host(inphared_name="Klebsiella pneumoniae", uniprot_taxon=1125630, bvbrc_id="1125630.4"),
    Host(inphared_name="Pseudomonas aeruginosa", uniprot_taxon=287, bvbrc_id="287.5706"),
    Host(inphared_name="Ralstonia solanacearum", uniprot_taxon=305, bvbrc_id="305.1006"),
    Host(inphared_name="Lactococcus lactis", uniprot_taxon=1360, bvbrc_id="1360.100"),
    Host(inphared_name="Staphylococcus aureus", uniprot_taxon=93061, bvbrc_id="93061.5"),
    Host(inphared_name="Salmonella", uniprot_taxon=28901, bvbrc_id="28901.3005"),
    Host(inphared_name="Acinetobacter baumannii", uniprot_taxon=470, bvbrc_id="470.2401"),
    Host(inphared_name="Streptococcus thermophilus", uniprot_taxon=1308, bvbrc_id="1308.482"),
)

#: Taxon id (as a string) -> host, the lookup the feature builder needs.
HOSTS_BY_TAXON: dict[str, Host] = {host.taxon_key: host for host in HOSTS}

#: INPHARED host name -> host.
HOSTS_BY_NAME: dict[str, Host] = {host.inphared_name: host for host in HOSTS}


def host_by_name(name: str) -> Host:
    """Look up a host by its INPHARED name."""
    try:
        return HOSTS_BY_NAME[name]
    except KeyError:
        raise KeyError(f"unknown host {name!r}; known: {sorted(HOSTS_BY_NAME)}") from None


def host_by_taxon(taxon: int | str) -> Host:
    """Look up a host by its UniProt taxon id."""
    try:
        return HOSTS_BY_TAXON[str(taxon)]
    except KeyError:
        raise KeyError(f"unknown taxon {taxon!r}; known: {sorted(HOSTS_BY_TAXON)}") from None


def corrections() -> list[Host]:
    """Hosts whose taxon id was deliberately changed from the original scripts."""
    return [host for host in HOSTS if host.corrected]
