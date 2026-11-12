"""Codon usage tables fetched from the Kazusa database.

Moved out of ``SmartAMR/scrapper.py``; the HTML scraping and the table layout
of ``__repr__`` are unchanged.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List
from urllib.request import urlopen

import numpy as np

from SmartAMR.config import KAZUSA_URL


class CodonTable:
    """Codon usage frequencies for one taxon, scraped from Kazusa.

    Parameters
    ----------
    base_url
        Base URL of the Kazusa codon usage CGI. Defaults to the project value.
    """

    def __init__(self, base_url: str = KAZUSA_URL) -> None:
        self.base_url: str = base_url
        self.tag: str = "PRE"
        self.reg_table: str = "<" + self.tag + ">(.*?)</" + self.tag + ">"
        self.reg_condon: str = r"([A-Z]{3})\s+([\d.]+)\(\s*(\d+)\)"
        self.reg_codon_swap: str = "U"
        self.taxon_id: str = ""
        self.table: Dict[str, float] = {}

    def __repr__(self) -> str:
        if not self.table:
            return f"<{self.__class__.__name__} for {self.taxon_id} (No data)>"

        bases: List[str] = ["U", "C", "A", "G"] if "UUU" in self.table else ["T", "C", "A", "G"]
        header: str = f"{'':<5}|{'U':^15}|{'C':^15}|{'A':^15}|{'G':^15}|"
        if "UUU" not in self.table:
            header = header.replace("U", "T", 1)
        separator: str = "-" * 5 + (":" + 15 * "-") * 4 + ":" + "-" * 5 + "\n"

        output = header + "\n" + separator
        for first_base in bases:
            for second_base in bases:
                line = f"  {first_base:<3}|"
                for third_base in bases:
                    codon = f"{first_base}{third_base}{second_base}"
                    value = self.table[codon]
                    line += f" {codon:<3} {value:<7.1f}   |"
                output += line + f"  {second_base:<3}\n"
            output += separator

        return f"<{self.__class__.__name__} for {self.taxon_id} with {len(self.table)} codons>\n{output}"

    def set_table(self, taxon_id: str, replace_uracil: bool = False) -> None:
        """Fetch and parse the codon usage table of ``taxon_id``."""

        url: str = f"{self.base_url}{taxon_id}"
        self.taxon_id = taxon_id

        with urlopen(url) as web_handle:  # noqa: S310 - fixed, project-owned URL
            html_content: str = web_handle.read().decode().replace("\n", " ")

        raw_table: List = re.findall(self.reg_table, html_content)
        if not raw_table:
            raise ValueError(f"No codon table found for taxon {taxon_id} at {url}")

        codon_element_table: List = re.findall(self.reg_condon, *raw_table)

        for codon, value_str, _count_str in codon_element_table:
            if replace_uracil:
                codon = re.sub(self.reg_codon_swap, "T", codon)
            self.table[codon] = float(value_str)

    def write_table(self, outdir: str | Path | None = None) -> Path:
        """Save the table as a ``<taxon_id>.npy`` file and return its path."""

        if outdir is None:
            raise ValueError("outdir is required, please provide a valid output directory")

        out_path: Path = Path(outdir)
        if not out_path.exists():
            raise IOError(f"Output path: {out_path} does not exists. Please provide valide path")

        target = out_path / f"{self.taxon_id}.npy"
        np.save(target, self.table)

        return target


def fetch_codon_table(taxon_id: str, replace_uracil: bool = False) -> CodonTable:
    """Build a :class:`CodonTable` already populated for ``taxon_id``."""
    table = CodonTable()
    table.set_table(taxon_id, replace_uracil=replace_uracil)
    return table
