import os
import re
import subprocess as sp
from pathlib import Path
from time import sleep
from typing import List, Dict, Tuple
from urllib.parse import urlsplit, unquote, urljoin
from urllib.request import urlopen

import numpy as np

PATRIC_FTP_BASE_URL = "ftp://ftp.bvbrc.org"
PATRIC_FTP_AMR_METADATA_URL = urljoin(PATRIC_FTP_BASE_URL, urljoin("RELEASE_NOTES/", "PATRIC_genomes_AMR.txt"))
PATRIC_FTP_GENOMES_URL = urljoin(PATRIC_FTP_BASE_URL, "genomes/")
PATRIC_FTP_GENOMES_METADATA_URL = urljoin(PATRIC_FTP_BASE_URL, urljoin("RELEASE_NOTES/", "genome_metadata"))

KAZUSA_URL = "https://www.kazusa.or.jp/codon/cgi-bin/showcodon.cgi?species="

class Downloader:
    def __init__(self, base_url : str , outdir : str = None, wait : int = 1) -> None:
        """AI is creating summary for __init__

        Parameters
        ----------
        base_url : str
            [description]
        outdir : str, optional
            [description], by default None
        wait : int, optional
            [description], by default 1

        Raises
        ------
        ValueError
            [description]
        """
        
        self.base_url = base_url
        self.wait = wait
        self.outdir = Path(outdir)

        if not self.outdir.exists():
            raise ValueError(f"Directory {self.outdir} does not exist, please provide existing directory as output")


    def _download(self, base_url: str , filename: str = None) -> None:
        """
        Generic downloading method.

        Parameters
        ----------
        base_url : str, optional
            [description], by default PATRIC_FTP_BASE_URL
        filename : str, optional
            [description], by default None
        """
    
        ftp_url: str = urljoin(base_url, filename)
        
        ftp_path: Path = Path(ftp_url)

        print(f"test : {ftp_url}")

        abstract_cmd: str = f"wget --quiet -o /dev/null -O {Path.joinpath(self.outdir, ftp_path.name)} --continue --timeout 15 {ftp_url}"

        try:
            sleep(self.wait)
            sp.call([abstract_cmd], shell=True)

        except Exception as e: 
            print(f"Error of download: {ftp_path} - {e}")

        return 
    
    def download(self, *args,  **kwargs) -> None:
        raise NotImplementedError("Subclasses must implement the download method.")


class GenomeContigDownloader(Downloader):
    def __init__(self, outdir : str = None, wait : int = 1) -> None:
        super().__init__(base_url = PATRIC_FTP_GENOMES_URL, outdir = outdir, wait = wait)

    def download(self, bvbrc_id : str = None) -> None:
        
        if bvbrc_id is None:
            raise KeyError(f"Id {bvbrc_id} does not exist.")
        
        filename: str = f"{bvbrc_id}.fna"
        ftp_url = f"{self.base_url}{bvbrc_id}/"
        sleep(self.wait)

        return self._download(base_url = ftp_url, filename = filename)
    
class GenomeProteinSeqDownloader(Downloader):
    def __init__(self, outdir : str = None, wait : int = 1) -> None:
        super().__init__(base_url = PATRIC_FTP_GENOMES_URL, outdir = outdir, wait = wait)

    def download(self, bvbrc_id : str = None) -> None:
        
        if bvbrc_id is None:
            raise KeyError(f"Id {bvbrc_id} does not exist.")
        
        filename: str = f"{bvbrc_id}.PATRIC.faa"
        ftp_url = f"{self.base_url}{bvbrc_id}/"
        sleep(self.wait)

        return self._download(base_url = ftp_url, filename = filename)
    
class GenomeFeatureDownloader(Downloader):
    def __init__(self, outdir : str = None, wait : int = 1) -> None:
        super().__init__(base_url = PATRIC_FTP_GENOMES_URL, outdir = outdir, wait = wait)

    def download(self, bvbrc_id : str = None) -> None:
        
        if bvbrc_id is None:
            raise KeyError(f"Id {bvbrc_id} does not exist.")
        
        filename: str = f"{bvbrc_id}.PATRIC.features.tab"
        ftp_url = f"{self.base_url}{bvbrc_id}/"
        sleep(self.wait)

        return self._download(base_url = ftp_url, filename = filename)
    
class GenomeFeatureSeqDownloader(Downloader):
    def __init__(self, outdir : str = None, wait : int = 1) -> None:
        super().__init__(base_url = PATRIC_FTP_GENOMES_URL, outdir = outdir, wait = wait)

    def download(self, bvbrc_id : str = None) -> None:
        
        if bvbrc_id is None:
            raise KeyError(f"Id {bvbrc_id} does not exist.")
        
        filename: str = f"{bvbrc_id}.PATRIC.ffn"
        ftp_url = f"{self.base_url}{bvbrc_id}/"
        sleep(self.wait)

        return self._download(base_url = ftp_url, filename = filename)
    
class GenomeRNASeqDownloader(Downloader):
    def __init__(self, outdir : str = None, wait : int = 1) -> None:
        super().__init__(base_url = PATRIC_FTP_GENOMES_URL, outdir = outdir, wait = wait)

    def download(self, bvbrc_id : str = None) -> None:
        
        if bvbrc_id is None:
            raise KeyError(f"Id {bvbrc_id} does not exist.")
        
        filename: str = f"{bvbrc_id}.PATRIC.frn"
        ftp_url = f"{self.base_url}{bvbrc_id}/"
        sleep(self.wait)

        return self._download(base_url = ftp_url, filename = filename)
    
class GenomeAnnotationDownloader(Downloader):
    def __init__(self, outdir : str = None, wait : int = 1) -> None:
        super().__init__(base_url = PATRIC_FTP_GENOMES_URL, outdir = outdir, wait = wait)

    def download(self, bvbrc_id : str = None) -> None:
        
        if bvbrc_id is None:
            raise KeyError(f"Id {bvbrc_id} does not exist.")
        
        filename: str = f"{bvbrc_id}.PATRIC.gff"
        ftp_url = f"{self.base_url}{bvbrc_id}/"
        sleep(self.wait)

        return self._download(base_url = ftp_url, filename = filename)
    
class GenomePathwayDownloader(Downloader):
    def __init__(self, outdir : str = None, wait : int = 1) -> None:
        super().__init__(base_url = PATRIC_FTP_GENOMES_URL, outdir = outdir, wait = wait)

    def download(self, bvbrc_id : str = None) -> None:
        
        if bvbrc_id is None:
            raise KeyError(f"Id {bvbrc_id} does not exist.")
        
        filename: str = f"{bvbrc_id}.PATRIC.pathway.tab"
        ftp_url = f"{self.base_url}{bvbrc_id}/"
        sleep(self.wait)

        return self._download(base_url = ftp_url, filename = filename)


class GenomeSpecialityGeneDownloader(Downloader):
    def __init__(self, outdir : str = None, wait : int = 1) -> None:
        super().__init__(base_url = PATRIC_FTP_GENOMES_URL, outdir = outdir, wait = wait)

    def download(self, bvbrc_id : str = None) -> None:
        
        if bvbrc_id is None:
            raise KeyError(f"Id {bvbrc_id} does not exist.")
        
        filename: str = f"{bvbrc_id}.PATRIC.spgene.tab"
        ftp_url = f"{self.base_url}{bvbrc_id}/"
        sleep(self.wait)

        return self._download(base_url = ftp_url, filename = filename)

class GenomeSubSystemDownloader(Downloader):
    def __init__(self, outdir : str = None, wait : int = 1) -> None:
        super().__init__(base_url = PATRIC_FTP_GENOMES_URL, outdir = outdir, wait = wait)

    def download(self, bvbrc_id : str = None) -> None:
        
        if bvbrc_id is None:
            raise KeyError(f"Id {bvbrc_id} does not exist.")
        
        filename: str = f"{bvbrc_id}.PATRIC.subsystem.tab"
        ftp_url = f"{self.base_url}{bvbrc_id}/"
        sleep(self.wait)

        return self._download(base_url = ftp_url, filename = filename)
    

class GlobalDownloader(Downloader):
    def __init__(self, outdir = None, wait = 1) -> None:
        super().__init__(base_url = PATRIC_FTP_GENOMES_URL, outdir = outdir, wait = wait)

    def download(self, bvbrc_id = None) -> None:

        if bvbrc_id is None:
            raise KeyError(f"Id {bvbrc_id} does not exist.")
        suffixes : list[str] = ["fna", "PATRIC.faa", "PATRIC.features.tab", "PATRIC.ffn", "PATRIC.frn", "PATRIC.gff", "PATRIC.pathway.tab", "PATRIC.spgene.tab", "PATRIC.subsystem.tab"]
        filenames: List = [f"{bvbrc_id}.{s}" for s in suffixes]

        for file in filenames:
            ftp_url = f"{self.base_url}{bvbrc_id}/"
            sleep(self.wait)

            self._download(base_url = ftp_url, filename = file)

    
class CodonTable():
    def __init__(self, base_url: str) -> None:
        self.base_url:str = base_url
        self.tag: str = 'PRE'
        self.reg_table: str = "<" + self.tag + ">(.*?)</" + self.tag + ">"
        self.reg_condon: str = r'([A-Z]{3})\s+([\d.]+)\(\s*(\d+)\)'
        self.reg_codon_swap: str = ('U')
        self.taxon_id: str = ""
        self.table: Dict = {}

    def __repr__(self) -> str:

        bases: List[str] = ['U', 'C', 'A', 'G'] if 'UUU' in self.table.keys() else ['T', 'C', 'A', 'G']
        header: str = f"{'':<5}|{'U':^15}|{'C':^15}|{'A':^15}|{'G':^15}|"
        if 'UUU' not in self.table.keys():
            header = header.replace('U', 'T', 1) # Replace the first 'U' with 'T'
        separator: str = '-'*5 + (':' + 15 * '-')*4 + ':' +'-'*5 + '\n'

        output = header + '\n' + separator
        # print(output)
        for first_base in bases:
            for second_base in bases:
                line = f"  {first_base:<3}|"
                for third_base in bases:
                    codon = f"{first_base}{third_base}{second_base}"
                    value = self.table[codon]
                    line += f" {codon:<3} {value:<7.1f}   |"
                output += line + f"  {second_base:<3}\n"
            output += separator

        if not self.table:
            return f"<{self.__class__.__name__} for {self.taxon_id} (No data)>"
        return f"<{self.__class__.__name__} for {self.taxon_id} with {len(self.table)} codons>\n{output}"

    def set_table(self, taxon_id: str, replace_uracil: bool = False)-> None:

        url: str = f"{self.base_url}{taxon_id}"
        self.taxon_id = taxon_id
        web_handle = urlopen(url)
        html_content: str = web_handle.read().decode().replace("\n", " ")

        raw_table: List = re.findall(self.reg_table, html_content)
        codon_element_table: List = re.findall(self.reg_condon, *raw_table)

        for codon, value_str, count_str in codon_element_table:

            if replace_uracil:
                codon: str =  re.sub(self.reg_codon_swap, "T", codon)
            
            self.table[codon] = float(value_str)

        return

    def write_table(self, outdir: str = None) -> None:
        
        out_path: Path = Path(outdir)
        out_filename = f"{self.taxon_id}.npy"

        if not out_path.exists():
            raise IOError(f"Output path: {out_path} does not exists. Please provide valide path")
        
        np.save(Path.joinpath(out_path, out_filename), self.table)



class MetadataUpdater(Downloader):
    def __init__(self, outdir = None, wait = 1):
        super().__init__(base_url = PATRIC_FTP_BASE_URL, outdir = outdir, wait = wait)

    def __call__(self) -> None:
        filename: str = urljoin("RELEASE_NOTES/", "PATRIC_genomes_AMR.txt")
        print(f"Updating Metdata ...")
        return self._download(base_url = self.base_url, filename = filename)


