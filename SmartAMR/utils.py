from pathlib import Path
from typing import List, Tuple, Dict,  Union

import numpy as np
import pandas as pd

from Bio import SeqIO

METADATA = 'PATRIC_genomes_AMR.txt'
SP_GENES_ROOT = '.PATRIC.spgene.tab'
FEATURES_ROOT = '.PATRIC.features.tab'

def extract_AMR_genes_names(file_path: str = None, gene_type: str = 'Antibiotic Resistance', genome_id: str = None) -> List[str]:
    """Extract the names of AMR genes from .spgene file
        Equivalent to `cut -f 8 XXXX.PATRIC.spgene.tab`
    """
    
    data_path: Path = Path.joinpath(Path(file_path), Path(genome_id + SP_GENES_ROOT))

    df : pd.DataFrame = pd.read_csv(data_path, sep = '\t')
    df = df[df['property'] == gene_type]

    return df['gene'].tolist()


def extract_genes_positions(file_path: str = None, genome_id: str = None, gene_name: str | List = None) -> Union[Tuple[str, int, int], Dict[str, Tuple[str, int, int]]]:
    """Extract the genomic coordinates of a given gene name in XXXX.PATRIC.features.tab or XXXX.PATRIC.gff
        Equivalent to `grep gene_name XXXX.PATRIC.features.tab XXXX.gff`
    """


    data_path: Path = Path.joinpath(Path(file_path), Path(genome_id + FEATURES_ROOT))
    df: pd.DataFrame = pd.read_csv(data_path, sep = '\t')

    if isinstance(gene_name, str):
        print("Str instance case")
        value: pd.Series = df[df['gene'] == gene_name]

        return (str(value['accession'].values[0]), int(value['start'].values[0]), int(value['end'].values[0]))
    
    elif isinstance(gene_name, list):
        
        value: pd.Series = df[df['gene'].isin(gene_name)]
        value_dict: dict = {k: (v[1]['accession'], v[1]['start'], v[1]['end']) for k, v in zip(gene_name, value.iterrows())}

        return value_dict

def extract_gene_sequence(file_path: str = None, genome_id: str = None, gene_name: str | List[str] = None) -> Dict[str, Tuple[str, str]]:
    """
    Extract gene sequence from a genome given a genome_id and genomic interval coordinates.
    """

    genome_file: Path = Path.joinpath(Path(file_path), Path(genome_id + '.fna'))

    if isinstance(gene_name, str):

        contig, start, stop = extract_genes_positions(file_path=file_path, genome_id=genome_id, gene_name=gene_name)

        for seq_record in SeqIO.parse(genome_file, "fasta"):

            if seq_record.id == contig:
                sequence: str = str(seq_record.seq[start:stop].upper())
    
        return {gene_name: (seq_record.id, sequence)} 
    

    elif isinstance(gene_name, list):

        sequences_dict = {} # Placeholder
        infos = extract_genes_positions(file_path=file_path, genome_id=genome_id, gene_name=gene_name)
        genes = infos.keys()
        contigs = [info[0] for info in infos.values()]
        starts = [info[1] for info in infos.values()]
        stops = [info[2] for info in infos.values()]

        for idx, gene in enumerate(genes):
            for seq_record in SeqIO.parse(genome_file, "fasta"):

                if seq_record.id == contigs[idx]:
                    sequence: str = str(seq_record.seq[starts[idx]: stops[idx]]).upper()

            sequences_dict[gene] = (contigs[idx], sequence)

        return sequences_dict


