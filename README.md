# Environment setting 

To install fresh environment for the project, execute the following the __env_setting.sh script__ with the following command:

```bash
chmod +x env_setting.sh;
bash _env_setting.sh;
```

Make sure that all the version of the different packages are printed as they are correctly installed.

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


## Scrap data

```python
# Get metadata

metadata_updater = MetadataUpdater(outdir="<YOUR_PATH>/scrapped/test_bacteria")

# Instanciate data downloader

## Global downloader for Acinetobacter calcoaceticus PHEA-2 --> id : 871585.3

global_downloader = GlobalDownloader(outdir = '<YOUR_PATH>/scrapped/test_bacteria', wait = 2)

global_downloader.download('871585.3') # start the downloading

## Speciality genes downloader for Acinetobacter calcoaceticus PHEA-2 --> id : 871585.3

spgenes_downloader = GenomeSpecialityGeneDownloader(outdir = '<YOUR_PATH>/scrapped/test_bacteria', wait = 2)

spgenes_downloader.download('871585.3')  # start the downloading


# Get codons usage table

## Codon table object instancation

codon_table = CodonTable(KAZUSA_URL)

## Get information about condon usage for the taxon of Acinetobacter baumannii --> taxon id: 470

codon_table.set_table('470', replace_uracil=False) # By default nucleotides are T, A, C, G

## Print formated table
print(codon_table)

# Save table as a python dictionary in .npy format through pickle
codon_table.write_table("<YOUR_PATH>/scrapped/test_bacteria")

```


## Filter metadata for species selection

```python
# Filter the BV BRC dmetadata to keep species and strains for which there is an information about antibiotic sensibility ('resistant_phenotype' column)

dft = filter_dataset("<YOUR_PATH>/scrapped/PATRIC_genomes_AMR.txt", filter = True, filter_field = "resistant_phenotype")

```


## Get sequences from species and genes


```python

# Get antibiotic resistant genes for Acinetobacter calcoaceticus PHEA-2 --> id : 871585.3

AMR_list = extract_AMR_genes_names(file_path='<YOUR_PATH>/scrapped/test_bacteria', genome_id = '871585.3')
print(AMR_list)

"""
>>> ['rpsL', 'fusA', 'tuf1', 'emrA', 'emrB', 'estR', 'nahR', nan, 'bla', 'bla', 'bla', 'gidB', nan, 'acrE', nan, 'quiX', 'lpxA', 'lpxA2', 'dxr', nan, nan, 'alr', 'qacF', 'mdtA', 'mexF', 'oprM', 'ampC', 'ampC', 'ampC', 'gyrA', nan, 'acrA', 'acrB', 'mexB', 'oprM', nan, 'yjdB', 'oprB', 'ugpQ', 'ybjG', 'gyrB', 'rplF', 'rpsJ', 'cmr', 'lpxC', 'ddlB', 'ileS', 'dadX', 'ant', nan, 'oprD', nan, 'tufA', 'rpoB', 'rpoB', 'rpoC', 'pgsA', 'norM', 'katG', 'folA', 'fabI', 'macB', 'macA', 'rho', 'murA']
"""

# Get genes genomic coordinates from gene names

selected_genes = ['katG', 'fusA']

gene_coordinates = extract_genes_positions(file_path='<YOUR_PATH>/scrapped/test_bacteria', genome_id = '871585.3', gene_name=selected_genes)

print(gene_coordinates)

"""
>>> {'katG': ('CP002177', 137175, 139313), 'fusA': ('CP002177', 3565997, 3568153)}
"""

# Get genes sequences from the previous gene set

gene_sequences = extract_gene_sequence(file_path='<YOUR_PATH>/scrapped/test_bacteria', genome_id = '871585.3', gene_name=['katG', 'fusA'])

print(gene_sequences)

"""
>>> {'katG': ('CP002177', 'TGGCTCGCCAAACC...ACGAGTAA'), 'fusA': ('CP002177', 'TAAGCTAAGTCAAAACGGTCAAGATTCA...TCGTCTGACAT')}
"""
```




