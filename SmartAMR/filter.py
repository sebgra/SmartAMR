from pathlib import Path
import numpy as np 
import pandas as pd

def filter_dataset(data: str = None, filter: bool = False, filter_field: str = 'resistant_phenotype') -> pd.DataFrame:
    """
    Filter out Nan values from an AMR dataframe considering an user definer column name

    Parameters
    ----------
    data : str, optional
        Path to the file to filter, by default None
    filter : bool, optional
        Set if the dataframe has to be filtered, by default False
    filter_field : str, optional
        Name of the column to filter out Nans from, by default 'resistant_phenotype'

    Returns
    -------
    pd.DataFrame
        Nan filtered DataFrame
    """


    data_path: Path = Path(data)

    if not data_path.exists():
        raise ValueError(f"path: {data} does not exist, please provide correct path")

    df: pd.DataFrame = pd.read_csv(data_path, sep = '\t')
    df['genome_id'] = df['genome_id'].astype(str)

    if not filter_field in df.columns:
        raise ValueError(f"{filter_field} not in the data column names. Please provide correct column name")
    
    df = df.dropna(subset=[filter_field])

    return df

def get_amr_data(data):
    """ Get all the informations from BVBRC metadata which have valid AMR phenotype."""
    pass

def get_amr_measurment_data(data):
    """Get all the information from BVBRC metadata for which there is a MIC measurment."""
    pass

def get_taxon_ids(data):
    """Return taxon IDs from metadata"""
    pass

def genome_ids(data):
    """Return genome IDs from metadata"""
    pass





