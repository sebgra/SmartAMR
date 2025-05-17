### Main python script for data generation
# TODO: connect evaluate_sequences to model to AI algorithm
import sys
import pandas as pd
import random

DATABASE_PATH = './data'

def evaluate_sequences(bacteria_sequence, phage_sequence):
  # TODO input into the model here
  """Coucou c'est là qu'il faut modifier et retourner un string ou un nombre"""
  binary = [1, 0]
  return random.choice(binary)

def parseDB(database):
  with open(f"{DATABASE_PATH}/{database}") as fasta:
    current_prot = ""
    current_name = ""
    for line in fasta.readlines():
      if line[0] == ">":
        if len(current_prot) > 0:
            yield current_name, current_prot
        current_name = line[1:].replace('\n', '')
        current_prot = ""
      else:
          current_prot += line.replace('\n', '')
    yield current_name, current_prot

def retrieve_prots(name, database):
  """Retrieves proteic sequence in the database"""
  proteic_sequences = []
  with open(f"{DATABASE_PATH}/{database}") as fasta:
    save_prot = False
    current_prot = ""
    for line in fasta.readlines():
      if line[0] == ">":
        if line[1:].replace('\n','') == name:
          save_prot = True
        else:
          if save_prot:
            proteic_sequences.append(current_prot)
            current_prot = ""
          save_prot = False
      else:
        if save_prot:
          current_prot += line.replace('\n', '')
    if save_prot:
      proteic_sequences.append(current_prot)
    return proteic_sequences


