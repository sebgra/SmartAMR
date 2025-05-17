# app.py

from flask import Flask, jsonify, request
from flask_cors import cross_origin
from main import *
import pandas as pd
import numpy as np
from Bio import Align

import os
from Bio.PDB import PDBList

BACTERIADB = 'bacteriaDB.fa'
PHAGEDB = 'phageDB.fa'

app = Flask(__name__)

@app.route('/')

def hello():
    return 'Hello from Flask!' 

@app.route('/api/evaluate/', methods=['POST'])
@cross_origin()
def evaluate_phage_bacteria():
    payload = request.get_json()  

    bacteriaData = payload['bacteria']
    phageData = payload['phage']

    results = {}
    k = 1

    # if required, get proteics sequences from database
    if len(bacteriaData['sequence']) == 0 and len(bacteriaData["name"]) > 0:
        bacteriaData['sequence'] = retrieve_prots(bacteriaData["name"], BACTERIADB)[0]
    if len(phageData['sequence']) == 0 and len(phageData["name"]) > 0:
        phageData['sequence'] = retrieve_prots(phageData["name"], PHAGEDB)[0]

    # if both sequences are provided, use the versus method directly
    if len(bacteriaData['sequence']) > 0 and len(phageData['sequence']) > 0:
        results[k] = {
            'bacteria':bacteriaData['name'],
            'phage':phageData['name'],
            'evaluation': evaluate_sequences(bacteriaData['sequence'], phageData['sequence'])
        }
        k += 1

    # if only one or the other sequence is provided, we iterate over the complementary database to produce an array of results
    else:
        if len(bacteriaData['sequence']) > 0:
            for name, sequence in parseDB(PHAGEDB):
                results[k] = {
                    'bacteria':bacteriaData['name'],
                    'phage':name,
                    'evaluation': evaluate_sequences(bacteriaData['sequence'], sequence)
                }
                k += 1
        if len(phageData['name']) > 0:
            for name, sequence in parseDB(BACTERIADB):
                results[k] = {
                    'bacteria':name,
                    'phage':phageData['name'],
                    'evaluation': evaluate_sequences(sequence, phageData['sequence'])
                }
                k += 1

    return jsonify({'status': 'ok', 'data': results})