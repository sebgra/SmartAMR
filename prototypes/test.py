import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, TimeDistributed
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import random

# --- 1. Define Parameters ---
seq_length = 1000  # Length of input/output sequences
embedding_dim = 64  # Dimensionality of the nucleotide embeddings
lstm_units_gen = 128    # Number of LSTM units for the generative model
lstm_units_amr = 128    # Number of LSTM units for the AMR prediction model
num_nucleotides = 4  # A, C, G, T
learning_rate_gen = 0.001
learning_rate_amr = 0.001
batch_size = 64
epochs = 5  # Reduced for demonstration, increase for real training
patience = 3       # For EarlyStopping

# --- 2. Define Nucleotide and Codon Mappings, Mutation Rates ---
nucleotides = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
reverse_nucleotides = {0: 'A', 1: 'C', 2:'G', 3: 'T'}

# Example codon usage table for E. coli (replace with your species)
codon_usage = {
    'AAA': 0.015, 'AAC': 0.018, 'AAG': 0.007, 'AAT': 0.019, 'ACA': 0.022, 'ACC': 0.035, 'ACG': 0.014, 'ACT': 0.027,
    'AGA': 0.012, 'AGC': 0.024, 'AGG': 0.013, 'AGT': 0.015, 'ATA': 0.017, 'ATC': 0.039, 'ATG': 0.023, 'ATT': 0.032,
    'CAA': 0.016, 'CAC': 0.021, 'CAG': 0.042, 'CAT': 0.017, 'CCA': 0.025, 'CCC': 0.033, 'CCG': 0.011, 'CCT': 0.025,
    'CGA': 0.006, 'CGC': 0.037, 'CGG': 0.011, 'CGT': 0.021, 'CTA': 0.007, 'CTC': 0.013, 'CTG': 0.040, 'CTT': 0.013,
    'GAA': 0.027, 'GAC': 0.022, 'GAG': 0.041, 'GAT': 0.030, 'GCA': 0.031, 'GCC': 0.040, 'GCG': 0.017, 'GCT': 0.035,
    'GGA': 0.017, 'GGC': 0.036, 'GGG': 0.017, 'GGT': 0.026, 'GTA': 0.011, 'GTC': 0.025, 'GTG': 0.028, 'GTT': 0.028,
    'TAA': 0.011, 'TAC': 0.013, 'TAG': 0.007, 'TAT': 0.016, 'TCA': 0.017, 'TCC': 0.022, 'TCG': 0.009, 'TCT': 0.018,
    'TGA': 0.013, 'TGC': 0.015, 'TGG': 0.013, 'TGT': 0.006, 'TTA': 0.014, 'TTC': 0.020, 'TTG': 0.013, 'TTT': 0.020,
}
canonical_codons = set(codon_usage.keys())

mutation_rates = {
    ('A', 'G'): 2.0, ('G', 'A'): 2.0, ('C', 'T'): 2.0, ('T', 'C'): 2.0,  # Transitions
    ('A', 'C'): 0.5, ('A', 'T'): 0.5, ('G', 'C'): 0.5, ('G', 'T'): 0.5,  # Transversions
    ('C', 'A'): 0.5, ('C', 'G'): 0.5, ('T', 'A'): 0.5, ('T', 'G'): 0.5,
}
total_rate = sum(mutation_rates.values())
mutation_probabilities = {k: v / total_rate for k, v in mutation_rates.items()}

def encode_sequence(seq, length):
    encoded = [nucleotides.get(base, 0) for base in seq]
    if len(encoded) < length:
        encoded += [0] * (length - len(encoded))
    elif len(encoded) > length:
        encoded = encoded[:length]
    return np.array(encoded)

def decode_sequence(encoded_seq):
    return "".join([reverse_nucleotides.get(i, 'N') for i in encoded_seq])

# --- 3. Prepare Dummy Training Data for Generative Model ---
# Replace this with your actual data loading from PATRIC
amr_gene_sequences_gen = [
    "ATGCGTAGCTAGCTAGCATCGATGCTAGCTAGCTACG" * 10,
    "CGTACGATCGATCGATCGATCGATCGATCGATCGATC" * 10,
    "GCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTA" * 10,
]
encoded_sequences_gen = np.array([encode_sequence(seq, seq_length) for seq in amr_gene_sequences_gen])


print(encoded_sequences_gen)

X_train_gen = encoded_sequences_gen[:, :-1]
y_train_gen_one_hot = tf.keras.utils.to_categorical(np.expand_dims(encoded_sequences_gen[:, 1:], axis=-1), num_classes=num_nucleotides)

# # --- 4. Build and Train the Generative LSTM Model ---
input_seq_gen = Input(shape=(seq_length - 1,), name='input_gen')
embedding_gen = Embedding(input_dim=num_nucleotides, output_dim=embedding_dim)(input_seq_gen)
lstm_gen = LSTM(units=lstm_units_gen, return_sequences=True)(embedding_gen)
output_probs_gen = TimeDistributed(Dense(units=num_nucleotides, activation='softmax'), name='output_gen')(lstm_gen)
generative_model = Model(inputs=input_seq_gen, outputs=output_probs_gen)

# optimizer_gen = Adam(learning_rate=learning_rate_gen)
# generative_model.compile(optimizer=optimizer_gen, loss='categorical_crossentropy')

# print("Training the generative model...")
# history_gen = generative_model.fit(
#     X_train_gen, y_train_gen_one_hot,
#     batch_size=batch_size,
#     epochs=epochs,
#     validation_split=0.2,
#     callbacks=[EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)]
# )
# print("Generative model training finished.")