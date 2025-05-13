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
reverse_nucleotides = {0: 'A', 1: 'C', 2: 'G': 3, 3: 'T'}

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
X_train_gen = encoded_sequences_gen[:, :-1]
y_train_gen_one_hot = tf.keras.utils.to_categorical(np.expand_dims(encoded_sequences_gen[:, 1:], axis=-1), num_classes=num_nucleotides)

# --- 4. Build and Train the Generative LSTM Model ---
input_seq_gen = Input(shape=(seq_length - 1,), name='input_gen')
embedding_gen = Embedding(input_dim=num_nucleotides, output_dim=embedding_dim)(input_seq_gen)
lstm_gen = LSTM(units=lstm_units_gen, return_sequences=True)(embedding_gen)
output_probs_gen = TimeDistributed(Dense(units=num_nucleotides, activation='softmax'), name='output_gen')(lstm_gen)
generative_model = Model(inputs=input_seq_gen, outputs=output_probs_gen)

optimizer_gen = Adam(learning_rate=learning_rate_gen)
generative_model.compile(optimizer=optimizer_gen, loss='categorical_crossentropy')

print("Training the generative model...")
history_gen = generative_model.fit(
    X_train_gen, y_train_gen_one_hot,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.2,
    callbacks=[EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)]
)
print("Generative model training finished.")

# --- 5. Function to Generate New Sequences with Constraints ---
def generate_sequence_constrained(model, seed_sequence, length, codon_usage, canonical_codons, mutation_probabilities):
    encoded_seed = encode_sequence(seed_sequence, seq_length - 1)
    generated_sequence = list(seed_sequence)

    for i in range(length - len(seed_sequence)):
        prediction_logits = model.predict(np.expand_dims(encoded_seed, axis=0))[0, -1, :]
        probabilities = tf.nn.softmax(prediction_logits).numpy()

        current_base = reverse_nucleotides.get(encoded_seed[-1], '') if encoded_seed.shape[0] > 0 else ''
        next_probs = np.zeros_like(probabilities)
        for j, next_base in reverse_nucleotides.items():
            mutation = tuple(sorted((current_base, next_base)))
            next_probs[j] = probabilities[j] * mutation_probabilities.get(mutation, 1.0) # Default to 1.0 if no rate

        if len(generated_sequence) >= 2 and (len(generated_sequence) + 1) % 3 == 0:
            potential_codon = "".join(generated_sequence[-2:])
            biased_probs = np.zeros_like(next_probs)
            for j, next_base in reverse_nucleotides.items():
                potential_full_codon = potential_codon + next_base
                bias = codon_usage.get(potential_full_codon, 1e-6)
                if potential_full_codon not in canonical_codons:
                    bias *= 0.1 # Further penalize non-canonical
                biased_probs[j] = next_probs[j] * bias
            next_probs = biased_probs / np.sum(biased_probs) if np.sum(biased_probs) > 0 else next_probs

        predicted_index = np.random.choice(range(num_nucleotides), p=next_probs / np.sum(next_probs) if np.sum(next_probs) > 0 else probabilities)
        predicted_nucleotide = reverse_nucleotides[predicted_index]
        generated_sequence.append(predicted_nucleotide)
        encoded_seed = np.roll(encoded_seed, -1)
        encoded_seed[-1] = predicted_index

    return "".join(generated_sequence)

# --- 6. Build and Train the AMR Prediction Model ---
# Dummy data for AMR prediction - replace with your actual data
amr_gene_sequences_amr_pos = [generate_sequence("ATGC", seq_length) for _ in range(100)]
amr_gene_sequences_amr_neg = [generate_sequence("CGTA", seq_length) for _ in range(100)]
encoded_sequences_amr_pos = np.array([encode_sequence(seq, seq_length) for seq in amr_gene_sequences_amr_pos])
encoded_sequences_amr_neg = np.array([encode_sequence(seq, seq_length) for seq in amr_gene_sequences_amr_neg])
X_train_amr = np.vstack([encoded_sequences_amr_pos, encoded_sequences_amr_neg])
y_train_amr = np.array([1] * 100 + [0] * 100)

input_seq_amr = Input(shape=(seq_length,), name='input_amr')
embedding_amr = Embedding(input_dim=num_nucleotides, output_dim=embedding_dim)(input_seq_amr)
lstm_amr = LSTM(units=lstm_units_amr)(embedding_amr)
output_amr = Dense(1, activation='sigmoid', name='output_amr')(lstm_amr)
amr_prediction_model = Model(inputs=input_seq_amr, outputs=output_amr)

optimizer_amr = Adam(learning_rate=learning_rate_amr)
amr_prediction_model.compile(optimizer=optimizer_amr, loss='binary_crossentropy', metrics=['accuracy'])

print("\nTraining the AMR prediction model...")
history_amr = amr_prediction_model.fit(
    X_train_amr, y_train_amr,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.2,
    callbacks=[EarlyStopping(monitor='val_accuracy', patience=patience, restore_best_weights=True)]
)
print("AMR prediction model training finished.")

# --- 7. Generate and Evaluate ---
print("\nGenerating and evaluating constrained mutant sequences...")
seed = amr_gene_sequences_gen[0][:50]
num_generated = 5
for i in range(num_generated):
    mutant_sequence = generate_sequence_constrained(
        generative_model, seed, seq_length, codon_usage, canonical_codons, mutation_probabilities
    )
    encoded_mutant = np.expand_dims(encode_sequence(mutant_sequence, seq_length), axis=0)
    amr_prediction = amr_prediction_model.predict(encoded_mutant)[0][0]
    print(f"Generated Mutant {i + 1}: {mutant_sequence[:20]}..., Predicted AMR Likelihood: {amr_prediction:.4f}")