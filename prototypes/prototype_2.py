import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, TimeDistributed
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import random
from tensorflow.keras import backend as K

# --- 1. Define Parameters (as before) ---
seq_length = 1000
embedding_dim = 64
lstm_units_gen = 128
lstm_units_amr = 128
num_nucleotides = 4
learning_rate_gen = 0.001
learning_rate_amr = 0.001
batch_size = 64
epochs = 5
patience = 3

# --- 2. Define Nucleotide and Codon Mappings, Mutation Rates (as before) ---
nucleotides = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
reverse_nucleotides = {0: 'A', 1: 'C', 2: 'G': 3, 3: 'T'}
codon_usage_tf = tf.constant([codon_usage.get(codon, 1e-6) for codon in [''.join(bases) for bases in np.array(list(np.ndindex((4, 4, 4))))]], dtype=tf.float32)
canonical_codons_set = set(codon_usage.keys())
mutation_matrix = np.zeros((4, 4), dtype=np.float32)
for (b1, b2), prob in mutation_probabilities.items():
    mutation_matrix[nucleotides[b1], nucleotides[b2]] = prob
mutation_probabilities_tf = tf.constant(mutation_matrix, dtype=tf.float32)

def get_codon_index_tf(seq_tensor):
    n_bases = tf.cast(tf.stack([4**2, 4**1, 4**0]), dtype=tf.int32)
    indices = tf.reduce_sum(tf.multiply(seq_tensor, n_bases), axis=-1)
    return indices

def is_canonical_codon_tf(codon_tensor):
    decoded_codons = tf.strings.reduce_join(tf.gather(list(nucleotides.keys()), tf.transpose(codon_tensor, perm=[1, 0])), axis=0)
    return tf.reduce_all(tf.map_fn(lambda x: tf.reduce_any(tf.equal(tf.constant(list(canonical_codons_set)), x)), decoded_codons), axis=0)

def mutation_penalty_tf(y_true, y_pred, mutation_probs):
    penalty = 0.0
    for i in tf.range(tf.shape(y_pred)[1]):
        current_pred_probs = y_pred[:, i, :]
        current_true_one_hot = y_true[:, i, :]
        current_pred_index = tf.argmax(current_pred_probs, axis=-1)
        current_true_index = tf.argmax(current_true_one_hot, axis=-1)
        mutation_prob = tf.gather_nd(mutation_probs, tf.stack([current_true_index, current_pred_index], axis=-1))
        penalty += (1.0 - mutation_prob)  # Penalize less likely mutations
    return penalty

def codon_usage_reward_tf(y_true, codon_usage_tensor):
    reward = 0.0
    for i in tf.range(0, tf.shape(y_true)[1] - 2, 3):
        codon_indices = tf.stack([tf.argmax(y_true[:, i, :], axis=-1),
                                  tf.argmax(y_true[:, i+1, :], axis=-1),
                                  tf.argmax(y_true[:, i+2, :], axis=-1)], axis=-1)
        flat_indices = get_codon_index