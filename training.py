import re
import random
import numpy as np
import tensorflow as tf
from keras.layers import Input, LSTM, Dense
from keras.models import Model

data_path = "Data/English.txt"
data_path2 = "Data/Hindi.txt"

# Load and preprocess the dataset
with open(data_path, 'r', encoding='utf-8') as f:
    lines = f.read().strip().split('\n')
with open(data_path2, 'r', encoding='utf-8') as f:
    lines2 = f.read().strip().split('\n')

lines = [" ".join(re.findall(r"[A-Za-z0-9]+", line)) for line in lines]
lines2 = [re.sub(r"[^\u0900-\u097F\s]", '', line) for line in lines2]  # Retain only Hindi characters and spaces

# Pair English and Hindi sentences and shuffle them
pairs = list(zip(lines, lines2))
random.shuffle(pairs)

# Tokenization and data preparation
input_docs, target_docs = [], []
input_tokens, target_tokens = set(), set()
for input_doc, target_doc in pairs:
    input_docs.append(input_doc)
    target_doc = f"<START> {target_doc} <END>"
    target_docs.append(target_doc)
    input_tokens.update(re.findall(r"[\w']+|[^\s\w]", input_doc))
    target_tokens.update(target_doc.split())

input_tokens = sorted(list(input_tokens))
target_tokens = sorted(list(target_tokens))
num_encoder_tokens = len(input_tokens)
num_decoder_tokens = len(target_tokens)

# Create token-to-index mappings
input_features_dict = {token: i for i, token in enumerate(input_tokens)}
target_features_dict = {token: i for i, token in enumerate(target_tokens)}
reverse_target_features_dict = {i: token for token, i in target_features_dict.items()}

# Sequence lengths
max_encoder_seq_length = max(len(re.findall(r"[\w']+|[^\s\w]", doc)) for doc in input_docs)
max_decoder_seq_length = max(len(doc.split()) for doc in target_docs)

# Prepare data arrays
encoder_input_data = np.zeros((len(input_docs), max_encoder_seq_length, num_encoder_tokens), dtype='float32')
decoder_input_data = np.zeros((len(input_docs), max_decoder_seq_length, num_decoder_tokens), dtype='float32')
decoder_target_data = np.zeros((len(input_docs), max_decoder_seq_length, num_decoder_tokens), dtype='float32')

for i, (input_doc, target_doc) in enumerate(zip(input_docs, target_docs)):
    for t, token in enumerate(re.findall(r"[\w']+|[^\s\w]", input_doc)):
        encoder_input_data[i, t, input_features_dict[token]] = 1.0
    for t, token in enumerate(target_doc.split()):
        decoder_input_data[i, t, target_features_dict[token]] = 1.0
        if t > 0:
            decoder_target_data[i, t - 1, target_features_dict[token]] = 1.0

# Define and compile the model
dimensionality = 256
batch_size = 128
epochs = 50

encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder_lstm = LSTM(dimensionality, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)
encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(None, num_decoder_tokens))
decoder_lstm = LSTM(dimensionality, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size, epochs=epochs, validation_split=0.2)

# Save the training model
model.save("Data/trained_model.keras")

# Define inference encoder model
encoder_model = Model(encoder_inputs, encoder_states)

# Define inference decoder model
decoder_state_input_h = Input(shape=(dimensionality,))
decoder_state_input_c = Input(shape=(dimensionality,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs
)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)

decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states
)

# Save inference models
encoder_model.save("Data/encoder_model.keras")
decoder_model.save("Data/decoder_model.keras")

# Save tokenization details
np.save("Data/input_features_dict.npy", input_features_dict)
np.save("Data/target_features_dict.npy", target_features_dict)
np.save("Data/reverse_target_features_dict.npy", reverse_target_features_dict)
np.save("Data/max_lengths.npy", [max_encoder_seq_length, max_decoder_seq_length])
