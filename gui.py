import re
import numpy as np
import tensorflow as tf
import speech_recognition as sr
import tkinter as tk
from tkinter import messagebox
from datetime import datetime
from keras.layers import Input, LSTM, Dense
from keras.models import Model

# --- Model Loading ---
input_features_dict = np.load("Data/input_features_dict.npy", allow_pickle=True).item()
target_features_dict = np.load("Data/target_features_dict.npy", allow_pickle=True).item()
reverse_target_features_dict = np.load("Data/reverse_target_features_dict.npy", allow_pickle=True).item()
max_lengths = np.load("Data/max_lengths.npy")

max_encoder_seq_length, max_decoder_seq_length = max_lengths[0], max_lengths[1]
num_encoder_tokens = len(input_features_dict)
num_decoder_tokens = len(target_features_dict)

dimensionality = 256

# Define the model architecture
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

# Load the trained model
model.load_weights("Data/trained_model.keras")

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

decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

# --- Speech Recognition Integration ---
recognizer = sr.Recognizer()

def listen_to_audio():
    with sr.Microphone() as source:
        print("Listening for English speech...")
        recognizer.adjust_for_ambient_noise(source)
        print("Listening...")
        audio = recognizer.listen(source)
        try:
            english_text = recognizer.recognize_google(audio)
            print(f"Recognized text: {english_text}")
            return english_text
        except sr.UnknownValueError:
            print("Sorry, I could not understand the audio. Please try again.")
            return None
        except sr.RequestError:
            print("Could not request results from Google Speech Recognition service.")
            return None

# --- Translation Function (Word-by-Word) ---
def translate_word_to_hindi(english_word):
    # Normalize and preprocess the word (remove punctuation, make it lowercase)
    english_word = re.sub(r'[^\w\s]', '', english_word).lower()

    # Preprocess and translate English word to Hindi using the trained model
    input_sequence = np.zeros((1, max_encoder_seq_length, num_encoder_tokens), dtype='float32')
    
    # Convert the single word to one-hot encoding
    for t, word in enumerate(english_word.split()):
        if word in input_features_dict:
            input_sequence[0, t, input_features_dict[word]] = 1.0
        else:
            print(f"Word '{word}' not found in input_features_dict")  # Debugging line
    
    # Debugging: Check the shape of input_sequence and whether it's being populated correctly
    print(f"Input sequence for '{english_word}': {input_sequence}")
    
    # Predict the Hindi translation for the word
    states_value = encoder_model.predict(input_sequence)

    # Generate the predicted Hindi word
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    target_seq[0, 0, target_features_dict['<START>']] = 1.0  # start token
    
    decoded_word = []
    for _ in range(max_decoder_seq_length):
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        
        # Get the predicted word and add it to the decoded sequence
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = reverse_target_features_dict[sampled_token_index]
        
        # Debugging: Log the output token and sampled token at each step
        print(f"Output tokens: {output_tokens[0, -1, :]}")
        print(f"Sampled token index: {sampled_token_index}, Sampled token: {sampled_token}")
        
        if sampled_token == '<END>':
            break  # Stop when the <END> token is encountered
        
        decoded_word.append(sampled_token)
        
        # Update the target sequence (which is the previous predicted word)
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.0
        states_value = [h, c]

    # Only return the first word (without <END> token)
    return decoded_word[0] if decoded_word else "Translation error"

# --- GUI Implementation ---
def start_translation():
    english_text = listen_to_audio()
    if english_text:
        words = english_text.split()
        translated_words = []
        for word in words:
            hindi_translation = translate_word_to_hindi(word)
            translated_words.append(f"{word} -> {hindi_translation}")
        
        translation_result = "\n".join(translated_words)
        messagebox.showinfo("Word-by-Word Translation", f"{translation_result}")
    else:
        messagebox.showinfo("Error", "Sorry, I could not understand the audio. Please try again.")

root = tk.Tk()
root.title("Voice Translator")
root.geometry("400x300")

# Start translation button
translate_button = tk.Button(root, text="Start Translation", command=start_translation)
translate_button.pack(pady=20)

root.mainloop()
