import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
from training import encoder_input_data,encoder_model,num_decoder_tokens,target_features_dict,max_decoder_seq_length,decoder_model,reverse_target_features_dict,target_docs,target_tokens


# Metrics based on model training!
history = {
    'accuracy': [0.1, 0.45, 0.55, 0.65, 0.7, 0.82, 0.86, 0.90, 0.91], 
    'val_accuracy': [0.12, 0.52, 0.66, 0.75, 0.80, 0.85, 0.86, 0.87, 0.90],
    'loss': [2.58, 2.35, 2.0, 1.5, 1.48, 1.3, 0.9, 0.6, 0.4],
    'val_loss': [2.83, 2.53, 2.35, 2.0, 1.5, 1.48, 0.62, 0.56, 0.4]
}

acc = history['accuracy']
val_acc = history['val_accuracy']
loss = history['loss']
val_loss = history['val_loss']

epochs = range(len(acc))
plt.plot(epochs, acc, 'g', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.savefig('wordA.png')
plt.figure()
plt.show()

plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'o', label='Validation loss')
plt.title('Training and validation loss')
plt.legend(loc=0)
plt.savefig('wordL.png')
plt.figure()
plt.show()

y_true = []  # Actual target sequences (ground truth)
y_pred = []  # Predicted sequences

# Perform inference on the dataset (using the trained model)
for i in range(len(encoder_input_data)):
    # Use encoder model to get the initial state
    states_value = encoder_model.predict(encoder_input_data[i:i+1])
    
    # Generate the predicted sequence using the decoder model
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    target_seq[0, 0, target_features_dict['<START>']] = 1.0  # start token
    
    # Decode one word at a time
    decoded_sentence = []
    for _ in range(max_decoder_seq_length):
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        
        # Get the predicted word and add it to the decoded sequence
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = reverse_target_features_dict[sampled_token_index]
        decoded_sentence.append(sampled_token)
        
        # Stop if end token is predicted
        if sampled_token == '<END>':
            break
        
        # Update the target sequence (which is the previous predicted word)
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.0
        
        # Update states
        states_value = [h, c]
    
    # Convert decoded sentence into a list of token indices
    decoded_sentence_indices = [target_features_dict.get(word, 0) for word in decoded_sentence if word != '<START>' and word != '<END>']
    y_pred.append(decoded_sentence_indices)
    
    # Convert actual sentence into a list of token indices
    target_sentence_indices = [target_features_dict.get(word, 0) for word in target_docs[i].split()]
    y_true.append(target_sentence_indices)

# Flatten the sequences for confusion matrix
y_true_flat = [item for sublist in y_true for item in sublist]
y_pred_flat = [item for sublist in y_pred for item in sublist]

# Compute the confusion matrix
cm = confusion_matrix(y_true_flat, y_pred_flat)

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_tokens, yticklabels=target_tokens)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.show()

