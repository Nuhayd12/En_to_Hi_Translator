import matplotlib.pyplot as plt


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
