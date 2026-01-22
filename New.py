# Author: JTithy
# Date: 21st January, 2026


# Required libraries
import random
import json
import pickle
import numpy as np
import tensorflow as tf
import nltk
from nltk.stem import WordNetLemmatizer

#NTKL Data Download
nltk.download('punkt')
nltk.download('wordnet')

# Lemmatizer reduces words to base form (e.g., running → run)
lemmatizer = WordNetLemmatizer()

# Load intents file
intents = json.loads(open('intents.json').read())

# Lists to store words, intent classes, and training documents
words = []
classes = []
documents = []

# Characters to ignore during training
ignoreLetters = ['?', '!', '.', ',']

# Loop through each intent and its patterns
for intent in intents['intents']:
    for pattern in intent['patterns']:
        wordList = nltk.word_tokenize(pattern)
        words.extend(wordList)
        documents.append((wordList, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
            
# Lemmatize words and remove ignored characters
words = [lemmatizer.lemmatize(word) for word in words if word not in ignoreLetters]

# Remove duplicates and sort
words = sorted(set(words))
classes = sorted(set(classes))

# Save processed words and classes
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Prepare training data
training = []
# Output template for one-hot encoding
outputEmpty = [0] * len(classes)

# Create bag-of-words and output vectors
for document in documents:
    bag = []
    wordPatterns = document[0]
    wordPatterns = [lemmatizer.lemmatize(word.lower()) for word in wordPatterns]
    for word in words: bag.append(1) if word in wordPatterns else bag.append(0)
    
    outputRow = list(outputEmpty)
    outputRow[classes.index(document[1])] = 1
    training.append(bag + outputRow)

# Shuffle training data
random.shuffle(training)
training = np.array(training)

# Split input (X) and output (Y)
trainX = training[:, :len(words)]
trainY = training[:, len(words):]


# Initialize neural network model
model = tf.keras.Sequential()

# input_shape defines the number of features (bag-of-words size)
model.add(tf.keras.layers.Dense(
    128,
    input_shape=(len(trainX[0]),),
    activation='relu'
))
model.add(tf.keras.layers.Dropout(0.5))

# Second hidden layer with 64 neurons
model.add(tf.keras.layers.Dense(
    64, 
    activation='relu'))

# Softmax converts outputs to probabilities
model.add(tf.keras.layers.Dense(
    len(trainY[0]), 
    activation = 'softmax'))

sgd = tf.keras.optimizers.SGD(learning_rate = 0.01, momentum = 0.9, nesterov = True)

model.compile(loss = 'categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])
hist = model.fit(np.array(trainX), np.array(trainY), epochs = 200, batch_size = 5, verbose=1)

# Save train model to a file
model.save('chatbot_demo.h5')
print("Executed")
