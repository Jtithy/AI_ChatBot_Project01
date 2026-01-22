# Author: JTithy
# Date: 22nd January, 2026

# Import libraries
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model

# Lemmatizer reduces words to base form
lemmatizer = WordNetLemmatizer()

# Load intents file
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

# Load the trained chatbot model
model = load_model("chatbot_demo.h5")

# Function to clean and preprocess user input
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

# Convert a sentence into a bag-of-words vector
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i]=1
                
    return np.array(bag)

# Predict the intent class of a given sentence
def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0] 
    
    # Minimum probability required to consider an intent
    ERROR_TRESHOLD = 0.25
    
    # Filter intents with probability above the threshold
    results = [[i,r] for i, r in enumerate(res) if r>ERROR_TRESHOLD]
    results.sort(key = lambda x:x[1], reverse = True)
    
    # Prepare final list of intents with probabilities
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability':str(r[1])})
    # Return predicted intents
    return return_list      

# Select a response based on predicted intent
def get_response(intents_list, intents_json):
    list_of_intents = intents_json['intents']
    tag = intents_list[0]['intents']
    # Search for matching intent in JSON
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['response'])
            break
    # Return the selected response
    return result
# Confirmation message
print("Great! Bot is running.")

# Run chatbot continuously
while True:
    message = input("")
    ints = predict_class(message)
    res = get_response(ints, intents)
    # Print chatbot response
    print(res)