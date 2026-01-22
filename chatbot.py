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
lemmatizer = WordNetLemmatizer(0)

# Load intents file
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

model = load_model("chatbot_demo.h5")
