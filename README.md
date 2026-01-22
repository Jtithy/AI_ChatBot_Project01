# AI_ChatBot_Project01 🤖

<p align="center">
  <img src="https://img.shields.io/badge/AI-Chatbot-blueviolet?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Framework-TensorFlow-orange?style=for-the-badge&logo=tensorflow" />
  <img src="https://img.shields.io/badge/Language-Python-blue?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/NLP-NLTK-green?style=for-the-badge" />
</p>

---

## 📌 Project Overview

**AI_ChatBot_Project01** is a simple **AI-based chatbot** built using **Python, NLP (NLTK), and TensorFlow/Keras**. The chatbot is trained on predefined intents using a **Bag-of-Words** approach and a **Neural Network** classifier to understand user queries and generate appropriate responses.

This project is suitable for **beginners learning AI, NLP, and chatbot development**.

---

## 🎯 Objectives

* Understand Natural Language Processing (NLP) basics
* Build and train a chatbot using intents
* Implement Bag-of-Words text representation
* Train a neural network classifier
* Run an interactive chatbot in the terminal

---

## 🧠 How the Chatbot Works

1. User input is tokenized and lemmatized using **NLTK**
2. Input sentence is converted into a **Bag-of-Words vector**
3. A trained **Neural Network model** predicts the intent
4. A suitable response is selected randomly from `intents.json`

---

## 📁 Project Structure

```
AI_ChatBot_Project01/
│-- intents.json
│-- train_chatbot.py
│-- chat.py
│-- words.pkl
│-- classes.pkl
│-- chatbot_demo.h5
│-- README.md
```

---

## 🧩 Files Description

* **intents.json** – Contains intents, patterns, and responses
* **train_chatbot.py** – Script to preprocess data and train the model
* **chat.py** – Script to run the chatbot in terminal
* **words.pkl** – Pickle file storing processed vocabulary
* **classes.pkl** – Pickle file storing intent labels
* **chatbot_demo.h5** – Trained chatbot model

---

## ⚙️ Model Architecture

* Input Layer (Bag-of-Words)
* Dense Layer (128 neurons, ReLU)
* Dropout Layer (0.5)
* Dense Layer (64 neurons, ReLU)
* Output Layer (Softmax)

---

## 🛠 Technologies Used

* Python
* TensorFlow / Keras
* NLTK
* NumPy
* Pickle
* JSON

---

## ▶️ How to Run the Project

### 1️⃣ Install Required Libraries

```bash
pip install tensorflow nltk numpy
```

### 2️⃣ Download NLTK Data (First Time Only)

```python
import nltk
nltk.download('punkt')
nltk.download('wordnet')
```

### 3️⃣ Train the Chatbot Model

```bash
python train_chatbot.py
```

This will generate:

* `chatbot_demo.h5`
* `words.pkl`
* `classes.pkl`

### 4️⃣ Run the Chatbot

```bash
python chat.py
```

Type your message and chat with the bot in the terminal.

---

## 🧪 Example Interaction

```
You: Hello
Bot: Hi! How can I help you today?

You: What can you do?
Bot: I can answer your questions and assist you.
```

---

## 🚀 Future Improvements

* Add more intents and responses
* Use word embeddings (Word2Vec / GloVe)
* Replace Bag-of-Words with LSTM or Transformer
* Add a GUI or a Web interface (Flask / Streamlit)
* Deploy chatbot on WhatsApp or Messenger

---

## 🌐 Useful Learning Resources
[Simplilearn - Python Chatbot Tutorial](https://youtu.be/t933Gh5fNrc?si=VkPxre0JEbRoUHGv)

---

## 📄 License

This project is created for **educational and learning purposes only**.

---

## ✍️ Author

**JTithy**
Date: January 2026

---

✨ *Happy Chatting with AI!* 🤖
