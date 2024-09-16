import streamlit as st
import os
import re
import string
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence

# Load the Keras model
if os.path.exists("./hateAndOffensiveDetection_model.h5"):
    model = load_model("./hateAndOffensiveDetection_model.h5")
else:
    print("Model file not found!")

# Load the tokenizer
# Check if the tokenizer file exists
if os.path.exists('tokenizer.pickle'):
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
else:
    print("Tokenizer file not found!")

# Function to clean text (without lemmatization)
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)  # Remove text within brackets
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
    text = re.sub(r'<.*?>+', '', text)  # Remove HTML tags
    text = re.sub(r'[^\x00-\x7F]+', '', text)  # Remove non-ASCII characters
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)  # Remove punctuation
    text = re.sub(r'\n', '', text)  # Remove newline characters
    text = re.sub(r'\w*\d\w*', '', text)  # Remove words with digits
    return text

# Streamlit interface
st.title("Hate and Offensive Language Detection")
st.write("Enter text to check if it's hate or offensive:")

# Input text box
user_input = st.text_area("Your text here:")

if st.button("Predict"):
    if user_input:
        # Clean the input text
        cleaned_text = clean_text(user_input)
        
        # Tokenize and pad the text
        seq = tokenizer.texts_to_sequences([cleaned_text])
        padded = sequence.pad_sequences(seq, maxlen=300)

        # Predict using the loaded model
        pred = model.predict(padded)

        # Display prediction result
        st.write("Prediction:", pred[0][0])
        if np.mean(pred) < 0.5:
            st.write("Result: No hate or offensive language detected.")
        else:
            st.write("Result: Hate and offensive language detected.")
    else:
        st.write("Please enter some text.")
