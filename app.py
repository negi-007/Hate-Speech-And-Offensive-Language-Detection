import streamlit as st
import re
import string
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Ensure NLTK resources are downloaded
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Load the Keras model
model = load_model("hateAndOffensiveDetection_model.h5")

# Load the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Function to clean and lemmatize text
def clean_and_lemmatize_text(text):
    # Clean the text
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)  # Remove text within brackets
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
    text = re.sub(r'<.*?>+', '', text)  # Remove HTML tags
    text = re.sub(r'[^\x00-\x7F]+', '', text)  # Remove non-ASCII characters
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)  # Remove punctuation
    text = re.sub(r'\n', '', text)  # Remove newline characters
    text = re.sub(r'\w*\d\w*', '', text)  # Remove words with digits

    # Tokenize the text
    words = nltk.word_tokenize(text)
    
    # Lemmatize each word
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    
    # Filter words that are in the tokenizer's vocabulary
    filtered_words = [word for word in lemmatized_words if word in tokenizer.word_index]
    
    # Join filtered words back into a sentence
    return ' '.join(filtered_words)

# Streamlit interface
st.title("Hate and Offensive Language Detection")
st.write("Enter text to check if it's hate or offensive:")

# Input text box
user_input = st.text_area("Your text here:")

if st.button("Predict"):
    if user_input:
        # Clean and lemmatize the input text
        cleaned_text = clean_and_lemmatize_text(user_input)
        
        # Tokenize and pad the cleaned text
        if cleaned_text:
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
            st.write("No valid words in the input to process.")
    else:
        st.write("Please enter some text.")
