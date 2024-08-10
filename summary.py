import streamlit as st
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Load SpaCy model
nlp = spacy.load('en_core_web_sm')

def extract_key_sentences(text, num_sentences=3):
    # Tokenize and split text into sentences
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]

    # Initialize TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(stop_words='english')

    # Transform sentences to TF-IDF matrix
    tfidf_matrix = vectorizer.fit_transform(sentences)

    # Calculate sentence scores based on TF-IDF
    scores = np.array(tfidf_matrix.sum(axis=1)).flatten()

    # Get indices of the top sentences
    ranked_indices = scores.argsort()[-num_sentences:][::-1]

    # Extract top sentences
    summary = ' '.join(np.array(sentences)[ranked_indices])

    return summary

# Streamlit app
st.title("Text Summarization with TF-IDF and SpaCy")

text = st.text_area("Enter the text to summarize", height=300)
num_sentences = st.slider("Number of sentences in summary", min_value=1, max_value=10, value=3)

if st.button("Summarize"):
    if text:
        summary = extract_key_sentences(text, num_sentences)
        st.subheader("Summary")
        st.write(summary)
    else:
        st.warning("Please enter some text to summarize.")
