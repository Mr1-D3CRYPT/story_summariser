import streamlit as st
import spacy
from spacy.cli import download
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest

# Download and load the spaCy model
def load_spacy_model():
    try:
        nlp = spacy.load('en_core_web_sm')
    except OSError:
        download('en_core_web_sm')
        nlp = spacy.load('en_core_web_sm')
    return nlp

nlp = load_spacy_model()

# Define a function for summarizing text
def summarize(text):
    doc = nlp(text)
    tokens = [token.text for token in doc]

    word_frequencies = {}
    for word in doc:
        if word.text.lower() not in STOP_WORDS and word.text.lower() not in punctuation:
            if word.text not in word_frequencies.keys():
                word_frequencies[word.text] = 1
            else:
                word_frequencies[word.text] += 1

    max_frequency = max(word_frequencies.values())
    for word in word_frequencies.keys():
        word_frequencies[word] = word_frequencies[word] / max_frequency

    sentence_tokens = [sent for sent in doc.sents]
    sentence_scores = {}
    for sent in sentence_tokens:
        for word in sent:
            if word.text.lower() in word_frequencies.keys():
                if sent not in sentence_scores.keys():
                    sentence_scores[sent] = word_frequencies[word.text.lower()]
                else:
                    sentence_scores[sent] += word_frequencies[word.text.lower()]

    select_length = int(len(sentence_tokens) * 0.3)
    summary_sentences = nlargest(select_length, sentence_scores, key=sentence_scores.get)
    
    # Sort the summary sentences in their original order
    summary_sentences = sorted(summary_sentences, key=lambda x: x.start)
    final_summary = [sent.text for sent in summary_sentences]
    summary = ' '.join(final_summary)
    return summary

# Define the Streamlit app
st.title("Story Summarizer")

# Input story with a larger text area
story = st.text_area("Enter the story text here:", height=300)

# Summarize the story
if st.button("Summarize Story"):
    if story.strip():
        # Generate summary
        summary = summarize(story)
        st.write("**Summary:**")
        st.text(summary)  # Use st.text to display the summary as a continuous block of text
    else:
        st.write("Please enter a story to get a summary.")
