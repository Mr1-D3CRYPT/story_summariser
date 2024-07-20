import streamlit as st
import pickle

# Load the trained model
with open('story_genre_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Define the Streamlit app
st.title("Story Genre Predictor")

# Input story with a larger text area
story = st.text_area("Enter the story text here:", height=300)

# Predict genre
if st.button("Predict Genre"):
    if story.strip():
        predicted_genre = model.predict([story])
        st.write(f"Predicted Genre: {predicted_genre[0]}")
    else:
        st.write("Please enter a story to get a prediction.")
