import streamlit as st
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from test_ascpects_based import AspectSentimentAnalyzer

aspectSentimentAnalyzer = AspectSentimentAnalyzer("fine-tuned-bert-base-multilingual-cased-asba", "fine-tuned-bert-base-multilingual-cased-asba")


st.title("Aspect Based Sentiment Analysis of Nepali Romanized English text")
st.write("This app performs aspect-based sentiment analysis on Nepali Romanized English text using a fine tuned bert base multi lingual cased model.")

# Get input text from the user
text = st.text_area("Enter Your Nepali Romanized English Review here:")

if text:
    results = aspectSentimentAnalyzer.predict(text)
    st.write("Aspect Based Sentiment Analysis:")
    for result in results:
        st.write(f"Aspect: {result['aspect']}, Sentiment: {result['label']}, Confidence Score: {result['score']:.2f}")
