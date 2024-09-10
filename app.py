import streamlit as st
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

model_path = "fine-tuned-model"

# Function to clean and normalize the text
def clean_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove special characters (except alphanumeric and spaces)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Load the fine-tuned model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Create a pipeline for sentiment analysis
sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Define the mapping of label indices to class names
label_map = {0: "neutral", 1: "positive", 2: "negative"}

st.title("Sentiment Analysis with BERT")

# Get input text from the user
text = st.text_area("Enter some text")

if text:
    # Clean the input text
    cleaned_text = clean_text(text)
    
    # Make predictions on cleaned text
    predictions = sentiment_pipeline(cleaned_text)

    # Map label indices to class names
    mapped_predictions = [{"label": label_map[int(pred['label'].split('_')[-1])], "score": pred['score']} for pred in predictions]
    
    # Display predictions
    st.write("Predictions:")
    for pred in mapped_predictions:
        st.write(f"Label: {pred['label']}, Score: {pred['score']:.4f}")
