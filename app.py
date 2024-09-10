import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Load the fine-tuned model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("fine-tuned-model")
tokenizer = AutoTokenizer.from_pretrained("fine-tuned-model")

# Create a pipeline for sentiment analysis
sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Define the mapping of label indices to class names
label_map = {0: "neutral", 1: "positive", 2: "negative"}

st.title("Sentiment Analysis with BERT")

text = st.text_area("Enter some text")

if text:
    # Make predictions
    predictions = sentiment_pipeline(text)

    # Map label indices to class names
    mapped_predictions = [{"label": label_map[int(pred['label'].split('_')[-1])], "score": pred['score']} for pred in predictions]
    
    # Display predictions
    st.write("Predictions:")
    for pred in mapped_predictions:
        st.write(f"Label: {pred['label']}, Score: {pred['score']:.4f}")
