import re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

class SentimentAnalyzer:
    
    def __init__(selft,model_path,tokenizer_path):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )
        
        self.label_map = {0: "neutral", 1: "positive", 2: "negative"}

    def clean_text(self,text):
        return re.sub(r'\s+',' ',text.strip().lower())

    def predict(self,text):
        cleaned_text = self.clean_text(text)
        prediction = self.sentiment_pipeline(cleaned_text)
        return {
            "label": self.label_map[int(prediction[0]['label'].split('_')[-1])],
            "score": prediction[0]['score']
        }

    