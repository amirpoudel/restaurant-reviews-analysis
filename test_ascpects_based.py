# from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
# import torch
# import re
# # Use the model for prediction
# # Load the fine-tuned model and tokenizer
# fine_tuned_model = AutoModelForSequenceClassification.from_pretrained("./fine-tuned-bert-base-multilingual-cased-asba")
# fine_tuned_tokenizer = AutoTokenizer.from_pretrained("./fine-tuned-bert-base-multilingual-cased-asba")

# # Create a sentiment analysis pipeline
# sentiment_pipeline = pipeline("sentiment-analysis", model=fine_tuned_model, tokenizer=fine_tuned_tokenizer, device=0 if torch.cuda.is_available() else -1)

# # Define the mapping of label indices to class names
# label_map = {0: "neutral", 1: "positive", 2: "negative"}

# # Clean text function (implement this according to your needs, such as removing special characters, etc.)
# def clean_text(text):
#     # Example cleaning: Lowercase the text and remove extra spaces
#     return re.sub(r'\s+', ' ', text.strip().lower())

# # Function to test sentiment for multiple aspects in a sentence
# def test_aspects_with_model(text, aspects, model_pipeline):
#     results = []
#     cleaned_text = clean_text(text)
    
#     for aspect in aspects:
#         # Combine text with the aspect
#         input_text = f"{cleaned_text} [ASPECT] {aspect}"
        
#         # Make prediction
#         prediction = model_pipeline(input_text)
        
#         # Map label index to class names and append result
#         results.append({
#             "aspect": aspect,
#             "label": label_map[int(prediction[0]['label'].split('_')[-1])],
#             "score": prediction[0]['score']
#         })
    
#     return results

# # Test sentence and aspects
# test_sentence = "Thakali chicken set khana mitho xa, environment ni ekdam peace family friends sanga dinner aauna ekdam fit hunxa"
# aspects = ["food", "environment"]  # Multiple aspects for the sentence

# # Test the model with multiple aspects
# predictions = test_aspects_with_model(test_sentence, aspects, sentiment_pipeline)

# # Print the results
# for result in predictions:
#     print(f"Aspect: {result['aspect']}, Sentiment: {result['label']}, Confidence Score: {result['score']:.2f}")





import re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

class AspectSentimentAnalyzer:
    def __init__(self, model_path, tokenizer_path):
        # Load the fine-tuned model and tokenizer
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        # Create a sentiment analysis pipeline
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis", 
            model=self.model, 
            tokenizer=self.tokenizer, 
            device=0 if torch.cuda.is_available() else -1
        )
        
        # Define the mapping of label indices to class names
        self.label_map = {0: "neutral", 1: "positive", 2: "negative"}

        # Predefined aspects to search for in the text
        self.predefined_aspects = {
            "food": ["food", "khana", "dish", "meal"],
            "service": ["service", "waiter", "staff"],
            "environment": ["environment", "ambience", "atmosphere"],
            "price": ["price", "cost", "value", "expensive", "cheap"]
        }
        

    def clean_text(self, text):
        """
        Clean text by lowercasing and removing extra spaces.
        """
        return re.sub(r'\s+', ' ', text.strip().lower())

    def find_aspect(self, text):
        """
        Find & match predefined aspects in the text.
        Returns a list of aspects found in the input text, or 'general' if none are found.
        """
        found_aspects = []
        cleaned_text = self.clean_text(text)
        text_words = cleaned_text.split()

        # Check predefined aspects by matching keywords in the text
        for aspect, keywords in self.predefined_aspects.items():
            if any(keyword in text_words for keyword in keywords):
                found_aspects.append(aspect)

        # If no aspect is found, return 'general'
        if not found_aspects:
            found_aspects.append("general")

        return found_aspects


    def predict(self, text, aspects):
        """
        Test sentiment for multiple aspects in a sentence.
        aspects: List of aspects to test.
        """
        results = []
        cleaned_text = self.clean_text(text)
        aspects = self.find_aspect(text)
        for aspect in aspects:
            # Combine text with the aspect
            input_text = f"{cleaned_text} [ASPECT] {aspect}"
            
            # Make prediction
            prediction = self.sentiment_pipeline(input_text)
            
            # Map label index to class names and append result
            results.append({
                "aspect": aspect,
                "label": self.label_map[int(prediction[0]['label'].split('_')[-1])],
                "score": prediction[0]['score']
            })
        
        return results
