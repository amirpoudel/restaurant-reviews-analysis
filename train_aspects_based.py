# #ASPECT BASED _TRAINING
# Import necessary libraries
import pandas as pd
import re
from datasets import load_dataset, ClassLabel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer , pipeline
from sklearn.model_selection import train_test_split
import torch

# Check if GPU is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load the dataset from Hugging Face
dataset = load_dataset("amirpoudel/restaurant_aspect_based_reviews_data")

# Explore the dataset structure
print(dataset)

# Function to flatten the dataset
def flatten_dataset(examples):
    texts, aspects, sentiments = [], [], []
    
    for i, aspect_list in enumerate(examples["aspects"]):
        for aspect_entry in aspect_list:
            texts.append(examples["text"][i])  # Same text for each aspect in the list
            aspects.append(aspect_entry["aspect"])  # Extract aspect
            sentiments.append(aspect_entry["sentiment"])  # Extract sentiment
    
    return {
        "text": texts,
        "aspect": aspects,
        "sentiment": sentiments
    }

# Apply the flattening to the dataset
flattened_dataset = dataset.map(flatten_dataset, batched=True, remove_columns=['aspects'])


# Label encoding for sentiment
sentiment_labels = ClassLabel(names=['neutral', 'positive', 'negative'])

def encode_labels(examples):
    examples['sentiment'] = sentiment_labels.str2int(examples['sentiment'])
    return examples

encoded_dataset = flattened_dataset.map(encode_labels)

# Tokenize the text and aspect together
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

def preprocess_and_tokenize(examples):
    inputs = [f"{text} [ASPECT] {aspect}" for text, aspect in zip(examples["text"], examples["aspect"])]
    tokenized_inputs = tokenizer(inputs, padding="max_length", truncation=True)
    tokenized_inputs["labels"] = examples["sentiment"]  # Ensure that labels are included
    return tokenized_inputs


# Apply tokenization
tokenized_dataset = encoded_dataset.map(preprocess_and_tokenize, batched=True, remove_columns=['text', 'aspect'])

# Split the dataset into training and validation sets
train_val_dataset = tokenized_dataset['train'].train_test_split(test_size=0.2)
train_dataset = train_val_dataset['train']
val_dataset = train_val_dataset['test']

print(f"Training dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")

# Load the Model and Define Training Arguments
model = AutoModelForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=3)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    logging_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir="./logs",
    fp16=True,  # Use mixed precision
    gradient_accumulation_steps=2  # Gradient accumulation
)

# Define the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Fine-tune the model
trainer.train()

# Save the model and tokenizer
model.save_pretrained("./fine-tuned-bert-base-multilingual-cased-asba")
tokenizer.save_pretrained("./fine-tuned-bert-base-multilingual-cased-asba")

# Evaluate the model
results = trainer.evaluate()
print(f"Evaluation results: {results}")



# Use the model for prediction
# Load the fine-tuned model and tokenizer
fine_tuned_model = AutoModelForSequenceClassification.from_pretrained("./fine-tuned-bert-base-multilingual-cased-asba")
fine_tuned_tokenizer = AutoTokenizer.from_pretrained("./fine-tuned-bert-base-multilingual-cased-asba")

# Create a sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis", model=fine_tuned_model, tokenizer=fine_tuned_tokenizer, device=0 if torch.cuda.is_available() else -1)

# Define the mapping of label indices to class names
label_map = {0: "neutral", 1: "positive", 2: "negative"}

# Clean text function (implement this according to your needs, such as removing special characters, etc.)
def clean_text(text):
    # Example cleaning: Lowercase the text and remove extra spaces
    return re.sub(r'\s+', ' ', text.strip().lower())

# Function to test sentiment for multiple aspects in a sentence
def test_aspects_with_model(text, aspects, model_pipeline):
    results = []
    cleaned_text = clean_text(text)
    
    for aspect in aspects:
        # Combine text with the aspect
        input_text = f"{cleaned_text} [ASPECT] {aspect}"
        
        # Make prediction
        prediction = model_pipeline(input_text)
        
        # Map label index to class names and append result
        results.append({
            "aspect": aspect,
            "label": label_map[int(prediction[0]['label'].split('_')[-1])],
            "score": prediction[0]['score']
        })
    
    return results

# Test sentence and aspects
test_sentence = "Thakali chicken set khana mitho xa, environment ni ekdam peace family friends sanga dinner aauna ekdam fit hunxa"
aspects = ["food", "environment"]  # Multiple aspects for the sentence

# Test the model with multiple aspects
predictions = test_aspects_with_model(test_sentence, aspects, sentiment_pipeline)

# Print the results
for result in predictions:
    print(f"Aspect: {result['aspect']}, Sentiment: {result['label']}, Confidence Score: {result['score']:.2f}")

