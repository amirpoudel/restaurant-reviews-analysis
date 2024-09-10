# Import necessary libraries
import pandas as pd
import re
from datasets import load_dataset, ClassLabel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, pipeline
from sklearn.model_selection import train_test_split
import torch

# Check if GPU is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load the dataset from Hugging Face
dataset = load_dataset("amirpoudel/bert-reviews-data")

# Explore the dataset
print(dataset)

# Define the classes and ensure they match with the dataset
class_names = ['neutral', 'positive', 'negative']
labels = ClassLabel(names=class_names)

# Function to clean and normalize the text
def clean_text(text):
    text = text.lower()  # Convert text to lowercase
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

# Encode Labels
def encode_labels(examples):
    examples['label'] = labels.str2int(examples['label'])
    return examples

# Apply label encoding to the dataset
encoded_dataset = dataset.map(encode_labels)

# Tokenize the Text - replace with bert-base-multilingual-cased
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def preprocess_and_tokenize(examples):
    examples['text'] = [clean_text(text) for text in examples['text']]  # Apply text cleaning
    return tokenizer(examples["text"], padding="max_length", truncation=True)  # Tokenize the cleaned text

# Apply preprocessing and tokenization
tokenized_dataset = encoded_dataset.map(preprocess_and_tokenize, batched=True)

# Split the dataset into training and validation sets
train_val_dataset = tokenized_dataset['train'].train_test_split(test_size=0.2)
train_dataset = train_val_dataset['train']
val_dataset = train_val_dataset['test']

print(train_dataset)
print(val_dataset)

# Load the Model and Define Training Arguments
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)

training_args = TrainingArguments(
    output_dir="./results",  # Directory to save results
    evaluation_strategy="epoch",  # Evaluate after each epoch
    logging_strategy="epoch",  # Log after each epoch
    learning_rate=2e-5,  # Learning rate
    per_device_train_batch_size=4,  # Batch size for training
    per_device_eval_batch_size=4,  # Batch size for evaluation
    num_train_epochs=5,  # Number of epochs
    weight_decay=0.01,  # Weight decay for regularization
    logging_dir="./logs",  # Directory to save logs
    fp16=True,  # Use mixed precision training to save memory
    gradient_accumulation_steps=2  # Accumulate gradients over multiple steps
)

# Define the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Fine-Tune the Model
trainer.train()

# Save the Model
model.save_pretrained("./fine-tuned-model-bert-base-uncased")
tokenizer.save_pretrained("./fine-tuned-model-bert-base-uncased")

# Evaluate the Model
results = trainer.evaluate()
print(results)

# Use the Model for Prediction
fine_tuned_model = AutoModelForSequenceClassification.from_pretrained("./fine-tuned-model-bert-base-uncased")
fine_tuned_tokenizer = AutoTokenizer.from_pretrained("./fine-tuned-model-bert-base-uncased")

# Create a pipeline for sentiment analysis
sentiment_pipeline = pipeline("sentiment-analysis", model=fine_tuned_model, tokenizer=fine_tuned_tokenizer, device=0 if torch.cuda.is_available() else -1)

# Define the mapping of label indices to class names
label_map = {0: "neutral", 1: "positive", 2: "negative"}

# Make predictions on new text (clean text before inference)
new_text = ["Thakali chicken set khana mitho xa, environment ni ekdam peace family friends sanga dinner aauna ekdam fit hunxa"]

# Clean the new text before passing it to the model
cleaned_text = [clean_text(text) for text in new_text]
predictions = sentiment_pipeline(cleaned_text)

# Map label indices to class names
mapped_predictions = [{"label": label_map[int(pred['label'].split('_')[-1])], "score": pred['score']} for pred in predictions]
print(mapped_predictions)
