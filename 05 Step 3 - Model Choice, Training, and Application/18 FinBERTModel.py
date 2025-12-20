import random
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification, BertConfig
    
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
    
# Load the tokenizer and the model
model_path = "ProsusAI/finbert"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = TFBertForSequenceClassification.from_pretrained(
    model_path,
    config=BertConfig.from_pretrained(model_path, num_labels=2)
)
    
# Create longer synthetic financial data
texts = [
    "The purchase price will be paid in cash and stock upon the closure of the transaction, scheduled for April 1, 2025",
    "With this, the company will exit the contract manufacturing service segment and enter the more profitable software and services market",
    "Commission income fell to SGD 2.1 mn from SGD 5.1 mn in the corresponding period in 2021",
    "Finnish media group Talentum has issued a profit warning followed by no dividend payment for 2023 and entering bankruptcy protection",
    "The loss for the second quarter of 2024 was EUR 0.1 mn smaller than the loss of the first quarter of 2024 indicating strong positive upturn in the business"
]
    
# Preprocess the texts
inputs = tokenizer(
    texts,
    return_tensors='tf',
    padding=True,
    truncation=True,
    max_length=512,
    add_special_tokens=True
)
    
# Ensure model focuses on relevant parts using attention mask
inputs['attention_mask'] = tf.cast(inputs['attention_mask'], dtype=tf.float32)
    
# Get sentiment predictions
outputs = model(**inputs)
predictions = tf.nn.softmax(outputs.logits, axis=–1).numpy()
    
# Define the sentiment labels
labels = ['negative', 'positive']
    
# Print the results
for i, text in enumerate(texts):
    sentiment = labels[tf.argmax(predictions[i]).numpy()]
    confidence = tf.reduce_max(predictions[i]).numpy()
    print(f"Text: {text}\nSentiment: {sentiment} (Confidence: {confidence:.2f}); Predictions: {predictions[i]}\n")