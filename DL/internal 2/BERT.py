from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf


# Step 1: Load pre-trained BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')

# Step 2: Encode the input text into BERT's format
text = "movie was not good"  # Example input
inputs = tokenizer(text, return_tensors='tf', truncation=True, padding=True, max_length=512)

# Step 3: Make a prediction
outputs = model(inputs)
logits = outputs.logits
print(logits)

# Step 4: Convert logits to probabilities (for sentiment analysis, we use sigmoid activation)
probabilities = tf.nn.softmax(logits, axis=-1)

# Step 5: Get the predicted sentiment (0 for negative, 1 for positive)
predicted_class = tf.argmax(probabilities, axis=-1).numpy()[0]
print("Predicted Sentiment:", "Positive" if predicted_class == 1 else "Negative")