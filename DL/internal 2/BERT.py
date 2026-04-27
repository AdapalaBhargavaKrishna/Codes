from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Step 1: Load pre-trained BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# Step 2: Encode the input text into BERT's format
text = "movie was not good"  # Example input
inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)

# Step 3: Make a prediction
with torch.no_grad():
    outputs = model(**inputs)
logits = outputs.logits
print(logits)

# Step 4: Convert logits to probabilities (for sentiment analysis, we use softmax activation)
probabilities = torch.nn.functional.softmax(logits, dim=-1)

# Step 5: Get the predicted sentiment (0 for negative, 1 for positive)
predicted_class = torch.argmax(probabilities, dim=-1).item()
print("Predicted Sentiment:", "Positive" if predicted_class == 1 else "Negative")

"""
================================================================================
BERT SENTIMENT ANALYSIS (PYTORCH VERSION)
================================================================================

1. ARCHITECTURE
================================================================================
Input: Text sequence ("movie was not good")
↓
BERT Tokenizer (PyTorch - 'pt' tensors):
- Converts text to tokens + attention mask + token type IDs
- Adds [CLS] at start, [SEP] at end
- ["[CLS]", "movie", "was", "not", "good", "[SEP]"]
- Returns PyTorch tensors (not TensorFlow)
↓
BERT-base-uncased Model (110M parameters):
- 12 Transformer Encoder Layers
- Hidden size: 768
- 12 Attention heads
- Feed-forward: 3072
- Activation: GELU
- LayerNorm after each sublayer
- Residual connections
↓
[CLS] token output (768-dim)
↓
Classification Head:
  - Dropout (0.1)
  - Linear: 768 → 2 (logits)
  - No activation (raw scores)
↓
Softmax: Probabilities for [Negative, Positive]

2. NUMBER OF LEARNING PARAMETERS
================================================================================
Component                    | Parameters
-----------------------------|-----------------------------------------------
BERT Base Model              | 109,482,240
Embedding (token, position)  | 23,440,896 + 786,432 = 24,227,328
12 Transformer Layers:
  - Multi-head attention (12) | 28,311,552
  - Feed-forward network     | 56,623,104
  - LayerNorm layers         | 36,864
Classification Head          | 768 × 2 + 2 = 1,538
-----------------------------|-----------------------------------------------
TOTAL TRAINABLE              | ~110 million (109,483,778)

3. LOSS CALCULATION FORMULA
================================================================================
Cross-Entropy Loss (PyTorch implementation):

During training (not in this code):
Loss = CrossEntropyLoss(logits, labels)

CrossEntropyLoss = -1/N Σ log(exp(logits[y_i]) / Σ exp(logits[c]))
                           i=1..N        c

Inference only (this code):
- No loss calculation
- Only forward pass with torch.no_grad()

For binary sentiment:
Binary Cross-Entropy (using logits directly):
Loss = -[y × log(σ(logits)) + (1-y) × log(1-σ(logits))]

where:
- y = true label (0 or 1)
- σ = sigmoid function
- logits = raw model outputs

4. WEIGHT UPDATE FORMULA
================================================================================
AdamW Optimizer (PyTorch implementation):

θ(t+1) = θ(t) - η × [(m̂(t) / (√v̂(t) + ε)) + λθ(t)]

PyTorch optimizer would be:
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

Features:
- Weight decay (λθ) applied separately from gradient
- Bias correction for moments
- Typically lr=2e-5 for BERT fine-tuning
- ε = 1e-8 (default)

5. SAMPLES USED FOR TRAINING, VALIDATION, TESTING
================================================================================
This code (inference only):
- Training: 0 samples
- Validation: 0 samples
- Testing: 1 sample ("movie was not good")

BERT was pre-trained on (before this code):
- Training: 3.3 billion words (BookCorpus + Wikipedia)
- Unlabeled text (Masked LM task)
- Validation: Typically 10% of pre-training data

For actual sentiment fine-tuning (not done here):
- Would need labeled data (e.g., SST-2: 67k training, 872 validation, 1.8k test)

6. LABELED OR UNLABELED DATA?
================================================================================
PRE-TRAINING (before loading model): Unlabeled Data ✓
- BERT learns from raw text without labels
- Masked Language Model (self-supervised)
- Predicts masked tokens using context

FINE-TUNING (not performed here): Labeled Data ✓
- Sentiment analysis requires labels (Positive/Negative)
- Would need labeled examples like:
  - "great movie" → Positive (1)
  - "terrible movie" → Negative (0)

CURRENT INFERENCE (this code): No training data
- No labels needed for prediction
- Pre-trained model predicts sentiment without fine-tuning

7. PYTORCH VS TENSORFLOW DIFFERENCES
================================================================================
Feature                 | TensorFlow Version | PyTorch Version
------------------------|--------------------|--------------------
Tensor Library          | tf.Tensor          | torch.Tensor
Return type             | return_tensors='tf'| return_tensors='pt'
Inference context       | No gradient tape   | torch.no_grad()
Softmax                 | tf.nn.softmax      | torch.nn.functional.softmax
Model class             | TFBert...          | Bert...
Argmax                  | tf.argmax().numpy()| torch.argmax().item()
Training               | model.fit()        | Manual train loop

8. TORCH.NO_GRAD() EXPLAINED
================================================================================
Context manager that disables gradient calculation:
with torch.no_grad():
    outputs = model(**inputs)

Benefits:
- No computation graph built
- Reduced memory usage
- Faster inference (no gradient tracking)
- Essential for inference (not training)

Without torch.no_grad():
- Would still work but waste memory
- Track gradients unnecessarily
- Slower for prediction

9. SOFTMAX vs SIGMOID FOR SENTIMENT
================================================================================
Model has 2 output neurons (binary classification):
Option 1 (model uses): Softmax
probabilities = softmax(logits)  # [p_neg, p_pos], sum=1
predicted_class = argmax(probabilities)

Option 2: Sigmoid (single neuron)
probability = sigmoid(logit)  # 0 to 1
predicted_class = 1 if probability > 0.5 else 0

BERT base model uses Softmax with 2 outputs

10. INPUT TENSORS (WHAT TOKENIZER RETURNS)
================================================================================
inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)

Returns dictionary:
{
    'input_ids': tensor([[101, 3185, 2001, 2025, 2204, 102, 0, ..., 0]]),  # shape (1, 512)
    'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 0, ..., 0]]),             # shape (1, 512)
    'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, ..., 0]])              # shape (1, 512)
}

Passing to model:
model(**inputs)  # Unpacks dictionary as keyword arguments
# Equivalent to: model(input_ids=..., attention_mask=..., token_type_ids=...)

11. LOGITS TO SENTIMENT PROCESS
================================================================================
Example output:
logits = [[1.2, -0.8]]  # Raw scores
↓
softmax → [[0.88, 0.12]]  # Probabilities
↓
argmax → 0 (index 0 has higher probability)
↓
Negative sentiment

Interpretation:
- Logits can be any real number
- Positive logit = evidence for that class
- Softmax converts to probability distribution
- Negative sentiment if p_negative > p_positive

12. WHY PRE-TRAINED BERT CAN DO SENTIMENT
================================================================================
1. Masked Language Modeling:
   - Predicts masked words like "[MASK] movie"
   - Learns "good" has positive associations
   - Learns "bad" has negative associations

2. Next Sentence Prediction:
   - Learns discourse relationships
   - Understands sentiment transition between sentences

3. Large-scale pre-training:
   - Trained on diverse text including reviews
   - Exposure to sentiment expressions
   - Can recognize negation ("not good")

Limitations without fine-tuning:
- Biased toward frequent patterns
- May fail on domain-specific sentiment
- More errors than fine-tuned model

13. PRACTICAL USAGE NOTES
================================================================================
Memory requirements:
- Model: ~420 MB (bert-base-uncased)
- With gradients: ~840 MB
- With torch.no_grad(): ~420 MB (inference only)

Inference speed:
- Single sentence: ~0.05-0.1 seconds on GPU
- Batch of 32: ~0.3-0.5 seconds

For production:
- Use batch inference for efficiency
- Fine-tune on domain data
- Consider smaller models (DistilBERT)
- Use quantization for edge deployment

14. COMPLETE TRAINING EXAMPLE (FOR REFERENCE)
================================================================================
# This would be added for actual training:
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader
import torch.optim as optim

model.train()
optimizer = optim.AdamW(model.parameters(), lr=2e-5)

for epoch in range(3):
    for batch in train_dataloader:
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
"""