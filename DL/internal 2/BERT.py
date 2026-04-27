from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')

text = "movie was not good"
inputs = tokenizer(text, return_tensors='tf', truncation=True, padding=True, max_length=512)

outputs = model(inputs)
logits = outputs.logits
print(logits)

probabilities = tf.nn.softmax(logits, axis=-1)
predicted_class = tf.argmax(probabilities, axis=-1).numpy()[0]
print("Predicted Sentiment:", "Positive" if predicted_class == 1 else "Negative")

"""
================================================================================
BERT SENTIMENT ANALYSIS (TRANSFER LEARNING)
================================================================================

1. ARCHITECTURE
================================================================================
Input: Text sequence ("movie was not good")
↓
BERT Tokenizer:
- Converts text to tokens + attention mask + token type IDs
- Adds [CLS] at start, [SEP] at end
- ["[CLS]", "movie", "was", "not", "good", "[SEP]"]
- Maps to token IDs (vocab size 30,522)
↓
BERT-base-uncased Model (12 layers):
Layer 0: Token Embeddings (30,522 → 768) + Position + Segment
Layer 1-12: Transformer Encoder Blocks
  - Multi-Head Self-Attention (12 heads)
  - Feed-Forward Network (768 → 3072 → 768)
  - LayerNorm + Residual connections
  - GELU activation
↓
[CLS] token output (classification token)
↓
Classification Head:
  - Dense: 768 → 2 neurons
  - Output: logits (2 numbers)
↓
Softmax: Probabilities for [Negative, Positive]

Model Details:
- Encoder layers: 12
- Hidden size: 768
- Attention heads: 12
- Total parameters: 110 million

2. NUMBER OF LEARNING PARAMETERS
================================================================================
Component                    | Parameters
-----------------------------|-----------------------------------------------
BERT Base Model              | 109,482,240 (frozen? No - trainable)
Embedding Layer              | 30,522 × 768 = 23,440,896
Position + Segment Embeddings| 768 × 512 × 2 = 786,432
12 Transformer Layers:
  - Self-attention (12 heads) | 12 × 4 × (768×768) = 28,311,552
  - Feed-forward (3072)       | 12 × (768×3072 + 3072×768) = 56,623,104
  - LayerNorm + Bias          | 12 × (768×2) × 2 = 36,864
Classification Head          | 768 × 2 + 2 = 1,538
-----------------------------|-----------------------------------------------
TOTAL TRAINABLE              | ~110 million parameters

3. LOSS CALCULATION FORMULA
================================================================================
Binary Classification (Negative vs Positive):
MFM (Masked Language Model) pre-training + Fine-tuning loss

During fine-tuning (sentiment analysis):
Loss = -1/N Σ [y_i × log(p_i) + (1-y_i) × log(1-p_i)]

where:
- N = batch size
- y_i = true label (0 = Negative, 1 = Positive)
- p_i = predicted probability (after softmax)

Example for "movie was not good" (Negative sentiment):
y = 0
p_negative = 0.85, p_positive = 0.15
Loss = -[0 × log(0.85) + 1 × log(0.15)] = -log(0.15) = 1.897

4. WEIGHT UPDATE FORMULA
================================================================================
AdamW Optimizer (BERT default):

θ(t+1) = θ(t) - η × [(m̂(t) / (√v̂(t) + ε)) + λθ(t)]

where:
- θ(t) = weights at step t
- η = learning rate (typically 2e-5 for fine-tuning)
- m̂(t) = bias-corrected first moment
- v̂(t) = bias-corrected second moment
- λ = weight decay (prevents overfitting)
- ε = 10⁻⁸

Key difference from Adam: Separate weight decay term (λθ)

5. SAMPLES USED FOR TRAINING, VALIDATION, TESTING
================================================================================
In this specific code (inference only):
- Training: 0 (no training performed)
- Validation: 0
- Testing: 1 (single sentence: "movie was not good")

BERT was PRE-TRAINED on:
- Training: 3.3 billion words (BookCorpus + English Wikipedia)
- Data: Unlabeled text for Masked LM + Next Sentence Prediction
- Fine-tuning (sentiment): Typically uses labeled datasets like IMDb (25k training)

This code uses PRE-TRAINED BERT for inference:
- No fine-tuning on sentiment data
- BERT was pre-trained on generic text, not specifically sentiment

6. LABELED OR UNLABELED DATA?
================================================================================
PRE-TRAINING: Unlabeled Data ✓
- BERT pre-training uses UNLABELED text
- Masked Language Model (predict masked words) - self-supervised
- Next Sentence Prediction (binary classification) - self-supervised

FINE-TUNING FOR SENTIMENT: Labeled Data ✓
- Sentiment analysis requires labeled data (positive/negative)
- This code does NOT fine-tune (uses pre-trained model directly)

For actual sentiment analysis: Would need labeled reviews:
- Input: "movie was great" → Label: Positive (1)
- Input: "movie was terrible" → Label: Negative (0)

================================================================================
BERT INPUT FORMAT
================================================================================
Tokenization Process:
Input text: "movie was not good"
↓
Add special tokens: ["[CLS]", "movie", "was", "not", "good", "[SEP]"]
↓
Token IDs: [101, 3185, 2001, 2025, 2204, 102]
↓
Attention Mask: [1, 1, 1, 1, 1, 1] (1 = real token, 0 = padding)
↓
Token Type IDs: [0, 0, 0, 0, 0, 0] (sentence A = 0)
↓
Padding to max_length=512: Adds zeros

Why [CLS] token?
- Special token at sequence start
- Final hidden state used for classification
- Aggregates entire sequence information

Why truncation=True?
- BERT max length = 512 tokens
- Longer texts are truncated

================================================================================
TRANSFORMER VS LSTM
================================================================================
Feature                 | LSTM (prev examples) | BERT Transformer
------------------------|----------------------|--------------------
Architecture            | Sequential (RNN)     | Parallel (Attention)
Context                 | Limited (vanishing)  | Full bidirectional
Positional info         | Natural (time steps) | Position embeddings
Training speed          | Slower (sequential)  | Faster (parallel)
Long-range dependencies | Struggles >100 tokens| Handles 512 tokens
Parameters (base)       | ~10M (custom)        | 110M (pre-trained)
Transfer learning       | From scratch         | Fine-tuning

================================================================================
ATTENTION MECHANISM (SIMPLIFIED)
================================================================================
Query, Key, Value (Q, K, V):
Attention(Q,K,V) = softmax(QK^T/√d_k) V

For "movie was not good":
- Each word pays attention to all other words
- "not" attends strongly to "good" (negation)
- "good" attends to "movie" (subject)
- Bidirectional: Sees left + right context

Multi-head attention (12 heads):
- Different heads learn different relationships
- Head1: Grammar patterns
- Head2: Negation detection
- Head3: Subject-verb agreement

================================================================================
WHAT BERT LEARNS
================================================================================
Pre-training tasks:
1. Masked Language Model (15% of tokens masked)
   - Predict masked word from context
   - "movie was [MASK] good" → predicts "not"
   - Learns word relationships

2. Next Sentence Prediction (50% real, 50% random)
   - Predict if sentence B follows sentence A
   - Learns discourse and coherence

Fine-tuning (this would need labeled data):
- Replace classification head (768→2)
- Train on labeled sentiment data
- Adjust all weights for sentiment task

================================================================================
WHY THIS CODE WORKS FOR SENTIMENT
================================================================================
Despite no fine-tuning, BERT has some sentiment capability:
- Pre-trained on diverse text (reviews, books, articles)
- Learned sentiment associations from context
- "good" → positive, "bad" → negative
- Can understand negation ("not good" → negative)

For production sentiment analysis:
- Would fine-tune on IMDb or SST-2 (92-94% accuracy)
- This code may make mistakes on subtle examples

================================================================================
LIMITATIONS OF THIS CODE
================================================================================
1. No fine-tuning (using generic BERT)
2. Single sentence inference only
3. No batch processing
4. No training loop
5. May struggle with sarcasm ("Yeah, great movie" said sarcastically)
6. No handling of multi-sentence inputs

================================================================================
TO ACTUALLY TRAIN FOR SENTIMENT
================================================================================
Would need:
1. Labeled dataset (IMDb reviews: 25k train, 25k test)
2. Fine-tuning loop:
    model.compile(optimizer=tf.keras.optimizers.Adam(2e-5),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=['accuracy'])
    model.fit(train_data, validation_data=test_data, epochs=3)

3. Typically achieves 92-94% accuracy on SST-2

================================================================================
BERT VARIANTS
================================================================================
- bert-base-uncased: 110M params (this one)
- bert-large-uncased: 340M params (24 layers)
- DistilBERT: 66M params (faster, slightly less accurate)
- RoBERTa: Optimized BERT (better performance)
- ALBERT: Parameter sharing (smaller memory)

================================================================================
APPLICATIONS
================================================================================
- Sentiment Analysis (reviews, social media)
- Question Answering (SQuAD)
- Named Entity Recognition (NER)
- Text Classification (topics, spam)
- Natural Language Inference (NLI)
"""