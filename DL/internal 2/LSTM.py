import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

corpus = [
    "I love machine learning",
    "I love deep learning",
    "I love natural language processing",
    "I enjoy coding in Python",
    "I enjoy solving problems"
]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1
print(total_words)

input_sequences = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    print(token_list)
    for i in range(1, len(token_list)):
        n_gram_seq = token_list[:i+1]
        input_sequences.append(n_gram_seq)
print(input_sequences)

from tensorflow.keras.preprocessing.sequence import pad_sequences
max_seq_len = max([len(x) for x in input_sequences])
input_sequences = pad_sequences(input_sequences, maxlen=max_seq_len, padding='pre')
print(input_sequences)
X = input_sequences[:, :-1]
y = input_sequences[:, -1]
y = to_categorical(y, num_classes=total_words)

model = Sequential()
model.add(Embedding(input_dim=total_words, output_dim=10, input_length=max_seq_len-1))
model.add(LSTM(100))
model.add(Dense(total_words, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=200, verbose=1)

def predict_next_word(seed_text):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_seq_len-1, padding='pre')
    print(token_list)
    predicted = model.predict(token_list, verbose=0)
    print(predicted)
    predicted_word_index = np.argmax(predicted)
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word

seed = "I love machine"
print(f"Next word prediction: {predict_next_word(seed)}")
print(tokenizer.word_index)

"""
================================================================================
LSTM TEXT GENERATION ANALYSIS (NEXT WORD PREDICTION)
================================================================================

1. ARCHITECTURE
================================================================================
Input: Sequence of word indices (length = max_seq_len - 1)
↓
Embedding Layer:
- input_dim = total_words (15 words in vocabulary)
- output_dim = 10 (dense vector representation)
- input_length = max_seq_len - 1 (4 words)
- Output shape: (batch, 4, 10)
↓
LSTM Layer:
- 100 units (memory cells)
- Returns last output only (default return_sequences=False)
- Output shape: (batch, 100)
↓
Dense Layer (Softmax):
- total_words neurons (15)
- Activation: softmax
- Output shape: (batch, 15)
↓
Output: Probability distribution over next word

Full vocabulary (15 words):
1. i, 2. love, 3. machine, 4. learning, 5. deep, 6. natural, 
7. language, 8. processing, 9. enjoy, 10. coding, 11. in, 
12. python, 13. solving, 14. problems

2. NUMBER OF LEARNING PARAMETERS (WEIGHTS & BIAS)
================================================================================
Layer                    | Weights/Params                     | Total
-------------------------|------------------------------------|---------------
Embedding (15×10)        | 15×10 = 150                        | 150
LSTM (10→100)           | 4 × [(100×10) + (100×100) + 100]   | 44,400
                         | = 4 × [1000 + 10000 + 100]          |
                         | = 4 × 11,100 = 44,400              |
Dense (100→15)          | 100×15 = 1,500 + 15 (bias)          | 1,515
TOTAL                   |                                       | 46,065

LSTM Calculation Detail:
For LSTM with input_dim=10, units=100:
- Input weights: 100 × 10 = 1,000
- Recurrent weights: 100 × 100 = 10,000
- Bias: 100
- Total per gate: 11,100
- 4 gates (forget, input, cell, output) = 44,400

3. LOSS CALCULATION FORMULA
================================================================================
Categorical Cross-Entropy (Multi-class classification over vocabulary):

For single prediction: Loss = - Σ y_c × log(p_c)
                              c=1..V

For batch: Loss = -1/N Σ Σ y_ic × log(p_ic)
                      i=1..N c=1..V

where:
- N = batch size (default 32)
- V = vocabulary size (15 words)
- y_ic = 1 if word c is the true next word, else 0 (one-hot encoded)
- p_ic = predicted probability of word c being next

Example: If true next word is "learning" and model predicts 0.7 for "learning":
Loss = -log(0.7) = 0.3567

4. WEIGHT UPDATE FORMULA
================================================================================
Adam Optimizer:

θ(t+1) = θ(t) - (η / (√v̂(t) + ε)) × m̂(t)

where:
- θ(t) = weights/biases at time step t
- η = learning rate (default 0.001)
- m̂(t) = bias-corrected first moment (mean of gradients)
- v̂(t) = bias-corrected second moment (uncentered variance)
- ε = 10⁻⁷

5. SAMPLES USED FOR TRAINING, VALIDATION, TESTING
================================================================================
Total corpus: 5 sentences
Input sequences generated: 13 n-grams

Breakdown of sequences:
- "I love" (from sentence 1)
- "I love machine" (from sentence 1)
- "I love machine learning" (from sentence 1)
- "I love deep" (from sentence 2)
- "I love deep learning" (from sentence 2)
- "I love natural" (from sentence 3)
- "I love natural language" (from sentence 3)
- "I love natural language processing" (from sentence 3)
- "I enjoy coding" (from sentence 4)
- "I enjoy coding in" (from sentence 4)
- "I enjoy coding in Python" (from sentence 4)
- "I enjoy solving" (from sentence 5)
- "I enjoy solving problems" (from sentence 5)

Training: All 13 sequences (no separate validation)
Testing: Single seed text "I love machine" (user-provided)
Validation: None (small dataset)

EPOCHS: 200 (many passes through same 13 sequences)

6. LABELED OR UNLABELED DATA?
================================================================================
Labeled Data (Self-Supervised) ✓

Process:
- Input sequences (X): First n-1 words from each n-gram
- Labels (y): Last word from each n-gram
- Labels automatically extracted from text (no manual labeling)
- This is SELF-SUPERVISED learning (predicts next word from context)

Example:
Sentence: "I love machine learning"
Sequences generated:
- X: ["I"] → y: "love"
- X: ["I", "love"] → y: "machine"
- X: ["I", "love", "machine"] → y: "learning"

================================================================================
DATA PREPROCESSING STEPS
================================================================================

1. TOKENIZATION:
   - Converts words to integers
   - Tokenizer.fit_on_texts() learns vocabulary
   - word_index: {'i':1, 'love':2, 'machine':3, 'learning':4, ...}

2. SEQUENCE CREATION:
   - Creates n-grams from each sentence
   - Example: "I love learning" → [1,2,4], [1,2], [1,2,4]

3. PADDING:
   - All sequences padded to same length (max_seq_len=5)
   - padding='pre' adds zeros at beginning
   - Example: [1,2,3] → [0,0,1,2,3]

4. SPLIT:
   - X: All words except last → input features
   - y: Last word → target label

5. ONE-HOT ENCODING:
   - y converted to one-hot vector of length 15
   - Example: label 4 → [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0]

================================================================================
EMBEDDING LAYER EXPLAINED
================================================================================
Purpose: Convert word indices to dense vectors
- input_dim=15: Vocabulary size
- output_dim=10: Each word represented by 10 numbers
- input_length=4: Each sequence has 4 words

Why Embedding?
- Captures semantic relationships between words
- Reduces dimensionality (one-hot would be 15 dims)
- Learns word similarities during training

Example vectors learned:
- "love" and "enjoy" become similar (both positive verbs)
- "machine" and "deep" become similar (both modify learning)

================================================================================
LSTM LAYER EXPLAINED
================================================================================
LSTM with 100 units (memory cells):

Gate functions:
1. Forget Gate: What to forget from previous state
2. Input Gate: What new information to store
3. Output Gate: What to output based on cell state

Why LSTM for text?
- Handles long-range dependencies
- Remembers context across sequences
- Prevents vanishing gradient problem

For this task:
- Previous words influence next word prediction
- "I love machine..." suggests "learning"
- "I enjoy..." suggests different patterns

================================================================================
PREDICTION PROCESS
================================================================================
Seed text: "I love machine"

Step 1: Tokenize → [1, 2, 3]
Step 2: Pad to length 4 → [0, 1, 2, 3]
Step 3: Model predicts probability distribution (15 numbers)
Step 4: argmax picks highest probability word index
Step 5: Map index back to word

Expected output:
- High probability for "learning"
- Lower for other words

================================================================================
TRAINING DETAILS
================================================================================
- Epochs: 200 (small dataset needs many epochs)
- Loss: Categorical Cross-Entropy
- Optimizer: Adam
- Batch size: Default (32 - but only 13 samples total)
- Verbose: 1 (shows progress bars)

Note: Very small dataset (5 sentences, 13 sequences)
- Expect overfitting
- model essentially memorizes patterns
- Works for demonstration but not real-world scale

================================================================================
CHALLENGES WITH THIS APPROACH
================================================================================
1. Small vocabulary (only 15 words)
2. No handling of unknown words
3. Cannot generate words not seen in training
4. Limited to short sequences (max 5 words)
5. No character-level or subword tokenization

================================================================================
APPLICATIONS
================================================================================
1. Autocomplete (Google Search, Gmail Smart Compose)
2. Chatbots and Conversational AI
3. Machine Translation
4. Speech Recognition
5. Text Generation (stories, poetry)
6. Code Completion

================================================================================
NEXT STEPS FOR IMPROVEMENT
================================================================================
1. Larger corpus (thousands of sentences)
2. Character-level tokenization for rare words
3. Bidirectional LSTM for context from both sides
4. Attention mechanism for important words
5. Transformer architecture (GPT, BERT)
6. Beam search instead of greedy prediction
"""