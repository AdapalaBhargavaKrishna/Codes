# ==================== ENCODER-DECODER FOR FRENCH TRANSLATION ====================
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 1. LOAD DATA
!wget -q http://www.manythings.org/anki/fra-eng.zip
!unzip -q fra-eng.zip

lines = open('fra.txt', encoding='utf-8').read().split('\n')
eng_sentences, fr_sentences = [], []
for line in lines[:10000]:
    if '\t' in line:
        eng, fr = line.split('\t')[:2]
        eng_sentences.append(eng.lower())
        fr_sentences.append("startseq " + fr.lower() + " endseq")

# 2. TOKENIZE
eng_token = Tokenizer()
eng_token.fit_on_texts(eng_sentences)
eng_seq = eng_token.texts_to_sequences(eng_sentences)

fr_token = Tokenizer(filters='')
fr_token.fit_on_texts(fr_sentences)
fr_seq = fr_token.texts_to_sequences(fr_sentences)

eng_vocab = len(eng_token.word_index) + 1
fr_vocab = len(fr_token.word_index) + 1

# 3. PADDING
max_eng = max(len(s) for s in eng_seq)
max_fr = max(len(s) for s in fr_seq)
enc_input = pad_sequences(eng_seq, maxlen=max_eng, padding='post')
dec_input = pad_sequences(fr_seq, maxlen=max_fr, padding='post')

# 4. DECODER OUTPUT (one-hot shifted by 1)
dec_output = np.zeros((len(fr_seq), max_fr, fr_vocab), dtype='float32')
for i, seq in enumerate(fr_seq):
    for t in range(1, len(seq)):
        dec_output[i, t-1, seq[t]] = 1.0

# 5. ENCODER
latent_dim = 256
enc_inp = Input(shape=(None,))
enc_emb = Embedding(eng_vocab, latent_dim)(enc_inp)
_, h, c = LSTM(latent_dim, return_state=True)(enc_emb)

# 6. DECODER
dec_inp = Input(shape=(None,))
dec_emb = Embedding(fr_vocab, latent_dim)(dec_inp)
dec_out, _, _ = LSTM(latent_dim, return_sequences=True, return_state=True)(dec_emb, initial_state=[h, c])
dec_out = Dense(fr_vocab, activation='softmax')(dec_out)

# 7. TRAIN
model = Model([enc_inp, dec_inp], dec_out)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit([enc_input, dec_input], dec_output, batch_size=64, epochs=20, verbose=1)

# 8. INFERENCE MODELS
enc_model = Model(enc_inp, [h, c])
dec_h = Input(shape=(latent_dim,))
dec_c = Input(shape=(latent_dim,))
dec_out_inf, h2, c2 = LSTM(latent_dim, return_sequences=True, return_state=True)(dec_emb, initial_state=[dec_h, dec_c])
dec_out_inf = Dense(fr_vocab, activation='softmax')(dec_out_inf)
dec_model = Model([dec_inp, dec_h, dec_c], [dec_out_inf, h2, c2])

# 9. TRANSLATE FUNCTION
reverse_fr = {v: k for k, v in fr_token.word_index.items()}
def translate(sentence):
    seq = pad_sequences(eng_token.texts_to_sequences([sentence]), maxlen=max_eng, padding='post')
    h, c = enc_model.predict(seq, verbose=0)
    target = np.zeros((1, 1))
    target[0,0] = fr_token.word_index['startseq']
    result = ''
    for _ in range(max_fr):
        out, h, c = dec_model.predict([target, h, c], verbose=0)
        idx = np.argmax(out[0,-1,:])
        if idx == 0: continue
        word = reverse_fr.get(idx, '')
        if word == 'endseq': break
        result += ' ' + word
        target[0,0] = idx
    return result.strip()

# 10. TEST
print("Input: how are you")
print("Output:", translate("how are you"))
print("\nInput: I love machine learning")
print("Output:", translate("I love machine learning"))

"""
================================================================================
LANGUAGE TRANSLATION USING ENCODER-DECODER ARCHITECTURE
================================================================================

1. ARCHITECTURE
================================================================================

===================== ENCODER =====================
Input: English sentence (sequence of word indices)
↓
Embedding Layer: (eng_vocab → 256) - converts words to dense vectors
↓
LSTM Layer: 256 units, return_state=True
↓
Output: Final hidden state (h) and cell state (c) → Context vector

===================== DECODER =====================
Input: French sentence (shifted by 1, starts with 'startseq')
↓
Embedding Layer: (fr_vocab → 256)
↓
LSTM Layer: 256 units, initialized with encoder's [h, c]
↓
Dense Layer: (256 → fr_vocab) with softmax
↓
Output: Probability distribution over French words (one-hot encoded)

Data Flow:
"how are you" → Encoder → [h, c] → Decoder → "comment allez vous"

2. NUMBER OF LEARNING PARAMETERS
================================================================================
Component               | Parameters
------------------------|--------------------------------------------------
Embedding (English)     | eng_vocab × 256
Embedding (French)      | fr_vocab × 256
Encoder LSTM (256)      | 4 × [(256×256) + (256×256) + 256] = 525,312
Decoder LSTM (256)      | 4 × [(256×256) + (256×256) + 256] = 525,312
Dense Layer             | 256 × fr_vocab + fr_vocab
------------------------|--------------------------------------------------
TOTAL (approx)          | ~1.5 - 2 million (depends on vocabulary size)

3. LOSS CALCULATION FORMULA
================================================================================
Categorical Cross-Entropy (per time step):

Loss = -1/T Σ Σ y_tc × log(p_tc)
           t=1..T c=1..V

where:
- T = max French sentence length
- V = French vocabulary size
- y_tc = 1 if word c is correct at position t
- p_tc = predicted probability

Teacher Forcing: Uses true previous word, not predicted

4. WEIGHT UPDATE FORMULA
================================================================================
RMSprop Optimizer:

E[g²]_t = ρ × E[g²]_{t-1} + (1-ρ) × g_t²
θ(t+1) = θ(t) - (η / √(E[g²]_t + ε)) × g_t

where:
- g_t = gradient at step t
- ρ = decay rate (default 0.9)
- η = learning rate (default 0.001)
- ε = 10⁻⁷ (stabilization)

5. SAMPLES USED
================================================================================
Training: 10,000 English-French sentence pairs
Validation: None (small dataset)
Testing: User-provided sentences ("how are you", "I love machine learning")

Training batch size: 64
Training epochs: 20

6. LABELED OR UNLABELED DATA?
================================================================================
Labeled Data (Supervised Learning) ✓

- Input: English sentence (source)
- Target: French sentence (translation)
- Parallel corpus provides aligned sentence pairs
- Each English sentence has correct French translation

7. KEY CONCEPTS
================================================================================

ENCODER:
- Reads input sentence word by word
- Final hidden state captures sentence meaning
- No output at each time step (return_state=True)

CONTEXT VECTOR:
- [h, c] from encoder's final LSTM state
- Compressed representation of English sentence
- Passed to decoder as initial state

DECODER:
- Generates French word by word
- Starts with 'startseq' token
- Uses teacher forcing during training
- Stops when 'endseq' is generated

TEACHER FORCING:
- Training: Uses true previous word
- Inference: Uses predicted previous word
- Speeds up training significantly

TOKENIZATION:
- Special tokens: 'startseq', 'endseq'
- Helps model know when to start/stop
- Filters='' preserves all punctuation

8. TRAINING DETAILS
================================================================================
Epochs: 20 (reduced from 200 for speed)
Batch size: 64 (increased from 2 for stability)
Loss: Categorical Cross-Entropy
Optimizer: RMSprop
Metric: Accuracy (per-word prediction accuracy)

9. INFERENCE PROCESS
================================================================================
1. Encode English sentence → [h, c]
2. Initialize decoder with 'startseq'
3. For each step:
   a. Predict next word probability
   b. Choose word with highest probability
   c. Feed back as next input
   d. Stop if 'endseq' or max length reached

10. COMPARISON WITH SIMPLE LSTM
================================================================================
Feature                 | Simple LSTM (prev example) | Encoder-Decoder
------------------------|----------------------------|--------------------
Input                    | Single sequence            | Two sequences
Output                   | Next word                  | Entire translation
Use case                 | Autocomplete               | Translation
Architecture             | Single model               | Two connected models
State transfer           | N/A                        | Encoder → Decoder

11. APPLICATIONS
================================================================================
- Machine Translation (Google Translate)
- Text Summarization
- Question Answering
- Image Captioning (CNN + Decoder)
- Chatbots

12. LIMITATIONS
================================================================================
- Fixed vocabulary (can't handle unknown words)
- LSTM forgets very long sentences
- Simple architecture (no attention mechanism)
- Small dataset (10k pairs) for demonstration
"""