import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
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

input_sequences = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

max_seq_len = max(len(x) for x in input_sequences)
input_sequences = pad_sequences(
    input_sequences,
    maxlen=max_seq_len,
    padding='pre'
)

X = input_sequences[:, :-1]
y = input_sequences[:, -1]
y = to_categorical(y, num_classes=total_words)

model = Sequential()
model.add(Embedding(input_dim=total_words, output_dim=10,input_length=max_seq_len-1))
model.add(LSTM(100))
model.add(Dense(total_words, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=200, verbose=1)

def predict_next_word(seed_text):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences(
        [token_list],
        maxlen=max_seq_len-1,
        padding='pre'
    )

    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted)
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word

seed_text = "I love machine"
next_word = predict_next_word(seed_text)
print("\nSeed Text :", seed_text)
print("Predicted Next Word :", next_word)