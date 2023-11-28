from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense
import numpy as np
'''
tensorflow version 2.14.0
'''
# Data
sentences = ["i love you", "he loves me", "she likes baseball",
             "i hate you", "sorry for that", "this is awful"]
labels = np.array([1., 1., 1., 0., 0., 0.])

# Data preprocessing
word_list = []
for i in sentences:
    word_list.append(i.split(" "))

embedding_size = len(sentences)
sentences_encode = [one_hot(d, embedding_size) for d in sentences]

x_train = pad_sequences(sentences_encode, maxlen=6)

# Model
model = Sequential([
    Embedding(input_dim=embedding_size, output_dim=32, input_length=6),
    Conv1D(8, (3), activation='relu'),
    MaxPooling1D(),
    Flatten(),
    Dense(1, activation='softmax')
])

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['acc'])

model.fit(x_train, labels, epochs=100)
