# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 17:56:33 2023

@author: knmoa
"""

import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import joblib
import pickle 

df = pd.read_excel('data (8).xlsx')
texts = df.iloc[:,0].tolist()
labels = df.iloc[:,1].tolist()

texts_train, texts_val, labels_train, labels_val = train_test_split(texts, labels, test_size=0.2, random_state=42)

tokenizer_train = Tokenizer()
tokenizer_train.fit_on_texts(texts_train)
word_index_train = tokenizer_train.word_index
total_words_train = len(word_index_train) + 1

sequences_train = tokenizer_train.texts_to_sequences(texts_train)
sequences_val = tokenizer_train.texts_to_sequences(texts_val)

max_sequence_length_train = max([len(seq) for seq in sequences_train])
max_sequence_length_val = max([len(seq) for seq in sequences_val])

sequences_train = pad_sequences(sequences_train, maxlen=max_sequence_length_train, padding='post')
sequences_val = pad_sequences(sequences_val, maxlen=max_sequence_length_train, padding='post')  # pad to the same length as training data

labels_train = tf.constant(labels_train)
labels_val = tf.constant(labels_val)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(total_words_train, 16, input_length=max_sequence_length_train),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(3, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00045), loss='binary_crossentropy', metrics=['accuracy'])

epochs = 60

es = EarlyStopping(monitor='val_loss', mode='min', baseline=0.2)

model.fit(sequences_train, labels_train, epochs=epochs, batch_size=1, 
                    validation_data=(sequences_val, labels_val))


model.save('my_model.h5')
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer_train, handle, protocol=pickle.HIGHEST_PROTOCOL)

