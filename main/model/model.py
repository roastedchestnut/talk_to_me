import tempfile
import pickle
import numpy as np
import pandas as pd
import main.preprocess.preprocess as pre
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Embedding
from tensorflow.keras.metrics import Precision, Recall
from sklearn.model_selection import train_test_split



class Model:
    def __init__(self, train_location, test_location, sep):
        self.train_location = train_location
        self.test_location = test_location
        self.sep = sep

    def train(self):
        dataset = pd.read_csv("../data/iseardataset.csv")

        X = []
        sentences = list(dataset['text'])
        for sen in sentences:
            X.append(pre.Preprocess(sen).preprocess_text())

        y = dataset['label']
        encoder = LabelBinarizer()
        y = encoder.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

        tokenizer = Tokenizer(num_words=10000)
        tokenizer.fit_on_texts(X_train)

        X_train = tokenizer.texts_to_sequences(X_train)
        X_test = tokenizer.texts_to_sequences(X_test)

        maxlen = 100

        X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
        X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

        embedding_dict = dict()
        with open('../data/glove.6B.100d.txt', encoding='UTF-8') as glove_file:
            for line in glove_file:
                records = line.split()
                word = records[0]
                vector_dimension = np.asarray(records[1:], dtype='float32')
                embedding_dict[word] = vector_dimension

        vocab_size = len(tokenizer.word_index) + 1
        embedding_matrix = np.zeros((vocab_size, 100))

        for word, index in tokenizer.word_index.items():
            embedding_vector = embedding_dict.get(word)
            if embedding_vector is not None:
                embedding_matrix[index] = embedding_vector

        model = Sequential([
            Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=maxlen, trainable=False),
            Bidirectional(LSTM(50, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)),
            Bidirectional(LSTM(54, dropout=0.3, recurrent_dropout=0.3, return_sequences=True)),
            Bidirectional(LSTM(60, dropout=0.3, recurrent_dropout=0.3)),
            Dense(64, activation="relu"),
            Dense(7, activation="softmax")])

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', Precision(), Recall()])

        history = model.fit(X_train, y_train, batch_size=128, epochs=100, verbose=1, validation_split=0.2)

        model.save("data/model_final.model")
        np.save("data/class_names.npy", encoder.classes_)

        with open('data/tokenizer.pickle', 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
