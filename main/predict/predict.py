import numpy as np
import pickle
import tensorflow.keras as keras
import main.preprocess.preprocess as pre


class Predict:
    def __init__(self, text):
        self.text = text

    def predict(self):
        class_name = np.load("../model/data/class_names.npy")

        with open("../model/data/tokenizer.pickle", 'rb') as handle:
            Tokenizer = pickle.load(handle)

        model = keras.models.load_model("../model/data/model_final.model")

        maxlen = 100

        pre_text = pre.Preprocess(self.text).preprocess_text()
        pre_text = [pre_text]
        pre_text = Tokenizer.texts_to_sequences(pre_text)
        pre_text = np.array(pre_text)
        pad_text = keras.preprocessing.sequence.pad_sequences(pre_text, padding='post', maxlen=maxlen)

        result = model.predict(pad_text)

        return class_name[np.argmax(result)]
