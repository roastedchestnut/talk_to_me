import numpy as np
import tensorflow.keras as keras
import main.preprocess.preprocess as pre
from main.util.aws import build_s3
from joblib import load
import tempfile


class Predict:
    def __init__(self, text, class_name_location="../model/data/class_names.npy", tokenizer_location="../model/data/tokenizer.pickle", model_location="../model/data/model_final.model"):
        self.text = text
        self.class_name_location = class_name_location
        self.tokenizer_location = tokenizer_location
        self.model_location = model_location
        self.s3 = build_s3()

    def predict(self):
        with tempfile.TemporaryFile() as fp:
            self.s3.Bucket('team07-public').download_fileobj(Fileobj=fp, Key='/model/model_final.model')
            fp.seek(0)
            model = load(fp)
            fp.close()

        with tempfile.TemporaryFile() as fp:
            self.s3.Bucket('team07-public').download_fileobj(Fileobj=fp, Key='/model/class_names.npy')
            fp.seek(0)
            class_name = load(fp)
            fp.close()

        with tempfile.TemporaryFile() as fp:
            self.s3.Bucket('team07-public').download_fileobj(Fileobj=fp, Key='/model/tokenizer.tokenizer')
            fp.seek(0)
            Tokenizer = load(fp)
            fp.close()

        maxlen = 100

        pre_text = pre.Preprocess(self.text).preprocess_text()
        pre_text = [pre_text]
        pre_text = Tokenizer.texts_to_sequences(pre_text)
        pre_text = np.array(pre_text)
        pad_text = keras.preprocessing.sequence.pad_sequences(pre_text, padding='post', maxlen=maxlen)

        result = model.predict(pad_text)

        return class_name[np.argmax(result)]

