import re


class Preprocess:
    def __init__(self, text):
        self.text = text

    def preprocess_text(self):
        # Removing html tags
        preprocess_text = re.compile(r'<[^>]+>').sub('', self.text)

        # Remove punctuations and numbers
        preprocess_text = re.sub('[^a-zA-Z]', ' ', preprocess_text)

        # Single character removal
        preprocess_text = re.sub(r"\s+[a-zA-Z]\s+", ' ', preprocess_text)

        # Removing multiple spaces
        preprocess_text = re.sub(r'\s+', ' ', preprocess_text)

        return preprocess_text
