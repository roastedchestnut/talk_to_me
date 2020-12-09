import sys
sys.path.append('../../')
print(sys.path)

import pandas as pd
from flask import Flask, request, jsonify
from main.predict.predict import Predict

def main():
    app = Flask(__name__)

    @app.route('/predict', methods=['POST'])
    def predict():
        data = request.get_json(force=True)
        mapped = {k: [v] for k, v in data.items()}
        df = pd.DataFrame.from_dict(mapped)
        text = df["text"][0]

        predictor = Predict(text)
        result = predictor.predict()

        return jsonify(
            result
        )

    app.run(host='0.0.0.0', port=5000, debug=False)


if __name__ == '__main__':
    main()
