import joblib
import numpy as np

class FraudPredictor:

    def __init__(self):
        self.model = joblib.load("models/model.pkl")
        self.scaler = joblib.load("models/scaler.pkl")

    def predict(self, features):
        data = np.array(features).reshape(1, -1)
        data = self.scaler.transform(data)

        pred = self.model.predict(data)[0]
        prob = self.model.predict_proba(data)[0][1]

        return int(pred), float(prob)