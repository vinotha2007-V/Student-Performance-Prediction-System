import pickle
import numpy as np

def predict_performance(features):
    with open("../models/model.pkl", "rb") as f:
        model = pickle.load(f)

    prediction = model.predict([features])
    return prediction[0]

# Example
if __name__ == "__main__":
    sample = [6, 80, 75, 7]
    print("Predicted Score:", predict_performance(sample))
