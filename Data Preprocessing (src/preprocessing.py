import pandas as pd

def load_data(path):
    return pd.read_csv(path)

def preprocess_data(df):
    X = df.drop("final_score", axis=1)
    y = df["final_score"]
    return X, y
