import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data():
    df = pd.read_csv("data/creditcard.csv")

    X = df.drop("Class", axis=1)
    y = df["Class"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("Dataset Ready:", X_scaled.shape)

    return X_scaled, y, scaler