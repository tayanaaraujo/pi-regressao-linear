
import pandas as pd
import numpy as np
import pickle
import os
import sys
from dahuffman import HuffmanCodec

os.makedirs("results", exist_ok=True)

csv_path = sys.argv[1]
df = pd.read_csv(csv_path)

with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("models/huffman.codec", "rb") as f:
    codec = pickle.load(f)

X = df.drop("time", axis=1, errors="ignore")
X_scaled = scaler.transform(X)

X_bytes = pickle.dumps(X_scaled)
decoded = codec.decode(codec.encode(X_bytes))
X_scaled = pickle.loads(decoded)

pred = model.predict(X_scaled)

df_out = df.copy()
df_out["prediction"] = pred

df_out.to_csv("results/predicoes.csv", index=False)

if "time" in df.columns:
    y_true = df["time"].values
    mse = np.mean((y_true - pred) ** 2)
    with open("results/desempenho.txt", "w") as f:
        f.write(f"MSE: {mse}\n")
