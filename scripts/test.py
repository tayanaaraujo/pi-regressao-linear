import os
import sys
import pickle
import pandas as pd
import numpy as np
from dahuffman import HuffmanCodec

# ------------------------------------
# 1. LEITURA DO CSV
# ------------------------------------
csv_path = sys.argv[1]
df = pd.read_csv(csv_path)

# ------------------------------------
# 2. CARREGAR MODELO
# ------------------------------------
with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("models/huffman.codec", "rb") as f:
    codec = pickle.load(f)

# ------------------------------------
# 3. PRÉ-PROCESSAMENTO
# ------------------------------------
X = df.drop("time", axis=1, errors="ignore").values
X_scaled = scaler.transform(X)

# desfazendo compressão
data_bytes = pickle.dumps(X_scaled)
decoded = codec.decode(codec.encode(data_bytes))
X_scaled = pickle.loads(decoded)

# ------------------------------------
# 4. PREVISÃO
# ------------------------------------
pred = model.predict(X_scaled)

df_out = df.copy()
df_out["prediction"] = pred

os.makedirs("results", exist_ok=True)
df_out.to_csv("results/predicoes.csv", index=False)

# ------------------------------------
# 5. AVALIAÇÃO SE EXISTIR RÓTULO
# ------------------------------------
if "time" in df.columns:
    y_true = df["time"].values
    mse = np.mean((y_true - pred) ** 2)

    with open("results/desempenho_real.txt", "w") as f:
        f.write(f"MSE real: {mse}\n")
