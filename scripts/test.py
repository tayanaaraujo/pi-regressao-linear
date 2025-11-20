import os
import sys
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from dahuffman import HuffmanCodec

# ------------------------------------
# 1. LEITURA DOS ARTEFATOS
# ------------------------------------
model = pickle.load(open("models/model.pkl", "rb"))
codec = pickle.load(open("models/huffman_codec.pkl", "rb"))

with open("models/encoded_scaler.bin", "rb") as f:
    encoded = f.read()

# decodifica via Huffman
decoded = codec.decode(list(encoded))
decoded = bytes(decoded)
scaler = pickle.loads(decoded)

# ------------------------------------
# 2. LEITURA DO CSV DE TESTE
# ------------------------------------
csv_path = sys.argv[1]
df = pd.read_csv(csv_path)

has_labels = "time" in df.columns

if has_labels:
    y_test = df["time"].values
    X_test = df.drop("time", axis=1).values
else:
    X_test = df.values

# ------------------------------------
# 3. NORMALIZAÇÃO
# ------------------------------------
X_test_scaled = scaler.transform(X_test)

# ------------------------------------
# 4. PREVISÃO
# ------------------------------------
pred = model.predict(X_test_scaled)

# ------------------------------------
# 5. EXPORTAÇÃO
# ------------------------------------
os.makedirs("results", exist_ok=True)

if has_labels:
    # gera arquivo com verdade real
    out = pd.DataFrame({
        "time": y_test,
        "prediction": pred
    })
    out.to_csv("results/predicoes.csv", index=False)

    mse = mean_squared_error(y_test, pred)
    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)

    with open("results/desempenho_real.txt", "w") as f:
        f.write(f"MSE: {mse}\nMAE: {mae}\nR2: {r2}\n")

else:
    # só gera as previsões
    pd.DataFrame({"prediction": pred}).to_csv("results/predicoes.csv", index=False)
