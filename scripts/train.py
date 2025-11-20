import os
import sys
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from dahuffman import HuffmanCodec
from collections import Counter

# ------------------------------------
# 1. LEITURA DO CSV
# ------------------------------------
csv_path = sys.argv[1]
df = pd.read_csv(csv_path)

# O valor alvo é "time"
y = df["time"].values
X = df.drop("time", axis=1).values

# ------------------------------------
# 2. NORMALIZAÇÃO MIN-MAX
# ------------------------------------
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# ------------------------------------
# 3. VALIDAÇÃO CRUZADA PARA SÉRIE TEMPORAL
# ------------------------------------
tscv = TimeSeriesSplit(n_splits=5)
model = LinearRegression()

scores = cross_val_score(model, X_scaled, y, cv=tscv, scoring="neg_mean_squared_error")
expected_performance = np.mean(scores)

# ------------------------------------
# 4. TREINO FINAL DO MODELO
# ------------------------------------
model.fit(X_scaled, y)

# ------------------------------------
# 5. COMPACTAÇÃO HUFFMAN DO SCALER
# ------------------------------------
scaler_bytes = pickle.dumps(scaler)
freqs = Counter(scaler_bytes)
codec = HuffmanCodec.from_frequencies(freqs)

encoded = codec.encode(scaler_bytes)  # lista de ints
encoded = bytes(encoded)  # converter para bytes antes de salvar

# ------------------------------------
# 6. SALVAR ARTEFATOS
# ------------------------------------
os.makedirs("models", exist_ok=True)

with open("models/model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("models/huffman_codec.pkl", "wb") as f:
    pickle.dump(codec, f)

with open("models/encoded_scaler.bin", "wb") as f:
    f.write(encoded)

with open("models/expected_performance.txt", "w") as f:
    f.write(f"Desempenho esperado (MSE CV): {expected_performance}\n")

