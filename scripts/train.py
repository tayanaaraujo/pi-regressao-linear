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

y = df["time"].values
X = df.drop("time", axis=1).values

# ------------------------------------
# 2. NORMALIZAÇÃO
# ------------------------------------
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# ------------------------------------
# 3. VALIDAÇÃO CRUZADA
# ------------------------------------
tscv = TimeSeriesSplit(n_splits=5)
model = LinearRegression()

scores = cross_val_score(model, X_scaled, y, cv=tscv,
                         scoring="neg_mean_squared_error")
expected_performance = np.mean(scores)

# ------------------------------------
# 4. TREINO FINAL
# ------------------------------------
model.fit(X_scaled, y)

# ------------------------------------
# 5. HUFFMAN – CODIFICAR X_scaled
# ------------------------------------
data_bytes = pickle.dumps(X_scaled)

# Frequências para gerar a árvore Huffman
freqs = Counter(data_bytes)

codec = HuffmanCodec.from_frequencies(freqs)

# Codificar
encoded = codec.encode(data_bytes)

# SALVAR AS PARTES: codec + dados codificados
os.makedirs("models", exist_ok=True)

# ---- SALVAR ARRAY CODIFICADO ----
with open("models/encoded_scaler.bin", "wb") as f:
    f.write(bytes(encoded))

# ---- SALVAR CODEC ----
with open("models/huffman_codec.pkl", "wb") as f:
    pickle.dump(codec, f)

# ------------------------------------
# 6. SALVAR MODELO E OUTROS ARTEFATOS
# ------------------------------------
with open("models/model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("models/expected_performance.txt", "w") as f:
    f.write(f"Desempenho esperado (MSE CV): {expected_performance}\n")

print("Treino concluído. Artefatos salvos na pasta models/")
