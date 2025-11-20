import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import json

print("🔄 Carregando dados...")
df = pd.read_csv("data/PI_train.csv")

# Separando entradas e rótulo
X = df.drop("time", axis=1)
y = df["time"]

print("🏋️ Treinando modelo de Regressão Linear...")
model = LinearRegression()
model.fit(X, y)

pred = model.predict(X)

# Métricas
mse = mean_squared_error(y, pred)
r2 = r2_score(y, pred)

print("📦 Salvando modelo treinado...")
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("📝 Salvando desempenho esperado...")
performance = {"MSE": mse, "R2": r2}
with open("performance.json", "w") as f:
    json.dump(performance, f, indent=4)

print("✅ Treinamento concluído!")
