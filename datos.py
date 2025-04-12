# Paso 1: Importar las librerías necesarias
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Paso 2: Establecer una semilla para tener resultados reproducibles
np.random.seed(42)

# Paso 3: Generar los 86,400 registros normales
normal_data = {
    "ConexionesPorMinuto": np.random.poisson(lam=100, size=86400),
    "TraficoMB": np.random.normal(loc=50, scale=5, size=86400),
    "CantidadIPs": np.random.poisson(lam=20, size=86400),
    "PuertosAccedidos": np.random.poisson(lam=10, size=86400),
    "RatioFallos": np.random.normal(loc=0.02, scale=0.01, size=86400),
    "Etiqueta": ["normal"] * 86400
}

# Paso 4: Generar los 120 registros anómalos
anomaly_data = {
    "ConexionesPorMinuto": np.random.poisson(lam=1000, size=120),
    "TraficoMB": np.random.normal(loc=200, scale=20, size=120),
    "CantidadIPs": np.random.poisson(lam=200, size=120),
    "PuertosAccedidos": np.random.poisson(lam=100, size=120),
    "RatioFallos": np.random.normal(loc=0.5, scale=0.1, size=120),
    "Etiqueta": ["anomalia"] * 120
}

# Paso 5: Convertirlos en DataFrames
df_normal = pd.DataFrame(normal_data)
df_anomalía = pd.DataFrame(anomaly_data)

# Paso 6: Unir ambos DataFrames en uno solo
df = pd.concat([df_normal, df_anomalía], ignore_index=True)

# Paso 7: Ver los primeros registros
print(df.head())

# Quitar las etiquetas
X = df.drop(columns=["Etiqueta"])
y = df["Etiqueta"].apply(lambda x: 1 if x == "anomalia" else 0)  # 1 = anomalía, 0 = normal

# 2. Escalar (muy importante para modelos como SVM y redes neuronales)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

