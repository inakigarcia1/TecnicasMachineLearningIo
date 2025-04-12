import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import precision_score, recall_score
import time
from datos import *

start_time = time.time()

input_dim = X_scaled.shape[1]

# Modelo muy simple
autoencoder = models.Sequential([
    layers.Input(shape=(input_dim,)),
    layers.Dense(8, activation='relu'),
    layers.Dense(4, activation='relu'),
    layers.Dense(8, activation='relu'),
    layers.Dense(input_dim)
])

autoencoder.compile(optimizer='adam', loss='mse')

# Solo entrenamos con datos normales
X_normal = X_scaled[y == 0]
autoencoder.fit(X_normal, X_normal, epochs=5, batch_size=128, verbose=0)

# Obtenemos los errores de reconstrucción
reconstructions = autoencoder.predict(X_scaled)
mse = np.mean(np.square(X_scaled - reconstructions), axis=1)

# Umbral manual (puede ajustarse): percentil 99.9
threshold = np.percentile(mse, 99.9)
pred_auto = [1 if e > threshold else 0 for e in mse]

end_time = time.time()
tiempo_auto = end_time - start_time

precision_auto = precision_score(y, pred_auto)
recall_auto = recall_score(y, pred_auto)

print("\n🔍 Autoencoder")
print("Precisión:", precision_auto)
print("Recall:", recall_auto)
print("Tiempo:", tiempo_auto, "segundos")
