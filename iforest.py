from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score
import time
from datos import *

start_time = time.time()

model_if = IsolationForest(contamination=0.0014, random_state=42)
model_if.fit(X_scaled)
pred_if = model_if.predict(X_scaled)

# Isolation Forest devuelve -1 para anomalías, 1 para normales
pred_if = [1 if p == -1 else 0 for p in pred_if]

end_time = time.time()
tiempo_if = end_time - start_time

precision_if = precision_score(y, pred_if)
recall_if = recall_score(y, pred_if)

print("🔍 Isolation Forest")
print("Precisión:", precision_if)
print("Recall:", recall_if)
print("Tiempo:", tiempo_if, "segundos")
