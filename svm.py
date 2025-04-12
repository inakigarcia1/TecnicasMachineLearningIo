from sklearn.svm import OneClassSVM
from sklearn.metrics import precision_score, recall_score
import time
from datos import *

start_time = time.time()

model_svm = OneClassSVM(nu=0.0014, kernel="rbf", gamma="auto")
model_svm.fit(X_scaled)
pred_svm = model_svm.predict(X_scaled)

# SVM también devuelve -1 para anomalía
pred_svm = [1 if p == -1 else 0 for p in pred_svm]

end_time = time.time()
tiempo_svm = end_time - start_time

precision_svm = precision_score(y, pred_svm)
recall_svm = recall_score(y, pred_svm)

print("\n🔍 One-Class SVM")
print("Precisión:", precision_svm)
print("Recall:", recall_svm)
print("Tiempo:", tiempo_svm, "segundos")
