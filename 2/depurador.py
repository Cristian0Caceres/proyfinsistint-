
import os
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.metrics import pairwise_distances_argmin
from sklearn.datasets import make_blobs
import sklearn

# ---------------------------------------------------------
# CONFIGURACIÓN (ajusta si quieres otra semilla o quantile)
RANDOM_STATE = 42
ESTIMATE_BANDWIDTH = True   # Si False, usa el bandwidth por defecto de MeanShift
BANDWIDTH_QUANTILE = 0.2    # si ESTIMATE_BANDWIDTH True -> quantile usado en estimate_bandwidth
# ---------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Posibles nombres de archivo
X_candidates = ['X.npy']
y_candidates = ['y.npy', '_.npy', 'labels.npy']

X_path = None
y_path = None

# Buscar archivos existentes
for name in X_candidates:
    p = os.path.join(BASE_DIR, name)
    if os.path.exists(p):
        X_path = p
        break

for name in y_candidates:
    p = os.path.join(BASE_DIR, name)
    if os.path.exists(p):
        y_path = p
        break

# Si faltan, crear datos sintéticos y guardarlos
if X_path is None or y_path is None:
    print("--- Archivos .npy no encontrados en:", BASE_DIR)
    print("Creando datos sintéticos de ejemplo (se guardarán como X.npy y y.npy)...")
    centers = [[-7, -6], [1.5, -6.5], [7.9, 0.5], [5.5, 10]]
    X_syn, y_syn = make_blobs(n_samples=200, centers=centers, cluster_std=1.0, random_state=RANDOM_STATE)
    X_path = os.path.join(BASE_DIR, 'X.npy')
    y_path = os.path.join(BASE_DIR, 'y.npy')
    np.save(X_path, X_syn)
    np.save(y_path, y_syn)
    print("Archivos creados:", X_path, y_path)
else:
    print("Archivos encontrados:")
    print(" X ->", X_path)
    print(" y ->", y_path)

# Cargar los datos
try:
    X = np.load(X_path)
except Exception as e:
    print("Error cargando X:", e)
    sys.exit(1)

try:
    y = np.load(y_path)
except Exception as e:
    print("Error cargando y:", e)
    # si no hay etiquetas originales, creamos etiquetas dummy para plotting
    y = np.zeros(len(X), dtype=int)
    print("Se generaron etiquetas por defecto (zeros).")

# Mensajes de versión
print("=" * 60)
print("Entorno:")
print(" Python executable:", sys.executable)
print(" numpy:", np.__version__)
print(" matplotlib:", mpl.__version__)
print(" scikit-learn:", sklearn.__version__)
print("=" * 60)

# Estimar bandwidth si activado
bandwidth = None
if ESTIMATE_BANDWIDTH:
    try:
        print("Estimando bandwidth con quantile =", BANDWIDTH_QUANTILE)
        bandwidth = estimate_bandwidth(X, quantile=BANDWIDTH_QUANTILE, random_state=RANDOM_STATE)
        print("Bandwidth estimado:", bandwidth)
    except Exception as e:
        print("No se pudo estimar bandwidth:", e)
        bandwidth = None

# Aplicar MeanShift
if bandwidth is not None:
    meanshift = MeanShift(bandwidth=bandwidth)
else:
    meanshift = MeanShift()

meanshift.fit(X)
labels = meanshift.labels_
centroides = meanshift.cluster_centers_

# Guardar CSV con X y labels para entregar
csv_path = os.path.join(BASE_DIR, 'X_labels.csv')
try:
    data_for_csv = np.hstack([X, labels.reshape(-1, 1)])
    header = "x,y,label"
    np.savetxt(csv_path, data_for_csv, delimiter=",", header=header, comments='', fmt='%.6f')
    print("CSV con X+labels guardado en:", csv_path)
except Exception as e:
    print("No se pudo guardar CSV:", e)

# ---------------------------
# 2.1 - Figura A: gráfico simple
fig_a = os.path.join(BASE_DIR, 'meanshift_figura_a.png')
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], alpha=0.6, s=50)
plt.title('Figura A - Datos de Entrenamiento')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True, alpha=0.3)
plt.savefig(fig_a, dpi=300, bbox_inches='tight')
plt.close()
print("Guardado:", fig_a)

# 2.2 - Figura B: colores según etiquetas reales (si hay)
fig_b = os.path.join(BASE_DIR, 'meanshift_figura_b.png')
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.6, s=50)
plt.title('Figura B - Blobs Resaltados por Colores')
plt.xlabel('X')
plt.ylabel('Y')
plt.colorbar(label='Cluster (etiquetas reales)')
plt.grid(True, alpha=0.3)
plt.savefig(fig_b, dpi=300, bbox_inches='tight')
plt.close()
print("Guardado:", fig_b)

# 2.3 - Figura C: clusters y centroides
fig_c = os.path.join(BASE_DIR, 'meanshift_figura_c.png')
plt.figure(figsize=(10, 7))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6, s=50)
# Centroides: estrella grande con borde blanco
plt.scatter(centroides[:, 0], centroides[:, 1],
            marker='*', c='black', s=500, edgecolors='white', linewidths=2,
            label='Centroides')
plt.title('Figura C - Clusters y Centroides con MeanShift')
plt.xlabel('X')
plt.ylabel('Y')
plt.colorbar(label='Cluster (etiquetado por MeanShift)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(fig_c, dpi=300, bbox_inches='tight')
plt.close()
print("Guardado:", fig_c)

# 2.4 - Etiquetas y conteo
etiquetas_unicas = np.unique(labels)
conteos = {int(e): int(np.sum(labels == e)) for e in etiquetas_unicas}

# 2.5 - Predicción de datos test (usando distancia al centroide)
test_data = np.array([[-7, -6], [1.5, -6.5], [7.9, 0.5], [5.5, 10]])
predicciones = pairwise_distances_argmin(test_data, centroides)

# 2.6 - Visualizar predicciones junto a centroides y entrenamiento
fig_pred = os.path.join(BASE_DIR, 'meanshift_predicciones.png')
plt.figure(figsize=(10, 7))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.4, s=50, label='Datos entrenamiento')
plt.scatter(centroides[:, 0], centroides[:, 1],
            marker='*', c='black', s=500, edgecolors='white', linewidths=2,
            label='Centroides')
plt.scatter(test_data[:, 0], test_data[:, 1],
            marker='X', c=predicciones, cmap='viridis', s=200,
            edgecolors='red', linewidths=2, label='Datos test')
plt.title('MeanShift - Predicción de Datos Test')
plt.xlabel('X')
plt.ylabel('Y')
plt.colorbar(label='Cluster')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(fig_pred, dpi=300, bbox_inches='tight')
plt.close()
print("Guardado:", fig_pred)

# Guardar reporte de texto con resultados resumidos
report_path = os.path.join(BASE_DIR, 'meanshift_report.txt')
with open(report_path, 'w') as f:
    f.write("MEANSHIFT - REPORT\n")
    f.write("=" * 20 + "\n")
    f.write(f"Python executable: {sys.executable}\n")
    f.write(f"numpy: {np.__version__}\n")
    f.write(f"matplotlib: {mpl.__version__}\n")
    f.write(f"scikit-learn: {sklearn.__version__}\n\n")
    f.write(f"Número de clusters encontrados: {len(centroides)}\n")
    f.write("Centroides:\n")
    for c in centroides:
        f.write(f"  {c.tolist()}\n")
    f.write("\nEtiquetas encontradas: " + np.array2string(etiquetas_unicas) + "\n")
    f.write(f"Cantidad total de datos: {len(X)}\n")
    f.write("Cantidad de datos por cluster:\n")
    for k, v in conteos.items():
        f.write(f"  Cluster {k}: {v} datos\n")
    f.write("\nPredicciones para puntos test (en orden):\n")
    for i, p in enumerate(test_data):
        f.write(f"  Punto {i+1}: {p.tolist()} -> Cluster {int(predicciones[i])}\n")
print("Reporte guardado en:", report_path)

print("\n" + "=" * 80)
print("ANÁLISIS COMPLETADO")
print("Gráficos guardados en:", BASE_DIR)
print("CSV guardado en:", csv_path)
print("Reporte guardado en:", report_path)
print("=" * 60)
