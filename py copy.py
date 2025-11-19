import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Cargar los datos (asegúrate de tener los archivos A.npy y _.npy en tu directorio)
# Si no tienes los archivos, genera datos sintéticos
try:
    X = np.load('A.npy')
    y = np.load('_.npy')
except:
    # Datos sintéticos de ejemplo si no existen los archivos
    from sklearn.datasets import make_blobs
    X, y = make_blobs(n_samples=300, centers=4, n_features=2, 
                      cluster_std=1.5, random_state=42)
    print("Usando datos sintéticos de ejemplo")

# 1. Gráfico Figura A - Datos iniciales sin colores
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.scatter(X[:, 0], X[:, 1], c='gray', s=50, alpha=0.6)
plt.title('Figura A: Datos de Entrenamiento')
plt.xlabel('X1')
plt.ylabel('X2')
plt.grid(True, alpha=0.3)

# 2. Gráfico Figura B - Blobs resaltados con colores (etiquetas originales si existen)
plt.subplot(1, 3, 2)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=50, alpha=0.6)
plt.title('Figura B: Blobs Coloreados')
plt.xlabel('X1')
plt.ylabel('X2')
plt.colorbar(label='Clase')
plt.grid(True, alpha=0.3)

# 3. KMeans - Clusters y Centroides
# Determinar el número de clusters (basado en las etiquetas únicas)
n_clusters = len(np.unique(y))
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
kmeans.fit(X)

# Obtener predicciones y centroides
labels = kmeans.labels_
centroides = kmeans.cluster_centers_

# Gráfico Figura C - Con centroides marcados
plt.subplot(1, 3, 3)
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50, alpha=0.6)
plt.scatter(centroides[:, 0], centroides[:, 1], 
           marker='*', s=500, c='black', edgecolors='yellow', 
           linewidths=2, label='Centroides')
plt.title('Figura C: Clusters con Centroides')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()
plt.colorbar(label='Cluster')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('kmeans_figuras.png', dpi=150, bbox_inches='tight')
plt.show()

# 4. Etiquetas y cantidad de datos
print("\n" + "="*60)
print("PREGUNTA 4: Información de las Etiquetas")
print("="*60)
etiquetas_unicas = np.unique(labels)
print(f"Etiquetas de los clusters: {etiquetas_unicas}")
print(f"Número total de datos: {len(X)}")
print(f"Número de clusters: {len(etiquetas_unicas)}")

# Contar datos por cluster
for i in etiquetas_unicas:
    count = np.sum(labels == i)
    print(f"  Cluster {i}: {count} datos")

# 5 y 6. Predicción de puntos de prueba
print("\n" + "="*60)
print("PREGUNTA 5 y 6: Predicción de Puntos de Prueba")
print("="*60)

test_data = np.array([[2, 5], [3.2, 6.5], [7, 2.5], [9, 3.2], [9, -6], [11, -8]])
predicciones = kmeans.predict(test_data)

print("\nPuntos de prueba y sus predicciones:")
for i, (punto, clase) in enumerate(zip(test_data, predicciones)):
    print(f"  Punto {i+1}: {punto} -> Clase {clase}")

# Visualización de puntos de prueba
plt.figure(figsize=(10, 7))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50, alpha=0.5, label='Datos entrenamiento')
plt.scatter(centroides[:, 0], centroides[:, 1], 
           marker='*', s=500, c='black', edgecolors='yellow', 
           linewidths=2, label='Centroides')
plt.scatter(test_data[:, 0], test_data[:, 1], 
           c=predicciones, cmap='viridis', s=200, marker='D', 
           edgecolors='red', linewidths=2, label='Puntos de prueba')

# Etiquetar los puntos de prueba
for i, (punto, clase) in enumerate(zip(test_data, predicciones)):
    plt.annotate(f'P{i+1}(C{clase})', 
                xy=punto, xytext=(5, 5), 
                textcoords='offset points', 
                fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

plt.title('KMeans: Datos de Entrenamiento y Puntos de Prueba', fontsize=14, fontweight='bold')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()
plt.colorbar(label='Cluster')
plt.grid(True, alpha=0.3)
plt.savefig('kmeans_prediccion.png', dpi=150, bbox_inches='tight')
plt.show()

# Información adicional del modelo
print("\n" + "="*60)
print("INFORMACIÓN ADICIONAL DEL MODELO")
print("="*60)
print(f"Inercia del modelo: {kmeans.inertia_:.2f}")
print(f"\nCoordenadas de los centroides:")
for i, centroide in enumerate(centroides):
    print(f"  Centroide {i}: ({centroide[0]:.2f}, {centroide[1]:.2f})")

print("\n" + "="*60)
print("RESUMEN DE RESPUESTAS")
print("="*60)
print("✓ Figura A: Datos iniciales graficados")
print("✓ Figura B: Blobs coloreados")
print("✓ Figura C: Clusters con centroides (estrellas negras)")
print(f"✓ Etiquetas: {etiquetas_unicas}, Total datos: {len(X)}")
print("✓ Predicciones realizadas para los 6 puntos de prueba")
print("✓ Clases asignadas a cada punto de prueba mostradas")
print("="*60)