import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import sys
import io

# Configurar la salida estándar para UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Cargar los datos de entrenamiento
# Asegúrate de tener los archivos A.npy y _.npy en el mismo directorio
X = np.load('A.npy')
y = np.load('_.npy')

print("="*60)
print("EJERCICIO 1: KMeans")
print("="*60)

# 1. Gráfico de la Figura A (datos sin procesar)
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.scatter(X[:, 0], X[:, 1], c='gray', alpha=0.6)
plt.title('Figura A: Datos de Entrenamiento')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True, alpha=0.3)

# 2. Gráfico de la Figura B (blobs con colores)
plt.subplot(1, 3, 2)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.6)
plt.title('Figura B: Blobs Coloreados')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True, alpha=0.3)

# 3. Aplicar KMeans y obtener clusters y centroides
# Determinar el número de clusters (usualmente basado en los datos)
n_clusters = len(np.unique(y))
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(X)

# Obtener las predicciones y centroides
labels = kmeans.labels_
centroides = kmeans.cluster_centers_

# Gráfico de la Figura C (clusters con centroides)
plt.subplot(1, 3, 3)
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6)
plt.scatter(centroides[:, 0], centroides[:, 1], 
            marker='*', s=500, c='black', edgecolors='yellow', 
            linewidths=2, label='Centroides')
plt.title('Figura C: Clusters con Centroides')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('kmeans_figuras.png', dpi=300, bbox_inches='tight')
plt.show()

# 4. Etiquetas y cantidad de datos
print("\n4. INFORMACIÓN DE LOS DATOS:")
print(f"   Etiquetas únicas: {np.unique(labels)}")
print(f"   Número de clusters: {len(np.unique(labels))}")
print(f"   Cantidad total de datos: {len(X)}")
print(f"   Distribución por cluster:")
for i in range(n_clusters):
    count = np.sum(labels == i)
    print(f"   - Cluster {i}: {count} datos")

# 5 y 6. Predicción de datos de prueba
test_data = np.array([[2, 5], [3.2, 6.5], [7, 2.5], [9, 3.2], [9, -6], [11, -8]])
predicciones = kmeans.predict(test_data)

print("\n5 y 6. PREDICCIONES PARA DATOS DE PRUEBA:")
print(f"   Datos de prueba: {test_data.tolist()}")
print(f"   Predicciones: {predicciones}")
print("\n   Detalle por punto:")
for i, (punto, clase) in enumerate(zip(test_data, predicciones)):
    print(f"   - Punto {i+1}: {punto} -> Clase {clase}")

# Visualización adicional con puntos de prueba
plt.figure(figsize=(10, 8))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.4, s=50, label='Datos de entrenamiento')
plt.scatter(centroides[:, 0], centroides[:, 1], 
            marker='*', s=500, c='black', edgecolors='yellow', 
            linewidths=2, label='Centroides')
plt.scatter(test_data[:, 0], test_data[:, 1], 
            c=predicciones, cmap='viridis', marker='X', s=200, 
            edgecolors='red', linewidths=2, label='Puntos de prueba')

# Etiquetar los puntos de prueba
for i, (punto, clase) in enumerate(zip(test_data, predicciones)):
    plt.annotate(f'P{i+1}(C{clase})', 
                xy=punto, xytext=(5, 5), 
                textcoords='offset points', 
                fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

plt.title('KMeans: Clusters, Centroides y Predicciones', fontsize=14, fontweight='bold')
plt.xlabel('X', fontsize=12)
plt.ylabel('Y', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.savefig('kmeans_predicciones.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*60)
print("INFORMACIÓN DEL MODELO KMEANS:")
print("="*60)
print(f"Inercia (suma de distancias cuadradas): {kmeans.inertia_:.2f}")
print(f"Número de iteraciones: {kmeans.n_iter_}")
print(f"\nCentroides:")
for i, centroide in enumerate(centroides):
    print(f"   Cluster {i}: [{centroide[0]:.2f}, {centroide[1]:.2f}]")

print("\nAnálisis completado exitosamente!")
print("  Archivos guardados: kmeans_figuras.png, kmeans_predicciones.png")