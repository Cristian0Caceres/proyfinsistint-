import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# ============================================================================
# CARGA DE DATOS
# ============================================================================
# Intenta cargar archivos de datos existentes, si no existen crea datos de ejemplo
try:
    X = np.load('A.npy')  # Características de los puntos (coordenadas X, Y)
    y = np.load('_.npy')  # Etiquetas verdaderas de cada punto
except:
    # Genera 300 puntos agrupados en 4 grupos con algo de dispersión
    from sklearn.datasets import make_blobs
    X, y = make_blobs(n_samples=300, centers=4, n_features=2, 
                      cluster_std=1.5, random_state=42)
    print("Usando datos sintéticos de ejemplo")

# ============================================================================
# VISUALIZACIÓN: Tres gráficos mostrando la progresión del análisis
# ============================================================================
plt.figure(figsize=(15, 5))

# FIGURA A: Puntos sin clasificar (todos grises)
plt.subplot(1, 3, 1)
plt.scatter(X[:, 0], X[:, 1], c='gray', s=50, alpha=0.6)
plt.title('Figura A: Datos de Entrenamiento')
plt.xlabel('X1')
plt.ylabel('X2')
plt.grid(True, alpha=0.3)

# FIGURA B: Puntos con sus etiquetas originales (coloreados por grupo real)
plt.subplot(1, 3, 2)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=50, alpha=0.6)
plt.title('Figura B: Blobs Coloreados')
plt.xlabel('X1')
plt.ylabel('X2')
plt.colorbar(label='Clase')
plt.grid(True, alpha=0.3)

# ============================================================================
# ALGORITMO KMEANS: Encuentra grupos automáticamente
# ============================================================================
# Cuenta cuántos grupos diferentes hay en los datos
n_clusters = len(np.unique(y))

# Crea y entrena el modelo KMeans para encontrar los centros de cada grupo
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
kmeans.fit(X)

# Obtiene a qué grupo pertenece cada punto y dónde está el centro de cada grupo
labels = kmeans.labels_
centroides = kmeans.cluster_centers_

# FIGURA C: Puntos agrupados por KMeans con sus centros marcados
plt.subplot(1, 3, 3)
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50, alpha=0.6)
# Marca los centros de cada grupo con estrellas grandes
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

# ============================================================================
# ANÁLISIS: Información sobre los grupos encontrados
# ============================================================================
print("\n" + "="*60)
print("PREGUNTA 4: Información de las Etiquetas")
print("="*60)
etiquetas_unicas = np.unique(labels)
print(f"Etiquetas de los clusters: {etiquetas_unicas}")
print(f"Número total de datos: {len(X)}")
print(f"Número de clusters: {len(etiquetas_unicas)}")

# Muestra cuántos puntos hay en cada grupo
for i in etiquetas_unicas:
    count = np.sum(labels == i)
    print(f"  Cluster {i}: {count} datos")

# ============================================================================
# PREDICCIÓN: Clasifica nuevos puntos no vistos antes
# ============================================================================
print("\n" + "="*60)
print("PREGUNTA 5 y 6: Predicción de Puntos de Prueba")
print("="*60)

# Define 6 puntos nuevos para clasificar
test_data = np.array([[2, 5], [3.2, 6.5], [7, 2.5], [9, 3.2], [9, -6], [11, -8]])
# El modelo decide a qué grupo pertenece cada punto nuevo
predicciones = kmeans.predict(test_data)

print("\nPuntos de prueba y sus predicciones:")
for i, (punto, clase) in enumerate(zip(test_data, predicciones)):
    print(f"  Punto {i+1}: {punto} -> Clase {clase}")

# ============================================================================
# VISUALIZACIÓN FINAL: Muestra todo junto
# ============================================================================
plt.figure(figsize=(10, 7))
# Puntos originales en color suave
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50, alpha=0.5, label='Datos entrenamiento')
# Centros de cada grupo
plt.scatter(centroides[:, 0], centroides[:, 1], 
           marker='*', s=500, c='black', edgecolors='yellow', 
           linewidths=2, label='Centroides')
# Puntos nuevos clasificados (diamantes con borde rojo)
plt.scatter(test_data[:, 0], test_data[:, 1], 
           c=predicciones, cmap='viridis', s=200, marker='D', 
           edgecolors='red', linewidths=2, label='Puntos de prueba')

# Etiqueta cada punto nuevo con su número y clase asignada
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

# ============================================================================
# MÉTRICAS DEL MODELO
# ============================================================================
print("\n" + "="*60)
print("INFORMACIÓN ADICIONAL DEL MODELO")
print("="*60)
# Inercia: qué tan compactos son los grupos (menor es mejor)
print(f"Inercia del modelo: {kmeans.inertia_:.2f}")
print(f"\nCoordenadas de los centroides:")
for i, centroide in enumerate(centroides):
    print(f"  Centroide {i}: ({centroide[0]:.2f}, {centroide[1]:.2f})")

# ============================================================================
# RESUMEN FINAL
# ============================================================================
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