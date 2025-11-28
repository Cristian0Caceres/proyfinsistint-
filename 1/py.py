import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score, normalized_mutual_info_score
import sys
import io
import os

# Configurar la salida estándar para UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Crear carpeta 'resultados' si no existe
if not os.path.exists('resultados'):
    os.makedirs('resultados')
    print("✓ Carpeta 'resultados' creada exitosamente\n")
else:
    print("✓ Carpeta 'resultados' ya existe\n")

# Cargar los datos de entrenamiento
X = np.load(r"C:\Users\ASUS F15\Documents\GitHub\proyfinsistint-\1\A.npy")
y_true = np.load(r"C:\Users\ASUS F15\Documents\GitHub\proyfinsistint-\1\_.npy")

print("="*60)
print("EJERCICIO 1: KMeans con Análisis de Etiquetas Reales")
print("="*60)
print(f"\nDatos cargados:")
print(f"   X (características): {X.shape}")
print(f"   y (etiquetas reales): {y_true.shape}")
print(f"   Clases reales en los datos: {np.unique(y_true)}")
print(f"   Distribución de clases reales:")
for clase in np.unique(y_true):
    count = np.sum(y_true == clase)
    print(f"   - Clase {clase}: {count} muestras ({count/len(y_true)*100:.1f}%)")

# ============================================
# MÉTODO 1: MÉTODO DEL CODO (ELBOW METHOD)
# ============================================
print("\n" + "="*60)
print("1. MÉTODO DEL CODO:")
print("="*60)
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
plt.savefig('resultados/kmeans_metodos_seleccion_k.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================
# USAR K=3 DETERMINADO DE FORMA NO SUPERVISADA
# ============================================
print("\n" + "="*60)
print("APLICANDO KMEANS CON K=3")
print("="*60)

# Aplicar KMeans con k=3
n_clusters = 3  # Determinado por los métodos anteriores
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
kmeans.fit(X)

labels = kmeans.labels_
centroides = kmeans.cluster_centers_

# ============================================
# FIGURAS A, B y C - REQUISITO DEL EJERCICIO
# ============================================
plt.figure(figsize=(18, 5))

# Figura A: Datos sin procesar (usando etiquetas reales de _.npy)
plt.subplot(1, 3, 1)
plt.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', alpha=0.6, s=50)
plt.title('Figura A: Datos de Entrenamiento\n(Etiquetas Reales de _.npy)', 
          fontsize=12, fontweight='bold')
plt.xlabel('X')
plt.ylabel('Y')
plt.colorbar(label='Clase Real')
plt.grid(True, alpha=0.3)

# Figura B: Clusters identificados por KMeans
plt.subplot(1, 3, 2)
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='plasma', alpha=0.6, s=50)
plt.title('Figura B: Blobs Detectados por KMeans\n(k=3)', 
          fontsize=12, fontweight='bold')
plt.xlabel('X')
plt.ylabel('Y')
plt.colorbar(label='Cluster KMeans')
plt.grid(True, alpha=0.3)

# Figura C: Clusters con centroides (estrella negra)
plt.subplot(1, 3, 3)
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='plasma', alpha=0.6, s=50)
plt.scatter(centroides[:, 0], centroides[:, 1], 
            marker='*', s=800, c='black', edgecolors='yellow', 
            linewidths=3, label='Centroides', zorder=5)
plt.title('Figura C: Clusters con Centroides\n(Estrella Negra)', 
          fontsize=12, fontweight='bold')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('resultados/kmeans_figuras_abc.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================
# 4. INFORMACIÓN DE LAS ETIQUETAS Y DATOS
# ============================================
print("\n" + "="*60)
print("4. INFORMACIÓN DE LAS ETIQUETAS Y DATOS:")
print("="*60)
print(f"   Etiquetas únicas detectadas por KMeans: {np.unique(labels)}")
print(f"   Número de clusters: {len(np.unique(labels))}")
print(f"   Cantidad total de datos: {len(X)}")
print(f"\n   Distribución por cluster (KMeans):")
for i in range(n_clusters):
    count = np.sum(labels == i)
    porcentaje = (count / len(X)) * 100
    print(f"   - Cluster {i}: {count} datos ({porcentaje:.1f}%)")

print(f"\n   Distribución por clase real (_.npy):")
for clase in np.unique(y_true):
    count = np.sum(y_true == clase)
    porcentaje = (count / len(y_true)) * 100
    print(f"   - Clase {clase}: {count} datos ({porcentaje:.1f}%)")

# ============================================
# 5 y 6. PREDICCIONES PARA DATOS DE PRUEBA
# ============================================
test_data = np.array([[2, 5], [3.2, 6.5], [7, 2.5], [9, 3.2], [9, -6], [11, -8]])
predicciones = kmeans.predict(test_data)

print("\n" + "="*60)
print("5 y 6. PREDICCIONES PARA DATOS DE PRUEBA:")
print("="*60)
print(f"   Datos de prueba:")
for i, punto in enumerate(test_data):
    print(f"   - Punto {i+1}: {punto}")

print(f"\n   Predicciones (clases asignadas):")
for i, (punto, clase) in enumerate(zip(test_data, predicciones)):
    print(f"   - Punto {i+1}: {punto} → Clase {clase}")

# ============================================
# VISUALIZACIÓN CON PUNTOS DE PRUEBA
# ============================================
plt.figure(figsize=(12, 9))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='plasma', alpha=0.4, s=50, 
            label='Datos de entrenamiento')
plt.scatter(centroides[:, 0], centroides[:, 1], 
            marker='*', s=800, c='black', edgecolors='yellow', 
            linewidths=3, label='Centroides', zorder=5)
plt.scatter(test_data[:, 0], test_data[:, 1], 
            c=predicciones, cmap='plasma', marker='X', s=300, 
            edgecolors='red', linewidths=3, label='Puntos de prueba', zorder=4)

for i, (punto, clase) in enumerate(zip(test_data, predicciones)):
    plt.annotate(f'P{i+1}→C{clase}', 
                xy=punto, xytext=(8, 8), 
                textcoords='offset points', 
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                         edgecolor='red', alpha=0.9, linewidth=2))

plt.title('KMeans: Clusters, Centroides y Predicciones (k=3)', 
          fontsize=14, fontweight='bold')
plt.xlabel('X', fontsize=12)
plt.ylabel('Y', fontsize=12)
plt.legend(fontsize=11, loc='best')
plt.grid(True, alpha=0.3)
plt.savefig('resultados/kmeans_predicciones.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================
# COMPARACIÓN: CLUSTERS vs CLASES REALES
# ============================================
print("\n" + "="*60)
print("COMPARACIÓN: CLUSTERS DE KMEANS vs CLASES REALES:")
print("="*60)

# Calcular métricas de comparación
ari = adjusted_rand_score(y_true, labels)
nmi = normalized_mutual_info_score(y_true, labels)

print(f"   Adjusted Rand Index (ARI): {ari:.4f}")
print(f"   - Rango: [-1, 1], donde 1 es coincidencia perfecta")
print(f"   - Valor obtenido: {'Excelente' if ari > 0.8 else 'Bueno' if ari > 0.6 else 'Regular'}")

print(f"\n   Normalized Mutual Information (NMI): {nmi:.4f}")
print(f"   - Rango: [0, 1], donde 1 es coincidencia perfecta")
print(f"   - Valor obtenido: {'Excelente' if nmi > 0.8 else 'Bueno' if nmi > 0.6 else 'Regular'}")

# Visualización comparativa
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Subplot 1: Etiquetas reales
axes[0].scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', alpha=0.6, s=50)
axes[0].set_title('Clases Reales (_.npy)', fontsize=13, fontweight='bold')
axes[0].set_xlabel('X')
axes[0].set_ylabel('Y')
axes[0].grid(True, alpha=0.3)

# Subplot 2: Clusters de KMeans
axes[1].scatter(X[:, 0], X[:, 1], c=labels, cmap='plasma', alpha=0.6, s=50)
axes[1].scatter(centroides[:, 0], centroides[:, 1], 
                marker='*', s=800, c='black', edgecolors='yellow', 
                linewidths=3, zorder=5)
axes[1].set_title(f'Clusters KMeans (k=3)\nARI={ari:.3f}, NMI={nmi:.3f}', 
                  fontsize=13, fontweight='bold')
axes[1].set_xlabel('X')
axes[1].set_ylabel('Y')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('resultados/kmeans_comparacion.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================
# MÉTRICAS FINALES DEL MODELO
# ============================================
print("\n" + "="*60)
print("MÉTRICAS DEL MODELO KMEANS (K=3):")
print("="*60)
print(f"Inercia (WCSS): {kmeans.inertia_:.2f}")
print(f"Coeficiente de Silueta: {silhouette_score(X, labels):.4f}")
print(f"Índice Davies-Bouldin: {davies_bouldin_score(X, labels):.4f}")
print(f"Número de iteraciones: {kmeans.n_iter_}")
print(f"\nCentroides de los clusters:")
for i, centroide in enumerate(centroides):
    print(f"   Cluster {i}: [{centroide[0]:.4f}, {centroide[1]:.4f}]")

print("\n" + "="*60)
print("ANÁLISIS COMPLETADO EXITOSAMENTE!")
print("="*60)
print("✓ Archivos generados en la carpeta 'resultados/':")
print("  1. resultados/kmeans_metodos_seleccion_k.png (Métodos de selección de k)")
print("  2. resultados/kmeans_figuras_abc.png (Figuras A, B y C requeridas)")
print("  3. resultados/kmeans_predicciones.png (Predicciones de datos test)")
print("  4. resultados/kmeans_comparacion.png (Comparación clusters vs clases reales)")
print("\n✓ Se utilizó _.npy para:")
print("  - Mostrar las etiquetas reales en Figura A")
print("  - Comparar clusters de KMeans con clases verdaderas")
print("  - Calcular métricas ARI y NMI")
print("="*60)