# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, homogeneity_score
from matplotlib.patches import Patch

# ============================================
# PARTE 6 - VISUALIZACIÓN DATASET IRIS
# ============================================

print("=" * 60)
print("PARTE 6 - DATASET IRIS CON KMEANS")
print("=" * 60)

# Cargar el dataset Iris
iris = load_iris()
X = iris.data  # Features: sepal length, sepal width, petal length, petal width
y = iris.target  # Especies: 0=setosa, 1=versicolor, 2=virginica
species_names = iris.target_names

print("\nInformacion del Dataset Iris:")
print(f"Numero de muestras: {len(X)}")
print(f"Caracteristicas: {iris.feature_names}")
print(f"Especies: {species_names}")
print(f"Forma de X: {X.shape}")

# Extraer las columnas que necesitamos para el gráfico
# Sepal Length (columna 0) y Sepal Width (columna 1)
sepal_length = X[:, 0]
sepal_width = X[:, 1]

print(f"\nRango Sepal Length: [{sepal_length.min():.2f}, {sepal_length.max():.2f}]")
print(f"Rango Sepal Width: [{sepal_width.min():.2f}, {sepal_width.max():.2f}]")

# ============================================
# APLICAR KMEANS AL DATASET IRIS
# ============================================

print("\nAplicando KMeans con k=3 clusters...")
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X)

# Obtener predicciones y centroides
labels_kmeans = kmeans.labels_
centroides = kmeans.cluster_centers_

print(f"Centroides encontrados:\n{centroides}")

# Contar muestras por cluster
for i in range(3):
    count = np.sum(labels_kmeans == i)
    print(f"Cluster {i}: {count} muestras")

# ============================================
# GRÁFICO PRINCIPAL - ESPECIES REALES
# ============================================

# ------------------------------------------------------------
# En este gráfico se observa claramente que la especie Setosa
# (amarillo) se separa muy bien de las otras dos, ya que tiene
# sépalos más pequeños. Versicolor y Virginica se superponen
# bastante, por lo que es difícil distinguirlas utilizando
# solo el largo y el ancho del sépalo.
# ------------------------------------------------------------

# Definir colores personalizados para que coincidan con la imagen
# Amarillo, Morado oscuro, Turquesa/Celeste
colors_map = {0: '#FFD700', 1: '#4B0082', 2: '#40E0D0'}  # Gold, Indigo, Turquoise
colors = [colors_map[label] for label in y]

plt.figure(figsize=(10, 7))

# Scatter plot con las especies reales
plt.scatter(
    sepal_length, sepal_width,
    c=colors,
    alpha=0.7,
    s=80,
    edgecolors='black',
    linewidth=0.5
)

# Título y etiquetas
plt.title('Iris Dataset - Sepal Length vs Sepal Width', fontsize=14, fontweight='bold')
plt.xlabel('Sepal Length', fontsize=12)
plt.ylabel('Sepal Width', fontsize=12)

# Configurar límites de los ejes
plt.xlim(4.0, 8.5)
plt.ylim(1.8, 4.7)

# Grid
plt.grid(True, alpha=0.3, linestyle='--')

# Leyenda manual
legend_elements = [
    Patch(facecolor='#FFD700', edgecolor='black', label='Setosa'),
    Patch(facecolor='#4B0082', edgecolor='black', label='Versicolor'),
    Patch(facecolor='#40E0D0', edgecolor='black', label='Virginica')
]
plt.legend(handles=legend_elements, loc='upper right', fontsize=10)

plt.tight_layout()
plt.savefig('iris_sepal_visualization.png', dpi=300, bbox_inches='tight')
print("\n[OK] Grafico principal guardado: iris_sepal_visualization.png")
plt.show()

# ============================================
# GRÁFICO CON CLUSTERS DE KMEANS
# ============================================

# ------------------------------------------------------------
# En este gráfico se ve cómo KMeans agrupa las flores sin
# utilizar las etiquetas reales. El algoritmo identifica bien
# la región de Setosa, pero mezcla parcialmente Versicolor y
# Virginica porque sus sépalos son similares. Las estrellas
# negras representan los centroides (el "promedio" de cada
# grupo encontrado por KMeans).
# ------------------------------------------------------------

plt.figure(figsize=(10, 7))

# Colores para los clusters de KMeans
colors_kmeans_map = {0: '#FFD700', 1: '#4B0082', 2: '#40E0D0'}
colors_kmeans = [colors_kmeans_map[label] for label in labels_kmeans]

plt.scatter(
    sepal_length, sepal_width,
    c=colors_kmeans,
    alpha=0.7,
    s=80,
    edgecolors='black',
    linewidth=0.5
)

# Graficar centroides como estrellas negras (usando las dimensiones de sépalo)
plt.scatter(
    centroides[:, 0], centroides[:, 1],
    marker='*',
    c='black',
    s=500,
    edgecolors='white',
    linewidths=2,
    label='Centroides',
    zorder=5
)

plt.title('Iris Dataset - Clusters KMeans (Sepal Length vs Sepal Width)',
          fontsize=14, fontweight='bold')
plt.xlabel('Sepal Length', fontsize=12)
plt.ylabel('Sepal Width', fontsize=12)
plt.xlim(4.0, 8.5)
plt.ylim(1.8, 4.7)
plt.grid(True, alpha=0.3, linestyle='--')
plt.legend(loc='upper right', fontsize=10)

plt.tight_layout()
plt.savefig('iris_kmeans_clusters.png', dpi=300, bbox_inches='tight')
print("[OK] Grafico con KMeans guardado: iris_kmeans_clusters.png")
plt.show()

# ============================================
# GRÁFICO COMPLETO - Todas las características
# ============================================

# ------------------------------------------------------------
# En la figura de análisis completo se muestran todas las
# combinaciones de pares de características. En general se
# observa que las variables del pétalo (largo y ancho del
# pétalo) separan mucho mejor las especies que las del sépalo.
# Setosa aparece muy aislada, mientras que Versicolor y
# Virginica se distinguen mejor cuando se usan medidas del
# pétalo.
# ------------------------------------------------------------

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Iris Dataset - Analisis Completo', fontsize=16, fontweight='bold')

# Pares de características: (índice1, índice2, nombre1, nombre2)
feature_pairs = [
    (0, 1, 'Sepal Length', 'Sepal Width'),    # Subplot 0
    (0, 2, 'Sepal Length', 'Petal Length'),   # Subplot 1
    (0, 3, 'Sepal Length', 'Petal Width'),    # Subplot 2
    (1, 2, 'Sepal Width', 'Petal Length'),    # Subplot 3
    (1, 3, 'Sepal Width', 'Petal Width'),     # Subplot 4
    (2, 3, 'Petal Length', 'Petal Width')     # Subplot 5
]

for idx, (feat1, feat2, name1, name2) in enumerate(feature_pairs):
    ax = axes[idx // 3, idx % 3]

    # Comentarios específicos para cada uno de los 6 subplots
    # (no afectan la ejecución, solo sirven como mini explicación)
    if idx == 0:
        # En este subplot (Sepal Length vs Sepal Width) se ve que
        # Setosa se separa, pero Versicolor y Virginica siguen muy
        # mezcladas. Las medidas del sépalo no son suficientes para
        # distinguir bien todas las especies.
        pass
    elif idx == 1:
        # En este subplot (Sepal Length vs Petal Length) se aprecia
        # una mejor separación, especialmente porque el largo del
        # pétalo ayuda a distinguir mucho más a Setosa del resto.
        pass
    elif idx == 2:
        # En este subplot (Sepal Length vs Petal Width) ya se nota
        # claramente que las medidas del pétalo permiten separar
        # mejor las especies que solo con el sépalo.
        pass
    elif idx == 3:
        # En este subplot (Sepal Width vs Petal Length) se ve que
        # el ancho del sépalo no aporta tanta separación, pero el
        # largo del pétalo vuelve a marcar bien la diferencia entre
        # Setosa y las otras dos especies.
        pass
    elif idx == 4:
        # En este subplot (Sepal Width vs Petal Width) se observa que
        # el ancho del pétalo aporta una clara separación entre las
        # especies, mientras que el ancho del sépalo sigue mostrando
        # algo de solapamiento.
        pass
    elif idx == 5:
        # En este subplot (Petal Length vs Petal Width) se obtiene
        # la mejor separación de todas: Setosa, Versicolor y
        # Virginica quedan bien diferenciadas, mostrando que las
        # características del pétalo son las más discriminantes.
        pass

    colors = [colors_map[label] for label in y]
    ax.scatter(
        X[:, feat1], X[:, feat2],
        c=colors,
        alpha=0.7,
        s=60,
        edgecolors='black',
        linewidth=0.5
    )

    ax.set_xlabel(name1, fontsize=10)
    ax.set_ylabel(name2, fontsize=10)
    ax.set_title(f'{name1} vs {name2}', fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')

# Leyenda en el último subplot
legend_elements = [
    Patch(facecolor='#FFD700', edgecolor='black', label='Setosa'),
    Patch(facecolor='#4B0082', edgecolor='black', label='Versicolor'),
    Patch(facecolor='#40E0D0', edgecolor='black', label='Virginica')
]
axes[1, 2].legend(handles=legend_elements, loc='center', fontsize=10)

plt.tight_layout()
plt.savefig('iris_complete_analysis.png', dpi=300, bbox_inches='tight')
print("[OK] Analisis completo guardado: iris_complete_analysis.png")
plt.show()

# ============================================
# ESTADÍSTICAS Y REPORTE
# ============================================

print("\n" + "=" * 60)
print("ESTADISTICAS DEL DATASET")
print("=" * 60)

for i, species in enumerate(species_names):
    mask = y == i
    print(f"\n{species.upper()}:")
    print(f"  Sepal Length: media={X[mask, 0].mean():.2f}, std={X[mask, 0].std():.2f}")
    print(f"  Sepal Width:  media={X[mask, 1].mean():.2f}, std={X[mask, 1].std():.2f}")
    print(f"  Petal Length: media={X[mask, 2].mean():.2f}, std={X[mask, 2].std():.2f}")
    print(f"  Petal Width:  media={X[mask, 3].mean():.2f}, std={X[mask, 3].std():.2f}")

# Comparar clustering real vs KMeans
ari = adjusted_rand_score(y, labels_kmeans)
homogeneity = homogeneity_score(y, labels_kmeans)

print("\n" + "=" * 60)
print("EVALUACION DEL CLUSTERING")
print("=" * 60)
print(f"Adjusted Rand Index: {ari:.4f}")
print(f"Homogeneity Score: {homogeneity:.4f}")
print("\nInterpretacion:")
if ari > 0.8:
    print("  -> Clustering MUY BUENO: KMeans identifico bien las especies")
elif ari > 0.5:
    print("  -> Clustering BUENO: KMeans capturo bastante estructura")
else:
    print("  -> Clustering MODERADO: Hay superposicion entre especies")

# Guardar reporte en archivo de texto
with open('iris_report.txt', 'w', encoding='utf-8') as f:
    f.write("=" * 60 + "\n")
    f.write("PARTE 6 - REPORTE DATASET IRIS\n")
    f.write("=" * 60 + "\n\n")

    f.write(f"Numero de muestras: {len(X)}\n")
    f.write(f"Caracteristicas: {iris.feature_names}\n")
    f.write(f"Especies: {list(species_names)}\n\n")

    f.write("CENTROIDES KMEANS:\n")
    for i, c in enumerate(centroides):
        f.write(f"  Cluster {i}: {c}\n")

    f.write(f"\nAdjusted Rand Index: {ari:.4f}\n")
    f.write(f"Homogeneity Score: {homogeneity:.4f}\n")

    f.write("\nESTADISTICAS POR ESPECIE:\n")
    for i, species in enumerate(species_names):
        mask = y == i
        f.write(f"\n{species.upper()}:\n")
        f.write(f"  Sepal Length: {X[mask, 0].mean():.2f} +/- {X[mask, 0].std():.2f}\n")
        f.write(f"  Sepal Width:  {X[mask, 1].mean():.2f} +/- {X[mask, 1].std():.2f}\n")

print("\n[OK] Reporte guardado: iris_report.txt")

print("\n" + "=" * 60)
print("ANALISIS COMPLETADO")
print("Archivos generados:")
print("  * iris_sepal_visualization.png (grafico principal)")
print("  * iris_kmeans_clusters.png (con centroides)")
print("  * iris_complete_analysis.png (analisis completo)")
print("  * iris_report.txt")
print("=" * 60)
