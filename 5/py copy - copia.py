import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import io
from mpl_toolkits.mplot3d import Axes3D

# -------------------------------------------------------
# CONFIGURAR SALIDA UTF-8
# -------------------------------------------------------
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
except:
    pass

# -------------------------------------------------------
# RUTA AL ARCHIVO .NPY  <<< CAMBIA ESTA RUTA >>>
# -------------------------------------------------------
RUTA_NPY = r"C:\Users\ASUS F15\Documents\GitHub\proyfinsistint-\5\puntos_3d.npy"

print("="*70)
print("EJERCICIO 5: CENTROIDE DE COLOR AZUL EN 3D")
print("="*70)

# -------------------------------------------------------
# VERIFICAR ARCHIVO
# -------------------------------------------------------
if not os.path.exists(RUTA_NPY):
    print("\n❌ ERROR: No se encontró el archivo:")
    print(RUTA_NPY)
    print("\n➡ Asegúrate de que la ruta es correcta.")
    sys.exit(1)

# -------------------------------------------------------
# CARGA DE DATOS (como pide el enunciado)
# -------------------------------------------------------
d3 = np.load(RUTA_NPY)

print("\nINFORMACIÓN DE LOS DATOS:")
print("-" * 70)
print(f"Forma del array: {d3.shape}")
print(f"Total de puntos: {len(d3)}")

# Crear DataFrame con 3 columnas (como pide el enunciado)
df = pd.DataFrame(d3)
df.columns = ['X', 'Y', 'Z']

print("\nPRIMERAS 10 FILAS DEL DATAFRAME:")
print("-" * 70)
print(df.head(10))

print("\nESTADÍSTICAS DESCRIPTIVAS:")
print("-" * 70)
print(df.describe())

# -------------------------------------------------------
# CALCULAR CENTROIDE
# -------------------------------------------------------
centroide_x = df['X'].mean()
centroide_y = df['Y'].mean()
centroide_z = df['Z'].mean()

print("\n" + "="*70)
print("CENTROIDE CALCULADO")
print("="*70)
print(f"Centroide X: {centroide_x:.4f}")
print(f"Centroide Y: {centroide_y:.4f}")
print(f"Centroide Z: {centroide_z:.4f}")
print(f"Coordenadas: ({centroide_x:.4f}, {centroide_y:.4f}, {centroide_z:.4f})")

# -------------------------------------------------------
# VISUALIZACIÓN 3D
# -------------------------------------------------------
fig = plt.figure(figsize=(14, 6))

# Gráfico 1: Vista SIN centroide
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(df['X'], df['Y'], df['Z'], c='gray', alpha=0.5, s=30, 
            label='Puntos 3D', edgecolors='darkgray', linewidths=0.5)

ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_title('Vista SIN Centroide')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Gráfico 2: Vista CON centroide AZUL
ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(df['X'], df['Y'], df['Z'], c='gray', alpha=0.3, s=20, 
            edgecolors='darkgray', label='Puntos 3D')

# ⭐ CENTROIDE EN AZUL (según enunciado)
ax2.scatter(centroide_x, centroide_y, centroide_z, c='blue', s=500,
            marker='*', edgecolors='darkblue', linewidths=2, 
            label='Centroide Azul', zorder=10)

# Dibujar ejes guía en azul
ax2.plot([centroide_x, centroide_x], [centroide_y, centroide_y],
         [df['Z'].min(), df['Z'].max()], 'b--', alpha=0.5)

ax2.plot([centroide_x, centroide_x], [df['Y'].min(), df['Y'].max()],
         [centroide_z, centroide_z], 'b--', alpha=0.5)

ax2.plot([df['X'].min(), df['X'].max()], [centroide_y, centroide_y],
         [centroide_z, centroide_z], 'b--', alpha=0.5)

ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
ax2.set_title('Vista CON Centroide Azul')
ax2.legend()
ax2.grid(True, alpha=0.3)

ax2.text(centroide_x, centroide_y, centroide_z, 
         f'  C({centroide_x:.2f}, {centroide_y:.2f}, {centroide_z:.2f})',
         color='blue', fontsize=10, fontweight='bold')

ax2.view_init(elev=20, azim=45)

plt.tight_layout()
plt.savefig('centroide_puntos_3d.png', dpi=300, bbox_inches='tight')
plt.show()

# -------------------------------------------------------
# PROYECCIONES 2D
# -------------------------------------------------------
fig2, axes = plt.subplots(2, 2, figsize=(12, 10))

# Proyección XY
axes[0, 0].scatter(df['X'], df['Y'], c='gray', alpha=0.5)
axes[0, 0].scatter(centroide_x, centroide_y, c='blue', s=200, marker='*',
                  edgecolors='darkblue', linewidths=2)
axes[0, 0].set_title('Proyección XY')
axes[0, 0].set_xlabel('X')
axes[0, 0].set_ylabel('Y')
axes[0, 0].axhline(centroide_y, color='blue', linestyle='--', alpha=0.3)
axes[0, 0].axvline(centroide_x, color='blue', linestyle='--', alpha=0.3)
axes[0, 0].grid(True, alpha=0.3)

# Proyección XZ
axes[0, 1].scatter(df['X'], df['Z'], c='gray', alpha=0.5)
axes[0, 1].scatter(centroide_x, centroide_z, c='blue', s=200, marker='*',
                  edgecolors='darkblue', linewidths=2)
axes[0, 1].set_title('Proyección XZ')
axes[0, 1].set_xlabel('X')
axes[0, 1].set_ylabel('Z')
axes[0, 1].axhline(centroide_z, color='blue', linestyle='--', alpha=0.3)
axes[0, 1].axvline(centroide_x, color='blue', linestyle='--', alpha=0.3)
axes[0, 1].grid(True, alpha=0.3)

# Proyección YZ
axes[1, 0].scatter(df['Y'], df['Z'], c='gray', alpha=0.5)
axes[1, 0].scatter(centroide_y, centroide_z, c='blue', s=200, marker='*',
                  edgecolors='darkblue', linewidths=2)
axes[1, 0].set_title('Proyección YZ')
axes[1, 0].set_xlabel('Y')
axes[1, 0].set_ylabel('Z')
axes[1, 0].axhline(centroide_z, color='blue', linestyle='--', alpha=0.3)
axes[1, 0].axvline(centroide_y, color='blue', linestyle='--', alpha=0.3)
axes[1, 0].grid(True, alpha=0.3)

# Información textual
axes[1, 1].axis('off')
axes[1, 1].text(0.05, 0.5, f"""
INFORMACIÓN DEL CENTROIDE

Total puntos: {len(df)}

Centroide (AZUL):
  X = {centroide_x:.4f}
  Y = {centroide_y:.4f}
  Z = {centroide_z:.4f}

Rangos de datos:
  X: [{df['X'].min():.2f}, {df['X'].max():.2f}]
  Y: [{df['Y'].min():.2f}, {df['Y'].max():.2f}]
  Z: [{df['Z'].min():.2f}, {df['Z'].max():.2f}]
""",
fontsize=11, family='monospace',
bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

plt.tight_layout()
plt.savefig('proyecciones_centroide_3d.png', dpi=300, bbox_inches='tight')
plt.show()

# -------------------------------------------------------
# ANÁLISIS DE DISTANCIAS
# -------------------------------------------------------
df_con_centroide = df.copy()
df_con_centroide['Distancia_al_Centroide'] = np.sqrt(
    (df['X'] - centroide_x)**2 +
    (df['Y'] - centroide_y)**2 +
    (df['Z'] - centroide_z)**2
)

print("\n" + "="*70)
print("ANÁLISIS ADICIONAL")
print("="*70)
print(f"Distancia promedio al centroide: {df_con_centroide['Distancia_al_Centroide'].mean():.4f}")
print(f"Distancia mínima: {df_con_centroide['Distancia_al_Centroide'].min():.4f}")
print(f"Distancia máxima: {df_con_centroide['Distancia_al_Centroide'].max():.4f}")

# Guardar CSV
df_con_centroide.to_csv('puntos_3d_con_distancias.csv', index=False)

print("\n" + "="*70)
print("ARCHIVOS GENERADOS")
print("="*70)
print("1. centroide_puntos_3d.png")
print("2. proyecciones_centroide_3d.png")
print("3. puntos_3d_con_distancias.csv")
print("\n✔ Ejercicio 5 completado exitosamente!")
print("="*70)