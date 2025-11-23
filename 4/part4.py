import numpy as np
import matplotlib.pyplot as plt

# ============================================
# PARTE 4 - DISTANCIA DEL COSENO
# ============================================

print("=" * 60)
print("PARTE 4 - SIMILARIDAD Y DISTANCIA DEL COSENO")
print("=" * 60)

# Definir los vectores segun el PDF
A = np.array([2, 1, 0, 2, 0, 1, 1, 1])
B = np.array([2, 1, 1, 1, 1, 0, 1, 1])
P = np.array([1, 2, 3, 0, 4, 6, 7, 9])
Q = np.array([2, 4, 5, 1, 8, 2, 4, 1])
S = np.array([2, 1, 4, 7, 1, 4, 5, 6])
T = np.array([3, 3, 5, 6, 1, 1, 7, 8])

# Funcion para calcular la similaridad del coseno
def cosine_similarity(vec1, vec2):
    """
    Calcula la similaridad del coseno entre dos vectores.
    cos(theta) = (A·B) / (||A|| * ||B||)
    """
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0
    
    return dot_product / (norm_vec1 * norm_vec2)

# Funcion para calcular la distancia del coseno
def cosine_distance(vec1, vec2):
    """
    Distancia del coseno = 1 - similaridad del coseno
    """
    return 1 - cosine_similarity(vec1, vec2)

# Funcion para calcular el angulo en radianes y grados
def angle_between_vectors(vec1, vec2):
    """
    Calcula el angulo theta entre dos vectores usando arccos(cos(theta))
    """
    cos_theta = cosine_similarity(vec1, vec2)
    # Asegurar que este en el rango [-1, 1] por errores numericos
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta_rad = np.arccos(cos_theta)
    theta_deg = np.degrees(theta_rad)
    return theta_rad, theta_deg

# ============================================
# CALCULOS
# ============================================

# Calcular similaridades del coseno
cos_AB = cosine_similarity(A, B)
cos_PQ = cosine_similarity(P, Q)
cos_ST = cosine_similarity(S, T)

# Calcular distancias del coseno
dist_AB = cosine_distance(A, B)
dist_PQ = cosine_distance(P, Q)
dist_ST = cosine_distance(S, T)

# Calcular angulos
angle_AB_rad, angle_AB_deg = angle_between_vectors(A, B)
angle_PQ_rad, angle_PQ_deg = angle_between_vectors(P, Q)
angle_ST_rad, angle_ST_deg = angle_between_vectors(S, T)

# ============================================
# PREGUNTA 1: EXPLICAR RESULTADOS
# ============================================

print("\n--- PREGUNTA 1: RESULTADOS Y EXPLICACION ---\n")

print(f"Vectores A = {A}")
print(f"Vectores B = {B}")
print(f"Cos(A,B) = {cos_AB:.6f}")
print(f"Distancia del Coseno(A,B) = {dist_AB:.6f}")
print(f"Interpretacion: ", end="")
if cos_AB > 0.9:
    print("Vectores MUY SIMILARES (casi paralelos)")
elif cos_AB > 0.7:
    print("Vectores SIMILARES (orientacion parecida)")
elif cos_AB > 0.3:
    print("Vectores MODERADAMENTE SIMILARES")
elif cos_AB > 0:
    print("Vectores POCO SIMILARES pero en la misma direccion general")
elif cos_AB == 0:
    print("Vectores ORTOGONALES (perpendiculares)")
else:
    print("Vectores en DIRECCIONES OPUESTAS")

print("\n" + "-" * 60 + "\n")

print(f"Vectores P = {P}")
print(f"Vectores Q = {Q}")
print(f"Cos(P,Q) = {cos_PQ:.6f}")
print(f"Distancia del Coseno(P,Q) = {dist_PQ:.6f}")
print(f"Interpretacion: ", end="")
if cos_PQ > 0.9:
    print("Vectores MUY SIMILARES (casi paralelos)")
elif cos_PQ > 0.7:
    print("Vectores SIMILARES (orientacion parecida)")
elif cos_PQ > 0.3:
    print("Vectores MODERADAMENTE SIMILARES")
elif cos_PQ > 0:
    print("Vectores POCO SIMILARES pero en la misma direccion general")
elif cos_PQ == 0:
    print("Vectores ORTOGONALES (perpendiculares)")
else:
    print("Vectores en DIRECCIONES OPUESTAS")

print("\n" + "-" * 60 + "\n")

print(f"Vectores S = {S}")
print(f"Vectores T = {T}")
print(f"Cos(S,T) = {cos_ST:.6f}")
print(f"Distancia del Coseno(S,T) = {dist_ST:.6f}")
print(f"Interpretacion: ", end="")
if cos_ST > 0.9:
    print("Vectores MUY SIMILARES (casi paralelos)")
elif cos_ST > 0.7:
    print("Vectores SIMILARES (orientacion parecida)")
elif cos_ST > 0.3:
    print("Vectores MODERADAMENTE SIMILARES")
elif cos_ST > 0:
    print("Vectores POCO SIMILARES pero en la misma direccion general")
elif cos_ST == 0:
    print("Vectores ORTOGONALES (perpendiculares)")
else:
    print("Vectores en DIRECCIONES OPUESTAS")

# ============================================
# PREGUNTA 2: ANGULOS
# ============================================

print("\n\n--- PREGUNTA 2: ANGULOS ENTRE VECTORES ---\n")

print(f"Angulo entre A y B:")
print(f"  theta = {angle_AB_rad:.6f} radianes = {angle_AB_deg:.2f} grados")

print(f"\nAngulo entre P y Q:")
print(f"  theta = {angle_PQ_rad:.6f} radianes = {angle_PQ_deg:.2f} grados")

print(f"\nAngulo entre S y T:")
print(f"  theta = {angle_ST_rad:.6f} radianes = {angle_ST_deg:.2f} grados")

# ============================================
# PREGUNTA 3: theta = 0 rad
# ============================================

print("\n\n--- PREGUNTA 3: SI theta = 0 rad ---\n")
print("Cuando theta = 0 radianes significa que:")
print("  * Los vectores apuntan en la MISMA DIRECCION")
print("  * Son PARALELOS y con el mismo sentido")
print("  * cos(0) = 1 -> Similaridad del Coseno = 1 (maxima similaridad)")
print("  * Distancia del Coseno = 1 - 1 = 0 (distancia minima)")
print("  * Los vectores son proporcionales: B = k*A (con k > 0)")
print("\nEjemplo: A = [1, 2, 3] y B = [2, 4, 6] -> theta = 0")

# ============================================
# PREGUNTA 4: theta = pi/2 rad
# ============================================

print("\n\n--- PREGUNTA 4: SI theta = pi/2 rad ---\n")
print("Cuando theta = pi/2 radianes (90 grados) significa que:")
print("  * Los vectores son ORTOGONALES (perpendiculares)")
print("  * No tienen similaridad direccional")
print("  * cos(pi/2) = 0 -> Similaridad del Coseno = 0")
print("  * Distancia del Coseno = 1 - 0 = 1")
print("  * El producto punto A*B = 0")
print("  * Son completamente INDEPENDIENTES en terminos de direccion")
print("\nEjemplo: A = [1, 0] y B = [0, 1] -> theta = pi/2")

# ============================================
# VISUALIZACION
# ============================================

# Crear grafico comparativo
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

pares = [(A, B, 'A y B'), (P, Q, 'P y Q'), (S, T, 'S y T')]
resultados = [(cos_AB, angle_AB_deg), (cos_PQ, angle_PQ_deg), (cos_ST, angle_ST_deg)]

for idx, ((vec1, vec2, nombre), (cos_val, angle_deg)) in enumerate(zip(pares, resultados)):
    ax = axes[idx]
    
    # Para visualizacion, proyectamos a 2D tomando las primeras 2 dimensiones
    ax.quiver(0, 0, vec1[0], vec1[1], angles='xy', scale_units='xy', scale=1, 
              color='blue', width=0.006, label=nombre.split()[0])
    ax.quiver(0, 0, vec2[0], vec2[1], angles='xy', scale_units='xy', scale=1, 
              color='red', width=0.006, label=nombre.split()[2])
    
    # Configurar limites
    max_val = max(abs(vec1[0]), abs(vec1[1]), abs(vec2[0]), abs(vec2[1])) * 1.2
    ax.set_xlim(-0.5, max_val)
    ax.set_ylim(-0.5, max_val)
    
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_title(f'{nombre}\ncos(theta)={cos_val:.4f}, theta={angle_deg:.1f} grados')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_aspect('equal')

plt.tight_layout()
plt.savefig('cosine_distance_visualization.png', dpi=300, bbox_inches='tight')
print("\n\nGrafico de visualizacion guardado como 'cosine_distance_visualization.png'")
plt.show()

# Grafico de barras comparativo
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Similaridades
nombres = ['Cos(A,B)', 'Cos(P,Q)', 'Cos(S,T)']
similaridades = [cos_AB, cos_PQ, cos_ST]
colors = ['#2ecc71', '#3498db', '#e74c3c']

ax1.bar(nombres, similaridades, color=colors, alpha=0.7, edgecolor='black')
ax1.set_ylabel('Similaridad del Coseno')
ax1.set_title('Comparacion de Similaridades')
ax1.set_ylim([0, 1])
ax1.grid(True, alpha=0.3, axis='y')
for i, v in enumerate(similaridades):
    ax1.text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold')

# Angulos
angulos = [angle_AB_deg, angle_PQ_deg, angle_ST_deg]
ax2.bar(nombres, angulos, color=colors, alpha=0.7, edgecolor='black')
ax2.set_ylabel('Angulo (grados)')
ax2.set_title('Comparacion de Angulos')
ax2.set_ylim([0, 90])
ax2.grid(True, alpha=0.3, axis='y')
for i, v in enumerate(angulos):
    ax2.text(i, v + 2, f'{v:.1f} grados', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('cosine_comparison.png', dpi=300, bbox_inches='tight')
print("Grafico comparativo guardado como 'cosine_comparison.png'")
plt.show()

# Guardar reporte
with open('cosine_report.txt', 'w', encoding='utf-8') as f:
    f.write("=" * 60 + "\n")
    f.write("PARTE 4 - REPORTE DE DISTANCIA DEL COSENO\n")
    f.write("=" * 60 + "\n\n")
    
    f.write("VECTORES:\n")
    f.write(f"A = {A}\n")
    f.write(f"B = {B}\n")
    f.write(f"P = {P}\n")
    f.write(f"Q = {Q}\n")
    f.write(f"S = {S}\n")
    f.write(f"T = {T}\n\n")
    
    f.write("RESULTADOS:\n")
    f.write(f"Cos(A,B) = {cos_AB:.6f}, Angulo = {angle_AB_rad:.6f} rad = {angle_AB_deg:.2f} grados\n")
    f.write(f"Cos(P,Q) = {cos_PQ:.6f}, Angulo = {angle_PQ_rad:.6f} rad = {angle_PQ_deg:.2f} grados\n")
    f.write(f"Cos(S,T) = {cos_ST:.6f}, Angulo = {angle_ST_rad:.6f} rad = {angle_ST_deg:.2f} grados\n\n")
    
    f.write("INTERPRETACIONES:\n")
    f.write("theta = 0 rad: Vectores paralelos (misma direccion), cos(theta) = 1\n")
    f.write("theta = pi/2 rad: Vectores ortogonales (perpendiculares), cos(theta) = 0\n")

print("Reporte guardado como 'cosine_report.txt'")

print("\n" + "=" * 60)
print("ANALISIS COMPLETADO")
print("=" * 60)