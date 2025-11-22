import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import sys
import io

# Configurar la salida estándar para UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("="*70)
print("EJERCICIO 3: REGRESIÓN LINEAL - PESO vs ALTURA")
print("="*70)

# Datos de entrenamiento
altura_train = np.array([1.60, 1.65, 1.70, 1.73, 1.80]).reshape(-1, 1)
peso_train = np.array([60.0, 65.0, 72.3, 75.0, 80.0])

# Datos de prueba
altura_test = np.array([1.58, 1.62, 1.69, 1.76, 1.82]).reshape(-1, 1)

print("\nDATOS DE ENTRENAMIENTO:")
print("-" * 70)
print(f"{'Altura (m)':<15} {'Peso (kg)':<15}")
print("-" * 70)
for h, p in zip(altura_train.flatten(), peso_train):
    print(f"{h:<15.2f} {p:<15.1f}")

# Crear y entrenar el modelo de regresión lineal
modelo = LinearRegression()
modelo.fit(altura_train, peso_train)

# Obtener parámetros del modelo
pendiente = modelo.coef_[0]
intercepto = modelo.intercept_

print("\n" + "="*70)
print("PARÁMETROS DEL MODELO")
print("="*70)
print(f"Ecuación de la recta: Peso = {pendiente:.4f} * Altura + {intercepto:.4f}")
print(f"Pendiente (m): {pendiente:.4f}")
print(f"Intercepto (b): {intercepto:.4f}")

# Realizar predicciones sobre datos de entrenamiento (para calcular RSS)
peso_pred_train = modelo.predict(altura_train)

# Realizar predicciones sobre datos de prueba
peso_pred_test = modelo.predict(altura_test)

print("\n" + "="*70)
print("PREDICCIONES PARA DATOS DE PRUEBA")
print("="*70)
print(f"{'Altura (m)':<15} {'Peso Predicho (kg)':<20}")
print("-" * 70)
for h, p in zip(altura_test.flatten(), peso_pred_test):
    print(f"{h:<15.2f} {p:<20.2f}")

# Calcular RSS (Residual Sum of Squares)
residuos = peso_train - peso_pred_train
rss = np.sum(residuos ** 2)

# Calcular métricas adicionales
mse = mean_squared_error(peso_train, peso_pred_train)
rmse = np.sqrt(mse)
r2 = r2_score(peso_train, peso_pred_train)

print("\n" + "="*70)
print("MÉTRICAS DEL MODELO")
print("="*70)
print(f"RSS (Residual Sum of Squares): {rss:.4f}")
print(f"MSE (Mean Squared Error): {mse:.4f}")
print(f"RMSE (Root Mean Squared Error): {rmse:.4f}")
print(f"R² (Coeficiente de Determinación): {r2:.4f}")

# Mostrar residuos
print("\n" + "="*70)
print("ANÁLISIS DE RESIDUOS (Datos de Entrenamiento)")
print("="*70)
print(f"{'Altura (m)':<12} {'Peso Real':<12} {'Peso Pred':<12} {'Residuo':<12}")
print("-" * 70)
for h, p_real, p_pred, res in zip(altura_train.flatten(), peso_train, peso_pred_train, residuos):
    print(f"{h:<12.2f} {p_real:<12.1f} {p_pred:<12.2f} {res:<12.4f}")

# Visualización
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Gráfico 1: Regresión Lineal con datos de entrenamiento y prueba
ax1 = axes[0]
ax1.scatter(altura_train, peso_train, color='blue', s=100, alpha=0.7, 
            label='Datos de Entrenamiento', edgecolors='black', linewidths=1.5)
ax1.scatter(altura_test, peso_pred_test, color='red', s=100, alpha=0.7, 
            marker='s', label='Predicciones (Test)', edgecolors='black', linewidths=1.5)

# Línea de regresión
altura_linea = np.linspace(altura_train.min(), altura_test.max(), 100).reshape(-1, 1)
peso_linea = modelo.predict(altura_linea)
ax1.plot(altura_linea, peso_linea, color='green', linewidth=2, 
         label=f'y = {pendiente:.2f}x + {intercepto:.2f}')

# Líneas de residuos
for h, p_real, p_pred in zip(altura_train.flatten(), peso_train, peso_pred_train):
    ax1.plot([h, h], [p_real, p_pred], 'r--', alpha=0.5, linewidth=1)

ax1.set_xlabel('Altura (m)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Peso (kg)', fontsize=12, fontweight='bold')
ax1.set_title('Regresión Lineal: Peso vs Altura', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Añadir texto con RSS
ax1.text(0.05, 0.95, f'RSS = {rss:.4f}\nR² = {r2:.4f}', 
         transform=ax1.transAxes, fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Gráfico 2: Residuos
ax2 = axes[1]
ax2.scatter(altura_train, residuos, color='purple', s=100, alpha=0.7, 
            edgecolors='black', linewidths=1.5)
ax2.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Residuo = 0')
ax2.set_xlabel('Altura (m)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Residuo (Peso Real - Peso Predicho)', fontsize=12, fontweight='bold')
ax2.set_title('Gráfico de Residuos', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('regresion_lineal_peso_altura.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*70)
print("INTERPRETACIÓN")
print("="*70)
print(f"-> Por cada metro adicional de altura, el peso aumenta {pendiente:.2f} kg")
print(f"-> El RSS de {rss:.4f} indica la suma de errores cuadrados del modelo")
print(f"-> El R² de {r2:.4f} indica que el modelo explica el {r2*100:.2f}% de la variabilidad")
print("\nArchivo guardado: regresion_lineal_peso_altura.png")
print("="*70)







