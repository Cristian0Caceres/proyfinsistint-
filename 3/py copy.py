import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

print("="*80)
print(" EJERCICIO 3: REGRESION LINEAL - PREDICCION DE PESO A PARTIR DE ALTURA")
print("="*80)

# Datos de entrenamiento según la tabla proporcionada
altura_train = np.array([1.60, 1.65, 1.70, 1.73, 1.80]).reshape(-1, 1)
peso_train = np.array([60.0, 65.0, 72.3, 75.0, 80.0])

# Datos de test según el enunciado
altura_test = np.array([1.58, 1.62, 1.69, 1.76, 1.82]).reshape(-1, 1)

print("\nDATOS DE ENTRENAMIENTO:")
print("-" * 80)
print(f"{'Altura (m)':<20} {'Peso (kg)':<20}")
print("-" * 80)
for h, p in zip(altura_train.flatten(), peso_train):
    print(f"{h:<20.2f} {p:<20.1f}")

# Crear y entrenar el modelo de regresión lineal
modelo = LinearRegression()
modelo.fit(altura_train, peso_train)

# Obtener parámetros del modelo
pendiente = modelo.coef_[0]
intercepto = modelo.intercept_

print("\n" + "="*80)
print("PARAMETROS DEL MODELO DE REGRESION LINEAL")
print("="*80)
print(f"Ecuación de la recta: Peso = {pendiente:.4f} × Altura + ({intercepto:.4f})")
print(f"Pendiente (m):  {pendiente:.4f} kg/m")
print(f"Intercepto (b): {intercepto:.4f} kg")
print(f"\nInterpretación: Por cada metro adicional de altura,")
print(f"                el peso aumenta aproximadamente {pendiente:.2f} kg")

# Realizar predicciones sobre datos de entrenamiento
peso_pred_train = modelo.predict(altura_train)

# Realizar predicciones sobre datos de test
peso_pred_test = modelo.predict(altura_test)

print("\n" + "="*80)
print("PREDICCIONES PARA DATOS DE TEST")
print("="*80)
print(f"{'Altura (m)':<20} {'Peso Predicho (kg)':<25}")
print("-" * 80)
for h, p in zip(altura_test.flatten(), peso_pred_test):
    print(f"{h:<20.2f} {p:<25.2f}")

# Calcular RSS (Residual Sum of Squares) - REQUISITO DEL EJERCICIO
residuos = peso_train - peso_pred_train
RSS = np.sum(residuos ** 2)

# Calcular métricas adicionales
mse = mean_squared_error(peso_train, peso_pred_train)
rmse = np.sqrt(mse)
r2 = r2_score(peso_train, peso_pred_train)

print("\n" + "="*80)
print("METRICAS DE EVALUACION DEL MODELO")
print("="*80)
print(f"RSS (Residual Sum of Squares):     {RSS:.6f}")
print(f"MSE (Mean Squared Error):          {mse:.6f}")
print(f"RMSE (Root Mean Squared Error):    {rmse:.6f}")
print(f"R² (Coeficiente de Determinación): {r2:.6f}")
print(f"\nEl modelo explica el {r2*100:.2f}% de la variabilidad en los datos")

# Análisis de residuos
print("\n" + "="*80)
print("ANALISIS DETALLADO DE RESIDUOS (Datos de Entrenamiento)")
print("="*80)
print(f"{'Altura (m)':<15} {'Peso Real':<15} {'Peso Pred':<15} {'Residuo':<15}")
print("-" * 80)
for h, p_real, p_pred, res in zip(altura_train.flatten(), peso_train, 
                                    peso_pred_train, residuos):
    print(f"{h:<15.2f} {p_real:<15.1f} {p_pred:<15.2f} {res:+15.4f}")

# Visualización
fig = plt.figure(figsize=(16, 6))

# Subplot 1: Regresión Lineal
ax1 = plt.subplot(1, 3, 1)
ax1.scatter(altura_train, peso_train, color='dodgerblue', s=150, alpha=0.8,
            label='Datos de Entrenamiento', edgecolors='black', linewidths=2, zorder=3)
ax1.scatter(altura_test, peso_pred_test, color='orangered', s=150, alpha=0.8,
            marker='s', label='Predicciones (Test)', edgecolors='black', linewidths=2, zorder=3)

# Línea de regresión
altura_linea = np.linspace(altura_train.min()-0.05, altura_test.max()+0.05, 200).reshape(-1, 1)
peso_linea = modelo.predict(altura_linea)
ax1.plot(altura_linea, peso_linea, color='green', linewidth=3,
         label=f'y = {pendiente:.2f}x + {intercepto:.2f}', zorder=2)

# Líneas de residuos
for h, p_real, p_pred in zip(altura_train.flatten(), peso_train, peso_pred_train):
    ax1.plot([h, h], [p_real, p_pred], 'r--', alpha=0.4, linewidth=1.5, zorder=1)

ax1.set_xlabel('Altura (m)', fontsize=13, fontweight='bold')
ax1.set_ylabel('Peso (kg)', fontsize=13, fontweight='bold')
ax1.set_title('Regresión Lineal: Peso vs Altura', fontsize=14, fontweight='bold', pad=15)
ax1.legend(fontsize=10, loc='upper left')
ax1.grid(True, alpha=0.3, linestyle='--')

# Añadir texto con métricas
textstr = f'RSS = {RSS:.4f}\nR² = {r2:.4f}\nRMSE = {rmse:.4f}'
ax1.text(0.98, 0.02, textstr, transform=ax1.transAxes, fontsize=10,
         verticalalignment='bottom', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

# Subplot 2: Gráfico de Residuos
ax2 = plt.subplot(1, 3, 2)
ax2.scatter(altura_train, residuos, color='purple', s=150, alpha=0.8,
            edgecolors='black', linewidths=2)
ax2.axhline(y=0, color='red', linestyle='--', linewidth=2.5, label='Residuo = 0', zorder=1)
ax2.set_xlabel('Altura (m)', fontsize=13, fontweight='bold')
ax2.set_ylabel('Residuo (kg)', fontsize=13, fontweight='bold')
ax2.set_title('Análisis de Residuos', fontsize=14, fontweight='bold', pad=15)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, linestyle='--')

# Subplot 3: Comparación Real vs Predicho
ax3 = plt.subplot(1, 3, 3)
ax3.scatter(peso_train, peso_pred_train, color='mediumseagreen', s=150, alpha=0.8,
            edgecolors='black', linewidths=2)
# Línea perfecta (y = x)
min_val = min(peso_train.min(), peso_pred_train.min())
max_val = max(peso_train.max(), peso_pred_train.max())
ax3.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2.5,
         label='Predicción Perfecta', zorder=1)
ax3.set_xlabel('Peso Real (kg)', fontsize=13, fontweight='bold')
ax3.set_ylabel('Peso Predicho (kg)', fontsize=13, fontweight='bold')
ax3.set_title('Real vs Predicho', fontsize=14, fontweight='bold', pad=15)
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('regresion_lineal_peso_altura.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*80)
print("INTERPRETACION Y CONCLUSIONES")
print("="*80)
print(f"* Pendiente: {pendiente:.2f} kg/m")
print(f"  -> Por cada metro adicional de altura, el peso aumenta {pendiente:.2f} kg")
print(f"\n* RSS = {RSS:.6f}")
print(f"  -> Suma de errores al cuadrado (menor es mejor)")
print(f"\n* R² = {r2:.6f}")
print(f"  -> El modelo explica el {r2*100:.2f}% de la variabilidad en los datos")
print(f"  -> Ajuste {'EXCELENTE' if r2 > 0.95 else 'BUENO' if r2 > 0.80 else 'REGULAR'}")
print(f"\n* RMSE = {rmse:.4f} kg")
print(f"  -> Error promedio de prediccion")
print("\n>> Grafico guardado: 'regresion_lineal_peso_altura.png'")
print("="*80)