import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# PARTE 8 - FILTRO COMPLEMENTARIO
# =============================================================================

print("=" * 70)
print("PARTE 8 - FILTRO COMPLEMENTARIO APLICADO A FILTRO DE KALMAN")
print("=" * 70)

# =============================================================================
# Configuración y generación de datos (mismo que Parte 7)
# =============================================================================
dt = 0.1
t = np.arange(0, 1000, dt)
pos_real = 0.1 * (3*t - t**2)

np.random.seed(42)
ruido_medicion = 2.0
mediciones = pos_real + np.random.normal(0, ruido_medicion, len(pos_real))

# =============================================================================
# Filtro de Kalman 1D (mismo código de Parte 7)
# =============================================================================
class KalmanFilter1D:
    def __init__(self, dt, process_variance, measurement_variance, 
                 initial_position, initial_velocity):
        self.dt = dt
        self.x = np.array([[initial_position], [initial_velocity]])
        self.F = np.array([[1, dt], [0, 1]])
        self.H = np.array([[1, 0]])
        self.P = np.array([[1000, 0], [0, 1000]])
        self.Q = process_variance * np.eye(2)
        self.R = np.array([[measurement_variance]])
        self.history_covariance = []

    def process(self, measurement):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        y = measurement - (self.H @ self.x)[0, 0]
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T / S[0, 0]
        self.x = self.x + K * y
        self.P = (np.eye(2) - K @ self.H) @ self.P
        self.history_covariance.append(np.sqrt(self.P[0, 0]))
        return self.x[0, 0], self.x[1, 0]

# Ejecutar Kalman
kf = KalmanFilter1D(
    dt, 
    process_variance=0.1, 
    measurement_variance=ruido_medicion**2,
    initial_position=mediciones[0], 
    initial_velocity=(mediciones[1] - mediciones[0]) / dt
)

pos_est_kalman, vel_est = zip(*[kf.process(m) for m in mediciones])
pos_est_kalman = np.array(pos_est_kalman)

# =============================================================================
# FILTRO COMPLEMENTARIO
# =============================================================================

def filtro_complementario(senal, alpha):
    """
    Aplica un filtro complementario a una señal.
    
    Fórmula: y[n] = alpha * y[n-1] + (1 - alpha) * x[n]
    
    Parámetros:
    - senal: señal de entrada (numpy array)
    - alpha: coeficiente de suavizado (0 < alpha < 1)
             - alpha cercano a 1: más suave pero más retardo
             - alpha cercano a 0: menos suave, respuesta más rápida
    
    Retorna:
    - senal_filtrada: señal suavizada
    """
    senal_filtrada = np.zeros_like(senal)
    senal_filtrada[0] = senal[0]  # Inicializar con el primer valor
    
    for i in range(1, len(senal)):
        senal_filtrada[i] = alpha * senal_filtrada[i-1] + (1 - alpha) * senal[i]
    
    return senal_filtrada

# =============================================================================
# Probar diferentes valores de alpha
# =============================================================================

alphas = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.98]

print("\n--- PROBANDO DIFERENTES VALORES DE ALPHA ---\n")

# Calcular errores para cada alpha
errores_mse = []
for alpha in alphas:
    pos_filtrada = filtro_complementario(pos_est_kalman, alpha)
    mse = np.mean((pos_filtrada - pos_real)**2)
    errores_mse.append(mse)
    print(f"Alpha = {alpha:.2f} | MSE = {mse:.4f}")

# Encontrar el mejor alpha
idx_mejor = np.argmin(errores_mse)
mejor_alpha = alphas[idx_mejor]
mejor_mse = errores_mse[idx_mejor]

print(f"\n{'='*70}")
print(f"MEJOR ALPHA ENCONTRADO: {mejor_alpha:.2f} con MSE = {mejor_mse:.4f}")
print(f"{'='*70}\n")

# Aplicar filtro complementario con el mejor alpha
pos_filtrada_mejor = filtro_complementario(pos_est_kalman, mejor_alpha)

# =============================================================================
# VISUALIZACIONES
# =============================================================================

# Gráfico 1: Comparación de diferentes alphas (zoom primeros 100s)
zoom_idx = int(100/dt)
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Filtro Complementario: Comparación de Diferentes Alphas', 
             fontsize=16, fontweight='bold')

alphas_muestra = [0.1, 0.5, 0.9, 0.98]
for idx, alpha in enumerate(alphas_muestra):
    ax = axes[idx//2, idx%2]
    pos_filt = filtro_complementario(pos_est_kalman, alpha)
    
    ax.plot(t[:zoom_idx], pos_real[:zoom_idx], 'g-', linewidth=2.5, 
            label='Posición Real', alpha=0.8)
    ax.plot(t[:zoom_idx], pos_est_kalman[:zoom_idx], 'b--', linewidth=1.5, 
            label='Kalman', alpha=0.6)
    ax.plot(t[:zoom_idx], pos_filt[:zoom_idx], 'r-', linewidth=2, 
            label=f'Complementario (alpha={alpha})')
    
    mse = np.mean((pos_filt[:zoom_idx] - pos_real[:zoom_idx])**2)
    ax.set_title(f'Alpha = {alpha} | MSE = {mse:.4f}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Tiempo (s)', fontsize=11)
    ax.set_ylabel('Posición (m)', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('filtro_complementario_comparacion.png', dpi=300, bbox_inches='tight')
print("Guardado: filtro_complementario_comparacion.png")
plt.show()

# Gráfico 2: Resultado final con mejor alpha (trayectoria completa)
plt.figure(figsize=(16, 8))
plt.plot(t, pos_real, 'g-', linewidth=3, label='Posición Real', alpha=0.8)
plt.plot(t, mediciones, 'gray', alpha=0.3, linewidth=0.8, label='Mediciones con Ruido')
plt.plot(t, pos_est_kalman, 'b--', linewidth=2, label='Filtro de Kalman', alpha=0.7)
plt.plot(t, pos_filtrada_mejor, 'r-', linewidth=2.5, 
         label=f'Filtro Complementario (alpha={mejor_alpha})')

plt.xlabel('Tiempo (s)', fontsize=14)
plt.ylabel('Posición (m)', fontsize=14)
plt.title(f'Resultado Final: Filtro Complementario con alpha={mejor_alpha} (MSE={mejor_mse:.4f})', 
          fontsize=16, fontweight='bold')
plt.legend(fontsize=13, loc='upper right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('filtro_complementario_resultado_completo.png', dpi=300, bbox_inches='tight')
print("Guardado: filtro_complementario_resultado_completo.png")
plt.show()

# Gráfico 3: Zoom en primeros 50 segundos
zoom_idx_50 = int(50/dt)
plt.figure(figsize=(16, 8))
plt.plot(t[:zoom_idx_50], pos_real[:zoom_idx_50], 'g-', linewidth=3, 
         label='Posición Real', alpha=0.8)
plt.plot(t[:zoom_idx_50], mediciones[:zoom_idx_50], 'gray', alpha=0.4, 
         linewidth=1, label='Mediciones con Ruido')
plt.plot(t[:zoom_idx_50], pos_est_kalman[:zoom_idx_50], 'b--', linewidth=2, 
         label='Filtro de Kalman', alpha=0.7)
plt.plot(t[:zoom_idx_50], pos_filtrada_mejor[:zoom_idx_50], 'r-', linewidth=2.5, 
         label=f'Filtro Complementario (alpha={mejor_alpha})')

plt.xlabel('Tiempo (s)', fontsize=14)
plt.ylabel('Posición (m)', fontsize=14)
plt.title(f'Zoom (0-50s): Filtro Complementario con alpha={mejor_alpha}', 
          fontsize=16, fontweight='bold')
plt.legend(fontsize=13)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('filtro_complementario_zoom_50s.png', dpi=300, bbox_inches='tight')
print("Guardado: filtro_complementario_zoom_50s.png")
plt.show()

# Gráfico 4: Gráfico de MSE vs Alpha
plt.figure(figsize=(12, 6))
plt.plot(alphas, errores_mse, 'b-o', linewidth=2, markersize=8)
plt.axvline(mejor_alpha, color='r', linestyle='--', linewidth=2, 
            label=f'Mejor alpha = {mejor_alpha}')
plt.xlabel('Alpha', fontsize=14)
plt.ylabel('Error Cuadrático Medio (MSE)', fontsize=14)
plt.title('Análisis de Error: MSE vs Alpha', fontsize=16, fontweight='bold')
plt.legend(fontsize=13)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('filtro_complementario_mse_vs_alpha.png', dpi=300, bbox_inches='tight')
print("Guardado: filtro_complementario_mse_vs_alpha.png")
plt.show()

# =============================================================================
# ANÁLISIS Y REPORTE
# =============================================================================

# Calcular métricas
mae_kalman = np.mean(np.abs(pos_est_kalman - pos_real))
mae_complementario = np.mean(np.abs(pos_filtrada_mejor - pos_real))

mse_kalman = np.mean((pos_est_kalman - pos_real)**2)
mse_complementario = mejor_mse

rmse_kalman = np.sqrt(mse_kalman)
rmse_complementario = np.sqrt(mse_complementario)

print("\n" + "="*70)
print("ANALISIS COMPARATIVO")
print("="*70)
print(f"\nFILTRO DE KALMAN:")
print(f"  MAE  = {mae_kalman:.4f}")
print(f"  MSE  = {mse_kalman:.4f}")
print(f"  RMSE = {rmse_kalman:.4f}")

print(f"\nFILTRO COMPLEMENTARIO (alpha={mejor_alpha}):")
print(f"  MAE  = {mae_complementario:.4f}")
print(f"  MSE  = {mse_complementario:.4f}")
print(f"  RMSE = {rmse_complementario:.4f}")

mejora_porcentual = ((mse_kalman - mse_complementario) / mse_kalman) * 100
print(f"\nMEJORA: {mejora_porcentual:.2f}% en MSE")

# Guardar reporte
with open('filtro_complementario_reporte.txt', 'w', encoding='utf-8') as f:
    f.write("="*70 + "\n")
    f.write("PARTE 8 - REPORTE FILTRO COMPLEMENTARIO\n")
    f.write("="*70 + "\n\n")
    
    f.write("EXPLICACION DEL FILTRO COMPLEMENTARIO:\n")
    f.write("-" * 70 + "\n")
    f.write("El filtro complementario combina dos fuentes de información:\n")
    f.write("  y[n] = alpha * y[n-1] + (1 - alpha) * x[n]\n\n")
    f.write("Donde:\n")
    f.write("  - alpha: coeficiente de suavizado (0 < alpha < 1)\n")
    f.write("  - y[n-1]: salida anterior (componente de baja frecuencia)\n")
    f.write("  - x[n]: entrada actual (componente de alta frecuencia)\n\n")
    f.write("INTERPRETACION DE ALPHA:\n")
    f.write("  - alpha cercano a 1: Más suave, más retardo, elimina más ruido\n")
    f.write("  - alpha cercano a 0: Menos suave, respuesta rápida, sigue más la señal\n\n")
    
    f.write("="*70 + "\n")
    f.write("RESULTADOS\n")
    f.write("="*70 + "\n\n")
    
    f.write(f"MEJOR ALPHA ENCONTRADO: {mejor_alpha}\n\n")
    
    f.write("COMPARACION DE ERRORES:\n")
    f.write(f"  Filtro de Kalman:\n")
    f.write(f"    MAE  = {mae_kalman:.4f}\n")
    f.write(f"    MSE  = {mse_kalman:.4f}\n")
    f.write(f"    RMSE = {rmse_kalman:.4f}\n\n")
    
    f.write(f"  Filtro Complementario (alpha={mejor_alpha}):\n")
    f.write(f"    MAE  = {mae_complementario:.4f}\n")
    f.write(f"    MSE  = {mse_complementario:.4f}\n")
    f.write(f"    RMSE = {rmse_complementario:.4f}\n\n")
    
    f.write(f"MEJORA: {mejora_porcentual:.2f}% en MSE\n\n")
    
    f.write("PRUEBAS REALIZADAS CON DIFERENTES ALPHAS:\n")
    for alpha, mse in zip(alphas, errores_mse):
        f.write(f"  alpha = {alpha:.2f} | MSE = {mse:.4f}\n")
    
    f.write("\n" + "="*70 + "\n")
    f.write("CONCLUSION\n")
    f.write("="*70 + "\n")
    f.write(f"El filtro complementario con alpha={mejor_alpha} proporciona una mejor\n")
    f.write("suavización de la curva predicha por el Filtro de Kalman, reduciendo\n")
    f.write(f"el error en un {mejora_porcentual:.2f}% y mejorando la precisión de la estimación.\n")

print("\nReporte guardado en: filtro_complementario_reporte.txt")

print("\n" + "="*70)
print("PARTE 8 COMPLETADA - TODOS LOS ARCHIVOS GENERADOS")
print("="*70)
print("\nArchivos generados:")
print("  1. filtro_complementario_comparacion.png")
print("  2. filtro_complementario_resultado_completo.png")
print("  3. filtro_complementario_zoom_50s.png")
print("  4. filtro_complementario_mse_vs_alpha.png")
print("  5. filtro_complementario_reporte.txt")
print("="*70)