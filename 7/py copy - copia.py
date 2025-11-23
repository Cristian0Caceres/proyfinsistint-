import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# Configuración y generación de datos
# =============================================================================
dt = 0.1
t = np.arange(0, 100, dt)  # Cambiado de 100 a 1000 segundos
pos_real = 0.1 * (3*t - t**2)

np.random.seed(42)
ruido_medicion = 2.0
mediciones = pos_real + np.random.normal(0, ruido_medicion, len(pos_real))

# =============================================================================
# Filtro de Kalman 1D
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
        # Predicción
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        # Actualización
        y = measurement - (self.H @ self.x)[0, 0]
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T / S[0, 0]
        self.x = self.x + K * y
        self.P = (np.eye(2) - K @ self.H) @ self.P
        
        self.history_covariance.append(np.sqrt(self.P[0, 0]))
        return self.x[0, 0], self.x[1, 0]

# =============================================================================
# Ejecutar Filtro de Kalman
# =============================================================================
kf = KalmanFilter1D(
    dt, 
    process_variance=0.1, 
    measurement_variance=ruido_medicion**2,
    initial_position=mediciones[0], 
    initial_velocity=(mediciones[1] - mediciones[0]) / dt
)

pos_est, vel_est = zip(*[kf.process(m) for m in mediciones])
pos_est = np.array(pos_est)
cov = np.array(kf.history_covariance)

# =============================================================================
# Función auxiliar para gráficos
# =============================================================================
def plot_kalman(t_data, pos_real_data, mediciones_data, pos_est_data, 
                cov_data=None, title='', zoom=False):
    plt.figure(figsize=(14, 6))
    plt.plot(t_data, pos_real_data, 'g-', linewidth=2.5, label='Posición Real')
    plt.plot(t_data, mediciones_data, 'r-', alpha=0.4 if not zoom else 0.5, 
             linewidth=1.2, label='Mediciones con Ruido' if not zoom else 'Mediciones')
    plt.plot(t_data, pos_est_data, 'b-', linewidth=2, label='Estimación Kalman')
    
    if cov_data is not None:
        plt.fill_between(t_data, pos_est_data - 2*cov_data, 
                         pos_est_data + 2*cov_data,
                         color='blue', alpha=0.15, label='Intervalo ±2σ')
    
    plt.xlabel('Tiempo (s)', fontsize=14)
    plt.ylabel('Posición (m)', fontsize=14)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# =============================================================================
# Gráficos
# =============================================================================
# Trayectoria real
plt.figure(figsize=(12, 6))
plt.plot(t, pos_real, 'g-', linewidth=2, label='Posición Real')
plt.xlabel('Tiempo (s)', fontsize=18)
plt.ylabel('Posición (m)', fontsize=18)
plt.title('Trayectoria Real de la Partícula', fontsize=16)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=14)
plt.tight_layout()
plt.show()

# Filtro de Kalman completo
plot_kalman(t, pos_real, mediciones, pos_est, 
            title='Filtro de Kalman: Predicción de Posición')

# Zoom 0-20 segundos
zoom_idx = int(20/dt)  # Corregido para mostrar 20 segundos
plot_kalman(t[:zoom_idx], pos_real[:zoom_idx], mediciones[:zoom_idx], 
            pos_est[:zoom_idx], cov[:zoom_idx],
            title='Zoom: Primeros 20 Segundos', zoom=True)