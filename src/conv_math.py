# conv_math.py
# Logica matematica pura de la convolucion numerica y correlacion discreta.

import math
import numpy as np
from scipy.integrate import quad
from scipy import signal # Añadido para procesamiento de señales discretas

# ---------------------------------------------------------------------------
# Funciones de senal auxiliares (MUNDO CONTINUO)
# ---------------------------------------------------------------------------

def _step(t):
    """Escalon unitario u(t): 1 si t >= 0, 0 si t < 0."""
    return 1.0 if t >= 0.0 else 0.0

def _rect(t):
    """Pulso rectangular centrado: 1 si |t| <= 0.5, 0 en otro caso."""
    return 1.0 if abs(t) <= 0.5 else 0.0

def _sqw(t, T=2 * math.pi):
    """Onda cuadrada de periodo T: +1 en la primera mitad, -1 en la segunda."""
    return 1.0 if (t % T) < T / 2 else -1.0

def _tri(t, T=2 * math.pi):
    """Onda triangular de periodo T entre -1 y +1."""
    phase = (t % T) / T          # fase normalizada entre 0 y 1
    return 4.0 * abs(phase - 0.5) - 1.0

def _ramp(t):
    """Rampa unitaria r(t): t si t >= 0, 0 si t < 0."""
    return float(t) if t >= 0.0 else 0.0

# Diccionario de nombres disponibles en las expresiones del usuario
SAFE_NAMES: dict = {
    "sin": np.sin,    "cos": np.cos,    "tan": np.tan,
    "exp": np.exp,    "log": np.log,    "sqrt": np.sqrt,
    "abs": np.abs,    "pi":  np.pi,     "e":   np.e,
    "arcsin": np.arcsin, "arccos": np.arccos, "arctan": np.arctan,
    # senales predefinidas
    "step": _step,    "rect": _rect,
    "sqw":  _sqw,     "tri":  _tri,    "ramp": _ramp,
}

# ---------------------------------------------------------------------------
# Construccion de funcion a partir de una cadena de texto
# ---------------------------------------------------------------------------

def construir_funcion(expr: str, T: float = 1.0):
    """
    Convierte la cadena  expr  en una funcion f(t) evaluable.
    """
    def f(t_val):
        local = {"t": t_val, "T": T, **SAFE_NAMES}
        return float(eval(expr, {"__builtins__": {}}, local))
    f(0.0)
    return f

# ---------------------------------------------------------------------------
# Convolucion numerica (MUNDO CONTINUO)
# ---------------------------------------------------------------------------

def calcular_convolucion(
    expr_f: str,
    expr_g: str,
    t_min_f: float,
    t_max_f: float,
    t_min_g: float,
    t_max_g: float,
    N: int = 2000,
) -> dict:
    # [TU CÓDIGO ORIGINAL SE MANTIENE INTACTO AQUÍ]
    T_f = t_max_f - t_min_f
    T_g = t_max_g - t_min_g

    f_base = construir_funcion(expr_f, T_f)
    g_base = construir_funcion(expr_g, T_g)

    def f(t):
        return f_base(t) if (t_min_f <= t <= t_max_f) else 0.0

    def g(t):
        return g_base(t) if (t_min_g <= t <= t_max_g) else 0.0

    t_arr_f = np.linspace(t_min_f, t_max_f, N)
    t_arr_g = np.linspace(t_min_g, t_max_g, N)

    f_arr = np.array([f(t) for t in t_arr_f], dtype=float)
    g_arr = np.array([g(t) for t in t_arr_g], dtype=float)

    t_conv = np.linspace(t_min_f + t_min_g, t_max_f + t_max_g, N)
    result_arr = np.zeros(N, dtype=float)
    for i, tau in enumerate(t_conv):
        def integrando(s, tau=tau):
            return f(s) * g(tau - s)
        valor, _ = quad(integrando, t_min_f, t_max_f,
                        limit=200, epsabs=1e-5, epsrel=1e-5)
        result_arr[i] = valor
    op_label = "(f \u2605 g)(\u03c4)"   

    f_arr_conv = np.array([f(s) for s in t_conv], dtype=float)

    return {
        "t_arr_f":    t_arr_f,
        "t_arr_g":    t_arr_g,
        "f_arr":      f_arr,
        "g_arr":      g_arr,
        "f_arr_conv": f_arr_conv,
        "t_conv":     t_conv,
        "conv_arr":   result_arr,
        "t_min_f":    t_min_f,
        "t_max_f":    t_max_f,
        "t_min_g":    t_min_g,
        "t_max_g":    t_max_g,
        "expr_f":     expr_f,
        "expr_g":     expr_g,
        "op_label":   op_label,
        "f_callable": f,
        "g_callable": g,
    }

# ---------------------------------------------------------------------------
# NUEVO: Procesamiento para Señales Digitales de Audio (MUNDO DISCRETO)
# ---------------------------------------------------------------------------

def correlacion_discreta_audio(audio1: np.ndarray, audio2: np.ndarray):
    """
    Calcula la correlación cruzada de dos arreglos de audio digital.
    Utiliza el método FFT ('fft') que es extremadamente rápido para 
    arreglos grandes (como audios de miles de muestras).
    
    Retorna:
    - lags: Arreglo de desplazamientos (útil para graficar o encontrar el offset temporal).
    - correlacion: Arreglo resultante de la similitud temporal.
    - similitud_max: El valor máximo de la correlación (indicador de similitud).
    """
    # Aseguramos que sean arrays 1D
    a1 = np.asarray(audio1).flatten()
    a2 = np.asarray(audio2).flatten()
    
    # Calcular correlación (método fft es vital para no congelar la PC)
    correlacion = signal.correlate(a1, a2, mode='full', method='fft')
    
    # Calcular los lags (desplazamientos)
    lags = signal.correlation_lags(len(a1), len(a2), mode='full')
    
    similitud_max = np.max(correlacion)
    
    return lags, correlacion, similitud_max

