# fourier_math.py
import math
import numpy as np
from scipy.integrate import quad

# --- Funciones de señal comunes (MUNDO CONTINUO) --------------------------

def _step(t):
    return 1.0 if t >= 0.0 else 0.0

def _rect(t):
    return 1.0 if abs(t) <= 0.5 else 0.0

def _sqw(t, T=2 * math.pi):
    return 1.0 if (t % T) < T / 2 else -1.0

def _tri(t, T=2 * math.pi):
    phase = (t % T) / T
    return 4.0 * abs(phase - 0.5) - 1.0

def _ramp(t):
    return float(t) if t >= 0.0 else 0.0

SAFE_NAMES: dict = {
    "sin": np.sin,   "cos": np.cos,   "tan": np.tan,
    "exp": np.exp,   "log": np.log,   "sqrt": np.sqrt,
    "abs": np.abs,   "pi":  np.pi,    "e":   np.e,
    "arcsin": np.arcsin, "arccos": np.arccos, "arctan": np.arctan,
    "step": _step,   "rect": _rect,
    "sqw":  _sqw,    "tri":  _tri,    "ramp": _ramp,
}

def construir_funcion(expr: str, T: float):
    def f(t_val):
        local = {"t": t_val, "T": T, **SAFE_NAMES}
        return float(eval(expr, {"__builtins__": {}}, local))
    return f

# --- Cálculo Analítico (MUNDO CONTINUO) -----------------------------------

def calcular_coeficientes(f, t0: float, t1: float, w0: float, n_max: int):
    T = t1 - t0
    a0, _ = quad(f, t0, t1)
    a0 *= 2 / T
    a, b = [], []
    for n in range(1, n_max + 1):
        an, _ = quad(lambda t, n=n: f(t) * math.cos(n * w0 * t), t0, t1)
        bn, _ = quad(lambda t, n=n: f(t) * math.sin(n * w0 * t), t0, t1)
        a.append((2 / T) * an)
        b.append((2 / T) * bn)
    return a0, a, b

def reconstruir(t_arr, a0: float, a: list, b: list, w0: float, n_max: int):
    res = np.full_like(t_arr, a0 / 2, dtype=float)
    for n in range(1, n_max + 1):
        res += a[n - 1] * np.cos(n * w0 * t_arr)
        res += b[n - 1] * np.sin(n * w0 * t_arr)
    return res

# --- NUEVO: Cálculo Discreto para Audio (MUNDO DISCRETO) ------------------

def coeficientes_serie_discreta_audio(audio_frame: np.ndarray, n_max: int):
    """
    Calcula los coeficientes a_n y b_n de la Serie Trigonométrica de Fourier
    para un fragmento (frame) de audio digital, utilizando sumatorias discretas
    en lugar de integrales.
    
    Parámetros:
    - audio_frame: Un arreglo numpy pequeño (ej. 1024 o 2048 muestras extraídas
                   del centro de una vocal de la palabra).
    - n_max: Cantidad de armónicos a extraer.
    
    Retorna: a0, lista_a, lista_b
    """
    x = np.asarray(audio_frame).flatten()
    N = len(x)
    
    if N == 0:
        return 0.0, [], []

    # a0 es el promedio de la señal multiplicado por 2
    a0 = (2.0 / N) * np.sum(x)
    
    a = []
    b = []
    
    # Índices espaciales k
    k = np.arange(N)
    # Frecuencia fundamental discreta
    w0 = 2 * np.pi / N 
    
    for n in range(1, n_max + 1):
        # Implementación vectorizada de las sumatorias discretas
        an = (2.0 / N) * np.sum(x * np.cos(n * w0 * k))
        bn = (2.0 / N) * np.sum(x * np.sin(n * w0 * k))
        a.append(an)
        b.append(bn)
        
    return a0, a, b