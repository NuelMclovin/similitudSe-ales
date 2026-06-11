import numpy as np
import sympy as sp
from scipy.integrate import quad_vec

# =============================================================================
# 1. MUNDO DISCRETO (Para procesar tus archivos de Audio real)
# =============================================================================

def espectro_fft_audio(audio_array: np.ndarray, fs: int):
    """
    Calcula la Transformada Rápida de Fourier (FFT) de un arreglo digital discreto.
    
    Parámetros:
    - audio_array: Arreglo 1D de numpy con las muestras de audio.
    - fs: Frecuencia de muestreo en Hz (ej. 44100).
    
    Retorna:
    - freqs_positivas: Arreglo de frecuencias en Hz.
    - magnitud: Arreglo con la amplitud normalizada de cada frecuencia.
    """
    N = len(audio_array)
    
    # Calcular la FFT
    fft_resultado = np.fft.fft(audio_array)
    
    # Calcular el eje de frecuencias correspondiente
    freqs = np.fft.fftfreq(N, d=1/fs)
    
    # Por el Teorema de Nyquist y la simetría de señales reales, 
    # solo nos interesa la primera mitad del espectro (frecuencias positivas).
    mitad = N // 2
    freqs_positivas = freqs[:mitad]
    
    # Obtener magnitud absoluta y normalizar por el número de muestras
    magnitud = np.abs(fft_resultado[:mitad]) / N
    
    return freqs_positivas, magnitud


# =============================================================================
# 2. MUNDO CONTINUO - MÉTODO SIMBÓLICO (Para análisis matemático exacto)
# =============================================================================

def transformada_simbolica(func_str: str):
    """
    Intenta resolver la Transformada de Fourier analíticamente mediante integración.
    
    Parámetros:
    - func_str: String de la función en sintaxis Sympy (ej. "exp(-Abs(t))").
    
    Retorna:
    - X_w_simp: La expresión de Sympy resultante de la transformada.
    
    Lanza ValueError si la integral no tiene una solución cerrada.
    """
    t = sp.symbols('t', real=True)
    w = sp.symbols('w', real=True)
    
    # Convertir string a expresión matemática
    x_t = sp.sympify(func_str)
    
    # Ecuación fundamental: Integral de -inf a inf de x(t)*e^(-jwt) dt
    X_w = sp.integrate(x_t * sp.exp(-sp.I * w * t), (t, -sp.oo, sp.oo))
    
    # Verificar si sympy dejó la expresión como una "Integral" sin resolver
    if any(X_w.atoms(sp.Integral)):
        raise ValueError("No se encontró una solución analítica cerrada. Usa el método numérico.")
        
    return sp.simplify(X_w)


# =============================================================================
# 3. MUNDO CONTINUO - MÉTODO NUMÉRICO (Para funciones continuas sin sol. cerrada)
# =============================================================================

def transformada_numerica(func_str: str, w_axis: np.ndarray, t_limite: float = 40.0):
    """
    Calcula la Transformada de Fourier de una función continua evaluando la
    integral numéricamente para un conjunto de frecuencias dadas.
    
    Parámetros:
    - func_str: String de la función (ej. "sin(pi*t)/(pi*t)").
    - w_axis: Arreglo de numpy con las frecuencias omega (w) a evaluar.
    - t_limite: Límite de integración en el tiempo [-t_limite, t_limite].
    
    Retorna:
    - X_vals: Arreglo de numpy complejo con el resultado para cada w.
    """
    t = sp.symbols('t', real=True)
    w = sp.symbols('w', real=True)
    x_t = sp.sympify(func_str)
    
    # Módulos de soporte, incluyendo la función escalón de Heaviside
    modules = ['numpy', {'Heaviside': lambda x: np.heaviside(x, 1)}]
    
    # Construir el integrando: x(t) * e^(-jwt)
    integrand_expr = x_t * sp.exp(-sp.I * w * t)
    f_integrand = sp.lambdify((t, w), integrand_expr, modules=modules)
    
    # quad_vec permite integrar una función que devuelve un vector (nuestro w_axis)
    # limitamos a -t_limite, t_limite en lugar de infinito para evitar cuelgues.
    X_vals, error = quad_vec(
        lambda tv: f_integrand(tv, w_axis),
        -t_limite, t_limite, 
        epsabs=1e-4, epsrel=1e-4
    )
    
    return X_vals