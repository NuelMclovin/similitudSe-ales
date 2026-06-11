# src/utils.py
import numpy as np
from scipy.io import wavfile
from pathlib import Path

def cargar_y_normalizar(ruta: Path):
    """Carga un archivo de audio WAV, lo convierte a mono y normaliza su amplitud."""
    fs, data = wavfile.read(str(ruta))
    
    # Convertir estéreo a mono si es necesario
    if len(data.shape) > 1: 
        data = data.mean(axis=1)
        
    # Normalización [-1, 1]
    max_amp = np.max(np.abs(data))
    if max_amp > 0: 
        data = data / max_amp
        
    return fs, data

def alinear_tamanos(a1, a2):
    """Añade ceros (padding) al audio más corto para que ambos arreglos midan lo mismo."""
    if len(a1) > len(a2): 
        a2 = np.pad(a2, (0, len(a1) - len(a2)), 'constant')
    elif len(a2) > len(a1): 
        a1 = np.pad(a1, (0, len(a2) - len(a1)), 'constant')
    return a1, a2