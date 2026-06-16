# src/detector.py
import numpy as np
from pathlib import Path
from scipy.spatial.distance import cosine
from scipy.signal import find_peaks
from src.utils import cargar_y_normalizar
from src.Transformada_fourier_math import espectro_fft_audio

REFERENCIAS = [
    {"archivo": "MeesiBueno.wav", "persona": "Messi",     "color": "#004aad"},
    {"archivo": "MessiIA.wav",    "persona": "Messi",     "color": "#5b8dd9"},
    {"archivo": "PodReal.wav",    "persona": "Cristiano", "color": "#c8102e"},
    {"archivo": "PodIA.wav",      "persona": "Cristiano", "color": "#e06080"},
]

FREQ_MIN     = 80
FREQ_MAX     = 8000
FREQ_F0_MIN  = 80     # rango de frecuencia fundamental de voz masculina
FREQ_F0_MAX  = 300
FREQ_FORM    = 3000   # formantes relevantes hasta 3kHz
N_FRAGMENTOS = 8
SEG_DURACION = 4


def _extraer_f0(audio: np.ndarray, fs: int) -> float:
    """Frecuencia fundamental dominante del audio (tono base de la voz)."""
    freqs, mag = espectro_fft_audio(audio, fs)
    mascara    = (freqs >= FREQ_F0_MIN) & (freqs <= FREQ_F0_MAX)
    if not np.any(mascara):
        return 0.0
    idx_pico = np.argmax(mag[mascara])
    return float(freqs[mascara][idx_pico])


def _huella_formantes(audio: np.ndarray, fs: int) -> np.ndarray:
    """
    Huella espectral enfocada en formantes (80–3000 Hz).
    Más discriminativa que el espectro completo para identificar voces.
    """
    freqs, mag = espectro_fft_audio(audio, fs)
    mascara    = (freqs >= FREQ_MIN) & (freqs <= FREQ_FORM)
    h          = mag[mascara].copy()
    norma      = np.linalg.norm(h)
    if norma > 0:
        h /= norma
    return h


def _huella_multi(audio: np.ndarray, fs: int) -> dict:
    """
    Calcula dos huellas promediadas sobre N_FRAGMENTOS del audio:
    - huella_formantes: espectro 80–3000 Hz normalizado
    - f0_medio: tono base promedio
    """
    muestras_seg = fs * SEG_DURACION
    total        = len(audio)

    if total <= muestras_seg:
        fragmentos = [audio]
    else:
        posiciones = np.linspace(muestras_seg // 2,
                                 total - muestras_seg // 2,
                                 N_FRAGMENTOS, dtype=int)
        fragmentos = [audio[p - muestras_seg // 2 : p + muestras_seg // 2]
                      for p in posiciones]

    huellas_form = []
    f0s          = []

    for frag in fragmentos:
        huellas_form.append(_huella_formantes(frag, fs))
        f0s.append(_extraer_f0(frag, fs))

    min_len      = min(len(h) for h in huellas_form)
    huella_prom  = np.mean([h[:min_len] for h in huellas_form], axis=0)
    f0_medio     = float(np.median(f0s))

    return {"huella": huella_prom, "f0": f0_medio}


def _sim_combinada(q: dict, ref: dict) -> float:
    """
    Similitud combinada: 70% coseno de formantes + 30% similitud de tono.
    El tono (f0) es muy estable por persona y ayuda a romper empates.
    """
    n       = min(len(q["huella"]), len(ref["huella"]))
    sim_cos = float(1.0 - cosine(q["huella"][:n], ref["huella"][:n]))

    # Similitud de f0: gaussiana — cae suavemente al alejarse del tono de referencia
    diff_f0 = abs(q["f0"] - ref["f0"])
    sim_f0  = float(np.exp(-(diff_f0 ** 2) / (2 * 20 ** 2)))  # sigma=20Hz

    return 0.70 * sim_cos + 0.30 * sim_f0


class DetectorVoz:
    def __init__(self, carpeta_audios: Path):
        self.carpeta     = carpeta_audios
        self.referencias = []
        self._cargar_referencias()

    def _cargar_referencias(self):
        for ref in REFERENCIAS:
            ruta = self.carpeta / ref["archivo"]
            if not ruta.exists():
                raise FileNotFoundError(f"No se encontró: {ruta}")
            fs, audio  = cargar_y_normalizar(ruta)
            datos      = _huella_multi(audio, fs)
            self.referencias.append({**ref, "fs": fs, **datos})

    def analizar(self, ruta: Path) -> dict:
        fs_q, audio_q = cargar_y_normalizar(ruta)
        query         = _huella_multi(audio_q, fs_q)

        # Similitud contra cada referencia
        scores = []
        for ref in self.referencias:
            sim = _sim_combinada(query, ref)
            scores.append({
                "persona":   ref["persona"],
                "archivo":   ref["archivo"],
                "color":     ref["color"],
                "similitud": sim,
                "f0_ref":    ref["f0"],
            })

        # Agrupar por persona: el score de la persona es el máximo de sus dos audios
        personas = {}
        for s in scores:
            p = s["persona"]
            if p not in personas or s["similitud"] > personas[p]["similitud"]:
                personas[p] = s

        ganador    = max(personas.values(), key=lambda x: x["similitud"])
        perdedor   = min(personas.values(), key=lambda x: x["similitud"])
        margen     = ganador["similitud"] - perdedor["similitud"]

        scores.sort(key=lambda x: x["similitud"], reverse=True)

        return {
            "persona":        ganador["persona"],
            "confianza":      ganador["similitud"],
            "margen":         margen,
            "f0_query":       query["f0"],
            "scores_persona": {p: v["similitud"] for p, v in personas.items()},
            "scores":         scores,
            "huella_query":   query["huella"],
            "nombre_archivo": ruta.name,
        }