# gui/visualizaciones.py
#
# Módulo de visualizaciones para el Analizador de Similitud de Voz.
# Genera un dashboard con 4 gráficas y 4 métricas de resumen.
#
# Dependencias: matplotlib, numpy, scipy  (ya presentes en el proyecto)
# Uso desde tu AnalizadorAudioApp:
#
#   from gui.visualizaciones import PanelVisualizaciones
#   panel = PanelVisualizaciones(frame_padre, audio1, audio2, fs)
#   panel.pack(fill="both", expand=True)

import numpy as np
import tkinter as tk
from tkinter import ttk
from scipy.spatial.distance import cosine

import matplotlib
matplotlib.use("TkAgg")                          # backend para Tkinter
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Colores del dashboard (funcionan en fondo oscuro y claro)
COLOR_REAL  = "#7B6FE8"   # violeta  — audio real
COLOR_IA    = "#1DBF8B"   # teal     — audio IA
COLOR_PICO  = "#EF9F27"   # ámbar    — pico de correlación
COLOR_FG    = "#E0DED8"   # texto sobre fondo oscuro
COLOR_BG    = "#1A1A1F"   # fondo del dashboard
COLOR_CARD  = "#24242B"   # fondo de cada gráfica
COLOR_GRID  = "#2E2E38"   # líneas de cuadrícula


# ---------------------------------------------------------------------------
# Cálculo de métricas
# ---------------------------------------------------------------------------

def _similitud_correlacion(a1: np.ndarray, a2: np.ndarray) -> float:
    """
    Similitud por correlación cruzada normalizada.
    Valor entre 0 y 1 (1 = idénticos).
    """
    from scipy import signal
    corr = signal.correlate(a1, a2, mode="full", method="fft")
    # Normalización por la energía de ambas señales
    norma = np.sqrt(np.sum(a1 ** 2) * np.sum(a2 ** 2))
    if norma == 0:
        return 0.0
    return float(np.max(np.abs(corr)) / norma)


def _similitud_coseno_espectral(a1: np.ndarray, a2: np.ndarray) -> float:
    """
    Similitud del coseno entre los espectros FFT de ambos audios.
    Valor entre 0 y 1 (1 = espectros idénticos).
    """
    m1 = np.abs(np.fft.rfft(a1))
    m2 = np.abs(np.fft.rfft(a2))
    if np.linalg.norm(m1) == 0 or np.linalg.norm(m2) == 0:
        return 0.0
    return float(1.0 - cosine(m1, m2))


def _desfase_temporal_ms(a1: np.ndarray, a2: np.ndarray, fs: int) -> float:
    """
    Retorna el desfase temporal en milisegundos en el que la correlación
    cruzada es máxima (positivo = audio IA está adelantado).
    """
    from scipy import signal
    corr = signal.correlate(a1, a2, mode="full", method="fft")
    lags  = signal.correlation_lags(len(a1), len(a2), mode="full")
    lag_optimo = lags[np.argmax(np.abs(corr))]
    return float(lag_optimo / fs * 1000)


def _armonico_dominante(audio_frame: np.ndarray, n_max: int = 10) -> int:
    """
    Devuelve el índice n (1-based) del armónico con mayor magnitud
    en la Serie de Fourier discreta del fragmento dado.
    """
    N = len(audio_frame)
    if N == 0:
        return 1
    x = np.asarray(audio_frame).flatten()
    k = np.arange(N)
    w0 = 2 * np.pi / N
    magnitudes = []
    for n in range(1, n_max + 1):
        an = (2.0 / N) * np.sum(x * np.cos(n * w0 * k))
        bn = (2.0 / N) * np.sum(x * np.sin(n * w0 * k))
        magnitudes.append(np.sqrt(an**2 + bn**2))
    return int(np.argmax(magnitudes)) + 1


# ---------------------------------------------------------------------------
# Widget principal
# ---------------------------------------------------------------------------

class PanelVisualizaciones(tk.Frame):
    """
    Frame embebible en cualquier ventana Tkinter.
    Recibe los dos audios normalizados y la frecuencia de muestreo,
    y muestra un dashboard completo con 4 gráficas y 4 métricas.

    Parámetros
    ----------
    parent : widget Tkinter padre
    audio_real : np.ndarray  — señal normalizada del audio real
    audio_ia   : np.ndarray  — señal normalizada del audio IA
    fs         : int         — frecuencia de muestreo en Hz
    n_max      : int         — armónicos a mostrar en la serie (default 10)
    tam_frame  : int         — muestras del fragmento central para armónicos
    """

    def __init__(
        self,
        parent,
        audio_real: np.ndarray,
        audio_ia: np.ndarray,
        fs: int,
        n_max: int = 10,
        tam_frame: int = 2048,
        **kwargs,
    ):
        super().__init__(parent, bg=COLOR_BG, **kwargs)

        self.audio_real = audio_real
        self.audio_ia   = audio_ia
        self.fs         = fs
        self.n_max      = n_max

        # Fragmento central para el análisis de armónicos
        centro = len(audio_real) // 2
        mitad  = tam_frame // 2
        self.frame_real = audio_real[max(0, centro - mitad) : centro + mitad]
        self.frame_ia   = audio_ia[max(0, centro - mitad)   : centro + mitad]

        self._construir_ui()

    # ------------------------------------------------------------------
    # Construcción del layout
    # ------------------------------------------------------------------

    def _construir_ui(self):
        # ── Título ──────────────────────────────────────────────────
        tk.Label(
            self,
            text="Dashboard de Similitud de Voz",
            font=("Helvetica", 14, "bold"),
            fg=COLOR_FG,
            bg=COLOR_BG,
        ).pack(pady=(14, 0))

        tk.Label(
            self,
            text="Transformada de Fourier · Correlación cruzada · Serie armónica",
            font=("Helvetica", 9),
            fg="#888880",
            bg=COLOR_BG,
        ).pack(pady=(2, 10))

        # ── Tarjetas de métricas ────────────────────────────────────
        self._construir_metricas()

        # ── Gráficas matplotlib ─────────────────────────────────────
        self._construir_graficas()

        # ── Leyenda inferior ────────────────────────────────────────
        leyenda = tk.Frame(self, bg=COLOR_BG)
        leyenda.pack(pady=(6, 10))
        for color, texto in [(COLOR_REAL, "Audio real (MeesiBueno)"), (COLOR_IA, "Audio IA (MessiIA)")]:
            dot = tk.Canvas(leyenda, width=12, height=12, bg=COLOR_BG, highlightthickness=0)
            dot.pack(side="left", padx=(10, 4))
            dot.create_oval(2, 2, 10, 10, fill=color, outline="")
            tk.Label(leyenda, text=texto, font=("Helvetica", 9), fg="#888880", bg=COLOR_BG).pack(side="left")

    def _construir_metricas(self):
        """Calcula y muestra las 4 tarjetas de métricas de resumen."""
        sim_corr  = _similitud_correlacion(self.audio_real, self.audio_ia)
        sim_cos   = _similitud_coseno_espectral(self.audio_real, self.audio_ia)
        desfase   = _desfase_temporal_ms(self.audio_real, self.audio_ia, self.fs)
        armonico  = _armonico_dominante(self.frame_real, self.n_max)

        metricas = [
            ("Similitud correlación",  f"{sim_corr * 100:.1f}%",  self._color_similitud(sim_corr)),
            ("Similitud espectral",    f"{sim_cos  * 100:.1f}%",  self._color_similitud(sim_cos)),
            ("Desfase temporal",       f"{desfase:+.0f} ms",      COLOR_FG),
            ("Armónico dominante",     f"n = {armonico}",         COLOR_FG),
        ]

        fila = tk.Frame(self, bg=COLOR_BG)
        fila.pack(fill="x", padx=16, pady=(0, 10))

        for label, valor, color_valor in metricas:
            card = tk.Frame(fila, bg=COLOR_CARD, bd=0, relief="flat")
            card.pack(side="left", expand=True, fill="x", padx=5, pady=2, ipady=8)
            tk.Label(card, text=label, font=("Helvetica", 8), fg="#888880", bg=COLOR_CARD).pack()
            tk.Label(card, text=valor, font=("Helvetica", 18, "bold"), fg=color_valor, bg=COLOR_CARD).pack()

    @staticmethod
    def _color_similitud(valor: float) -> str:
        if valor >= 0.80:
            return "#1DBF8B"   # verde-teal: alta similitud
        if valor >= 0.50:
            return "#EF9F27"   # ámbar: similitud media
        return "#E24B4A"       # rojo: baja similitud

    # ------------------------------------------------------------------
    # Gráficas matplotlib
    # ------------------------------------------------------------------

    def _construir_graficas(self):
        """Crea la figura con GridSpec 2x2 y la embebe en Tkinter."""
        plt.style.use("dark_background")

        fig = plt.figure(figsize=(11, 7), facecolor=COLOR_BG)
        gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35,
                                left=0.07, right=0.97, top=0.93, bottom=0.08)

        ax_wave = fig.add_subplot(gs[0, :])   # fila 0, columnas 0 y 1
        ax_fft  = fig.add_subplot(gs[1, 0])
        ax_corr = fig.add_subplot(gs[1, 1])

        self._grafica_waveform(ax_wave)
        self._grafica_fft(ax_fft)
        self._grafica_correlacion(ax_corr)

        # Segunda figura para armónicos (ocuparía mucho en el mismo GridSpec)
        fig2 = plt.figure(figsize=(11, 3.2), facecolor=COLOR_BG)
        ax_harm = fig2.add_subplot(111)
        self._grafica_armonicos(ax_harm)
        fig2.subplots_adjust(left=0.07, right=0.97, top=0.85, bottom=0.15)

        for f in (fig, fig2):
            canvas = FigureCanvasTkAgg(f, master=self)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True, padx=12, pady=4)

    # ── Gráfica 1: Forma de onda ──────────────────────────────────

    def _grafica_waveform(self, ax):
        # Reducir puntos para velocidad de render (máx 4000 puntos)
        MAX_PTS = 4000
        r = max(1, len(self.audio_real) // MAX_PTS)
        t = np.arange(0, len(self.audio_real), r) / self.fs

        ax.plot(t, self.audio_real[::r], color=COLOR_REAL, lw=0.8,
                alpha=0.9, label="Real")
        ax.plot(t, self.audio_ia[::r],   color=COLOR_IA,   lw=0.8,
                alpha=0.8, label="IA", linestyle="--")

        self._estilo_ax(ax, "Forma de onda · Dominio del tiempo",
                        "Tiempo (s)", "Amplitud norm.")
        ax.set_ylim(-1.15, 1.15)

    # ── Gráfica 2: Espectro FFT ───────────────────────────────────

    def _grafica_fft(self, ax):
        from src.Transformada_fourier_math import espectro_fft_audio

        f1, m1 = espectro_fft_audio(self.audio_real, self.fs)
        f2, m2 = espectro_fft_audio(self.audio_ia,   self.fs)

        # Solo hasta 8 kHz (rango de voz)
        MAX_HZ = 8000
        mask1 = f1 <= MAX_HZ
        mask2 = f2 <= MAX_HZ

        ax.fill_between(f1[mask1], m1[mask1], alpha=0.35, color=COLOR_REAL)
        ax.fill_between(f2[mask2], m2[mask2], alpha=0.30, color=COLOR_IA)
        ax.plot(f1[mask1], m1[mask1], color=COLOR_REAL, lw=1.0)
        ax.plot(f2[mask2], m2[mask2], color=COLOR_IA,   lw=1.0)

        self._estilo_ax(ax, "Espectro FFT · Dominio de la frecuencia",
                        "Frecuencia (Hz)", "Magnitud norm.")

    # ── Gráfica 3: Correlación cruzada ───────────────────────────

    def _grafica_correlacion(self, ax):
        from scipy import signal
        from src.conv_math import correlacion_discreta_audio

        lags, corr, _ = correlacion_discreta_audio(self.audio_real, self.audio_ia)

        # Normalizar para que el pico sea 1
        corr_norm = corr / (np.max(np.abs(corr)) + 1e-12)

        # Reducir puntos
        MAX_PTS = 3000
        r = max(1, len(lags) // MAX_PTS)
        lags_s = lags[::r] / self.fs * 1000   # lags en ms

        colores = np.where(corr_norm[::r] == np.max(corr_norm[::r]),
                           COLOR_PICO, COLOR_REAL + "88")

        ax.bar(lags_s, corr_norm[::r], width=lags_s[1] - lags_s[0] if len(lags_s) > 1 else 1,
               color=colores, edgecolor="none")

        # Marcar el pico
        pico_lag = lags[np.argmax(corr_norm)] / self.fs * 1000
        pico_val = np.max(corr_norm)
        ax.axvline(pico_lag, color=COLOR_PICO, lw=1.2, linestyle="--", alpha=0.8)
        ax.text(pico_lag + 5, pico_val * 0.92,
                f"lag = {pico_lag:+.0f} ms",
                color=COLOR_PICO, fontsize=8)

        self._estilo_ax(ax, "Correlación cruzada · Similitud temporal",
                        "Desfase (ms)", "Correlación norm.")

    # ── Gráfica 4: Armónicos ──────────────────────────────────────

    def _grafica_armonicos(self, ax):
        from src.fourier_math import coeficientes_serie_discreta_audio

        _, a_r, b_r = coeficientes_serie_discreta_audio(self.frame_real, self.n_max)
        _, a_i, b_i = coeficientes_serie_discreta_audio(self.frame_ia,   self.n_max)

        mag_r = [np.sqrt(a**2 + b**2) for a, b in zip(a_r, b_r)]
        mag_i = [np.sqrt(a**2 + b**2) for a, b in zip(a_i, b_i)]

        ns = np.arange(1, self.n_max + 1)
        ancho = 0.35

        ax.bar(ns - ancho/2, mag_r, width=ancho, color=COLOR_REAL,
               alpha=0.85, label="Real")
        ax.bar(ns + ancho/2, mag_i, width=ancho, color=COLOR_IA,
               alpha=0.85, label="IA")

        ax.set_xticks(ns)
        ax.set_xticklabels([f"n={n}" for n in ns], fontsize=8)
        ax.legend(fontsize=8, framealpha=0.2)

        self._estilo_ax(ax, "Magnitud de armónicos · Serie de Fourier discreta",
                        "Armónico", "|aₙ² + bₙ²|^½")



    @staticmethod
    def _estilo_ax(ax, titulo: str, xlabel: str, ylabel: str):
        ax.set_facecolor(COLOR_CARD)
        ax.set_title(titulo, fontsize=10, fontweight="bold",
                     color=COLOR_FG, pad=8)
        ax.set_xlabel(xlabel, fontsize=8, color="#888880")
        ax.set_ylabel(ylabel, fontsize=8, color="#888880")
        ax.tick_params(colors="#888880", labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#2E2E38")
        ax.grid(True, color=COLOR_GRID, linewidth=0.5, linestyle="--", alpha=0.7)
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)



if __name__ == "__main__":
    import sys, os
    # Añadir la raíz del proyecto al path
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from src.utils import cargar_y_normalizar, alinear_tamanos
    import config

    fs1, a1 = cargar_y_normalizar(config.AUDIO_ORIGINAL)
    fs2, a2 = cargar_y_normalizar(config.AUDIO_IA)
    a1, a2  = alinear_tamanos(a1, a2)

    root = tk.Tk()
    root.title("Dashboard — Similitud de Voz con Fourier")
    root.configure(bg=COLOR_BG)
    root.geometry("1100x900")

    panel = PanelVisualizaciones(root, a1, a2, fs1)
    panel.pack(fill="both", expand=True)

    root.mainloop()