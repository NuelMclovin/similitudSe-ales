# gui/tab_detector.py
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from src.detector import DetectorVoz, REFERENCIAS
from src.Transformada_fourier_math import espectro_fft_audio
from src.utils import cargar_y_normalizar

COLOR_BG   = "#1A1A1F"
COLOR_CARD = "#24242B"
COLOR_FG   = "#E0DED8"
COLOR_SUB  = "#888880"
COLOR_GRID = "#2E2E38"
COLOR_BAR  = "#7B6FE8"
COLOR_WIN  = "#EF9F27"
COLOR_MESSI     = "#004aad"
COLOR_CRISTIANO = "#c8102e"


class TabDetector(tk.Frame):
    def __init__(self, parent, carpeta_audios: Path, **kwargs):
        super().__init__(parent, bg=COLOR_BG, **kwargs)
        self.carpeta_audios = carpeta_audios
        self.detector       = DetectorVoz(carpeta_audios)
        self.resultado      = None
        self._construir_ui()

    def _construir_ui(self):
        header = tk.Frame(self, bg=COLOR_BG)
        header.pack(fill="x", padx=20, pady=(16, 0))
        tk.Label(header, text="Detector de Voz",
                 font=("Helvetica", 15, "bold"), fg=COLOR_FG, bg=COLOR_BG).pack(side="left")
        tk.Label(header,
                 text="Selecciona un audio · el sistema identifica quién habla usando frecuencia fundamental y formantes",
                 font=("Helvetica", 9), fg=COLOR_SUB, bg=COLOR_BG).pack(side="left", padx=12)

        btn_frame = tk.Frame(self, bg=COLOR_BG)
        btn_frame.pack(pady=14)
        tk.Button(btn_frame, text="📂  Seleccionar audio",
                  font=("Helvetica", 11, "bold"),
                  bg="#7B6FE8", fg="white",
                  activebackground="#5a54c4", activeforeground="white",
                  relief="flat", padx=20, pady=8, cursor="hand2",
                  command=self._seleccionar_y_analizar).pack()
        self.lbl_archivo = tk.Label(btn_frame, text="Ningún archivo seleccionado",
                                    font=("Helvetica", 9), fg=COLOR_SUB, bg=COLOR_BG)
        self.lbl_archivo.pack(pady=(6, 0))

        self.frame_resultados = tk.Frame(self, bg=COLOR_BG)
        self.frame_resultados.pack(fill="both", expand=True, padx=20, pady=(0, 16))

        # Veredicto
        self.frame_veredicto = tk.Frame(self.frame_resultados, bg=COLOR_BG)
        self.frame_veredicto.pack(fill="x", pady=(0, 10))

        bloque = tk.Frame(self.frame_veredicto, bg=COLOR_CARD, padx=24, pady=14)
        bloque.pack(side="left")
        tk.Label(bloque, text="PERSONA DETECTADA",
                 font=("Helvetica", 9), fg=COLOR_SUB, bg=COLOR_CARD).pack()
        self.lbl_persona = tk.Label(bloque, text="—",
                                    font=("Helvetica", 36, "bold"),
                                    fg=COLOR_FG, bg=COLOR_CARD)
        self.lbl_persona.pack()
        self.lbl_detalle = tk.Label(bloque, text="",
                                    font=("Helvetica", 10), fg=COLOR_SUB, bg=COLOR_CARD)
        self.lbl_detalle.pack(pady=(4, 0))

        bloque_f0 = tk.Frame(self.frame_veredicto, bg=COLOR_CARD, padx=24, pady=14)
        bloque_f0.pack(side="left", padx=(10, 0))
        tk.Label(bloque_f0, text="TONO BASE (F0)",
                 font=("Helvetica", 9), fg=COLOR_SUB, bg=COLOR_CARD).pack()
        self.lbl_f0 = tk.Label(bloque_f0, text="—",
                                font=("Helvetica", 36, "bold"),
                                fg=COLOR_FG, bg=COLOR_CARD)
        self.lbl_f0.pack()
        tk.Label(bloque_f0, text="Hz detectados en el audio",
                 font=("Helvetica", 9), fg=COLOR_SUB, bg=COLOR_CARD).pack(pady=(4, 0))

        self.canvas_barra = tk.Canvas(self.frame_resultados, height=18,
                                      bg=COLOR_CARD, highlightthickness=0)
        self.canvas_barra.pack(fill="x", pady=(0, 12))

        self.frame_graficas = tk.Frame(self.frame_resultados, bg=COLOR_BG)
        self.frame_graficas.pack(fill="both", expand=True)

    def _seleccionar_y_analizar(self):
        ruta = filedialog.askopenfilename(
            title="Selecciona un audio para analizar",
            initialdir=str(self.carpeta_audios),
            filetypes=[("WAV", "*.wav"), ("Todos", "*.*")],
        )
        if not ruta:
            return
        ruta = Path(ruta)
        self.lbl_archivo.config(text=f"Analizando: {ruta.name} ...")
        self.update_idletasks()
        try:
            self.resultado = self.detector.analizar(ruta)
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.lbl_archivo.config(text="Error al procesar el archivo.")
            return
        self.lbl_archivo.config(text=f"Archivo analizado: {ruta.name}")
        self._mostrar_resultado(ruta)

    def _mostrar_resultado(self, ruta: Path):
        r          = self.resultado
        color_win  = COLOR_MESSI if r["persona"] == "Messi" else COLOR_CRISTIANO

        self.lbl_persona.config(text=r["persona"], fg=color_win)
        self.lbl_detalle.config(
            text=f"Confianza: {r['confianza'] * 100:.1f}%  ·  Margen: {r['margen'] * 100:.1f}%"
        )
        self.lbl_f0.config(text=f"{r['f0_query']:.0f}", fg=color_win)

        self.canvas_barra.update_idletasks()
        ancho = self.canvas_barra.winfo_width()
        self.canvas_barra.delete("all")
        self.canvas_barra.create_rectangle(0, 0, ancho, 18, fill=COLOR_CARD, outline="")
        fill_w = int(ancho * min(r["confianza"], 1.0))
        self.canvas_barra.create_rectangle(0, 0, fill_w, 18, fill=color_win, outline="")
        self.canvas_barra.create_text(
            ancho // 2, 9,
            text=f"{r['confianza'] * 100:.1f}% confianza · margen sobre el otro: {r['margen'] * 100:.1f}%",
            fill="white", font=("Helvetica", 9, "bold"),
        )

        for widget in self.frame_graficas.winfo_children():
            widget.destroy()
        self._dibujar_graficas(ruta)

    def _dibujar_graficas(self, ruta: Path):
        plt.style.use("dark_background")
        fig = plt.figure(figsize=(13, 4.5), facecolor=COLOR_BG)
        gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38,
                                left=0.06, right=0.97, top=0.88, bottom=0.14)

        self._grafica_personas(fig.add_subplot(gs[0, 0]))
        self._grafica_f0(fig.add_subplot(gs[0, 1]))
        self._grafica_espectros(fig.add_subplot(gs[0, 2]), ruta)

        canvas = FigureCanvasTkAgg(fig, master=self.frame_graficas)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def _grafica_personas(self, ax):
        r        = self.resultado
        sp       = r["scores_persona"]
        personas = list(sp.keys())
        valores  = [sp[p] * 100 for p in personas]
        colores  = []
        for p in personas:
            if p == r["persona"]:
                colores.append(COLOR_MESSI if p == "Messi" else COLOR_CRISTIANO)
            else:
                colores.append(COLOR_BAR + "88")

        bars = ax.barh(personas, valores, color=colores, height=0.45)
        for bar, val in zip(bars, valores):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}%", va="center", color=COLOR_FG, fontsize=10, fontweight="bold")
        ax.set_xlim(0, 115)
        self._estilo_ax(ax, "Similitud por persona\n(formantes + tono)", "Similitud (%)", "")

    def _grafica_f0(self, ax):
        r = self.resultado

        # F0 de cada referencia
        refs_f0 = {}
        for ref in self.detector.referencias:
            p = ref["persona"]
            if p not in refs_f0:
                refs_f0[p] = []
            refs_f0[p].append(ref["f0"])
        f0_messi     = np.mean(refs_f0.get("Messi", [0]))
        f0_cristiano = np.mean(refs_f0.get("Cristiano", [0]))

        categorias = ["Messi\n(referencia)", "Cristiano\n(referencia)", "Audio\nanalizado"]
        valores    = [f0_messi, f0_cristiano, r["f0_query"]]
        colores    = [COLOR_MESSI, COLOR_CRISTIANO, COLOR_WIN]

        bars = ax.bar(categorias, valores, color=colores, width=0.5)
        for bar, val in zip(bars, valores):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                    f"{val:.0f} Hz", ha="center", color=COLOR_FG, fontsize=9, fontweight="bold")
        ax.set_ylim(0, max(valores) * 1.25)
        self._estilo_ax(ax, "Frecuencia fundamental (F0)\nTono base de cada voz",
                        "", "Frecuencia (Hz)")

    def _grafica_espectros(self, ax, ruta: Path):
        r = self.resultado

        fs_q, audio_q  = cargar_y_normalizar(ruta)
        freqs_q, mag_q = espectro_fft_audio(audio_q, fs_q)
        mask_q = (freqs_q >= 80) & (freqs_q <= 3000)

        color_win = COLOR_MESSI if r["persona"] == "Messi" else COLOR_CRISTIANO

        ref_info = next(x for x in REFERENCIAS if x["persona"] == r["persona"])
        fs_r, audio_r  = cargar_y_normalizar(self.carpeta_audios / ref_info["archivo"])
        freqs_r, mag_r = espectro_fft_audio(audio_r, fs_r)
        mask_r = (freqs_r >= 80) & (freqs_r <= 3000)

        ax.fill_between(freqs_q[mask_q], mag_q[mask_q], alpha=0.4, color=COLOR_WIN)
        ax.plot(freqs_q[mask_q], mag_q[mask_q], color=COLOR_WIN, lw=1.3, label="Analizado")
        ax.fill_between(freqs_r[mask_r], mag_r[mask_r], alpha=0.3, color=color_win)
        ax.plot(freqs_r[mask_r], mag_r[mask_r], color=color_win, lw=1.0,
                linestyle="--", label=f"Ref: {r['persona']}")
        ax.axvline(r["f0_query"], color=COLOR_WIN, lw=1.2, linestyle=":",
                   label=f"F0 detectado: {r['f0_query']:.0f} Hz")
        ax.legend(fontsize=8, framealpha=0.2)
        self._estilo_ax(ax, "Formantes vocálicas (80–3000 Hz)\nZona de identificación de voz",
                        "Frecuencia (Hz)", "Magnitud norm.")

    @staticmethod
    def _estilo_ax(ax, titulo, xlabel, ylabel):
        ax.set_facecolor(COLOR_CARD)
        ax.set_title(titulo, fontsize=9, fontweight="bold", color=COLOR_FG, pad=8)
        ax.set_xlabel(xlabel, fontsize=8, color=COLOR_SUB)
        ax.set_ylabel(ylabel, fontsize=8, color=COLOR_SUB)
        ax.tick_params(colors=COLOR_SUB, labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(COLOR_GRID)
        ax.grid(True, color=COLOR_GRID, linewidth=0.5, linestyle="--", alpha=0.7)