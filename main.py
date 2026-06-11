import os
import sys
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk
from tkinter import ttk, messagebox

# =============================================================================
# CONFIGURACIÓN Y RUTAS
# =============================================================================
MESSI_REAL = r"D:\t.escom\Elem escuela\5 Semestre\Señales\Proyecto_Similitud_Audio\MeesiBueno.wav"
MESSI_IA = r"D:\t.escom\Elem escuela\5 Semestre\Señales\Proyecto_Similitud_Audio\MessiIA.wav"
DIRECTORIO_PROYECTO = r"D:\t.escom\Elem escuela\5 Semestre\Señales\Proyecto_Similitud_Audio"
sys.path.append(DIRECTORIO_PROYECTO)

try:
    from fourier_math import coeficientes_serie_discreta_audio
    from Transformada_fourier_math import espectro_fft_audio
    from conv_math import correlacion_discreta_audio
except ImportError as e:
    print(f"Error importando módulos: {e}")
    sys.exit(1)

# =============================================================================
# LÓGICA DE PROCESAMIENTO MATEMÁTICO
# =============================================================================
def cargar_y_normalizar(ruta):
    fs, data = wavfile.read(ruta)
    if len(data.shape) > 1: data = data.mean(axis=1)
    max_amp = np.max(np.abs(data))
    if max_amp > 0: data = data / max_amp
    return fs, data

def alinear_tamanos(a1, a2):
    if len(a1) > len(a2): a2 = np.pad(a2, (0, len(a1) - len(a2)), 'constant')
    elif len(a2) > len(a1): a1 = np.pad(a1, (0, len(a2) - len(a1)), 'constant')
    return a1, a2

# =============================================================================
# CLASE PRINCIPAL DE LA INTERFAZ (GUI)
# =============================================================================
class AnalizadorAudioApp:
    def __init__(self, root, ruta_real, ruta_ia, tam_ventana=15.0):
        self.root = root
        self.root.title("Sistema de Análisis y Detección de Audio Clonado - ESCOM")
        self.root.geometry("1400x850")
        
        self.tam_ventana = tam_ventana
        
        # Variables para almacenar datos procesados
        self.fs = 0
        self.a_real = None
        self.a_ia = None
        self.resultados_ventanas = []
        
        self.cargar_datos(ruta_real, ruta_ia)
        self.construir_interfaz()

    def cargar_datos(self, r_real, r_ia):
        if not os.path.exists(r_real) or not os.path.exists(r_ia):
            messagebox.showerror("Error", "No se encontraron los audios en la ruta especificada.")
            sys.exit(1)
            
        fs1, a1 = cargar_y_normalizar(r_real)
        fs2, a2 = cargar_y_normalizar(r_ia)
        self.fs = fs1
        self.a_real, self.a_ia = alinear_tamanos(a1, a2)
        
        self.duracion_total = len(self.a_real) / self.fs
        self.num_ventanas = int(self.duracion_total // self.tam_ventana) or 1
        
        # Pre-procesar ventanas para la tabla y gráficas globales
        self.prom_mag_real = 0
        self.prom_mag_ia = 0
        self.eje_freqs = None
        
        for i in range(self.num_ventanas):
            inicio_idx = int(i * self.tam_ventana * self.fs)
            fin_idx = int((i + 1) * self.tam_ventana * self.fs)
            seg1 = self.a_real[inicio_idx:fin_idx]
            seg2 = self.a_ia[inicio_idx:fin_idx]
            
            # FFT
            freqs, mag1 = espectro_fft_audio(seg1, self.fs)
            _, mag2 = espectro_fft_audio(seg2, self.fs)
            if self.eje_freqs is None: self.eje_freqs = freqs
            self.prom_mag_real += mag1
            self.prom_mag_ia += mag2
            
            # Correlación y MSE
            _, _, max_corr = correlacion_discreta_audio(seg1, seg2)
            _, _, max_auto = correlacion_discreta_audio(seg1, seg1)
            sim_pct = (max_corr / max_auto) * 100 if max_auto > 0 else 0
            mse = np.mean((mag1 - mag2)**2)
            
            # Veredicto de coincidencia
            estado = "Alta Coincidencia" if sim_pct > 70 else ("Similitud Media" if sim_pct > 30 else "Desfase / IA Alterada")
            
            self.resultados_ventanas.append({
                "ventana": i + 1, "inicio": i * self.tam_ventana, "fin": (i+1) * self.tam_ventana,
                "similitud": sim_pct, "mse": mse, "estado": estado
            })
            
        self.prom_mag_real /= self.num_ventanas
        self.prom_mag_ia /= self.num_ventanas

    def construir_interfaz(self):
        # Crear sistema de pestañas
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.tab_global = ttk.Frame(self.notebook)
        self.tab_individual = ttk.Frame(self.notebook)
        self.tab_tabla = ttk.Frame(self.notebook)
        
        self.notebook.add(self.tab_global, text="1. Dashboard Comparativo Global")
        self.notebook.add(self.tab_individual, text="2. Análisis Individual y Muestreo")
        self.notebook.add(self.tab_tabla, text="3. Tabla de Coincidencias")
        
        self.construir_tab_global()
        self.construir_tab_individual()
        self.construir_tab_tabla()

    # ==================== PESTAÑA 1: GLOBAL ====================
    def construir_tab_global(self):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={'height_ratios': [2, 1]})
        fig.patch.set_facecolor('#f0f0f0')
        
        # Espectro Promedio
        ax1.plot(self.eje_freqs, self.prom_mag_real, label="Voz Original", color='#004aad')
        ax1.plot(self.eje_freqs, self.prom_mag_ia, label="Voz IA", color='#d90429', alpha=0.7)
        ax1.set_xlim(0, 8000)
        ax1.set_title("Firma Tonal Promedio (Transformada de Fourier)")
        ax1.set_ylabel("Magnitud")
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Similitud Temporal (Barras)
        ventanas_x = [res["ventana"] for res in self.resultados_ventanas]
        similitudes = [res["similitud"] for res in self.resultados_ventanas]
        ax2.bar(ventanas_x, similitudes, color='#2FA572')
        ax2.axhline(50, color='red', linestyle='--', label="Umbral Crítico")
        ax2.set_title("Coherencia Temporal por Ventana de 15s")
        ax2.set_ylabel("Similitud (%)")
        ax2.set_xticks(ventanas_x)
        ax2.legend()
        
        plt.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.tab_global)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # ==================== PESTAÑA 2: INDIVIDUAL Y MUESTREO ====================
    def construir_tab_individual(self):
        # Panel de Controles
        ctrl_frame = ttk.Frame(self.tab_individual, padding=10)
        ctrl_frame.pack(side=tk.TOP, fill=tk.X)
        
        ttk.Label(ctrl_frame, text="Seleccionar Señal:").pack(side=tk.LEFT, padx=5)
        self.senal_var = tk.StringVar(value="Original")
        ttk.Combobox(ctrl_frame, textvariable=self.senal_var, values=["Original", "IA Clon"], state="readonly").pack(side=tk.LEFT, padx=5)
        
        ttk.Label(ctrl_frame, text="Factor de Sub-Muestreo (Decimación):").pack(side=tk.LEFT, padx=(20, 5))
        self.muestreo_var = tk.IntVar(value=100) # Tomar 1 de cada 100 muestras
        ttk.Scale(ctrl_frame, from_=10, to=500, variable=self.muestreo_var, orient=tk.HORIZONTAL, length=200).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(ctrl_frame, text="Actualizar Análisis", command=self.actualizar_grafico_individual).pack(side=tk.LEFT, padx=20)
        
        # Área de Gráficas
        self.fig_indiv, (self.ax_onda, self.ax_armonicos) = plt.subplots(1, 2, figsize=(10, 4))
        self.fig_indiv.patch.set_facecolor('#f0f0f0')
        self.canvas_indiv = FigureCanvasTkAgg(self.fig_indiv, master=self.tab_individual)
        self.canvas_indiv.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.actualizar_grafico_individual() # Primera carga

    def actualizar_grafico_individual(self):
        self.ax_onda.clear()
        self.ax_armonicos.clear()
        
        # Extraer solo 0.1 segundos (100 ms) del medio del audio para que se vea la onda clara
        centro = int((self.duracion_total / 2) * self.fs)
        fin = centro + int(0.1 * self.fs)
        
        datos = self.a_real[centro:fin] if self.senal_var.get() == "Original" else self.a_ia[centro:fin]
        color = '#004aad' if self.senal_var.get() == "Original" else '#d90429'
        
        # 1. Gráfica de Muestreo (Stem Plot)
        factor = self.muestreo_var.get()
        datos_muestreados = datos[::factor]
        tiempo_muestreado = np.arange(len(datos_muestreados)) * factor
        
        self.ax_onda.plot(datos, color='gray', alpha=0.3, label="Señal Continua")
        self.ax_onda.stem(tiempo_muestreado, datos_muestreados, linefmt=color, markerfmt='o', basefmt=" ", label=f"Muestreo (1 cada {factor})")
        self.ax_onda.set_title("Visualización de Muestreo (100 ms)")
        self.ax_onda.legend()
        
        # 2. Extracción de Armónicos (Serie Trigonométrica)
        # Extraemos armónicos de ese pequeño frame
        a0, list_a, list_b = coeficientes_serie_discreta_audio(datos, n_max=10)
        magnitudes_armonicos = [np.sqrt(a**2 + b**2) for a, b in zip(list_a, list_b)]
        
        armonicos_x = np.arange(1, 11)
        self.ax_armonicos.bar(armonicos_x, magnitudes_armonicos, color=color)
        self.ax_armonicos.set_title("Primeros 10 Armónicos (Serie Trig. Fourier)")
        self.ax_armonicos.set_xlabel("Número de Armónico (n)")
        self.ax_armonicos.set_xticks(armonicos_x)
        
        self.fig_indiv.tight_layout()
        self.canvas_indiv.draw()

    # ==================== PESTAÑA 3: TABLA DE DATOS ====================
    def construir_tab_tabla(self):
        columnas = ("Ventana", "Tiempo", "Similitud Temporal", "Error MSE (Espectro)", "Veredicto del Sistema")
        self.tree = ttk.Treeview(self.tab_tabla, columns=columnas, show="headings", height=15)
        
        # Configurar cabeceras
        for col in columnas:
            self.tree.heading(col, text=col)
            self.tree.column(col, anchor=tk.CENTER, width=150)
        self.tree.column("Veredicto del Sistema", width=250)
            
        # Insertar datos
        for res in self.resultados_ventanas:
            tiempo_str = f"{res['inicio']:.1f}s - {res['fin']:.1f}s"
            sim_str = f"{res['similitud']:.2f} %"
            mse_str = f"{res['mse']:.4e}"
            self.tree.insert("", tk.END, values=(res["ventana"], tiempo_str, sim_str, mse_str, res["estado"]))
            
        # Scrollbar para la tabla
        scrollbar = ttk.Scrollbar(self.tab_tabla, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscroll=scrollbar.set)
        
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)


# =============================================================================
# EJECUCIÓN
# =============================================================================
if __name__ == "__main__":
    root = tk.Tk()
    # Aplicar un estilo más moderno a la interfaz
    style = ttk.Style(root)
    style.theme_use('clam') 
    
    app = AnalizadorAudioApp(root, MESSI_REAL, MESSI_IA, tam_ventana=15.0)
    root.mainloop()