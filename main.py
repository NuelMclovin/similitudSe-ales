# main.py
import tkinter as tk
from tkinter import ttk
import sys

# Importamos la configuración y la app visual
try:
    import config
    from gui.app import AnalizadorAudioApp
except ImportError as e:
    print(f"Error crítico al iniciar el sistema: {e}")
    print("Asegúrate de ejecutar main.py desde la raíz del proyecto.")
    sys.exit(1)

if __name__ == "__main__":
    # Inicializamos la ventana principal de Tkinter
    root = tk.Tk()
    
    # Aplicar un estilo moderno a la interfaz
    style = ttk.Style(root)
    style.theme_use('clam') 
    
    # Invocamos la clase pasando las rutas configuradas en config.py
    app = AnalizadorAudioApp(
        root=root, 
        ruta_real=config.AUDIO_ORIGINAL, 
        ruta_ia=config.AUDIO_IA, 
        tam_ventana=15.0
    )
    
    # Arrancar el bucle principal de la aplicación
    root.mainloop()