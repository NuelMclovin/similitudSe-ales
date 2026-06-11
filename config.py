# config.py
from pathlib import Path

# BASE_DIR calcula dinámicamente la ruta absoluta donde se encuentra este archivo config.py
BASE_DIR = Path(__file__).resolve().parent

# Carpetas principales del proyecto
AUDIOS_DIR = BASE_DIR / "audios"
SRC_DIR = BASE_DIR / "src"

# Rutas específicas de los archivos de audio (basado en los nombres de tu imagen)
AUDIO_ORIGINAL = AUDIOS_DIR / "MeesiBueno.wav"
AUDIO_IA = AUDIOS_DIR / "MessiIA.wav"