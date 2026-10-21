"""Constantes globales: rutas, tema visual y valores por defecto."""
from pathlib import Path

DIR = Path(__file__).resolve().parent
RUTA_ACTIVO = DIR / "modelo_mnist.keras"      # modelo que usa la pestaña Reconocer
DIR_MODELOS = DIR / "modelos"                 # un archivo por entrenamiento
RUTA_EXPERIMENTOS = DIR / "experimentos.json"

TAM = 400        # lado del lienzo de dibujo
VISTA = 112      # lado de la vista previa 28x28

# ------------------------- Tema negro elegante (acento dorado) -------------------------
BG = "#09090b"        # fondo general (casi negro)
PANEL = "#131316"     # tarjetas / paneles
PANEL2 = "#1c1c21"    # campos, pistas, banners
TXT = "#ececf1"
MUTED = "#8e8e9a"
BORDE = "#2c2c34"
ACENTO = "#c9a45c"    # dorado champagne
ACENTO_HOVER = "#dbb878"
ACENTO_OFF = "#4a4130"
TXT_ACENTO = "#0b0b0d"  # texto sobre fondo dorado
VERDE = "#4ade80"
ROJO = "#f87171"
AMARILLO = "#fbbf24"
CIAN = "#7dd3fc"
ROSA = "#f0abfc"
FUENTE = "Segoe UI"

# ------------------------------ Opciones de UI ------------------------------
ARQUITECTURAS = ["MLP", "LeNet-5", "CNN simple", "CNN profunda"]
OPTIMIZADORES = ["Adam", "SGD", "SGD + Momentum", "RMSprop", "Nadam", "Adagrad"]
PROGRAMACIONES = ["Constante", "Decaimiento exponencial", "Coseno", "Reducir en meseta"]
BATCHES = ["32", "64", "128", "256", "512"]

# Valores iniciales del formulario de entrenamiento.
# Pensados para PROBAR rápido: 20 % de los datos y 3 épocas (unos segundos por modelo).
# Para un entrenamiento "serio" sube subset a 100 y epocas a 10-15.
DEFECTOS = {
    "arq": "CNN simple", "opt": "Adam", "lr": "0.001", "sched": "Constante",
    "dropout": 0.3, "l2": "0", "epocas": 3, "batch": "128", "val": 10,
    "subset": 20, "es": True, "paciencia": 2, "aug": False, "semilla": 42,
}

# nombre -> (clave de config, tipo, valores por defecto del barrido)
VARIABLES_BARRIDO = {
    "Tasa de aprendizaje": ("lr", float, "0.01, 0.001, 0.0001"),
    "Optimizador": ("opt", str, "Adam, SGD + Momentum, RMSprop"),
    "Arquitectura": ("arq", str, "MLP, LeNet-5, CNN simple"),
    "Dropout": ("dropout", float, "0.0, 0.3, 0.6"),
    "Tamaño de batch": ("batch", int, "64, 128, 256"),
}
