from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

# Ruta datos
DATA_DIR = ROOT_DIR / 'data'
RAW_DATA_DIR = DATA_DIR / 'data.csv'

# Ruta modelos
MODEL_DIR = ROOT_DIR / 'models'
XGB_MODEL_DIR = MODEL_DIR / 'model.pkl'

# Comprobación de directorios y archivos
if not DATA_DIR.exists():
    print(f"Advertencia: El directorio de datos {DATA_DIR} no existe.")
if not MODEL_DIR.exists():
    print(f"Advertencia: El directorio de modelos {MODEL_DIR} no existe.")
if not RAW_DATA_DIR.exists():
    print(f"Advertencia: El archivo de datos {RAW_DATA_DIR} no existe.")
if not XGB_MODEL_DIR.exists():
    print(f"Advertencia: El archivo del modelo {XGB_MODEL_DIR} no existe.")