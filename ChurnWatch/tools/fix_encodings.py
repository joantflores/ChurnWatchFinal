from pathlib import Path
import sys

# Ruta raíz del proyecto (sube un nivel desde el archivo actual)
ROOT = Path(__file__).resolve().parent.parent

# Busca todos los archivos .py dentro del proyecto
py_files = list(ROOT.rglob('*.py'))

# Lista para almacenar los archivos convertidos
changed = []

for p in py_files:
    # Omitir carpetas de entorno virtual 
    if '\\.venv' in str(p) or '/.venv' in str(p):
        continue

    try:
        # Intentar leer el archivo como UTF-8
        # Si funciona, no es necesario modificarlo
        text = p.read_text(encoding='utf-8')
        continue

    except Exception:
        try:
            # Lee el archivo en binario
            raw = p.read_bytes()

            text = raw.decode('latin-1')

            # Guardar nuevamente usando UTF-8
            p.write_text(text, encoding='utf-8')

            # Registrar el archivo convertido
            changed.append(p.relative_to(ROOT))

        except Exception as e:

            # Mostrar error si no se pudo convertir
            print(f"Failed to fix {p}: {e}")

# Mostrar resumen de archivos modificados
print('Re-encoded files:')

for c in changed:
    print(' -', c)

print('\nDone.')

sys.exit(0)