# ChurnWatch

## Cómo ejecutar el proyecto en Visual Studio 2022

### 1. Abrir el proyecto

1. Abrir Visual Studio 2022.
2. Seleccionar **File > Open > Folder**.
3. Elegir la carpeta `ChurnWatch`.

### 2. Crear o seleccionar entorno de Python

1. Ir a **View > Other Windows > Python Environments**.
2. Crear un entorno virtual o seleccionar uno existente.
3. Usar Python 3.10 o superior.

### 3. Instalar dependencias

Abrir la terminal dentro de Visual Studio, asegurarse de estar dentro de la carpeta `ChurnWatch` y ejecutar:

```bash
python -m pip install -r requirements.txt
```

4. Ejecutar la aplicación
Desde la misma terminal, ejecutar:

python app.py

5. Abrir la app
Cuando el servidor inicie, abrir en el navegador:

http://localhost:8050