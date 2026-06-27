# Detector de Daños Viales — Mendoza

Aplicación web desarrollada con **Streamlit** y **YOLOv8/Ultralytics** para detectar daños viales en imágenes y sugerir una derivación automática al organismo correspondiente.

## Funcionalidades

- Subida de imagen `.jpg`, `.jpeg` o `.png`.
- Inferencia con modelo YOLO entrenado para detectar:
  - `Pothole` → bache.
  - `Crack` → grieta en calzada.
  - `Manhole` → tapa de alcantarillado / boca de acceso.
- Visualización de bounding boxes sobre la imagen.
- Tabla con clase detectada, confianza y coordenadas de cada caja.
- Extracción de ubicación GPS desde metadatos EXIF cuando la imagen los conserva.
- Geocodificación inversa: conversión de coordenadas GPS a dirección visible.
- Uso de OpenStreetMap/Nominatim sin API key y soporte opcional para Google Maps Geocoding API.
- Selección manual del departamento si la imagen no tiene GPS o si la detección automática no coincide.
- Derivación sugerida:
  - `Manhole` → Aguas Mendocinas.
  - `Pothole` o `Crack` → Municipalidad del departamento.
  - Departamento desconocido → Dirección Provincial de Vialidad como fallback.
- Generación de borrador de correo descargable en `.txt`.

## Estructura del repositorio

```text
root/
├── README.md
├── requirements.txt
├── runtime.txt
├── data/
│   └── link_dataset.txt
├── dev/
│   ├── notebooks/
│   │   └── entrenamiento.ipynb
│   └── modelo.pt              # modelo YOLO final incluido
└── prod/
    ├── app.py
    └── utils.py
```

## Modelo incluido

Esta versión ya incluye el modelo YOLO final en:

```text
dev/modelo.pt
```

No hay que descomprimir ni renombrar nada dentro de esta versión del ZIP.

## Ubicación y EXIF

La app lee las coordenadas GPS desde los metadatos EXIF de la imagen original. Para que funcione, la foto debe conservar esos metadatos. Muchas redes sociales y apps de mensajería pueden eliminarlos al comprimir o reenviar imágenes.

El flujo aplicado es:

```text
Imagen original → EXIF GPS → latitud/longitud → geocodificación inversa → dirección/departamento
```

Por defecto, la app usa **OpenStreetMap/Nominatim** mediante `requests`, por lo que no requiere instalar librerías adicionales ni configurar una API key. Sí requiere conexión a internet en el entorno donde corre Streamlit.

Opcionalmente, se puede configurar Google Maps Geocoding API en Streamlit Secrets:

```toml
GOOGLE_MAPS_API_KEY = "TU_API_KEY_DE_GOOGLE_MAPS"
```

Si se configura esa clave, la app intenta usar Google Maps primero. Si Google falla o no hay clave, usa OpenStreetMap/Nominatim como fallback.

Importante: la app conserva una copia de la imagen original para leer EXIF antes de convertirla a RGB, porque esa conversión puede eliminar los metadatos.

## Clases configuradas

El notebook final usa este orden de clases:

```python
0: "Pothole"
1: "Crack"
2: "Manhole"
```

La app usa ese mismo orden para interpretar las detecciones.

## Cómo correr localmente

Desde la raíz del proyecto:

```bash
python -m venv .venv
```

En Windows:

```bash
.venv\Scripts\activate
```

En Linux/macOS:

```bash
source .venv/bin/activate
```

Instalar dependencias:

```bash
pip install -r requirements.txt
```

Ejecutar la app:

```bash
streamlit run prod/app.py
```

## Si falta el modelo

La app está preparada para tres opciones:

1. **Modelo local**: colocar `dev/modelo.pt`.
2. **Modelo temporal**: subir el `.pt` desde la barra lateral de la app.
3. **Modelo remoto**: configurar `MODEL_URL` en Streamlit Cloud Secrets.

## Despliegue en Streamlit Cloud

1. Subir este repositorio a GitHub.
2. Entrar a Streamlit Cloud.
3. Crear una nueva app.
4. Elegir el repositorio y rama `main`.
5. En `Main file path`, colocar:

```text
prod/app.py
```

## Dataset

Dataset utilizado:

```text
Road Damage Dataset: Potholes, Cracks and Manholes
https://www.kaggle.com/datasets/lorenzoarcioni/road-damage-dataset-potholes-cracks-and-manholes
```

También está documentado en:

```text
data/link_dataset.txt
```

## Integrantes

Completar:

- Agustin Leyes — 49491
- Joaquin Giuliani — 50038
- Martin Beron — 49952
- Matias Almendros — 49926

## App desplegada

Completar luego del deploy:

```text
https://redes-neuronales-xqectxpgagmqvnzcux3aw8.streamlit.app/
```
