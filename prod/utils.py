"""
Funciones auxiliares para la app Streamlit de detección de daños viales.

Modelo esperado: YOLOv8 entrenado con Ultralytics.
Clases del entrenamiento:
    0 -> Pothole
    1 -> Crack
    2 -> Manhole
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Optional
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests
from PIL import Image
from PIL.ExifTags import GPSTAGS, TAGS
from ultralytics import YOLO

# ─────────────────────────────────────────────────────────────
# RUTAS Y CONSTANTES DEL PROYECTO
# ─────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "dev" / "modelo.pt"
TMP_MODEL_PATH = Path("/tmp/modelo.pt")

# Orden real usado en el notebook final.
CLASS_NAMES: dict[int, str] = {
    0: "Pothole",
    1: "Crack",
    2: "Manhole",
}

CLASS_NAMES_ES: dict[str, str] = {
    "Pothole": "Bache",
    "Crack": "Grieta en calzada",
    "Manhole": "Tapa de alcantarillado / boca de acceso",
}

# Valores usados en el notebook final para visualización/inferencia.
YOLO_IMG_SIZE = 640
YOLO_CONF_FOR_PREDICTION = 0.25
YOLO_IOU_FOR_NMS = 0.70
CONFIDENCE_THRESHOLD = 0.40

AGUAS_MENDOCINAS_EMAIL = "reclamos@aysam.com.ar"
FALLBACK_EMAIL = "dpvmza@mendoza.gov.ar"

MUNICIPALIDADES: dict[str, str] = {
    "Capital": "contacto@ciudaddemendoza.gov.ar",
    "Godoy Cruz": "contacto@godoycruz.gob.ar",
    "Guaymallén": "contacto@guaymallen.gov.ar",
    "Las Heras": "contacto@municipalidadlasheras.gob.ar",
    "Luján de Cuyo": "contacto@lujan.gob.ar",
    "Maipú": "contacto@maipu.gob.ar",
    "Lavalle": "contacto@lavalle.gob.ar",
    "San Martín": "contacto@sanmartin.gob.ar",
    "Rivadavia": "contacto@municipalidadrivadavia.gob.ar",
    "Junín": "contacto@junin.gob.ar",
    "La Paz": "contacto@lapaz.gob.ar",
    "Santa Rosa": "contacto@santarosa.gob.ar",
    "San Carlos": "contacto@sancarlos.gob.ar",
    "Tunuyán": "contacto@tunuyan.gob.ar",
    "Tupungato": "contacto@tupungato.gob.ar",
    "San Rafael": "contacto@sanrafael.gob.ar",
    "General Alvear": "contacto@alvearmendoza.gob.ar",
    "Malargüe": "contacto@malargue.gob.ar",
}

DEPARTMENT_OPTIONS = ["Desconocido", *MUNICIPALIDADES.keys()]


# ─────────────────────────────────────────────────────────────
# CARGA DEL MODELO
# ─────────────────────────────────────────────────────────────

def model_exists() -> bool:
    """Indica si existe el modelo local esperado por la app."""
    return DEFAULT_MODEL_PATH.exists()


def download_model(model_url: str, local_path: Path = TMP_MODEL_PATH) -> Path:
    """Descarga el modelo desde una URL directa y devuelve la ruta local."""
    local_path.parent.mkdir(parents=True, exist_ok=True)

    if local_path.exists() and local_path.stat().st_size > 0:
        return local_path

    response = requests.get(model_url, stream=True, timeout=60)
    response.raise_for_status()

    with open(local_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

    return local_path


def load_model(model_path: str | Path | None = None, model_url: str | None = None) -> YOLO:
    """
    Carga el modelo YOLO.

    Prioridad:
    1. model_path explícito.
    2. model_url directo, por ejemplo desde Streamlit Secrets.
    3. dev/modelo.pt dentro del repositorio.
    """
    if model_path is not None:
        selected_path = Path(model_path)
    elif model_url:
        selected_path = download_model(model_url)
    else:
        selected_path = DEFAULT_MODEL_PATH

    if not selected_path.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo en {selected_path}. "
            "Colocá el archivo dev/modelo.pt o configurá MODEL_URL en Streamlit Secrets."
        )

    return YOLO(str(selected_path))


def save_uploaded_model(uploaded_model_file) -> Path:
    """Guarda un modelo subido desde la interfaz en /tmp para probar la app."""
    tmp_path = Path("/tmp/modelo_subido.pt")
    with open(tmp_path, "wb") as f:
        f.write(uploaded_model_file.getbuffer())
    return tmp_path


# ─────────────────────────────────────────────────────────────
# INFERENCIA YOLO
# ─────────────────────────────────────────────────────────────

def run_inference(
    model: YOLO,
    image: Image.Image,
    conf: float = YOLO_CONF_FOR_PREDICTION,
    iou: float = YOLO_IOU_FOR_NMS,
    imgsz: int = YOLO_IMG_SIZE,
):
    """Ejecuta YOLO sobre una imagen PIL y devuelve el primer resultado."""
    if image.mode != "RGB":
        image = image.convert("RGB")

    results = model.predict(
        source=np.array(image),
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device="cpu",
        verbose=False,
    )
    return results[0]


def result_to_dataframe(result) -> pd.DataFrame:
    """Convierte las detecciones de Ultralytics a un DataFrame legible."""
    rows = []
    boxes = getattr(result, "boxes", None)

    if boxes is None or len(boxes) == 0:
        return pd.DataFrame(columns=["clase", "clase_es", "confianza", "x1", "y1", "x2", "y2"])

    for box in boxes:
        cls_id = int(box.cls.item())
        confidence = float(box.conf.item())
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]

        class_name = CLASS_NAMES.get(cls_id, str(cls_id))
        rows.append({
            "clase": class_name,
            "clase_es": CLASS_NAMES_ES.get(class_name, class_name),
            "confianza": confidence,
            "x1": round(x1, 2),
            "y1": round(y1, 2),
            "x2": round(x2, 2),
            "y2": round(y2, 2),
        })

    df = pd.DataFrame(rows)
    return df.sort_values("confianza", ascending=False).reset_index(drop=True)


def plot_result_rgb(result) -> np.ndarray:
    """Devuelve la imagen con cajas dibujadas en formato RGB para Streamlit."""
    plotted_bgr = result.plot()
    return plotted_bgr[..., ::-1]


def get_primary_detection(detections_df: pd.DataFrame) -> Optional[dict]:
    """Devuelve la detección de mayor confianza o None si no hay detecciones."""
    if detections_df.empty:
        return None
    return detections_df.iloc[0].to_dict()


# ─────────────────────────────────────────────────────────────
# EXIF / GPS
# ─────────────────────────────────────────────────────────────

def _to_float(value) -> float:
    """Convierte IFDRational o fracciones EXIF a float."""
    try:
        return float(value)
    except TypeError:
        return float(value[0]) / float(value[1])


def _dms_to_decimal(dms_value) -> float:
    """Convierte grados/minutos/segundos EXIF a grados decimales."""
    degrees = _to_float(dms_value[0])
    minutes = _to_float(dms_value[1]) / 60.0
    seconds = _to_float(dms_value[2]) / 3600.0
    return degrees + minutes + seconds


def extract_gps_from_exif(image: Image.Image) -> Optional[tuple[float, float]]:
    """Extrae latitud y longitud desde los metadatos EXIF, si existen.

    En JPEG, la etiqueta GPSInfo puede contener solamente un puntero al bloque GPS.
    Por eso se usa ``get_ifd(34853)`` cuando Pillow lo admite; leer la etiqueta de
    forma directa puede devolver un entero y hacer que una imagen con GPS parezca no
    tener metadata.
    """
    try:
        exif_data = image.getexif()
    except Exception:
        return None

    if not exif_data:
        return None

    gps_tag_id = next(
        (tag_id for tag_id, tag_name in TAGS.items() if tag_name == "GPSInfo"),
        34853,
    )

    try:
        # Método correcto para obtener el subdirectorio GPS en Pillow moderno.
        gps_info = exif_data.get_ifd(gps_tag_id)
    except (AttributeError, KeyError, TypeError, ValueError):
        # Compatibilidad con imágenes/Pillow que ya devuelven un diccionario directo.
        gps_info = exif_data.get(gps_tag_id)

    if not gps_info or not hasattr(gps_info, "items"):
        return None

    gps_data = {
        GPSTAGS.get(key, key): value
        for key, value in gps_info.items()
    }

    if "GPSLatitude" not in gps_data or "GPSLongitude" not in gps_data:
        return None

    try:
        lat = _dms_to_decimal(gps_data["GPSLatitude"])
        lon = _dms_to_decimal(gps_data["GPSLongitude"])
    except (TypeError, ValueError, IndexError, ZeroDivisionError):
        return None

    lat_ref = gps_data.get("GPSLatitudeRef", "N")
    lon_ref = gps_data.get("GPSLongitudeRef", "E")

    if isinstance(lat_ref, bytes):
        lat_ref = lat_ref.decode(errors="ignore")
    if isinstance(lon_ref, bytes):
        lon_ref = lon_ref.decode(errors="ignore")

    if str(lat_ref).upper() == "S":
        lat = -lat
    if str(lon_ref).upper() == "W":
        lon = -lon

    return lat, lon


# ─────────────────────────────────────────────────────────────
# GEOCODIFICACIÓN INVERSA
# ─────────────────────────────────────────────────────────────

def reverse_geocode(lat: float, lon: float) -> dict:
    """Convierte coordenadas GPS en dirección/departamento con Nominatim."""
    url = "https://nominatim.openstreetmap.org/reverse"
    params = {
        "lat": lat,
        "lon": lon,
        "format": "jsonv2",
        "addressdetails": 1,
        "accept-language": "es",
    }
    headers = {"User-Agent": "RoadDamageDetectorSemana4/1.0"}

    response = requests.get(url, params=params, headers=headers, timeout=10)
    response.raise_for_status()
    data = response.json()

    address = data.get("address", {})
    raw_department = (
        address.get("county")
        or address.get("municipality")
        or address.get("city")
        or address.get("town")
        or address.get("city_district")
        or "Desconocido"
    )

    department = normalize_department(raw_department)

    return {
        "address": data.get("display_name", "Dirección desconocida"),
        "department": department,
        "province": address.get("state", ""),
        "country": address.get("country", ""),
        "lat": lat,
        "lon": lon,
    }


def normalize_department(raw: str) -> str:
    """Normaliza nombres devueltos por Nominatim para que coincidan con el diccionario."""
    if not raw:
        return "Desconocido"

    normalized = raw.lower().strip()
    replacements = {
        "departamento capital": "Capital",
        "capital": "Capital",
        "ciudad de mendoza": "Capital",
        "mendoza": "Capital",
        "departamento godoy cruz": "Godoy Cruz",
        "godoy cruz": "Godoy Cruz",
        "departamento guaymallén": "Guaymallén",
        "guaymallén": "Guaymallén",
        "guaymallen": "Guaymallén",
        "departamento las heras": "Las Heras",
        "las heras": "Las Heras",
        "departamento luján de cuyo": "Luján de Cuyo",
        "luján de cuyo": "Luján de Cuyo",
        "lujan de cuyo": "Luján de Cuyo",
        "departamento maipú": "Maipú",
        "maipú": "Maipú",
        "maipu": "Maipú",
        "departamento lavalle": "Lavalle",
        "lavalle": "Lavalle",
        "departamento san martín": "San Martín",
        "san martín": "San Martín",
        "san martin": "San Martín",
        "departamento rivadavia": "Rivadavia",
        "rivadavia": "Rivadavia",
        "departamento junín": "Junín",
        "junín": "Junín",
        "junin": "Junín",
        "departamento la paz": "La Paz",
        "la paz": "La Paz",
        "departamento santa rosa": "Santa Rosa",
        "santa rosa": "Santa Rosa",
        "departamento san carlos": "San Carlos",
        "san carlos": "San Carlos",
        "departamento tunuyán": "Tunuyán",
        "tunuyán": "Tunuyán",
        "tunuyan": "Tunuyán",
        "departamento tupungato": "Tupungato",
        "tupungato": "Tupungato",
        "departamento san rafael": "San Rafael",
        "san rafael": "San Rafael",
        "departamento general alvear": "General Alvear",
        "general alvear": "General Alvear",
        "departamento malargüe": "Malargüe",
        "malargüe": "Malargüe",
        "malargue": "Malargüe",
    }
    return replacements.get(normalized, raw)


def make_empty_location() -> dict:
    """Ubicación por defecto cuando la imagen no trae GPS."""
    return {
        "address": "No disponible: la imagen no contiene metadatos GPS EXIF.",
        "department": "Desconocido",
        "province": "Mendoza",
        "country": "Argentina",
        "lat": None,
        "lon": None,
    }


# ─────────────────────────────────────────────────────────────
# DERIVACIÓN Y CORREO
# ─────────────────────────────────────────────────────────────

def determine_responsible_entity(class_name: str | None, department: str) -> dict[str, str]:
    """Determina organismo responsable según clase detectada y departamento."""
    if class_name is None:
        return {
            "entity": "No determinado",
            "email": "",
            "reason": "No hubo detecciones válidas, por lo tanto no se genera una derivación automática.",
        }

    if class_name == "Manhole":
        return {
            "entity": "Aguas Mendocinas (AYSAM)",
            "email": AGUAS_MENDOCINAS_EMAIL,
            "reason": "Se detectó una tapa de alcantarillado o boca de acceso. La derivación sugerida es Aguas Mendocinas.",
        }

    if class_name in {"Pothole", "Crack"}:
        email = MUNICIPALIDADES.get(department, FALLBACK_EMAIL)
        if email == FALLBACK_EMAIL:
            entity = "Dirección Provincial de Vialidad de Mendoza"
            reason = (
                "Se detectó un daño en calzada, pero no se pudo determinar el departamento. "
                "Se usa la Dirección Provincial de Vialidad como fallback."
            )
        else:
            entity = f"Municipalidad de {department}"
            damage = CLASS_NAMES_ES.get(class_name, class_name).lower()
            reason = f"Se detectó {damage}. La derivación sugerida es la municipalidad correspondiente al departamento."

        return {
            "entity": entity,
            "email": email,
            "reason": reason,
        }

    return {
        "entity": "No determinado",
        "email": "",
        "reason": "La clase detectada no está contemplada por la lógica de derivación.",
    }


def build_email_body(
    detections_df: pd.DataFrame,
    primary_detection: Optional[dict],
    location_info: dict,
    responsible: dict[str, str],
) -> str:
    """Genera un borrador de correo para reportar los daños detectados."""
    if primary_detection is None or detections_df.empty:
        return "No se generó un reporte porque el modelo no detectó daños con el umbral configurado."

    # Se listan los daños detectados sin confianza ni coordenadas de caja,
    # para que el borrador sea más simple y legible para el organismo receptor.
    counts = detections_df["clase_es"].value_counts(sort=False)
    detections_lines = []
    for label, count in counts.items():
        if int(count) == 1:
            detections_lines.append(f" - {label}")
        else:
            detections_lines.append(f" - {label} ({int(count)} detecciones)")

    lat = location_info.get("lat")
    lon = location_info.get("lon")
    if lat is not None and lon is not None:
        coords_text = f"{lat:.6f}, {lon:.6f}"
        maps_link = f"https://maps.google.com/?q={lat},{lon}"
    else:
        coords_text = "No disponibles"
        maps_link = "No disponible"

    return f"""Estimados,

Me dirijo a ustedes para reportar uno o más daños viales detectados mediante análisis automático de imagen.

ORGANISMO SUGERIDO:
- {responsible.get('entity', 'No determinado')}
- Correo: {responsible.get('email', 'No determinado')}

DAÑOS DETECTADOS:
{chr(10).join(detections_lines)}

UBICACIÓN:
- Dirección: {location_info.get('address', 'No disponible')}
- Departamento: {location_info.get('department', 'No determinado')}
- Provincia: {location_info.get('province', 'Mendoza')}
- Coordenadas: {coords_text}
- Google Maps: {maps_link}

Motivo de derivación:
{responsible.get('reason', '')}

Adjunto a este correo la imagen original para su evaluación.

Saludos cordiales.
"""

def build_mailto_link(email: str, subject: str, body: str) -> str:
    """Construye un link mailto. Para cuerpos largos puede convenir copiar el texto manualmente."""
    return f"mailto:{email}?subject={quote(subject)}&body={quote(body)}"
