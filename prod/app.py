"""
App Streamlit - Detector de Daños Viales con YOLO.

Ruta principal para Streamlit Cloud:
    prod/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Permite importar utils.py aunque Streamlit ejecute desde la raíz del repo.
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

import pandas as pd
import streamlit as st
from PIL import Image

from utils import (
    CLASS_NAMES,
    CONFIDENCE_THRESHOLD,
    DEFAULT_MODEL_PATH,
    DEPARTMENT_OPTIONS,
    YOLO_CONF_FOR_PREDICTION,
    YOLO_IMG_SIZE,
    YOLO_IOU_FOR_NMS,
    build_email_body,
    build_mailto_link,
    determine_responsible_entity,
    extract_gps_from_exif,
    get_primary_detection,
    load_model,
    make_empty_location,
    plot_result_rgb,
    result_to_dataframe,
    reverse_geocode,
    run_inference,
)

st.set_page_config(
    page_title="Detector de Daños Viales — Mendoza",
    page_icon="🛣️",
    layout="wide",
)

st.title("🛣️ Detector de Daños Viales — Mendoza")
st.caption(
    "Aplicación de Semana 4: sube una imagen, detecta baches/grietas/tapas con YOLO "
    "y genera una derivación sugerida al organismo correspondiente."
)


@st.cache_resource(show_spinner=False)
def get_model_cached(model_path: str | None, model_url: str | None):
    return load_model(model_path=model_path, model_url=model_url)


def get_secret_value(key: str, default: str = "") -> str:
    """Lee un secreto de Streamlit sin romper la app si no existe secrets.toml."""
    try:
        return st.secrets.get(key, default)
    except FileNotFoundError:
        return default


# ─────────────────────────────────────────────────────────────
# CARGA DEL MODELO
# ─────────────────────────────────────────────────────────────

model = None
model_error = None
model_url = get_secret_value("MODEL_URL", "")

try:
    if model_url:
        model = get_model_cached(None, model_url)
    else:
        model = get_model_cached(str(DEFAULT_MODEL_PATH), None)
except Exception as exc:
    model_error = exc

with st.expander("Información del modelo, clases y parámetros", expanded=False):
    st.write("Clases configuradas según el notebook final:")
    st.table(pd.DataFrame([{"id": k, "clase": v} for k, v in CLASS_NAMES.items()]))

    st.write("Parámetros de inferencia:")
    yolo_img_size = st.slider(
        "Tamaño de entrada (imgsz)",
        min_value=320,
        max_value=1280,
        value=YOLO_IMG_SIZE,
        step=32,
        help=(
            "Resolución usada por YOLO durante la predicción. Un valor mayor puede "
            "detectar detalles más pequeños, pero demora más y consume más recursos."
        ),
        key="yolo_img_size",
    )
    yolo_confidence = st.slider(
        "Umbral de confianza (conf)",
        min_value=0.05,
        max_value=0.95,
        value=float(YOLO_CONF_FOR_PREDICTION),
        step=0.05,
        format="%.2f",
        help=(
            "Confianza mínima para mostrar una detección. Si se baja, aparecen más "
            "detecciones pero también pueden aumentar los falsos positivos."
        ),
        key="yolo_confidence",
    )
    yolo_iou = st.slider(
        "Umbral de solapamiento para NMS (iou)",
        min_value=0.10,
        max_value=0.95,
        value=float(YOLO_IOU_FOR_NMS),
        step=0.05,
        format="%.2f",
        help=(
            "Controla la eliminación de cajas superpuestas sobre el mismo daño. "
            "NMS conserva la detección más confiable y descarta cajas redundantes."
        ),
        key="yolo_iou",
    )

if model is None:
    st.error("No se pudo cargar el modelo YOLO.")
    st.write("Detalle del error:")
    st.code(str(model_error), language="text")
    st.markdown(
        "Soluciones posibles:\n"
        "1. Colocar el archivo `modelo.pt` en `dev/modelo.pt`.\n"
        "2. En Streamlit Cloud, agregar `MODEL_URL` en Settings → Secrets si el modelo se descarga desde Drive/Hugging Face."
    )
    st.stop()

# ─────────────────────────────────────────────────────────────
# CARGA DE IMAGEN
# ─────────────────────────────────────────────────────────────

st.divider()
st.subheader("1. Subir imagen")

uploaded_image = st.file_uploader(
    "Seleccioná una imagen JPG, JPEG o PNG",
    type=["jpg", "jpeg", "png"],
    help="Para extraer ubicación automática, la foto debe conservar metadatos EXIF con GPS.",
)

if uploaded_image is None:
    st.info("Subí una imagen para ejecutar la detección.")
    st.stop()

# Se extrae EXIF antes de convertir la imagen para conservar todos los metadatos.
original_image = Image.open(uploaded_image)
gps_coords = extract_gps_from_exif(original_image)

image = original_image
if image.mode != "RGB":
    image = image.convert("RGB")

# ─────────────────────────────────────────────────────────────
# UBICACIÓN
# ─────────────────────────────────────────────────────────────

st.divider()
st.subheader("2. Ubicación")

location_info = make_empty_location()

if gps_coords:
    lat, lon = gps_coords
    try:
        with st.spinner("Obteniendo dirección desde coordenadas GPS..."):
            location_info = reverse_geocode(lat, lon)
        st.success("La imagen contiene coordenadas GPS EXIF.")
    except Exception as exc:
        st.warning(f"La imagen tiene GPS, pero falló la geocodificación inversa: {exc}")
        location_info = {
            "address": "No disponible por error de geocodificación.",
            "department": "Desconocido",
            "province": "Mendoza",
            "country": "Argentina",
            "lat": lat,
            "lon": lon,
        }
else:
    st.warning("La imagen no contiene datos GPS EXIF. Podés elegir el departamento manualmente.")

# Corrección manual del departamento.
detected_department = location_info.get("department", "Desconocido")
if detected_department not in DEPARTMENT_OPTIONS:
    detected_department = "Desconocido"

manual_department = st.selectbox(
    "Departamento para la derivación",
    options=DEPARTMENT_OPTIONS,
    index=DEPARTMENT_OPTIONS.index(detected_department),
)
location_info["department"] = manual_department

col_location_a, col_location_b = st.columns([2, 1])
with col_location_a:
    st.write(f"**Dirección:** {location_info.get('address', 'No disponible')}")
    st.write(f"**Departamento usado:** {location_info.get('department', 'Desconocido')}")

with col_location_b:
    lat = location_info.get("lat")
    lon = location_info.get("lon")
    if lat is not None and lon is not None:
        st.write(f"**Latitud:** {lat:.6f}")
        st.write(f"**Longitud:** {lon:.6f}")
        st.map(pd.DataFrame({"lat": [lat], "lon": [lon]}), zoom=14)
    else:
        st.write("**Coordenadas:** no disponibles")

# ─────────────────────────────────────────────────────────────
# INFERENCIA
# ─────────────────────────────────────────────────────────────

st.divider()
st.subheader("3. Detección con YOLO")

with st.spinner("Ejecutando inferencia..."):
    result = run_inference(
        model=model,
        image=image,
        conf=yolo_confidence,
        iou=yolo_iou,
        imgsz=yolo_img_size,
    )
    detections_df = result_to_dataframe(result)
    primary_detection = get_primary_detection(detections_df)
    plotted_image = plot_result_rgb(result)

col_image, col_results = st.columns([1, 1])

with col_image:
    st.image(image, caption="Imagen original", use_container_width=True)
    st.image(plotted_image, caption="Resultado YOLO", use_container_width=True)

with col_results:
    if detections_df.empty:
        st.warning("No se detectaron daños con el umbral de confianza configurado.")
    else:
        st.metric("Cantidad de detecciones", len(detections_df))
        st.dataframe(
            detections_df.assign(confianza=lambda df: (df["confianza"] * 100).round(1).astype(str) + "%"),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("**Daños detectados:**")
        for i, row in detections_df.iterrows():
            st.write(f"- Detección {i + 1}: {row['clase_es']} ({row['confianza'] * 100:.1f}%)")

        low_confidence = detections_df[detections_df["confianza"] < CONFIDENCE_THRESHOLD]
        if not low_confidence.empty:
            st.warning(
                f"Hay {len(low_confidence)} detección/es con confianza menor a "
                f"{CONFIDENCE_THRESHOLD * 100:.0f}%. Conviene verificar manualmente antes de reportar."
            )

# ─────────────────────────────────────────────────────────────
# DERIVACIÓN Y CORREO
# ─────────────────────────────────────────────────────────────

st.divider()
st.subheader("4. Derivación sugerida")

primary_class = primary_detection["clase"] if primary_detection else None
responsible = determine_responsible_entity(primary_class, location_info["department"])

if responsible["email"]:
    st.success(f"**Organismo:** {responsible['entity']}")
    st.write(f"**Correo:** `{responsible['email']}`")
else:
    st.warning("No se generó organismo responsable porque no hubo detecciones válidas.")
st.caption(responsible["reason"])

st.divider()
st.subheader("5. Borrador de correo")

email_body = build_email_body(
    detections_df=detections_df,
    primary_detection=primary_detection,
    location_info=location_info,
    responsible=responsible,
)

st.text_area(
    "Texto sugerido para copiar en el correo",
    value=email_body,
    height=360,
)

st.download_button(
    "Descargar borrador .txt",
    data=email_body,
    file_name="reporte_dano_vial.txt",
    mime="text/plain",
)

if responsible["email"] and primary_detection:
    detected_labels = detections_df["clase_es"].drop_duplicates().tolist()
    subject = "Reporte de daño vial - " + ", ".join(detected_labels)
    mailto_link = build_mailto_link(responsible["email"], subject, email_body)
    st.link_button("Abrir cliente de correo", mailto_link)

st.divider()
st.caption(
    "Sistema desarrollado para el proyecto de Redes Neuronales Profundas. "
    "El resultado depende del modelo entrenado, el umbral de confianza y la calidad de la imagen."
)
