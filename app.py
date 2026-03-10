import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from supabase import create_client, Client
import uuid
import os

# ------------------------
# PAGE CONFIG
# ------------------------

st.set_page_config(
    page_title="Digitales Fundbüro",
    page_icon="🔎",
    layout="wide"
)

# ------------------------
# DESIGN (CSS)
# ------------------------

st.markdown("""
<style>

.main {
    background-color: #f5f7fb;
}

.stButton>button {
    background-color: #4CAF50;
    color: white;
    border-radius: 8px;
    height: 3em;
    width: 100%;
}

.card {
    background-color: white;
    padding: 15px;
    border-radius: 12px;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
    margin-bottom: 20px;
}

.title {
    font-size: 40px;
    font-weight: bold;
}

.subtitle {
    color: grey;
    margin-bottom: 20px;
}

</style>
""", unsafe_allow_html=True)

# ------------------------
# SUPABASE CONFIG
# ------------------------

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    st.error("SUPABASE_URL oder SUPABASE_KEY ist nicht gesetzt!")
    st.stop()

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

BUCKET = "lost-items"

# ------------------------
# MODEL LADEN
# ------------------------

model = tf.keras.models.load_model("keras_model.h5")

with open("labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

# ------------------------
# IMAGE PREPROCESS
# ------------------------

def preprocess_image(image):
    image = image.resize((224, 224))
    image_array = np.asarray(image)
    image_array = image_array.astype(np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# ------------------------
# VORHERSAGE
# ------------------------

def predict(image):
    processed = preprocess_image(image)
    prediction = model.predict(processed)
    index = np.argmax(prediction)
    label = labels[index]
    confidence = prediction[0][index]
    return label, confidence

# ------------------------
# HEADER
# ------------------------

st.markdown('<div class="title">🔎 Digitales Fundbüro</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Finde verlorene Gegenstände oder melde gefundene Objekte</div>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📤 Fund melden", "🔍 Gegenstand suchen"])

# ------------------------
# TAB 1: FUND MELDEN
# ------------------------

with tab1:

    st.subheader("Fundstück hochladen")

    uploaded_file = st.file_uploader("Bild hochladen", type=["jpg","png","jpeg"])

    if uploaded_file:

        col1, col2 = st.columns(2)

        image = Image.open(uploaded_file)

        with col1:
            st.image(image, caption="Hochgeladenes Bild", use_column_width=True)

        label, confidence = predict(image)

        with col2:
            st.markdown("### 🤖 KI-Erkennung")

            st.success(f"Objekt: **{label}**")
            st.write("Sicherheit:", round(float(confidence)*100,2), "%")

            if st.button("📦 Fund speichern"):

                file_id = str(uuid.uuid4()) + ".jpg"
                image_bytes = uploaded_file.getvalue()

                # Bild hochladen
                supabase.storage.from_(BUCKET).upload(
                    file_id,
                    image_bytes,
                    {"content-type": "image/jpeg"}
                )

                public_url = supabase.storage.from_(BUCKET).get_public_url(file_id)

                # Datenbank speichern
                supabase.table("lost_items").insert({
                    "label": label,
                    "image_url": public_url
                }).execute()

                st.success("Fund erfolgreich gespeichert!")

# ------------------------
# TAB 2: SUCHE
# ------------------------

with tab2:

    st.subheader("Fundstücke durchsuchen")

    search = st.text_input("🔍 Objekt suchen")

    show_all = st.checkbox("Alle Gegenstände anzeigen")

    if st.button("Suchen"):

        query = supabase.table("lost_items").select("*")

        if not show_all:
            if search != "":
                query = query.ilike("label", f"%{search}%")

        result = query.execute()

        if len(result.data) == 0:
            st.info("Keine Gegenstände gefunden")

        cols = st.columns(3)

        for i, item in enumerate(result.data):

            with cols[i % 3]:

                st.markdown('<div class="card">', unsafe_allow_html=True)

                st.image(item["image_url"], use_column_width=True)

                st.markdown(f"**Kategorie:** {item['label']}")

                st.markdown('</div>', unsafe_allow_html=True)
