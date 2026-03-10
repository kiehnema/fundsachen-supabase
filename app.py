import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from supabase import create_client, Client
import uuid
import os

# ------------------------
# SUPABASE CONFIG
# ------------------------

# Supabase URL und Key über Secrets oder Environment Variablen
SUPABASE_URL = os.environ.get("SUPABASE_URL")  # z.B. "https://gtwnoeacbgpxpojizzub.supabase.co"
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")  # z.B. ANON-KEY

# Prüfen, ob URL und Key gesetzt sind
if not SUPABASE_URL or not SUPABASE_KEY:
    st.error("SUPABASE_URL oder SUPABASE_KEY ist nicht gesetzt!")
    st.stop()

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

BUCKET = "lost-items"  # Name des Storage-Buckets

# ------------------------
# MODEL LADEN
# ------------------------

model = tf.keras.models.load_model("keras_model.h5")

with open("labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

# ------------------------
# FUNKTION: IMAGE PREPROCESS
# ------------------------

def preprocess_image(image):
    image = image.resize((224, 224))
    image_array = np.asarray(image)
    image_array = image_array.astype(np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# ------------------------
# FUNKTION: VORHERSAGE
# ------------------------

def predict(image):
    processed = preprocess_image(image)
    prediction = model.predict(processed)
    index = np.argmax(prediction)
    label = labels[index]
    confidence = prediction[0][index]
    return label, confidence

# ------------------------
# STREAMLIT UI
# ------------------------

st.title("🔎 Digitales Fundbüro")

tab1, tab2 = st.tabs(["Fund hochladen", "Suche"])

# ------------------------
# TAB 1: UPLOAD
# ------------------------

with tab1:
    uploaded_file = st.file_uploader("Bild hochladen", type=["jpg","png","jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Hochgeladenes Bild", use_column_width=True)

        label, confidence = predict(image)

        st.write("Erkanntes Objekt:", label)
        st.write("Sicherheit:", round(float(confidence)*100,2), "%")

        if st.button("Fund speichern"):
            file_id = str(uuid.uuid4()) + ".jpg"
            image_bytes = uploaded_file.getvalue()

            # Upload zu Supabase Storage
            supabase.storage.from_(lost-items).upload(
                file_id,
                image_bytes,
                {"content-type": "image/jpeg"}
            )

            public_url = supabase.storage.from_(lost-items).get_public_url(file_id)

            # In Datenbank speichern
            supabase.table("lost_items").insert({
                "label": label,
                "image_url": public_url
            }).execute()

            st.success("Fund gespeichert!")

# ------------------------
# TAB 2: SUCHE
# ------------------------

with tab2:
    search = st.text_input("Nach Objekt suchen")

    if st.button("Suchen"):
        query = supabase.table("lost_items").select("*")

        if search != "":
            query = query.ilike("label", f"%{search}%")

        result = query.execute()

        for item in result.data:
            st.image(item["image_url"], width=200)
            st.write("Kategorie:", item["label"])
            st.write("---")
