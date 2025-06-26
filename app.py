import streamlit as st
import face_recognition
import numpy as np
from PIL import Image, ImageDraw
import os
import shutil

st.title("Facial Recognition App")

KNOWN_FACES_DIR = "known_faces"

def load_known_faces():
    encodings = []
    names = []
    for filename in os.listdir(KNOWN_FACES_DIR):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image = face_recognition.load_image_file(os.path.join(KNOWN_FACES_DIR, filename))
            faces = face_recognition.face_encodings(image)
            if faces:
                encodings.append(faces[0])
                names.append(os.path.splitext(filename)[0])
    return encodings, names

def save_new_face(image_file, name):
    # Save uploaded image to known_faces directory with the given name
    ext = os.path.splitext(image_file.name)[1]
    save_path = os.path.join(KNOWN_FACES_DIR, f"{name}{ext}")
    with open(save_path, "wb") as f:
        f.write(image_file.getbuffer())

# Section: Add New Face
st.header("Add a New Face")
with st.form("add_face_form", clear_on_submit=True):
    new_face_image = st.file_uploader("Upload a clear photo of the person", type=['jpg', 'jpeg', 'png'], key="add_face")
    new_face_name = st.text_input("Enter the name of the person (no spaces or special characters)")
    submitted = st.form_submit_button("Add Face")
    if submitted:
        if new_face_image and new_face_name:
            save_new_face(new_face_image, new_face_name)
            st.success(f"Added {new_face_name} to the known faces database.")
        else:
            st.warning("Please provide both a photo and a name.")

# Load known faces (reload after adding)
known_face_encodings, known_face_names = load_known_faces()

# Section: Face Recognition
st.header("Recognize Faces in an Image")
uploaded_file = st.file_uploader("Upload an image for recognition", type=['jpg', 'jpeg', 'png'], key="recognition")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    face_locations = face_recognition.face_locations(img_array)
    face_encodings = face_recognition.face_encodings(img_array, face_locations)

    draw = ImageDraw.Draw(image)
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 255, 0), width=3)
        draw.text((left, bottom + 10), name, fill=(255, 0, 0))
    st.image(image, caption="Processed Image", use_column_width=True)
    st.success(f"Found {len(face_locations)} face(s).")
