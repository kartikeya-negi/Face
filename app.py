import streamlit as st
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw
import os
import json

st.title("Face Detection & Labeling App (Streamlit Cloud Compatible)")

LABELED_FACES_FILE = "labeled_faces.json"

# --- Helper functions ---
def load_labeled_faces():
    if os.path.exists(LABELED_FACES_FILE):
        with open(LABELED_FACES_FILE, "r") as f:
            return json.load(f)
    return {}

def save_labeled_faces(data):
    with open(LABELED_FACES_FILE, "w") as f:
        json.dump(data, f)

# --- Section: Add a Label to a Face ---
st.header("Add a Label to a Detected Face")
labeled_faces = load_labeled_faces()

add_face_image = st.file_uploader("Upload an image (clear face, for labeling)", type=['jpg', 'jpeg', 'png'], key="add_face")
add_face_name = st.text_input("Enter a label/name for the face")

if st.button("Add Face Label"):
    if add_face_image and add_face_name:
        image = Image.open(add_face_image).convert('RGB')
        img_array = np.array(image)

        mp_face_detection = mp.solutions.face_detection
        with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.7) as face_detection:
            results = face_detection.process(img_array)
            if results.detections:
                # Use the bounding box as a "face signature"
                bbox = results.detections[0].location_data.relative_bounding_box
                key = f"{bbox.xmin:.3f}_{bbox.ymin:.3f}_{bbox.width:.3f}_{bbox.height:.3f}"
                labeled_faces[key] = add_face_name
                save_labeled_faces(labeled_faces)
                st.success(f"Labeled face as '{add_face_name}'.")
            else:
                st.warning("No face detected in the image.")
    else:
        st.warning("Please upload an image and enter a name.")

# --- Section: Detect (and label) Faces in Uploaded Image ---
st.header("Detect Faces in an Image")
uploaded_file = st.file_uploader("Upload an image for detection", type=['jpg', 'jpeg', 'png'], key="detect_face")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    img_array = np.array(image)

    mp_face_detection = mp.solutions.face_detection
    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(img_array)

        draw = ImageDraw.Draw(image)
        face_count = 0
        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                ih, iw, _ = img_array.shape
                x = int(bbox.xmin * iw)
                y = int(bbox.ymin * ih)
                w = int(bbox.width * iw)
                h = int(bbox.height * ih)
                draw.rectangle([x, y, x + w, y + h], outline=(0, 255, 0), width=3)

                # Try to match with labeled faces (simple bbox match)
                key = f"{bbox.xmin:.3f}_{bbox.ymin:.3f}_{bbox.width:.3f}_{bbox.height:.3f}"
                label = labeled_faces.get(key, "Unknown")
                draw.text((x, y + h + 5), label, fill=(255, 0, 0))
                face_count += 1

            st.image(image, caption="Detected Faces", use_column_width=True)
            st.success(f"Found {face_count} face(s).")
        else:
            st.image(image, caption="No faces detected", use_column_width=True)
            st.warning("No faces found in the image.")

# --- Show all labeled faces ---
st.header("All Labeled Faces")
if labeled_faces:
    for k, v in labeled_faces.items():
        st.write(f"Label: **{v}** (bbox signature: {k})")
else:
    st.write("No faces labeled yet.")
