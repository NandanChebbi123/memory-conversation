import os
import streamlit as st
from PIL import Image, ImageDraw
from facenet_pytorch import MTCNN

mtcnn = MTCNN(keep_all=True)

def get_upload_dirs():
    if "profile_dir" not in st.session_state:
        raise RuntimeError("Profile directory not initialized. Select a profile first.")
    profile_dir = st.session_state["profile_dir"]
    upload_dir = os.path.join(profile_dir, "uploads")
    processed_dir = os.path.join(profile_dir, "processed")
    os.makedirs(upload_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    return upload_dir, processed_dir

def draw_faces(image_path, processed_dir):
    """Detect faces and draw bounding boxes"""
    image = Image.open(image_path).convert("RGB")
    boxes, probs = mtcnn.detect(image)
    draw = ImageDraw.Draw(image)
    if boxes is not None:
        for i, (box, prob) in enumerate(zip(boxes, probs)):
            draw.rectangle(box.tolist(), outline="red", width=3)
            draw.text((box[0], box[1] - 10), f"Face {i+1} ({prob:.2f})", fill="red")

    processed_path = os.path.join(processed_dir, os.path.basename(image_path))
    image.save(processed_path)
    return processed_path

def run():
    st.subheader("Photo Panel")
    upload_dir, processed_dir = get_upload_dirs()
    uploaded_files = st.file_uploader("Upload images", accept_multiple_files=True, type=["jpg", "png", "jpeg"])
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_path = os.path.join(upload_dir, uploaded_file.name)
            if not os.path.exists(file_path):
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
            st.success(f"Uploaded {uploaded_file.name}")

    st.subheader("Uploaded Photos")
    image_files = sorted(os.listdir(upload_dir))
    selected_image = None
    if image_files:
        cols = st.columns(3)
        for idx, img in enumerate(image_files):
            path = os.path.join(upload_dir, img)
            with cols[idx % 3]:
                if st.button(f"Select {img}", key=f"{st.session_state.active_profile}_{img}"):
                    processed_path = draw_faces(path, processed_dir)
                    profile_key = f"selected_image_{st.session_state.active_profile}"
                    st.session_state[profile_key] = processed_path
                    selected_image = processed_path
                st.image(path, caption=img, use_container_width=True)

    profile_key = f"selected_image_{st.session_state.active_profile}"
    if profile_key in st.session_state:
        selected_image = st.session_state[profile_key]
        st.subheader("Selected Image with Detected Faces")
        st.image(selected_image, use_column_width=True)

    return selected_image