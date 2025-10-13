import os
from dotenv import load_dotenv

load_dotenv()
print("OPENAI_API_KEY:", os.getenv("OPENAI_API_KEY"))


for var in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"]:
    if var in os.environ:
        del os.environ[var]

from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

import sys, os
import streamlit as st
import networkx as nx

sys.path.append(os.path.join(os.path.dirname(__file__), "pkg"))
from pkg import gpt4v, profile
from pkg.memory_kg import MemoryKG, LocalFileAdapter
from langchain.embeddings.openai import OpenAIEmbeddings
from dotenv import load_dotenv
from pkg.logger import get_logger

@st.cache_resource(show_spinner=False)
def get_memory(profile_name: str):
    adapter = LocalFileAdapter(
        profile_name=profile_name,
        embeddings=OpenAIEmbeddings()
    )
    return MemoryKG(adapter=adapter, profile_name=profile_name)

if "active_profile" not in st.session_state or st.session_state.active_profile is None:
    profile.run()
    st.stop()

active_profile = st.session_state.active_profile
logger = get_logger(active_profile)

if (
    "memory_kg" not in st.session_state
    or st.session_state.memory_kg.profile_name != active_profile
):
    st.session_state.memory_kg = get_memory(active_profile)

memory_kg = st.session_state.memory_kg
selected_image_key = f"selected_image_{active_profile}"

st.sidebar.title(f"Profile: {active_profile}")
default_page = st.session_state.get("selected_page", "Upload & Select")
page = st.sidebar.radio(
    "Go to",
    ["Upload & Select", "Chat with GPT", "Knowledge Graph"],
    index=["Upload & Select", "Chat with GPT", "Knowledge Graph"].index(default_page)
)
st.session_state["selected_page"] = page

if page == "Upload & Select":
    from pkg import image_upload_display
    st.header("Upload or Select a Photo")
    processed_path = image_upload_display.run()
    if processed_path:
        st.session_state[selected_image_key] = processed_path
        st.success("Photo selected successfully!")

elif page == "Chat with GPT":
    image_path = st.session_state.get(selected_image_key)
    if image_path and os.path.exists(image_path):
        st.info(f"Using image: {os.path.basename(image_path)}")
        gpt4v.run(image_path=image_path)
    else:
        st.warning("Please upload and select an image first.")

elif page == "Knowledge Graph":
    st.header("Knowledge Graph Viewer")
    memory_kg.display_streamlit(height=700, width=900)