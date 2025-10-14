import streamlit as st
import os
import json
import shutil
from pkg.memory_kg import MemoryKG, LocalFileAdapter
from langchain_community.embeddings import OpenAIEmbeddings

PROFILES_FILE = "profiles.json"

def load_profiles():
    if os.path.exists(PROFILES_FILE):
        with open(PROFILES_FILE, "r") as f:
            return json.load(f)
    return {}

def save_profiles(profiles):
    with open(PROFILES_FILE, "w") as f:
        json.dump(profiles, f, indent=2)

def ensure_profile_dirs(profile_name):
    base = os.path.join("data", profile_name)
    os.makedirs(base, exist_ok=True)
    os.makedirs(os.path.join(base, "uploads"), exist_ok=True)
    os.makedirs(os.path.join(base, "processed"), exist_ok=True)
    return base

def delete_profile(profile_name):
    profiles = load_profiles()
    if profile_name in profiles:
        profiles.pop(profile_name)
        save_profiles(profiles)

    profile_dir = os.path.join("data", profile_name)
    if os.path.exists(profile_dir):
        shutil.rmtree(profile_dir)

    if st.session_state.get("active_profile") == profile_name:
        st.session_state.active_profile = None
        st.session_state.memory_kg = None
        st.session_state.profile_dir = None
    st.success(f"Profile '{profile_name}' deleted!")

def profile_selector():
    st.title("Choose Your Profile")

    if "active_profile" not in st.session_state:
        st.session_state.active_profile = None
    if "profiles" not in st.session_state:
        st.session_state.profiles = load_profiles()

    profiles = st.session_state.profiles
    cols = st.columns(len(profiles) + 2)

    for i, (name, data) in enumerate(profiles.items()):
        with cols[i]:
            if st.button(name, key=f"profile_{name}"):
                st.session_state.active_profile = name
                profile_dir = ensure_profile_dirs(name)
                st.session_state["profile_dir"] = profile_dir
                adapter = LocalFileAdapter(
                    profile_name=name,
                    embeddings=OpenAIEmbeddings()
                )
                st.session_state.memory_kg = MemoryKG(adapter=adapter, profile_name=name)
                st.success(f"Profile '{name}' selected!")

            if st.button(f"Delete {name}", key=f"delete_{name}"):
                delete_profile(name)
            st.session_state.refresh = not st.session_state.get("refresh", False)  # triggers rerun


    with cols[-2]:
        new_name = st.text_input("New Profile")
        if new_name and st.button("Add"):
            if new_name not in profiles:
                profiles[new_name] = {"theme": "default", "conversation": []}
                save_profiles(profiles)
                st.session_state.profiles = profiles

            st.session_state.active_profile = new_name
            profile_dir = ensure_profile_dirs(new_name)
            st.session_state["profile_dir"] = profile_dir
            adapter = LocalFileAdapter(
                profile_name=new_name,
                embeddings=OpenAIEmbeddings()
            )
            st.session_state.memory_kg = MemoryKG(adapter=adapter, profile_name=new_name)
            st.success(f"Profile '{new_name}' created and selected!")

    if not profiles:
        st.info("No profiles yet. Add a new profile above.")

def run():
    profile_selector()