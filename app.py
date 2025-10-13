import streamlit as st
import os

st.sidebar.title("Team Login")
username = st.sidebar.text_input("Username")
password = st.sidebar.text_input("Password", type="password")

if username != "team" or password != "secure123":
    st.error("Invalid credentials")
    st.stop()

st.sidebar.success("Welcome, team 👋")

from pkg import profile
profile.run()
