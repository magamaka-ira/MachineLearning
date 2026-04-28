import streamlit as st

st.title("Mon app Streamlit 🚀")
st.write("Bienvenue dans notre projet Immo")

name = st.text_input("Ton prénom")
if name:
    st.write(f"Hello {name} 👋")

    
st.title("Test Streamlit")
st.write("Si tu vois ça, tout marche")