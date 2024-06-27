import streamlit as st
import pandas as pd
from PIL import Image
import pytesseract as tess


def createdDisplay():
    st.logo("./image/logo_explain.png")
    st.title("Projet LIPSTIP - EFREI")
    choice = st.selectbox("Choisissez comment entrer le brevet", ["Analyse de texte", "Analyse de fichier (pdf ou txt)"])
    if choice == "Analyse de texte":
        entered_text = st.text_area("Entrez le texte à analyser")
    elif choice == "Analyse de fichier (pdf ou txt)":
        uploaded_file = st.file_uploader("Choisissez un fichier", type=["pdf", "txt"])
    if st.button("Lancer l'analyse"):
        if uploaded_file is not None:
            description = readFile(uploaded_file)
            st.write(description)
        elif entered_text is not None:
            description = readText(entered_text)
            st.write(description)
        else:
            st.write("Veuillez insérer un fichier ou bien un texte")
        else:
            readTXT(entered_text)


def readFile(uploaded_file):
    file_type = uploaded_file.name.split(".")[-1]
    if file_type == "pdf":
        image = Image.open(uploaded_file.read())
        description = tess.image_to_string(image, lang='eng')
        
    elif file_type == "txt":
        description = uploaded_file.read().decode("utf-8")
    return description

def readText(text):
    description = text
    return description

if __name__ == "__main__":
    createdDisplay()