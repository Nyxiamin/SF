import streamlit as st
import pandas as pd

from functionDataFrame import readDataframe, createdCleanCSV

df = pd.read_csv("../EFREI_LIPSTIP_50k_elements_EPO_clean.csv", sep=',', encoding='utf-8', nrows=1000)

def createdDisplay():
    global df
    st.write("Projet LIPSTIP - EFREI")
    st.write("veuillez soit insérer un fichier ou bien copier coller votre texte dans la zone de texte")
    # uploaded_file = st.file_uploader("Uploader votre fichier", type=["pdf"])
    entered_text = st.text_area("Entrez votre texte ici")
    if st.button("Soumettre"):
        
        # À garder si on lis les PDF
        """if uploaded_file is not None and entered_text is not None:
            st.write("Veuillez insérer soit un fichier ou bien un texte, pas les deux")
        elif uploaded_file is not None:
            st.write("Fichier inséré")
        elif entered_text is not None:
            st.write(entered_text)"""
        if entered_text is None:
            st.write("Veuillez insérer un fichier ou bien un texte")
        else:
            readTXT(entered_text)

def readPDF(file):
    pass

def readTXT(text):
    pass

if __name__ == "__main__":
    createdDisplay()