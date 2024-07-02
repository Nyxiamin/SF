import streamlit as st
import pandas as pd
from PIL import Image
import pytesseract as tess
import fitz
import streamlit_scrollable_textbox as stx
import regex as re
from annotated_text import annotated_text


from functionresult import functionresult

css_style = """
<style>
h1 {
    color: #52d274;
}
h2 {
    color: #52d274;
}
</style>
"""

dic_relation_lettre_domaine = {'A' : "Human necessities (agriculture, health)",
                               'B' : "Performing operations & transporting (printing, microstructural tech)",
                               'C' :  "Chemistry & metallurgy (chemistry, metallurgy)",
                               'D' : "Textiles & paper (textiles, paper)",
                               'E' : "Fixed constructions (building, mining)",
                               'F' : "Mechanical engineering & Lighting & Heating & Weapons & Blasting (lighting, weapons)",
                               'G' : "Physics (instruments, nucleonics)",
                               'H' : "Electricity (semiconductors, electric power)",
                               'Y' : "General tagging of new technological developments"}

def createdDisplay(df_cleaned):
    st.markdown(css_style, unsafe_allow_html=True)
    st.logo("./img/logo_explain.png") #si cette ligne ne fonctionne pas, il faut remplacer logo par image
    st.title("Projet LIPSTIP - EFREI")
    st.write("Ce projet a pour but de prédire les codes CPC d'un brevet et de trouver des documents similaires en fonction de la description du brevet.")
    
    entered_text = None
    uploaded_file = None
    
    # création d'un module déroulant pour le choix de l'entrée du brevet
    choice = st.selectbox("Choisissez comment entrer le brevet", ["Analyse de fichier (pdf ou txt)", "Analyse de texte"])
    if choice == "Analyse de texte":
        entered_text = st.text_area("Entrez le texte à analyser")
    elif choice == "Analyse de fichier (pdf ou txt)":
        uploaded_file = st.file_uploader("Choisissez un fichier", type=["pdf", "txt"])
    
    # création d'un bouton pour lancer l'analyse en fonction de l'entrée choisie
    if st.button("Lancer l'analyse"):
        if uploaded_file is not None:
            description = readFile(uploaded_file)
        elif entered_text is not None:
            description = readText(entered_text)
        else:
            st.write("Veuillez insérer un fichier ou bien un texte")
            description = None

        if description is not None:
            with st.spinner("Analyse en cours..."):
                predicted_cpc, similar_documents, new_imp_words = functionresult(description, df_cleaned)
            
            # Affichage des codes CPC probables
            st.header("Voici les codes CPC prédits")
            st.subheader("Le code du brevet le plus probable est : ")
            st.markdown("**" + predicted_cpc[0] + "**" + " correspondant au domaine : " + dic_relation_lettre_domaine[predicted_cpc[0]] )
            st.subheader("Les deux autres codes du brevet les plus probables sont : ")
            for i in range(1,len(predicted_cpc)):
                st.write("**" + predicted_cpc[i] + "**" + " correspondant au domaine : " + dic_relation_lettre_domaine[predicted_cpc[i]])
            
            # Affichage des documents similaires
            st.header("Voici les documents similaires")
            for similar_document in similar_documents:
                for code in similar_document[1]:
                    st.markdown("Le code du document similaire est "+ "**" + code + "**" + ", correspondant au domaine : " + dic_relation_lettre_domaine[code])
                st.write("Le pourcentage de similarité est de {:.2f}%".format(similar_document[2]))
                stx.scrollableTextbox(similar_document[0], height=250)
                st.divider()
            
            # Affichage du texte avec les mots importants surlignés
            st.header("Voici les mots importants")
            highlighted_text = highligh_words(description, new_imp_words)
            with st.spinner("Mise en évidence des mots importants..."):
                annotated_text(highlighted_text)

def readFile(uploaded_file):
    '''Fonction pour lire le contenu d'un fichier pdf ou txt et le retourner sous forme de texte'''
    file_type = uploaded_file.name.split(".")[-1]
    if file_type == "pdf":
        description = readPDF(uploaded_file)
        
    elif file_type == "txt":
        description = uploaded_file.read().decode("utf-8")
    return description

def readText(text):
    '''Fonction pour lire le contenu d'un texte et le retourner'''
    description = text
    return description

def readPDF(uploaded_file):
    pdf = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    description = ""
    for i in range(pdf.page_count):
        page = pdf.load_page(i)
        
        page_text = page.get_text()
        if page_text:
            description += page_text
            
        else:
            image_list = page.get_images(full=True)
            for image in image_list:
                xref = image[0]
                base_image = pdf.extract_image(xref)
                image_bytes = base_image["image"]
                image = Image.open(image_bytes)
                image_text = tess.image_to_string(image)
                description += image_text
                
    return description

def highligh_words(text, imp_words):
    '''Fonction pour mettre en évidence les mots importants dans le texte'''
    text = text.lower()
    # Enlever les balises HTML
    text = re.sub(r'<.*?>', '', text)
    #text = re.sub(r'[^\w\s]', '', text)
    list_words = text.split()
    imp_words = imp_words.split()
    highlighted_text =  []
    for word in list_words:
        if word in imp_words:
            highlighted_text.append((word, "", "#afa"))
            highlighted_text.append(" ")
        else: 
            highlighted_text.append(word)
            highlighted_text.append(" ")
    return highlighted_text

@st.cache_resource
def load_data_test():
    return pd.read_csv("../EFREI_LIPSTIP_50k_elements_EPO_clean.csv")

if __name__ == "__main__":
    st.set_page_config(page_title="Projet LIPSTIP - EFREI", page_icon="./img/logo_explain.png")
    df_test = load_data_test()
    createdDisplay(df_test)