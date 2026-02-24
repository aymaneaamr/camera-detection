import streamlit as st
import cv2
import numpy as np
from collections import defaultdict
import pandas as pd
from datetime import datetime
import json
import base64
from io import BytesIO
import openpyxl
from openpyxl.styles import Font, Alignment, PatternFill
from pyzbar.pyzbar import decode
import re

# Configuration de la page
st.set_page_config(
    page_title="Gestionnaire d'Inventaire Multi-Articles",
    page_icon="📦",
    layout="wide"
)

# CSS personnalisé
st.markdown("""
<style>
    .barcode-scanner {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 1rem;
    }
    .success-box {
        background: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
    }
    .code-display {
        background: #2c3e50;
        color: #ecf0f1;
        padding: 1rem;
        border-radius: 5px;
        font-family: monospace;
        font-size: 1.5rem;
        text-align: center;
        margin: 1rem 0;
    }
    .location-badge {
        background: #17a2b8;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 5px;
        font-size: 0.8rem;
        margin-left: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

class GestionnaireArticles:
    def __init__(self):
        """Initialise le gestionnaire d'articles"""
        self.articles = {}  # {nom_article: {"photos": [], "emplacement": "", "libelle": ""}}
    
    def creer_nouvel_article(self, nom_article, emplacement="", libelle=""):
        """Crée un nouvel article dans l'inventaire"""
        if nom_article and nom_article not in self.articles:
            self.articles[nom_article] = {
                'photos': [],
                'emplacement': emplacement,
                'libelle': libelle,
                'date_creation': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            return True
        return False
    
    def ajouter_photo_article(self, nom_article, frame_original, frame_analyse, nb_articles):
        """Ajoute une photo analysée à un article existant"""
        if nom_article in self.articles:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            _, buffer_original = cv2.imencode('.jpg', frame_original)
            _, buffer_analyse = cv2.imencode('.jpg', frame_analyse)
            
            photo_data = {
                'timestamp': timestamp,
                'nb_articles': nb_articles,
                'image_originale': base64.b64encode(buffer_original).decode('utf-8'),
                'image_analyse': base64.b64encode(buffer_analyse).decode('utf-8'),
                'id': len(self.articles[nom_article]['photos'])
            }
            self.articles[nom_article]['photos'].append(photo_data)
            return True
        return False
    
    def get_total_article(self, nom_article):
        if nom_article in self.articles:
            return sum(photo['nb_articles'] for photo in self.articles[nom_article]['photos'])
        return 0

    def generer_excel(self):
        output = BytesIO()
        workbook = openpyxl.Workbook()
        
        # Feuille Résumé
        sheet_resume = workbook.active
        sheet_resume.title = "Inventaire"
        headers = ["Nom de l'article", "Libellé", "Emplacement", "Quantité totale", "Nombre de photos", "Dernière MAJ"]
        
        for col, header in enumerate(headers, 1):
            cell = sheet_resume.cell(row=1, column=col)
            cell.value = header
            cell.font = Font(color="FFFFFF", bold=True)
            cell.fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
        
        row = 2
        for nom, data in self.articles.items():
            total = sum(p['nb_articles'] for p in data['photos'])
            sheet_resume.cell(row=row, column=1).value = nom
            sheet_resume.cell(row=row, column=2).value = data.get('libelle', '')
            sheet_resume.cell(row=row, column=3).value = data.get('emplacement', '')
            sheet_resume.cell(row=row, column=4).value = total
            sheet_resume.cell(row=row, column=5).value = len(data['photos'])
            sheet_resume.cell(row=row, column=6).value = data['photos'][-1]['timestamp'] if data['photos'] else data.get('date_creation')
            row += 1
        
        workbook.save(output)
        output.seek(0)
        return output

# --- Fonctions de détection (inchangées mais renommées pour la cohérence) ---
def detecter_code_barre(image):
    resultat = image.copy()
    codes_detectes = []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    codes = decode(gray)
    for code in codes:
        data = code.data.decode('utf-8')
        codes_detectes.append({'data': data, 'type': code.type})
    return resultat, codes_detectes

def detecter_articles(image):
    resultat = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valides = [c for c in contours if cv2.contourArea(c) > 200]
    for c in valides:
        cv2.drawContours(resultat, [c], -1, (0, 255, 0), 2)
    return resultat, len(valides)

def base64_to_image(base64_string):
    img_data = base64.b64decode(base64_string)
    nparr = np.frombuffer(img_data, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

# --- Initialisation Session State ---
if 'gestionnaire' not in st.session_state:
    st.session_state.gestionnaire = GestionnaireArticles()
if 'page' not in st.session_state:
    st.session_state.page = "saisie"

gst = st.session_state.gestionnaire

# --- Interface Latérale ---
with st.sidebar:
    st.header("📋 Liste des Articles")
    for nom in gst.articles.keys():
        if st.button(f"📦 {nom}", key=f"sel_{nom}", use_container_width=True):
            st.session_state.article_selectionne = nom
            st.session_state.page = "details"
    
    st.divider()
    if st.button("➕ Nouvel Article", use_container_width=True):
        st.session_state.page = "saisie"
        st.session_state.scan_effectue = False
        st.rerun()

    if gst.articles:
        excel = gst.generer_excel()
        st.download_button("📥 Export Excel", data=excel, file_name="inventaire.xlsx", use_container_width=True)

# --- Contenu Principal ---
if st.session_state.page == "saisie":
    st.header("➕ Ajouter un article")
    
    # Zone Scan
    st.markdown('<div class="barcode-scanner">', unsafe_allow_html=True)
    img_barcode = st.camera_input("Scanner le code-barres")
    if img_barcode:
        frame = cv2.imdecode(np.frombuffer(img_barcode.getvalue(), np.uint8), cv2.IMREAD_COLOR)
        _, codes = detecter_code_barre(frame)
        if codes:
            st.session_state.code_detecte = codes[0]['data']
            st.success(f"Code détecté : {st.session_state.code_detecte}")
    st.markdown('</div>', unsafe_allow_html=True)

    # Formulaire
    col_nom, col_lib, col_emp = st.columns([1.5, 1.5, 1])
    with col_nom:
        nom_art = st.text_input("Référence / Nom *", value=st.session_state.get('code_detecte', ""))
    with col_lib:
        libelle_art = st.text_input("Libellé (optionnel)", placeholder="Description courte")
    with col_emp:
        emp_art = st.text_input("Emplacement", placeholder="Rayon/Bac")

    if st.button("✅ Créer l'article"):
        if nom_art:
            if gst.creer_nouvel_article(nom_art, emp_art, libelle_art):
                st.session_state.article_selectionne = nom_art
                st.session_state.page = "details"
                st.rerun()
            else:
                st.error("Cet article existe déjà.")
        else:
            st.error("Le nom est obligatoire.")

elif st.session_state.page == "details" and st.session_state.get('article_selectionne'):
    nom = st.session_state.article_selectionne
    data = gst.articles[nom]
    
    st.header(f"📦 Article : {nom}")
    if data['libelle']: st.subheader(f"📝 {data['libelle']}")
    
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Quantité Totale", gst.get_total_article(nom))
    col_m2.metric("Emplacement", data['emplacement'] if data['emplacement'] else "Non défini")
    col_m3.metric("Photos", len(data['photos']))

    # Ajouter photo
    uploaded = st.file_uploader("Ajouter une photo pour comptage", type=['jpg','png'])
    if uploaded:
        frame = cv2.imdecode(np.frombuffer(uploaded.read(), np.uint8), cv2.IMREAD_COLOR)
        res, count = detecter_articles(frame)
        if gst.ajouter_photo_article(nom, frame, res, count):
            st.success(f"{count} éléments détectés !")
            st.rerun()

    # Grille de photos
    if data['photos']:
        cols = st.columns(3)
        for i, p in enumerate(data['photos']):
            with cols[i % 3]:
                img = base64_to_image(p['image_analyse'])
                st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=f"{p['nb_articles']} articles - {p['timestamp']}")
