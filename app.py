import streamlit as st
import cv2
import numpy as np
from datetime import datetime
import base64
import openpyxl
from io import BytesIO
from pyzbar.pyzbar import decode
import os
import subprocess
import time

# ==================== Dictionnaire des articles prédéfinis ====================
ARTICLES_PREDEFINIS = {
    "10751037": {
        "libelle": "Capacitor E54.G85-203G30 Un 1260 V DC / 750 AC MKP 20µF",
        "emplacement": "A191"
    },
    "10751038": {
        "libelle": "Contacteur principal Bipolaire",
        "emplacement": "A204"
    },
    "10751039": {
        "libelle": "Contacteur de précharge Bipolaire",
        "emplacement": "A204"
    },
    "10751040": {
        "libelle": "Coupe circuit 1A, 480VAC, 3Poles",
        "emplacement": "A192"
    },
    "10751050": {
        "libelle": "Cosse à sertir 50x8",
        "emplacement": "A194"
    }
}

st.set_page_config(page_title="Gestionnaire d'Inventaire", page_icon="📦", layout="wide")

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
    .camera-box {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #667eea;
        margin: 1rem 0;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

class GestionnairePieces:
    def __init__(self):
        self.articles = {}
    
    def creer_nouvel_article(self, code_article, libelle="", emplacement=""):
        if code_article and code_article not in self.articles:
            if code_article in ARTICLES_PREDEFINIS:
                if not libelle:
                    libelle = ARTICLES_PREDEFINIS[code_article]["libelle"]
                if not emplacement:
                    emplacement = ARTICLES_PREDEFINIS[code_article]["emplacement"]
            self.articles[code_article] = {
                'libelle': libelle,
                'photos': [],
                'emplacement': emplacement,
                'date_creation': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            return True
        return False
    
    def ajouter_photo_article(self, code_article, frame_original, frame_analyse, nb_pieces):
        if code_article in self.articles:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            _, buffer_original = cv2.imencode('.jpg', frame_original)
            _, buffer_analyse = cv2.imencode('.jpg', frame_analyse)
            photo_data = {
                'timestamp': timestamp,
                'nb_pieces': nb_pieces,
                'image_originale': base64.b64encode(buffer_original).decode('utf-8'),
                'image_analyse': base64.b64encode(buffer_analyse).decode('utf-8'),
                'id': len(self.articles[code_article]['photos'])
            }
            self.articles[code_article]['photos'].append(photo_data)
            return True
        return False
    
    def get_total_article(self, code_article):
        if code_article in self.articles:
            return sum(p['nb_pieces'] for p in self.articles[code_article]['photos'])
        return 0
    
    def get_photos_article(self, code_article):
        return self.articles.get(code_article, {}).get('photos', [])
    
    def get_emplacement_article(self, code_article):
        return self.articles.get(code_article, {}).get('emplacement', '')
    
    def get_libelle_article(self, code_article):
        return self.articles.get(code_article, {}).get('libelle', '')
    
    def supprimer_photo(self, code_article, photo_id):
        if code_article in self.articles and 0 <= photo_id < len(self.articles[code_article]['photos']):
            del self.articles[code_article]['photos'][photo_id]
            for i, p in enumerate(self.articles[code_article]['photos']):
                p['id'] = i
            return True
        return False
    
    def supprimer_article(self, code_article):
        if code_article in self.articles:
            del self.articles[code_article]
            return True
        return False
    
    def get_tous_les_totaux(self):
        return {code: self.get_total_article(code) for code in self.articles}
    
    def generer_excel(self):
        output = BytesIO()
        workbook = openpyxl.Workbook()
        sheet = workbook.active
        sheet.title = "Inventaire"
        headers = ["Code Article", "Libellé", "Emplacement", "Quantité totale", "Photos", "Date création"]
        for col, h in enumerate(headers, 1):
            cell = sheet.cell(row=1, column=col)
            cell.value = h
            cell.font = openpyxl.styles.Font(bold=True)
        row = 2
        for code, data in self.articles.items():
            total = sum(p['nb_pieces'] for p in data['photos'])
            sheet.cell(row=row, column=1).value = code
            sheet.cell(row=row, column=2).value = data.get('libelle', '')
            sheet.cell(row=row, column=3).value = data.get('emplacement', '')
            sheet.cell(row=row, column=4).value = total
            sheet.cell(row=row, column=5).value = len(data['photos'])
            sheet.cell(row=row, column=6).value = data.get('date_creation', '')
            row += 1
        workbook.save(output)
        output.seek(0)
        return output
    
    def reinitialiser_tout(self):
        self.articles = {}

def detecter_code_barre(image):
    resultat = image.copy()
    codes_detectes = []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    codes = decode(gray)
    for code in codes:
        data = code.data.decode('utf-8')
        type_code = code.type
        points = code.polygon
        if len(points) == 4:
            pts = np.array([(p.x, p.y) for p in points], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(resultat, [pts], True, (0, 255, 0), 3)
        cv2.putText(resultat, f"{type_code}: {data}", (code.rect.left, code.rect.top-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        codes_detectes.append({'data': data, 'type': type_code})
    return resultat, codes_detectes

def detecter_pieces(image):
    resultat = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)
    kernel = np.ones((3,3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)
    edges = cv2.erode(edges, kernel, iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pieces = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 200:
            pieces.append(cnt)
    nb = len(pieces)
    for cnt in pieces:
        cv2.drawContours(resultat, [cnt], -1, (0,255,0), 2)
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"]/M["m00"])
            cy = int(M["m01"]/M["m00"])
            cv2.circle(resultat, (cx,cy), 3, (0,0,255), -1)
    cv2.putText(resultat, f"Pieces: {nb}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    return resultat, nb

def base64_to_image(b64):
    img_data = base64.b64decode(b64)
    nparr = np.frombuffer(img_data, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

# Initialisation session
if 'gestionnaire' not in st.session_state:
    st.session_state.gestionnaire = GestionnairePieces()
if 'page' not in st.session_state:
    st.session_state.page = "saisie"
if 'article_selectionne' not in st.session_state:
    st.session_state.article_selectionne = None
if 'photo_selectionnee' not in st.session_state:
    st.session_state.photo_selectionnee = None
if 'code_detecte' not in st.session_state:
    st.session_state.code_detecte = None
if 'scan_effectue' not in st.session_state:
    st.session_state.scan_effectue = False

gestionnaire = st.session_state.gestionnaire

# Barre latérale
with st.sidebar:
    st.header("📋 Articles")
    if gestionnaire.articles:
        for code in gestionnaire.articles.keys():
            total = gestionnaire.get_total_article(code)
            lib = gestionnaire.get_libelle_article(code)
            emp = gestionnaire.get_emplacement_article(code)
            if st.button(f"📦 {code} ({total})", key=f"side_{code}", use_container_width=True):
                st.session_state.article_selectionne = code
                st.session_state.page = "details"
                st.rerun()
            if lib or emp:
                st.caption(f"{lib[:30]}... {emp}")
        st.divider()
        if st.button("➕ Nouvel article", use_container_width=True):
            st.session_state.page = "saisie"
            st.session_state.article_selectionne = None
            st.rerun()
        if gestionnaire.articles:
            st.divider()
            excel = gestionnaire.generer_excel()
            st.download_button("📥 Excel", excel, "inventaire.xlsx", use_container_width=True)
            if st.button("🔄 Reset", type="primary", use_container_width=True):
                gestionnaire.reinitialiser_tout()
                st.rerun()
    else:
        st.info("Aucun article")

# ==================== PAGE DE SAISIE ====================
if st.session_state.page == "saisie":
    st.title("📦 Gestionnaire d'Inventaire")
    st.header("➕ Nouvel article")
    
    st.markdown("### 📷 Scanner le code-barres")
    
    # Solution avec l'application Windows Camera
    st.markdown("""
    <div class="camera-box">
        <h4>📸 Utilisation de votre webcam Logitech C310</h4>
        <p>La webcam fonctionne mais le composant Streamlit a un problème. Voici une méthode alternative :</p>
        <ol>
            <li>Cliquez sur le bouton ci-dessous pour ouvrir l'application <strong>Camera Windows</strong>.</li>
            <li>Prenez une photo du code-barres avec cette application.</li>
            <li>La photo est automatiquement sauvegardée dans votre dossier <strong>Images</strong>.</li>
            <li>Revenez ici et uploadez la photo.</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("📸 Ouvrir l'application Camera Windows", use_container_width=True):
        try:
            subprocess.Popen('start microsoft.windows.camera:', shell=True)
            st.success("✅ Application Camera ouverte ! Prenez votre photo, puis revenez.")
        except Exception as e:
            st.error(f"Erreur : {e}")
    
    uploaded_barcode = st.file_uploader("Uploader la photo du code-barres", type=['jpg', 'jpeg', 'png'], key="barcode_upload")
    if uploaded_barcode:
        with st.spinner("Analyse..."):
            file_bytes = np.asarray(bytearray(uploaded_barcode.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            image_annotee, codes = detecter_code_barre(frame)
            st.image(cv2.cvtColor(image_annotee, cv2.COLOR_BGR2RGB), use_column_width=True)
            if codes:
                st.session_state.code_detecte = codes[0]['data']
                st.session_state.scan_effectue = True
                st.markdown(f"<div class='success-box'><h4>✅ Code détecté !</h4><div class='code-display'>{codes[0]['data']}</div></div>", unsafe_allow_html=True)
            else:
                st.warning("❌ Aucun code-barres détecté")
    
    st.markdown("---")
    st.markdown("### 📝 Informations article")
    
    default_code = st.session_state.code_detecte if st.session_state.code_detecte else ""
    code_article = st.text_input("Code article *", value=default_code)
    
    if code_article and code_article in ARTICLES_PREDEFINIS:
        libelle = st.text_input("Libellé", value=ARTICLES_PREDEFINIS[code_article]["libelle"])
        emplacement = st.text_input("Emplacement", value=ARTICLES_PREDEFINIS[code_article]["emplacement"])
        st.info(f"📍 Emplacement suggéré: {ARTICLES_PREDEFINIS[code_article]['emplacement']}")
    else:
        libelle = st.text_input("Libellé (optionnel)")
        emplacement = st.text_input("Emplacement (optionnel)")
    
    if st.button("✅ Créer l'article", use_container_width=True):
        if code_article:
            if gestionnaire.creer_nouvel_article(code_article, libelle, emplacement):
                st.success("✅ Article créé!")
                st.session_state.article_selectionne = code_article
                st.session_state.page = "details"
                st.session_state.code_detecte = None
                st.session_state.scan_effectue = False
                st.rerun()
            else:
                st.error("❌ Code existe déjà")
        else:
            st.error("❌ Code requis")

# ==================== PAGE DÉTAILS ====================
elif st.session_state.page == "details" and st.session_state.article_selectionne:
    code = st.session_state.article_selectionne
    photos = gestionnaire.get_photos_article(code)
    total = gestionnaire.get_total_article(code)
    lib = gestionnaire.get_libelle_article(code)
    emp = gestionnaire.get_emplacement_article(code)
    
    st.title(f"📦 {code}")
    if lib:
        st.subheader(lib)
    if emp:
        st.info(f"📍 {emp}")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total", total)
    with col2:
        st.metric("Photos", len(photos))
    
    col_a1, col_a2, col_a3 = st.columns(3)
    with col_a1:
        if st.button("⬅️ Retour"):
            st.session_state.page = "saisie"
            st.rerun()
    with col_a2:
        if st.button("📸 Ajouter photo"):
            st.session_state.ajout_photo = True
    with col_a3:
        if st.button("🗑️ Supprimer", type="primary"):
            gestionnaire.supprimer_article(code)
            st.session_state.page = "saisie"
            st.rerun()
    
    if st.session_state.get('ajout_photo', False):
        st.divider()
        st.subheader("📸 Ajouter une photo")
        
        # Même principe avec l'application Camera
        st.markdown("""
        <div class="camera-box">
            <p>Utilisez l'application Camera Windows pour prendre la photo des pièces.</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("📸 Ouvrir l'application Camera", key="open_cam_photo"):
            subprocess.Popen('start microsoft.windows.camera:', shell=True)
            st.info("Prenez la photo, puis uploadez-la.")
        
        uploaded_photo = st.file_uploader("Uploader la photo", type=['jpg', 'jpeg', 'png'], key="photo_upload")
        if uploaded_photo:
            file_bytes = np.asarray(bytearray(uploaded_photo.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            resultat, nb = detecter_pieces(frame)
            if gestionnaire.ajouter_photo_article(code, frame, resultat, nb):
                st.success(f"✅ {nb} pièces détectées et ajoutées!")
                st.session_state.ajout_photo = False
                st.rerun()
        
        if st.button("❌ Annuler"):
            st.session_state.ajout_photo = False
            st.rerun()
    
    if photos:
        st.divider()
        st.subheader("📸 Photos")
        cols = st.columns(3)
        for i, photo in enumerate(photos):
            with cols[i % 3]:
                img = base64_to_image(photo['image_analyse'])
                img = cv2.resize(img, (200,150))
                st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_column_width=True)
                st.caption(f"{photo['nb_pieces']} pièces")
                if st.button("🗑️", key=f"del_{i}"):
                    gestionnaire.supprimer_photo(code, i)
                    st.rerun()

# Pied de page
st.markdown("---")
st.caption(f"📦 Total global: {sum(gestionnaire.get_tous_les_totaux().values())} pièces | {len(gestionnaire.articles)} articles")
