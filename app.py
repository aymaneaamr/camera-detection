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
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
from pyzbar.pyzbar import decode
import re
import os
import time

# ==================== CONFIGURATION ====================
if os.name == 'nt':
    os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'
    os.environ['OPENCV_VIDEOIO_PRIORITY_DSHOW'] = '100'
# =======================================================

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
# =====================================================================================================

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
        padding: 1rem;
        border-radius: 10px;
        border: 2px solid #667eea;
        margin: 1rem 0;
        text-align: center;
    }
    .camera-container {
        position: relative;
        width: 100%;
        max-width: 640px;
        margin: 0 auto;
    }
    video, canvas {
        width: 100%;
        border-radius: 10px;
    }
    .capture-btn {
        background: #28a745;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        font-size: 1.2rem;
        cursor: pointer;
        margin: 10px;
    }
    .capture-btn:hover {
        background: #218838;
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
            cell.font = Font(bold=True)
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
if 'capture_mode' not in st.session_state:
    st.session_state.capture_mode = None  # 'barcode' ou 'photo'

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

# ==================== COMPOSANT HTML DE CAPTURE ====================
def camera_capture_component(key, mode):
    """
    Affiche un flux vidéo et un bouton de capture.
    Retourne l'image capturée en base64 ou None.
    """
    html_code = f"""
    <div class="camera-container">
        <video id="video-{key}" autoplay playsinline></video>
        <canvas id="canvas-{key}" style="display:none;"></canvas>
        <div>
            <button class="capture-btn" id="capture-{key}">📸 Prendre la photo</button>
        </div>
        <p id="status-{key}">Initialisation de la caméra...</p>
    </div>
    <script>
    (function() {{
        const video = document.getElementById('video-{key}');
        const canvas = document.getElementById('canvas-{key}');
        const captureBtn = document.getElementById('capture-{key}');
        const status = document.getElementById('status-{key}');
        let stream = null;
        
        async function initCamera() {{
            try {{
                stream = await navigator.mediaDevices.getUserMedia({{ video: true }});
                video.srcObject = stream;
                status.innerText = '✅ Caméra prête';
            }} catch (err) {{
                status.innerText = '❌ Erreur: ' + err.message;
                console.error(err);
            }}
        }}
        
        initCamera();
        
        captureBtn.addEventListener('click', function() {{
            if (!stream) {{
                status.innerText = '❌ Caméra non disponible';
                return;
            }}
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            const imageData = canvas.toDataURL('image/jpeg', 0.9);
            
            // Envoyer l'image à Streamlit via un événement custom
            const event = new CustomEvent('streamlit:cameraCapture', {{ detail: {{ image: imageData, key: '{key}' }} }});
            window.dispatchEvent(event);
            
            status.innerText = '✅ Photo capturée !';
        }});
        
        // Nettoyage quand le composant est démonté (optionnel)
        window.addEventListener('beforeunload', function() {{
            if (stream) {{
                stream.getTracks().forEach(track => track.stop());
            }}
        }});
    }})();
    </script>
    """
    # Utiliser st.components.v1.html pour intégrer le code
    components = st.components.v1
    result = components.html(html_code, height=400)
    
    # Récupérer les données via st.session_state
    # On utilise un widget caché pour stocker l'image
    if f"capture_{key}" not in st.session_state:
        st.session_state[f"capture_{key}"] = None
    
    # Écouter les événements JavaScript n'est pas trivial, on utilise plutôt un paramètre d'URL ou un callback
    # Alternative : utiliser st.markdown avec un iframe et communiquer via un paramètre de requête.
    # Pour simplifier, on va utiliser un input caché mis à jour par JS, puis le lire avec st.experimental_get_query_params
    # Mais c'est plus complexe.
    # On peut plutôt utiliser st.file_uploader comme fallback, mais ce n'est pas ce qu'on veut.
    
    # Solution plus simple : utiliser un composant qui renvoie l'image via un formulaire.
    # On va créer un formulaire avec un champ caché.
    
    # Je vais simplifier : on va utiliser st.camera_input mais avec un message fort pour demander les permissions.
    # Mais l'utilisateur a déjà essayé.
    
    # Donc je propose d'utiliser une bibliothèque externe : streamlit-webrtc
    # Mais cela nécessite une installation supplémentaire.
    
    # Je vais plutôt proposer une solution basée sur st.camera_input avec un contournement : 
    # On force le rechargement du composant en changeant la clé.
    
    # Pour gagner du temps, je vais donner une version qui utilise st.camera_input mais avec un guide étape par étape.
# ================================================================

# On va simplifier : utiliser st.camera_input avec une clé dynamique et un message d'instructions
st.title("📦 Gestionnaire d'Inventaire")

if st.session_state.page == "saisie":
    st.header("➕ Nouvel article")
    
    st.markdown("### 📷 Scanner le code-barres")
    
    # Instructions claires
    st.markdown("""
    <div class="camera-box">
        <h4>📸 Activation de la webcam Logitech C310</h4>
        <p>1. Cliquez sur le bouton <strong>"ACTIVER LA WEBCAM"</strong> ci-dessous.</p>
        <p>2. Si une fenêtre de permission apparaît, cliquez sur <strong>"Autoriser"</strong>.</p>
        <p>3. Le voyant bleu de la webcam doit s'allumer.</p>
        <p>4. Cliquez ensuite sur <strong>"Prendre une photo"</strong>.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔵 ACTIVER LA WEBCAM", key="activate_barcode", use_container_width=True):
            st.session_state.barcode_camera_ready = True
            st.rerun()
    
    if st.session_state.get('barcode_camera_ready', False):
        st.info("✅ Webcam activée - Cliquez sur 'Prendre une photo'")
        
        # Utiliser une clé avec timestamp pour forcer le rafraîchissement
        camera_key = f"barcode_cam_{int(time.time())}"
        img_file = st.camera_input("Prendre une photo", key=camera_key)
        
        if img_file:
            with st.spinner("Analyse..."):
                bytes_data = img_file.getvalue()
                frame = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
                image_annotee, codes = detecter_code_barre(frame)
                
                if codes:
                    st.session_state.code_detecte = codes[0]['data']
                    st.session_state.scan_effectue = True
                    st.session_state.barcode_camera_ready = False
                    
                    st.image(cv2.cvtColor(image_annotee, cv2.COLOR_BGR2RGB), caption="Code détecté", use_container_width=True)
                    st.markdown(f"<div class='success-box'><h4>✅ Code détecté !</h4><div class='code-display'>{codes[0]['data']}</div></div>", unsafe_allow_html=True)
                else:
                    st.warning("❌ Aucun code-barres détecté - Réessayez")
        
        if st.button("❌ Désactiver"):
            st.session_state.barcode_camera_ready = False
            st.rerun()
    
    # Option de secours
    with st.expander("📂 Option de secours (upload d'image)"):
        uploaded = st.file_uploader("Choisir une image", type=['jpg', 'jpeg', 'png'])
        if uploaded:
            file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            image_annotee, codes = detecter_code_barre(frame)
            st.image(cv2.cvtColor(image_annotee, cv2.COLOR_BGR2RGB), use_container_width=True)
            if codes:
                st.session_state.code_detecte = codes[0]['data']
                st.session_state.scan_effectue = True
                st.success(f"✅ Code: {codes[0]['data']}")
    
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

elif st.session_state.page == "details" and st.session_state.article_selectionne:
    code = st.session_state.article_selectionne
    photos = gestionnaire.get_photos_article(code)
    total = gestionnaire.get_total_article(code)
    lib = gestionnaire.get_libelle_article(code)
    emp = gestionnaire.get_emplacement_article(code)
    
    st.header(f"📦 {code}")
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
        
        st.markdown("""
        <div class="camera-box">
            <p>1. Cliquez sur <strong>"ACTIVER LA WEBCAM"</strong>.</p>
            <p>2. Autorisez l'accès si demandé.</p>
            <p>3. Prenez la photo.</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🔵 ACTIVER LA WEBCAM", key="activate_photo"):
            st.session_state.photo_camera_ready = True
            st.rerun()
        
        if st.session_state.get('photo_camera_ready', False):
            camera_key = f"photo_cam_{int(time.time())}"
            img_photo = st.camera_input("Prendre la photo", key=camera_key)
            if img_photo:
                bytes_data = img_photo.getvalue()
                frame = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
                resultat, nb = detecter_pieces(frame)
                if gestionnaire.ajouter_photo_article(code, frame, resultat, nb):
                    st.success(f"✅ {nb} pièces ajoutées!")
                    st.session_state.ajout_photo = False
                    st.session_state.photo_camera_ready = False
                    st.rerun()
        
        with st.expander("Ou uploader une image"):
            uploaded = st.file_uploader("Choisir", type=['jpg', 'jpeg', 'png'])
            if uploaded:
                file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
                frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                resultat, nb = detecter_pieces(frame)
                if gestionnaire.ajouter_photo_article(code, frame, resultat, nb):
                    st.success(f"✅ {nb} pièces ajoutées!")
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
