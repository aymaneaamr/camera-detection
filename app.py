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
import subprocess
import platform
import tempfile

# ==================== CONFIGURATION ====================
if os.name == 'nt':  # Windows
    os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'
    os.environ['OPENCV_VIDEOIO_PRIORITY_DSHOW'] = '100'
# =======================================================

# ==================== Dictionnaire des articles prédéfinis avec leurs emplacements ====================
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

# Configuration de la page
st.set_page_config(
    page_title="Gestionnaire d'Inventaire Multi-Pièces",
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
    .barcode-result {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
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
    .label-badge {
        background: #28a745;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 5px;
        font-size: 0.8rem;
        margin-left: 0.5rem;
    }
    .article-found {
        background: #cce5ff;
        color: #004085;
        padding: 0.5rem;
        border-radius: 5px;
        border-left: 5px solid #004085;
        margin: 0.5rem 0;
    }
    .camera-info {
        background: #e3f2fd;
        color: #0d47a1;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
        font-size: 1rem;
        text-align: center;
        border: 2px solid #2196f3;
    }
    .camera-selector {
        background: #f5f5f5;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .warning-box {
        background: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
    }
    .info-box {
        background: #d1ecf1;
        color: #0c5460;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #17a2b8;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ==================== FONCTIONS DE DÉTECTION DES CAMÉRAS ====================
def detecter_toutes_cameras_agressif():
    """
    Version agressive de détection des caméras - teste toutes les combinaisons possibles
    """
    cameras = []
    
    # Liste de tous les backends possibles
    backends = [
        (cv2.CAP_DSHOW, "DirectShow"),
        (cv2.CAP_MSMF, "Media Foundation"),
        (cv2.CAP_ANY, "Auto"),
        (cv2.CAP_VFW, "VFW"),
        (cv2.CAP_FFMPEG, "FFMPEG"),
        (cv2.CAP_IMAGES, "Images"),
        (cv2.CAP_OPENCV_MJPEG, "MJPEG"),
    ]
    
    # Tester les index 0 à 9
    for index in range(10):
        for backend, nom_backend in backends:
            try:
                # Essayer d'ouvrir la caméra
                cap = cv2.VideoCapture(index, backend)
                
                if cap.isOpened():
                    # Attendre l'initialisation
                    time.sleep(0.5)
                    
                    # Lire plusieurs frames pour être sûr
                    frames_valides = 0
                    frames_total = 0
                    
                    for _ in range(5):
                        ret, frame = cap.read()
                        if ret and frame is not None and frame.size > 0:
                            frames_valides += 1
                        frames_total += 1
                        time.sleep(0.1)
                    
                    # Si au moins 3 frames sur 5 sont valides
                    if frames_valides >= 3:
                        # Récupérer les propriétés
                        largeur = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        hauteur = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        
                        camera_info = {
                            'index': index,
                            'disponible': True,
                            'nom': f"Caméra {index} ({nom_backend})",
                            'resolution': f"{largeur}x{hauteur}",
                            'fps': f"{fps:.1f}" if fps > 0 else "Inconnu",
                            'backend': nom_backend,
                            'frames_valides': frames_valides
                        }
                        
                        # Éviter les doublons
                        if not any(c['index'] == index and c['backend'] == nom_backend for c in cameras):
                            cameras.append(camera_info)
                    
                    cap.release()
            except Exception as e:
                continue
    
    return cameras

def test_camera_simple():
    """
    Test simple pour voir si la caméra 0 fonctionne
    """
    resultats = []
    
    # Test avec différents backends
    backends = [
        (cv2.CAP_DSHOW, "DirectShow"),
        (cv2.CAP_MSMF, "Media Foundation"),
        (cv2.CAP_ANY, "Auto"),
    ]
    
    for backend, nom in backends:
        try:
            cap = cv2.VideoCapture(0, backend)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    resultats.append(f"✅ {nom}: OK")
                else:
                    resultats.append(f"⚠️ {nom}: Ouvert mais pas d'image")
                cap.release()
            else:
                resultats.append(f"❌ {nom}: Non ouvert")
        except Exception as e:
            resultats.append(f"❌ {nom}: Erreur {str(e)[:30]}")
    
    return resultats

# ============================================================================

class GestionnairePieces:
    def __init__(self):
        """Initialise le gestionnaire de pièces"""
        self.articles = {}
        self.reset_article_courant()
    
    def reset_article_courant(self):
        """Réinitialise l'article en cours de saisie"""
        self.article_courant = {
            'code': '',
            'libelle': '',
            'emplacement': '',
            'photos': [],
            'total_pieces': 0
        }
    
    def creer_nouvel_article(self, code_article, libelle="", emplacement=""):
        """Crée un nouvel article dans l'inventaire"""
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
        """Ajoute une photo analysée à un article existant"""
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
        """Retourne le total de pièces pour un article donné"""
        if code_article in self.articles:
            return sum(photo['nb_pieces'] for photo in self.articles[code_article]['photos'])
        return 0
    
    def get_photos_article(self, code_article):
        """Retourne toutes les photos d'un article"""
        if code_article in self.articles:
            return self.articles[code_article]['photos']
        return []
    
    def get_emplacement_article(self, code_article):
        """Retourne l'emplacement d'un article"""
        if code_article in self.articles:
            return self.articles[code_article].get('emplacement', '')
        return ''
    
    def get_libelle_article(self, code_article):
        """Retourne le libellé d'un article"""
        if code_article in self.articles:
            return self.articles[code_article].get('libelle', '')
        return ''
    
    def supprimer_photo(self, code_article, photo_id):
        """Supprime une photo d'un article"""
        if code_article in self.articles and 0 <= photo_id < len(self.articles[code_article]['photos']):
            del self.articles[code_article]['photos'][photo_id]
            for i, photo in enumerate(self.articles[code_article]['photos']):
                photo['id'] = i
            return True
        return False
    
    def supprimer_article(self, code_article):
        """Supprime complètement un article"""
        if code_article in self.articles:
            del self.articles[code_article]
            return True
        return False
    
    def get_tous_les_totaux(self):
        """Retourne un dictionnaire avec tous les totaux par article"""
        return {code: self.get_total_article(code) for code in self.articles}
    
    def get_tous_emplacements(self):
        """Retourne un dictionnaire avec tous les emplacements par article"""
        return {code: self.get_emplacement_article(code) for code in self.articles}
    
    def get_tous_libelles(self):
        """Retourne un dictionnaire avec tous les libellés par article"""
        return {code: self.get_libelle_article(code) for code in self.articles}
    
    def generer_excel(self):
        """Génère un fichier Excel avec l'inventaire complet"""
        output = BytesIO()
        workbook = openpyxl.Workbook()
        
        sheet_resume = workbook.active
        sheet_resume.title = "Inventaire"
        
        headers = ["Code Article", "Libellé", "Emplacement", "Quantité totale", "Nombre de photos", "Dernière mise à jour"]
        for col, header in enumerate(headers, 1):
            cell = sheet_resume.cell(row=1, column=col)
            cell.value = header
            cell.font = Font(bold=True)
            cell.fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
            cell.font = Font(color="FFFFFF", bold=True)
            cell.alignment = Alignment(horizontal="center")
        
        row = 2
        for code_article, data in self.articles.items():
            total = sum(p['nb_pieces'] for p in data['photos'])
            nb_photos = len(data['photos'])
            derniere_date = data['photos'][-1]['timestamp'] if data['photos'] else data.get('date_creation', 'N/A')
            emplacement = data.get('emplacement', '')
            libelle = data.get('libelle', '')
            
            sheet_resume.cell(row=row, column=1).value = code_article
            sheet_resume.cell(row=row, column=2).value = libelle
            sheet_resume.cell(row=row, column=3).value = emplacement
            sheet_resume.cell(row=row, column=4).value = total
            sheet_resume.cell(row=row, column=5).value = nb_photos
            sheet_resume.cell(row=row, column=6).value = derniere_date
            row += 1
        
        sheet_resume.column_dimensions['A'].width = 20
        sheet_resume.column_dimensions['B'].width = 30
        sheet_resume.column_dimensions['C'].width = 20
        sheet_resume.column_dimensions['D'].width = 15
        sheet_resume.column_dimensions['E'].width = 15
        sheet_resume.column_dimensions['F'].width = 22
        
        sheet_detail = workbook.create_sheet("Détail des photos")
        
        detail_headers = ["Code Article", "Libellé", "Emplacement", "Photo #", "Date", "Nombre de pièces"]
        for col, header in enumerate(detail_headers, 1):
            cell = sheet_detail.cell(row=1, column=col)
            cell.value = header
            cell.font = Font(bold=True)
            cell.fill = PatternFill(start_color="92D050", end_color="92D050", fill_type="solid")
            cell.alignment = Alignment(horizontal="center")
        
        row = 2
        for code_article, data in self.articles.items():
            libelle = data.get('libelle', '')
            emplacement = data.get('emplacement', '')
            for i, photo in enumerate(data['photos'], 1):
                sheet_detail.cell(row=row, column=1).value = code_article
                sheet_detail.cell(row=row, column=2).value = libelle
                sheet_detail.cell(row=row, column=3).value = emplacement
                sheet_detail.cell(row=row, column=4).value = f"Photo {i}"
                sheet_detail.cell(row=row, column=5).value = photo['timestamp']
                sheet_detail.cell(row=row, column=6).value = photo['nb_pieces']
                row += 1
        
        sheet_detail.column_dimensions['A'].width = 20
        sheet_detail.column_dimensions['B'].width = 30
        sheet_detail.column_dimensions['C'].width = 20
        sheet_detail.column_dimensions['D'].width = 12
        sheet_detail.column_dimensions['E'].width = 22
        sheet_detail.column_dimensions['F'].width = 18
        
        workbook.save(output)
        output.seek(0)
        return output
    
    def reinitialiser_tout(self):
        """Réinitialise complètement l'inventaire"""
        self.articles = {}

# Fonction pour détecter et lire les codes-barres
def detecter_code_barre(image):
    """Détecte et lit les codes-barres dans une image"""
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
        
        cv2.putText(resultat, f"{type_code}: {data}", 
                   (code.rect.left, code.rect.top - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        codes_detectes.append({
            'data': data,
            'type': type_code
        })
    
    return resultat, codes_detectes

# Fonction pour détecter les pièces dans une image
def detecter_pieces(image):
    """Détecte et compte les pièces dans une image"""
    resultat = image.copy()
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)
    edges = cv2.erode(edges, kernel, iterations=1)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    pieces_valides = []
    for contour in contours:
        aire = cv2.contourArea(contour)
        if aire > 200:
            pieces_valides.append(contour)
    
    nb_pieces = len(pieces_valides)
    
    for contour in pieces_valides:
        cv2.drawContours(resultat, [contour], -1, (0, 255, 0), 2)
        
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.circle(resultat, (cx, cy), 3, (0, 0, 255), -1)
    
    cv2.putText(resultat, f"Pieces: {nb_pieces}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    return resultat, nb_pieces

# Fonction pour décoder l'image base64
def base64_to_image(base64_string):
    img_data = base64.b64decode(base64_string)
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

# Fonction pour capturer depuis l'application Windows Camera
def capturer_avec_app_windows():
    """
    Utilise l'application Camera de Windows pour prendre une photo
    """
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown("""
    ### 📷 Mode de secours - Application Windows Camera
    
    **Pourquoi ?** OpenCV n'arrive pas à détecter votre webcam, mais elle fonctionne dans l'application Windows.
    
    **Instructions:**
    1. Cliquez sur le bouton ci-dessous pour ouvrir l'application Camera
    2. Prenez votre photo dans l'application
    3. La photo est automatiquement sauvegardée dans votre dossier "Pictures"
    4. Uploader la photo ici
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📸 Ouvrir l'application Camera", use_container_width=True):
            try:
                if platform.system() == "Windows":
                    # Ouvrir l'application Camera Windows
                    subprocess.Popen(['start', 'microsoft.windows.camera:'], shell=True)
                    st.success("✅ Application Camera ouverte!")
                    st.info("Prenez votre photo, puis revenez ici pour l'uploader.")
            except Exception as e:
                st.error(f"Erreur: {e}")
    
    with col2:
        st.markdown("**Ou utilisez votre téléphone:**")
        st.markdown("Prenez une photo avec votre téléphone et uploader-la")
    
    # Upload de l'image
    uploaded_file = st.file_uploader("Uploader la photo", type=['jpg', 'jpeg', 'png'], key="windows_camera_upload")
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        return frame
    
    return None

# Initialisation
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
if 'mode_secours' not in st.session_state:
    st.session_state.mode_secours = False

# Détection des caméras
with st.spinner("🔍 Recherche des caméras disponibles..."):
    cameras_trouvees = detecter_toutes_cameras_agressif()
    test_simple = test_camera_simple()

gestionnaire = st.session_state.gestionnaire

# Interface principale
st.title("📦 Gestionnaire d'Inventaire Multi-Pièces avec Scan Code-Barres")

# Barre latérale avec diagnostic
with st.sidebar:
    st.header("📋 Articles en inventaire")
    
    if gestionnaire.articles:
        for code_article in gestionnaire.articles.keys():
            total = gestionnaire.get_total_article(code_article)
            libelle = gestionnaire.get_libelle_article(code_article)
            emplacement = gestionnaire.get_emplacement_article(code_article)
            
            with st.container():
                col1, col2 = st.columns([3, 1])
                with col1:
                    if st.button(f"📦 {code_article}", key=f"select_{code_article}", use_container_width=True):
                        st.session_state.article_selectionne = code_article
                        st.session_state.page = "details"
                with col2:
                    st.write(f"**{total}**")
            
            if libelle or emplacement:
                badge_text = ""
                if libelle:
                    badge_text += f"📝 {libelle}"
                if libelle and emplacement:
                    badge_text += " | "
                if emplacement:
                    badge_text += f"📍 {emplacement}"
                if badge_text:
                    st.caption(badge_text)
        
        st.divider()
        
        if st.button("➕ Nouvel article", use_container_width=True):
            st.session_state.page = "saisie"
            st.session_state.article_selectionne = None
            st.session_state.code_detecte = None
            st.session_state.scan_effectue = False
            st.rerun()
        
        st.divider()
        
        if gestionnaire.articles:
            st.header("📊 Export")
            excel_file = gestionnaire.generer_excel()
            st.download_button(
                label="📥 Télécharger Excel",
                data=excel_file,
                file_name=f"inventaire_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
            
            if st.button("🔄 Tout réinitialiser", type="primary", use_container_width=True):
                gestionnaire.reinitialiser_tout()
                st.session_state.page = "saisie"
                st.session_state.article_selectionne = None
                st.rerun()
    else:
        st.info("Aucun article pour le moment")
    
    # Diagnostic
    st.divider()
    st.header("🔧 Diagnostic")
    
    if cameras_trouvees:
        st.success(f"✅ {len(cameras_trouvees)} caméra(s) détectée(s)")
        for cam in cameras_trouvees[:3]:  # Afficher les 3 premières
            st.text(f"📷 {cam['nom']} - {cam['resolution']}")
    else:
        st.error("❌ Aucune caméra détectée par OpenCV")
        st.info("""
        **Solutions:**
        1. Vérifiez qu'aucune autre app n'utilise la caméra
        2. Exécutez en tant qu'administrateur
        3. Utilisez le mode de secours ci-dessous
        """)
        
        if st.button("🆘 Activer le mode de secours", use_container_width=True):
            st.session_state.mode_secours = True
            st.rerun()

# Contenu principal
if st.session_state.page == "saisie":
    st.header("➕ Ajouter un nouvel article")
    
    # Mode de secours ou normal ?
    if st.session_state.mode_secours or not cameras_trouvees:
        st.markdown("""
        <div class="warning-box">
            <strong>⚠️ Mode de secours activé</strong><br>
            Utilisez l'application Windows Camera ou votre téléphone pour prendre les photos.
        </div>
        """, unsafe_allow_html=True)
        
        # Scan de code-barres en mode secours
        st.markdown("### 📷 Scanner le code-barres")
        
        uploaded_barcode = st.file_uploader("Uploader une photo du code-barres", type=['jpg', 'jpeg', 'png'], key="barcode_secours")
        if uploaded_barcode:
            with st.spinner("🔍 Analyse du code-barres..."):
                file_bytes = np.asarray(bytearray(uploaded_barcode.read()), dtype=np.uint8)
                frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                
                image_annotee, codes = detecter_code_barre(frame)
                st.image(cv2.cvtColor(image_annotee, cv2.COLOR_BGR2RGB), 
                        caption="Image analysée", use_container_width=True)
                
                if codes:
                    code_trouve = codes[0]['data']
                    st.session_state.code_detecte = code_trouve
                    st.session_state.scan_effectue = True
                    
                    st.markdown(f"""
                    <div class="success-box">
                        <h4>✅ Code-barres détecté !</h4>
                        <div class="code-display">{code_trouve}</div>
                        <p><strong>Type :</strong> {codes[0]['type']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning("❌ Aucun code-barres détecté")
        
        if st.button("🔄 Désactiver le mode de secours", use_container_width=True):
            st.session_state.mode_secours = False
            st.rerun()
    
    else:
        # Mode normal avec sélection de caméra
        st.markdown("### 📷 Scanner le code-barres")
        
        if cameras_trouvees:
            # Créer un sélecteur de caméra
            options = []
            for cam in cameras_trouvees:
                options.append(f"Caméra {cam['index']} - {cam['resolution']} ({cam['backend']})")
            
            camera_choice = st.selectbox("Choisissez votre caméra", options)
            
            if camera_choice:
                index = int(camera_choice.split(" - ")[0].replace("Caméra ", ""))
                
                if st.button(f"📷 Activer la caméra {index}", use_container_width=True):
                    st.session_state.camera_active = True
                    st.session_state.camera_index = index
                    st.rerun()
                
                if st.session_state.get('camera_active', False):
                    st.info(f"Caméra {index} activée")
                    
                    img_barcode = st.camera_input("Prendre une photo", key=f"cam_{index}")
                    if img_barcode:
                        with st.spinner("Analyse..."):
                            bytes_data = img_barcode.getvalue()
                            frame = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
                            
                            image_annotee, codes = detecter_code_barre(frame)
                            
                            if codes:
                                code_trouve = codes[0]['data']
                                st.session_state.code_detecte = code_trouve
                                st.session_state.scan_effectue = True
                                
                                st.image(cv2.cvtColor(image_annotee, cv2.COLOR_BGR2RGB), 
                                        caption="Code détecté", use_container_width=True)
                                
                                st.markdown(f"""
                                <div class="success-box">
                                    <h4>✅ Code détecté !</h4>
                                    <div class="code-display">{code_trouve}</div>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                st.session_state.camera_active = False
                            else:
                                st.warning("❌ Aucun code-barres détecté")
        else:
            # Fallback vers upload si pas de caméra
            st.warning("Aucune caméra détectée - Utilisation du mode upload")
            uploaded_barcode = st.file_uploader("Uploader une image", type=['jpg', 'jpeg', 'png'])
            if uploaded_barcode:
                with st.spinner("Analyse..."):
                    file_bytes = np.asarray(bytearray(uploaded_barcode.read()), dtype=np.uint8)
                    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    image_annotee, codes = detecter_code_barre(frame)
                    
                    if codes:
                        code_trouve = codes[0]['data']
                        st.session_state.code_detecte = code_trouve
                        st.session_state.scan_effectue = True
                        st.image(cv2.cvtColor(image_annotee, cv2.COLOR_BGR2RGB), 
                                caption="Code détecté", use_container_width=True)
                        
                        st.markdown(f"""
                        <div class="success-box">
                            <h4>✅ Code détecté !</h4>
                            <div class="code-display">{code_trouve}</div>
                        </div>
                        """, unsafe_allow_html=True)
    
    # Formulaire de création d'article
    st.markdown("---")
    st.markdown("### 📝 Informations de l'article")
    
    default_code = st.session_state.code_detecte if st.session_state.code_detecte else ""
    
    col_code, col_lib, col_emp = st.columns([2, 2, 1])
    
    with col_code:
        code_article = st.text_input("Code article *", value=default_code, key="code_input")
    
    with col_lib:
        if code_article and code_article in ARTICLES_PREDEFINIS:
            libelle_value = ARTICLES_PREDEFINIS[code_article]["libelle"]
        else:
            libelle_value = ""
        libelle = st.text_input("Libellé", value=libelle_value, key="libelle_input")
    
    with col_emp:
        if code_article and code_article in ARTICLES_PREDEFINIS:
            emp_value = ARTICLES_PREDEFINIS[code_article]["emplacement"]
        else:
            emp_value = ""
        emplacement = st.text_input("Emplacement", value=emp_value, key="emp_input")
    
    if code_article and code_article in ARTICLES_PREDEFINIS:
        st.info(f"📝 Article trouvé: {ARTICLES_PREDEFINIS[code_article]['libelle']}")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Créer l'article", use_container_width=True):
            if code_article:
                if gestionnaire.creer_nouvel_article(code_article, libelle, emplacement):
                    st.success(f"✅ Article créé!")
                    st.session_state.article_selectionne = code_article
                    st.session_state.page = "details"
                    st.session_state.code_detecte = None
                    st.session_state.scan_effectue = False
                    st.rerun()
                else:
                    st.error("❌ Erreur")
            else:
                st.error("❌ Code requis")
    
    with col2:
        if st.button("❌ Annuler", use_container_width=True):
            st.session_state.code_detecte = None
            st.session_state.scan_effectue = False
            st.rerun()

elif st.session_state.page == "details" and st.session_state.article_selectionne:
    # Page de détails (identique à votre code original)
    code_article = st.session_state.article_selectionne
    photos = gestionnaire.get_photos_article(code_article)
    total = gestionnaire.get_total_article(code_article)
    libelle = gestionnaire.get_libelle_article(code_article)
    emplacement = gestionnaire.get_emplacement_article(code_article)
    
    st.header(f"📦 {code_article}")
    if libelle:
        st.markdown(f"**{libelle}**")
    if emplacement:
        st.markdown(f"📍 {emplacement}")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total pièces", total)
    with col2:
        st.metric("Photos", len(photos))
    
    col_o1, col_o2, col_o3 = st.columns(3)
    with col_o1:
        if st.button("⬅️ Retour"):
            st.session_state.page = "saisie"
            st.rerun()
    with col_o2:
        if st.button("📸 Ajouter photo"):
            st.session_state.ajout_photo = True
    with col_o3:
        if st.button("🗑️ Supprimer", type="primary"):
            if gestionnaire.supprimer_article(code_article):
                st.session_state.page = "saisie"
                st.rerun()
    
    # Ajout de photo avec gestion du mode secours
    if st.session_state.get('ajout_photo', False):
        st.subheader("📸 Ajouter une photo")
        
        if st.session_state.mode_secours or not cameras_trouvees:
            # Mode secours pour les photos
            frame = capturer_avec_app_windows()
            if frame is not None:
                with st.spinner("Analyse..."):
                    resultat, nb_pieces = detecter_pieces(frame)
                    if gestionnaire.ajouter_photo_article(code_article, frame, resultat, nb_pieces):
                        st.success(f"✅ {nb_pieces} pièces ajoutées!")
                        st.session_state.ajout_photo = False
                        st.rerun()
        else:
            # Mode normal avec caméra
            if cameras_trouvees:
                options = [f"Caméra {cam['index']}" for cam in cameras_trouvees]
                camera_idx = st.selectbox("Caméra", range(len(cameras_trouvees)), format_func=lambda x: options[x])
                camera_index = cameras_trouvees[camera_idx]['index']
                
                if st.button(f"📷 Activer"):
                    st.session_state.photo_camera_active = True
                
                if st.session_state.get('photo_camera_active', False):
                    img_file = st.camera_input("Prendre photo", key=f"photo_{camera_index}")
                    if img_file:
                        bytes_data = img_file.getvalue()
                        frame = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
                        resultat, nb_pieces = detecter_pieces(frame)
                        
                        if gestionnaire.ajouter_photo_article(code_article, frame, resultat, nb_pieces):
                            st.success(f"✅ {nb_pieces} pièces ajoutées!")
                            st.session_state.ajout_photo = False
                            st.session_state.photo_camera_active = False
                            st.rerun()
            else:
                st.warning("Aucune caméra - utilisez l'upload")
                uploaded = st.file_uploader("Choisir image", type=['jpg', 'jpeg', 'png'])
                if uploaded:
                    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
                    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    resultat, nb_pieces = detecter_pieces(frame)
                    if gestionnaire.ajouter_photo_article(code_article, frame, resultat, nb_pieces):
                        st.success(f"✅ {nb_pieces} pièces ajoutées!")
                        st.session_state.ajout_photo = False
                        st.rerun()
        
        if st.button("❌ Annuler photo"):
            st.session_state.ajout_photo = False
            st.rerun()
    
    # Affichage des photos
    if photos:
        st.subheader("📸 Photos")
        cols = st.columns(3)
        for i, photo in enumerate(photos):
            with cols[i % 3]:
                img = base64_to_image(photo['image_analyse'])
                img_mini = cv2.resize(img, (200, 150))
                st.image(cv2.cvtColor(img_mini, cv2.COLOR_BGR2RGB), use_column_width=True)
                st.caption(f"{photo['nb_pieces']} pièces - {photo['timestamp'][:10]}")
                
                col_b1, col_b2 = st.columns(2)
                with col_b1:
                    if st.button("🔍", key=f"view_{i}"):
                        st.session_state.photo_selectionnee = i
                        st.session_state.page = "photo_detail"
                        st.rerun()
                with col_b2:
                    if st.button("🗑️", key=f"del_{i}"):
                        gestionnaire.supprimer_photo(code_article, i)
                        st.rerun()
    else:
        st.info("Aucune photo")

elif st.session_state.page == "photo_detail":
    # Détail photo (similaire à votre code)
    code_article = st.session_state.article_selectionne
    photos = gestionnaire.get_photos_article(code_article)
    photo_id = st.session_state.photo_selectionnee
    
    if 0 <= photo_id < len(photos):
        photo = photos[photo_id]
        
        st.header(f"Photo {photo_id + 1}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Originale")
            img_orig = base64_to_image(photo['image_originale'])
            st.image(cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB), use_column_width=True)
        
        with col2:
            st.subheader(f"Analyse - {photo['nb_pieces']} pièces")
            img_anal = base64_to_image(photo['image_analyse'])
            st.image(cv2.cvtColor(img_anal, cv2.COLOR_BGR2RGB), use_column_width=True)
        
        st.caption(f"Date: {photo['timestamp']}")
        
        if st.button("⬅️ Retour"):
            st.session_state.page = "details"
            st.session_state.photo_selectionnee = None
            st.rerun()
    else:
        st.error("Photo introuvable")

# Pied de page
st.markdown("---")
col_f1, col_f2, col_f3 = st.columns(3)
with col_f1:
    st.caption("📦 Gestionnaire d'Inventaire v4.0")
with col_f2:
    total_global = sum(gestionnaire.get_tous_les_totaux().values())
    st.caption(f"🧩 Total: {total_global} pièces")
with col_f3:
    st.caption(f"📊 Articles: {len(gestionnaire.articles)}")
