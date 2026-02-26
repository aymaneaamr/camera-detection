import streamlit as st
import cv2
import numpy as np
from collections import defaultdict
from datetime import datetime
import base64
from io import BytesIO
import openpyxl
from openpyxl.styles import Font, Alignment, PatternFill
from pyzbar.pyzbar import decode
import re
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
    .webcam-container {
        border: 3px solid #667eea;
        border-radius: 10px;
        padding: 20px;
        margin: 20px 0;
        background: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# ==================== FONCTIONS DE DÉTECTION ====================
class DetecteurPieces:
    def __init__(self):
        self.couleurs = {
            'rouge': {
                'lower1': np.array([0, 100, 100]), 'upper1': np.array([10, 255, 255]),
                'lower2': np.array([160, 100, 100]), 'upper2': np.array([180, 255, 255]),
            },
            'bleu': {
                'lower': np.array([100, 150, 50]), 'upper': np.array([140, 255, 255]),
            },
            'vert': {
                'lower': np.array([40, 70, 70]), 'upper': np.array([80, 255, 255]),
            },
            'jaune': {
                'lower': np.array([20, 100, 100]), 'upper': np.array([30, 255, 255]),
            }
        }
        
        self.seuils_taille = {
            'P': (0, 500),
            'M': (500, 2000),
            'G': (2000, 5000),
            'TG': (5000, float('inf'))
        }
    
    def get_couleur_piece(self, hsv, contour):
        """Détermine la couleur d'une pièce"""
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        
        best_couleur = '?'
        best_score = 0
        
        for nom_couleur, params in self.couleurs.items():
            if 'lower1' in params:
                mask1 = cv2.inRange(hsv, params['lower1'], params['upper1'])
                mask2 = cv2.inRange(hsv, params['lower2'], params['upper2'])
                mask_couleur = cv2.bitwise_or(mask1, mask2)
            else:
                mask_couleur = cv2.inRange(hsv, params['lower'], params['upper'])
            
            mask_combine = cv2.bitwise_and(mask_couleur, mask)
            pixels_couleur = cv2.countNonZero(mask_combine)
            pixels_total = cv2.countNonZero(mask)
            
            if pixels_total > 0:
                score = pixels_couleur / pixels_total
                if score > best_score and score > 0.2:
                    best_score = score
                    best_couleur = nom_couleur
        
        return best_couleur
    
    def get_taille_piece(self, aire):
        """Détermine la taille d'une pièce"""
        for nom_taille, (min_vol, max_vol) in self.seuils_taille.items():
            if min_vol <= aire < max_vol:
                return nom_taille
        return '?'
    
    def detecter(self, image):
        """Détecte et compte les pièces dans une image"""
        resultat = image.copy()
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Détection des contours
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=2)
        edges = cv2.erode(edges, kernel, iterations=1)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        pieces_valides = []
        stats_couleur = defaultdict(int)
        stats_taille = defaultdict(int)
        
        couleurs_bbox = {
            'rouge': (0, 0, 255),
            'bleu': (255, 0, 0),
            'vert': (0, 255, 0),
            'jaune': (0, 255, 255),
            '?': (128, 128, 128)
        }
        
        for contour in contours:
            aire = cv2.contourArea(contour)
            if aire < 200:
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            
            couleur_nom = self.get_couleur_piece(hsv, contour)
            taille_nom = self.get_taille_piece(aire)
            
            pieces_valides.append(contour)
            stats_couleur[couleur_nom] += 1
            stats_taille[taille_nom] += 1
            
            # Dessiner la pièce
            cv2.rectangle(resultat, (x, y), (x+w, y+h), couleurs_bbox.get(couleur_nom, (128,128,128)), 2)
            cv2.circle(resultat, (x + w//2, y + h//2), 3, (255, 255, 255), -1)
        
        nb_pieces = len(pieces_valides)
        
        # Ajouter les informations sur l'image
        h, w = resultat.shape[:2]
        
        # TOTAL
        cv2.putText(resultat, f"Pieces: {nb_pieces}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Statistiques couleurs
        y_stats = 60
        cv2.putText(resultat, f"Couleurs: R:{stats_couleur.get('rouge',0)} B:{stats_couleur.get('bleu',0)} V:{stats_couleur.get('vert',0)} J:{stats_couleur.get('jaune',0)}", 
                   (10, y_stats), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Statistiques tailles
        y_stats += 20
        cv2.putText(resultat, f"Tailles: P:{stats_taille.get('P',0)} M:{stats_taille.get('M',0)} G:{stats_taille.get('G',0)} TG:{stats_taille.get('TG',0)}", 
                   (10, y_stats), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return resultat, nb_pieces, stats_couleur, stats_taille

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
        
        codes_detectes.append({'data': data, 'type': type_code})
    
    return resultat, codes_detectes

def base64_to_image(base64_string):
    img_data = base64.b64decode(base64_string)
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

# ==================== CLASSE GESTIONNAIRE ====================
class GestionnairePieces:
    def __init__(self):
        self.articles = {}
        self.detecteur = DetecteurPieces()
    
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
    
    def ajouter_photo(self, code_article, frame):
        """Ajoute une photo avec détection automatique"""
        if code_article in self.articles:
            # Détecter les pièces
            resultat, nb_pieces, stats_couleur, stats_taille = self.detecteur.detecter(frame)
            
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            _, buffer_original = cv2.imencode('.jpg', frame)
            _, buffer_analyse = cv2.imencode('.jpg', resultat)
            
            photo_data = {
                'timestamp': timestamp,
                'nb_pieces': nb_pieces,
                'stats_couleur': dict(stats_couleur),
                'stats_taille': dict(stats_taille),
                'image_originale': base64.b64encode(buffer_original).decode('utf-8'),
                'image_analyse': base64.b64encode(buffer_analyse).decode('utf-8'),
                'id': len(self.articles[code_article]['photos'])
            }
            
            self.articles[code_article]['photos'].append(photo_data)
            return True, nb_pieces, resultat
        return False, 0, None
    
    def get_total_article(self, code_article):
        if code_article in self.articles:
            return sum(photo['nb_pieces'] for photo in self.articles[code_article]['photos'])
        return 0
    
    def get_photos_article(self, code_article):
        if code_article in self.articles:
            return self.articles[code_article]['photos']
        return []
    
    def get_emplacement_article(self, code_article):
        if code_article in self.articles:
            return self.articles[code_article].get('emplacement', '')
        return ''
    
    def get_libelle_article(self, code_article):
        if code_article in self.articles:
            return self.articles[code_article].get('libelle', '')
        return ''
    
    def supprimer_photo(self, code_article, photo_id):
        if code_article in self.articles and 0 <= photo_id < len(self.articles[code_article]['photos']):
            del self.articles[code_article]['photos'][photo_id]
            for i, photo in enumerate(self.articles[code_article]['photos']):
                photo['id'] = i
            return True
        return False
    
    def supprimer_article(self, code_article):
        if code_article in self.articles:
            del self.articles[code_article]
            return True
        return False
    
    def generer_excel(self):
        output = BytesIO()
        workbook = openpyxl.Workbook()
        
        # Feuille résumé
        sheet = workbook.active
        sheet.title = "Inventaire"
        
        headers = ["Code Article", "Libellé", "Emplacement", "Total pièces", "Photos", "Dernière mise à jour"]
        for col, header in enumerate(headers, 1):
            cell = sheet.cell(row=1, column=col)
            cell.value = header
            cell.font = Font(bold=True)
            cell.fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
            cell.font = Font(color="FFFFFF", bold=True)
        
        row = 2
        for code, data in self.articles.items():
            total = sum(p['nb_pieces'] for p in data['photos'])
            sheet.cell(row=row, column=1).value = code
            sheet.cell(row=row, column=2).value = data.get('libelle', '')
            sheet.cell(row=row, column=3).value = data.get('emplacement', '')
            sheet.cell(row=row, column=4).value = total
            sheet.cell(row=row, column=5).value = len(data['photos'])
            sheet.cell(row=row, column=6).value = data['photos'][-1]['timestamp'] if data['photos'] else data['date_creation']
            row += 1
        
        workbook.save(output)
        output.seek(0)
        return output

# ==================== INITIALISATION ====================
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

gestionnaire = st.session_state.gestionnaire

# ==================== INTERFACE ====================
st.title("📦 Gestionnaire d'Inventaire avec Caméra USB")
st.markdown("Caméra du téléphone (index 2) - Compatible USB")

# Barre latérale
with st.sidebar:
    st.header("📋 Articles")
    if gestionnaire.articles:
        for code in gestionnaire.articles.keys():
            total = gestionnaire.get_total_article(code)
            libelle = gestionnaire.get_libelle_article(code)
            if st.button(f"📦 {code} - {total} pcs", key=f"btn_{code}", use_container_width=True):
                st.session_state.article_selectionne = code
                st.session_state.page = "details"
                st.rerun()
        
        st.divider()
        if st.button("➕ Nouvel article", use_container_width=True):
            st.session_state.page = "saisie"
            st.session_state.article_selectionne = None
            st.rerun()
        
        if gestionnaire.articles:
            excel_file = gestionnaire.generer_excel()
            st.download_button(
                label="📥 Excel",
                data=excel_file,
                file_name=f"inventaire.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
    else:
        st.info("Aucun article")

# ==================== PAGE SAISIE ====================
if st.session_state.page == "saisie":
    st.header("➕ Nouvel article")
    
    # === SECTION WEBCAM SIMPLE AVEC st.camera_input ===
    st.markdown('<div class="webcam-container">', unsafe_allow_html=True)
    st.subheader("📷 Caméra USB (index 2)")
    
    # st.camera_input fonctionne parfaitement avec les caméras externes !
    img_file = st.camera_input("Prendre une photo avec la caméra USB", key="webcam_usb")
    
    if img_file:
        # Lire l'image
        bytes_data = img_file.getvalue()
        frame = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        
        # Détecter les pièces
        resultat, nb_pieces, stats_couleur, stats_taille = gestionnaire.detecteur.detecter(frame)
        
        # Afficher les résultats
        col1, col2 = st.columns(2)
        with col1:
            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Image originale")
        with col2:
            st.image(cv2.cvtColor(resultat, cv2.COLOR_BGR2RGB), caption=f"Analyse - {nb_pieces} pièces")
        
        # Statistiques
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            st.metric("Total pièces", nb_pieces)
        with col_s2:
            st.metric("Rouge/Bleu", f"{stats_couleur.get('rouge',0)}/{stats_couleur.get('bleu',0)}")
        with col_s3:
            st.metric("Vert/Jaune", f"{stats_couleur.get('vert',0)}/{stats_couleur.get('jaune',0)}")
        
        # Sauvegarder dans session_state
        if 'derniere_photo' not in st.session_state:
            st.session_state.derniere_photo = {
                'frame': frame,
                'resultat': resultat,
                'nb_pieces': nb_pieces
            }
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # === SCAN CODE-BARRES ===
    st.markdown("---")
    st.markdown('<div class="barcode-scanner">', unsafe_allow_html=True)
    st.subheader("📷 Scanner code-barres")
    
    img_barcode = st.camera_input("Prendre photo du code-barres", key="barcode_cam")
    if img_barcode:
        bytes_data = img_barcode.getvalue()
        frame = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        image_annotee, codes = detecter_code_barre(frame)
        
        st.image(cv2.cvtColor(image_annotee, cv2.COLOR_BGR2RGB), use_column_width=True)
        
        if codes:
            code_trouve = codes[0]['data']
            st.session_state.code_detecte = code_trouve
            st.markdown(f"""
            <div class="success-box">
                <h4>✅ Code détecté !</h4>
                <div class="code-display">{code_trouve}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("❌ Aucun code détecté")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # === FORMULAIRE ===
    st.markdown("---")
    st.subheader("📝 Informations article")
    
    default_code = st.session_state.code_detecte if st.session_state.code_detecte else ""
    
    col1, col2 = st.columns(2)
    with col1:
        code = st.text_input("Code article *", value=default_code)
    with col2:
        emplacement = st.text_input("Emplacement (optionnel)")
    
    libelle = st.text_input("Libellé (optionnel)")
    
    if code and code in ARTICLES_PREDEFINIS:
        st.info(f"📦 {ARTICLES_PREDEFINIS[code]['libelle']}")
    
    col_b1, col_b2 = st.columns(2)
    with col_b1:
        if st.button("✅ Créer article", use_container_width=True):
            if code:
                if gestionnaire.creer_nouvel_article(code, libelle, emplacement):
                    # Ajouter la photo si elle existe
                    if 'derniere_photo' in st.session_state:
                        photo = st.session_state.derniere_photo
                        gestionnaire.ajouter_photo(code, photo['frame'])
                        del st.session_state.derniere_photo
                    
                    st.success(f"✅ Article {code} créé")
                    st.session_state.article_selectionne = code
                    st.session_state.page = "details"
                    st.rerun()
                else:
                    st.error("Code existe déjà")
            else:
                st.error("Code requis")

# ==================== PAGE DÉTAILS ====================
elif st.session_state.page == "details" and st.session_state.article_selectionne:
    code = st.session_state.article_selectionne
    photos = gestionnaire.get_photos_article(code)
    total = gestionnaire.get_total_article(code)
    libelle = gestionnaire.get_libelle_article(code)
    emplacement = gestionnaire.get_emplacement_article(code)
    
    # En-tête
    st.header(f"📦 {code}")
    if libelle:
        st.subheader(libelle)
    if emplacement:
        st.info(f"📍 {emplacement}")
    
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.metric("Total pièces", total)
    with col_m2:
        st.metric("Photos", len(photos))
    with col_m3:
        st.metric("Moyenne/photo", round(total/len(photos) if photos else 0, 1))
    
    # Boutons
    col_b1, col_b2, col_b3 = st.columns(3)
    with col_b1:
        if st.button("⬅️ Retour"):
            st.session_state.page = "saisie"
            st.rerun()
    with col_b2:
        if st.button("📸 Ajouter photo"):
            st.session_state.ajout_photo = True
    with col_b3:
        if st.button("🗑️ Supprimer article", type="primary"):
            if gestionnaire.supprimer_article(code):
                st.session_state.page = "saisie"
                st.rerun()
    
    # Ajout de photo
    if st.session_state.get('ajout_photo', False):
        st.divider()
        st.subheader("📸 Nouvelle photo")
        
        img_file = st.camera_input("Prendre photo")
        if img_file:
            bytes_data = img_file.getvalue()
            frame = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            success, nb, resultat = gestionnaire.ajouter_photo(code, frame)
            
            if success:
                st.success(f"✅ {nb} pièces détectées!")
                st.image(cv2.cvtColor(resultat, cv2.COLOR_BGR2RGB))
                st.session_state.ajout_photo = False
                time.sleep(1)
                st.rerun()
    
    # Liste des photos
    if photos:
        st.divider()
        st.subheader("📸 Photos")
        
        cols = st.columns(3)
        for i, photo in enumerate(reversed(photos)):  # Plus récentes d'abord
            with cols[i % 3]:
                img = base64_to_image(photo['image_analyse'])
                img_mini = cv2.resize(img, (200, 150))
                st.image(cv2.cvtColor(img_mini, cv2.COLOR_BGR2RGB))
                st.caption(f"{photo['nb_pieces']} pièces - {photo['timestamp'][:10]}")
                
                if st.button(f"🗑️", key=f"del_{i}"):
                    if gestionnaire.supprimer_photo(code, len(photos)-1-i):
                        st.rerun()

# Pied de page
st.markdown("---")
st.caption("📦 Version USB - Compatible caméra téléphone (index 2)")
