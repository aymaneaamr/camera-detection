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

# Dictionnaire des articles prédéfinis
ARTICLES_PREDEFINIS = {
    "10751037": "Capacitor E54.G85-203G30 Un 1260 V DC / 750 AC MKP 20µF",
    "10751038": "Contacteur principal Bipolaire",
    "10751039": "Contacteur de précharge Bipolaire",
    "10751040": "Coupe circuit 1A, 480VAC, 3Poles",
    "10751050": "Cosse à sertir 50x8"
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
</style>
""", unsafe_allow_html=True)

class GestionnairePieces:
    def __init__(self):
        self.articles = {}
        self.reset_article_courant()
    
    def reset_article_courant(self):
        self.article_courant = {
            'code': '',
            'libelle': '',
            'emplacement': '',
            'photos': [],
            'total_pieces': 0
        }
    
    def creer_nouvel_article(self, code_article, libelle="", emplacement=""):
        if code_article and code_article not in self.articles:
            if code_article in ARTICLES_PREDEFINIS and not libelle:
                libelle = ARTICLES_PREDEFINIS[code_article]
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
    
    def get_tous_les_totaux(self):
        return {code: self.get_total_article(code) for code in self.articles}
    
    def get_tous_emplacements(self):
        return {code: self.get_emplacement_article(code) for code in self.articles}
    
    def get_tous_libelles(self):
        return {code: self.get_libelle_article(code) for code in self.articles}
    
    def generer_excel(self):
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
            sheet_resume.cell(row=row, column=1).value = code_article
            sheet_resume.cell(row=row, column=2).value = data.get('libelle', '')
            sheet_resume.cell(row=row, column=3).value = data.get('emplacement', '')
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
        cv2.putText(resultat, f"{type_code}: {data}", 
                   (code.rect.left, code.rect.top - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        codes_detectes.append({'data': data, 'type': type_code})
    return resultat, codes_detectes

def detecter_pieces(image):
    resultat = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)
    edges = cv2.erode(edges, kernel, iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pieces_valides = [c for c in contours if cv2.contourArea(c) > 200]
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

def base64_to_image(base64_string):
    img_data = base64.b64decode(base64_string)
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

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

gestionnaire = st.session_state.gestionnaire

# Interface principale
st.title("📦 Gestionnaire d'Inventaire Multi-Pièces avec Scan Code-Barres")
st.markdown("""
Cette application permet de gérer l'inventaire de plusieurs types de pièces :
1. **Scanner** un code-barres pour identifier automatiquement l'article
2. **Ajouter** un libellé descriptif (optionnel)
3. **Ajouter** un emplacement de stockage (optionnel)
4. **Ajouter** plusieurs photos pour cet article
5. **Changer** d'article et répéter
6. **Exporter** un fichier Excel avec tous les totaux
""")

# Barre latérale
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

# Contenu principal
if st.session_state.page == "saisie":
    st.header("➕ Ajouter un nouvel article")
    
    # Section scan
    st.markdown('<div class="barcode-scanner">', unsafe_allow_html=True)
    st.markdown("### 📷 Scanner le code-barres de l'article")
    st.markdown("Prenez une photo du code-barres pour identifier automatiquement l'article")
    col_scan1, col_scan2 = st.columns(2)
    with col_scan1:
        scan_option = st.radio("Source", ["📸 Caméra", "🖼️ Upload"], horizontal=True, key="scan_source")
    if scan_option == "📸 Caméra":
        img_barcode = st.camera_input("Prendre une photo du code-barres", key="camera_barcode")
        if img_barcode:
            with st.spinner("🔍 Analyse du code-barres..."):
                bytes_data = img_barcode.getvalue()
                frame = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
                image_annotee, codes = detecter_code_barre(frame)
                if codes:
                    code_trouve = codes[0]['data']
                    st.session_state.code_detecte = code_trouve
                    st.session_state.scan_effectue = True
                    st.image(cv2.cvtColor(image_annotee, cv2.COLOR_BGR2RGB), caption="Code-barres détecté", use_container_width=True)
                    st.markdown(f"""
                    <div class="success-box">
                        <h4>✅ Code-barres détecté !</h4>
                        <div class="code-display">{code_trouve}</div>
                        <p><strong>Type :</strong> {codes[0]['type']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning("❌ Aucun code-barres détecté. Veuillez réessayer avec une image plus claire.")
    else:
        uploaded_barcode = st.file_uploader("Choisir une image de code-barres", type=['jpg', 'jpeg', 'png'], key="upload_barcode")
        if uploaded_barcode:
            with st.spinner("🔍 Analyse du code-barres..."):
                file_bytes = np.asarray(bytearray(uploaded_barcode.read()), dtype=np.uint8)
                frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                image_annotee, codes = detecter_code_barre(frame)
                st.image(cv2.cvtColor(image_annotee, cv2.COLOR_BGR2RGB), caption="Image analysée", use_container_width=True)
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
                    st.warning("❌ Aucun code-barres détecté. Veuillez réessayer avec une image plus claire.")
    st.markdown('</div>', unsafe_allow_html=True)
    
    if st.session_state.scan_effectue:
        if st.button("🔄 Nouveau scan", use_container_width=True):
            st.session_state.scan_effectue = False
            st.session_state.code_detecte = None
            st.rerun()
    
    st.markdown("---")
    
    # ========== SOLUTION SIMPLE : SI CODE ARTICLE ALORS LIBELLÉ ==========
    st.markdown("### 📝 Informations de l'article")
    
    col_code, col_lib, col_emp = st.columns([2, 2, 1])
    
    with col_code:
        # Valeur par défaut depuis le scan
        default_code = st.session_state.code_detecte if st.session_state.code_detecte else ""
        code_article = st.text_input(
            "Code article *",
            value=default_code,
            placeholder="Code article (obligatoire)",
            key="code_article"
        )
        
        # Message si article trouvé
        if code_article and code_article in ARTICLES_PREDEFINIS:
            st.markdown(f"""
            <div class="article-found">
                <strong>📝 Article trouvé :</strong> {ARTICLES_PREDEFINIS[code_article]}
            </div>
            """, unsafe_allow_html=True)
    
    with col_lib:
        # SI LE CODE EXISTE, ON AFFICHE LE LIBELLÉ DIRECTEMENT
        if code_article and code_article in ARTICLES_PREDEFINIS:
            libelle_value = ARTICLES_PREDEFINIS[code_article]
        else:
            libelle_value = ""
        
        libelle = st.text_input(
            "Libellé (optionnel)",
            value=libelle_value,
            placeholder="Description de l'article",
            key="libelle"
        )
    
    with col_emp:
        emplacement = st.text_input(
            "Emplacement (optionnel)",
            placeholder="Ex: A-12, Rayon 3...",
            key="emplacement"
        )
    
    st.caption("* Champ obligatoire")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Créer l'article", use_container_width=True):
            if code_article:
                if gestionnaire.creer_nouvel_article(code_article, libelle, emplacement):
                    st.success(f"✅ Article '{code_article}' créé avec succès!")
                    st.session_state.article_selectionne = code_article
                    st.session_state.page = "details"
                    st.session_state.code_detecte = None
                    st.session_state.scan_effectue = False
                    st.rerun()
                else:
                    if code_article in gestionnaire.articles:
                        st.error("❌ Ce code article existe déjà")
                    else:
                        st.error("❌ Erreur lors de la création de l'article")
            else:
                st.error("❌ Veuillez entrer un code article")
    with col2:
        if st.button("❌ Annuler", use_container_width=True):
            st.session_state.code_detecte = None
            st.session_state.scan_effectue = False
            st.rerun()

elif st.session_state.page == "details" and st.session_state.article_selectionne:
    # Page de détails d'un article
    code_article = st.session_state.article_selectionne
    photos = gestionnaire.get_photos_article(code_article)
    total = gestionnaire.get_total_article(code_article)
    libelle = gestionnaire.get_libelle_article(code_article)
    emplacement = gestionnaire.get_emplacement_article(code_article)
    
    # En-tête avec libellé et emplacement
    col_h1, col_h2, col_h3, col_h4, col_h5 = st.columns([2, 1, 1, 1, 1])
    with col_h1:
        st.header(f"📦 {code_article}")
        if libelle:
            st.markdown(f"<span class='label-badge'>📝 {libelle}</span>", unsafe_allow_html=True)
        if emplacement:
            st.markdown(f"<span class='location-badge'>📍 {emplacement}</span>", unsafe_allow_html=True)
    with col_h2:
        st.metric("Total pièces", total)
    with col_h3:
        st.metric("Photos", len(photos))
    with col_h4:
        if libelle:
            st.metric("Libellé", libelle[:20] + "..." if len(libelle) > 20 else libelle)
    with col_h5:
        if emplacement:
            st.metric("Emplacement", emplacement)
    
    # Afficher un badge si le code est un code-barres
    if re.match(r'^[A-Z0-9-]+$', code_article):
        st.info(f"🔖 Code produit: {code_article}")
    
    # Options
    col_o1, col_o2, col_o3 = st.columns(3)
    with col_o1:
        if st.button("⬅️ Retour à la saisie", use_container_width=True):
            st.session_state.page = "saisie"
            st.session_state.code_detecte = None
            st.session_state.scan_effectue = False
            st.rerun()
    with col_o2:
        if st.button("📸 Ajouter une photo", use_container_width=True):
            st.session_state.ajout_photo = True
    with col_o3:
        if st.button("🗑️ Supprimer cet article", use_container_width=True, type="primary"):
            if gestionnaire.supprimer_article(code_article):
                st.success(f"✅ Article '{code_article}' supprimé")
                st.session_state.page = "saisie"
                st.rerun()
    
    st.divider()
    
    # Ajout de photo
    if st.session_state.get('ajout_photo', False):
        st.subheader("📸 Ajouter une photo")
        
        col_p1, col_p2 = st.columns([2, 1])
        with col_p2:
            if st.button("❌ Annuler"):
                st.session_state.ajout_photo = False
                st.rerun()
        
        with col_p1:
            source = st.radio("Source", ["📸 Prendre une photo", "🖼️ Choisir une image"], horizontal=True)
        
        if source == "📸 Prendre une photo":
            img_file = st.camera_input("Prendre une photo")
            if img_file:
                with st.spinner("Analyse..."):
                    bytes_data = img_file.getvalue()
                    frame = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
                    resultat, nb_pieces = detecter_pieces(frame)
                    
                    if gestionnaire.ajouter_photo_article(code_article, frame, resultat, nb_pieces):
                        st.success(f"✅ {nb_pieces} pièces détectées et ajoutées!")
                        st.session_state.ajout_photo = False
                        st.rerun()
        
        else:  # Choisir une image
            uploaded_file = st.file_uploader("Choisir une image", type=['jpg', 'jpeg', 'png'])
            if uploaded_file:
                with st.spinner("Analyse..."):
                    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    resultat, nb_pieces = detecter_pieces(frame)
                    
                    if gestionnaire.ajouter_photo_article(code_article, frame, resultat, nb_pieces):
                        st.success(f"✅ {nb_pieces} pièces détectées et ajoutées!")
                        st.session_state.ajout_photo = False
                        st.rerun()
    
    # Affichage des photos existantes
    if photos:
        st.subheader("📸 Photos enregistrées")
        
        # Options d'affichage
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            tri = st.selectbox("Trier par", ["Plus récente", "Plus ancienne", "Plus de pièces", "Moins de pièces"])
        
        # Trier les photos
        photos_affichees = photos.copy()
        if tri == "Plus récente":
            photos_affichees = list(reversed(photos_affichees))
        elif tri == "Plus ancienne":
            photos_affichees = photos_affichees
        elif tri == "Plus de pièces":
            photos_affichees = sorted(photos_affichees, key=lambda x: x['nb_pieces'], reverse=True)
        elif tri == "Moins de pièces":
            photos_affichees = sorted(photos_affichees, key=lambda x: x['nb_pieces'])
        
        # Afficher les photos en grille
        cols = st.columns(3)
        for i, photo in enumerate(photos_affichees):
            with cols[i % 3]:
                # Afficher la miniature
                img = base64_to_image(photo['image_analyse'])
                img_mini = cv2.resize(img, (200, 150))
                st.image(cv2.cvtColor(img_mini, cv2.COLOR_BGR2RGB), use_column_width=True)
                
                # Informations
                st.caption(f"📅 {photo['timestamp'][:10]}")
                st.caption(f"🔢 {photo['nb_pieces']} pièces")
                
                # Boutons
                col_b1, col_b2 = st.columns(2)
                with col_b1:
                    if st.button("🔍 Voir", key=f"view_{code_article}_{i}"):
                        st.session_state.photo_selectionnee = photo['id']
                        st.session_state.page = "photo_detail"
                        st.rerun()
                with col_b2:
                    if st.button("🗑️", key=f"del_{code_article}_{i}"):
                        if gestionnaire.supprimer_photo(code_article, photo['id']):
                            st.rerun()
    
    else:
        st.info("📸 Aucune photo pour cet article. Cliquez sur 'Ajouter une photo' pour commencer.")

elif st.session_state.page == "photo_detail" and st.session_state.article_selectionne and st.session_state.photo_selectionnee is not None:
    # Détail d'une photo spécifique
    code_article = st.session_state.article_selectionne
    photos = gestionnaire.get_photos_article(code_article)
    photo_id = st.session_state.photo_selectionnee
    
    if 0 <= photo_id < len(photos):
        photo = photos[photo_id]
        libelle = gestionnaire.get_libelle_article(code_article)
        
        st.header(f"🔍 Détail de la photo - {code_article}")
        if libelle:
            st.subheader(libelle)
        
        # Afficher les deux images
        col_img1, col_img2 = st.columns(2)
        
        with col_img1:
            st.subheader("📸 Image originale")
            img_originale = base64_to_image(photo['image_originale'])
            st.image(cv2.cvtColor(img_originale, cv2.COLOR_BGR2RGB), use_column_width=True)
        
        with col_img2:
            st.subheader(f"🔍 Analyse - {photo['nb_pieces']} pièces")
            img_analyse = base64_to_image(photo['image_analyse'])
            st.image(cv2.cvtColor(img_analyse, cv2.COLOR_BGR2RGB), use_column_width=True)
        
        # Informations
        st.metric("Nombre de pièces", photo['nb_pieces'])
        st.caption(f"Date: {photo['timestamp']}")
        
        # Boutons
        col_b1, col_b2 = st.columns(2)
        with col_b1:
            if st.button("⬅️ Retour à l'article", use_container_width=True):
                st.session_state.page = "details"
                st.session_state.photo_selectionnee = None
                st.rerun()
        with col_b2:
            if st.button("🗑️ Supprimer cette photo", use_container_width=True, type="primary"):
                if gestionnaire.supprimer_photo(code_article, photo_id):
                    st.session_state.page = "details"
                    st.session_state.photo_selectionnee = None
                    st.rerun()
    else:
        st.error("Photo non trouvée")
        if st.button("Retour"):
            st.session_state.page = "details"
            st.session_state.photo_selectionnee = None
            st.rerun()

# Pied de page
st.markdown("---")
col_f1, col_f2, col_f3, col_f4, col_f5 = st.columns(5)
with col_f1:
    st.caption("📦 Gestionnaire d'Inventaire v3.0 - Avec scan code-barres")
with col_f2:
    total_global = sum(gestionnaire.get_tous_les_totaux().values())
    st.caption(f"🧩 Total global: {total_global} pièces")
with col_f3:
    st.caption(f"📊 Articles: {len(gestionnaire.articles)}")
with col_f4:
    emplacements_renseignes = sum(1 for e in gestionnaire.get_tous_emplacements().values() if e)
    st.caption(f"📍 Emplacements: {emplacements_renseignes}/{len(gestionnaire.articles)}")
with col_f5:
    libelles_renseignes = sum(1 for l in gestionnaire.get_tous_libelles().values() if l)
    st.caption(f"📝 Libellés: {libelles_renseignes}/{len(gestionnaire.articles)}")
