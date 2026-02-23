import streamlit as st
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import json
import base64
from io import BytesIO
import openpyxl
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
import re

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
    .info-box {
        background: #e8f4fd;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #2196F3;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class GestionnairePieces:
    def __init__(self):
        """Initialise le gestionnaire de pièces"""
        self.pieces = {}
        self.reset_piece_courante()
    
    def reset_piece_courante(self):
        """Réinitialise la pièce en cours de saisie"""
        self.piece_courante = {
            'nom': '',
            'photos': [],
            'total_pieces': 0
        }
    
    def creer_nouvelle_piece(self, nom_piece):
        """Crée une nouvelle pièce dans l'inventaire"""
        if nom_piece and nom_piece not in self.pieces:
            self.pieces[nom_piece] = []
            return True
        return False
    
    def ajouter_photo_piece(self, nom_piece, frame_original, frame_analyse, nb_pieces):
        """Ajoute une photo analysée à une pièce existante"""
        if nom_piece in self.pieces:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            _, buffer_original = cv2.imencode('.jpg', frame_original)
            _, buffer_analyse = cv2.imencode('.jpg', frame_analyse)
            
            photo_data = {
                'timestamp': timestamp,
                'nb_pieces': nb_pieces,
                'image_originale': base64.b64encode(buffer_original).decode('utf-8'),
                'image_analyse': base64.b64encode(buffer_analyse).decode('utf-8'),
                'id': len(self.pieces[nom_piece])
            }
            
            self.pieces[nom_piece].append(photo_data)
            return True
        return False
    
    def get_total_piece(self, nom_piece):
        """Retourne le total de pièces pour un nom donné"""
        if nom_piece in self.pieces:
            return sum(photo['nb_pieces'] for photo in self.pieces[nom_piece])
        return 0
    
    def get_photos_piece(self, nom_piece):
        """Retourne toutes les photos d'une pièce"""
        return self.pieces.get(nom_piece, [])
    
    def supprimer_photo(self, nom_piece, photo_id):
        """Supprime une photo d'une pièce"""
        if nom_piece in self.pieces and 0 <= photo_id < len(self.pieces[nom_piece]):
            del self.pieces[nom_piece][photo_id]
            for i, photo in enumerate(self.pieces[nom_piece]):
                photo['id'] = i
            return True
        return False
    
    def supprimer_piece(self, nom_piece):
        """Supprime complètement une pièce"""
        if nom_piece in self.pieces:
            del self.pieces[nom_piece]
            return True
        return False
    
    def get_tous_les_totaux(self):
        """Retourne un dictionnaire avec tous les totaux par pièce"""
        return {nom: self.get_total_piece(nom) for nom in self.pieces}
    
    def generer_excel(self):
        """Génère un fichier Excel avec l'inventaire complet"""
        output = BytesIO()
        workbook = openpyxl.Workbook()
        
        # Style
        header_font = Font(bold=True, color="FFFFFF")
        header_fill_bleu = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
        header_fill_vert = PatternFill(start_color="92D050", end_color="92D050", fill_type="solid")
        border = Border(
            left=Side(style='thin'),
            right=Side(style='thin'),
            top=Side(style='thin'),
            bottom=Side(style='thin')
        )
        
        # Feuille principale
        sheet_resume = workbook.active
        sheet_resume.title = "Inventaire"
        
        headers = ["Nom de la pièce", "Quantité totale", "Nombre de photos", "Dernière mise à jour"]
        for col, header in enumerate(headers, 1):
            cell = sheet_resume.cell(row=1, column=col)
            cell.value = header
            cell.font = header_font
            cell.fill = header_fill_bleu
            cell.alignment = Alignment(horizontal="center")
            cell.border = border
        
        row = 2
        for nom_piece, photos in self.pieces.items():
            total = sum(p['nb_pieces'] for p in photos)
            nb_photos = len(photos)
            derniere_date = photos[-1]['timestamp'] if photos else "N/A"
            
            sheet_resume.cell(row=row, column=1).value = nom_piece
            sheet_resume.cell(row=row, column=2).value = total
            sheet_resume.cell(row=row, column=3).value = nb_photos
            sheet_resume.cell(row=row, column=4).value = derniere_date
            
            for col in range(1, 5):
                cell = sheet_resume.cell(row=row, column=col)
                cell.border = border
            row += 1
        
        for col in range(1, 5):
            sheet_resume.column_dimensions[openpyxl.utils.get_column_letter(col)].width = 22
        
        # Feuille de détail
        sheet_detail = workbook.create_sheet("Détail des photos")
        
        detail_headers = ["Pièce", "Photo #", "Date", "Nombre de pièces"]
        for col, header in enumerate(detail_headers, 1):
            cell = sheet_detail.cell(row=1, column=col)
            cell.value = header
            cell.font = header_font
            cell.fill = header_fill_vert
            cell.alignment = Alignment(horizontal="center")
            cell.border = border
        
        row = 2
        for nom_piece, photos in self.pieces.items():
            for i, photo in enumerate(photos, 1):
                sheet_detail.cell(row=row, column=1).value = nom_piece
                sheet_detail.cell(row=row, column=2).value = f"Photo {i}"
                sheet_detail.cell(row=row, column=3).value = photo['timestamp']
                sheet_detail.cell(row=row, column=4).value = photo['nb_pieces']
                
                for col in range(1, 5):
                    cell = sheet_detail.cell(row=row, column=col)
                    cell.border = border
                row += 1
        
        sheet_detail.column_dimensions['A'].width = 25
        sheet_detail.column_dimensions['B'].width = 12
        sheet_detail.column_dimensions['C'].width = 22
        sheet_detail.column_dimensions['D'].width = 18
        
        workbook.save(output)
        output.seek(0)
        return output
    
    def reinitialiser_tout(self):
        """Réinitialise complètement l'inventaire"""
        self.pieces = {}

# Version simplifiée de lecture de code-barres (sans pyzbar)
def lire_code_barre_simple(image):
    """
    Version simplifiée qui simule la lecture de code-barres
    ou utilise des motifs pour détecter des formes de codes
    """
    resultat = image.copy()
    codes = []
    
    # Détection de contours pour trouver des zones rectangulaires
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # Trouver les contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:  # Zone significative
            # Obtenir le rectangle englobant
            x, y, w, h = cv2.boundingRect(contour)
            
            # Vérifier si c'est potentiellement un code-barres (ratio)
            aspect_ratio = w / h if h > 0 else 0
            if 2 < aspect_ratio < 5 or (w < h and 0.2 < aspect_ratio < 0.5):
                # Dessiner un rectangle
                cv2.rectangle(resultat, (x, y), (x + w, y + h), (0, 255, 0), 3)
                
                # Simuler un code trouvé
                code_simule = f"PROD-{np.random.randint(1000, 9999)}"
                codes.append({
                    'data': code_simule,
                    'type': 'CODE128 (simulé)',
                    'rect': (x, y, w, h)
                })
                
                cv2.putText(resultat, code_simule, (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return resultat, codes

# Version avec saisie manuelle de code (recommandée)
def saisie_code_manuel():
    """Interface pour la saisie manuelle de code"""
    with st.form("saisie_code"):
        st.markdown("### ✏️ Saisie manuelle du code")
        code = st.text_input("Entrez le code de la pièce", placeholder="Ex: PROD-12345, VIS-M8, etc.")
        col1, col2 = st.columns(2)
        with col1:
            valider = st.form_submit_button("✅ Valider", use_container_width=True)
        with col2:
            annuler = st.form_submit_button("❌ Annuler", use_container_width=True)
        
        if valider and code:
            return code
    return None

def formater_nom_piece(code):
    """Formate le code en nom de pièce lisible"""
    # Supprimer les caractères spéciaux
    nom = re.sub(r'[^\w\s-]', '', code)
    return nom

def detecter_pieces(image):
    """Détecte et compte les pièces dans une image"""
    resultat = image.copy()
    
    # Conversion en niveaux de gris
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Flou pour réduire le bruit
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Détection des contours
    edges = cv2.Canny(blur, 50, 150)
    
    # Dilatation et érosion
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)
    edges = cv2.erode(edges, kernel, iterations=1)
    
    # Trouver les contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filtrer les petits contours
    pieces_valides = []
    for contour in contours:
        aire = cv2.contourArea(contour)
        if aire > 200:
            pieces_valides.append(contour)
    
    nb_pieces = len(pieces_valides)
    
    # Dessiner les contours
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
if 'piece_selectionnee' not in st.session_state:
    st.session_state.piece_selectionnee = None
if 'photo_selectionnee' not in st.session_state:
    st.session_state.photo_selectionnee = None
if 'code_temporaire' not in st.session_state:
    st.session_state.code_temporaire = ""

gestionnaire = st.session_state.gestionnaire

# Interface principale
st.title("📦 Gestionnaire d'Inventaire Multi-Pièces")
st.markdown("""
Cette application permet de gérer l'inventaire de plusieurs types de pièces :
1. **Scanner** un code-barres (ou saisie manuelle)
2. **Ajouter** plusieurs photos pour cette pièce
3. **Changer** de pièce et répéter
4. **Exporter** un fichier Excel
""")

# Barre latérale
with st.sidebar:
    st.header("📋 Pièces en inventaire")
    
    if gestionnaire.pieces:
        for nom_piece in gestionnaire.pieces.keys():
            total = gestionnaire.get_total_piece(nom_piece)
            col1, col2 = st.columns([3, 1])
            with col1:
                if st.button(f"📦 {nom_piece}", key=f"select_{nom_piece}", use_container_width=True):
                    st.session_state.piece_selectionnee = nom_piece
                    st.session_state.page = "details"
            with col2:
                st.write(f"**{total}**")
        
        st.divider()
        
        if st.button("➕ Nouvelle pièce", use_container_width=True):
            st.session_state.page = "saisie"
            st.session_state.piece_selectionnee = None
        
        st.divider()
        
        if gestionnaire.pieces:
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
                st.session_state.piece_selectionnee = None
                st.rerun()
    else:
        st.info("Aucune pièce pour le moment")

# Contenu principal
if st.session_state.page == "saisie":
    st.header("➕ Ajouter une nouvelle pièce")
    
    # Options de saisie
    option_saisie = st.radio(
        "Méthode d'identification",
        ["📝 Saisie manuelle", "📷 Scanner code-barres (simulation)"],
        horizontal=True
    )
    
    code_trouve = None
    
    if option_saisie == "📷 Scanner code-barres (simulation)":
        st.markdown('<div class="barcode-scanner">', unsafe_allow_html=True)
        st.markdown("### 📷 Simulation de scan")
        st.info("💡 Version simplifiée - Entrez le code manuellement ou utilisez la simulation")
        
        col_scan1, col_scan2 = st.columns(2)
        
        with col_scan1:
            scan_option = st.radio("Source", ["📸 Caméra", "🖼️ Upload"], horizontal=True)
        
        if scan_option == "📸 Caméra":
            img_barcode = st.camera_input("Prendre une photo")
            if img_barcode:
                with st.spinner("🔍 Analyse..."):
                    bytes_data = img_barcode.getvalue()
                    frame = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
                    
                    # Version simplifiée
                    image_avec_code, codes = lire_code_barre_simple(frame)
                    
                    if codes:
                        code_trouve = codes[0]['data']
                        st.session_state.code_temporaire = code_trouve
                        st.image(cv2.cvtColor(image_avec_code, cv2.COLOR_BGR2RGB), 
                                caption="Code détecté (simulation)", use_container_width=True)
                    else:
                        st.warning("❌ Aucun code détecté. Utilisez la saisie manuelle.")
        
        else:
            uploaded_barcode = st.file_uploader("Choisir une image", type=['jpg', 'jpeg', 'png'])
            if uploaded_barcode:
                with st.spinner("🔍 Analyse..."):
                    file_bytes = np.asarray(bytearray(uploaded_barcode.read()), dtype=np.uint8)
                    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    
                    image_avec_code, codes = lire_code_barre_simple(frame)
                    
                    if codes:
                        code_trouve = codes[0]['data']
                        st.session_state.code_temporaire = code_trouve
                        st.image(cv2.cvtColor(image_avec_code, cv2.COLOR_BGR2RGB), 
                                caption="Code détecté (simulation)", use_container_width=True)
                    else:
                        st.warning("❌ Aucun code détecté. Utilisez la saisie manuelle.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Formulaire de création
    with st.form("nouvelle_piece"):
        if option_saisie == "📝 Saisie manuelle":
            nom_piece = st.text_input("Nom de la pièce", placeholder="Ex: Vis M8, Écrou, Rondelle...")
            code_manuel = st.text_input("Code (optionnel)", placeholder="Ex: PROD-12345")
            if code_manuel:
                st.session_state.code_temporaire = code_manuel
                st.info(f"✨ Code enregistré: {code_manuel}")
        else:
            if st.session_state.code_temporaire:
                nom_piece = st.text_input(
                    "Nom de la pièce (modifiable)",
                    value=formater_nom_piece(st.session_state.code_temporaire),
                    placeholder="Nom de la pièce"
                )
                st.info(f"✨ Code détecté: {st.session_state.code_temporaire}")
            else:
                nom_piece = st.text_input(
                    "Nom de la pièce",
                    placeholder="Ex: Vis M8, Écrou, Rondelle..."
                )
                st.warning("⚠️ Aucun code scanné - Vous pouvez entrer le nom manuellement")
        
        col1, col2 = st.columns(2)
        with col1:
            submitted = st.form_submit_button("✅ Créer la pièce", use_container_width=True)
        with col2:
            cancelled = st.form_submit_button("❌ Annuler", use_container_width=True)
    
    if submitted:
        if nom_piece:
            if gestionnaire.creer_nouvelle_piece(nom_piece):
                st.success(f"✅ Pièce '{nom_piece}' créée avec succès!")
                st.session_state.piece_selectionnee = nom_piece
                st.session_state.page = "details"
                st.session_state.code_temporaire = ""
                st.rerun()
            else:
                st.error("❌ Ce nom de pièce existe déjà")
        else:
            st.error("❌ Veuillez entrer un nom de pièce")

# ... (le reste du code pour les pages "details" et "photo_detail" reste identique à votre version précédente)

# Pied de page
st.markdown("---")
col_f1, col_f2, col_f3 = st.columns(3)
with col_f1:
    st.caption("📦 Gestionnaire d'Inventaire v2.0")
with col_f2:
    total_global = sum(gestionnaire.get_tous_les_totaux().values())
    st.caption(f"🧩 Total global: {total_global} pièces")
with col_f3:
    st.caption(f"📊 Types: {len(gestionnaire.pieces)}")
