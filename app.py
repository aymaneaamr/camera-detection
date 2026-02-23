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
        display: inline-block;
    }
    .piece-card {
        background: white;
        border-radius: 8px;
        padding: 10px;
        margin-bottom: 8px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .piece-card:hover {
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        border-color: #667eea;
    }
    .piece-name {
        font-weight: bold;
        color: #2c3e50;
    }
    .piece-location {
        background: #17a2b8;
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.8rem;
        display: inline-block;
        margin-left: 5px;
    }
    .piece-total {
        background: #667eea;
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.8rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

class GestionnairePieces:
    def __init__(self):
        """Initialise le gestionnaire de pièces"""
        self.pieces = {}  # Dictionnaire {nom_piece: {"photos": [], "emplacement": ""}}
        self.reset_piece_courante()
    
    def reset_piece_courante(self):
        """Réinitialise la pièce en cours de saisie"""
        self.piece_courante = {
            'nom': '',
            'emplacement': '',
            'photos': [],
            'total_pieces': 0
        }
    
    def creer_nouvelle_piece(self, nom_piece, emplacement=""):
        """Crée une nouvelle pièce dans l'inventaire avec son emplacement"""
        if nom_piece and nom_piece not in self.pieces:
            self.pieces[nom_piece] = {
                'photos': [],
                'emplacement': emplacement,
                'date_creation': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            return True
        return False
    
    def ajouter_photo_piece(self, nom_piece, frame_original, frame_analyse, nb_pieces):
        """Ajoute une photo analysée à une pièce existante"""
        if nom_piece in self.pieces:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Convertir les images en base64
            _, buffer_original = cv2.imencode('.jpg', frame_original)
            _, buffer_analyse = cv2.imencode('.jpg', frame_analyse)
            
            photo_data = {
                'timestamp': timestamp,
                'nb_pieces': nb_pieces,
                'image_originale': base64.b64encode(buffer_original).decode('utf-8'),
                'image_analyse': base64.b64encode(buffer_analyse).decode('utf-8'),
                'id': len(self.pieces[nom_piece]['photos'])
            }
            
            self.pieces[nom_piece]['photos'].append(photo_data)
            return True
        return False
    
    def get_total_piece(self, nom_piece):
        """Retourne le total de pièces pour un nom donné"""
        if nom_piece in self.pieces:
            return sum(photo['nb_pieces'] for photo in self.pieces[nom_piece]['photos'])
        return 0
    
    def get_photos_piece(self, nom_piece):
        """Retourne toutes les photos d'une pièce"""
        if nom_piece in self.pieces:
            return self.pieces[nom_piece]['photos']
        return []
    
    def get_emplacement_piece(self, nom_piece):
        """Retourne l'emplacement d'une pièce"""
        if nom_piece in self.pieces:
            return self.pieces[nom_piece].get('emplacement', '')
        return ''
    
    def supprimer_photo(self, nom_piece, photo_id):
        """Supprime une photo d'une pièce"""
        if nom_piece in self.pieces and 0 <= photo_id < len(self.pieces[nom_piece]['photos']):
            del self.pieces[nom_piece]['photos'][photo_id]
            # Réindexer les IDs
            for i, photo in enumerate(self.pieces[nom_piece]['photos']):
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
    
    def get_tous_emplacements(self):
        """Retourne un dictionnaire avec tous les emplacements par pièce"""
        return {nom: self.get_emplacement_piece(nom) for nom in self.pieces}
    
    def generer_excel(self):
        """Génère un fichier Excel avec l'inventaire complet"""
        # Créer un nouveau classeur Excel
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
        
        # Feuille principale - Résumé
        sheet_resume = workbook.active
        sheet_resume.title = "Inventaire"
        
        # En-têtes (ajout de la colonne Emplacement)
        headers = ["Nom de la pièce", "Emplacement", "Quantité totale", "Nombre de photos", "Dernière mise à jour"]
        for col, header in enumerate(headers, 1):
            cell = sheet_resume.cell(row=1, column=col)
            cell.value = header
            cell.font = header_font
            cell.fill = header_fill_bleu
            cell.alignment = Alignment(horizontal="center")
            cell.border = border
        
        # Données du résumé
        row = 2
        for nom_piece, data in self.pieces.items():
            total = sum(p['nb_pieces'] for p in data['photos'])
            nb_photos = len(data['photos'])
            derniere_date = data['photos'][-1]['timestamp'] if data['photos'] else data.get('date_creation', 'N/A')
            emplacement = data.get('emplacement', '')
            
            sheet_resume.cell(row=row, column=1).value = nom_piece
            sheet_resume.cell(row=row, column=2).value = emplacement
            sheet_resume.cell(row=row, column=3).value = total
            sheet_resume.cell(row=row, column=4).value = nb_photos
            sheet_resume.cell(row=row, column=5).value = derniere_date
            
            for col in range(1, 6):
                cell = sheet_resume.cell(row=row, column=col)
                cell.border = border
            row += 1
        
        # Ajuster la largeur des colonnes
        sheet_resume.column_dimensions['A'].width = 25
        sheet_resume.column_dimensions['B'].width = 20
        sheet_resume.column_dimensions['C'].width = 15
        sheet_resume.column_dimensions['D'].width = 15
        sheet_resume.column_dimensions['E'].width = 22
        
        # Feuille de détail
        sheet_detail = workbook.create_sheet("Détail des photos")
        
        # En-têtes détail (ajout de la colonne Emplacement)
        detail_headers = ["Pièce", "Emplacement", "Photo #", "Date", "Nombre de pièces"]
        for col, header in enumerate(detail_headers, 1):
            cell = sheet_detail.cell(row=1, column=col)
            cell.value = header
            cell.font = header_font
            cell.fill = header_fill_vert
            cell.alignment = Alignment(horizontal="center")
            cell.border = border
        
        # Données détaillées
        row = 2
        for nom_piece, data in self.pieces.items():
            emplacement = data.get('emplacement', '')
            for i, photo in enumerate(data['photos'], 1):
                sheet_detail.cell(row=row, column=1).value = nom_piece
                sheet_detail.cell(row=row, column=2).value = emplacement
                sheet_detail.cell(row=row, column=3).value = f"Photo {i}"
                sheet_detail.cell(row=row, column=4).value = photo['timestamp']
                sheet_detail.cell(row=row, column=5).value = photo['nb_pieces']
                
                for col in range(1, 6):
                    cell = sheet_detail.cell(row=row, column=col)
                    cell.border = border
                row += 1
        
        # Ajuster les colonnes du détail
        sheet_detail.column_dimensions['A'].width = 25
        sheet_detail.column_dimensions['B'].width = 20
        sheet_detail.column_dimensions['C'].width = 12
        sheet_detail.column_dimensions['D'].width = 22
        sheet_detail.column_dimensions['E'].width = 18
        
        workbook.save(output)
        output.seek(0)
        return output
    
    def reinitialiser_tout(self):
        """Réinitialise complètement l'inventaire"""
        self.pieces = {}

# Fonction pour détecter et lire les codes-barres
def detecter_code_barre(image):
    """Détecte et lit les codes-barres dans une image"""
    resultat = image.copy()
    codes_detectes = []
    
    # Conversion en niveaux de gris
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Décoder les codes-barres
    codes = decode(gray)
    
    for code in codes:
        # Extraire les données
        data = code.data.decode('utf-8')
        type_code = code.type
        
        # Dessiner le rectangle autour du code
        points = code.polygon
        if len(points) == 4:
            pts = np.array([(p.x, p.y) for p in points], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(resultat, [pts], True, (0, 255, 0), 3)
        
        # Ajouter le texte
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
    
    # Filtrer les petits contours (bruit)
    pieces_valides = []
    for contour in contours:
        aire = cv2.contourArea(contour)
        if aire > 200:  # Seuil minimum
            pieces_valides.append(contour)
    
    nb_pieces = len(pieces_valides)
    
    # Dessiner les contours
    for contour in pieces_valides:
        # Dessiner le contour en vert
        cv2.drawContours(resultat, [contour], -1, (0, 255, 0), 2)
        
        # Ajouter un point au centre
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.circle(resultat, (cx, cy), 3, (0, 0, 255), -1)
    
    # Ajouter le compteur
    cv2.putText(resultat, f"Pieces: {nb_pieces}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    return resultat, nb_pieces

# Fonction pour décoder l'image base64
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
if 'code_detecte' not in st.session_state:
    st.session_state.code_detecte = None
if 'scan_effectue' not in st.session_state:
    st.session_state.scan_effectue = False
if 'ajout_photo' not in st.session_state:
    st.session_state.ajout_photo = False

gestionnaire = st.session_state.gestionnaire

# Interface principale
st.title("📦 Gestionnaire d'Inventaire Multi-Pièces avec Scan Code-Barres")
st.markdown("""
Cette application permet de gérer l'inventaire de plusieurs types de pièces :
1. **Scanner** un code-barres pour identifier automatiquement la pièce
2. **Ajouter** un emplacement de stockage (optionnel)
3. **Ajouter** plusieurs photos pour cette pièce
4. **Changer** de pièce et répéter
5. **Exporter** un fichier Excel avec tous les totaux
""")

# Barre latérale avec la liste des pièces
with st.sidebar:
    st.header("📋 Pièces en inventaire")
    
    if gestionnaire.pieces:
        for nom_piece in gestionnaire.pieces.keys():
            total = gestionnaire.get_total_piece(nom_piece)
            emplacement = gestionnaire.get_emplacement_piece(nom_piece)
            
            # Créer un conteneur pour chaque pièce
            with st.container():
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    # Afficher le nom avec l'emplacement en markdown
                    if emplacement:
                        st.markdown(f"**📦 {nom_piece}**  \n📍 `{emplacement}`")
                    else:
                        st.markdown(f"**📦 {nom_piece}**")
                    
                    # Bouton de sélection
                    if st.button("Sélectionner", key=f"select_{nom_piece}", use_container_width=True):
                        st.session_state.piece_selectionnee = nom_piece
                        st.session_state.page = "details"
                        st.rerun()
                
                with col2:
                    st.markdown(f"**{total}**")
                
                st.divider()
        
        # Bouton pour retourner à la saisie
        if st.button("➕ Nouvelle pièce", use_container_width=True):
            st.session_state.page = "saisie"
            st.session_state.piece_selectionnee = None
            st.session_state.code_detecte = None
            st.session_state.scan_effectue = False
            st.rerun()
        
        st.divider()
        
        # Export Excel
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
            
            # Réinitialisation
            if st.button("🔄 Tout réinitialiser", type="primary", use_container_width=True):
                gestionnaire.reinitialiser_tout()
                st.session_state.page = "saisie"
                st.session_state.piece_selectionnee = None
                st.rerun()
    else:
        st.info("Aucune pièce pour le moment")

# Contenu principal
if st.session_state.page == "saisie":
    # Page de saisie d'une nouvelle pièce avec scan de code-barres
    st.header("➕ Ajouter une nouvelle pièce")
    
    # Section scan de code-barres
    st.markdown('<div class="barcode-scanner">', unsafe_allow_html=True)
    st.markdown("### 📷 Scanner le code-barres de la pièce")
    st.markdown("Prenez une photo du code-barres pour identifier automatiquement la pièce")
    
    col_scan1, col_scan2 = st.columns(2)
    
    with col_scan1:
        scan_option = st.radio("Source", ["📸 Caméra", "🖼️ Upload"], horizontal=True, key="scan_source")
    
    if scan_option == "📸 Caméra":
        img_barcode = st.camera_input("Prendre une photo du code-barres", key="camera_barcode")
        if img_barcode:
            with st.spinner("🔍 Analyse du code-barres..."):
                bytes_data = img_barcode.getvalue()
                frame = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
                
                # Détection du code-barres
                image_annotee, codes = detecter_code_barre(frame)
                
                if codes:
                    # Prendre le premier code détecté
                    code_trouve = codes[0]['data']
                    st.session_state.code_detecte = code_trouve
                    st.session_state.scan_effectue = True
                    
                    # Afficher l'image avec le code détecté
                    st.image(cv2.cvtColor(image_annotee, cv2.COLOR_BGR2RGB), 
                            caption="Code-barres détecté", use_container_width=True)
                    
                    st.markdown(f"""
                    <div class="success-box">
                        <h4>✅ Code-barres détecté !</h4>
                        <div class="code-display">{code_trouve}</div>
                        <p><strong>Type :</strong> {codes[0]['type']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning("❌ Aucun code-barres détecté. Veuillez réessayer avec une image plus claire.")
    
    else:  # Upload
        uploaded_barcode = st.file_uploader("Choisir une image de code-barres", type=['jpg', 'jpeg', 'png'], key="upload_barcode")
        if uploaded_barcode:
            with st.spinner("🔍 Analyse du code-barres..."):
                file_bytes = np.asarray(bytearray(uploaded_barcode.read()), dtype=np.uint8)
                frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                
                # Détection du code-barres
                image_annotee, codes = detecter_code_barre(frame)
                
                # Afficher l'image
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
                    st.warning("❌ Aucun code-barres détecté. Veuillez réessayer avec une image plus claire.")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Bouton pour réinitialiser le scan
    if st.session_state.scan_effectue:
        if st.button("🔄 Nouveau scan", use_container_width=True):
            st.session_state.scan_effectue = False
            st.session_state.code_detecte = None
            st.rerun()
    
    st.markdown("---")
    
    # Formulaire de création de pièce avec emplacement optionnel
    st.markdown("### 📝 Informations de la pièce")
    
    # Utiliser une clé dynamique pour le text_input qui change quand le code est détecté
    input_key = f"nom_piece_input_{st.session_state.code_detecte or 'manuel'}"
    
    # Définir la valeur par défaut en fonction du code détecté
    default_value = st.session_state.code_detecte if st.session_state.code_detecte else ""
    
    # Deux colonnes pour le nom et l'emplacement
    col_nom, col_emp = st.columns([2, 1])
    
    with col_nom:
        nom_piece = st.text_input(
            "Nom de la pièce *",
            value=default_value,
            placeholder="Nom de la pièce (obligatoire)",
            key=input_key
        )
    
    with col_emp:
        emplacement = st.text_input(
            "Emplacement (optionnel)",
            placeholder="Ex: A-12, Rayon 3...",
            key="emplacement_input"
        )
    
    st.caption("* Champ obligatoire")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Créer la pièce", use_container_width=True):
            if nom_piece:
                if gestionnaire.creer_nouvelle_piece(nom_piece, emplacement):
                    st.success(f"✅ Pièce '{nom_piece}' créée avec succès!")
                    if emplacement:
                        st.info(f"📍 Emplacement: {emplacement}")
                    st.session_state.piece_selectionnee = nom_piece
                    st.session_state.page = "details"
                    st.session_state.code_detecte = None
                    st.session_state.scan_effectue = False
                    st.rerun()
                else:
                    st.error("❌ Ce nom de pièce existe déjà")
            else:
                st.error("❌ Veuillez entrer un nom de pièce")
    with col2:
        if st.button("❌ Annuler", use_container_width=True):
            st.session_state.code_detecte = None
            st.session_state.scan_effectue = False
            st.rerun()

elif st.session_state.page == "details" and st.session_state.piece_selectionnee:
    # Page de détails d'une pièce
    nom_piece = st.session_state.piece_selectionnee
    photos = gestionnaire.get_photos_piece(nom_piece)
    total = gestionnaire.get_total_piece(nom_piece)
    emplacement = gestionnaire.get_emplacement_piece(nom_piece)
    
    # En-tête avec emplacement
    col_h1, col_h2, col_h3, col_h4 = st.columns([2, 1, 1, 1])
    with col_h1:
        st.header(f"📦 {nom_piece}")
        if emplacement:
            st.markdown(f"<span class='location-badge'>📍 {emplacement}</span>", unsafe_allow_html=True)
    with col_h2:
        st.metric("Total pièces", total)
    with col_h3:
        st.metric("Photos", len(photos))
    with col_h4:
        if emplacement:
            st.metric("Emplacement", emplacement)
    
    # Afficher un badge si le nom est un code
    if re.match(r'^[A-Z0-9-]+$', nom_piece):
        st.info(f"🔖 Code produit: {nom_piece}")
    
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
        if st.button("🗑️ Supprimer cette pièce", use_container_width=True, type="primary"):
            if gestionnaire.supprimer_piece(nom_piece):
                st.success(f"✅ Pièce '{nom_piece}' supprimée")
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
                    
                    if gestionnaire.ajouter_photo_piece(nom_piece, frame, resultat, nb_pieces):
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
                    
                    if gestionnaire.ajouter_photo_piece(nom_piece, frame, resultat, nb_pieces):
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
                    if st.button("🔍 Voir", key=f"view_{nom_piece}_{i}"):
                        st.session_state.photo_selectionnee = photo['id']
                        st.session_state.page = "photo_detail"
                        st.rerun()
                with col_b2:
                    if st.button("🗑️", key=f"del_{nom_piece}_{i}"):
                        if gestionnaire.supprimer_photo(nom_piece, photo['id']):
                            st.rerun()
    
    else:
        st.info("📸 Aucune photo pour cette pièce. Cliquez sur 'Ajouter une photo' pour commencer.")

elif st.session_state.page == "photo_detail" and st.session_state.piece_selectionnee and st.session_state.photo_selectionnee is not None:
    # Détail d'une photo spécifique
    nom_piece = st.session_state.piece_selectionnee
    photos = gestionnaire.get_photos_piece(nom_piece)
    photo_id = st.session_state.photo_selectionnee
    
    if 0 <= photo_id < len(photos):
        photo = photos[photo_id]
        
        st.header(f"🔍 Détail de la photo - {nom_piece}")
        
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
            if st.button("⬅️ Retour à la pièce", use_container_width=True):
                st.session_state.page = "details"
                st.session_state.photo_selectionnee = None
                st.rerun()
        with col_b2:
            if st.button("🗑️ Supprimer cette photo", use_container_width=True, type="primary"):
                if gestionnaire.supprimer_photo(nom_piece, photo_id):
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
col_f1, col_f2, col_f3, col_f4 = st.columns(4)
with col_f1:
    st.caption("📦 Gestionnaire d'Inventaire v3.0 - Avec scan code-barres")
with col_f2:
    total_global = sum(gestionnaire.get_tous_les_totaux().values())
    st.caption(f"🧩 Total global: {total_global} pièces")
with col_f3:
    st.caption(f"📊 Types: {len(gestionnaire.pieces)}")
with col_f4:
    emplacements_renseignes = sum(1 for e in gestionnaire.get_tous_emplacements().values() if e)
    st.caption(f"📍 Emplacements: {emplacements_renseignes}/{len(gestionnaire.pieces)}")
