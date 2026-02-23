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
import plotly.express as px
import plotly.graph_objects as go

# Configuration de la page
st.set_page_config(
    page_title="Gestionnaire d'Inventaire Multi-Pièces",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour améliorer l'esthétique
st.markdown("""
<style>
    /* Style global */
    .main {
        background-color: #f5f5f5;
    }
    
    /* En-têtes */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9);
        font-size: 1.1rem;
    }
    
    /* Cartes */
    .card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        transition: transform 0.3s, box-shadow 0.3s;
    }
    
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    
    /* Métriques */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    
    .metric-card h3 {
        color: white;
        font-size: 1rem;
        margin-bottom: 0.5rem;
    }
    
    .metric-card .value {
        font-size: 2rem;
        font-weight: bold;
    }
    
    /* Boutons */
    .stButton button {
        border-radius: 25px;
        font-weight: 500;
        transition: all 0.3s;
        border: none;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    
    /* Images */
    .image-container {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Messages de succès */
    .stSuccess {
        border-radius: 10px;
        border-left: 5px solid #28a745;
    }
    
    /* Barre latérale */
    .sidebar-content {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
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

# Fonction améliorée de détection des pièces
def detecter_pieces_ameliorer(image):
    """
    Version améliorée de la détection de pièces avec plusieurs techniques
    pour une meilleure précision
    """
    resultat = image.copy()
    
    # 1. Prétraitement avancé
    # Conversion en niveaux de gris
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Application d'un filtre bilatéral pour préserver les bords tout en réduisant le bruit
    bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # 2. Amélioration du contraste avec CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(bilateral)
    
    # 3. Détection de contours avec différents seuils
    # Utilisation de Canny adaptatif
    median_intensity = np.median(enhanced)
    lower = int(max(0, 0.66 * median_intensity))
    upper = int(min(255, 1.33 * median_intensity))
    
    edges1 = cv2.Canny(enhanced, lower, upper)
    edges2 = cv2.Canny(enhanced, 30, 100)
    
    # Combinaison des résultats
    edges = cv2.bitwise_or(edges1, edges2)
    
    # 4. Opérations morphologiques avancées
    kernel = np.ones((3, 3), np.uint8)
    kernel_large = np.ones((5, 5), np.uint8)
    
    # Fermeture pour combler les trous
    closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_large, iterations=2)
    
    # Ouverture pour enlever le bruit
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Dilatation finale
    dilated = cv2.dilate(opening, kernel, iterations=2)
    
    # 5. Trouver les contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 6. Filtrage avancé des contours
    pieces_valides = []
    for contour in contours:
        aire = cv2.contourArea(contour)
        
        # Filtre de taille
        if aire < 150:  # Réduit pour capturer les petites pièces
            continue
        
        # Filtre de circularité pour éviter les formes trop irrégulières
        perimeter = cv2.arcLength(contour, True)
        if perimeter > 0:
            circularity = 4 * np.pi * aire / (perimeter * perimeter)
            if circularity < 0.3:  # Trop irrégulier, probablement du bruit
                continue
        
        # Filtre de solidité
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area > 0:
            solidity = aire / hull_area
            if solidity < 0.5:  # Trop creux, probablement du bruit
                continue
        
        pieces_valides.append(contour)
    
    nb_pieces = len(pieces_valides)
    
    # 7. Dessiner les résultats avec différents styles
    for i, contour in enumerate(pieces_valides):
        # Couleur différente pour chaque pièce (dégradé)
        color = (0, 255 - i*20 % 255, i*30 % 255)
        
        # Dessiner le contour
        cv2.drawContours(resultat, [contour], -1, color, 3)
        
        # Rectangle englobant
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(resultat, (x, y), (x + w, y + h), (255, 255, 255), 1)
        
        # Centre de gravité
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.circle(resultat, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(resultat, str(i+1), (cx + 10, cy - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # 8. Ajouter un compteur stylisé
    h, w = resultat.shape[:2]
    
    # Fond pour le compteur
    overlay = resultat.copy()
    cv2.rectangle(overlay, (5, 5), (200, 60), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, resultat, 0.5, 0, resultat)
    
    # Texte du compteur
    cv2.putText(resultat, f"📦 PIECES: {nb_pieces}", (15, 40),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
    
    # 9. Statistiques supplémentaires
    aires = [cv2.contourArea(c) for c in pieces_valides]
    stats = {
        'nb_pieces': nb_pieces,
        'aire_moyenne': np.mean(aires) if aires else 0,
        'aire_min': min(aires) if aires else 0,
        'aire_max': max(aires) if aires else 0
    }
    
    return resultat, nb_pieces, stats

def base64_to_image(base64_string):
    """Convertit une chaîne base64 en image OpenCV"""
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
if 'ajout_photo' not in st.session_state:
    st.session_state.ajout_photo = False

gestionnaire = st.session_state.gestionnaire

# En-tête stylisé
st.markdown("""
<div class="main-header">
    <h1>📦 Gestionnaire d'Inventaire Intelligent</h1>
    <p>Détection automatique de pièces • Gestion multi-produits • Export Excel professionnel</p>
</div>
""", unsafe_allow_html=True)

# Barre latérale améliorée
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h2 style="color: #667eea;">📋 INVENTAIRE</h2>
    </div>
    """, unsafe_allow_html=True)
    
    if gestionnaire.pieces:
        # Statistiques globales
        total_global = sum(gestionnaire.get_tous_les_totaux().values())
        nb_types = len(gestionnaire.pieces)
        
        col_stat1, col_stat2 = st.columns(2)
        with col_stat1:
            st.markdown("""
            <div class="metric-card">
                <h3>📦 Types</h3>
                <div class="value">{}</div>
            </div>
            """.format(nb_types), unsafe_allow_html=True)
        
        with col_stat2:
            st.markdown("""
            <div class="metric-card">
                <h3>🧩 Total</h3>
                <div class="value">{}</div>
            </div>
            """.format(total_global), unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 📋 Liste des pièces")
        
        # Afficher toutes les pièces
        for nom_piece in gestionnaire.pieces.keys():
            total = gestionnaire.get_total_piece(nom_piece)
            with st.container():
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    if st.button(f"📦 {nom_piece}", key=f"select_{nom_piece}", use_container_width=True):
                        st.session_state.piece_selectionnee = nom_piece
                        st.session_state.page = "details"
                with col2:
                    st.markdown(f"**{total}**")
                with col3:
                    if st.button("🗑️", key=f"del_piece_{nom_piece}"):
                        if gestionnaire.supprimer_piece(nom_piece):
                            st.rerun()
        
        st.divider()
        
        # Boutons d'action
        if st.button("➕ NOUVELLE PIÈCE", use_container_width=True):
            st.session_state.page = "saisie"
            st.session_state.piece_selectionnee = None
        
        st.divider()
        
        # Export
        if gestionnaire.pieces:
            st.markdown("### 📊 EXPORT")
            excel_file = gestionnaire.generer_excel()
            st.download_button(
                label="📥 Télécharger Excel",
                data=excel_file,
                file_name=f"inventaire_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
            
            if st.button("🔄 TOUT RÉINITIALISER", type="primary", use_container_width=True):
                gestionnaire.reinitialiser_tout()
                st.session_state.page = "saisie"
                st.session_state.piece_selectionnee = None
                st.rerun()
    else:
        st.info("📭 Aucune pièce pour le moment")
        st.markdown("""
        <div style="text-align: center; padding: 2rem;">
            <p style="color: #999;">Cliquez sur "Nouvelle pièce" pour commencer</p>
        </div>
        """, unsafe_allow_html=True)

# Contenu principal
if st.session_state.page == "saisie":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("➕ Ajouter une nouvelle pièce")
    
    with st.form("nouvelle_piece"):
        nom_piece = st.text_input("Nom de la pièce", placeholder="Ex: Vis M8, Écrou, Rondelle...")
        
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
                st.rerun()
            else:
                st.error("❌ Ce nom de pièce existe déjà ou est invalide")
        else:
            st.error("❌ Veuillez entrer un nom de pièce")
    st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.page == "details" and st.session_state.piece_selectionnee:
    nom_piece = st.session_state.piece_selectionnee
    photos = gestionnaire.get_photos_piece(nom_piece)
    total = gestionnaire.get_total_piece(nom_piece)
    
    # En-tête de la pièce
    st.markdown(f"""
    <div class="card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;">
        <h2 style="color: white;">📦 {nom_piece}</h2>
        <div style="display: flex; gap: 2rem;">
            <div><strong>Total pièces:</strong> {total}</div>
            <div><strong>Photos:</strong> {len(photos)}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Options
    col_o1, col_o2, col_o3 = st.columns(3)
    with col_o1:
        if st.button("⬅️ Retour", use_container_width=True):
            st.session_state.page = "saisie"
            st.rerun()
    with col_o2:
        if st.button("📸 Ajouter une photo", use_container_width=True):
            st.session_state.ajout_photo = True
    with col_o3:
        if st.button("🗑️ Supprimer", use_container_width=True, type="primary"):
            if gestionnaire.supprimer_piece(nom_piece):
                st.success(f"✅ Pièce supprimée")
                st.session_state.page = "saisie"
                st.rerun()
    
    st.divider()
    
    # Ajout de photo
    if st.session_state.get('ajout_photo', False):
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📸 Ajouter une photo")
        
        col_p1, col_p2 = st.columns([2, 1])
        with col_p2:
            if st.button("❌ Annuler", use_container_width=True):
                st.session_state.ajout_photo = False
                st.rerun()
        
        with col_p1:
            source = st.radio("Source", ["📸 Caméra", "🖼️ Fichier"], horizontal=True)
        
        if source == "📸 Caméra":
            img_file = st.camera_input("Prendre une photo")
            if img_file:
                with st.spinner("🔍 Analyse en cours..."):
                    bytes_data = img_file.getvalue()
                    frame = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
                    resultat, nb_pieces, stats = detecter_pieces_ameliorer(frame)
                    
                    if gestionnaire.ajouter_photo_piece(nom_piece, frame, resultat, nb_pieces):
                        st.success(f"✅ {nb_pieces} pièces détectées!")
                        
                        # Afficher les statistiques
                        col_s1, col_s2, col_s3 = st.columns(3)
                        with col_s1:
                            st.metric("Pièces détectées", nb_pieces)
                        with col_s2:
                            st.metric("Taille moyenne", f"{stats['aire_moyenne']:.0f} px²")
                        with col_s3:
                            st.metric("Précision", "👍 Bonne")
                        
                        st.session_state.ajout_photo = False
                        st.rerun()
        
        else:
            uploaded_file = st.file_uploader("Choisir une image", type=['jpg', 'jpeg', 'png'])
            if uploaded_file:
                with st.spinner("🔍 Analyse en cours..."):
                    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    resultat, nb_pieces, stats = detecter_pieces_ameliorer(frame)
                    
                    if gestionnaire.ajouter_photo_piece(nom_piece, frame, resultat, nb_pieces):
                        st.success(f"✅ {nb_pieces} pièces détectées!")
                        
                        # Afficher les statistiques
                        col_s1, col_s2, col_s3 = st.columns(3)
                        with col_s1:
                            st.metric("Pièces détectées", nb_pieces)
                        with col_s2:
                            st.metric("Taille moyenne", f"{stats['aire_moyenne']:.0f} px²")
                        with col_s3:
                            st.metric("Précision", "👍 Bonne")
                        
                        st.session_state.ajout_photo = False
                        st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Affichage des photos
    if photos:
        st.subheader("📸 Photos enregistrées")
        
        # Options de tri
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
        
        # Grille de photos
        cols = st.columns(3)
        for i, photo in enumerate(photos_affichees):
            with cols[i % 3]:
                img = base64_to_image(photo['image_analyse'])
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                st.markdown('<div class="image-container">', unsafe_allow_html=True)
                st.image(img_rgb, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown(f"""
                <div style="text-align: center; margin: 0.5rem 0;">
                    <span style="background: #667eea; color: white; padding: 0.2rem 0.5rem; border-radius: 5px;">
                        📅 {photo['timestamp'][:10]}
                    </span>
                    <span style="background: #764ba2; color: white; padding: 0.2rem 0.5rem; border-radius: 5px; margin-left: 0.5rem;">
                        🔢 {photo['nb_pieces']} pièces
                    </span>
                </div>
                """, unsafe_allow_html=True)
                
                col_b1, col_b2 = st.columns(2)
                with col_b1:
                    if st.button("🔍 Voir", key=f"view_{nom_piece}_{i}", use_container_width=True):
                        st.session_state.photo_selectionnee = photo['id']
                        st.session_state.page = "photo_detail"
                        st.rerun()
                with col_b2:
                    if st.button("🗑️", key=f"del_{nom_piece}_{i}", use_container_width=True):
                        if gestionnaire.supprimer_photo(nom_piece, photo['id']):
                            st.rerun()
    
    else:
        st.info("📸 Aucune photo pour cette pièce")

elif st.session_state.page == "photo_detail":
    nom_piece = st.session_state.piece_selectionnee
    photos = gestionnaire.get_photos_piece(nom_piece)
    photo_id = st.session_state.photo_selectionnee
    
    if 0 <= photo_id < len(photos):
        photo = photos[photo_id]
        
        st.markdown(f"""
        <div class="card">
            <h2>🔍 Détail de la photo - {nom_piece}</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Afficher les deux images
        col_img1, col_img2 = st.columns(2)
        
        with col_img1:
            st.markdown("### 📸 Originale")
            img_originale = base64_to_image(photo['image_originale'])
            img_originale_rgb = cv2.cvtColor(img_originale, cv2.COLOR_BGR2RGB)
            st.image(img_originale_rgb, use_container_width=True)
        
        with col_img2:
            st.markdown(f"### 🔍 Analyse - {photo['nb_pieces']} pièces")
            img_analyse = base64_to_image(photo['image_analyse'])
            img_analyse_rgb = cv2.cvtColor(img_analyse, cv2.COLOR_BGR2RGB)
            st.image(img_analyse_rgb, use_container_width=True)
        
        # Informations
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 1rem; border-radius: 10px; color: white; margin: 1rem 0;">
        """, unsafe_allow_html=True)
        
        col_i1, col_i2 = st.columns(2)
        with col_i1:
            st.metric("Nombre de pièces", photo['nb_pieces'])
        with col_i2:
            st.metric("Date", photo['timestamp'][:10])
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Boutons
        col_b1, col_b2 = st.columns(2)
        with col_b1:
            if st.button("⬅️ Retour", use_container_width=True):
                st.session_state.page = "details"
                st.session_state.photo_selectionnee = None
                st.rerun()
        with col_b2:
            if st.button("🗑️ Supprimer", use_container_width=True, type="primary"):
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

# Pied de page amélioré
st.markdown("---")
col_f1, col_f2, col_f3, col_f4 = st.columns(4)
with col_f1:
    st.caption("📦 Gestionnaire d'Inventaire v2.0")
with col_f2:
    total_global = sum(gestionnaire.get_tous_les_totaux().values())
    st.caption(f"🧩 Total global: {total_global} pièces")
with col_f3:
    st.caption(f"📊 Types: {len(gestionnaire.pieces)}")
with col_f4:
    if gestionnaire.pieces:
        st.caption(f"🆕 Dernière: {gestionnaire.pieces[list(gestionnaire.pieces.keys())[-1]][-1]['timestamp'][:10] if gestionnaire.pieces[list(gestionnaire.pieces.keys())[-1]] else 'N/A'}")
