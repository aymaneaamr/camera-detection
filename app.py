import streamlit as st
import cv2
import numpy as np
from collections import defaultdict
from PIL import Image
import time
import os
from pathlib import Path

# Configuration de la page
st.set_page_config(
    page_title="Compteur de Pièces",
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="auto"
)

# ============================================
# 1. DÉFINITION DE LA CLASSE D'ABORD
# ============================================
class CompteurPieces:
    def __init__(self):
        """Initialise le compteur de pièces"""
        # Couleurs HSV
        self.couleurs = {
            'rouge': {
                'lower1': np.array([0, 100, 100]), 'upper1': np.array([10, 255, 255]),
                'lower2': np.array([160, 100, 100]), 'upper2': np.array([180, 255, 255]),
                'couleur_bbox': (0, 0, 255)
            },
            'bleu': {
                'lower': np.array([100, 150, 50]), 'upper': np.array([140, 255, 255]),
                'couleur_bbox': (255, 0, 0)
            },
            'vert': {
                'lower': np.array([40, 70, 70]), 'upper': np.array([80, 255, 255]),
                'couleur_bbox': (0, 255, 0)
            },
            'jaune': {
                'lower': np.array([20, 100, 100]), 'upper': np.array([30, 255, 255]),
                'couleur_bbox': (0, 255, 255)
            }
        }
        
        # Seuils de taille
        self.seuils_taille = {
            'P': (0, 500),      # Petite
            'M': (500, 2000),    # Moyenne
            'G': (2000, 5000),   # Grande
            'TG': (5000, float('inf'))  # Très Grande
        }
        
        self.reset_compteur()
    
    def reset_compteur(self):
        """Réinitialise tous les compteurs"""
        self.stats_couleur_total = defaultdict(int)
        self.stats_taille_total = defaultdict(int)
        self.total_pieces_cumule = 0
    
    def get_couleur_piece(self, hsv, contour):
        """Détermine la couleur d'une pièce"""
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        
        best_couleur = '?'
        best_score = 0
        best_color_bbox = (128, 128, 128)
        
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
                    best_color_bbox = params['couleur_bbox']
        
        return best_couleur, best_color_bbox
    
    def get_taille_piece(self, aire):
        """Détermine la taille d'une pièce"""
        for nom_taille, (min_vol, max_vol) in self.seuils_taille.items():
            if min_vol <= aire < max_vol:
                return nom_taille
        return '?'
    
    def traiter_frame(self, frame):
        """Traite une frame et retourne les pièces détectées"""
        resultat = frame.copy()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Détection des contours
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=2)
        edges = cv2.erode(edges, kernel, iterations=1)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        pieces_actuelles = []
        stats_couleur_actuelles = defaultdict(int)
        stats_taille_actuelles = defaultdict(int)
        
        for contour in contours:
            aire = cv2.contourArea(contour)
            if aire < 200:
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            centre = (x + w//2, y + h//2)
            
            couleur_nom, couleur_bbox = self.get_couleur_piece(hsv, contour)
            taille_nom = self.get_taille_piece(aire)
            
            pieces_actuelles.append({
                'contour': contour,
                'aire': aire,
                'bbox': (x, y, w, h),
                'couleur': couleur_nom,
                'taille': taille_nom,
                'centre': centre
            })
            
            stats_couleur_actuelles[couleur_nom] += 1
            stats_taille_actuelles[taille_nom] += 1
            
            # Dessiner la pièce
            cv2.rectangle(resultat, (x, y), (x+w, y+h), couleur_bbox, 2)
            cv2.circle(resultat, centre, 3, (255, 255, 255), -1)
            cv2.putText(resultat, f"#{len(pieces_actuelles)}", (x+5, y+20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        total_actuel = len(pieces_actuelles)
        
        return resultat, pieces_actuelles, stats_couleur_actuelles, stats_taille_actuelles, total_actuel

# ============================================
# 2. DÉTECTION DU TYPE D'APPAREIL (simplifiée)
# ============================================
def detecter_appareil():
    """Détecte si l'utilisateur est sur mobile (version simplifiée)"""
    try:
        # Version simplifiée sans JavaScript
        # On utilise la largeur de l'écran via les métadonnées
        return False  # Par défaut, on suppose PC
    except:
        return False

# ============================================
# 3. INITIALISATION DES ÉTATS DE SESSION
# ============================================
if 'compteur' not in st.session_state:
    st.session_state.compteur = CompteurPieces()  # Maintenant la classe est définie
if 'frame_count' not in st.session_state:
    st.session_state.frame_count = 0
if 'is_mobile' not in st.session_state:
    st.session_state.is_mobile = False  # On désactive la détection mobile pour l'instant
if 'camera_active' not in st.session_state:
    st.session_state.camera_active = False
if 'photos_prises' not in st.session_state:
    st.session_state.photos_prises = []

# ============================================
# 4. INTERFACE PRINCIPALE
# ============================================
st.title("🧩 Compteur de Pièces")

# Afficher le mode actuel
device_emoji = "📱" if st.session_state.is_mobile else "💻"
st.caption(f"{device_emoji} Mode : {'Téléphone' if st.session_state.is_mobile else 'PC'}")

# Sidebar simplifiée pour le téléphone
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Options adaptées au téléphone (simplifiées)
    source = st.radio(
        "Source",
        ["📸 Appareil photo", "🖼️ Galerie", "🧪 Mode démo"],
        horizontal=True  # Horizontal pour mobile
    )
    
    st.markdown("---")
    
    if st.button("🔄 Réinitialiser"):
        st.session_state.compteur.reset_compteur()
        st.session_state.frame_count = 0
        st.rerun()
    
    st.markdown("---")
    
    # Légende simplifiée
    with st.expander("📝 Légende"):
        st.markdown("""
        - 🔴 Rouge
        - 🔵 Bleu  
        - 🟢 Vert
        - 🟡 Jaune
        - **P** < 500 px
        - **M** 500-2000 px
        - **G** 2000-5000 px
        - **TG** > 5000 px
        """)

# ============================================
# 5. TRAITEMENT SELON LA SOURCE
# ============================================
if source == "📸 Appareil photo":
    st.subheader("📸 Prenez une photo")
    
    # Widget caméra
    img_file = st.camera_input(
        "Appuyez pour prendre une photo",
        key=f"camera_{time.time()}",
        help="Utilisez l'appareil photo de votre téléphone"
    )
    
    if img_file is not None:
        with st.spinner("🔍 Analyse en cours..."):
            try:
                # Lire l'image
                bytes_data = img_file.getvalue()
                frame = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
                
                if frame is not None:
                    # Redimensionner pour le téléphone
                    height, width = frame.shape[:2]
                    if width > 400:
                        scale = 400 / width
                        new_width = int(width * scale)
                        new_height = int(height * scale)
                        frame = cv2.resize(frame, (new_width, new_height))
                    
                    # Traitement
                    resultat, pieces, stats_couleur, stats_taille, total_actuel = st.session_state.compteur.traiter_frame(frame)
                    st.session_state.frame_count += 1
                    
                    # Sauvegarder dans l'historique
                    st.session_state.photos_prises.append({
                        'time': time.time(),
                        'total': total_actuel
                    })
                    
                    # Afficher le résultat
                    st.image(cv2.cvtColor(resultat, cv2.COLOR_BGR2RGB), 
                            caption=f"🎯 {total_actuel} pièces", use_column_width=True)
                    
                    # Statistiques simples
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Total", total_actuel)
                    with col2:
                        st.metric("Frame", st.session_state.frame_count)
                    
                    # Détail des couleurs
                    st.write("**Couleurs détectées:**")
                    cols = st.columns(4)
                    couleurs = ['rouge', 'bleu', 'vert', 'jaune']
                    emojis = ['🔴', '🔵', '🟢', '🟡']
                    for i, couleur in enumerate(couleurs):
                        with cols[i]:
                            count = stats_couleur.get(couleur, 0)
                            st.metric(emojis[i], count)
            
            except Exception as e:
                st.error(f"Erreur lors de l'analyse: {str(e)}")

elif source == "🖼️ Galerie":
    st.subheader("🖼️ Choisir une photo")
    
    uploaded_file = st.file_uploader(
        "Sélectionner une photo",
        type=['jpg', 'jpeg', 'png'],
        help="Choisissez une photo dans votre galerie"
    )
    
    if uploaded_file:
        with st.spinner("🔍 Analyse en cours..."):
            try:
                bytes_data = uploaded_file.getvalue()
                frame = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
                
                if frame is not None:
                    # Redimensionner
                    height, width = frame.shape[:2]
                    if width > 400:
                        scale = 400 / width
                        new_width = int(width * scale)
                        new_height = int(height * scale)
                        frame = cv2.resize(frame, (new_width, new_height))
                    
                    # Traitement
                    resultat, pieces, stats_couleur, stats_taille, total_actuel = st.session_state.compteur.traiter_frame(frame)
                    
                    # Affichage
                    st.image(cv2.cvtColor(resultat, cv2.COLOR_BGR2RGB), 
                            caption=f"🎯 {total_actuel} pièces", use_column_width=True)
                    
                    # Statistiques
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Total", total_actuel)
                    
            except Exception as e:
                st.error(f"Erreur: {str(e)}")

else:  # Mode démo
    st.subheader("🧪 Mode démo")
    
    if st.button("🎲 Générer une image test"):
        # Créer une image de test
        test_img = np.zeros((480, 640, 3), dtype=np.uint8)
        test_img.fill(255)
        
        # Dessiner des cercles
        cv2.circle(test_img, (200, 200), 50, (0, 0, 255), -1)
        cv2.circle(test_img, (350, 250), 40, (255, 0, 0), -1)
        cv2.circle(test_img, (500, 200), 45, (0, 255, 0), -1)
        cv2.circle(test_img, (300, 350), 35, (0, 255, 255), -1)
        
        # Traitement
        resultat, pieces, stats_couleur, stats_taille, total = st.session_state.compteur.traiter_frame(test_img)
        
        # Affichage
        st.image(cv2.cvtColor(resultat, cv2.COLOR_BGR2RGB), 
                caption=f"🎯 {total} pièces", use_column_width=True)

# ============================================
# 6. HISTORIQUE SIMPLIFIÉ
# ============================================
if st.session_state.photos_prises:
    with st.expander("📜 Dernières photos"):
        for i, photo in enumerate(reversed(st.session_state.photos_prises[-5:])):
            st.write(f"Photo {i+1}: {photo['total']} pièces")

# ============================================
# 7. PIED DE PAGE
# ============================================
st.markdown("---")
st.caption("🧩 Compteur de Pièces v4.1 - Version Téléphone")
