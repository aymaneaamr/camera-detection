import streamlit as st
import cv2
import numpy as np
from collections import defaultdict
from PIL import Image
import time

# Configuration de la page
st.set_page_config(
    page_title="Compteur de Pièces - Téléphone",
    page_icon="📱",
    layout="centered"  # Centré pour mobile
)

class CompteurPieces:
    def __init__(self):
        """Initialise le compteur de pièces"""
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
        
        self.seuils_taille = {
            'P': (0, 500),
            'M': (500, 2000),
            'G': (2000, 5000),
            'TG': (5000, float('inf'))
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
            
            cv2.rectangle(resultat, (x, y), (x+w, y+h), couleur_bbox, 2)
            cv2.circle(resultat, centre, 3, (255, 255, 255), -1)
            cv2.putText(resultat, f"#{len(pieces_actuelles)}", (x+5, y+20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        total_actuel = len(pieces_actuelles)
        
        return resultat, pieces_actuelles, stats_couleur_actuelles, stats_taille_actuelles, total_actuel

# Initialisation
if 'compteur' not in st.session_state:
    st.session_state.compteur = CompteurPieces()
if 'frame_count' not in st.session_state:
    st.session_state.frame_count = 0

# ============================================
# INTERFACE SIMPLIFIÉE POUR TÉLÉPHONE
# ============================================
st.title("📱 Compteur de Pièces")

# Grand bouton bien visible pour ouvrir la caméra
st.markdown("""
<style>
    .stCameraInput {
        border: 3px solid #4CAF50;
        border-radius: 15px;
        padding: 10px;
    }
    .stCameraInput button {
        background-color: #4CAF50;
        color: white;
        font-size: 20px;
        padding: 20px;
        border-radius: 15px;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Option 1: Grand bouton caméra
st.subheader("📸 Étape 1: Appuyez pour ouvrir la caméra")

# Créer un conteneur pour la caméra
camera_container = st.container()

with camera_container:
    # Le widget camera_input avec un key unique qui change à chaque rafraîchissement
    img_file = st.camera_input(
        "📱 Appuyez ici pour utiliser l'appareil photo",
        key=f"camera_{int(time.time())}",
        help="Appuyez pour prendre une photo"
    )

# Option 2: Bouton d'aide si la caméra ne s'ouvre pas
with st.expander("🔧 La caméra ne s'ouvre pas ?"):
    st.markdown("""
    **Sur iPhone:**
    1. Allez dans Réglages > Safari
    2. Descendez jusqu'à "Caméra"
    3. Sélectionnez "Autoriser"
    
    **Sur Android:**
    1. Allez dans Paramètres > Applications > Chrome
    2. Appuyez sur "Autorisations"
    3. Activez "Appareil photo"
    
    **Puis rafraîchissez la page** 🔄
    """)

# Traitement de l'image
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
                
                # Afficher le résultat
                st.subheader(f"✅ Résultat: {total_actuel} pièces")
                
                # Image résultat
                st.image(cv2.cvtColor(resultat, cv2.COLOR_BGR2RGB), 
                        caption=f"🎯 {total_actuel} pièces détectées", 
                        use_column_width=True)
                
                # Statistiques en grille
                st.subheader("📊 Détails")
                
                # Par couleur
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("🔴 Rouge", stats_couleur.get('rouge', 0))
                with col2:
                    st.metric("🔵 Bleu", stats_couleur.get('bleu', 0))
                with col3:
                    st.metric("🟢 Vert", stats_couleur.get('vert', 0))
                with col4:
                    st.metric("🟡 Jaune", stats_couleur.get('jaune', 0))
                
                # Par taille
                st.write("**Taille des pièces:**")
                col_t1, col_t2, col_t3, col_t4 = st.columns(4)
                with col_t1:
                    st.metric("P", stats_taille.get('P', 0))
                with col_t2:
                    st.metric("M", stats_taille.get('M', 0))
                with col_t3:
                    st.metric("G", stats_taille.get('G', 0))
                with col_t4:
                    st.metric("TG", stats_taille.get('TG', 0))
                
                # Bouton pour nouvelle photo
                if st.button("📸 Prendre une autre photo", use_container_width=True):
                    st.rerun()
                    
        except Exception as e:
            st.error(f"Erreur: {str(e)}")
            st.button("🔄 Réessayer", on_click=st.rerun)

else:
    # Message d'instruction quand aucune photo n'est prise
    st.info("👆 Appuyez sur le bouton vert pour ouvrir la caméra")
    
    # Option galerie
    with st.expander("📁 Ou choisir une photo depuis la galerie"):
        uploaded_file = st.file_uploader("Sélectionner une photo", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file:
            with st.spinner("🔍 Analyse..."):
                bytes_data = uploaded_file.getvalue()
                frame = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
                
                if frame is not None:
                    resultat, pieces, stats_couleur, stats_taille, total = st.session_state.compteur.traiter_frame(frame)
                    st.image(cv2.cvtColor(resultat, cv2.COLOR_BGR2RGB), 
                            caption=f"Résultat: {total} pièces", use_column_width=True)

# Pied de page
st.markdown("---")
st.caption("""
📱 Version optimisée pour téléphone
• Appuyez sur le bouton vert pour la caméra
• Autorisez l'accès si demandé
""")
