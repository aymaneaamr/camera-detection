import streamlit as st
import cv2
import numpy as np
from collections import defaultdict
from PIL import Image
import tempfile
import os
import time

# Configuration de la page
st.set_page_config(
    page_title="Compteur de Pièces",
    page_icon="🧩",
    layout="wide"
)

class CompteurPieces:
    def __init__(self):
        """Initialise le compteur de pièces"""
        # Couleurs HSV
        self.couleurs = {
            'rouge': {
                'lower1': np.array([0, 100, 100]), 'upper1': np.array([10, 255, 255]),
                'lower2': np.array([160, 100, 100]), 'upper2': np.array([180, 255, 255]),
                'couleur_bbox': (0, 0, 255)  # BGR pour OpenCV
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

# Initialisation du compteur dans la session
if 'compteur' not in st.session_state:
    st.session_state.compteur = CompteurPieces()
if 'frame_count' not in st.session_state:
    st.session_state.frame_count = 0
if 'historique_images' not in st.session_state:
    st.session_state.historique_images = []

# Interface Streamlit
st.title("🧩 Compteur de Pièces")
st.markdown("""
Cette application détecte et compte automatiquement les pièces en temps réel :
- **Détection par couleur** (rouge, bleu, vert, jaune)
- **Classification par taille** (P, M, G, TG)
- **Comptage en direct**
""")

# Sidebar pour les paramètres
with st.sidebar:
    st.header("⚙️ Configuration")
    
    source = st.radio(
        "Source vidéo",
        ["Caméra", "Uploader une image", "Uploader une vidéo"]
    )
    
    camera_index = st.number_input("Index caméra", min_value=0, max_value=3, value=0)
    
    st.markdown("---")
    st.header("📊 Statistiques")
    
    if st.button("🔄 Réinitialiser compteurs"):
        st.session_state.compteur.reset_compteur()
        st.session_state.frame_count = 0
        st.rerun()
    
    st.markdown("---")
    st.markdown("""
    **Commandes :**
    - **Q** : Quitter
    - **R** : Réinitialiser
    """)

# Zone principale
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📹 Flux vidéo")
    video_placeholder = st.empty()
    
    if source == "Caméra":
        run = st.checkbox("Démarrer la caméra", key="cam_toggle")
        FRAME_WINDOW = st.empty()
        
        if run:
            cap = cv2.VideoCapture(camera_index)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            
            stop_button = st.button("⏹️ Arrêter")
            
            while run and not stop_button:
                ret, frame = cap.read()
                if not ret:
                    st.error("Erreur de lecture caméra")
                    break
                
                st.session_state.frame_count += 1
                
                # Traitement
                resultat, pieces, stats_couleur, stats_taille, total_actuel = st.session_state.compteur.traiter_frame(frame)
                
                # Mise à jour du total cumulé (optionnel)
                # st.session_state.compteur.total_pieces_cumule += total_actuel
                
                # Conversion BGR -> RGB pour Streamlit
                resultat_rgb = cv2.cvtColor(resultat, cv2.COLOR_BGR2RGB)
                
                # Ajout des stats sur l'image
                h, w = resultat_rgb.shape[:2]
                
                # Total actuel
                cv2.putText(resultat_rgb, f"TOTAL: {total_actuel}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                
                # Stats par couleur
                y_start = 60
                for couleur, count in stats_couleur.items():
                    color_map = {'rouge': (255,0,0), 'bleu': (0,0,255), 
                                'vert': (0,255,0), 'jaune': (0,255,255), '?': (128,128,128)}
                    cv2.putText(resultat_rgb, f"{couleur}: {count}", (10, y_start),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_map.get(couleur, (255,255,255)), 1)
                    y_start += 20
                
                # Frame count
                cv2.putText(resultat_rgb, f"Frame: {st.session_state.frame_count}", (w-150, h-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                FRAME_WINDOW.image(resultat_rgb)
                
                # Mise à jour des stats dans la colonne 2
                with col2:
                    st.metric("Pièces détectées", total_actuel)
                    st.metric("Total cumulé", st.session_state.compteur.total_pieces_cumule)
                    
                    # Stats détaillées
                    st.write("**Par couleur :**")
                    for couleur, count in stats_couleur.items():
                        st.write(f"- {couleur}: {count}")
                    
                    st.write("**Par taille :**")
                    for taille, count in stats_taille.items():
                        st.write(f"- {taille}: {count}")
                
                time.sleep(0.05)
            else:
                if 'cap' in locals():
                    cap.release()
    
    elif source == "Uploader une image":
        uploaded_file = st.file_uploader("Choisir une image", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file:
            # Lire l'image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            # Traitement
            resultat, pieces, stats_couleur, stats_taille, total_actuel = st.session_state.compteur.traiter_frame(frame)
            
            # Affichage
            col_img1, col_img2 = st.columns(2)
            with col_img1:
                st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Originale")
            with col_img2:
                st.image(cv2.cvtColor(resultat, cv2.COLOR_BGR2RGB), 
                        caption=f"Détection: {total_actuel} pièces")
            
            # Stats
            with col2:
                st.metric("Pièces détectées", total_actuel)
                st.write("**Détails :**")
                for i, piece in enumerate(pieces, 1):
                    st.write(f"Pièce #{i} : {piece['couleur']} - {piece['taille']}")

with col2:
    st.subheader("📊 Analyse")
    
    if 'total_actuel' in locals():
        st.metric("Pièces détectées", total_actuel)
    
    st.markdown("---")
    st.markdown("""
    ### 📝 Légende couleurs
    - 🔴 **Rouge** : Boîte rouge
    - 🔵 **Bleu** : Boîte bleue
    - 🟢 **Vert** : Boîte verte
    - 🟡 **Jaune** : Boîte jaune
    
    ### 📏 Tailles
    - **P** : < 500 pixels
    - **M** : 500-2000 pixels
    - **G** : 2000-5000 pixels
    - **TG** : > 5000 pixels
    """)

# Pied de page
st.markdown("---")
st.caption("🧩 Compteur de Pièces v1.0 - Compatible Python 3.13")
