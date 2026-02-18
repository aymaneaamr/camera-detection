import streamlit as st
import cv2
import numpy as np
from collections import defaultdict
from PIL import Image
import time
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

# Configuration de la page
st.set_page_config(
    page_title="Compteur de Pièces - Direct",
    page_icon="🎥",
    layout="centered"
)

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
            'P': (0, 500),
            'M': (500, 2000),
            'G': (2000, 5000),
            'TG': (5000, float('inf'))
        }
        
        self.reset_compteur()
    
    def reset_compteur(self):
        """Réinitialise tous les compteurs"""
        self.stats_couleur = defaultdict(int)
        self.stats_taille = defaultdict(int)
        self.total_pieces = 0
    
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
        
        # Réinitialiser les stats pour cette frame
        stats_couleur_frame = defaultdict(int)
        stats_taille_frame = defaultdict(int)
        
        for contour in contours:
            aire = cv2.contourArea(contour)
            if aire < 200:  # Ignorer les petits contours
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            centre = (x + w//2, y + h//2)
            
            couleur_nom, couleur_bbox = self.get_couleur_piece(hsv, contour)
            taille_nom = self.get_taille_piece(aire)
            
            stats_couleur_frame[couleur_nom] += 1
            stats_taille_frame[taille_nom] += 1
            
            # Dessiner la pièce
            cv2.rectangle(resultat, (x, y), (x+w, y+h), couleur_bbox, 2)
            cv2.circle(resultat, centre, 3, (255, 255, 255), -1)
            cv2.putText(resultat, f"{couleur_nom[0]}{taille_nom}", (x+5, y+20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        total_actuel = len(contours)
        
        # Mettre à jour les stats cumulées
        self.stats_couleur = stats_couleur_frame
        self.stats_taille = stats_taille_frame
        self.total_pieces = total_actuel
        
        # Ajouter le compteur sur l'image
        h, w = resultat.shape[:2]
        cv2.putText(resultat, f"TOTAL: {total_actuel}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        return resultat

# Créer une instance globale du compteur
compteur = CompteurPieces()

# Interface
st.title("🎥 Compteur de Pièces - Temps Réel")
st.markdown("""
**Pointez la caméra vers les pièces - La détection est automatique !**
""")

# Sidebar avec les stats
with st.sidebar:
    st.header("📊 Statistiques")
    
    # Métrique principale
    total_placeholder = st.empty()
    total_placeholder.metric("Pièces détectées", 0)
    
    st.markdown("---")
    
    # Par couleur
    st.subheader("🎨 Par couleur")
    col1, col2 = st.columns(2)
    rouge_placeholder = col1.empty()
    bleu_placeholder = col2.empty()
    vert_placeholder = col1.empty()
    jaune_placeholder = col2.empty()
    
    st.markdown("---")
    
    # Par taille
    st.subheader("📏 Par taille")
    p_placeholder = st.empty()
    m_placeholder = st.empty()
    g_placeholder = st.empty()
    tg_placeholder = st.empty()
    
    st.markdown("---")
    
    if st.button("🔄 Réinitialiser", use_container_width=True):
        compteur.reset_compteur()
        st.rerun()

# Zone principale - Flux vidéo
st.subheader("📹 Flux en direct")

# Processeur vidéo pour WebRTC (version corrigée)
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.compteur_local = compteur  # Utiliser l'instance globale
    
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Traiter la frame
        resultat = self.compteur_local.traiter_frame(img)
        
        return av.VideoFrame.from_ndarray(resultat, format="bgr24")

# Lancer le flux vidéo
ctx = webrtc_streamer(
    key="object-detection",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# Mise à jour des stats (exécuté à chaque rafraîchissement)
if compteur.total_pieces > 0 or True:  # Toujours mettre à jour
    total_placeholder.metric("Pièces détectées", compteur.total_pieces)
    
    # Mise à jour des couleurs
    rouge_placeholder.metric("🔴 Rouge", compteur.stats_couleur.get('rouge', 0))
    bleu_placeholder.metric("🔵 Bleu", compteur.stats_couleur.get('bleu', 0))
    vert_placeholder.metric("🟢 Vert", compteur.stats_couleur.get('vert', 0))
    jaune_placeholder.metric("🟡 Jaune", compteur.stats_couleur.get('jaune', 0))
    
    # Mise à jour des tailles
    p_placeholder.metric("P", compteur.stats_taille.get('P', 0))
    m_placeholder.metric("M", compteur.stats_taille.get('M', 0))
    g_placeholder.metric("G", compteur.stats_taille.get('G', 0))
    tg_placeholder.metric("TG", compteur.stats_taille.get('TG', 0))

# Instructions
if not ctx.state.playing:
    st.info("""
    👆 **Cliquez sur 'START' pour activer la caméra**
    
    Puis pointez vers des pièces colorées !
    """)
else:
    st.success("✅ Caméra active - Détection en cours...")

# Pied de page
st.markdown("---")
st.caption("""
🎥 Détection en temps réel - Pointez et comptez !
• Rouge, Bleu, Vert, Jaune
• Classification par taille (P, M, G, TG)
""")
