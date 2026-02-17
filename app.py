import streamlit as st
import cv2
import numpy as np
from collections import defaultdict
import time
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, ClientSettings
import av

# Configuration de la page
st.set_page_config(
    page_title="Compteur de Pièces - Live",
    page_icon="🎥",
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
        
        # Ajouter des informations sur l'image
        h, w = resultat.shape[:2]
        cv2.putText(resultat, f"TOTAL: {total_actuel}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        return resultat, pieces_actuelles, stats_couleur_actuelles, stats_taille_actuelles, total_actuel

# Initialisation
if 'compteur' not in st.session_state:
    st.session_state.compteur = CompteurPieces()
if 'frame_count' not in st.session_state:
    st.session_state.frame_count = 0
if 'stats_couleur' not in st.session_state:
    st.session_state.stats_couleur = defaultdict(int)
if 'stats_taille' not in st.session_state:
    st.session_state.stats_taille = defaultdict(int)
if 'total_actuel' not in st.session_state:
    st.session_state.total_actuel = 0

# Interface
st.title("🎥 Compteur de Pièces - Temps Réel")

with st.sidebar:
    st.header("⚙️ Configuration")
    
    if st.button("🔄 Réinitialiser compteurs"):
        st.session_state.compteur.reset_compteur()
        st.session_state.frame_count = 0
        st.session_state.stats_couleur = defaultdict(int)
        st.session_state.stats_taille = defaultdict(int)
        st.session_state.total_actuel = 0
        st.rerun()
    
    st.markdown("---")
    st.markdown("""
    ### 📝 Légende
    - 🔴 Rouge
    - 🔵 Bleu  
    - 🟢 Vert
    - 🟡 Jaune
    
    ### 📏 Tailles
    - **P** : < 500 px
    - **M** : 500-2000 px
    - **G** : 2000-5000 px
    - **TG** : > 5000 px
    """)

# Zone principale
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📹 Flux vidéo en direct")
    
    # Configuration client pour WebRTC
    client_settings = ClientSettings(
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
    )
    
    class VideoProcessor(VideoProcessorBase):
        def __init__(self):
            self.frame_count = 0
        
        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            
            # Vérifier que l'image n'est pas vide
            if img is None or img.size == 0:
                return frame
            
            # Traitement
            try:
                resultat, pieces, stats_couleur, stats_taille, total_actuel = st.session_state.compteur.traiter_frame(img)
                
                # Mise à jour des stats
                st.session_state.frame_count += 1
                st.session_state.stats_couleur = stats_couleur
                st.session_state.stats_taille = stats_taille
                st.session_state.total_actuel = total_actuel
                
                # Ajouter des infos
                h, w = resultat.shape[:2]
                cv2.putText(resultat, f"Frame: {st.session_state.frame_count}", (w-150, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                return av.VideoFrame.from_ndarray(resultat, format="bgr24")
            except Exception as e:
                # En cas d'erreur, retourner l'image originale
                return frame
    
    # Lancer WebRTC avec configuration explicite
    ctx = webrtc_streamer(
        key="object-detection-live",
        mode=WebRtcMode.SENDRECV,
        client_settings=client_settings,
        video_processor_factory=VideoProcessor,
        async_processing=True,
    )
    
    # Instructions
    if not ctx.state.playing:
        st.info("""
        👆 **Cliquez sur 'START' pour activer la caméra**
        
        Si l'écran reste noir :
        1. Vérifiez que vous avez autorisé l'accès à la caméra
        2. Rafraîchissez la page
        3. Essayez avec un autre navigateur (Chrome recommandé)
        """)
    else:
        st.success("✅ Caméra active - Détection en cours...")

with col2:
    st.subheader("📊 Analyse en direct")
    
    # Stats dynamiques
    stats_placeholder = st.empty()
    
    # Mise à jour des stats
    with stats_placeholder.container():
        st.metric("Pièces dans l'image", st.session_state.total_actuel)
        
        st.markdown("---")
        st.write("**🎨 Par couleur :**")
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.write(f"🔴 Rouge: {st.session_state.stats_couleur.get('rouge', 0)}")
            st.write(f"🔵 Bleu: {st.session_state.stats_couleur.get('bleu', 0)}")
        with col_c2:
            st.write(f"🟢 Vert: {st.session_state.stats_couleur.get('vert', 0)}")
            st.write(f"🟡 Jaune: {st.session_state.stats_couleur.get('jaune', 0)}")
        
        st.markdown("---")
        st.write("**📏 Par taille :**")
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.write(f"P: {st.session_state.stats_taille.get('P', 0)}")
            st.write(f"M: {st.session_state.stats_taille.get('M', 0)}")
        with col_t2:
            st.write(f"G: {st.session_state.stats_taille.get('G', 0)}")
            st.write(f"TG: {st.session_state.stats_taille.get('TG', 0)}")
        
        st.markdown("---")
        st.info(f"🔄 Frames: {st.session_state.frame_count}")

# Pied de page
st.markdown("---")
st.caption("""
🎥 Compteur de Pièces v3.1 - Flux vidéo en direct
• Utilise WebRTC avec configuration STUN
• Compatible Chrome, Firefox, Edge
""")
