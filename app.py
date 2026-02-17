import streamlit as st
import cv2
import numpy as np
from collections import defaultdict
from PIL import Image
import time
import base64
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av

# Configuration de la page
st.set_page_config(
    page_title="Compteur de Pièces - Live",
    page_icon="🎥",
    layout="wide"
)

# Vérifier si streamlit-webrtc est installé
try:
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
    import av
    webrtc_available = True
except ImportError:
    webrtc_available = False
    st.error("""
    ❌ streamlit-webrtc n'est pas installé.
    
    Pour installer : 
    ```bash
    pip install streamlit-webrtc av
    ```
    
    Ou ajoutez au requirements.txt :
    ```
    streamlit-webrtc
    av
    ```
    """)

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
if 'stats_couleur' not in st.session_state:
    st.session_state.stats_couleur = defaultdict(int)
if 'stats_taille' not in st.session_state:
    st.session_state.stats_taille = defaultdict(int)
if 'total_actuel' not in st.session_state:
    st.session_state.total_actuel = 0

# Interface Streamlit
st.title("🎥 Compteur de Pièces - Temps Réel")
st.markdown("""
Cette application détecte et compte automatiquement les pièces **en direct** :
- **Flux vidéo continu** comme dans Visual Studio
- **Détection par couleur** (rouge, bleu, vert, jaune)
- **Classification par taille** (P, M, G, TG)
- **Fonctionne dans votre navigateur** sans installation
""")

# Sidebar pour les paramètres
with st.sidebar:
    st.header("⚙️ Configuration")
    
    st.markdown("---")
    st.header("📊 Statistiques en direct")
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.metric("Pièces détectées", st.session_state.total_actuel)
    with col_s2:
        st.metric("Frames", st.session_state.frame_count)
    
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
    
    # Option 1: Avec streamlit-webrtc (meilleure performance)
    if webrtc_available:
        class VideoProcessor(VideoProcessorBase):
            def __init__(self):
                self.frame_count = 0
            
            def recv(self, frame):
                img = frame.to_ndarray(format="bgr24")
                
                # Traitement
                resultat, pieces, stats_couleur, stats_taille, total_actuel = st.session_state.compteur.traiter_frame(img)
                
                # Mise à jour des stats dans la session
                st.session_state.frame_count += 1
                st.session_state.stats_couleur = stats_couleur
                st.session_state.stats_taille = stats_taille
                st.session_state.total_actuel = total_actuel
                
                # Ajout d'information sur l'image
                h, w = resultat.shape[:2]
                cv2.putText(resultat, f"TOTAL: {total_actuel}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(resultat, f"Frame: {st.session_state.frame_count}", (w-150, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                return av.VideoFrame.from_ndarray(resultat, format="bgr24")
        
        ctx = webrtc_streamer(
            key="object-detection",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=VideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        
        if not ctx.state.playing:
            st.info("👆 Cliquez sur 'START' pour activer la caméra")
    
    else:
        st.warning("""
        ⚠️ streamlit-webrtc n'est pas installé.
        
        Pour le flux vidéo en direct, installez :
        ```bash
        pip install streamlit-webrtc av
        ```
        
        En attendant, utilisez le mode photo ci-dessous :
        """)
        
        # Fallback sur st.camera_input
        img_file = st.camera_input("Prenez une photo")
        if img_file:
            bytes_data = img_file.getvalue()
            frame = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            resultat, pieces, stats_couleur, stats_taille, total_actuel = st.session_state.compteur.traiter_frame(frame)
            st.session_state.total_actuel = total_actuel
            st.session_state.stats_couleur = stats_couleur
            st.session_state.stats_taille = stats_taille
            st.image(cv2.cvtColor(resultat, cv2.COLOR_BGR2RGB))

with col2:
    st.subheader("📊 Analyse en direct")
    
    # Créer des placeholders pour les stats dynamiques
    stats_container = st.container()
    
    with stats_container:
        # Métrique principale
        st.metric("Pièces dans l'image", st.session_state.total_actuel)
        
        st.markdown("---")
        
        # Stats par couleur
        st.write("**🎨 Par couleur :**")
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.write(f"🔴 Rouge: {st.session_state.stats_couleur.get('rouge', 0)}")
            st.write(f"🔵 Bleu: {st.session_state.stats_couleur.get('bleu', 0)}")
            st.write(f"🟢 Vert: {st.session_state.stats_couleur.get('vert', 0)}")
        with col_c2:
            st.write(f"🟡 Jaune: {st.session_state.stats_couleur.get('jaune', 0)}")
            st.write(f"⚪ Autre: {st.session_state.stats_couleur.get('?', 0)}")
        
        st.markdown("---")
        
        # Stats par taille
        st.write("**📏 Par taille :**")
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.write(f"P: {st.session_state.stats_taille.get('P', 0)}")
            st.write(f"M: {st.session_state.stats_taille.get('M', 0)}")
        with col_t2:
            st.write(f"G: {st.session_state.stats_taille.get('G', 0)}")
            st.write(f"TG: {st.session_state.stats_taille.get('TG', 0)}")
        
        st.markdown("---")
        
        # Informations
        st.info(f"🔄 Frames traitées: {st.session_state.frame_count}")
        
        if st.session_state.frame_count > 0:
            fps = st.session_state.frame_count / max(1, (time.time() - st.session_state.get('start_time', time.time())))
            st.write(f"⚡ FPS: {fps:.1f}")

# Initialiser le temps de démarrage
if 'start_time' not in st.session_state:
    st.session_state.start_time = time.time()

# Pied de page
st.markdown("---")
st.caption("""
🎥 Compteur de Pièces v3.0 - Flux vidéo en direct
• Utilise WebRTC pour la capture vidéo temps réel
• Compatible Streamlit Cloud
• Même expérience que Visual Studio
""")
