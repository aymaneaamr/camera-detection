import streamlit as st
import cv2
import numpy as np
from collections import defaultdict
from PIL import Image
import time
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
import os
import tempfile
from pathlib import Path
import requests
from io import BytesIO

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
        self.stats_couleur = defaultdict(int)
        self.stats_taille = defaultdict(int)
        self.total_pieces = 0
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
            cv2.putText(resultat, f"{couleur_nom[0]}{taille_nom}", (x+5, y+20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        total_actuel = len(pieces_actuelles)
        
        # Mise à jour des stats
        self.stats_couleur = stats_couleur_actuelles
        self.stats_taille = stats_taille_actuelles
        self.total_pieces = total_actuel
        
        # Ajouter le compteur total sur l'image
        h, w = resultat.shape[:2]
        cv2.putText(resultat, f"Total: {total_actuel}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        return resultat, pieces_actuelles, stats_couleur_actuelles, stats_taille_actuelles, total_actuel

# Classe pour le traitement vidéo en temps réel
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.compteur = compteur
    
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        resultat, _, _, _, _ = self.compteur.traiter_frame(img)
        return av.VideoFrame.from_ndarray(resultat, format="bgr24")

# Fonction pour importer depuis OneDrive
def importer_depuis_onedrive():
    """Interface pour importer des photos depuis OneDrive"""
    st.subheader("☁️ Importer depuis OneDrive")
    
    # Option 1: Lien de partage OneDrive
    with st.expander("🔗 Importer par lien de partage", expanded=True):
        st.markdown("""
        1. Allez dans OneDrive
        2. Cliquez droit sur l'image → **Partager**
        3. Copiez le lien de partage
        4. Collez-le ci-dessous
        """)
        
        onedrive_url = st.text_input("Lien de partage OneDrive:", placeholder="https://1drv.ms/i/s!...")
        
        if onedrive_url:
            if st.button("📥 Importer depuis ce lien", use_container_width=True):
                with st.spinner("Téléchargement en cours..."):
                    try:
                        # Convertir le lien de partage en lien de téléchargement direct
                        if "1drv.ms" in onedrive_url:
                            # Pour les liens courts OneDrive
                            response = requests.get(onedrive_url, allow_redirects=True)
                            final_url = response.url
                            # Extraire l'ID et créer le lien de téléchargement
                            if "redir?" in final_url:
                                import re
                                file_id = re.search(r'[0-9A-Fa-f]{8,}', final_url)
                                if file_id:
                                    direct_url = f"https://api.onedrive.com/v1.0/shares/u!{file_id}/root/content"
                                else:
                                    direct_url = final_url.replace("redir", "download")
                            else:
                                direct_url = final_url.replace("/view", "/download")
                        else:
                            direct_url = onedrive_url
                        
                        # Télécharger l'image
                        headers = {
                            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                        }
                        img_response = requests.get(direct_url, headers=headers, timeout=30)
                        
                        if img_response.status_code == 200:
                            img_data = BytesIO(img_response.content)
                            pil_image = Image.open(img_data)
                            # Convertir en format OpenCV
                            open_cv_image = np.array(pil_image)
                            if len(open_cv_image.shape) == 3:
                                open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
                            
                            st.session_state.onedrive_image = open_cv_image
                            st.session_state.onedrive_image_loaded = True
                            st.success("✅ Image importée avec succès!")
                            st.rerun()
                        else:
                            st.error(f"❌ Erreur de téléchargement: {img_response.status_code}")
                    except Exception as e:
                        st.error(f"❌ Erreur: {str(e)}")
    
    # Option 2: Upload direct depuis OneDrive (via fichier local)
    with st.expander("📁 Ouvrir depuis OneDrive synchro"):
        st.markdown("""
        Si vous avez OneDrive synchronisé sur votre PC, vous pouvez aussi :
        1. Ouvrir votre dossier OneDrive dans l'explorateur
        2. Glisser-déposer les photos directement ci-dessous
        """)
        
        uploaded_file = st.file_uploader(
            "Glissez vos photos OneDrive ici",
            type=['jpg', 'jpeg', 'png', 'gif', 'bmp'],
            key="onedrive_upload"
        )
        
        if uploaded_file:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            st.session_state.onedrive_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            st.session_state.onedrive_image_loaded = True
            st.success(f"✅ Image chargée: {uploaded_file.name}")
            st.rerun()
    
    # Option 3: Guide d'export OneDrive
    with st.expander("📱 Depuis l'application mobile OneDrive"):
        st.markdown("""
        **Depuis votre téléphone :**
        1. Ouvrez l'application OneDrive
        2. Trouvez votre photo
        3. Tapez sur les **3 points** → **Exporter** → **Enregistrer sur l'appareil**
        4. Revenez sur ce PC et utilisez l'option **Uploader une image**
        """)

# Initialisation du compteur dans la session
if 'compteur' not in st.session_state:
    st.session_state.compteur = CompteurPieces()
if 'frame_count' not in st.session_state:
    st.session_state.frame_count = 0
if 'mode' not in st.session_state:
    st.session_state.mode = None
if 'onedrive_image' not in st.session_state:
    st.session_state.onedrive_image = None
if 'onedrive_image_loaded' not in st.session_state:
    st.session_state.onedrive_image_loaded = False

compteur = st.session_state.compteur

# Interface Streamlit
st.title("🧩 Compteur de Pièces - Interface Adaptative")
st.markdown("""
Cette application détecte et compte automatiquement les pièces :
- **Détection par couleur** (rouge, bleu, vert, jaune)
- **Classification par taille** (P, M, G, TG)
- **S'adapte automatiquement à votre appareil**
""")

# Détection du type d'appareil
user_agent = st.query_params.get("user_agent", [""])[0] if hasattr(st, 'query_params') else ""
is_mobile = any(x in user_agent.lower() for x in ['android', 'iphone', 'mobile']) if user_agent else None

# Si on ne peut pas détecter automatiquement, demander à l'utilisateur
if is_mobile is None:
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📱 Je suis sur téléphone", use_container_width=True):
            st.session_state.mode = "mobile"
            st.rerun()
    with col2:
        if st.button("💻 Je suis sur PC", use_container_width=True):
            st.session_state.mode = "pc"
            st.rerun()
else:
    st.session_state.mode = "mobile" if is_mobile else "pc"

# Interface selon le mode détecté
if st.session_state.mode == "mobile":
    # ========== INTERFACE MOBILE (TÉLÉPHONE) ==========
    st.info("📱 Mode téléphone détecté - Interface optimisée pour mobile")
    
    # Interface simplifiée pour mobile
    with st.container():
        st.subheader("📸 Prendre une photo")
        
        # Affichage compact
        col1, col2 = st.columns([1, 1])
        with col1:
            source = st.radio(
                "Source",
                ["📸 Caméra", "🖼️ Galerie", "🧪 Démo"],
                label_visibility="collapsed"
            )
        
        if source == "📸 Caméra":
            img_file = st.camera_input("Prendre une photo", key="mobile_camera")
            
            if img_file is not None:
                with st.spinner("🔍 Analyse..."):
                    bytes_data = img_file.getvalue()
                    frame = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
                    
                    resultat, pieces, stats_couleur, stats_taille, total_actuel = compteur.traiter_frame(frame)
                    
                    st.success(f"✅ **{total_actuel} pièces**")
                    
                    # Affichage compact
                    st.image(cv2.cvtColor(resultat, cv2.COLOR_BGR2RGB), use_column_width=True)
                    
                    # Stats en lignes
                    st.write("**Couleurs:** " + ", ".join([f"{c}:{stats_couleur.get(c,0)}" for c in ['rouge','bleu','vert','jaune'] if stats_couleur.get(c,0)>0]))
                    st.write("**Tailles:** " + ", ".join([f"{t}:{stats_taille.get(t,0)}" for t in ['P','M','G','TG'] if stats_taille.get(t,0)>0]))
        
        elif source == "🖼️ Galerie":
            uploaded_file = st.file_uploader("Choisir image", type=['jpg', 'jpeg', 'png'], label_visibility="collapsed")
            
            if uploaded_file:
                with st.spinner("🔍 Analyse..."):
                    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    
                    resultat, pieces, stats_couleur, stats_taille, total_actuel = compteur.traiter_frame(frame)
                    
                    st.success(f"✅ **{total_actuel} pièces**")
                    st.image(cv2.cvtColor(resultat, cv2.COLOR_BGR2RGB), use_column_width=True)
                    
                    # Stats compactes
                    with st.expander("📊 Détails"):
                        st.write("**Par couleur:**", dict(stats_couleur))
                        st.write("**Par taille:**", dict(stats_taille))
        
        else:  # Mode démo
            if st.button("🎲 Générer test", use_container_width=True):
                with st.spinner("..."):
                    test_img = np.zeros((480, 640, 3), dtype=np.uint8)
                    test_img.fill(255)
                    
                    cv2.circle(test_img, (200, 200), 50, (0, 0, 255), -1)
                    cv2.circle(test_img, (350, 250), 40, (255, 0, 0), -1)
                    cv2.circle(test_img, (500, 200), 45, (0, 255, 0), -1)
                    cv2.circle(test_img, (300, 350), 35, (0, 255, 255), -1)
                    
                    resultat, pieces, stats_couleur, stats_taille, total_actuel = compteur.traiter_frame(test_img)
                    
                    st.success(f"✅ **{total_actuel} pièces**")
                    st.image(cv2.cvtColor(resultat, cv2.COLOR_BGR2RGB), use_column_width=True)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Rouge", stats_couleur.get('rouge',0))
                        st.metric("Bleu", stats_couleur.get('bleu',0))
                    with col2:
                        st.metric("Vert", stats_couleur.get('vert',0))
                        st.metric("Jaune", stats_couleur.get('jaune',0))

else:
    # ========== INTERFACE PC (ORDINATEUR) ==========
    st.info("💻 Mode PC détecté - Interface complète avec OneDrive")
    
    # Sidebar pour les paramètres
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        source = st.radio(
            "Source",
            ["📸 Prendre une photo", "🎥 Flux en direct", "🖼️ Uploader une image", "☁️ OneDrive", "🧪 Mode démo"]
        )
        
        st.markdown("---")
        st.header("📊 Statistiques")
        
        if st.button("🔄 Réinitialiser compteurs", use_container_width=True):
            compteur.reset_compteur()
            st.session_state.frame_count = 0
            st.session_state.onedrive_image = None
            st.session_state.onedrive_image_loaded = False
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
    
    # Zone principale PC
    if source == "📸 Prendre une photo":
        st.subheader("📸 Prenez une photo")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            img_file = st.camera_input("Cliquez pour prendre une photo", key="pc_camera")
        
        if img_file is not None:
            with st.spinner("🔍 Analyse en cours..."):
                bytes_data = img_file.getvalue()
                frame = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
                
                resultat, pieces, stats_couleur, stats_taille, total_actuel = compteur.traiter_frame(frame)
                st.session_state.frame_count += 1
                
                st.success(f"✅ **{total_actuel} pièces** détectées !")
                
                col_img1, col_img2 = st.columns(2)
                with col_img1:
                    st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 
                            caption="📸 Photo originale", use_column_width=True)
                with col_img2:
                    st.image(cv2.cvtColor(resultat, cv2.COLOR_BGR2RGB), 
                            caption=f"🎯 {total_actuel} pièces détectées", use_column_width=True)
                
                # Statistiques détaillées
                st.subheader("📊 Détail par couleur et taille")
                
                col_m1, col_m2, col_m3 = st.columns(3)
                with col_m1:
                    st.metric("Total pièces", total_actuel)
                with col_m2:
                    st.metric("Couleurs différentes", len([c for c in stats_couleur.values() if c > 0]))
                with col_m3:
                    st.metric("Frame", st.session_state.frame_count)
                
                # Tableau des couleurs
                st.write("**🎨 Répartition par couleur :**")
                cols = st.columns(5)
                couleurs_list = ['rouge', 'bleu', 'vert', 'jaune', 'autre']
                color_emoji = {'rouge': '🔴', 'bleu': '🔵', 'vert': '🟢', 'jaune': '🟡', 'autre': '⚪'}
                
                for i, couleur in enumerate(couleurs_list):
                    with cols[i]:
                        count = stats_couleur.get(couleur if couleur != 'autre' else '?', 0)
                        st.metric(f"{color_emoji[couleur]} {couleur}", count)
                
                # Tableau des tailles
                st.write("**📏 Répartition par taille :**")
                cols = st.columns(4)
                tailles_list = ['P', 'M', 'G', 'TG']
                for i, taille in enumerate(tailles_list):
                    with cols[i]:
                        count = stats_taille.get(taille, 0)
                        st.metric(f"Taille {taille}", count)
                
                # Liste détaillée des pièces
                with st.expander("🔍 Voir le détail de chaque pièce"):
                    for i, piece in enumerate(pieces, 1):
                        st.write(f"Pièce #{i} : {piece['couleur']} - {piece['taille']} (aire: {piece['aire']:.0f} px)")
    
    elif source == "🎥 Flux en direct":
        st.subheader("🎥 Flux vidéo en temps réel")
        
        # Stats en direct dans la sidebar
        with st.sidebar:
            st.metric("Pièces actuellement", compteur.total_pieces)
            st.write("**Couleurs:**")
            for c in ['rouge', 'bleu', 'vert', 'jaune']:
                if compteur.stats_couleur.get(c, 0) > 0:
                    st.write(f"- {c}: {compteur.stats_couleur.get(c, 0)}")
        
        # Lancer le flux vidéo
        ctx = webrtc_streamer(
            key="object-detection-pc",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=VideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        
        if not ctx.state.playing:
            st.info("👆 **Cliquez sur 'START' pour activer la caméra**")
    
    elif source == "🖼️ Uploader une image":
        st.subheader("🖼️ Analyse d'image")
        
        uploaded_file = st.file_uploader("Choisissez une image", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file:
            with st.spinner("🔍 Analyse en cours..."):
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                
                resultat, pieces, stats_couleur, stats_taille, total_actuel = compteur.traiter_frame(frame)
                st.session_state.frame_count += 1
                
                st.success(f"✅ **{total_actuel} pièces** détectées !")
                
                col_img1, col_img2 = st.columns(2)
                with col_img1:
                    st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 
                            caption="🖼️ Image originale", use_column_width=True)
                with col_img2:
                    st.image(cv2.cvtColor(resultat, cv2.COLOR_BGR2RGB), 
                            caption=f"🎯 {total_actuel} pièces détectées", use_column_width=True)
                
                st.subheader("📊 Résultats")
                col_s1, col_s2 = st.columns(2)
                with col_s1:
                    st.write("**Par couleur :**")
                    for couleur in ['rouge', 'bleu', 'vert', 'jaune', '?']:
                        count = stats_couleur.get(couleur, 0)
                        if count > 0:
                            st.write(f"- {couleur}: {count}")
                with col_s2:
                    st.write("**Par taille :**")
                    for taille in ['P', 'M', 'G', 'TG']:
                        count = stats_taille.get(taille, 0)
                        if count > 0:
                            st.write(f"- {taille}: {count}")
    
    elif source == "☁️ OneDrive":
        # Interface OneDrive
        importer_depuis_onedrive()
        
        # Si une image OneDrive a été chargée
        if st.session_state.onedrive_image_loaded and st.session_state.onedrive_image is not None:
            st.markdown("---")
            st.subheader("📸 Image importée de OneDrive")
            
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("🔍 Analyser cette image", use_container_width=True):
                    with st.spinner("🔍 Analyse en cours..."):
                        frame = st.session_state.onedrive_image
                        resultat, pieces, stats_couleur, stats_taille, total_actuel = compteur.traiter_frame(frame)
                        st.session_state.frame_count += 1
                        
                        st.success(f"✅ **{total_actuel} pièces** détectées !")
                        
                        col_img1, col_img2 = st.columns(2)
                        with col_img1:
                            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 
                                    caption="☁️ Image OneDrive", use_column_width=True)
                        with col_img2:
                            st.image(cv2.cvtColor(resultat, cv2.COLOR_BGR2RGB), 
                                    caption=f"🎯 {total_actuel} pièces détectées", use_column_width=True)
                        
                        # Résultats
                        st.subheader("📊 Résultats")
                        col_r1, col_r2 = st.columns(2)
                        with col_r1:
                            st.write("**Couleurs:**", dict(stats_couleur))
                        with col_r2:
                            st.write("**Tailles:**", dict(stats_taille))
            
            with col2:
                if st.button("🗑️ Effacer l'image", use_container_width=True):
                    st.session_state.onedrive_image = None
                    st.session_state.onedrive_image_loaded = False
                    st.rerun()
    
    else:  # Mode démo
        st.subheader("🧪 Mode démo")
        
        if st.button("🎲 Générer une image de test"):
            with st.spinner("🔍 Analyse..."):
                test_img = np.zeros((480, 640, 3), dtype=np.uint8)
                test_img.fill(255)
                
                cv2.circle(test_img, (200, 200), 50, (0, 0, 255), -1)
                cv2.circle(test_img, (350, 250), 40, (255, 0, 0), -1)
                cv2.circle(test_img, (500, 200), 45, (0, 255, 0), -1)
                cv2.circle(test_img, (300, 350), 35, (0, 255, 255), -1)
                cv2.circle(test_img, (450, 350), 60, (100, 100, 100), -1)
                
                resultat, pieces, stats_couleur, stats_taille, total_actuel = compteur.traiter_frame(test_img)
                st.session_state.frame_count += 1
                
                st.success(f"✅ **{total_actuel} pièces** détectées en mode démo !")
                
                col_img1, col_img2 = st.columns(2)
                with col_img1:
                    st.image(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB), 
                            caption="🧪 Image de test", use_column_width=True)
                with col_img2:
                    st.image(cv2.cvtColor(resultat, cv2.COLOR_BGR2RGB), 
                            caption=f"🎯 {total_actuel} pièces détectées", use_column_width=True)
                
                col_d1, col_d2 = st.columns(2)
                with col_d1:
                    st.write("**Couleurs:**", dict(stats_couleur))
                with col_d2:
                    st.write("**Tailles:**", dict(stats_taille))

# Pied de page commun
st.markdown("---")
st.caption("""
🧩 Compteur de Pièces v3.1 - Interface Adaptative avec OneDrive
• S'adapte automatiquement à votre appareil (mobile/PC)
• Importez vos photos directement depuis OneDrive
• Interface optimisée pour chaque type d'écran
""")
