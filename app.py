import streamlit as st
import cv2
import numpy as np
from collections import defaultdict
from PIL import Image
import time
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av

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
        self.compteur = st.session_state.compteur
    
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        resultat, _, _, _, _ = self.compteur.traiter_frame(img)
        return av.VideoFrame.from_ndarray(resultat, format="bgr24")

# Fonction OneDrive simplifiée - sans dépendances supplémentaires
def interface_onedrive():
    """Interface simple pour importer depuis OneDrive"""
    
    st.subheader("☁️ Importer depuis OneDrive")
    
    st.markdown("""
    ### 📋 3 méthodes simples :
    
    **Méthode 1 : Télécharger depuis OneDrive.com**  
    → Le plus simple et fiable à 100%
    """)
    
    # Méthode 1: Upload après téléchargement manuel
    with st.expander("📤 Méthode 1: Télécharger depuis OneDrive.com", expanded=True):
        st.markdown("""
        **Étapes :**
        1. Allez sur [**OneDrive.com**](https://onedrive.com) dans votre navigateur
        2. Connectez-vous avec votre compte Microsoft
        3. Trouvez votre photo
        4. Cliquez sur la photo pour l'ouvrir
        5. Cliquez sur **Télécharger** (icône ⬇️ en haut)
        6. Enregistrez la photo sur votre PC
        7. Glissez-déposez la photo ci-dessous :
        """)
        
        uploaded_file = st.file_uploader(
            "Choisir l'image téléchargée de OneDrive",
            type=['jpg', 'jpeg', 'png', 'bmp', 'gif'],
            key="onedrive_upload_simple",
            help="Sélectionnez l'image que vous avez téléchargée"
        )
        
        if uploaded_file:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            st.session_state.onedrive_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            st.success(f"✅ Image chargée: {uploaded_file.name}")
    
    # Méthode 2: Dossier OneDrive synchronisé
    with st.expander("📁 Méthode 2: Dossier OneDrive synchronisé"):
        st.markdown("""
        **Si vous avez l'application OneDrive installée sur votre PC :**
        1. Ouvrez votre **dossier OneDrive** dans l'explorateur Windows
        2. Trouvez votre photo
        3. Glissez-déposez la photo directement ici :
        """)
        
        uploaded_file2 = st.file_uploader(
            "Glisser depuis le dossier OneDrive",
            type=['jpg', 'jpeg', 'png', 'bmp', 'gif'],
            key="onedrive_folder_simple",
            help="Ouvrez votre dossier OneDrive et glissez l'image ici"
        )
        
        if uploaded_file2:
            file_bytes = np.asarray(bytearray(uploaded_file2.read()), dtype=np.uint8)
            st.session_state.onedrive_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            st.success(f"✅ Image chargée depuis le dossier: {uploaded_file2.name}")
    
    # Méthode 3: Depuis un téléphone
    with st.expander("📱 Méthode 3: Depuis un téléphone"):
        st.markdown("""
        **Sur votre téléphone :**
        1. Ouvrez l'application **OneDrive**
        2. Trouvez votre photo
        3. Appuyez sur les **3 points** (⋯) à côté de la photo
        4. Sélectionnez **Exporter** → **Enregistrer sur l'appareil**
        5. Envoyez-vous la photo par email ou messagerie
        6. Téléchargez-la sur ce PC
        7. Utilisez la Méthode 1 ou 2
        """)
    
    # Afficher l'image chargée
    if st.session_state.onedrive_image is not None:
        st.markdown("---")
        st.subheader("📸 Image chargée de OneDrive")
        
        # Aperçu
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(cv2.cvtColor(st.session_state.onedrive_image, cv2.COLOR_BGR2RGB), 
                    caption="Aperçu", width=200)
        
        with col2:
            st.write("**Image prête à être analysée !**")
            
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if st.button("🔍 Analyser maintenant", use_container_width=True):
                    with st.spinner("🔍 Analyse en cours..."):
                        frame = st.session_state.onedrive_image
                        resultat, pieces, stats_couleur, stats_taille, total_actuel = compteur.traiter_frame(frame)
                        st.session_state.frame_count += 1
                        
                        st.success(f"✅ **{total_actuel} pièces** détectées !")
                        
                        col_img1, col_img2 = st.columns(2)
                        with col_img1:
                            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 
                                    caption="Originale (OneDrive)", use_column_width=True)
                        with col_img2:
                            st.image(cv2.cvtColor(resultat, cv2.COLOR_BGR2RGB), 
                                    caption=f"Analysée - {total_actuel} pièces", use_column_width=True)
                        
                        st.write("**Résultats :**")
                        st.write(f"- Couleurs: {dict(stats_couleur)}")
                        st.write(f"- Tailles: {dict(stats_taille)}")
            
            with col_btn2:
                if st.button("🗑️ Effacer", use_container_width=True):
                    st.session_state.onedrive_image = None
                    st.rerun()

# Initialisation du compteur dans la session
if 'compteur' not in st.session_state:
    st.session_state.compteur = CompteurPieces()
if 'frame_count' not in st.session_state:
    st.session_state.frame_count = 0
if 'mode' not in st.session_state:
    st.session_state.mode = None
if 'onedrive_image' not in st.session_state:
    st.session_state.onedrive_image = None

compteur = st.session_state.compteur

# Interface Streamlit
st.title("🧩 Compteur de Pièces - Interface Adaptative")
st.markdown("""
Cette application détecte et compte automatiquement les pièces :
- **Détection par couleur** (rouge, bleu, vert, jaune)
- **Classification par taille** (P, M, G, TG)
- **Import OneDrive simplifié** (téléchargez d'abord l'image)
""")

# Interface principale
st.info("📱💻 Interface adaptative - Utilisez les options ci-dessous")

# Tabs pour organiser l'interface
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📸 Prendre photo", 
    "🎥 Flux direct", 
    "🖼️ Upload image", 
    "☁️ OneDrive", 
    "🧪 Mode démo"
])

with tab1:
    st.subheader("📸 Prendre une photo avec la caméra")
    
    img_file = st.camera_input("Prenez une photo", key="main_camera")
    
    if img_file is not None:
        with st.spinner("🔍 Analyse..."):
            bytes_data = img_file.getvalue()
            frame = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            
            resultat, pieces, stats_couleur, stats_taille, total_actuel = compteur.traiter_frame(frame)
            st.session_state.frame_count += 1
            
            st.success(f"✅ **{total_actuel} pièces** détectées !")
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 
                        caption="Photo originale", use_column_width=True)
            with col2:
                st.image(cv2.cvtColor(resultat, cv2.COLOR_BGR2RGB), 
                        caption=f"Résultat - {total_actuel} pièces", use_column_width=True)
            
            st.metric("Total", total_actuel)
            st.write("**Couleurs:**", dict(stats_couleur))
            st.write("**Tailles:**", dict(stats_taille))

with tab2:
    st.subheader("🎥 Flux vidéo en direct")
    
    ctx = webrtc_streamer(
        key="object-detection-live",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    
    if not ctx.state.playing:
        st.info("👆 **Cliquez sur 'START' pour activer la caméra**")
    else:
        st.success("✅ Caméra active - Détection en temps réel")

with tab3:
    st.subheader("🖼️ Analyser une image")
    
    uploaded_file = st.file_uploader(
        "Choisissez une image", 
        type=['jpg', 'jpeg', 'png', 'bmp'],
        key="upload_tab"
    )
    
    if uploaded_file:
        with st.spinner("🔍 Analyse..."):
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            resultat, pieces, stats_couleur, stats_taille, total_actuel = compteur.traiter_frame(frame)
            st.session_state.frame_count += 1
            
            st.success(f"✅ **{total_actuel} pièces** détectées !")
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 
                        caption="Image originale", use_column_width=True)
            with col2:
                st.image(cv2.cvtColor(resultat, cv2.COLOR_BGR2RGB), 
                        caption=f"Résultat - {total_actuel} pièces", use_column_width=True)
            
            st.write("**Résultats détaillés :**")
            st.write(f"- Par couleur: {dict(stats_couleur)}")
            st.write(f"- Par taille: {dict(stats_taille)}")

with tab4:
    # Interface OneDrive simplifiée
    interface_onedrive()

with tab5:
    st.subheader("🧪 Mode démo - Image de test")
    
    if st.button("🎲 Générer une image de test", use_container_width=True):
        with st.spinner("Génération et analyse..."):
            # Créer une image de test
            test_img = np.zeros((480, 640, 3), dtype=np.uint8)
            test_img.fill(255)
            
            # Dessiner des cercles colorés
            cv2.circle(test_img, (200, 200), 50, (0, 0, 255), -1)   # Rouge
            cv2.circle(test_img, (350, 250), 40, (255, 0, 0), -1)   # Bleu
            cv2.circle(test_img, (500, 200), 45, (0, 255, 0), -1)   # Vert
            cv2.circle(test_img, (300, 350), 35, (0, 255, 255), -1) # Jaune
            cv2.circle(test_img, (450, 350), 60, (100, 100, 100), -1) # Gris
            
            resultat, pieces, stats_couleur, stats_taille, total_actuel = compteur.traiter_frame(test_img)
            
            st.success(f"✅ **{total_actuel} pièces** détectées en mode démo !")
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB), 
                        caption="🧪 Image de test", use_column_width=True)
            with col2:
                st.image(cv2.cvtColor(resultat, cv2.COLOR_BGR2RGB), 
                        caption=f"Résultat", use_column_width=True)
            
            st.write("**Résultats :**")
            st.write(f"- Couleurs: {dict(stats_couleur)}")
            st.write(f"- Tailles: {dict(stats_taille)}")

# Bouton de réinitialisation global
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col2:
    if st.button("🔄 Réinitialiser tous les compteurs", use_container_width=True):
        compteur.reset_compteur()
        st.session_state.frame_count = 0
        st.session_state.onedrive_image = None
        st.success("✅ Compteurs réinitialisés !")
        st.rerun()

# Pied de page
st.markdown("---")
st.caption("""
🧩 Compteur de Pièces v3.5 - Sans dépendances inutiles  
• Interface simple avec onglets  
• Pour OneDrive : téléchargez d'abord l'image sur votre PC  
• 100% fonctionnel sans packages supplémentaires
""")
