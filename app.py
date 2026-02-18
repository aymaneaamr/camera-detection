import streamlit as st
import cv2
import numpy as np
from collections import defaultdict
from PIL import Image
import time
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
import requests
import json
import msal
import webbrowser
from urllib.parse import quote, urlencode
import base64

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

# Configuration OneDrive
CLIENT_ID = "YOUR_CLIENT_ID"  # À obtenir sur https://portal.azure.com
REDIRECT_URI = "http://localhost:8501"
SCOPE = ["Files.Read", "Files.Read.All", "offline_access"]

# Fonction pour l'authentification OneDrive
def authenticate_onedrive():
    """Authentifie l'utilisateur avec OneDrive"""
    
    # Créer une application MSAL
    app = msal.PublicClientApplication(
        client_id=CLIENT_ID,
        authority="https://login.microsoftonline.com/common"
    )
    
    # Vérifier si on a déjà un token
    accounts = app.get_accounts()
    if accounts:
        # Essayer de récupérer silencieusement un token
        result = app.acquire_token_silent(SCOPE, account=accounts[0])
        if result:
            return result
    
    # Sinon, faire l'authentification interactive
    # Note: En production, utilisez un flux serveur
    auth_url = app.get_authorization_request_url(
        SCOPE,
        redirect_uri=REDIRECT_URI
    )
    
    st.markdown(f"""
    ### 🔑 Connexion OneDrive requise
    
    [Cliquez ici pour vous connecter à OneDrive]({auth_url})
    
    Après la connexion, copiez l'URL complète de la page et collez-la ci-dessous :
    """)
    
    auth_response = st.text_input("URL de redirection après connexion :")
    
    if auth_response:
        # Extraire le code de l'URL
        from urllib.parse import urlparse, parse_qs
        parsed = urlparse(auth_response)
        params = parse_qs(parsed.query)
        
        if 'code' in params:
            code = params['code'][0]
            result = app.acquire_token_by_authorization_code(code, SCOPE, REDIRECT_URI)
            return result
    
    return None

# Fonction pour lister les photos OneDrive
def list_onedrive_photos(token):
    """Liste les photos dans OneDrive"""
    
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }
    
    # Chercher dans le dossier Photos
    response = requests.get(
        "https://graph.microsoft.com/v1.0/me/drive/special/photos/children",
        headers=headers
    )
    
    if response.status_code == 200:
        return response.json().get('value', [])
    
    # Sinon, chercher à la racine
    response = requests.get(
        "https://graph.microsoft.com/v1.0/me/drive/root/children",
        headers=headers
    )
    
    if response.status_code == 200:
        files = response.json().get('value', [])
        # Filtrer les images
        images = [f for f in files if f.get('file', {}).get('mimeType', '').startswith('image/')]
        return images
    
    return []

# Fonction pour télécharger une photo OneDrive
def download_onedrive_photo(token, item_id):
    """Télécharge une photo depuis OneDrive"""
    
    headers = {
        'Authorization': f'Bearer {token}'
    }
    
    # Obtenir l'URL de téléchargement
    response = requests.get(
        f"https://graph.microsoft.com/v1.0/me/drive/items/{item_id}/content",
        headers=headers,
        stream=True
    )
    
    if response.status_code == 200:
        # Convertir en image OpenCV
        img_data = response.content
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    
    return None

# Fonction principale OneDrive
def interface_onedrive():
    """Interface principale pour OneDrive"""
    
    st.subheader("☁️ Importer depuis OneDrive")
    
    # Étape 1: Configuration
    st.info("""
    ### 📋 Configuration requise
    
    Pour utiliser OneDrive, vous devez :
    1. Aller sur [Azure Portal](https://portal.azure.com)
    2. Créer une application
    3. Obtenir un Client ID
    """)
    
    # Option pour utiliser l'API publique (simplifiée)
    st.markdown("---")
    st.markdown("### 🔄 Méthode simple : Téléchargement manuel")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **📱 Depuis votre téléphone :**
        1. Ouvrez OneDrive
        2. Trouvez votre photo
        3. Tapez sur les **3 points** → **Exporter** → **Enregistrer sur l'appareil**
        4. Envoyez-vous la photo par email
        5. Téléchargez-la sur ce PC
        """)
    
    with col2:
        st.markdown("""
        **💻 Depuis votre PC :**
        1. Ouvrez OneDrive dans votre navigateur
        2. Trouvez votre photo
        3. Cliquez sur **Télécharger**
        4. Enregistrez la photo sur votre PC
        5. Utilisez l'upload ci-dessous
        """)
    
    st.markdown("---")
    st.markdown("### 📤 Uploader l'image téléchargée")
    
    uploaded_file = st.file_uploader(
        "Choisissez l'image que vous avez téléchargée de OneDrive",
        type=['jpg', 'jpeg', 'png', 'gif', 'bmp'],
        key="onedrive_final_upload"
    )
    
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        return img
    
    return None

# Initialisation du compteur dans la session
if 'compteur' not in st.session_state:
    st.session_state.compteur = CompteurPieces()
if 'frame_count' not in st.session_state:
    st.session_state.frame_count = 0
if 'mode' not in st.session_state:
    st.session_state.mode = None
if 'onedrive_image' not in st.session_state:
    st.session_state.onedrive_image = None
if 'onedrive_token' not in st.session_state:
    st.session_state.onedrive_token = None

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
    # ========== INTERFACE MOBILE ==========
    st.info("📱 Mode téléphone détecté - Interface optimisée pour mobile")
    
    with st.container():
        st.subheader("📸 Prendre une photo")
        
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
                    st.image(cv2.cvtColor(resultat, cv2.COLOR_BGR2RGB), use_column_width=True)
                    
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
    # ========== INTERFACE PC ==========
    st.info("💻 Mode PC détecté - Interface complète")
    
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
    if source == "📸 Prendre une photo":
        st.subheader("📸 Prenez une photo")
        
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
                
                st.subheader("📊 Détail par couleur et taille")
                
                col_m1, col_m2, col_m3 = st.columns(3)
                with col_m1:
                    st.metric("Total pièces", total_actuel)
                with col_m2:
                    st.metric("Couleurs différentes", len([c for c in stats_couleur.values() if c > 0]))
                with col_m3:
                    st.metric("Frame", st.session_state.frame_count)
    
    elif source == "🎥 Flux en direct":
        st.subheader("🎥 Flux vidéo en temps réel")
        
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
        # Interface OneDrive simplifiée mais fonctionnelle
        st.subheader("☁️ Importer depuis OneDrive")
        
        # Méthode 1: Upload direct (recommandée)
        with st.expander("📤 Méthode 1: Upload direct (RECOMMANDÉE)", expanded=True):
            st.success("✅ **La méthode la plus fiable**")
            st.markdown("""
            1. Allez sur [OneDrive.com](https://onedrive.com) dans votre navigateur
            2. Trouvez votre photo
            3. Cliquez sur **Télécharger** (⬇️)
            4. Enregistrez la photo sur votre PC
            5. Glissez-déposez-la ci-dessous
            """)
            
            uploaded_file = st.file_uploader(
                "Choisir l'image téléchargée de OneDrive",
                type=['jpg', 'jpeg', 'png', 'bmp'],
                key="onedrive_upload",
                help="Téléchargez d'abord l'image depuis OneDrive"
            )
            
            if uploaded_file:
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                st.session_state.onedrive_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                st.success(f"✅ Image chargée: {uploaded_file.name}")
        
        # Méthode 2: Extension Chrome (alternative)
        with st.expander("🌐 Méthode 2: Extension Chrome"):
            st.info("""
            **Utiliser l'extension "OneDrive Downloader" :**
            1. Installez l'extension depuis le Chrome Web Store
            2. Allez sur OneDrive, cliquez droit sur l'image
            3. Sélectionnez "Télécharger avec l'extension"
            4. Puis uploadez l'image ci-dessus
            """)
        
        # Méthode 3: Application de bureau
        with st.expander("💻 Méthode 3: Application OneDrive"):
            st.info("""
            **Si vous avez l'application OneDrive installée :**
            1. Ouvrez votre dossier OneDrive dans l'explorateur
            2. Trouvez la photo
            3. Glissez-déposez-la directement ici
            """)
            
            uploaded_file2 = st.file_uploader(
                "Ou glissez depuis votre dossier OneDrive",
                type=['jpg', 'jpeg', 'png', 'bmp'],
                key="onedrive_folder",
                help="Ouvrez votre dossier OneDrive et glissez l'image ici"
            )
            
            if uploaded_file2:
                file_bytes = np.asarray(bytearray(uploaded_file2.read()), dtype=np.uint8)
                st.session_state.onedrive_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                st.success(f"✅ Image chargée depuis le dossier: {uploaded_file2.name}")
        
        # Afficher l'image chargée
        if st.session_state.onedrive_image is not None:
            st.markdown("---")
            st.subheader("📸 Image chargée")
            
            # Aperçu
            st.image(cv2.cvtColor(st.session_state.onedrive_image, cv2.COLOR_BGR2RGB), 
                    caption="Aperçu", width=300)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔍 Analyser cette image", use_container_width=True):
                    with st.spinner("🔍 Analyse..."):
                        frame = st.session_state.onedrive_image
                        resultat, pieces, stats_couleur, stats_taille, total_actuel = compteur.traiter_frame(frame)
                        st.session_state.frame_count += 1
                        
                        st.success(f"✅ **{total_actuel} pièces** détectées !")
                        
                        col_img1, col_img2 = st.columns(2)
                        with col_img1:
                            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 
                                    caption="Originale", use_column_width=True)
                        with col_img2:
                            st.image(cv2.cvtColor(resultat, cv2.COLOR_BGR2RGB), 
                                    caption="Analysée", use_column_width=True)
                        
                        st.write("**Couleurs:**", dict(stats_couleur))
                        st.write("**Tailles:**", dict(stats_taille))
            
            with col2:
                if st.button("🗑️ Effacer", use_container_width=True):
                    st.session_state.onedrive_image = None
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
                
                st.success(f"✅ **{total_actuel} pièces** détectées !")
                
                col_img1, col_img2 = st.columns(2)
                with col_img1:
                    st.image(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB), 
                            caption="🧪 Image de test", use_column_width=True)
                with col_img2:
                    st.image(cv2.cvtColor(resultat, cv2.COLOR_BGR2RGB), 
                            caption=f"🎯 {total_actuel} pièces", use_column_width=True)
                
                col_d1, col_d2 = st.columns(2)
                with col_d1:
                    st.write("**Couleurs:**", dict(stats_couleur))
                with col_d2:
                    st.write("**Tailles:**", dict(stats_taille))

# Pied de page
st.markdown("---")
st.caption("""
🧩 Compteur de Pièces v3.4 - Interface Adaptative
• S'adapte automatiquement à votre appareil (mobile/PC)
• Pour OneDrive : téléchargez d'abord l'image sur votre PC
• Interface claire et simple d'utilisation
""")
