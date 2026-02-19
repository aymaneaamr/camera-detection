import streamlit as st
import cv2
import numpy as np
from collections import defaultdict
from PIL import Image
import time
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

# Configuration de la page
st.set_page_config(
    page_title="Compteur de Pièces - Version Avancée",
    page_icon="🔧",
    layout="wide"
)

class CompteurPieces:
    def __init__(self):
        """Initialise le compteur de pièces avec technologies avancées"""
        # Couleurs HSV (inclut le gris pour les boulons)
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
            },
            'gris': {
                'lower': np.array([0, 0, 50]), 'upper': np.array([180, 50, 255]),
                'couleur_bbox': (128, 128, 128)
            }
        }
        
        # Seuils de taille ajustables
        self.seuils_taille = {
            'P': (0, 100),      # Très petite
            'M': (100, 500),    # Petite
            'G': (500, 2000),   # Moyenne
            'TG': (2000, 5000), # Grande
            'EX': (5000, float('inf'))  # Extra large
        }
        
        # Paramètres avancés
        self.params = {
            'seuil_aire_min': 30,
            'seuil_canny_bas': 30,
            'seuil_canny_haut': 100,
            'sensibilite_couleur': 0.15,
            'seuil_circularite': 0.7,
            'seuil_separation': 0.4,
            'utiliser_watershed': True,
            'utiliser_distance_transform': True,
            'utiliser_circularite': True,
            'mode_detection': "Tous"
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
    
    def est_circulaire(self, contour):
        """Vérifie si un contour est approximativement circulaire"""
        perimeter = cv2.arcLength(contour, True)
        area = cv2.contourArea(contour)
        
        if perimeter == 0 or area == 0:
            return False
        
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        return circularity > self.params['seuil_circularite']
    
    def separer_objets_distance_transform(self, contours, frame_shape):
        """Sépare les objets collés using distance transform"""
        nouveaux_contours = []
        
        for contour in contours:
            aire = cv2.contourArea(contour)
            
            if aire > 500:  # Seuil pour considérer la séparation
                x, y, w, h = cv2.boundingRect(contour)
                bbox_aire = w * h
                rapport = aire / bbox_aire if bbox_aire > 0 else 0
                
                if rapport < 0.6:  # Probablement plusieurs objets
                    # Créer un masque avec une marge
                    marge = 20
                    mask = np.zeros((h + 2*marge, w + 2*marge), dtype=np.uint8)
                    contour_shift = contour - [x - marge, y - marge]
                    cv2.drawContours(mask, [contour_shift], -1, 255, -1)
                    
                    # Distance transform
                    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
                    cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
                    
                    # Seuillage adaptatif
                    _, dist_thresh = cv2.threshold(dist, self.params['seuil_separation'], 1.0, cv2.THRESH_BINARY)
                    
                    # Nettoyage morphologique
                    kernel = np.ones((3, 3), np.uint8)
                    dist_thresh = cv2.erode(dist_thresh, kernel, iterations=1)
                    dist_thresh = cv2.dilate(dist_thresh, kernel, iterations=2)
                    
                    # Trouver les nouveaux contours
                    dist_contours, _ = cv2.findContours(
                        (dist_thresh * 255).astype(np.uint8),
                        cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE
                    )
                    
                    for dist_contour in dist_contours:
                        if cv2.contourArea(dist_contour) > self.params['seuil_aire_min']:
                            dist_contour = dist_contour + [x - marge, y - marge]
                            nouveaux_contours.append(dist_contour)
                    
                    continue
            
            nouveaux_contours.append(contour)
        
        return nouveaux_contours
    
    def separer_avec_watershed(self, frame, contours):
        """Utilise l'algorithme Watershed pour séparer les objets collés"""
        if len(contours) == 0:
            return contours
        
        # Créer un masque avec tous les contours
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        for contour in contours:
            cv2.drawContours(mask, [contour], -1, 255, -1)
        
        # Dilater légèrement
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        
        # Distance transform pour trouver les marqueurs
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
        
        # Trouver les pics locaux (centres présumés des objets)
        local_max = peak_local_max(dist, min_distance=20, exclude_border=False, indices=False)
        
        # Marqueurs pour watershed
        markers = ndimage.label(local_max)[0]
        markers = markers.astype(np.int32)
        
        # Appliquer watershed
        cv2.watershed(frame, markers)
        
        # Extraire les nouveaux contours
        nouveaux_contours = []
        for i in range(1, markers.max() + 1):
            marker_mask = np.zeros(markers.shape, dtype=np.uint8)
            marker_mask[markers == i] = 255
            
            # Nettoyer le masque
            marker_mask = cv2.erode(marker_mask, kernel, iterations=1)
            marker_mask = cv2.dilate(marker_mask, kernel, iterations=2)
            
            marker_contours, _ = cv2.findContours(
                marker_mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            for marker_contour in marker_contours:
                if cv2.contourArea(marker_contour) > self.params['seuil_aire_min']:
                    nouveaux_contours.append(marker_contour)
        
        return nouveaux_contours if nouveaux_contours else contours
    
    def ameliorer_contours(self, frame):
        """Améliore la détection des contours avec des techniques avancées"""
        # Prétraitement adaptatif
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Égalisation d'histogramme adaptative
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        
        # Filtre bilatéral pour préserver les bords
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Détection de contours avec Canny adaptatif
        median = np.median(gray)
        lower = int(max(0, (1.0 - 0.33) * median))
        upper = int(min(255, (1.0 + 0.33) * median))
        
        edges = cv2.Canny(gray, lower, upper)
        
        # Opérations morphologiques avancées
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel, iterations=1)
        
        return edges
    
    def get_couleur_piece(self, hsv, contour):
        """Détermine la couleur d'une pièce avec seuil ajustable"""
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        
        best_couleur = '?'
        best_score = 0
        best_color_bbox = (128, 128, 128)
        
        couleurs_a_verifier = self.couleurs.keys()
        if self.params['mode_detection'] == "Boulons":
            couleurs_a_verifier = ['gris']
        elif self.params['mode_detection'] == "Pièces colorées":
            couleurs_a_verifier = [c for c in self.couleurs.keys() if c != 'gris']
        
        for nom_couleur in couleurs_a_verifier:
            params = self.couleurs[nom_couleur]
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
                if score > best_score and score > self.params['sensibilite_couleur']:
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
        """Traite une frame avec toutes les technologies avancées"""
        resultat = frame.copy()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Détection améliorée des contours
        edges = self.ameliorer_contours(frame)
        
        # Trouver les contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filtrer par taille minimum
        contours = [c for c in contours if cv2.contourArea(c) > self.params['seuil_aire_min']]
        
        # Appliquer les techniques de séparation
        if self.params['utiliser_distance_transform']:
            contours = self.separer_objets_distance_transform(contours, frame.shape)
        
        if self.params['utiliser_watershed'] and len(contours) > 0:
            contours = self.separer_avec_watershed(frame, contours)
        
        pieces_actuelles = []
        stats_couleur_actuelles = defaultdict(int)
        stats_taille_actuelles = defaultdict(int)
        
        for contour in contours:
            aire = cv2.contourArea(contour)
            
            # Filtre de circularité optionnel
            if self.params['utiliser_circularite'] and not self.est_circulaire(contour):
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
                'centre': centre,
                'circularite': 4 * np.pi * aire / (cv2.arcLength(contour, True)**2) if cv2.arcLength(contour, True) > 0 else 0
            })
            
            stats_couleur_actuelles[couleur_nom] += 1
            stats_taille_actuelles[taille_nom] += 1
            
            # Dessiner avec style amélioré
            cv2.rectangle(resultat, (x, y), (x+w, y+h), couleur_bbox, 2)
            cv2.circle(resultat, centre, 3, (255, 255, 255), -1)
            
            # Ajouter des informations détaillées
            info_text = f"{couleur_nom[0]}{taille_nom}"
            cv2.putText(resultat, info_text, (x+5, y+20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Ajouter un petit indicateur de circularité
            if pieces_actuelles[-1]['circularite'] > 0.8:
                cv2.putText(resultat, "●", (x+w-15, y+20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        total_actuel = len(pieces_actuelles)
        
        # Mise à jour des stats
        self.stats_couleur = stats_couleur_actuelles
        self.stats_taille = stats_taille_actuelles
        self.total_pieces = total_actuel
        
        # Ajouter le compteur total avec style
        h, w = resultat.shape[:2]
        cv2.putText(resultat, f"Total: {total_actuel}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Ajouter un compteur de FPS si en temps réel
        if hasattr(self, 'fps'):
            cv2.putText(resultat, f"FPS: {self.fps:.1f}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return resultat, pieces_actuelles, stats_couleur_actuelles, stats_taille_actuelles, total_actuel

# Classe pour le traitement vidéo en temps réel avec FPS
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.compteur = compteur
        self.last_time = time.time()
        self.frame_count = 0
    
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Calcul FPS
        self.frame_count += 1
        current_time = time.time()
        if current_time - self.last_time >= 1.0:
            self.compteur.fps = self.frame_count / (current_time - self.last_time)
            self.frame_count = 0
            self.last_time = current_time
        
        resultat, _, _, _, _ = self.compteur.traiter_frame(img)
        return av.VideoFrame.from_ndarray(resultat, format="bgr24")

# Initialisation du compteur dans la session
if 'compteur' not in st.session_state:
    st.session_state.compteur = CompteurPieces()
if 'frame_count' not in st.session_state:
    st.session_state.frame_count = 0
if 'mode' not in st.session_state:
    st.session_state.mode = None

compteur = st.session_state.compteur

# Interface Streamlit
st.title("🔧 Compteur de Pièces - Version Technologique Avancée")
st.markdown("""
Cette application utilise des technologies de pointe en vision par ordinateur :
- **Détection multi-couleurs** (rouge, bleu, vert, jaune, gris)
- **Classification intelligente par taille** (P, M, G, TG, EX)
- **Séparation d'objets collés** (Distance Transform + Watershed)
- **Analyse de circularité** pour formes rondes
- **Détection adaptative des contours** (Canny auto-ajusté)
- **Interface adaptative** mobile/PC
""")

# Détection du type d'appareil
user_agent = st.query_params.get("user_agent", [""])[0] if hasattr(st, 'query_params') else ""
is_mobile = any(x in user_agent.lower() for x in ['android', 'iphone', 'mobile']) if user_agent else None

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
    st.info("📱 Mode téléphone - Interface optimisée")
    
    with st.container():
        st.subheader("📸 Prendre une photo")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            source = st.radio(
                "Source",
                ["📸 Caméra", "🖼️ Galerie", "🧪 Démo"],
                label_visibility="collapsed"
            )
        
        # Paramètres simplifiés pour mobile
        with st.expander("⚙️ Paramètres avancés"):
            mode_detection = st.selectbox(
                "Mode détection",
                ["Tous", "Pièces colorées", "Boulons"],
                index=0
            )
            compteur.params['mode_detection'] = mode_detection
            compteur.params['seuil_aire_min'] = st.slider("Taille min", 10, 200, 30)
        
        if source == "📸 Caméra":
            img_file = st.camera_input("Prendre une photo", key="mobile_camera")
            
            if img_file is not None:
                with st.spinner("🔍 Analyse avancée en cours..."):
                    bytes_data = img_file.getvalue()
                    frame = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
                    
                    resultat, pieces, stats_couleur, stats_taille, total_actuel = compteur.traiter_frame(frame)
                    
                    st.success(f"✅ **{total_actuel} pièces** détectées")
                    st.image(cv2.cvtColor(resultat, cv2.COLOR_BGR2RGB), use_column_width=True)
                    
                    # Stats avec tous les types
                    toutes_couleurs = ['rouge','bleu','vert','jaune','gris','?']
                    couleurs_affichees = [c for c in toutes_couleurs if stats_couleur.get(c,0)>0]
                    st.write("**Couleurs:** " + ", ".join([f"{c}:{stats_couleur.get(c,0)}" for c in couleurs_affichees]))
                    st.write("**Tailles:** " + ", ".join([f"{t}:{stats_taille.get(t,0)}" for t in ['P','M','G','TG','EX'] if stats_taille.get(t,0)>0]))
        
        elif source == "🖼️ Galerie":
            uploaded_file = st.file_uploader("Choisir image", type=['jpg', 'jpeg', 'png'], label_visibility="collapsed")
            
            if uploaded_file:
                with st.spinner("🔍 Analyse avancée en cours..."):
                    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    
                    resultat, pieces, stats_couleur, stats_taille, total_actuel = compteur.traiter_frame(frame)
                    
                    st.success(f"✅ **{total_actuel} pièces** détectées")
                    st.image(cv2.cvtColor(resultat, cv2.COLOR_BGR2RGB), use_column_width=True)
                    
                    with st.expander("📊 Détails complets"):
                        st.write("**Par couleur:**", dict(stats_couleur))
                        st.write("**Par taille:**", dict(stats_taille))
        
        else:  # Mode démo
            if st.button("🎲 Générer test avancé", use_container_width=True):
                with st.spinner("..."):
                    test_img = np.zeros((480, 640, 3), dtype=np.uint8)
                    test_img.fill(255)
                    
                    # Créer une scène de test avec objets variés
                    cv2.circle(test_img, (200, 200), 50, (0, 0, 255), -1)
                    cv2.circle(test_img, (350, 250), 40, (255, 0, 0), -1)
                    cv2.circle(test_img, (500, 200), 45, (0, 255, 0), -1)
                    cv2.circle(test_img, (300, 350), 35, (0, 255, 255), -1)
                    cv2.circle(test_img, (150, 350), 25, (128, 128, 128), -1)
                    
                    # Ajouter des objets collés
                    cv2.ellipse(test_img, (400, 400), (30, 30), 0, 0, 360, (0, 0, 255), -1)
                    cv2.ellipse(test_img, (440, 400), (30, 30), 0, 0, 360, (255, 0, 0), -1)
                    
                    resultat, pieces, stats_couleur, stats_taille, total_actuel = compteur.traiter_frame(test_img)
                    
                    st.success(f"✅ **{total_actuel} pièces** détectées")
                    st.image(cv2.cvtColor(resultat, cv2.COLOR_BGR2RGB), use_column_width=True)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        for c in ['rouge', 'bleu', 'vert', 'jaune', 'gris']:
                            if stats_couleur.get(c, 0) > 0:
                                st.metric(c.capitalize(), stats_couleur.get(c, 0))
                    with col2:
                        for t in ['P', 'M', 'G', 'TG', 'EX']:
                            if stats_taille.get(t, 0) > 0:
                                st.metric(f"Taille {t}", stats_taille.get(t, 0))

else:
    # ========== INTERFACE PC COMPLÈTE ==========
    st.info("💻 Mode PC - Interface complète avec tous les paramètres")
    
    # Sidebar pour tous les paramètres avancés
    with st.sidebar:
        st.header("⚙️ Configuration avancée")
        
        source = st.radio(
            "Source d'image",
            ["📸 Prendre une photo", "🎥 Flux en direct", "🖼️ Uploader une image", "🧪 Mode démo"]
        )
        
        st.markdown("---")
        st.header("🔬 Paramètres de détection")
        
        mode_detection = st.selectbox(
            "Mode de détection",
            ["Tous", "Pièces colorées", "Boulons"],
            index=0
        )
        compteur.params['mode_detection'] = mode_detection
        
        col1, col2 = st.columns(2)
        with col1:
            compteur.params['seuil_aire_min'] = st.slider("Aire minimum", 10, 500, 30)
            compteur.params['sensibilite_couleur'] = st.slider("Sensibilité couleur", 0.05, 0.5, 0.15, 0.05)
        with col2:
            compteur.params['seuil_circularite'] = st.slider("Seuil circularité", 0.3, 0.9, 0.7, 0.05)
            compteur.params['seuil_separation'] = st.slider("Sensibilité séparation", 0.2, 0.8, 0.4, 0.05)
        
        st.markdown("---")
        st.header("🧪 Technologies activées")
        
        compteur.params['utiliser_watershed'] = st.checkbox("Watershed (séparation avancée)", True)
        compteur.params['utiliser_distance_transform'] = st.checkbox("Distance Transform", True)
        compteur.params['utiliser_circularite'] = st.checkbox("Filtre de circularité", True)
        
        st.markdown("---")
        st.header("📊 Statistiques temps réel")
        
        stats_container = st.empty()
        
        if st.button("🔄 Réinitialiser compteurs", use_container_width=True):
            compteur.reset_compteur()
            st.session_state.frame_count = 0
            st.rerun()
        
        st.markdown("---")
        st.markdown("""
        ### 📝 Légende complète
        **Couleurs :**
        - 🔴 Rouge
        - 🔵 Bleu  
        - 🟢 Vert
        - 🟡 Jaune
        - ⚪ Gris (boulons)
        
        **Tailles :**
        - **P** : < 100 px
        - **M** : 100-500 px
        - **G** : 500-2000 px
        - **TG** : 2000-5000 px
        - **EX** : > 5000 px
        
        **Indicateurs :**
        - ● : Forme circulaire
        """)
    
    # Zone principale PC
    if source == "📸 Prendre une photo":
        st.subheader("📸 Capture photo avec analyse avancée")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            img_file = st.camera_input("Cliquez pour prendre une photo", key="pc_camera")
        
        if img_file is not None:
            with st.spinner("🔍 Analyse avec technologies avancées..."):
                bytes_data = img_file.getvalue()
                frame = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
                
                resultat, pieces, stats_couleur, stats_taille, total_actuel = compteur.traiter_frame(frame)
                st.session_state.frame_count += 1
                
                st.success(f"✅ **{total_actuel} pièces** détectées avec succès !")
                
                # Mise à jour des stats dans sidebar
                with stats_container:
                    st.metric("Pièces détectées", total_actuel)
                    st.metric("Frame", st.session_state.frame_count)
                
                col_img1, col_img2 = st.columns(2)
                with col_img1:
                    st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 
                            caption="📸 Image originale", use_column_width=True)
                with col_img2:
                    st.image(cv2.cvtColor(resultat, cv2.COLOR_BGR2RGB), 
                            caption=f"🎯 Analyse avancée - {total_actuel} pièces", use_column_width=True)
                
                # Statistiques détaillées avec tous les types
                st.subheader("📊 Analyse détaillée")
                
                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                with col_m1:
                    st.metric("Total pièces", total_actuel)
                with col_m2:
                    st.metric("Couleurs différentes", len([c for c in stats_couleur.values() if c > 0]))
                with col_m3:
                    st.metric("Tailles différentes", len([t for t in stats_taille.values() if t > 0]))
                with col_m4:
                    st.metric("Frame analyse", st.session_state.frame_count)
                
                # Tableau complet des couleurs
                st.write("**🎨 Répartition par couleur :**")
                cols = st.columns(6)
                couleurs_list = ['rouge', 'bleu', 'vert', 'jaune', 'gris', 'autre']
                color_emoji = {'rouge': '🔴', 'bleu': '🔵', 'vert': '🟢', 'jaune': '🟡', 'gris': '⚪', 'autre': '❓'}
                
                for i, couleur in enumerate(couleurs_list):
                    with cols[i]:
                        count = stats_couleur.get(couleur if couleur != 'autre' else '?', 0)
                        st.metric(f"{color_emoji[couleur]} {couleur}", count)
                
                # Tableau complet des tailles
                st.write("**📏 Répartition par taille :**")
                cols = st.columns(5)
                tailles_list = ['P', 'M', 'G', 'TG', 'EX']
                for i, taille in enumerate(tailles_list):
                    with cols[i]:
                        count = stats_taille.get(taille, 0)
                        st.metric(f"Taille {taille}", count)
                
                # Liste détaillée des pièces avec métriques avancées
                with st.expander("🔍 Voir l'analyse détaillée de chaque pièce"):
                    for i, piece in enumerate(pieces, 1):
                        col_p1, col_p2, col_p3 = st.columns(3)
                        with col_p1:
                            st.write(f"**Pièce #{i}**")
                            st.write(f"Couleur: {piece['couleur']}")
                            st.write(f"Taille: {piece['taille']}")
                        with col_p2:
                            st.write(f"Aire: {piece['aire']:.0f} px²")
                            st.write(f"Position: ({piece['centre'][0]}, {piece['centre'][1]})")
                        with col_p3:
                            st.write(f"Circularité: {piece['circularite']:.3f}")
                            st.write(f"Dimensions: {piece['bbox'][2]}x{piece['bbox'][3]}")
                        st.divider()
    
    elif source == "🎥 Flux en direct":
        st.subheader("🎥 Flux vidéo en temps réel avec technologies avancées")
        
        # Stats en direct dans la sidebar
        with stats_container:
            st.metric("Pièces actuellement", compteur.total_pieces)
            st.write("**Couleurs détectées:**")
            for c in ['rouge', 'bleu', 'vert', 'jaune', 'gris']:
                if compteur.stats_couleur.get(c, 0) > 0:
                    st.write(f"- {c}: {compteur.stats_couleur.get(c, 0)}")
        
        # Lancer le flux vidéo
        ctx = webrtc_streamer(
            key="object-detection-avance",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=VideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        
        if not ctx.state.playing:
            st.info("👆 **Cliquez sur 'START' pour activer la caméra avec analyse en temps réel**")
    
    elif source == "🖼️ Uploader une image":
        st.subheader("🖼️ Analyse d'image avec technologies avancées")
        
        uploaded_file = st.file_uploader("Choisissez une image (JPG, JPEG, PNG)", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file:
            with st.spinner("🔍 Analyse avancée en cours..."):
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
                            caption=f"🎯 Analyse avancée - {total_actuel} pièces", use_column_width=True)
                
                st.subheader("📊 Résultats de l'analyse")
                col_s1, col_s2 = st.columns(2)
                with col_s1:
                    st.write("**Par couleur :**")
                    for couleur in ['rouge', 'bleu', 'vert', 'jaune', 'gris', '?']:
                        count = stats_couleur.get(couleur, 0)
                        if count > 0:
                            st.write(f"- {couleur}: {count}")
                with col_s2:
                    st.write("**Par taille :**")
                    for taille in ['P', 'M', 'G', 'TG', 'EX']:
                        count = stats_taille.get(taille, 0)
                        if count > 0:
                            st.write(f"- {taille}: {count}")
    
    else:  # Mode démo
        st.subheader("🧪 Mode démo - Test de toutes les technologies")
        
        if st.button("🎲 Générer scène de test complexe", use_container_width=True):
            with st.spinner("🔍 Analyse avec toutes les technologies..."):
                test_img = np.zeros((600, 800, 3), dtype=np.uint8)
                test_img.fill(240)
                
                # Créer une scène de test complexe avec objets variés et collés
                # Objets individuels
                cv2.circle(test_img, (200, 150), 45, (0, 0, 255), -1)      # rouge
                cv2.circle(test_img, (350, 150), 35, (255, 0, 0), -1)      # bleu
                cv2.circle(test_img, (500, 150), 40, (0, 255, 0), -1)      # vert
                cv2.circle(test_img, (650, 150), 30, (0, 255, 255), -1)    # jaune
                cv2.circle(test_img, (100, 300), 25, (128, 128, 128), -1)  # gris
                
                # Objets collés (pour tester la séparation)
                # Groupe de 3 cercles collés
                cv2.circle(test_img, (250, 350), 35, (0, 0, 255), -1)
                cv2.circle(test_img, (300, 350), 35, (0, 0, 255), -1)
                cv2.circle(test_img, (275, 300), 35, (0, 0, 255), -1)
                
                # Groupe de 2 ellipses collées
                cv2.ellipse(test_img, (500, 350), (40, 30), 0, 0, 360, (255, 0, 0), -1)
                cv2.ellipse(test_img, (550, 350), (40, 30), 0, 0, 360, (255, 0, 0), -1)
                
                # Objets de différentes tailles
                cv2.circle(test_img, (150, 500), 60, (0, 255, 0), -1)      # grand
                cv2.circle(test_img, (400, 500), 20, (255, 255, 0), -1)    # petit
                cv2.circle(test_img, (600, 500), 80, (128, 128, 128), -1)  # très grand gris
                
                resultat, pieces, stats_couleur, stats_taille, total_actuel = compteur.traiter_frame(test_img)
                st.session_state.frame_count += 1
                
                st.success(f"✅ **{total_actuel} pièces** détectées avec les technologies avancées !")
                
                col_img1, col_img2 = st.columns(2)
                with col_img1:
                    st.image(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB), 
                            caption="🧪 Scène de test complexe", use_column_width=True)
                with col_img2:
                    st.image(cv2.cvtColor(resultat, cv2.COLOR_BGR2RGB), 
                            caption=f"🎯 Résultat - {total_actuel} pièces séparées", use_column_width=True)
                
                # Afficher les performances des technologies
                st.subheader("📊 Performance des technologies utilisées")
                col_t1, col_t2, col_t3, col_t4 = st.columns(4)
                with col_t1:
                    st.metric("Watershed", "Activé" if compteur.params['utiliser_watershed'] else "Désactivé")
                with col_t2:
                    st.metric("Distance Transform", "Activé" if compteur.params['utiliser_distance_transform'] else "Désactivé")
                with col_t3:
                    st.metric("Filtre circularité", "Activé" if compteur.params['utiliser_circularite'] else "Désactivé")
                with col_t4:
                    st.metric("Détection adaptative", "Activé")
                
                # Résultats détaillés
                col_d1, col_d2 = st.columns(2)
                with col_d1:
                    st.write("**Couleurs détectées:**", dict(stats_couleur))
                with col_d2:
                    st.write("**Tailles détectées:**", dict(stats_taille))
                
                # Statistiques de séparation
                with st.expander("📈 Statistiques avancées"):
                    total_aire = sum(p['aire'] for p in pieces)
                    moyenne_aire = total_aire / len(pieces) if pieces else 0
                    st.write(f"**Aire totale:** {total_aire:.0f} px²")
                    st.write(f"**Aire moyenne:** {moyenne_aire:.0f} px²")
                    st.write(f"**Circularité moyenne:** {np.mean([p['circularite'] for p in pieces]):.3f}")
                    st.write(f"**Nombre de contours traités:** {len(pieces)}")

# Pied de page commun avec crédits technologiques
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>🔧 <strong>Compteur de Pièces v4.0 - Édition Technologies Avancées</strong></p>
    <p style='font-size: 0.8em; color: gray;'>
        Technologies intégrées : OpenCV • Watershed • Distance Transform • Détection adaptative Canny • Analyse de circularité • Filtrage HSV multi-plages • Streamlit WebRTC
    </p>
</div>
""", unsafe_allow_html=True)
