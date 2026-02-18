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
        # Couleurs HSV - Ajustées pour mieux détecter les boulons
        self.couleurs = {
            'rouge': {
                'lower1': np.array([0, 100, 100]), 'upper1': np.array([10, 255, 255]),
                'lower2': np.array([160, 100, 100]), 'upper2': np.array([180, 255, 255]),
                'couleur_bbox': (0, 0, 255)  # BGR pour OpenCV
            },
            'bleu': {
                'lower': np.array([100, 100, 50]), 'upper': np.array([130, 255, 255]),
                'couleur_bbox': (255, 0, 0)
            },
            'vert': {
                'lower': np.array([40, 50, 50]), 'upper': np.array([80, 255, 255]),
                'couleur_bbox': (0, 255, 0)
            },
            'jaune': {
                'lower': np.array([20, 100, 100]), 'upper': np.array([35, 255, 255]),
                'couleur_bbox': (0, 255, 255)
            },
            'gris': {
                'lower': np.array([0, 0, 50]), 'upper': np.array([180, 50, 200]),
                'couleur_bbox': (128, 128, 128)
            }
        }
        
        # Seuils de taille - Adaptés pour les boulons
        self.seuils_taille = {
            'P': (100, 500),     # Petite
            'M': (500, 1500),     # Moyenne
            'G': (1500, 3000),    # Grande
            'TG': (3000, 10000)   # Très Grande
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
        
        # Calculer la couleur moyenne de la pièce
        mean_val = cv2.mean(hsv, mask=mask)
        h_mean, s_mean, v_mean = mean_val[0], mean_val[1], mean_val[2]
        
        # Détection spéciale pour les boulons (gris/métal)
        if s_mean < 50 and 50 < v_mean < 200:
            best_couleur = 'gris'
            best_color_bbox = (128, 128, 128)
            best_score = 0.8
        
        for nom_couleur, params in self.couleurs.items():
            if nom_couleur == 'gris':
                continue
                
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
                if score > best_score and score > 0.3:  # Seuil plus élevé
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
        
        # Amélioration du contraste
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        frame_enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Détection des contours améliorée
        gray = cv2.cvtColor(frame_enhanced, cv2.COLOR_BGR2GRAY)
        
        # Filtre bilatéral pour préserver les bords
        blur = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Détection de contours adaptative
        edges = cv2.Canny(blur, 30, 100)
        
        # Opérations morphologiques pour fermer les contours
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Trouver les contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        pieces_actuelles = []
        stats_couleur_actuelles = defaultdict(int)
        stats_taille_actuelles = defaultdict(int)
        
        # Filtrer les contours par forme (circularité pour les boulons)
        for contour in contours:
            aire = cv2.contourArea(contour)
            
            # Ignorer les trop petits ou trop grands contours
            if aire < 100 or aire > 10000:
                continue
            
            # Calculer la circularité
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * aire / (perimeter * perimeter)
            else:
                circularity = 0
            
            # Les boulons ont généralement une forme allongée (faible circularité)
            # ou des formes avec des trous
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            
            # Critères pour identifier un boulon
            est_boulon = False
            
            # Boulon allongé
            if aspect_ratio > 1.5 or aspect_ratio < 0.67:
                est_boulon = True
            # Boulon avec circularité faible (forme non circulaire)
            elif circularity < 0.6:
                est_boulon = True
            # Boulon avec aire moyenne
            elif 300 < aire < 3000:
                est_boulon = True
            
            if not est_boulon:
                continue
            
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
                'circularite': circularity,
                'ratio': aspect_ratio
            })
            
            stats_couleur_actuelles[couleur_nom] += 1
            stats_taille_actuelles[taille_nom] += 1
            
            # Dessiner la pièce avec des infos supplémentaires
            cv2.rectangle(resultat, (x, y), (x+w, y+h), couleur_bbox, 2)
            cv2.circle(resultat, centre, 3, (255, 255, 255), -1)
            
            # Afficher le type de pièce
            if couleur_nom == 'gris':
                label = f"Boulon #{len(pieces_actuelles)}"
            else:
                label = f"#{len(pieces_actuelles)} {couleur_nom[0]}{taille_nom}"
            
            cv2.putText(resultat, label, (x+5, y+20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        total_actuel = len(pieces_actuelles)
        
        # Mise à jour des stats
        self.stats_couleur = stats_couleur_actuelles
        self.stats_taille = stats_taille_actuelles
        self.total_pieces = total_actuel
        
        # Ajouter le compteur total sur l'image
        h, w = resultat.shape[:2]
        cv2.putText(resultat, f"Total: {total_actuel}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return resultat, pieces_actuelles, stats_couleur_actuelles, stats_taille_actuelles, total_actuel

# Classe pour le traitement vidéo en temps réel
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.compteur = st.session_state.compteur
    
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        resultat, _, _, _, _ = self.compteur.traiter_frame(img)
        return av.VideoFrame.from_ndarray(resultat, format="bgr24")

# Fonction pour analyser une image avec debug
def analyser_image_debug(frame):
    """Version debug qui montre les étapes de traitement"""
    st.subheader("🔍 Analyse détaillée")
    
    # Créer les différentes étapes
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 9, 75, 75)
    edges = cv2.Canny(blur, 30, 100)
    
    kernel = np.ones((3, 3), np.uint8)
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Afficher les étapes
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.image(gray, caption="Niveaux de gris", use_column_width=True)
    with col2:
        st.image(blur, caption="Filtre bilatéral", use_column_width=True)
    with col3:
        st.image(edges, caption="Contours bruts", use_column_width=True)
    with col4:
        st.image(edges_closed, caption="Contours fermés", use_column_width=True)
    
    # Compter les contours
    contours, _ = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    st.write(f"**Nombre total de contours bruts:** {len(contours)}")
    
    # Filtrer par aire
    contours_filtres = [c for c in contours if 100 < cv2.contourArea(c) < 10000]
    st.write(f"**Après filtre de taille:** {len(contours_filtres)}")
    
    return len(contours_filtres)

# Initialisation du compteur dans la session
if 'compteur' not in st.session_state:
    st.session_state.compteur = CompteurPieces()
if 'frame_count' not in st.session_state:
    st.session_state.frame_count = 0
if 'mode' not in st.session_state:
    st.session_state.mode = None

compteur = st.session_state.compteur

# Interface Streamlit
st.title("🧩 Compteur de Pièces/Boulons")
st.markdown("""
Cette application détecte et compte automatiquement les pièces et boulons :
- **Détection améliorée pour les boulons** (formes allongées, couleur métal)
- **Classification par taille et couleur**
- **Filtrage intelligent** pour éviter les faux positifs
""")

# Interface principale
st.info("📱💻 Interface adaptative - Utilisez les options ci-dessous")

# Tabs pour organiser l'interface
tab1, tab2, tab3, tab4 = st.tabs([
    "📸 Prendre photo", 
    "🎥 Flux direct", 
    "🖼️ Upload image",
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
            
            st.success(f"✅ **{total_actuel} boulons/pièces** détectés !")
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 
                        caption="Photo originale", use_column_width=True)
            with col2:
                st.image(cv2.cvtColor(resultat, cv2.COLOR_BGR2RGB), 
                        caption=f"Résultat - {total_actuel} détectés", use_column_width=True)
            
            # Statistiques
            st.subheader("📊 Détails")
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                st.metric("Total", total_actuel)
                if 'gris' in stats_couleur:
                    st.metric("Boulons (gris)", stats_couleur['gris'])
            with col_s2:
                st.write("**Par taille:**")
                for t in ['P', 'M', 'G', 'TG']:
                    if stats_taille.get(t, 0) > 0:
                        st.write(f"- {t}: {stats_taille[t]}")
            
            # Option debug
            with st.expander("🔧 Voir l'analyse détaillée"):
                analyser_image_debug(frame)

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
        
        # Afficher les stats en direct
        st.metric("Boulons détectés", compteur.total_pieces)

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
            
            st.success(f"✅ **{total_actuel} boulons/pièces** détectés !")
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 
                        caption="Image originale", use_column_width=True)
            with col2:
                st.image(cv2.cvtColor(resultat, cv2.COLOR_BGR2RGB), 
                        caption=f"Résultat - {total_actuel} détectés", use_column_width=True)
            
            st.write("**Résultats détaillés :**")
            st.write(f"- Par couleur: {dict(stats_couleur)}")
            st.write(f"- Par taille: {dict(stats_taille)}")
            
            # Option debug
            with st.expander("🔧 Voir l'analyse détaillée"):
                analyser_image_debug(frame)

with tab4:
    st.subheader("🧪 Mode démo - Image de test")
    
    if st.button("🎲 Générer une image de test", use_container_width=True):
        with st.spinner("Génération et analyse..."):
            # Créer une image de test avec des formes de boulons
            test_img = np.zeros((480, 640, 3), dtype=np.uint8)
            test_img.fill(255)
            
            # Dessiner des boulons (rectangles allongés)
            cv2.rectangle(test_img, (150, 180), (200, 220), (100, 100, 100), -1)  # Gris
            cv2.rectangle(test_img, (300, 150), (380, 200), (100, 100, 100), -1)  # Gris allongé
            cv2.rectangle(test_img, (450, 200), (500, 280), (100, 100, 100), -1)  # Gris vertical
            cv2.circle(test_img, (250, 350), 30, (100, 100, 100), -1)  # Rond gris
            cv2.ellipse(test_img, (400, 350), (40, 20), 0, 0, 360, (100, 100, 100), -1)  # Ovale
            
            # Ajouter des pièces colorées
            cv2.circle(test_img, (200, 400), 25, (0, 0, 255), -1)   # Rouge
            cv2.circle(test_img, (500, 400), 25, (255, 0, 0), -1)   # Bleu
            
            resultat, pieces, stats_couleur, stats_taille, total_actuel = compteur.traiter_frame(test_img)
            
            st.success(f"✅ **{total_actuel} objets** détectés en mode démo !")
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB), 
                        caption="🧪 Image de test", use_column_width=True)
            with col2:
                st.image(cv2.cvtColor(resultat, cv2.COLOR_BGR2RGB), 
                        caption=f"Résultat - {total_actuel} détectés", use_column_width=True)
            
            st.write("**Résultats :**")
            st.write(f"- Couleurs: {dict(stats_couleur)}")
            st.write(f"- Tailles: {dict(stats_taille)}")

# Bouton de réinitialisation
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col2:
    if st.button("🔄 Réinitialiser les compteurs", use_container_width=True):
        compteur.reset_compteur()
        st.session_state.frame_count = 0
        st.success("✅ Compteurs réinitialisés !")
        st.rerun()

# Pied de page
st.markdown("---")
st.caption("""
🧩 Compteur de Pièces/Boulons v4.0 - Détection améliorée
• Détection spéciale pour les boulons (formes allongées, couleur métal)
• Filtrage intelligent pour éviter les faux positifs
• Interface claire et simple d'utilisation
""")
