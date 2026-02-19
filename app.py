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
    page_title="Compteur de Pièces - Séparation avancée",
    page_icon="🔧",
    layout="wide"
)

class CompteurPieces:
    def __init__(self):
        """Initialise le compteur de pièces avec séparation des objets collés"""
        # Couleurs HSV
        self.couleurs = {
            'rouge': {
                'lower1': np.array([0, 100, 100]), 'upper1': np.array([10, 255, 255]),
                'lower2': np.array([160, 100, 100]), 'upper2': np.array([180, 255, 255]),
                'couleur_bbox': (0, 0, 255)  # BGR
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
            'gris': {   # Ajout pour boulons
                'lower': np.array([0, 0, 30]), 'upper': np.array([180, 60, 255]),
                'couleur_bbox': (128, 128, 128)
            }
        }
        
        # Seuils de taille (ajustés pour petits objets)
        self.seuils_taille = {
            'P': (0, 100),
            'M': (100, 500),
            'G': (500, 2000),
            'TG': (2000, 5000),
            'EX': (5000, float('inf'))
        }
        
        # Paramètres de séparation
        self.force_separation = 3  # 1 à 5
        self.seuil_aire_min = 30
        
        self.reset_compteur()
    
    def reset_compteur(self):
        """Réinitialise tous les compteurs"""
        self.stats_couleur = defaultdict(int)
        self.stats_taille = defaultdict(int)
        self.total_pieces = 0
        self.stats_couleur_total = defaultdict(int)
        self.stats_taille_total = defaultdict(int)
        self.total_pieces_cumule = 0
    
    def separer_objets_colles(self, contours, frame_shape):
        """
        Sépare les contours qui pourraient représenter plusieurs objets collés
        en utilisant distance transform et composants connectés.
        """
        nouveaux_contours = []
        
        for contour in contours:
            aire = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            bbox_aire = w * h
            rapport = aire / bbox_aire if bbox_aire > 0 else 0
            
            # Critères pour considérer qu'un contour est "collé"
            est_collé = False
            if aire > 500 and rapport < 0.6:  # grand et peu rempli
                est_collé = True
            elif aire > 1000:  # très grand, on force la tentative
                est_collé = True
            
            # Ajustement selon la force de séparation
            force = self.force_separation / 5.0  # normalisation 0.2 à 1.0
            if aire > 200 * force and rapport < (0.7 - 0.1 * force):
                est_collé = True
            
            if est_collé:
                # Créer un masque du contour avec une petite marge
                marge = 10
                mask = np.zeros((h + 2*marge, w + 2*marge), dtype=np.uint8)
                contour_shift = contour - [x - marge, y - marge]
                cv2.drawContours(mask, [contour_shift], -1, 255, -1)
                
                # Distance transform pour trouver les centres
                dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
                cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
                
                # Seuillage adaptatif (plus agressif si force élevée)
                seuil_dist = max(0.2, 0.5 - 0.1 * force)
                _, dist_thresh = cv2.threshold(dist, seuil_dist, 1.0, cv2.THRESH_BINARY)
                
                # Nettoyage morphologique
                kernel = np.ones((3, 3), np.uint8)
                dist_thresh = cv2.erode(dist_thresh, kernel, iterations=1)
                dist_thresh = cv2.dilate(dist_thresh, kernel, iterations=2)
                
                # Trouver les composants connectés
                num_labels, labels = cv2.connectedComponents((dist_thresh * 255).astype(np.uint8))
                
                if num_labels > 1:  # au moins un composant (fond exclu)
                    for i in range(1, num_labels):
                        # Créer un masque pour chaque composant
                        comp_mask = np.zeros_like(mask)
                        comp_mask[labels == i] = 255
                        
                        # Dilater pour retrouver une taille proche de l'original
                        comp_mask = cv2.dilate(comp_mask, kernel, iterations=2)
                        
                        # Trouver le contour du composant
                        comp_contours, _ = cv2.findContours(
                            comp_mask,
                            cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE
                        )
                        for comp_c in comp_contours:
                            if cv2.contourArea(comp_c) > self.seuil_aire_min:
                                # Replacer le contour dans le repère original
                                comp_c = comp_c + [x - marge, y - marge]
                                nouveaux_contours.append(comp_c)
                    continue  # on ne garde pas le contour original
            
            # Si pas considéré comme collé, on garde le contour original
            nouveaux_contours.append(contour)
        
        return nouveaux_contours
    
    def get_couleur_piece(self, hsv, contour):
        """Détermine la couleur dominante dans le contour"""
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
                if score > best_score and score > 0.15:
                    best_score = score
                    best_couleur = nom_couleur
                    best_color_bbox = params['couleur_bbox']
        
        return best_couleur, best_color_bbox
    
    def get_taille_piece(self, aire):
        """Détermine la taille selon les seuils"""
        for nom_taille, (min_vol, max_vol) in self.seuils_taille.items():
            if min_vol <= aire < max_vol:
                return nom_taille
        return '?'
    
    def traiter_frame(self, frame):
        """Traite une frame avec détection et séparation des objets collés"""
        resultat = frame.copy()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Détection des contours améliorée
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 30, 100)  # seuils plus bas pour plus de détails
        
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=2)
        edges = cv2.erode(edges, kernel, iterations=1)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filtrer par taille minimum
        contours = [c for c in contours if cv2.contourArea(c) > self.seuil_aire_min]
        
        # Étape clé : séparer les objets collés
        contours = self.separer_objets_colles(contours, frame.shape)
        
        pieces_actuelles = []
        stats_couleur_actuelles = defaultdict(int)
        stats_taille_actuelles = defaultdict(int)
        
        for idx, contour in enumerate(contours):
            aire = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            centre = (x + w//2, y + h//2)
            
            couleur_nom, couleur_bbox = self.get_couleur_piece(hsv, contour)
            taille_nom = self.get_taille_piece(aire)
            
            pieces_actuelles.append({
                'id': idx+1,
                'contour': contour,
                'aire': aire,
                'bbox': (x, y, w, h),
                'couleur': couleur_nom,
                'taille': taille_nom,
                'centre': centre
            })
            
            stats_couleur_actuelles[couleur_nom] += 1
            stats_taille_actuelles[taille_nom] += 1
            
            # Dessiner avec un style plus informatif
            cv2.rectangle(resultat, (x, y), (x+w, y+h), couleur_bbox, 2)
            cv2.circle(resultat, centre, 3, (255, 255, 255), -1)
            cv2.putText(resultat, f"#{idx+1}", (x+5, y-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, couleur_bbox, 1)
            cv2.putText(resultat, f"{couleur_nom[0]}{taille_nom}", (x+5, y+20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        total_actuel = len(pieces_actuelles)
        
        # Mise à jour des stats
        self.stats_couleur = stats_couleur_actuelles
        self.stats_taille = stats_taille_actuelles
        self.total_pieces = total_actuel
        
        # Ajouter le compteur total avec objectif 32
        h, w = resultat.shape[:2]
        cv2.rectangle(resultat, (5, 5), (250, 70), (0, 0, 0), -1)
        cv2.putText(resultat, f"Detecte: {total_actuel}", (15, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(resultat, "Objectif: 32", (15, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        
        return resultat, pieces_actuelles, stats_couleur_actuelles, stats_taille_actuelles, total_actuel

# Classe pour le traitement vidéo en temps réel
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.compteur = compteur
        self.last_time = time.time()
        self.frame_count = 0
    
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        # Calcul FPS optionnel
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
st.title("🔧 Compteur de Pièces - Séparation des objets collés")
st.markdown("""
Cette application détecte et compte automatiquement les pièces, **même si elles sont collées**.
- **Détection multi-couleurs** (rouge, bleu, vert, jaune, gris)
- **Classification par taille** (P, M, G, TG, EX)
- **Séparation intelligente** des objets proches
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
        
        # Paramètres simplifiés
        with st.expander("⚙️ Réglages"):
            force = st.slider("Force séparation", 1, 5, 3)
            compteur.force_separation = force
            seuil = st.slider("Taille min", 10, 100, 30)
            compteur.seuil_aire_min = seuil
        
        source = st.radio("Source", ["📸 Caméra", "🖼️ Galerie", "🧪 Démo"], label_visibility="collapsed")
        
        if source == "📸 Caméra":
            img_file = st.camera_input("Prendre une photo", key="mobile_camera")
            if img_file:
                with st.spinner("🔍 Analyse..."):
                    bytes_data = img_file.getvalue()
                    frame = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
                    resultat, pieces, stats_couleur, stats_taille, total = compteur.traiter_frame(frame)
                    
                    st.image(cv2.cvtColor(resultat, cv2.COLOR_BGR2RGB), use_column_width=True)
                    
                    if total == 32:
                        st.success(f"✅ Parfait ! {total}/32 pièces")
                    else:
                        st.warning(f"⚠️ {total}/32 pièces détectées ({32-total} manquantes)")
                    
                    st.write("**Couleurs:**", dict(stats_couleur))
                    st.write("**Tailles:**", dict(stats_taille))
        
        elif source == "🖼️ Galerie":
            uploaded = st.file_uploader("Choisir image", type=['jpg','jpeg','png'], label_visibility="collapsed")
            if uploaded:
                with st.spinner("🔍 Analyse..."):
                    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
                    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    resultat, pieces, stats_couleur, stats_taille, total = compteur.traiter_frame(frame)
                    
                    st.image(cv2.cvtColor(resultat, cv2.COLOR_BGR2RGB), use_column_width=True)
                    st.write(f"**Total:** {total}/32")
        
        else:  # Démo
            if st.button("🎲 Générer test"):
                with st.spinner("..."):
                    test_img = np.zeros((480, 640, 3), dtype=np.uint8)
                    test_img.fill(255)
                    # Créer quelques objets collés
                    for i in range(4):
                        for j in range(2):
                            x = 150 + i*120
                            y = 150 + j*120
                            cv2.circle(test_img, (x, y), 30, (0,0,255), -1)
                            if i<3:
                                cv2.circle(test_img, (x+40, y+20), 30, (0,0,255), -1)
                    resultat, pieces, stats_couleur, stats_taille, total = compteur.traiter_frame(test_img)
                    st.image(cv2.cvtColor(resultat, cv2.COLOR_BGR2RGB), use_column_width=True)
                    st.write(f"**Détecté:** {total} pièces")
else:
    # ========== INTERFACE PC COMPLÈTE ==========
    st.info("💻 Mode PC - Paramètres avancés")
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        source = st.radio("Source", ["📸 Prendre une photo", "🎥 Flux en direct", "🖼️ Uploader une image", "🧪 Mode démo"])
        
        st.markdown("---")
        st.header("🔧 Paramètres de séparation")
        
        force = st.slider("💪 Force de séparation", 1, 5, 3,
                         help="Plus la valeur est élevée, plus l'algorithme divise les objets collés")
        compteur.force_separation = force
        
        seuil_min = st.slider("📏 Taille minimum (px)", 10, 100, 30,
                            help="Aire minimum pour considérer un objet")
        compteur.seuil_aire_min = seuil_min
        
        st.markdown("---")
        st.header("📊 Objectif 32 pièces")
        
        # Affichage dynamique (sera mis à jour après analyse)
        st.metric("Actuellement", compteur.total_pieces)
        if compteur.total_pieces < 32:
            st.warning(f"❌ Manque {32 - compteur.total_pieces}")
        elif compteur.total_pieces > 32:
            st.warning(f"⚠️ Trop de pièces (+{compteur.total_pieces - 32})")
        else:
            st.success("✅ Objectif atteint !")
        
        if st.button("🔄 Réinitialiser", use_container_width=True):
            compteur.reset_compteur()
            st.rerun()
        
        st.markdown("---")
        st.markdown("""
        ### 📝 Légende
        - 🔴 Rouge
        - 🔵 Bleu  
        - 🟢 Vert
        - 🟡 Jaune
        - ⚪ Gris (boulons)
        
        ### 📏 Tailles
        - **P** : <100
        - **M** : 100-500
        - **G** : 500-2000
        - **TG** : 2000-5000
        - **EX** : >5000
        """)
    
    # Zone principale
    if source == "📸 Prendre une photo":
        st.subheader("📸 Prenez une photo")
        
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            img_file = st.camera_input("Cliquez pour prendre une photo", key="pc_camera")
        
        if img_file:
            with st.spinner("🔍 Analyse avec séparation..."):
                bytes_data = img_file.getvalue()
                frame = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
                resultat, pieces, stats_couleur, stats_taille, total = compteur.traiter_frame(frame)
                st.session_state.frame_count += 1
                
                col_img1, col_img2 = st.columns(2)
                with col_img1:
                    st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Originale")
                with col_img2:
                    st.image(cv2.cvtColor(resultat, cv2.COLOR_BGR2RGB), caption=f"Détection ({total} pièces)")
                
                # Statistiques
                st.subheader("📊 Résultats")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total", total)
                with col2:
                    st.metric("Objectif", 32)
                with col3:
                    st.metric("Différence", total - 32)
                
                # Détails
                col_c1, col_c2 = st.columns(2)
                with col_c1:
                    st.write("**Couleurs:**")
                    for c in ['rouge','bleu','vert','jaune','gris']:
                        if stats_couleur.get(c,0)>0:
                            st.write(f"- {c}: {stats_couleur[c]}")
                with col_c2:
                    st.write("**Tailles:**")
                    for t in ['P','M','G','TG','EX']:
                        if stats_taille.get(t,0)>0:
                            st.write(f"- {t}: {stats_taille[t]}")
    
    elif source == "🎥 Flux en direct":
        st.subheader("🎥 Flux vidéo en temps réel")
        
        # Stats en direct dans la sidebar
        with st.sidebar:
            st.metric("Pièces en direct", compteur.total_pieces)
        
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
        
        uploaded = st.file_uploader("Choisissez une image", type=['jpg','jpeg','png'])
        if uploaded:
            with st.spinner("🔍 Analyse..."):
                file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
                frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                resultat, pieces, stats_couleur, stats_taille, total = compteur.traiter_frame(frame)
                
                st.image(cv2.cvtColor(resultat, cv2.COLOR_BGR2RGB), use_column_width=True)
                st.write(f"**Total:** {total} / 32 pièces")
    
    else:  # Mode démo
        st.subheader("🧪 Mode démo - Test de séparation")
        
        if st.button("🎲 Générer une image de test (objets collés)"):
            with st.spinner("🔍 Analyse..."):
                test_img = np.zeros((480, 640, 3), dtype=np.uint8)
                test_img.fill(255)
                
                # Groupe de 3 cercles collés
                cv2.circle(test_img, (200, 200), 35, (0,0,255), -1)
                cv2.circle(test_img, (250, 200), 35, (0,0,255), -1)
                cv2.circle(test_img, (225, 160), 35, (0,0,255), -1)
                
                # Autre groupe
                cv2.circle(test_img, (400, 300), 30, (255,0,0), -1)
                cv2.circle(test_img, (440, 300), 30, (255,0,0), -1)
                
                # Objet isolé
                cv2.circle(test_img, (500, 150), 40, (0,255,0), -1)
                
                resultat, pieces, stats_couleur, stats_taille, total = compteur.traiter_frame(test_img)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.image(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB), caption="Image de test")
                with col2:
                    st.image(cv2.cvtColor(resultat, cv2.COLOR_BGR2RGB), caption=f"Résultat ({total} pièces détectées)")
                
                st.write("**Détails:**", dict(stats_couleur))

# Pied de page
st.markdown("---")
st.caption("🔧 Compteur de Pièces v5.0 - Séparation des objets collés par distance transform")
