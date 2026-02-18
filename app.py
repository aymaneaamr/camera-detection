import streamlit as st
import cv2
import numpy as np
from collections import defaultdict
from PIL import Image
import time
import os
from pathlib import Path

# Configuration de la page
st.set_page_config(
    page_title="Compteur de Pièces",
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="auto"
)

# Détection du type d'appareil (téléphone ou PC)
def detecter_appareil():
    """Détecte si l'utilisateur est sur mobile"""
    try:
        # Méthode 1: Via les headers (si disponible)
        if hasattr(st, 'query_params'):
            user_agent = st.query_params.get('user_agent', [''])[0]
            if 'mobile' in user_agent.lower():
                return True
        
        # Méthode 2: Via la largeur d'écran (approximatif)
        # On utilise st.markdown avec du JavaScript
        import streamlit.components.v1 as components
        
        mobile_detection_script = """
        <script>
            // Détection simple du mobile
            const isMobile = /iPhone|iPad|iPod|Android/i.test(navigator.userAgent);
            const isSmallScreen = window.innerWidth < 768;
            
            // Envoyer le résultat à Streamlit
            const mobile = isMobile || isSmallScreen;
            window.parent.postMessage({type: 'mobile_detection', isMobile: mobile}, '*');
        </script>
        """
        
        components.html(mobile_detection_script, height=0)
        
        # Par défaut, on suppose que c'est un PC
        # La valeur sera mise à jour si le JS s'exécute
        return False
        
    except:
        # En cas d'erreur, on suppose PC
        return False

# Initialisation des états de session
if 'compteur' not in st.session_state:
    st.session_state.compteur = CompteurPieces()
if 'frame_count' not in st.session_state:
    st.session_state.frame_count = 0
if 'is_mobile' not in st.session_state:
    st.session_state.is_mobile = detecter_appareil()
if 'camera_active' not in st.session_state:
    st.session_state.camera_active = False
if 'photos_prises' not in st.session_state:
    st.session_state.photos_prises = []

class CompteurPieces:
    # ... (votre classe CompteurPieces existante) ...
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
        
        return resultat, pieces_actuelles, stats_couleur_actuelles, stats_taille_actuelles, total_actuel

# Interface principale
st.title("🧩 Compteur de Pièces")

# Afficher le mode actuel
device_emoji = "📱" if st.session_state.is_mobile else "💻"
st.caption(f"{device_emoji} Mode : {'Téléphone' if st.session_state.is_mobile else 'PC'}")

# Sidebar adaptative
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Options adaptées à l'appareil
    if st.session_state.is_mobile:
        source = st.radio(
            "Source",
            ["📸 Appareil photo", "🖼️ Galerie", "🧪 Mode démo"],
            horizontal=True
        )
    else:
        source = st.radio(
            "Source",
            ["📸 Appareil photo", "🖼️ Galerie", "📁 OneDrive", "🧪 Mode démo"],
            horizontal=False
        )
    
    st.markdown("---")
    st.header("📊 Statistiques")
    
    if st.button("🔄 Réinitialiser compteurs"):
        st.session_state.compteur.reset_compteur()
        st.session_state.frame_count = 0
        st.rerun()
    
    st.markdown("---")
    
    # Légende adaptative
    if st.session_state.is_mobile:
        with st.expander("📝 Légende"):
            st.markdown("""
            - 🔴 Rouge
            - 🔵 Bleu  
            - 🟢 Vert
            - 🟡 Jaune
            - **P** < 500 px
            - **M** 500-2000 px
            - **G** 2000-5000 px
            - **TG** > 5000 px
            """)
    else:
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

# Zone principale selon la source choisie
if source == "📸 Appareil photo" or source == "📸 Appareil photo (téléphone)":
    st.subheader("📸 Prendre une photo")
    
    if st.session_state.is_mobile:
        # Interface optimisée pour mobile
        st.markdown("""
        <div style='text-align: center; padding: 10px; background: #f0f2f6; border-radius: 10px; margin-bottom: 10px;'>
            <p style='font-size: 1.2em;'>📱 Appuyez pour prendre une photo</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Widget caméra avec paramètres mobile
        img_file = st.camera_input(
            "Prendre une photo",
            key=f"camera_mobile_{time.time()}",
            help="Appuyez pour utiliser l'appareil photo"
        )
        
    else:
        # Interface PC
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            img_file = st.camera_input(
                "Cliquez pour prendre une photo",
                key=f"camera_pc_{time.time()}"
            )
    
    if img_file is not None:
        with st.spinner("🔍 Analyse en cours..."):
            # Traitement de l'image
            bytes_data = img_file.getvalue()
            frame = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            
            # Redimensionnement adaptatif
            height, width = frame.shape[:2]
            max_width = 400 if st.session_state.is_mobile else 800
            if width > max_width:
                scale = max_width / width
                new_width = int(width * scale)
                new_height = int(height * scale)
                frame = cv2.resize(frame, (new_width, new_height))
            
            # Traitement
            resultat, pieces, stats_couleur, stats_taille, total_actuel = st.session_state.compteur.traiter_frame(frame)
            st.session_state.frame_count += 1
            
            # Sauvegarder dans l'historique
            st.session_state.photos_prises.append({
                'time': time.time(),
                'total': total_actuel,
                'stats_couleur': stats_couleur,
                'stats_taille': stats_taille
            })
            
            # Affichage des résultats (adaptatif)
            if st.session_state.is_mobile:
                # Affichage vertical pour mobile
                st.image(cv2.cvtColor(resultat, cv2.COLOR_BGR2RGB), 
                        caption=f"🎯 {total_actuel} pièces", use_column_width=True)
                
                # Stats en colonnes
                col_m1, col_m2 = st.columns(2)
                with col_m1:
                    st.metric("Total", total_actuel)
                with col_m2:
                    st.metric("Frame", st.session_state.frame_count)
                
                # Détail des couleurs en horizontal
                st.write("**Couleurs:**")
                cols = st.columns(4)
                coul_ordre = ['rouge', 'bleu', 'vert', 'jaune']
                for i, c in enumerate(coul_ordre):
                    with cols[i]:
                        count = stats_couleur.get(c, 0)
                        emoji = ['🔴', '🔵', '🟢', '🟡'][i]
                        st.metric(emoji, count)
                
            else:
                # Affichage horizontal pour PC
                col_img1, col_img2 = st.columns(2)
                with col_img1:
                    st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 
                            caption="Originale", use_column_width=True)
                with col_img2:
                    st.image(cv2.cvtColor(resultat, cv2.COLOR_BGR2RGB), 
                            caption=f"Résultat: {total_actuel} pièces", use_column_width=True)
                
                # Statistiques
                st.subheader("📊 Détail")
                col_s1, col_s2 = st.columns(2)
                with col_s1:
                    st.write("**Par couleur :**")
                    for couleur, count in stats_couleur.items():
                        if count > 0:
                            emoji = {'rouge': '🔴', 'bleu': '🔵', 'vert': '🟢', 'jaune': '🟡'}.get(couleur, '⚪')
                            st.write(f"{emoji} {couleur}: {count}")
                with col_s2:
                    st.write("**Par taille :**")
                    for taille in ['P', 'M', 'G', 'TG']:
                        count = stats_taille.get(taille, 0)
                        if count > 0:
                            st.write(f"- {taille}: {count}")

elif source == "🖼️ Galerie":
    st.subheader("🖼️ Choisir une photo existante")
    
    # Interface adaptative pour la galerie
    if st.session_state.is_mobile:
        # Version mobile avec sélection simple
        uploaded_file = st.file_uploader(
            "Sélectionner une photo",
            type=['jpg', 'jpeg', 'png'],
            help="Choisissez une photo dans votre galerie"
        )
    else:
        # Version PC avec options supplémentaires
        col_up, col_path = st.columns([2, 1])
        with col_up:
            uploaded_file = st.file_uploader(
                "Choisir une photo",
                type=['jpg', 'jpeg', 'png']
            )
        with col_path:
            if st.button("📁 Ouvrir OneDrive"):
                home = str(Path.home())
                onedrive = os.path.join(home, "OneDrive", "Pictures", "Camera Roll")
                if os.path.exists(onedrive):
                    try:
                        os.startfile(onedrive)
                    except:
                        st.error("Impossible d'ouvrir le dossier")
    
    if uploaded_file:
        # Traitement de l'image (identique aux deux versions)
        bytes_data = uploaded_file.getvalue()
        frame = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        
        # Redimensionnement adaptatif
        height, width = frame.shape[:2]
        max_width = 400 if st.session_state.is_mobile else 800
        if width > max_width:
            scale = max_width / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            frame = cv2.resize(frame, (new_width, new_height))
        
        # Traitement
        resultat, pieces, stats_couleur, stats_taille, total_actuel = st.session_state.compteur.traiter_frame(frame)
        
        # Affichage adaptatif (similaire à la section caméra)
        if st.session_state.is_mobile:
            st.image(cv2.cvtColor(resultat, cv2.COLOR_BGR2RGB), 
                    caption=f"🎯 {total_actuel} pièces", use_column_width=True)
            st.metric("Total", total_actuel)
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Originale")
            with col2:
                st.image(cv2.cvtColor(resultat, cv2.COLOR_BGR2RGB), caption=f"Résultat: {total_actuel}")

elif source == "📁 OneDrive" and not st.session_state.is_mobile:
    st.subheader("📁 Photos OneDrive")
    
    # Interface OneDrive (uniquement sur PC)
    home = str(Path.home())
    onedrive_photos = os.path.join(home, "OneDrive", "Pictures", "Camera Roll")
    
    if os.path.exists(onedrive_photos):
        photos = [f for f in os.listdir(onedrive_photos) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        photos.sort(reverse=True)
        
        selected = st.selectbox("Choisir une photo", photos[:20])
        
        if selected:
            chemin = os.path.join(onedrive_photos, selected)
            image = Image.open(chemin)
            st.image(image, caption=selected, use_column_width=True)
            
            if st.button("🔍 Analyser"):
                frame = np.array(image)
                if len(frame.shape) == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                resultat, pieces, stats_couleur, stats_taille, total = st.session_state.compteur.traiter_frame(frame)
                st.image(cv2.cvtColor(resultat, cv2.COLOR_BGR2RGB), caption=f"Résultat: {total} pièces")

else:  # Mode démo
    st.subheader("🧪 Mode démo")
    
    if st.button("🎲 Générer une image de test"):
        # Création d'une image de test
        test_img = np.zeros((480, 640, 3), dtype=np.uint8)
        test_img.fill(255)
        
        cv2.circle(test_img, (200, 200), 50, (0, 0, 255), -1)
        cv2.circle(test_img, (350, 250), 40, (255, 0, 0), -1)
        cv2.circle(test_img, (500, 200), 45, (0, 255, 0), -1)
        cv2.circle(test_img, (300, 350), 35, (0, 255, 255), -1)
        
        resultat, pieces, stats_couleur, stats_taille, total = st.session_state.compteur.traiter_frame(test_img)
        
        # Affichage adaptatif
        if st.session_state.is_mobile:
            st.image(cv2.cvtColor(resultat, cv2.COLOR_BGR2RGB), 
                    caption=f"🎯 {total} pièces", use_column_width=True)
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.image(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB), caption="Test")
            with col2:
                st.image(cv2.cvtColor(resultat, cv2.COLOR_BGR2RGB), caption=f"Résultat: {total}")

# Historique des dernières photos (adaptatif)
if st.session_state.photos_prises:
    with st.expander("📜 Historique des dernières analyses"):
        for i, photo in enumerate(reversed(st.session_state.photos_prises[-5:])):
            st.write(f"Photo {i+1}: {photo['total']} pièces")
            if not st.session_state.is_mobile:
                st.write(f"   Couleurs: {dict(photo['stats_couleur'])}")

# Pied de page adaptatif
st.markdown("---")
st.caption(f"""
🧩 Compteur de Pièces v4.0 - Interface adaptative
• Mode {'📱 Téléphone' if st.session_state.is_mobile else '💻 PC'} détecté automatiquement
• Utilisez {'l\'appareil photo' if st.session_state.is_mobile else 'la caméra ou OneDrive'}
""")
