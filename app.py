import streamlit as st
import cv2
import numpy as np
from collections import defaultdict
from PIL import Image
import time

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

# Interface Streamlit
st.title("🧩 Compteur de Pièces")
st.markdown("""
Cette application détecte et compte automatiquement les pièces :
- **Détection par couleur** (rouge, bleu, vert, jaune)
- **Classification par taille** (P, M, G, TG)
- **Fonctionne sur téléphone et PC**
""")

# Sidebar pour les paramètres
with st.sidebar:
    st.header("⚙️ Configuration")
    
    source = st.radio(
        "Source",
        ["📸 Prendre une photo", "🖼️ Uploader une image", "🧪 Mode démo"]
    )
    
    st.markdown("---")
    st.header("📊 Statistiques")
    
    if st.button("🔄 Réinitialiser compteurs"):
        st.session_state.compteur.reset_compteur()
        st.session_state.frame_count = 0
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
    st.subheader("📸 Prenez une photo avec votre caméra")
    
    # Instructions pour le téléphone
    st.info("""
    **📱 Sur téléphone :**
    1. Appuyez sur le bouton ci-dessous
    2. Autorisez l'accès à la caméra si demandé
    3. Prenez la photo
    """)
    
    # Widget caméra avec paramètres adaptés au mobile
    img_file = st.camera_input(
        "Prendre une photo",
        key="camera_phone",
        help="Appuyez pour prendre une photo"
    )
    
    if img_file is not None:
        with st.spinner("🔍 Analyse en cours..."):
            try:
                # Lire l'image avec PIL d'abord (plus compatible)
                pil_image = Image.open(img_file)
                
                # Convertir en numpy array
                frame = np.array(pil_image)
                
                # Convertir RGB -> BGR pour OpenCV si nécessaire
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # Redimensionner si trop grande pour éviter les lenteurs
                height, width = frame.shape[:2]
                if width > 800:
                    scale = 800 / width
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    frame = cv2.resize(frame, (new_width, new_height))
                
                # Traitement
                resultat, pieces, stats_couleur, stats_taille, total_actuel = st.session_state.compteur.traiter_frame(frame)
                st.session_state.frame_count += 1
                
                # Affichage des résultats
                st.success(f"✅ **{total_actuel} pièces** détectées !")
                
                # Afficher les images
                col_img1, col_img2 = st.columns(2)
                with col_img1:
                    # Reconvertir BGR -> RGB pour affichage
                    display_original = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    st.image(display_original, caption="📸 Photo originale", use_column_width=True)
                with col_img2:
                    display_result = cv2.cvtColor(resultat, cv2.COLOR_BGR2RGB)
                    st.image(display_result, caption=f"🎯 {total_actuel} pièces", use_column_width=True)
                
                # Statistiques détaillées
                st.subheader("📊 Détail par couleur et taille")
                
                # Métriques principales
                col_m1, col_m2, col_m3 = st.columns(3)
                with col_m1:
                    st.metric("Total pièces", total_actuel)
                with col_m2:
                    st.metric("Couleurs diff.", len([c for c in stats_couleur.values() if c > 0]))
                with col_m3:
                    st.metric("Frame", st.session_state.frame_count)
                
                # Tableau des couleurs adapté au mobile
                st.write("**🎨 Répartition par couleur :**")
                
                # Version responsive pour mobile
                if st.session_state.get('mobile', False):
                    # Affichage vertical pour mobile
                    for couleur in ['rouge', 'bleu', 'vert', 'jaune', '?']:
                        count = stats_couleur.get(couleur, 0)
                        emoji = {'rouge': '🔴', 'bleu': '🔵', 'vert': '🟢', 'jaune': '🟡', '?': '⚪'}
                        st.write(f"{emoji.get(couleur, '⚪')} {couleur}: {count}")
                else:
                    # Affichage horizontal pour PC
                    cols = st.columns(5)
                    couleurs_list = ['rouge', 'bleu', 'vert', 'jaune', 'autre']
                    color_emoji = {'rouge': '🔴', 'bleu': '🔵', 'vert': '🟢', 'jaune': '🟡', 'autre': '⚪'}
                    
                    for i, couleur in enumerate(couleurs_list):
                        with cols[i]:
                            count = stats_couleur.get(couleur if couleur != 'autre' else '?', 0)
                            st.metric(f"{color_emoji[couleur]}", count)
                
                # Tableau des tailles adapté au mobile
                st.write("**📏 Répartition par taille :**")
                
                if st.session_state.get('mobile', False):
                    # Affichage vertical pour mobile
                    for taille in ['P', 'M', 'G', 'TG']:
                        count = stats_taille.get(taille, 0)
                        st.write(f"Taille {taille}: {count}")
                else:
                    cols = st.columns(4)
                    tailles_list = ['P', 'M', 'G', 'TG']
                    for i, taille in enumerate(tailles_list):
                        with cols[i]:
                            count = stats_taille.get(taille, 0)
                            st.metric(f"{taille}", count)
                
                # Détail des pièces
                with st.expander("🔍 Voir le détail de chaque pièce"):
                    for i, piece in enumerate(pieces, 1):
                        st.write(f"Pièce #{i} : {piece['couleur']} - {piece['taille']} (aire: {piece['aire']:.0f} px)")
            
            except Exception as e:
                st.error(f"Erreur lors de l'analyse: {str(e)}")
                st.info("Essayez de prendre une autre photo avec un meilleur éclairage")

elif source == "🖼️ Uploader une image":
    st.subheader("🖼️ Analyse d'image")
    
    uploaded_file = st.file_uploader("Choisissez une image", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file:
        with st.spinner("🔍 Analyse en cours..."):
            try:
                # Lire l'image
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                
                # Redimensionner si trop grande
                height, width = frame.shape[:2]
                if width > 800:
                    scale = 800 / width
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    frame = cv2.resize(frame, (new_width, new_height))
                
                # Traitement
                resultat, pieces, stats_couleur, stats_taille, total_actuel = st.session_state.compteur.traiter_frame(frame)
                st.session_state.frame_count += 1
                
                # Affichage
                st.success(f"✅ **{total_actuel} pièces** détectées !")
                
                col_img1, col_img2 = st.columns(2)
                with col_img1:
                    st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 
                            caption="🖼️ Originale", use_column_width=True)
                with col_img2:
                    st.image(cv2.cvtColor(resultat, cv2.COLOR_BGR2RGB), 
                            caption=f"🎯 {total_actuel} pièces", use_column_width=True)
                
                # Statistiques simplifiées pour mobile
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
            
            except Exception as e:
                st.error(f"Erreur: {str(e)}")

else:  # Mode démo
    st.subheader("🧪 Mode démo")
    st.info("Génération d'images de test pour démonstration")
    
    if st.button("🎲 Générer une image de test"):
        with st.spinner("🔍 Analyse..."):
            # Créer une image de test
            test_img = np.zeros((480, 640, 3), dtype=np.uint8)
            test_img.fill(255)
            
            # Dessiner des pièces
            cv2.circle(test_img, (200, 200), 50, (0, 0, 255), -1)  # Rouge
            cv2.circle(test_img, (350, 250), 40, (255, 0, 0), -1)  # Bleu
            cv2.circle(test_img, (500, 200), 45, (0, 255, 0), -1)  # Vert
            cv2.circle(test_img, (300, 350), 35, (0, 255, 255), -1)  # Jaune
            
            # Traitement
            resultat, pieces, stats_couleur, stats_taille, total_actuel = st.session_state.compteur.traiter_frame(test_img)
            
            # Affichage
            st.success(f"✅ **{total_actuel} pièces** détectées !")
            
            col_img1, col_img2 = st.columns(2)
            with col_img1:
                st.image(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB), 
                        caption="🧪 Test", use_column_width=True)
            with col_img2:
                st.image(cv2.cvtColor(resultat, cv2.COLOR_BGR2RGB), 
                        caption=f"🎯 {total_actuel}", use_column_width=True)

# Détection mobile (approximative)
import streamlit as st
if 'mobile' not in st.session_state:
    # Vérifier si l'écran est petit (probablement mobile)
    try:
        import streamlit.components.v1 as components
        mobile_script = """
        <script>
            const isMobile = /iPhone|iPad|iPod|Android/i.test(navigator.userAgent);
            window.parent.postMessage({isMobile: isMobile}, '*');
        </script>
        """
        components.html(mobile_script, height=0)
        st.session_state.mobile = False  # Par défaut
    except:
        st.session_state.mobile = False

# Pied de page
st.markdown("---")
st.caption("""
🧩 Compteur de Pièces v2.1 - Optimisé mobile
• Compatible iPhone et Android
• Redimensionnement automatique
• Interface adaptative
""")
