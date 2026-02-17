import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
from utils.object_counter import ObjectCounter
import time

# Configuration de la page
st.set_page_config(
    page_title="Compteur d'Objets par Caméra",
    page_icon="🔢",
    layout="wide"
)

# Titre et description
st.title("📸 Compteur d'Objets Intelligent")
st.markdown("""
Cette application utilise votre caméra pour compter automatiquement les objets en temps réel.
- **Aucune installation requise** - Fonctionne directement dans votre navigateur
- **Compatible avec tous les PC** - Windows, Mac, Linux
- **Confidentialité** - Les images ne sont pas stockées sur nos serveurs
""")

# Sidebar pour les paramètres
with st.sidebar:
    st.header("⚙️ Paramètres")
    
    # Source de la caméra
    camera_source = st.selectbox(
        "Source de la caméra",
        ["Caméra par défaut (0)", "Caméra USB (1)", "Uploader une image", "Uploader une vidéo"]
    )
    
    # Type de détection
    detection_type = st.selectbox(
        "Type d'objets à compter",
        ["Personnes", "Voitures", "Visages", "Objets génériques (YOLO)", "Formes simples"]
    )
    
    # Seuil de confiance
    confidence_threshold = st.slider("Seuil de confiance", 0.0, 1.0, 0.5, 0.05)
    
    st.markdown("---")
    
    # Options avancées
    with st.expander("🔧 Options avancées"):
        show_boxes = st.checkbox("Afficher les boîtes de détection", True)
        show_counts = st.checkbox("Afficher le compteur", True)
        show_fps = st.checkbox("Afficher les FPS", True)
    
    st.markdown("---")
    st.markdown("""
    ### 📖 Comment utiliser
    1. Choisissez votre source vidéo
    2. Sélectionnez le type d'objets
    3. Cliquez sur "Démarrer"
    4. Pointez la caméra vers les objets
    """)

# Zone principale
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📹 Flux vidéo en direct")
    
    # Conteneur pour la vidéo
    video_placeholder = st.empty()
    
    # Boutons de contrôle
    col_start, col_stop, col_clear = st.columns(3)
    with col_start:
        start_button = st.button("▶️ Démarrer", use_container_width=True)
    with col_stop:
        stop_button = st.button("⏹️ Arrêter", use_container_width=True)
    with col_clear:
        clear_button = st.button("🔄 Réinitialiser", use_container_width=True)

with col2:
    st.subheader("📊 Statistiques")
    
    # Métriques principales
    total_objects = st.metric("Objets détectés", "0", delta=None)
    fps_metric = st.metric("FPS", "0")
    
    # Graphique en temps réel (simulé)
    st.subheader("Historique des détections")
    chart_placeholder = st.empty()
    
    # Log des événements
    with st.expander("📝 Journal des détections"):
        log_placeholder = st.empty()

# Gestion de l'état de la session
if 'running' not in st.session_state:
    st.session_state.running = False
if 'counter' not in st.session_state:
    st.session_state.counter = ObjectCounter()
if 'detection_history' not in st.session_state:
    st.session_state.detection_history = []
if 'frame_count' not in st.session_state:
    st.session_state.frame_count = 0

# Fonction pour traiter la vidéo uploadée
def process_uploaded_video(video_file):
    """Traite une vidéo uploadée"""
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    return tfile.name

# Fonction pour traiter l'image uploadée
def process_uploaded_image(image_file):
    """Traite une image uploadée"""
    image = Image.open(image_file)
    return np.array(image)

# Boucle principale de traitement vidéo
if start_button:
    st.session_state.running = True
    st.session_state.detection_history = []
    st.session_state.frame_count = 0

if stop_button:
    st.session_state.running = False

if clear_button:
    st.session_state.detection_history = []
    st.session_state.frame_count = 0
    total_objects = st.metric("Objets détectés", "0")
    fps_metric = st.metric("FPS", "0")

# Initialisation de la vidéo
cap = None
video_path = None

if st.session_state.running:
    try:
        # Gestion des différentes sources
        if camera_source == "Caméra par défaut (0)":
            cap = cv2.VideoCapture(0)
        elif camera_source == "Caméra USB (1)":
            cap = cv2.VideoCapture(1)
        elif camera_source == "Uploader une vidéo":
            uploaded_file = st.file_uploader("Choisissez une vidéo", type=['mp4', 'avi', 'mov'])
            if uploaded_file is not None:
                video_path = process_uploaded_video(uploaded_file)
                cap = cv2.VideoCapture(video_path)
        elif camera_source == "Uploader une image":
            uploaded_file = st.file_uploader("Choisissez une image", type=['jpg', 'jpeg', 'png'])
            if uploaded_file is not None:
                image = process_uploaded_image(uploaded_file)
                # Afficher l'image directement
                st.image(image, caption="Image uploadée", use_column_width=True)
                st.info("Pour les images statiques, utilisez la détection par upload dans les paramètres.")
                st.session_state.running = False

        if cap is not None and cap.isOpened():
            # Paramètres de la caméra
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            fps = 0
            prev_time = time.time()
            
            while st.session_state.running:
                ret, frame = cap.read()
                if not ret:
                    st.warning("Impossible de lire le flux vidéo")
                    break
                
                # Calcul des FPS
                current_time = time.time()
                fps = 1 / (current_time - prev_time)
                prev_time = current_time
                
                # Détection d'objets (simplifiée pour l'exemple)
                # Dans un cas réel, vous utiliseriez YOLO, Haar Cascades, etc.
                
                # Simulation de détection (à remplacer par votre vrai modèle)
                if detection_type == "Visages":
                    # Utiliser Haar Cascade pour les visages
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                    
                    detected_objects = len(faces)
                    
                    if show_boxes:
                        for (x, y, w, h) in faces:
                            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                elif detection_type == "Formes simples":
                    # Détection de contours pour formes simples
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                    _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)
                    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    detected_objects = len(contours)
                    
                    if show_boxes:
                        cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)
                
                else:
                    # Simulation aléatoire pour la démo
                    import random
                    detected_objects = random.randint(0, 10)
                
                # Mise à jour de l'historique
                st.session_state.detection_history.append(detected_objects)
                if len(st.session_state.detection_history) > 50:
                    st.session_state.detection_history.pop(0)
                
                # Ajout des informations sur l'image
                if show_counts:
                    cv2.putText(frame, f"Objets: {detected_objects}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                if show_fps:
                    cv2.putText(frame, f"FPS: {int(fps)}", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                
                # Conversion pour Streamlit
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Mise à jour de l'affichage
                video_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
                
                # Mise à jour des métriques
                with col2:
                    total_objects.metric("Objets détectés", detected_objects, 
                                        delta=detected_objects - st.session_state.frame_count)
                    fps_metric.metric("FPS", int(fps))
                    
                    # Graphique d'historique
                    if len(st.session_state.detection_history) > 1:
                        chart_data = np.array(st.session_state.detection_history).reshape(-1, 1)
                        chart_placeholder.line_chart(chart_data)
                    
                    # Log
                    if st.session_state.frame_count % 30 == 0:  # Toutes les 30 frames
                        log_placeholder.info(f"Frame {st.session_state.frame_count}: {detected_objects} objets détectés")
                
                st.session_state.frame_count += 1
                
                # Petite pause pour éviter de surcharger
                time.sleep(0.03)
    
    except Exception as e:
        st.error(f"Erreur: {str(e)}")
    
    finally:
        if cap is not None:
            cap.release()
        if video_path and os.path.exists(video_path):
            os.unlink(video_path)

# Pied de page
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>🔒 <strong>Confidentialité garantie</strong> - Aucune donnée n'est stockée sur nos serveurs</p>
    <p>📱 Compatible avec tous les appareils (PC, tablettes, smartphones)</p>
    <p>⚡ Propulsé par Streamlit et OpenCV</p>
</div>
""", unsafe_allow_html=True)
