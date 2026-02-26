import streamlit as st
import cv2
import numpy as np
from collections import defaultdict
from datetime import datetime
import base64
from io import BytesIO
import openpyxl
from openpyxl.styles import Font, Alignment, PatternFill
from pyzbar.pyzbar import decode
import re
import time
from threading import Thread
import platform
import subprocess

# ==================== Dictionnaire des articles prédéfinis avec leurs emplacements ====================
ARTICLES_PREDEFINIS = {
    "10751037": {
        "libelle": "Capacitor E54.G85-203G30 Un 1260 V DC / 750 AC MKP 20µF",
        "emplacement": "A191"
    },
    "10751038": {
        "libelle": "Contacteur principal Bipolaire",
        "emplacement": "A204"
    },
    "10751039": {
        "libelle": "Contacteur de précharge Bipolaire",
        "emplacement": "A204"
    },
    "10751040": {
        "libelle": "Coupe circuit 1A, 480VAC, 3Poles",
        "emplacement": "A192"
    },
    "10751050": {
        "libelle": "Cosse à sertir 50x8",
        "emplacement": "A194"
    }
}
# =====================================================================================================

# Configuration de la page
st.set_page_config(
    page_title="Gestionnaire d'Inventaire Multi-Pièces",
    page_icon="📦",
    layout="wide"
)

# CSS personnalisé
st.markdown("""
<style>
    .barcode-scanner {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 1rem;
    }
    .barcode-result {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
    }
    .success-box {
        background: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
    }
    .code-display {
        background: #2c3e50;
        color: #ecf0f1;
        padding: 1rem;
        border-radius: 5px;
        font-family: monospace;
        font-size: 1.5rem;
        text-align: center;
        margin: 1rem 0;
    }
    .location-badge {
        background: #17a2b8;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 5px;
        font-size: 0.8rem;
        margin-left: 0.5rem;
    }
    .label-badge {
        background: #28a745;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 5px;
        font-size: 0.8rem;
        margin-left: 0.5rem;
    }
    .article-found {
        background: #cce5ff;
        color: #004085;
        padding: 0.5rem;
        border-radius: 5px;
        border-left: 5px solid #004085;
        margin: 0.5rem 0;
    }
    .webcam-container {
        border: 3px solid #667eea;
        border-radius: 10px;
        padding: 10px;
        margin: 10px 0;
        background: #f8f9fa;
    }
    .stats-box {
        background: #f8f9fa;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #28a745;
        margin: 10px 0;
    }
    .diagnostic-box {
        background: #fff3cd;
        color: #856404;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #ffc107;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ==================== CLASSE POUR LA WEBCAM EN DIRECT ====================
class WebcamStream:
    def __init__(self, camera_id=0, backend=cv2.CAP_DSHOW):
        self.camera_id = camera_id
        self.backend = backend
        self.cap = None
        self.running = False
        self.frame = None
        self.nb_pieces = 0
        self.stats_couleur = defaultdict(int)
        self.stats_taille = defaultdict(int)
        self.fps = 0
        self.frame_count = 0
        self.last_time = time.time()
        
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
        
        self.seuils_taille = {
            'P': (0, 500),
            'M': (500, 2000),
            'G': (2000, 5000),
            'TG': (5000, float('inf'))
        }
    
    def get_couleur_piece(self, hsv, contour):
        """Détermine la couleur d'une pièce"""
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        
        best_couleur = '?'
        best_score = 0
        
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
        
        return best_couleur
    
    def get_taille_piece(self, aire):
        """Détermine la taille d'une pièce"""
        for nom_taille, (min_vol, max_vol) in self.seuils_taille.items():
            if min_vol <= aire < max_vol:
                return nom_taille
        return '?'
    
    def detecter_pieces_live(self, frame):
        """Détecte et compte les pièces dans une frame en direct"""
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
        
        pieces_valides = []
        stats_couleur_actuelles = defaultdict(int)
        stats_taille_actuelles = defaultdict(int)
        
        for contour in contours:
            aire = cv2.contourArea(contour)
            if aire < 200:
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            
            couleur_nom = self.get_couleur_piece(hsv, contour)
            taille_nom = self.get_taille_piece(aire)
            
            pieces_valides.append(contour)
            stats_couleur_actuelles[couleur_nom] += 1
            stats_taille_actuelles[taille_nom] += 1
            
            # Dessiner la pièce
            couleur_bbox = self.couleurs.get(couleur_nom, {}).get('couleur_bbox', (128, 128, 128))
            cv2.rectangle(resultat, (x, y), (x+w, y+h), couleur_bbox, 2)
            cv2.circle(resultat, (x + w//2, y + h//2), 3, (255, 255, 255), -1)
        
        nb_pieces = len(pieces_valides)
        
        # Ajouter les informations sur l'image
        h, w = resultat.shape[:2]
        
        # TOTAL ACTUEL
        cv2.putText(resultat, f"Pieces: {nb_pieces}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Statistiques
        y_stats = 60
        cv2.putText(resultat, f"Couleurs: R:{stats_couleur_actuelles.get('rouge',0)} B:{stats_couleur_actuelles.get('bleu',0)} V:{stats_couleur_actuelles.get('vert',0)} J:{stats_couleur_actuelles.get('jaune',0)}", 
                   (10, y_stats), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        y_stats += 20
        cv2.putText(resultat, f"Tailles: P:{stats_taille_actuelles.get('P',0)} M:{stats_taille_actuelles.get('M',0)} G:{stats_taille_actuelles.get('G',0)} TG:{stats_taille_actuelles.get('TG',0)}", 
                   (10, y_stats), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # FPS
        self.frame_count += 1
        current_time = time.time()
        if current_time - self.last_time >= 1.0:
            self.fps = self.frame_count
            self.frame_count = 0
            self.last_time = current_time
        
        cv2.putText(resultat, f"FPS: {self.fps}", (w-100, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return resultat, nb_pieces, stats_couleur_actuelles, stats_taille_actuelles
    
    def start(self):
        """Démarre le flux webcam avec différents backends"""
        backends_to_try = [
            (self.backend, "Backend spécifié"),
            (cv2.CAP_DSHOW, "DirectShow"),
            (cv2.CAP_MSMF, "Media Foundation"),
            (cv2.CAP_ANY, "Auto"),
        ]
        
        for backend, backend_name in backends_to_try:
            try:
                self.cap = cv2.VideoCapture(self.camera_id, backend)
                if self.cap.isOpened():
                    # Tester la capture
                    ret, test_frame = self.cap.read()
                    if ret and test_frame is not None:
                        print(f"✅ Caméra {self.camera_id} ouverte avec {backend_name}")
                        
                        # Configuration
                        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        self.cap.set(cv2.CAP_PROP_FPS, 30)
                        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        
                        self.running = True
                        self.thread = Thread(target=self._update)
                        self.thread.daemon = True
                        self.thread.start()
                        return True
                    else:
                        self.cap.release()
            except Exception as e:
                print(f"Erreur avec backend {backend_name}: {e}")
                continue
        
        return False
    
    def _update(self):
        """Met à jour la frame en continu"""
        while self.running:
            try:
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    # Traiter la frame
                    resultat, nb_pieces, stats_couleur, stats_taille = self.detecter_pieces_live(frame)
                    self.frame = resultat
                    self.nb_pieces = nb_pieces
                    self.stats_couleur = stats_couleur
                    self.stats_taille = stats_taille
                time.sleep(0.03)  # ~30 FPS
            except Exception as e:
                print(f"Erreur dans _update: {e}")
                time.sleep(0.1)
    
    def read(self):
        """Lit la dernière frame traitée"""
        return self.frame
    
    def stop(self):
        """Arrête le flux"""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()

# ==================== FONCTIONS DE DIAGNOSTIC ====================
def get_system_info():
    """Récupère les informations système"""
    info = {
        "os": platform.system(),
        "os_version": platform.version(),
        "python_version": platform.python_version(),
        "opencv_version": cv2.__version__,
        "processor": platform.processor(),
        "machine": platform.machine()
    }
    return info

def test_single_camera(index, backend):
    """Teste une caméra spécifique avec un backend donné"""
    try:
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            # Obtenir les propriétés
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Lire une frame
            ret, frame = cap.read()
            cap.release()
            
            if ret and frame is not None:
                return {
                    "success": True,
                    "width": int(width),
                    "height": int(height),
                    "fps": fps,
                    "frame_shape": frame.shape
                }
            else:
                return {"success": False, "error": "Pas de flux vidéo"}
        else:
            return {"success": False, "error": "Non disponible"}
    except Exception as e:
        return {"success": False, "error": str(e)}

def diagnostic_cameras():
    """Outil de diagnostic complet des caméras"""
    with st.expander("🔧 Diagnostic des caméras", expanded=False):
        st.markdown('<div class="diagnostic-box">', unsafe_allow_html=True)
        
        # Informations système
        sys_info = get_system_info()
        st.write("### 💻 Informations système")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(f"**OS:** {sys_info['os']}")
            st.write(f"**Python:** {sys_info['python_version']}")
        with col2:
            st.write(f"**OpenCV:** {sys_info['opencv_version']}")
            st.write(f"**Machine:** {sys_info['machine']}")
        with col3:
            st.write("**Backends disponibles:**")
            st.write("- CAP_DSHOW (DirectShow)")
            st.write("- CAP_MSMF (Media Foundation)")
            st.write("- CAP_ANY (Auto)")
        
        st.divider()
        
        # Test automatique
        st.write("### 🔍 Scan automatique des caméras")
        
        if st.button("🔍 Lancer le diagnostic complet", use_container_width=True):
            with st.spinner("Scan des caméras en cours..."):
                results = []
                backends = [
                    (cv2.CAP_DSHOW, "DirectShow"),
                    (cv2.CAP_MSMF, "Media Foundation"),
                    (cv2.CAP_ANY, "Auto"),
                ]
                
                for backend, backend_name in backends:
                    st.write(f"**Test avec {backend_name}:**")
                    cols = st.columns(5)
                    for i in range(5):
                        result = test_single_camera(i, backend)
                        with cols[i]:
                            if result["success"]:
                                st.success(f"✅ Caméra {i}")
                                st.caption(f"{result['width']}x{result['height']}")
                            else:
                                st.error(f"❌ Caméra {i}")
        
        # Test manuel
        st.write("### 🎯 Test manuel")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            test_index = st.number_input("Index à tester", min_value=0, max_value=10, value=0)
        with col2:
            test_backend = st.selectbox(
                "Backend",
                ["DirectShow", "Media Foundation", "Auto"],
                index=0
            )
            backend_map = {
                "DirectShow": cv2.CAP_DSHOW,
                "Media Foundation": cv2.CAP_MSMF,
                "Auto": cv2.CAP_ANY
            }
        
        with col3:
            if st.button("Tester", use_container_width=True):
                result = test_single_camera(test_index, backend_map[test_backend])
                if result["success"]:
                    st.success(f"✅ Caméra {test_index} OK")
                    st.json(result)
                else:
                    st.error(f"❌ Caméra {test_index}: {result.get('error', 'Inconnu')}")
        
        # Conseils
        st.divider()
        st.write("### 💡 Conseils de dépannage")
        st.markdown("""
        1. **Vérifiez les permissions** : 
           - Windows : Paramètres > Confidentialité > Caméra > Autoriser les applications
           
        2. **Vérifiez que la caméra n'est pas utilisée** :
           - Fermez les autres applications (Teams, Zoom, etc.)
           
        3. **Testez avec l'application native** :
           - Ouvrez l'application "Appareil photo" de Windows
           
        4. **Mettez à jour les pilotes** :
           - Gestionnaire de périphériques > Caméras > Mettre à jour
           
        5. **Essayez différents indices** :
           - 0 = Webcam intégrée
           - 1 = Caméra externe
           - 2+ = Autres périphériques
        """)
        
        st.markdown('</div>', unsafe_allow_html=True)

# ==================== FONCTIONS EXISTANTES ====================
def detecter_code_barre(image):
    """Détecte et lit les codes-barres dans une image"""
    resultat = image.copy()
    codes_detectes = []
    
    # Conversion en niveaux de gris
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Décoder les codes-barres
    codes = decode(gray)
    
    for code in codes:
        # Extraire les données
        data = code.data.decode('utf-8')
        type_code = code.type
        
        # Dessiner le rectangle autour du code
        points = code.polygon
        if len(points) == 4:
            pts = np.array([(p.x, p.y) for p in points], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(resultat, [pts], True, (0, 255, 0), 3)
        
        # Ajouter le texte
        cv2.putText(resultat, f"{type_code}: {data}", 
                   (code.rect.left, code.rect.top - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        codes_detectes.append({
            'data': data,
            'type': type_code
        })
    
    return resultat, codes_detectes

def detecter_pieces(image):
    """Détecte et compte les pièces dans une image"""
    resultat = image.copy()
    
    # Conversion en niveaux de gris
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Flou pour réduire le bruit
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Détection des contours
    edges = cv2.Canny(blur, 50, 150)
    
    # Dilatation et érosion
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)
    edges = cv2.erode(edges, kernel, iterations=1)
    
    # Trouver les contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filtrer les petits contours (bruit)
    pieces_valides = []
    for contour in contours:
        aire = cv2.contourArea(contour)
        if aire > 200:  # Seuil minimum
            pieces_valides.append(contour)
    
    nb_pieces = len(pieces_valides)
    
    # Dessiner les contours
    for contour in pieces_valides:
        # Dessiner le contour en vert
        cv2.drawContours(resultat, [contour], -1, (0, 255, 0), 2)
        
        # Ajouter un point au centre
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.circle(resultat, (cx, cy), 3, (0, 0, 255), -1)
    
    # Ajouter le compteur
    cv2.putText(resultat, f"Pieces: {nb_pieces}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    return resultat, nb_pieces

def base64_to_image(base64_string):
    """Convertit une chaîne base64 en image OpenCV"""
    img_data = base64.b64decode(base64_string)
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def webcam_section():
    """Section webcam avec OpenCV direct et diagnostic"""
    st.markdown('<div class="webcam-container">', unsafe_allow_html=True)
    st.markdown("### 📷 Webcam en direct - Comptage automatique")
    
    # Ajouter le diagnostic
    diagnostic_cameras()
    
    # Initialisation dans session_state
    if 'webcam_stream' not in st.session_state:
        st.session_state.webcam_stream = None
    if 'webcam_active' not in st.session_state:
        st.session_state.webcam_active = False
    if 'camera_index' not in st.session_state:
        st.session_state.camera_index = 0
    
    # Interface de contrôle
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    
    with col1:
        # Choix du backend
        backend_option = st.selectbox(
            "Backend",
            ["DirectShow", "Media Foundation", "Auto"],
            index=0,
            key="backend_select"
        )
        
        backend_map = {
            "DirectShow": cv2.CAP_DSHOW,
            "Media Foundation": cv2.CAP_MSMF,
            "Auto": cv2.CAP_ANY
        }
        backend = backend_map[backend_option]
    
    with col2:
        camera_id = st.number_input("Index caméra", 
                                   min_value=0, 
                                   max_value=10, 
                                   value=st.session_state.camera_index,
                                   step=1,
                                   key="camera_input")
        st.session_state.camera_index = camera_id
    
    with col3:
        if not st.session_state.webcam_active:
            if st.button("▶️ Démarrer", use_container_width=True):
                stream = WebcamStream(camera_id, backend)
                if stream.start():
                    st.session_state.webcam_stream = stream
                    st.session_state.webcam_active = True
                    st.rerun()
                else:
                    st.error(f"❌ Impossible d'ouvrir la caméra {camera_id}")
        else:
            if st.button("⏹️ Arrêter", use_container_width=True):
                if st.session_state.webcam_stream:
                    st.session_state.webcam_stream.stop()
                st.session_state.webcam_stream = None
                st.session_state.webcam_active = False
                st.rerun()
    
    with col4:
        if st.session_state.webcam_active:
            st.success("✅ Active")
    
    # Affichage du flux
    if st.session_state.webcam_active and st.session_state.webcam_stream:
        # Créer un placeholder pour l'image
        image_placeholder = st.empty()
        stats_placeholder = st.empty()
        
        frame = st.session_state.webcam_stream.read()
        if frame is not None:
            # Afficher l'image
            image_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 
                                   channels="RGB", use_column_width=True)
            
            # Statistiques en direct
            stream = st.session_state.webcam_stream
            with stats_placeholder.container():
                col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                
                with col_s1:
                    st.metric("Total pièces", stream.nb_pieces)
                with col_s2:
                    rouge = stream.stats_couleur.get('rouge', 0)
                    bleu = stream.stats_couleur.get('bleu', 0)
                    st.metric("Rouge/Bleu", f"{rouge}/{bleu}")
                with col_s3:
                    vert = stream.stats_couleur.get('vert', 0)
                    jaune = stream.stats_couleur.get('jaune', 0)
                    st.metric("Vert/Jaune", f"{vert}/{jaune}")
                with col_s4:
                    st.metric("FPS", stream.fps)
                
                # Option pour capturer
                if st.button("📸 Capturer cette image pour l'article"):
                    # Convertir l'image en base64 pour la sauvegarde
                    _, buffer = cv2.imencode('.jpg', frame)
                    img_base64 = base64.b64encode(buffer).decode('utf-8')
                    st.session_state.derniere_capture = {
                        'image': img_base64,
                        'nb_pieces': stream.nb_pieces,
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    st.success(f"✅ Image capturée avec {stream.nb_pieces} pièces!")
            
            # Refresh automatique
            time.sleep(0.1)
            st.rerun()
    else:
        st.info("👆 Cliquez sur 'Démarrer' pour activer la webcam")
        st.info("💡 Si la caméra ne fonctionne pas, utilisez l'outil de diagnostic ci-dessus")
    
    st.markdown('</div>', unsafe_allow_html=True)

# ==================== CLASSE GESTIONNAIRE ====================
class GestionnairePieces:
    def __init__(self):
        """Initialise le gestionnaire de pièces"""
        self.articles = {}  # Dictionnaire {code_article: {"libelle": "", "photos": [], "emplacement": ""}}
    
    def creer_nouvel_article(self, code_article, libelle="", emplacement=""):
        """Crée un nouvel article dans l'inventaire avec son libellé et emplacement"""
        if code_article and code_article not in self.articles:
            # Si le code existe dans les prédéfinis et que le libellé ou l'emplacement sont vides, on les remplit
            if code_article in ARTICLES_PREDEFINIS:
                if not libelle:
                    libelle = ARTICLES_PREDEFINIS[code_article]["libelle"]
                if not emplacement:
                    emplacement = ARTICLES_PREDEFINIS[code_article]["emplacement"]
            
            self.articles[code_article] = {
                'libelle': libelle,
                'photos': [],
                'emplacement': emplacement,
                'date_creation': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            return True
        return False
    
    def ajouter_photo_article(self, code_article, frame_original, frame_analyse, nb_pieces):
        """Ajoute une photo analysée à un article existant"""
        if code_article in self.articles:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Convertir les images en base64
            _, buffer_original = cv2.imencode('.jpg', frame_original)
            _, buffer_analyse = cv2.imencode('.jpg', frame_analyse)
            
            photo_data = {
                'timestamp': timestamp,
                'nb_pieces': nb_pieces,
                'image_originale': base64.b64encode(buffer_original).decode('utf-8'),
                'image_analyse': base64.b64encode(buffer_analyse).decode('utf-8'),
                'id': len(self.articles[code_article]['photos'])
            }
            
            self.articles[code_article]['photos'].append(photo_data)
            return True
        return False
    
    def ajouter_capture_webcam(self, code_article, capture_data):
        """Ajoute une capture depuis la webcam"""
        if code_article in self.articles and capture_data:
            # Convertir base64 en image
            img_data = base64.b64decode(capture_data['image'])
            frame = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
            
            # Analyser l'image
            resultat, nb_pieces = detecter_pieces(frame)
            
            # Ajouter la photo
            return self.ajouter_photo_article(code_article, frame, resultat, nb_pieces)
        return False
    
    def get_total_article(self, code_article):
        """Retourne le total de pièces pour un article donné"""
        if code_article in self.articles:
            return sum(photo['nb_pieces'] for photo in self.articles[code_article]['photos'])
        return 0
    
    def get_photos_article(self, code_article):
        """Retourne toutes les photos d'un article"""
        if code_article in self.articles:
            return self.articles[code_article]['photos']
        return []
    
    def get_emplacement_article(self, code_article):
        """Retourne l'emplacement d'un article"""
        if code_article in self.articles:
            return self.articles[code_article].get('emplacement', '')
        return ''
    
    def get_libelle_article(self, code_article):
        """Retourne le libellé d'un article"""
        if code_article in self.articles:
            return self.articles[code_article].get('libelle', '')
        return ''
    
    def supprimer_photo(self, code_article, photo_id):
        """Supprime une photo d'un article"""
        if code_article in self.articles and 0 <= photo_id < len(self.articles[code_article]['photos']):
            del self.articles[code_article]['photos'][photo_id]
            # Réindexer les IDs
            for i, photo in enumerate(self.articles[code_article]['photos']):
                photo['id'] = i
            return True
        return False
    
    def supprimer_article(self, code_article):
        """Supprime complètement un article"""
        if code_article in self.articles:
            del self.articles[code_article]
            return True
        return False
    
    def get_tous_les_totaux(self):
        """Retourne un dictionnaire avec tous les totaux par article"""
        return {code: self.get_total_article(code) for code in self.articles}
    
    def get_tous_emplacements(self):
        """Retourne un dictionnaire avec tous les emplacements par article"""
        return {code: self.get_emplacement_article(code) for code in self.articles}
    
    def get_tous_libelles(self):
        """Retourne un dictionnaire avec tous les libellés par article"""
        return {code: self.get_libelle_article(code) for code in self.articles}
    
    def generer_excel(self):
        """Génère un fichier Excel avec l'inventaire complet"""
        # Créer un nouveau classeur Excel
        output = BytesIO()
        workbook = openpyxl.Workbook()
        
        # Feuille principale - Résumé
        sheet_resume = workbook.active
        sheet_resume.title = "Inventaire"
        
        # En-têtes
        headers = ["Code Article", "Libellé", "Emplacement", "Quantité totale", "Nombre de photos", "Dernière mise à jour"]
        for col, header in enumerate(headers, 1):
            cell = sheet_resume.cell(row=1, column=col)
            cell.value = header
            cell.font = Font(bold=True)
            cell.fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
            cell.font = Font(color="FFFFFF", bold=True)
            cell.alignment = Alignment(horizontal="center")
        
        # Données du résumé
        row = 2
        for code_article, data in self.articles.items():
            total = sum(p['nb_pieces'] for p in data['photos'])
            nb_photos = len(data['photos'])
            derniere_date = data['photos'][-1]['timestamp'] if data['photos'] else data.get('date_creation', 'N/A')
            emplacement = data.get('emplacement', '')
            libelle = data.get('libelle', '')
            
            sheet_resume.cell(row=row, column=1).value = code_article
            sheet_resume.cell(row=row, column=2).value = libelle
            sheet_resume.cell(row=row, column=3).value = emplacement
            sheet_resume.cell(row=row, column=4).value = total
            sheet_resume.cell(row=row, column=5).value = nb_photos
            sheet_resume.cell(row=row, column=6).value = derniere_date
            row += 1
        
        # Ajuster la largeur des colonnes
        sheet_resume.column_dimensions['A'].width = 20
        sheet_resume.column_dimensions['B'].width = 30
        sheet_resume.column_dimensions['C'].width = 20
        sheet_resume.column_dimensions['D'].width = 15
        sheet_resume.column_dimensions['E'].width = 15
        sheet_resume.column_dimensions['F'].width = 22
        
        # Feuille de détail
        sheet_detail = workbook.create_sheet("Détail des photos")
        
        # En-têtes détail
        detail_headers = ["Code Article", "Libellé", "Emplacement", "Photo #", "Date", "Nombre de pièces"]
        for col, header in enumerate(detail_headers, 1):
            cell = sheet_detail.cell(row=1, column=col)
            cell.value = header
            cell.font = Font(bold=True)
            cell.fill = PatternFill(start_color="92D050", end_color="92D050", fill_type="solid")
            cell.alignment = Alignment(horizontal="center")
        
        # Données détaillées
        row = 2
        for code_article, data in self.articles.items():
            libelle = data.get('libelle', '')
            emplacement = data.get('emplacement', '')
            for i, photo in enumerate(data['photos'], 1):
                sheet_detail.cell(row=row, column=1).value = code_article
                sheet_detail.cell(row=row, column=2).value = libelle
                sheet_detail.cell(row=row, column=3).value = emplacement
                sheet_detail.cell(row=row, column=4).value = f"Photo {i}"
                sheet_detail.cell(row=row, column=5).value = photo['timestamp']
                sheet_detail.cell(row=row, column=6).value = photo['nb_pieces']
                row += 1
        
        # Ajuster les colonnes du détail
        sheet_detail.column_dimensions['A'].width = 20
        sheet_detail.column_dimensions['B'].width = 30
        sheet_detail.column_dimensions['C'].width = 20
        sheet_detail.column_dimensions['D'].width = 12
        sheet_detail.column_dimensions['E'].width = 22
        sheet_detail.column_dimensions['F'].width = 18
        
        workbook.save(output)
        output.seek(0)
        return output
    
    def reinitialiser_tout(self):
        """Réinitialise complètement l'inventaire"""
        self.articles = {}

# ==================== INITIALISATION ====================
if 'gestionnaire' not in st.session_state:
    st.session_state.gestionnaire = GestionnairePieces()
if 'page' not in st.session_state:
    st.session_state.page = "saisie"
if 'article_selectionne' not in st.session_state:
    st.session_state.article_selectionne = None
if 'photo_selectionnee' not in st.session_state:
    st.session_state.photo_selectionnee = None
if 'code_detecte' not in st.session_state:
    st.session_state.code_detecte = None
if 'scan_effectue' not in st.session_state:
    st.session_state.scan_effectue = False
if 'derniere_capture' not in st.session_state:
    st.session_state.derniere_capture = None

gestionnaire = st.session_state.gestionnaire

# ==================== INTERFACE PRINCIPALE ====================
st.title("📦 Gestionnaire d'Inventaire Multi-Pièces avec Webcam")
st.markdown("""
Cette application permet de gérer l'inventaire de plusieurs types de pièces :
1. **Scanner** un code-barres pour identifier automatiquement l'article
2. **Ajouter** un libellé descriptif (optionnel)
3. **Ajouter** un emplacement de stockage (optionnel)
4. **Utiliser la webcam** en direct pour compter les pièces
5. **Ajouter** plusieurs photos pour cet article
6. **Changer** d'article et répéter
7. **Exporter** un fichier Excel avec tous les totaux
""")

# Barre latérale avec la liste des articles
with st.sidebar:
    st.header("📋 Articles en inventaire")
    
    if gestionnaire.articles:
        # Afficher tous les articles avec leurs totaux, libellés et emplacements
        for code_article in gestionnaire.articles.keys():
            total = gestionnaire.get_total_article(code_article)
            libelle = gestionnaire.get_libelle_article(code_article)
            emplacement = gestionnaire.get_emplacement_article(code_article)
            
            with st.container():
                col1, col2 = st.columns([3, 1])
                with col1:
                    # Créer un bouton avec le code article
                    if st.button(f"📦 {code_article}", key=f"select_{code_article}", use_container_width=True):
                        st.session_state.article_selectionne = code_article
                        st.session_state.page = "details"
                with col2:
                    st.write(f"**{total}**")
            
            # Afficher les badges séparément
            if libelle or emplacement:
                badge_text = ""
                if libelle:
                    badge_text += f"📝 {libelle}"
                if libelle and emplacement:
                    badge_text += " | "
                if emplacement:
                    badge_text += f"📍 {emplacement}"
                
                if badge_text:
                    st.caption(badge_text)
        
        st.divider()
        
        # Bouton pour retourner à la saisie
        if st.button("➕ Nouvel article", use_container_width=True):
            st.session_state.page = "saisie"
            st.session_state.article_selectionne = None
            st.session_state.code_detecte = None
            st.session_state.scan_effectue = False
            st.rerun()
        
        st.divider()
        
        # Export Excel
        if gestionnaire.articles:
            st.header("📊 Export")
            excel_file = gestionnaire.generer_excel()
            st.download_button(
                label="📥 Télécharger Excel",
                data=excel_file,
                file_name=f"inventaire_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
            
            # Réinitialisation
            if st.button("🔄 Tout réinitialiser", type="primary", use_container_width=True):
                gestionnaire.reinitialiser_tout()
                st.session_state.page = "saisie"
                st.session_state.article_selectionne = None
                st.rerun()
    else:
        st.info("Aucun article pour le moment")

# Contenu principal
if st.session_state.page == "saisie":
    # Page de saisie d'un nouvel article avec scan de code-barres
    st.header("➕ Ajouter un nouvel article")
    
    # Section webcam en direct
    webcam_section()
    
    # Capture depuis la webcam
    if st.session_state.derniere_capture and st.session_state.article_selectionne:
        st.info(f"📸 Dernière capture : {st.session_state.derniere_capture['nb_pieces']} pièces")
        if st.button("✅ Ajouter cette capture à l'article"):
            if gestionnaire.ajouter_capture_webcam(st.session_state.article_selectionne, st.session_state.derniere_capture):
                st.success("✅ Capture ajoutée avec succès!")
                st.session_state.derniere_capture = None
                st.rerun()
    
    st.markdown("---")
    
    # Section scan de code-barres
    st.markdown('<div class="barcode-scanner">', unsafe_allow_html=True)
    st.markdown("### 📷 Scanner le code-barres de l'article")
    st.markdown("Prenez une photo du code-barres pour identifier automatiquement l'article")
    
    col_scan1, col_scan2 = st.columns(2)
    
    with col_scan1:
        scan_option = st.radio("Source", ["📸 Caméra", "🖼️ Upload"], horizontal=True, key="scan_source")
    
    if scan_option == "📸 Caméra":
        img_barcode = st.camera_input("Prendre une photo du code-barres", key="camera_barcode")
        if img_barcode:
            with st.spinner("🔍 Analyse du code-barres..."):
                bytes_data = img_barcode.getvalue()
                frame = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
                
                # Détection du code-barres
                image_annotee, codes = detecter_code_barre(frame)
                
                if codes:
                    # Prendre le premier code détecté
                    code_trouve = codes[0]['data']
                    st.session_state.code_detecte = code_trouve
                    st.session_state.scan_effectue = True
                    
                    # Afficher l'image avec le code détecté
                    st.image(cv2.cvtColor(image_annotee, cv2.COLOR_BGR2RGB), 
                            caption="Code-barres détecté", use_container_width=True)
                    
                    st.markdown(f"""
                    <div class="success-box">
                        <h4>✅ Code-barres détecté !</h4>
                        <div class="code-display">{code_trouve}</div>
                        <p><strong>Type :</strong> {codes[0]['type']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning("❌ Aucun code-barres détecté. Veuillez réessayer avec une image plus claire.")
    
    else:  # Upload
        uploaded_barcode = st.file_uploader("Choisir une image de code-barres", type=['jpg', 'jpeg', 'png'], key="upload_barcode")
        if uploaded_barcode:
            with st.spinner("🔍 Analyse du code-barres..."):
                file_bytes = np.asarray(bytearray(uploaded_barcode.read()), dtype=np.uint8)
                frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                
                # Détection du code-barres
                image_annotee, codes = detecter_code_barre(frame)
                
                # Afficher l'image
                st.image(cv2.cvtColor(image_annotee, cv2.COLOR_BGR2RGB), 
                        caption="Image analysée", use_container_width=True)
                
                if codes:
                    code_trouve = codes[0]['data']
                    st.session_state.code_detecte = code_trouve
                    st.session_state.scan_effectue = True
                    
                    st.markdown(f"""
                    <div class="success-box">
                        <h4>✅ Code-barres détecté !</h4>
                        <div class="code-display">{code_trouve}</div>
                        <p><strong>Type :</strong> {codes[0]['type']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning("❌ Aucun code-barres détecté. Veuillez réessayer avec une image plus claire.")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Bouton pour réinitialiser le scan
    if st.session_state.scan_effectue:
        if st.button("🔄 Nouveau scan", use_container_width=True):
            st.session_state.scan_effectue = False
            st.session_state.code_detecte = None
            st.rerun()
    
    st.markdown("---")
    
    # ==================== FORMULAIRE AVEC SAISIE AUTOMATIQUE DU LIBELLÉ ET DE L'EMPLACEMENT ====================
    st.markdown("### 📝 Informations de l'article")
    
    # Valeur par défaut pour le code (depuis le scan)
    default_code = st.session_state.code_detecte if st.session_state.code_detecte else ""
    
    # Trois colonnes pour le code, le libellé et l'emplacement
    col_code, col_lib, col_emp = st.columns([2, 2, 1])
    
    with col_code:
        code_article = st.text_input(
            "Code article *",
            value=default_code,
            placeholder="Code article (obligatoire)",
            key="code_article_input"
        )
    
    with col_lib:
        # Déterminer le libellé en fonction du code
        if code_article and code_article in ARTICLES_PREDEFINIS:
            libelle_value = ARTICLES_PREDEFINIS[code_article]["libelle"]
        else:
            libelle_value = ""
        
        libelle = st.text_input(
            "Libellé (optionnel)",
            value=libelle_value,
            placeholder="Description de l'article",
            key="libelle_input"
        )
    
    with col_emp:
        # Déterminer l'emplacement en fonction du code
        if code_article and code_article in ARTICLES_PREDEFINIS:
            emplacement_value = ARTICLES_PREDEFINIS[code_article]["emplacement"]
        else:
            emplacement_value = ""
        
        emplacement = st.text_input(
            "Emplacement (optionnel)",
            value=emplacement_value,
            placeholder="Ex: A-12, Rayon 3...",
            key="emplacement_input"
        )
    
    # Afficher le message si l'article est trouvé
    if code_article and code_article in ARTICLES_PREDEFINIS:
        st.markdown(f"""
        <div class="article-found">
            <strong>📝 Article trouvé :</strong> {ARTICLES_PREDEFINIS[code_article]["libelle"]}<br>
            <strong>📍 Emplacement :</strong> {ARTICLES_PREDEFINIS[code_article]["emplacement"]}
        </div>
        """, unsafe_allow_html=True)
    
    st.caption("* Champ obligatoire")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Créer l'article", use_container_width=True):
            if code_article:
                if gestionnaire.creer_nouvel_article(code_article, libelle, emplacement):
                    st.success(f"✅ Article '{code_article}' créé avec succès!")
                    if libelle:
                        st.info(f"📝 Libellé: {libelle}")
                    if emplacement:
                        st.info(f"📍 Emplacement: {emplacement}")
                    st.session_state.article_selectionne = code_article
                    st.session_state.page = "details"
                    st.session_state.code_detecte = None
                    st.session_state.scan_effectue = False
                    st.rerun()
                else:
                    if code_article in gestionnaire.articles:
                        st.error("❌ Ce code article existe déjà")
                    else:
                        st.error("❌ Erreur lors de la création de l'article")
            else:
                st.error("❌ Veuillez entrer un code article")
    with col2:
        if st.button("❌ Annuler", use_container_width=True):
            st.session_state.code_detecte = None
            st.session_state.scan_effectue = False
            st.rerun()

elif st.session_state.page == "details" and st.session_state.article_selectionne:
    # Page de détails d'un article
    code_article = st.session_state.article_selectionne
    photos = gestionnaire.get_photos_article(code_article)
    total = gestionnaire.get_total_article(code_article)
    libelle = gestionnaire.get_libelle_article(code_article)
    emplacement = gestionnaire.get_emplacement_article(code_article)
    
    # En-tête avec libellé et emplacement
    col_h1, col_h2, col_h3, col_h4, col_h5 = st.columns([2, 1, 1, 1, 1])
    with col_h1:
        st.header(f"📦 {code_article}")
        if libelle:
            st.markdown(f"<span class='label-badge'>📝 {libelle}</span>", unsafe_allow_html=True)
        if emplacement:
            st.markdown(f"<span class='location-badge'>📍 {emplacement}</span>", unsafe_allow_html=True)
    with col_h2:
        st.metric("Total pièces", total)
    with col_h3:
        st.metric("Photos", len(photos))
    with col_h4:
        if libelle:
            st.metric("Libellé", libelle[:20] + "..." if len(libelle) > 20 else libelle)
    with col_h5:
        if emplacement:
            st.metric("Emplacement", emplacement)
    
    # Afficher un badge si le code est un code-barres
    if re.match(r'^[A-Z0-9-]+$', code_article):
        st.info(f"🔖 Code produit: {code_article}")
    
    # Options
    col_o1, col_o2, col_o3 = st.columns(3)
    with col_o1:
        if st.button("⬅️ Retour à la saisie", use_container_width=True):
            st.session_state.page = "saisie"
            st.session_state.code_detecte = None
            st.session_state.scan_effectue = False
            st.rerun()
    with col_o2:
        if st.button("📸 Ajouter une photo", use_container_width=True):
            st.session_state.ajout_photo = True
    with col_o3:
        if st.button("🗑️ Supprimer cet article", use_container_width=True, type="primary"):
            if gestionnaire.supprimer_article(code_article):
                st.success(f"✅ Article '{code_article}' supprimé")
                st.session_state.page = "saisie"
                st.rerun()
    
    st.divider()
    
    # Ajout de photo
    if st.session_state.get('ajout_photo', False):
        st.subheader("📸 Ajouter une photo")
        
        col_p1, col_p2 = st.columns([2, 1])
        with col_p2:
            if st.button("❌ Annuler"):
                st.session_state.ajout_photo = False
                st.rerun()
        
        with col_p1:
            source = st.radio("Source", ["📸 Prendre une photo", "🖼️ Choisir une image"], horizontal=True)
        
        if source == "📸 Prendre une photo":
            img_file = st.camera_input("Prendre une photo")
            if img_file:
                with st.spinner("Analyse..."):
                    bytes_data = img_file.getvalue()
                    frame = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
                    resultat, nb_pieces = detecter_pieces(frame)
                    
                    if gestionnaire.ajouter_photo_article(code_article, frame, resultat, nb_pieces):
                        st.success(f"✅ {nb_pieces} pièces détectées et ajoutées!")
                        st.session_state.ajout_photo = False
                        st.rerun()
        
        else:  # Choisir une image
            uploaded_file = st.file_uploader("Choisir une image", type=['jpg', 'jpeg', 'png'])
            if uploaded_file:
                with st.spinner("Analyse..."):
                    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    resultat, nb_pieces = detecter_pieces(frame)
                    
                    if gestionnaire.ajouter_photo_article(code_article, frame, resultat, nb_pieces):
                        st.success(f"✅ {nb_pieces} pièces détectées et ajoutées!")
                        st.session_state.ajout_photo = False
                        st.rerun()
    
    # Affichage des photos existantes
    if photos:
        st.subheader("📸 Photos enregistrées")
        
        # Options d'affichage
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            tri = st.selectbox("Trier par", ["Plus récente", "Plus ancienne", "Plus de pièces", "Moins de pièces"])
        
        # Trier les photos
        photos_affichees = photos.copy()
        if tri == "Plus récente":
            photos_affichees = list(reversed(photos_affichees))
        elif tri == "Plus ancienne":
            photos_affichees = photos_affichees
        elif tri == "Plus de pièces":
            photos_affichees = sorted(photos_affichees, key=lambda x: x['nb_pieces'], reverse=True)
        elif tri == "Moins de pièces":
            photos_affichees = sorted(photos_affichees, key=lambda x: x['nb_pieces'])
        
        # Afficher les photos en grille
        cols = st.columns(3)
        for i, photo in enumerate(photos_affichees):
            with cols[i % 3]:
                # Afficher la miniature
                img = base64_to_image(photo['image_analyse'])
                img_mini = cv2.resize(img, (200, 150))
                st.image(cv2.cvtColor(img_mini, cv2.COLOR_BGR2RGB), use_column_width=True)
                
                # Informations
                st.caption(f"📅 {photo['timestamp'][:10]}")
                st.caption(f"🔢 {photo['nb_pieces']} pièces")
                
                # Boutons
                col_b1, col_b2 = st.columns(2)
                with col_b1:
                    if st.button("🔍 Voir", key=f"view_{code_article}_{i}"):
                        st.session_state.photo_selectionnee = photo['id']
                        st.session_state.page = "photo_detail"
                        st.rerun()
                with col_b2:
                    if st.button("🗑️", key=f"del_{code_article}_{i}"):
                        if gestionnaire.supprimer_photo(code_article, photo['id']):
                            st.rerun()
    
    else:
        st.info("📸 Aucune photo pour cet article. Cliquez sur 'Ajouter une photo' pour commencer.")

elif st.session_state.page == "photo_detail" and st.session_state.article_selectionne and st.session_state.photo_selectionnee is not None:
    # Détail d'une photo spécifique
    code_article = st.session_state.article_selectionne
    photos = gestionnaire.get_photos_article(code_article)
    photo_id = st.session_state.photo_selectionnee
    
    if 0 <= photo_id < len(photos):
        photo = photos[photo_id]
        libelle = gestionnaire.get_libelle_article(code_article)
        
        st.header(f"🔍 Détail de la photo - {code_article}")
        if libelle:
            st.subheader(libelle)
        
        # Afficher les deux images
        col_img1, col_img2 = st.columns(2)
        
        with col_img1:
            st.subheader("📸 Image originale")
            img_originale = base64_to_image(photo['image_originale'])
            st.image(cv2.cvtColor(img_originale, cv2.COLOR_BGR2RGB), use_column_width=True)
        
        with col_img2:
            st.subheader(f"🔍 Analyse - {photo['nb_pieces']} pièces")
            img_analyse = base64_to_image(photo['image_analyse'])
            st.image(cv2.cvtColor(img_analyse, cv2.COLOR_BGR2RGB), use_column_width=True)
        
        # Informations
        st.metric("Nombre de pièces", photo['nb_pieces'])
        st.caption(f"Date: {photo['timestamp']}")
        
        # Boutons
        col_b1, col_b2 = st.columns(2)
        with col_b1:
            if st.button("⬅️ Retour à l'article", use_container_width=True):
                st.session_state.page = "details"
                st.session_state.photo_selectionnee = None
                st.rerun()
        with col_b2:
            if st.button("🗑️ Supprimer cette photo", use_container_width=True, type="primary"):
                if gestionnaire.supprimer_photo(code_article, photo_id):
                    st.session_state.page = "details"
                    st.session_state.photo_selectionnee = None
                    st.rerun()
    else:
        st.error("Photo non trouvée")
        if st.button("Retour"):
            st.session_state.page = "details"
            st.session_state.photo_selectionnee = None
            st.rerun()

# Pied de page
st.markdown("---")
col_f1, col_f2, col_f3, col_f4, col_f5 = st.columns(5)
with col_f1:
    st.caption("📦 Gestionnaire d'Inventaire v3.0 - Avec webcam en direct")
with col_f2:
    total_global = sum(gestionnaire.get_tous_les_totaux().values())
    st.caption(f"🧩 Total global: {total_global} pièces")
with col_f3:
    st.caption(f"📊 Articles: {len(gestionnaire.articles)}")
with col_f4:
    emplacements_renseignes = sum(1 for e in gestionnaire.get_tous_emplacements().values() if e)
    st.caption(f"📍 Emplacements: {emplacements_renseignes}/{len(gestionnaire.articles)}")
with col_f5:
    libelles_renseignes = sum(1 for l in gestionnaire.get_tous_libelles().values() if l)
    st.caption(f"📝 Libellés: {libelles_renseignes}/{len(gestionnaire.articles)}")
