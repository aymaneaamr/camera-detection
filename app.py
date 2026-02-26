import streamlit as st
import cv2
import numpy as np
from collections import defaultdict
from datetime import datetime
import base64
from io import BytesIO
import openpyxl
from pyzbar.pyzbar import decode
import re
import time
import threading

# ==================== Configuration ====================
st.set_page_config(
    page_title="Compteur de Pièces - Caméra USB",
    page_icon="📷",
    layout="wide"
)

# CSS
st.markdown("""
<style>
    .video-container {
        border: 3px solid #ff4b4b;
        border-radius: 10px;
        padding: 10px;
        margin: 10px 0;
        background: #1e1e1e;
    }
    .stats-box {
        background: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #ff4b4b;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ==================== Détecteur de pièces (votre code original) ====================
class CompteurPieces:
    def __init__(self):
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
        for nom_taille, (min_vol, max_vol) in self.seuils_taille.items():
            if min_vol <= aire < max_vol:
                return nom_taille
        return '?'
    
    def traiter_frame(self, frame):
        """Version exacte de votre fonction originale"""
        resultat = frame.copy()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=2)
        edges = cv2.erode(edges, kernel, iterations=1)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        pieces = []
        stats_couleur = defaultdict(int)
        stats_taille = defaultdict(int)
        
        for contour in contours:
            aire = cv2.contourArea(contour)
            if aire < 200:
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            centre = (x + w//2, y + h//2)
            
            couleur_nom = self.get_couleur_piece(hsv, contour)
            taille_nom = self.get_taille_piece(aire)
            
            pieces.append(contour)
            stats_couleur[couleur_nom] += 1
            stats_taille[taille_nom] += 1
            
            # Dessiner
            couleur_bbox = self.couleurs.get(couleur_nom, {}).get('couleur_bbox', (128, 128, 128))
            cv2.rectangle(resultat, (x, y), (x+w, y+h), couleur_bbox, 2)
            cv2.circle(resultat, centre, 3, (255, 255, 255), -1)
        
        total_actuel = len(pieces)
        
        # Ajouter les stats sur l'image
        h, w = resultat.shape[:2]
        cv2.putText(resultat, f"TOTAL: {total_actuel}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        y = 60
        cv2.putText(resultat, f"Rouge:{stats_couleur.get('rouge',0)} Bleu:{stats_couleur.get('bleu',0)} Vert:{stats_couleur.get('vert',0)} Jaune:{stats_couleur.get('jaune',0)}", 
                   (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        y += 20
        cv2.putText(resultat, f"P:{stats_taille.get('P',0)} M:{stats_taille.get('M',0)} G:{stats_taille.get('G',0)} TG:{stats_taille.get('TG',0)}", 
                   (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return resultat, total_actuel, stats_couleur, stats_taille

# ==================== Capture vidéo en continu ====================
class VideoCapture:
    def __init__(self, camera_id=2):
        self.camera_id = camera_id
        self.cap = None
        self.running = False
        self.frame = None
        self.compteur = CompteurPieces()
        self.stats = {"total": 0, "couleurs": {}, "tailles": {}}
        self.lock = threading.Lock()
    
    def start(self):
        """Démarre la capture vidéo"""
        self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            return False
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        self.running = True
        self.thread = threading.Thread(target=self._update)
        self.thread.daemon = True
        self.thread.start()
        return True
    
    def _update(self):
        """Boucle de capture"""
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                # Traiter la frame
                resultat, total, stats_couleur, stats_taille = self.compteur.traiter_frame(frame)
                
                with self.lock:
                    self.frame = resultat
                    self.stats["total"] = total
                    self.stats["couleurs"] = dict(stats_couleur)
                    self.stats["tailles"] = dict(stats_taille)
            time.sleep(0.03)  # ~30 FPS
    
    def read(self):
        """Lit la dernière frame"""
        with self.lock:
            return self.frame.copy() if self.frame is not None else None, self.stats.copy()
    
    def stop(self):
        """Arrête la capture"""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()

# ==================== Initialisation ====================
if 'video' not in st.session_state:
    st.session_state.video = None
if 'capture_active' not in st.session_state:
    st.session_state.capture_active = False
if 'photos' not in st.session_state:
    st.session_state.photos = []

# ==================== Interface ====================
st.title("📷 Compteur de Pièces - Caméra USB (Index 2)")
st.markdown("Utilisation de la caméra du téléphone en USB")

# Contrôles
col1, col2, col3, col4 = st.columns(4)

with col1:
    camera_index = st.number_input("Index caméra", min_value=0, max_value=10, value=2)

with col2:
    if not st.session_state.capture_active:
        if st.button("▶️ Démarrer la caméra", use_container_width=True):
            video = VideoCapture(int(camera_index))
            if video.start():
                st.session_state.video = video
                st.session_state.capture_active = True
                st.rerun()
            else:
                st.error(f"❌ Impossible d'ouvrir la caméra {camera_index}")
    else:
        if st.button("⏹️ Arrêter", use_container_width=True):
            if st.session_state.video:
                st.session_state.video.stop()
            st.session_state.video = None
            st.session_state.capture_active = False
            st.rerun()

with col3:
    if st.session_state.capture_active:
        st.success("✅ Caméra active")

with col4:
    if st.button("📸 Capturer", use_container_width=True):
        st.session_state.capture_request = True

# Affichage vidéo en direct
if st.session_state.capture_active and st.session_state.video:
    # Lire la dernière frame
    frame, stats = st.session_state.video.read()
    
    if frame is not None:
        # Créer un placeholder pour la vidéo
        video_placeholder = st.empty()
        
        # Afficher la frame
        video_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 
                               channels="RGB", use_column_width=True)
        
        # Statistiques
        with st.container():
            st.markdown('<div class="stats-box">', unsafe_allow_html=True)
            
            col_s1, col_s2, col_s3, col_s4 = st.columns(4)
            
            with col_s1:
                st.metric("TOTAL PIÈCES", stats["total"])
            
            with col_s2:
                couleurs = stats["couleurs"]
                st.metric("Rouge/Bleu", f"{couleurs.get('rouge',0)}/{couleurs.get('bleu',0)}")
            
            with col_s3:
                st.metric("Vert/Jaune", f"{couleurs.get('vert',0)}/{couleurs.get('jaune',0)}")
            
            with col_s4:
                tailles = stats["tailles"]
                st.metric("P/M/G", f"{tailles.get('P',0)}/{tailles.get('M',0)}/{tailles.get('G',0)}")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Capture d'image
        if st.session_state.get('capture_request', False):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(f"capture_{timestamp}.jpg", frame)
            
            # Ajouter aux photos
            st.session_state.photos.append({
                'timestamp': datetime.now().strftime("%H:%M:%S"),
                'total': stats["total"],
                'frame': frame
            })
            
            st.success(f"✅ Image capturée avec {stats['total']} pièces!")
            st.session_state.capture_request = False
        
        # Rafraîchissement automatique
        time.sleep(0.1)
        st.rerun()

# Galerie des captures
if st.session_state.photos:
    st.divider()
    st.subheader("📸 Dernières captures")
    
    cols = st.columns(3)
    for i, photo in enumerate(reversed(st.session_state.photos[-6:])):  # 6 dernières
        with cols[i % 3]:
            st.image(cv2.cvtColor(photo['frame'], cv2.COLOR_BGR2RGB), 
                    caption=f"{photo['timestamp']} - {photo['total']} pièces")
            
            if st.button(f"🗑️ Supprimer", key=f"del_photo_{i}"):
                st.session_state.photos.remove(photo)
                st.rerun()

# Instructions
with st.expander("📋 Instructions"):
    st.markdown("""
    ### Comment utiliser :
    1. Branchez votre téléphone en mode USB camera
    2. Vérifiez que l'index 2 est bien votre téléphone
    3. Cliquez sur **Démarrer la caméra**
    4. Le flux vidéo s'affiche en direct avec comptage automatique
    5. Cliquez sur **Capturer** pour sauvegarder une image
    6. Les captures apparaissent dans la galerie
    
    ### Fonctionnalités :
    - ✅ Flux vidéo en temps réel
    - ✅ Comptage automatique des pièces
    - ✅ Détection des couleurs (rouge, bleu, vert, jaune)
    - ✅ Classification par taille (P, M, G, TG)
    - ✅ Capture d'images
    - ✅ Galerie des captures
    """)
