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
from skimage import measure
import matplotlib.pyplot as plt

# Configuration de la page
st.set_page_config(
    page_title="Compteur de Pièces - Ultra Séparation",
    page_icon="🔧",
    layout="wide"
)

class CompteurPieces:
    def __init__(self):
        """Initialise le compteur de pièces avec technologies ultra avancées"""
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
            },
            'gris': {
                'lower': np.array([0, 0, 30]), 'upper': np.array([180, 60, 255]),
                'couleur_bbox': (128, 128, 128)
            }
        }
        
        # Seuils de taille ultra fins
        self.seuils_taille = {
            'XS': (0, 50),      # Extra small
            'P': (50, 150),     # Petit
            'M': (150, 400),    # Moyen
            'G': (400, 1000),   # Grand
            'TG': (1000, 2500), # Très Grand
            'EX': (2500, float('inf'))  # Extra large
        }
        
        # Paramètres ultra agressifs
        self.params = {
            'seuil_aire_min': 15,  # Encore plus petit
            'seuil_canny_bas': 20,
            'seuil_canny_haut': 80,
            'sensibilite_couleur': 0.1,
            'seuil_circularite': 0.5,  # Moins strict
            'seuil_separation': 0.25,   # Plus agressif
            'utiliser_watershed': True,
            'utiliser_distance_transform': True,
            'utiliser_circularite': False,  # Désactivé par défaut
            'mode_detection': "Tous",
            'force_separation': 3,  # Niveau de force de séparation (1-5)
            'min_distance_peaks': 15,  # Distance minimale entre pics
            'erosion_iterations': 2,
            'dilatation_iterations': 3
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
    
    def separateur_ultra_agressif(self, contours, frame_shape):
        """
        Méthode ultra agressive pour séparer les objets collés
        Combine multiple techniques
        """
        if len(contours) == 0:
            return contours
        
        nouveaux_contours = []
        
        for contour in contours:
            aire = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            
            # Critères pour détecter les objets collés
            est_collé = False
            facteurs = []
            
            # 1. Rapport aire/bbox
            bbox_aire = w * h
            rapport = aire / bbox_aire if bbox_aire > 0 else 0
            facteurs.append(rapport < 0.5)  # Si rapport < 0.5, probablement collé
            
            # 2. Convexité
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidite = aire / hull_area if hull_area > 0 else 0
            facteurs.append(solidite < 0.7)  # Si solidité < 0.7, forme irrégulière
            
            # 3. Compacité
            perimeter = cv2.arcLength(contour, True)
            compacite = (perimeter ** 2) / (4 * np.pi * aire) if aire > 0 else 0
            facteurs.append(compacite > 2.0)  # Si compacité > 2, forme complexe
            
            # 4. Nombre de pics de convexité
            hull = cv2.convexHull(contour, returnPoints=False)
            defects = cv2.convexityDefects(contour, hull)
            if defects is not None:
                nb_defauts = len(defects)
                facteurs.append(nb_defauts > 3)  # Beaucoup de défauts = objets collés
            
            # Décision : si au moins 2 facteurs sont vrais, c'est collé
            if sum(facteurs) >= 2:
                est_collé = True
            
            if est_collé or aire > 1000:  # Force la séparation pour les gros objets
                # Technique 1: Distance transform agressive
                mask = np.zeros(frame_shape[:2], dtype=np.uint8)
                cv2.drawContours(mask, [contour], -1, 255, -1)
                
                # Distance transform
                dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
                cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
                
                # Seuillage adaptatif basé sur la force de séparation
                seuil = max(0.1, 0.4 - (self.params['force_separation'] * 0.05))
                _, dist_thresh = cv2.threshold(dist, seuil, 1.0, cv2.THRESH_BINARY)
                
                # Trouver les pics locaux
                local_max = peak_local_max(
                    dist, 
                    min_distance=self.params['min_distance_peaks'],
                    exclude_border=False,
                    num_peaks=10
                )
                
                if len(local_max) > 1:  # Plusieurs pics détectés
                    # Créer des marqueurs pour watershed
                    markers = np.zeros(dist.shape, dtype=np.int32)
                    for i, peak in enumerate(local_max):
                        markers[peak[0], peak[1]] = i + 1
                    
                    # Watershed
                    markers = ndimage.label(markers)[0]
                    markers = markers.astype(np.int32)
                    
                    # Préparer image pour watershed
                    img_rgb = cv2.cvtColor(
                        np.dstack([mask, mask, mask]) * 255, 
                        cv2.COLOR_GRAY2RGB
                    ).astype(np.uint8)
                    
                    cv2.watershed(img_rgb, markers)
                    
                    # Extraire les nouveaux contours
                    for i in range(1, markers.max() + 1):
                        marker_mask = np.zeros(markers.shape, dtype=np.uint8)
                        marker_mask[markers == i] = 255
                        
                        # Opérations morphologiques
                        kernel = np.ones((3, 3), np.uint8)
                        marker_mask = cv2.morphologyEx(marker_mask, cv2.MORPH_OPEN, kernel)
                        
                        # Trouver les contours
                        new_contours, _ = cv2.findContours(
                            marker_mask,
                            cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE
                        )
                        
                        for new_c in new_contours:
                            if cv2.contourArea(new_c) > self.params['seuil_aire_min']:
                                nouveaux_contours.append(new_c)
                else:
                    # Si un seul pic, garder le contour original
                    nouveaux_contours.append(contour)
            else:
                nouveaux_contours.append(contour)
        
        return nouveaux_contours
    
    def detection_contours_multi_echelle(self, frame):
        """Détection multi-échelle des contours"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Plusieurs échelles de flou
        tous_contours = []
        
        for blur_size in [(3,3), (5,5), (7,7)]:
            blur = cv2.GaussianBlur(gray, blur_size, 0)
            
            # Plusieurs seuils Canny
            for seuil_bas in [20, 30, 40]:
                edges = cv2.Canny(blur, seuil_bas, seuil_bas * 3)
                
                # Dilatation pour connecter les bords
                kernel = np.ones((2, 2), np.uint8)
                edges = cv2.dilate(edges, kernel, iterations=1)
                
                contours, _ = cv2.findContours(
                    edges, 
                    cv2.RETR_EXTERNAL, 
                    cv2.CHAIN_APPROX_SIMPLE
                )
                
                tous_contours.extend(contours)
        
        # Fusionner les contours proches
        contours_fusionnes = self.fusionner_contours_proches(tous_contours)
        
        return contours_fusionnes
    
    def fusionner_contours_proches(self, contours, distance_max=20):
        """Fusionne les contours très proches"""
        if len(contours) < 2:
            return contours
        
        # Grouper par proximité
        groupes = []
        utilises = [False] * len(contours)
        
        for i, c1 in enumerate(contours):
            if utilises[i]:
                continue
            
            groupe = [c1]
            utilises[i] = True
            x1, y1, w1, h1 = cv2.boundingRect(c1)
            centre1 = (x1 + w1//2, y1 + h1//2)
            
            for j, c2 in enumerate(contours[i+1:], i+1):
                if utilises[j]:
                    continue
                
                x2, y2, w2, h2 = cv2.boundingRect(c2)
                centre2 = (x2 + w2//2, y2 + h2//2)
                
                # Distance entre centres
                dist = np.sqrt((centre1[0] - centre2[0])**2 + (centre1[1] - centre2[1])**2)
                
                if dist < distance_max:
                    groupe.append(c2)
                    utilises[j] = True
            
            if len(groupe) > 1:
                # Fusionner les contours du groupe
                points = np.vstack([c.reshape(-1, 2) for c in groupe])
                hull = cv2.convexHull(points)
                groupes.append(hull)
            else:
                groupes.append(c1)
        
        return groupes
    
    def get_couleur_piece(self, hsv, contour):
        """Détermine la couleur avec analyse multi-zone"""
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        
        # Analyser plusieurs zones du contour (périphérie et centre)
        x, y, w, h = cv2.boundingRect(contour)
        
        # Masque pour le centre
        centre_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        cv2.rectangle(centre_mask, 
                     (x + w//4, y + h//4), 
                     (x + 3*w//4, y + 3*h//4), 
                     255, -1)
        centre_mask = cv2.bitwise_and(centre_mask, mask)
        
        # Masque pour la périphérie
        peri_mask = cv2.bitwise_xor(mask, centre_mask)
        
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
            
            # Créer masque couleur
            if 'lower1' in params:
                mask1 = cv2.inRange(hsv, params['lower1'], params['upper1'])
                mask2 = cv2.inRange(hsv, params['lower2'], params['upper2'])
                mask_couleur = cv2.bitwise_or(mask1, mask2)
            else:
                mask_couleur = cv2.inRange(hsv, params['lower'], params['upper'])
            
            # Score pour le centre (pondéré plus fort)
            centre_match = cv2.bitwise_and(mask_couleur, centre_mask)
            pixels_centre = cv2.countNonZero(centre_match)
            pixels_centre_total = cv2.countNonZero(centre_mask)
            score_centre = pixels_centre / pixels_centre_total if pixels_centre_total > 0 else 0
            
            # Score pour la périphérie
            peri_match = cv2.bitwise_and(mask_couleur, peri_mask)
            pixels_peri = cv2.countNonZero(peri_match)
            pixels_peri_total = cv2.countNonZero(peri_mask)
            score_peri = pixels_peri / pixels_peri_total if pixels_peri_total > 0 else 0
            
            # Score combiné (centre plus important)
            score_total = (score_centre * 0.7 + score_peri * 0.3)
            
            if score_total > best_score and score_total > self.params['sensibilite_couleur']:
                best_score = score_total
                best_couleur = nom_couleur
                best_color_bbox = params['couleur_bbox']
        
        return best_couleur, best_color_bbox
    
    def get_taille_piece(self, aire):
        """Détermine la taille avec catégories ultra-fines"""
        for nom_taille, (min_vol, max_vol) in self.seuils_taille.items():
            if min_vol <= aire < max_vol:
                return nom_taille
        return '?'
    
    def traiter_frame(self, frame):
        """Traite une frame avec ultra séparation"""
        resultat = frame.copy()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Détection multi-échelle des contours
        contours = self.detection_contours_multi_echelle(frame)
        
        # Filtrer par taille
        contours = [c for c in contours if cv2.contourArea(c) > self.params['seuil_aire_min']]
        
        # Application de la séparation ultra agressive
        contours = self.separateur_ultra_agressif(contours, frame.shape)
        
        pieces_actuelles = []
        stats_couleur_actuelles = defaultdict(int)
        stats_taille_actuelles = defaultdict(int)
        
        # Ajouter un identifiant unique à chaque pièce
        for idx, contour in enumerate(contours):
            aire = cv2.contourArea(contour)
            
            x, y, w, h = cv2.boundingRect(contour)
            centre = (x + w//2, y + h//2)
            
            couleur_nom, couleur_bbox = self.get_couleur_piece(hsv, contour)
            taille_nom = self.get_taille_piece(aire)
            
            pieces_actuelles.append({
                'id': idx + 1,
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
            
            # Ajouter ID et infos
            info_text = f"{idx+1}:{couleur_nom[0]}{taille_nom}"
            cv2.putText(resultat, info_text, (x+5, y+20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Ajouter un point de couleur pour l'ID
            cv2.putText(resultat, f"#{idx+1}", (x+5, y-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, couleur_bbox, 1)
        
        total_actuel = len(pieces_actuelles)
        
        # Mise à jour des stats
        self.stats_couleur = stats_couleur_actuelles
        self.stats_taille = stats_taille_actuelles
        self.total_pieces = total_actuel
        
        # Ajouter le compteur total avec style
        h, w = resultat.shape[:2]
        
        # Fond pour le texte
        cv2.rectangle(resultat, (5, 5), (200, 70), (0, 0, 0), -1)
        cv2.putText(resultat, f"Total: {total_actuel}", (15, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(resultat, f"Attendu: 32", (15, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Ajouter FPS
        if hasattr(self, 'fps'):
            cv2.putText(resultat, f"FPS: {self.fps:.1f}", (w-100, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return resultat, pieces_actuelles, stats_couleur_actuelles, stats_taille_actuelles, total_actuel

# [Le reste du code Streamlit reste similaire, avec mise à jour des catégories de taille]

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

# Initialisation
if 'compteur' not in st.session_state:
    st.session_state.compteur = CompteurPieces()
if 'frame_count' not in st.session_state:
    st.session_state.frame_count = 0
if 'mode' not in st.session_state:
    st.session_state.mode = None

compteur = st.session_state.compteur

# Interface Streamlit
st.title("🔧 Compteur de Pièces - Ultra Séparation V5.0")
st.markdown("""
<div style='background-color: #ffffcc; padding: 10px; border-radius: 5px;'>
⚠️ <strong>Objectif : 32 pièces</strong> - Détection actuelle : 29 pièces<br>
Problème : Objets collés détectés comme un seul article<br>
✅ <strong>Solution : Ultra séparation agressive activée</strong>
</div>
""", unsafe_allow_html=True)

# Paramètres ultra agressifs dans la sidebar
with st.sidebar:
    st.header("⚙️ Ultra Séparation")
    
    force_separation = st.slider("💪 Force de séparation", 1, 5, 3, 
                                help="Plus la valeur est élevée, plus la séparation est agressive")
    compteur.params['force_separation'] = force_separation
    
    min_distance = st.slider("📏 Distance min entre objets", 5, 30, 15,
                            help="Distance minimale entre les pics pour considérer 2 objets distincts")
    compteur.params['min_distance_peaks'] = min_distance
    
    seuil_aire = st.slider("🔍 Taille minimum", 5, 50, 15,
                          help="Aire minimum pour considérer un objet")
    compteur.params['seuil_aire_min'] = seuil_aire
    
    st.markdown("---")
    st.header("📊 Stats temps réel")
    
    # Afficher le compteur
    st.metric("Pièces détectées", compteur.total_pieces, 
              delta=compteur.total_pieces - 32, 
              delta_color="inverse")
    
    if compteur.total_pieces < 32:
        st.warning(f"❌ Il manque {32 - compteur.total_pieces} pièces")
    elif compteur.total_pieces > 32:
        st.warning(f"⚠️ Trop de pièces détectées (+{compteur.total_pieces - 32})")
    else:
        st.success("✅ Objectif atteint !")
    
    if st.button("🔄 Analyser à nouveau", use_container_width=True):
        st.rerun()

# Suite de l'interface...
