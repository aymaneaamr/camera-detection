import streamlit as st
import cv2
import numpy as np
from collections import defaultdict
from PIL import Image
import time
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
import pandas as pd
from datetime import datetime
import json
import os

# Configuration de la page
st.set_page_config(
    page_title="Compteur de Pièces - Inventaire Entrepôt",
    page_icon="🏭",
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
        self.stats_couleur = defaultdict(int)
        self.stats_taille = defaultdict(int)
        self.total_pieces = 0
        self.stats_couleur_total = defaultdict(int)
        self.stats_taille_total = defaultdict(int)
        self.total_pieces_cumule = 0
        self.historique_photos = []
        self.inventaire_total = defaultdict(lambda: defaultdict(int))
    
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
    
    def ajouter_photo_analyse(self, stats_couleur, stats_taille, total_actuel, nom_photo=""):
        """Ajoute les résultats d'une photo à l'inventaire cumulé"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Créer l'entrée pour l'historique
        entree_photo = {
            'timestamp': timestamp,
            'nom_photo': nom_photo if nom_photo else f"Photo_{len(self.historique_photos)+1}",
            'total_pieces': total_actuel,
            'stats_couleur': dict(stats_couleur),
            'stats_taille': dict(stats_taille)
        }
        
        self.historique_photos.append(entree_photo)
        
        # Mettre à jour les totaux cumulés
        for couleur, count in stats_couleur.items():
            self.stats_couleur_total[couleur] += count
            # Mettre à jour l'inventaire par taille et couleur
            for taille, count_taille in stats_taille.items():
                if count_taille > 0:
                    self.inventaire_total[couleur][taille] += count // max(1, len([t for t in stats_taille.values() if t > 0]))
        
        for taille, count in stats_taille.items():
            self.stats_taille_total[taille] += count
        
        self.total_pieces_cumule += total_actuel
        
        return entree_photo
    
    def get_inventaire_dataframe(self):
        """Retourne l'inventaire sous forme de DataFrame"""
        data = []
        for couleur in ['rouge', 'bleu', 'vert', 'jaune', '?']:
            for taille in ['P', 'M', 'G', 'TG']:
                quantite = self.inventaire_total.get(couleur, {}).get(taille, 0)
                if quantite > 0 or couleur == '?' or taille == 'P':
                    data.append({
                        'Couleur': couleur.capitalize(),
                        'Taille': taille,
                        'Quantité': quantite
                    })
        
        return pd.DataFrame(data)
    
    def exporter_inventaire_json(self):
        """Exporte l'inventaire au format JSON"""
        inventaire = {
            'total_pieces_cumule': self.total_pieces_cumule,
            'stats_couleur_total': dict(self.stats_couleur_total),
            'stats_taille_total': dict(self.stats_taille_total),
            'historique_photos': self.historique_photos,
            'inventaire_detail': {
                couleur: dict(taille) for couleur, taille in self.inventaire_total.items()
            },
            'date_export': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        return json.dumps(inventaire, indent=2, ensure_ascii=False)

# Classe pour le traitement vidéo en temps réel
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.compteur = compteur
    
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        resultat, _, _, _, _ = self.compteur.traiter_frame(img)
        return av.VideoFrame.from_ndarray(resultat, format="bgr24")

# Initialisation du compteur dans la session
if 'compteur' not in st.session_state:
    st.session_state.compteur = CompteurPieces()
if 'frame_count' not in st.session_state:
    st.session_state.frame_count = 0
if 'mode' not in st.session_state:
    st.session_state.mode = None
if 'derniere_analyse' not in st.session_state:
    st.session_state.derniere_analyse = None

compteur = st.session_state.compteur

# Interface Streamlit
st.title("🏭 Compteur de Pièces - Gestion d'Inventaire Entrepôt")
st.markdown("""
Cette application permet de gérer l'inventaire de votre entrepôt :
- **Détection automatique** des pièces par couleur et taille
- **Accumulation des résultats** de plusieurs photos
- **Suivi d'inventaire** en temps réel
- **Export des données** pour votre système de gestion
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
    # ========== INTERFACE MOBILE (TÉLÉPHONE) ==========
    st.info("📱 Mode téléphone détecté - Interface optimisée pour mobile")
    
    # Interface simplifiée pour mobile avec onglets
    tab1, tab2, tab3 = st.tabs(["📸 Analyse", "📊 Inventaire", "⚙️ Paramètres"])
    
    with tab1:
        st.subheader("📸 Prendre une photo")
        
        # Affichage compact
        col1, col2 = st.columns([1, 1])
        with col1:
            source = st.radio(
                "Source",
                ["📸 Caméra", "🖼️ Galerie", "🧪 Démo"],
                label_visibility="collapsed"
            )
        
        with col2:
            nom_photo = st.text_input("Nom du lot", placeholder="ex: Lot A-123", key="nom_photo_mobile")
        
        if source == "📸 Caméra":
            img_file = st.camera_input("Prendre une photo", key="mobile_camera")
            
            if img_file is not None:
                with st.spinner("🔍 Analyse..."):
                    bytes_data = img_file.getvalue()
                    frame = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
                    
                    resultat, pieces, stats_couleur, stats_taille, total_actuel = compteur.traiter_frame(frame)
                    
                    # Ajouter à l'inventaire
                    entree = compteur.ajouter_photo_analyse(stats_couleur, stats_taille, total_actuel, nom_photo)
                    st.session_state.derniere_analyse = entree
                    
                    st.success(f"✅ **{total_actuel} pièces** ajoutées à l'inventaire")
                    
                    # Affichage compact
                    st.image(cv2.cvtColor(resultat, cv2.COLOR_BGR2RGB), use_column_width=True)
                    
                    # Stats en lignes
                    st.write("**Couleurs:** " + ", ".join([f"{c}:{stats_couleur.get(c,0)}" for c in ['rouge','bleu','vert','jaune'] if stats_couleur.get(c,0)>0]))
                    st.write("**Tailles:** " + ", ".join([f"{t}:{stats_taille.get(t,0)}" for t in ['P','M','G','TG'] if stats_taille.get(t,0)>0]))
        
        elif source == "🖼️ Galerie":
            uploaded_file = st.file_uploader("Choisir image", type=['jpg', 'jpeg', 'png'], label_visibility="collapsed")
            
            if uploaded_file:
                with st.spinner("🔍 Analyse..."):
                    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    
                    resultat, pieces, stats_couleur, stats_taille, total_actuel = compteur.traiter_frame(frame)
                    
                    # Ajouter à l'inventaire
                    entree = compteur.ajouter_photo_analyse(stats_couleur, stats_taille, total_actuel, nom_photo)
                    st.session_state.derniere_analyse = entree
                    
                    st.success(f"✅ **{total_actuel} pièces** ajoutées à l'inventaire")
                    st.image(cv2.cvtColor(resultat, cv2.COLOR_BGR2RGB), use_column_width=True)
        
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
                    
                    # Ajouter à l'inventaire
                    entree = compteur.ajouter_photo_analyse(stats_couleur, stats_taille, total_actuel, "Mode démo")
                    st.session_state.derniere_analyse = entree
                    
                    st.success(f"✅ **{total_actuel} pièces** ajoutées à l'inventaire")
                    st.image(cv2.cvtColor(resultat, cv2.COLOR_BGR2RGB), use_column_width=True)
    
    with tab2:
        st.subheader(f"📊 Inventaire Total: {compteur.total_pieces_cumule} pièces")
        
        # Affichage des métriques principales
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric("Photos analysées", len(compteur.historique_photos))
        with col_m2:
            st.metric("Total pièces", compteur.total_pieces_cumule)
        with col_m3:
            st.metric("Dernier ajout", compteur.historique_photos[-1]['total_pieces'] if compteur.historique_photos else 0)
        
        # Tableau d'inventaire
        st.write("**📦 Inventaire par couleur et taille:**")
        df_inventaire = compteur.get_inventaire_dataframe()
        if not df_inventaire.empty:
            st.dataframe(df_inventaire, use_container_width=True, hide_index=True)
        
        # Répartition par couleur
        st.write("**🎨 Répartition par couleur:**")
        if compteur.stats_couleur_total:
            cols = st.columns(len(compteur.stats_couleur_total))
            for i, (couleur, count) in enumerate(compteur.stats_couleur_total.items()):
                if count > 0:
                    with cols[i % len(cols)]:
                        st.metric(couleur.capitalize(), count)
        
        # Historique des photos
        with st.expander("📜 Historique des analyses"):
            for i, photo in enumerate(reversed(compteur.historique_photos[-10:]), 1):
                st.write(f"{i}. {photo['timestamp']} - {photo['nom_photo']}: {photo['total_pieces']} pièces")
                st.caption(f"   Couleurs: {photo['stats_couleur']}")
        
        # Boutons d'export
        col_b1, col_b2 = st.columns(2)
        with col_b1:
            if st.button("📥 Exporter CSV", use_container_width=True):
                csv = df_inventaire.to_csv(index=False)
                st.download_button(
                    label="Télécharger CSV",
                    data=csv,
                    file_name=f"inventaire_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        with col_b2:
            if st.button("📤 Exporter JSON", use_container_width=True):
                json_data = compteur.exporter_inventaire_json()
                st.download_button(
                    label="Télécharger JSON",
                    data=json_data,
                    file_name=f"inventaire_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
    
    with tab3:
        st.subheader("⚙️ Configuration")
        
        if st.button("🔄 Réinitialiser l'inventaire", use_container_width=True, type="primary"):
            compteur.reset_compteur()
            st.session_state.frame_count = 0
            st.success("✅ Inventaire réinitialisé")
            st.rerun()
        
        st.write("**📏 Seuils de taille (pixels):**")
        st.write("- P: < 500")
        st.write("- M: 500-2000")
        st.write("- G: 2000-5000")
        st.write("- TG: > 5000")

else:
    # ========== INTERFACE PC (ORDINATEUR) ==========
    st.info("💻 Mode PC détecté - Interface complète avec gestion d'inventaire")
    
    # Sidebar pour les paramètres et l'inventaire
    with st.sidebar:
        st.header("📦 INVENTAIRE")
        
        # Métriques principales
        st.metric("📸 Photos analysées", len(compteur.historique_photos))
        st.metric("🧩 Total pièces", compteur.total_pieces_cumule)
        
        if compteur.historique_photos:
            st.metric("🆕 Dernier ajout", f"{compteur.historique_photos[-1]['total_pieces']} pièces")
        
        st.markdown("---")
        
        # Aperçu rapide de l'inventaire
        st.subheader("🎨 Par couleur")
        for couleur, count in compteur.stats_couleur_total.items():
            if count > 0:
                st.write(f"- {couleur}: {count}")
        
        st.subheader("📏 Par taille")
        for taille, count in compteur.stats_taille_total.items():
            if count > 0:
                st.write(f"- {taille}: {count}")
        
        st.markdown("---")
        st.header("⚙️ Configuration")
        
        source = st.radio(
            "Source d'analyse",
            ["📸 Prendre une photo", "🎥 Flux en direct", "🖼️ Uploader une image", "🧪 Mode démo"]
        )
        
        nom_lot = st.text_input("🏷️ Nom du lot", placeholder="ex: Lot A-123", key="nom_lot_pc")
        
        if st.button("🔄 Réinitialiser inventaire", use_container_width=True, type="primary"):
            compteur.reset_compteur()
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
    
    # Zone principale PC avec onglets
    tab_main1, tab_main2, tab_main3 = st.tabs(["🔍 Analyse", "📊 Inventaire complet", "📈 Statistiques"])
    
    with tab_main1:
        if source == "📸 Prendre une photo":
            st.subheader("📸 Prenez une photo pour l'inventaire")
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                img_file = st.camera_input("Cliquez pour prendre une photo", key="pc_camera")
            
            if img_file is not None:
                with st.spinner("🔍 Analyse en cours..."):
                    bytes_data = img_file.getvalue()
                    frame = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
                    
                    resultat, pieces, stats_couleur, stats_taille, total_actuel = compteur.traiter_frame(frame)
                    
                    # Ajouter à l'inventaire
                    entree = compteur.ajouter_photo_analyse(stats_couleur, stats_taille, total_actuel, nom_lot)
                    st.session_state.derniere_analyse = entree
                    st.session_state.frame_count += 1
                    
                    st.success(f"✅ **{total_actuel} pièces** ajoutées à l'inventaire !")
                    
                    col_img1, col_img2 = st.columns(2)
                    with col_img1:
                        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 
                                caption="📸 Photo originale", use_column_width=True)
                    with col_img2:
                        st.image(cv2.cvtColor(resultat, cv2.COLOR_BGR2RGB), 
                                caption=f"🎯 {total_actuel} pièces détectées", use_column_width=True)
                    
                    # Statistiques détaillées
                    st.subheader("📊 Résultats de l'analyse")
                    
                    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                    with col_m1:
                        st.metric("Pièces détectées", total_actuel)
                    with col_m2:
                        st.metric("Couleurs différentes", len([c for c in stats_couleur.values() if c > 0]))
                    with col_m3:
                        st.metric("Tailles différentes", len([t for t in stats_taille.values() if t > 0]))
                    with col_m4:
                        st.metric("Total inventaire", compteur.total_pieces_cumule)
                    
                    # Tableau des couleurs
                    st.write("**🎨 Répartition par couleur :**")
                    cols = st.columns(5)
                    couleurs_list = ['rouge', 'bleu', 'vert', 'jaune', 'autre']
                    color_emoji = {'rouge': '🔴', 'bleu': '🔵', 'vert': '🟢', 'jaune': '🟡', 'autre': '⚪'}
                    
                    for i, couleur in enumerate(couleurs_list):
                        with cols[i]:
                            count = stats_couleur.get(couleur if couleur != 'autre' else '?', 0)
                            st.metric(f"{color_emoji[couleur]} {couleur}", count)
                    
                    # Tableau des tailles
                    st.write("**📏 Répartition par taille :**")
                    cols = st.columns(4)
                    tailles_list = ['P', 'M', 'G', 'TG']
                    for i, taille in enumerate(tailles_list):
                        with cols[i]:
                            count = stats_taille.get(taille, 0)
                            st.metric(f"Taille {taille}", count)
        
        elif source == "🎥 Flux en direct":
            st.subheader("🎥 Flux vidéo en temps réel")
            
            st.warning("⚠️ En mode flux en direct, les pièces ne sont pas automatiquement ajoutées à l'inventaire. Utilisez 'Prendre une photo' pour l'inventaire.")
            
            # Lancer le flux vidéo
            ctx = webrtc_streamer(
                key="object-detection-pc",
                mode=WebRtcMode.SENDRECV,
                video_processor_factory=VideoProcessor,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
            )
            
            if ctx.state.playing:
                if st.button("📸 Capturer et ajouter à l'inventaire"):
                    st.info("Fonctionnalité à implémenter - Utilisez 'Prendre une photo' pour l'instant")
        
        elif source == "🖼️ Uploader une image":
            st.subheader("🖼️ Analyse d'image pour inventaire")
            
            uploaded_file = st.file_uploader("Choisissez une image", type=['jpg', 'jpeg', 'png'])
            
            if uploaded_file:
                with st.spinner("🔍 Analyse en cours..."):
                    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    
                    resultat, pieces, stats_couleur, stats_taille, total_actuel = compteur.traiter_frame(frame)
                    
                    # Ajouter à l'inventaire
                    entree = compteur.ajouter_photo_analyse(stats_couleur, stats_taille, total_actuel, nom_lot or uploaded_file.name)
                    st.session_state.derniere_analyse = entree
                    st.session_state.frame_count += 1
                    
                    st.success(f"✅ **{total_actuel} pièces** ajoutées à l'inventaire !")
                    
                    col_img1, col_img2 = st.columns(2)
                    with col_img1:
                        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 
                                caption="🖼️ Image originale", use_column_width=True)
                    with col_img2:
                        st.image(cv2.cvtColor(resultat, cv2.COLOR_BGR2RGB), 
                                caption=f"🎯 {total_actuel} pièces détectées", use_column_width=True)
        
        else:  # Mode démo
            st.subheader("🧪 Mode démo - Génération de données de test")
            
            if st.button("🎲 Générer et ajouter à l'inventaire"):
                with st.spinner("🔍 Analyse..."):
                    test_img = np.zeros((480, 640, 3), dtype=np.uint8)
                    test_img.fill(255)
                    
                    # Générer des formes de test
                    cv2.circle(test_img, (200, 200), 50, (0, 0, 255), -1)
                    cv2.circle(test_img, (350, 250), 40, (255, 0, 0), -1)
                    cv2.circle(test_img, (500, 200), 45, (0, 255, 0), -1)
                    cv2.circle(test_img, (300, 350), 35, (0, 255, 255), -1)
                    cv2.circle(test_img, (450, 350), 60, (100, 100, 100), -1)
                    
                    resultat, pieces, stats_couleur, stats_taille, total_actuel = compteur.traiter_frame(test_img)
                    
                    # Ajouter à l'inventaire
                    entree = compteur.ajouter_photo_analyse(stats_couleur, stats_taille, total_actuel, "Mode démo")
                    st.session_state.derniere_analyse = entree
                    st.session_state.frame_count += 1
                    
                    st.success(f"✅ **{total_actuel} pièces** ajoutées à l'inventaire en mode démo !")
                    
                    col_img1, col_img2 = st.columns(2)
                    with col_img1:
                        st.image(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB), 
                                caption="🧪 Image de test", use_column_width=True)
                    with col_img2:
                        st.image(cv2.cvtColor(resultat, cv2.COLOR_BGR2RGB), 
                                caption=f"🎯 {total_actuel} pièces détectées", use_column_width=True)
    
    with tab_main2:
        st.subheader("📊 INVENTAIRE COMPLET")
        
        # Métriques globales
        col_g1, col_g2, col_g3, col_g4 = st.columns(4)
        with col_g1:
            st.metric("📸 Total photos", len(compteur.historique_photos))
        with col_g2:
            st.metric("🧩 Total pièces", compteur.total_pieces_cumule)
        with col_g3:
            st.metric("🎨 Couleurs", len([c for c in compteur.stats_couleur_total.values() if c > 0]))
        with col_g4:
            st.metric("📏 Tailles", len([t for t in compteur.stats_taille_total.values() if t > 0]))
        
        # Tableau d'inventaire détaillé
        st.write("### 📦 Inventaire par couleur et taille")
        df_inventaire = compteur.get_inventaire_dataframe()
        
        if not df_inventaire.empty:
            # Pivot table pour une meilleure visualisation
            pivot_df = df_inventaire.pivot(index='Couleur', columns='Taille', values='Quantité').fillna(0).astype(int)
            
            col_p1, col_p2 = st.columns([2, 1])
            with col_p1:
                st.dataframe(pivot_df, use_container_width=True)
            with col_p2:
                # Graphique simple
                st.bar_chart(pivot_df.sum(axis=1))
        
        # Répartition par couleur
        st.write("### 🎨 Répartition par couleur")
        if compteur.stats_couleur_total:
            cols = st.columns(len(compteur.stats_couleur_total))
            for i, (couleur, count) in enumerate(compteur.stats_couleur_total.items()):
                if count > 0:
                    with cols[i % len(cols)]:
                        st.metric(couleur.capitalize(), count, delta=f"{count/compteur.total_pieces_cumule*100:.1f}%" if compteur.total_pieces_cumule > 0 else "0%")
        
        # Répartition par taille
        st.write("### 📏 Répartition par taille")
        if compteur.stats_taille_total:
            cols = st.columns(len(compteur.stats_taille_total))
            for i, (taille, count) in enumerate(compteur.stats_taille_total.items()):
                if count > 0:
                    with cols[i % len(cols)]:
                        st.metric(f"Taille {taille}", count, delta=f"{count/compteur.total_pieces_cumule*100:.1f}%" if compteur.total_pieces_cumule > 0 else "0%")
        
        # Historique des analyses
        st.write("### 📜 Historique des analyses")
        if compteur.historique_photos:
            # Créer un DataFrame pour l'historique
            hist_data = []
            for photo in compteur.historique_photos:
                hist_data.append({
                    'Date': photo['timestamp'],
                    'Lot': photo['nom_photo'],
                    'Pièces': photo['total_pieces'],
                    'Détail': f"C:{sum(photo['stats_couleur'].values())} pièces"
                })
            
            df_hist = pd.DataFrame(hist_data)
            st.dataframe(df_hist, use_container_width=True, hide_index=True)
        else:
            st.info("Aucune analyse pour le moment")
        
        # Boutons d'export
        st.write("### 📤 Export des données")
        col_e1, col_e2, col_e3 = st.columns(3)
        
        with col_e1:
            csv = df_inventaire.to_csv(index=False)
            st.download_button(
                label="📥 Télécharger CSV",
                data=csv,
                file_name=f"inventaire_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col_e2:
            json_data = compteur.exporter_inventaire_json()
            st.download_button(
                label="📥 Télécharger JSON",
                data=json_data,
                file_name=f"inventaire_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
        
        with col_e3:
            if st.button("🔄 Réinitialiser inventaire", use_container_width=True):
                compteur.reset_compteur()
                st.success("✅ Inventaire réinitialisé")
                st.rerun()
    
    with tab_main3:
        st.subheader("📈 Statistiques et Analyses")
        
        if compteur.historique_photos:
            # Créer un DataFrame pour les analyses temporelles
            df_temps = pd.DataFrame([
                {
                    'Date': photo['timestamp'],
                    'Total': photo['total_pieces'],
                    **{f"C_{c}": photo['stats_couleur'].get(c, 0) for c in ['rouge', 'bleu', 'vert', 'jaune', '?']},
                    **{f"T_{t}": photo['stats_taille'].get(t, 0) for t in ['P', 'M', 'G', 'TG']}
                }
                for photo in compteur.historique_photos
            ])
            
            # Graphique d'évolution
            st.write("### 📈 Évolution du nombre de pièces par analyse")
            st.line_chart(df_temps.set_index('Date')['Total'])
            
            # Statistiques descriptives
            st.write("### 📊 Statistiques descriptives")
            col_s1, col_s2, col_s3, col_s4 = st.columns(4)
            with col_s1:
                st.metric("Moyenne par lot", f"{df_temps['Total'].mean():.1f}")
            with col_s2:
                st.metric("Médiane", f"{df_temps['Total'].median():.1f}")
            with col_s3:
                st.metric("Min", df_temps['Total'].min())
            with col_s4:
                st.metric("Max", df_temps['Total'].max())
            
            # Distribution des couleurs
            st.write("### 🎨 Distribution des couleurs")
            couleurs_data = {
                'Rouge': compteur.stats_couleur_total.get('rouge', 0),
                'Bleu': compteur.stats_couleur_total.get('bleu', 0),
                'Vert': compteur.stats_couleur_total.get('vert', 0),
                'Jaune': compteur.stats_couleur_total.get('jaune', 0),
                'Autre': compteur.stats_couleur_total.get('?', 0)
            }
            df_couleurs = pd.DataFrame([couleurs_data])
            st.bar_chart(df_couleurs.T)
            
            # Distribution des tailles
            st.write("### 📏 Distribution des tailles")
            tailles_data = {
                'Petite (P)': compteur.stats_taille_total.get('P', 0),
                'Moyenne (M)': compteur.stats_taille_total.get('M', 0),
                'Grande (G)': compteur.stats_taille_total.get('G', 0),
                'Très Grande (TG)': compteur.stats_taille_total.get('TG', 0)
            }
            df_tailles = pd.DataFrame([tailles_data])
            st.bar_chart(df_tailles.T)
            
        else:
            st.info("📊 Aucune donnée statistique disponible. Commencez par analyser des photos.")

# Pied de page commun
st.markdown("---")
col_f1, col_f2, col_f3 = st.columns(3)
with col_f1:
    st.caption(f"🏭 Inventaire Entrepôt v4.0")
with col_f2:
    st.caption(f"📸 Photos analysées: {len(compteur.historique_photos)}")
with col_f3:
    st.caption(f"🧩 Total pièces: {compteur.total_pieces_cumule}")
