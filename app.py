import streamlit as st
import cv2
import numpy as np
from collections import defaultdict
import pandas as pd
from datetime import datetime
import json
import base64
from io import BytesIO
import openpyxl
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side

# Import pyzbar pour la lecture de codes-barres
try:
    from pyzbar.pyzbar import decode as decode_barcode
    BARCODE_AVAILABLE = True
except ImportError:
    BARCODE_AVAILABLE = False

# Configuration de la page
st.set_page_config(
    page_title="Gestionnaire d'Inventaire Multi-Pièces",
    page_icon="📦",
    layout="wide"
)

class GestionnairePieces:
    def __init__(self):
        """Initialise le gestionnaire de pièces"""
        self.pieces = {}  # Dictionnaire {nom_piece: [liste_des_photos]}
        self.reset_piece_courante()
    
    def reset_piece_courante(self):
        """Réinitialise la pièce en cours de saisie"""
        self.piece_courante = {
            'nom': '',
            'photos': [],
            'total_pieces': 0
        }
    
    def creer_nouvelle_piece(self, nom_piece):
        """Crée une nouvelle pièce dans l'inventaire"""
        if nom_piece and nom_piece not in self.pieces:
            self.pieces[nom_piece] = []
            return True
        return False
    
    def ajouter_photo_piece(self, nom_piece, frame_original, frame_analyse, nb_pieces):
        """Ajoute une photo analysée à une pièce existante"""
        if nom_piece in self.pieces:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            _, buffer_original = cv2.imencode('.jpg', frame_original)
            _, buffer_analyse = cv2.imencode('.jpg', frame_analyse)
            
            photo_data = {
                'timestamp': timestamp,
                'nb_pieces': nb_pieces,
                'image_originale': base64.b64encode(buffer_original).decode('utf-8'),
                'image_analyse': base64.b64encode(buffer_analyse).decode('utf-8'),
                'id': len(self.pieces[nom_piece])
            }
            
            self.pieces[nom_piece].append(photo_data)
            return True
        return False
    
    def get_total_piece(self, nom_piece):
        """Retourne le total de pièces pour un nom donné"""
        if nom_piece in self.pieces:
            return sum(photo['nb_pieces'] for photo in self.pieces[nom_piece])
        return 0
    
    def get_photos_piece(self, nom_piece):
        """Retourne toutes les photos d'une pièce"""
        return self.pieces.get(nom_piece, [])
    
    def supprimer_photo(self, nom_piece, photo_id):
        """Supprime une photo d'une pièce"""
        if nom_piece in self.pieces and 0 <= photo_id < len(self.pieces[nom_piece]):
            del self.pieces[nom_piece][photo_id]
            for i, photo in enumerate(self.pieces[nom_piece]):
                photo['id'] = i
            return True
        return False
    
    def supprimer_piece(self, nom_piece):
        """Supprime complètement une pièce"""
        if nom_piece in self.pieces:
            del self.pieces[nom_piece]
            return True
        return False
    
    def get_tous_les_totaux(self):
        """Retourne un dictionnaire avec tous les totaux par pièce"""
        return {nom: self.get_total_piece(nom) for nom in self.pieces}
    
    def generer_excel(self):
        """Génère un fichier Excel avec l'inventaire complet"""
        output = BytesIO()
        workbook = openpyxl.Workbook()
        
        sheet_resume = workbook.active
        sheet_resume.title = "Inventaire"
        
        headers = ["Nom de la pièce", "Quantité totale", "Nombre de photos", "Dernière mise à jour"]
        for col, header in enumerate(headers, 1):
            cell = sheet_resume.cell(row=1, column=col)
            cell.value = header
            cell.font = Font(bold=True)
            cell.fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
            cell.font = Font(color="FFFFFF", bold=True)
            cell.alignment = Alignment(horizontal="center")
        
        row = 2
        for nom_piece, photos in self.pieces.items():
            total = sum(p['nb_pieces'] for p in photos)
            nb_photos = len(photos)
            derniere_date = photos[-1]['timestamp'] if photos else "N/A"
            
            sheet_resume.cell(row=row, column=1).value = nom_piece
            sheet_resume.cell(row=row, column=2).value = total
            sheet_resume.cell(row=row, column=3).value = nb_photos
            sheet_resume.cell(row=row, column=4).value = derniere_date
            row += 1
        
        for col in range(1, 5):
            sheet_resume.column_dimensions[openpyxl.utils.get_column_letter(col)].width = 20
        
        sheet_detail = workbook.create_sheet("Détail des photos")
        
        detail_headers = ["Pièce", "Photo #", "Date", "Nombre de pièces"]
        for col, header in enumerate(detail_headers, 1):
            cell = sheet_detail.cell(row=1, column=col)
            cell.value = header
            cell.font = Font(bold=True)
            cell.fill = PatternFill(start_color="92D050", end_color="92D050", fill_type="solid")
            cell.alignment = Alignment(horizontal="center")
        
        row = 2
        for nom_piece, photos in self.pieces.items():
            for i, photo in enumerate(photos, 1):
                sheet_detail.cell(row=row, column=1).value = nom_piece
                sheet_detail.cell(row=row, column=2).value = f"Photo {i}"
                sheet_detail.cell(row=row, column=3).value = photo['timestamp']
                sheet_detail.cell(row=row, column=4).value = photo['nb_pieces']
                row += 1
        
        for col in range(1, 5):
            sheet_detail.column_dimensions[openpyxl.utils.get_column_letter(col)].width = 25
        
        workbook.save(output)
        output.seek(0)
        return output
    
    def reinitialiser_tout(self):
        """Réinitialise complètement l'inventaire"""
        self.pieces = {}


def lire_code_barre(image):
    """
    Lit le code-barres d'une image et retourne la valeur décodée.
    Retourne (valeur, image_annotée) ou (None, image_originale) si aucun code trouvé.
    """
    if not BARCODE_AVAILABLE:
        return None, image

    resultat = image.copy()
    
    # Convertir en niveaux de gris pour une meilleure détection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Tenter de décoder les codes-barres
    barcodes = decode_barcode(gray)
    
    # Si rien trouvé en gris, essayer avec l'image originale
    if not barcodes:
        barcodes = decode_barcode(image)
    
    if barcodes:
        # Prendre le premier code-barres trouvé
        barcode = barcodes[0]
        valeur = barcode.data.decode('utf-8')
        type_code = barcode.type
        
        # Dessiner un rectangle autour du code-barres
        points = barcode.polygon
        if len(points) == 4:
            pts = np.array([[p.x, p.y] for p in points], dtype=np.int32)
            cv2.polylines(resultat, [pts], True, (0, 255, 0), 3)
        else:
            rect = barcode.rect
            cv2.rectangle(resultat,
                          (rect.left, rect.top),
                          (rect.left + rect.width, rect.top + rect.height),
                          (0, 255, 0), 3)
        
        # Afficher la valeur sur l'image
        cv2.putText(resultat, valeur, (barcode.rect.left, barcode.rect.top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(resultat, f"Type: {type_code}", (barcode.rect.left, barcode.rect.top - 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
        
        return valeur, resultat
    
    return None, resultat


# Fonction pour détecter les pièces dans une image
def detecter_pieces(image):
    """Détecte et compte les pièces dans une image"""
    resultat = image.copy()
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)
    edges = cv2.erode(edges, kernel, iterations=1)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    pieces_valides = []
    for contour in contours:
        aire = cv2.contourArea(contour)
        if aire > 200:
            pieces_valides.append(contour)
    
    nb_pieces = len(pieces_valides)
    
    for contour in pieces_valides:
        cv2.drawContours(resultat, [contour], -1, (0, 255, 0), 2)
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.circle(resultat, (cx, cy), 3, (0, 0, 255), -1)
    
    cv2.putText(resultat, f"Pieces: {nb_pieces}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    return resultat, nb_pieces


def base64_to_image(base64_string):
    img_data = base64.b64decode(base64_string)
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


# Initialisation
if 'gestionnaire' not in st.session_state:
    st.session_state.gestionnaire = GestionnairePieces()
if 'page' not in st.session_state:
    st.session_state.page = "saisie"
if 'piece_selectionnee' not in st.session_state:
    st.session_state.piece_selectionnee = None
if 'photo_selectionnee' not in st.session_state:
    st.session_state.photo_selectionnee = None
if 'nom_scanne' not in st.session_state:
    st.session_state.nom_scanne = ""

gestionnaire = st.session_state.gestionnaire

# Interface principale
st.title("📦 Gestionnaire d'Inventaire Multi-Pièces")
st.markdown("""
Cette application permet de gérer l'inventaire de plusieurs types de pièces :
1. **Scanner** le code-barres d'une pièce (ou saisir manuellement)
2. **Ajouter** plusieurs photos pour cette pièce
3. **Changer** de pièce et répéter
4. **Exporter** un fichier Excel avec tous les totaux
""")

# Barre latérale
with st.sidebar:
    st.header("📋 Pièces en inventaire")
    
    if gestionnaire.pieces:
        for nom_piece in gestionnaire.pieces.keys():
            total = gestionnaire.get_total_piece(nom_piece)
            col1, col2 = st.columns([3, 1])
            with col1:
                if st.button(f"📦 {nom_piece}", key=f"select_{nom_piece}", use_container_width=True):
                    st.session_state.piece_selectionnee = nom_piece
                    st.session_state.page = "details"
            with col2:
                st.write(f"**{total}**")
        
        st.divider()
        
        if st.button("➕ Nouvelle pièce", use_container_width=True):
            st.session_state.page = "saisie"
            st.session_state.piece_selectionnee = None
            st.session_state.nom_scanne = ""
        
        st.divider()
        
        if gestionnaire.pieces:
            st.header("📊 Export")
            excel_file = gestionnaire.generer_excel()
            st.download_button(
                label="📥 Télécharger Excel",
                data=excel_file,
                file_name=f"inventaire_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
            
            if st.button("🔄 Tout réinitialiser", type="primary", use_container_width=True):
                gestionnaire.reinitialiser_tout()
                st.session_state.page = "saisie"
                st.session_state.piece_selectionnee = None
                st.session_state.nom_scanne = ""
                st.rerun()
    else:
        st.info("Aucune pièce pour le moment")

# ─── PAGE SAISIE (NOUVELLE PIÈCE) ───────────────────────────────────────────
if st.session_state.page == "saisie":
    st.header("➕ Ajouter une nouvelle pièce")

    if not BARCODE_AVAILABLE:
        st.warning("⚠️ La bibliothèque `pyzbar` n'est pas installée. Installez-la avec : `pip install pyzbar` et `libzbar0` (Linux) pour activer le scan de codes-barres.")

    # Tabs : Scanner | Saisie manuelle
    tab_scan, tab_manuel = st.tabs(["📷 Scanner un code-barres", "⌨️ Saisie manuelle"])

    # ── Onglet scan ──────────────────────────────────────────────────────────
    with tab_scan:
        if not BARCODE_AVAILABLE:
            st.error("La lecture de codes-barres n'est pas disponible. Utilisez la saisie manuelle.")
        else:
            st.markdown("Prenez une photo ou importez une image contenant le code-barres de la pièce.")
            
            source_scan = st.radio(
                "Source",
                ["📸 Prendre une photo", "🖼️ Importer une image"],
                horizontal=True,
                key="source_scan"
            )
            
            image_scan = None
            
            if source_scan == "📸 Prendre une photo":
                img_file = st.camera_input("Pointez la caméra vers le code-barres", key="cam_barcode")
                if img_file:
                    bytes_data = img_file.getvalue()
                    image_scan = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            else:
                uploaded = st.file_uploader("Choisir une image avec le code-barres", type=['jpg', 'jpeg', 'png'], key="up_barcode")
                if uploaded:
                    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
                    image_scan = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            if image_scan is not None:
                with st.spinner("🔍 Lecture du code-barres..."):
                    valeur, img_annotee = lire_code_barre(image_scan)
                
                col_img, col_info = st.columns([2, 1])
                with col_img:
                    st.image(cv2.cvtColor(img_annotee, cv2.COLOR_BGR2RGB),
                             caption="Résultat du scan", use_column_width=True)
                
                with col_info:
                    if valeur:
                        st.success(f"✅ Code-barres détecté !")
                        st.info(f"**Valeur :** `{valeur}`")
                        
                        # Pré-remplir le nom
                        st.session_state.nom_scanne = valeur
                        
                        # Permettre de modifier avant confirmation
                        nom_confirme = st.text_input(
                            "Nom de la pièce (modifiable)",
                            value=st.session_state.nom_scanne,
                            key="nom_confirme_scan"
                        )
                        
                        if st.button("✅ Créer la pièce", use_container_width=True, key="btn_creer_scan"):
                            nom_final = nom_confirme.strip()
                            if nom_final:
                                if gestionnaire.creer_nouvelle_piece(nom_final):
                                    st.success(f"✅ Pièce '{nom_final}' créée !")
                                    st.session_state.piece_selectionnee = nom_final
                                    st.session_state.page = "details"
                                    st.session_state.nom_scanne = ""
                                    st.rerun()
                                else:
                                    st.warning(f"⚠️ La pièce '{nom_final}' existe déjà. Sélectionnez-la dans la barre latérale.")
                            else:
                                st.error("❌ Le nom ne peut pas être vide.")
                    else:
                        st.error("❌ Aucun code-barres trouvé.")
                        st.markdown("""
                        **Conseils :**
                        - Assurez-vous que le code-barres est bien visible et net
                        - Éclairez correctement l'image
                        - Évitez les reflets
                        - Utilisez l'onglet **Saisie manuelle** en cas de difficulté
                        """)

    # ── Onglet saisie manuelle ────────────────────────────────────────────────
    with tab_manuel:
        with st.form("nouvelle_piece_manuelle"):
            nom_piece = st.text_input(
                "Nom de la pièce",
                placeholder="Ex: Vis M8, Écrou, Rondelle...",
                value=st.session_state.get("nom_scanne", "")
            )
            
            col1, col2 = st.columns(2)
            with col1:
                submitted = st.form_submit_button("✅ Créer la pièce", use_container_width=True)
            with col2:
                cancelled = st.form_submit_button("❌ Annuler", use_container_width=True)
        
        if submitted:
            if nom_piece:
                if gestionnaire.creer_nouvelle_piece(nom_piece):
                    st.success(f"✅ Pièce '{nom_piece}' créée avec succès!")
                    st.session_state.piece_selectionnee = nom_piece
                    st.session_state.page = "details"
                    st.session_state.nom_scanne = ""
                    st.rerun()
                else:
                    st.error("❌ Ce nom de pièce existe déjà ou est invalide")
            else:
                st.error("❌ Veuillez entrer un nom de pièce")

# ─── PAGE DÉTAILS D'UNE PIÈCE ────────────────────────────────────────────────
elif st.session_state.page == "details" and st.session_state.piece_selectionnee:
    nom_piece = st.session_state.piece_selectionnee
    photos = gestionnaire.get_photos_piece(nom_piece)
    total = gestionnaire.get_total_piece(nom_piece)
    
    col_h1, col_h2, col_h3 = st.columns([2, 1, 1])
    with col_h1:
        st.header(f"📦 {nom_piece}")
    with col_h2:
        st.metric("Total pièces", total)
    with col_h3:
        st.metric("Photos", len(photos))
    
    col_o1, col_o2, col_o3 = st.columns(3)
    with col_o1:
        if st.button("⬅️ Retour à la saisie", use_container_width=True):
            st.session_state.page = "saisie"
            st.rerun()
    with col_o2:
        if st.button("📸 Ajouter une photo", use_container_width=True):
            st.session_state.ajout_photo = True
    with col_o3:
        if st.button("🗑️ Supprimer cette pièce", use_container_width=True, type="primary"):
            if gestionnaire.supprimer_piece(nom_piece):
                st.success(f"✅ Pièce '{nom_piece}' supprimée")
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
                    
                    if gestionnaire.ajouter_photo_piece(nom_piece, frame, resultat, nb_pieces):
                        st.success(f"✅ {nb_pieces} pièces détectées et ajoutées!")
                        st.session_state.ajout_photo = False
                        st.rerun()
        else:
            uploaded_file = st.file_uploader("Choisir une image", type=['jpg', 'jpeg', 'png'])
            if uploaded_file:
                with st.spinner("Analyse..."):
                    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    resultat, nb_pieces = detecter_pieces(frame)
                    
                    if gestionnaire.ajouter_photo_piece(nom_piece, frame, resultat, nb_pieces):
                        st.success(f"✅ {nb_pieces} pièces détectées et ajoutées!")
                        st.session_state.ajout_photo = False
                        st.rerun()
    
    # Affichage des photos existantes
    if photos:
        st.subheader("📸 Photos enregistrées")
        
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            tri = st.selectbox("Trier par", ["Plus récente", "Plus ancienne", "Plus de pièces", "Moins de pièces"])
        
        photos_affichees = photos.copy()
        if tri == "Plus récente":
            photos_affichees = list(reversed(photos_affichees))
        elif tri == "Plus de pièces":
            photos_affichees = sorted(photos_affichees, key=lambda x: x['nb_pieces'], reverse=True)
        elif tri == "Moins de pièces":
            photos_affichees = sorted(photos_affichees, key=lambda x: x['nb_pieces'])
        
        cols = st.columns(3)
        for i, photo in enumerate(photos_affichees):
            with cols[i % 3]:
                img = base64_to_image(photo['image_analyse'])
                img_mini = cv2.resize(img, (200, 150))
                st.image(cv2.cvtColor(img_mini, cv2.COLOR_BGR2RGB), use_column_width=True)
                st.caption(f"📅 {photo['timestamp'][:10]}")
                st.caption(f"🔢 {photo['nb_pieces']} pièces")
                
                col_b1, col_b2 = st.columns(2)
                with col_b1:
                    if st.button("🔍 Voir", key=f"view_{nom_piece}_{i}"):
                        st.session_state.photo_selectionnee = photo['id']
                        st.session_state.page = "photo_detail"
                        st.rerun()
                with col_b2:
                    if st.button("🗑️", key=f"del_{nom_piece}_{i}"):
                        if gestionnaire.supprimer_photo(nom_piece, photo['id']):
                            st.rerun()
    else:
        st.info("📸 Aucune photo pour cette pièce. Cliquez sur 'Ajouter une photo' pour commencer.")

# ─── PAGE DÉTAIL D'UNE PHOTO ─────────────────────────────────────────────────
elif st.session_state.page == "photo_detail" and st.session_state.piece_selectionnee and st.session_state.photo_selectionnee is not None:
    nom_piece = st.session_state.piece_selectionnee
    photos = gestionnaire.get_photos_piece(nom_piece)
    photo_id = st.session_state.photo_selectionnee
    
    if 0 <= photo_id < len(photos):
        photo = photos[photo_id]
        
        st.header(f"🔍 Détail de la photo - {nom_piece}")
        
        col_img1, col_img2 = st.columns(2)
        with col_img1:
            st.subheader("📸 Image originale")
            img_originale = base64_to_image(photo['image_originale'])
            st.image(cv2.cvtColor(img_originale, cv2.COLOR_BGR2RGB), use_column_width=True)
        with col_img2:
            st.subheader(f"🔍 Analyse - {photo['nb_pieces']} pièces")
            img_analyse = base64_to_image(photo['image_analyse'])
            st.image(cv2.cvtColor(img_analyse, cv2.COLOR_BGR2RGB), use_column_width=True)
        
        st.metric("Nombre de pièces", photo['nb_pieces'])
        st.caption(f"Date: {photo['timestamp']}")
        
        col_b1, col_b2 = st.columns(2)
        with col_b1:
            if st.button("⬅️ Retour à la pièce", use_container_width=True):
                st.session_state.page = "details"
                st.session_state.photo_selectionnee = None
                st.rerun()
        with col_b2:
            if st.button("🗑️ Supprimer cette photo", use_container_width=True, type="primary"):
                if gestionnaire.supprimer_photo(nom_piece, photo_id):
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
col_f1, col_f2, col_f3 = st.columns(3)
with col_f1:
    st.caption("📦 Gestionnaire d'Inventaire v2.0 — Scan Code-Barres")
with col_f2:
    total_global = sum(gestionnaire.get_tous_les_totaux().values())
    st.caption(f"🧩 Total global: {total_global} pièces")
with col_f3:
    st.caption(f"📊 Types de pièces: {len(gestionnaire.pieces)}")
