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
import re
import sqlite3
import os

# ==================== Configuration de la page ====================
st.set_page_config(
    page_title="Gestionnaire d'Inventaire Multi-Pièces",
    page_icon="📦",
    layout="wide"
)

# ==================== Fonctions de persistance SQLite ====================

def init_database():
    conn = sqlite3.connect('inventaire.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS articles
                 (code TEXT PRIMARY KEY,
                  libelle TEXT,
                  emplacement TEXT,
                  date_creation TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS photos
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  code_article TEXT,
                  timestamp TEXT,
                  nb_pieces INTEGER,
                  image_originale TEXT,
                  image_analyse TEXT,
                  FOREIGN KEY (code_article) REFERENCES articles(code))''')
    conn.commit()
    conn.close()

def charger_donnees():
    gestionnaire = GestionnairePieces()
    conn = sqlite3.connect('inventaire.db')
    c = conn.cursor()
    c.execute("SELECT code, libelle, emplacement, date_creation FROM articles")
    for code, libelle, emplacement, date_creation in c.fetchall():
        gestionnaire.articles[code] = {
            'libelle': libelle, 'photos': [],
            'emplacement': emplacement, 'date_creation': date_creation
        }
    c.execute("SELECT code_article, timestamp, nb_pieces, image_originale, image_analyse FROM photos ORDER BY timestamp")
    for code_article, timestamp, nb_pieces, img_o, img_a in c.fetchall():
        if code_article in gestionnaire.articles:
            gestionnaire.articles[code_article]['photos'].append({
                'timestamp': timestamp, 'nb_pieces': nb_pieces,
                'image_originale': img_o, 'image_analyse': img_a,
                'id': len(gestionnaire.articles[code_article]['photos'])
            })
    conn.close()
    return gestionnaire

def sauvegarder_article(code, libelle, emplacement, date_creation):
    conn = sqlite3.connect('inventaire.db')
    conn.cursor().execute(
        "INSERT OR REPLACE INTO articles VALUES (?, ?, ?, ?)",
        (code, libelle, emplacement, date_creation))
    conn.commit(); conn.close()

def sauvegarder_photo(code_article, timestamp, nb_pieces, img_o, img_a):
    conn = sqlite3.connect('inventaire.db')
    conn.cursor().execute(
        "INSERT INTO photos (code_article,timestamp,nb_pieces,image_originale,image_analyse) VALUES (?,?,?,?,?)",
        (code_article, timestamp, nb_pieces, img_o, img_a))
    conn.commit(); conn.close()

def supprimer_article_db(code):
    conn = sqlite3.connect('inventaire.db')
    c = conn.cursor()
    c.execute("DELETE FROM photos WHERE code_article=?", (code,))
    c.execute("DELETE FROM articles WHERE code=?", (code,))
    conn.commit(); conn.close()

def supprimer_photo_db(photo_id):
    conn = sqlite3.connect('inventaire.db')
    conn.cursor().execute("DELETE FROM photos WHERE id=?", (photo_id,))
    conn.commit(); conn.close()

def get_photo_db_id(code_article, timestamp):
    conn = sqlite3.connect('inventaire.db')
    c = conn.cursor()
    c.execute("SELECT id FROM photos WHERE code_article=? AND timestamp=?", (code_article, timestamp))
    r = c.fetchone(); conn.close()
    return r[0] if r else None


# ==================== DÉTECTION AMÉLIORÉE ====================

def _draw_counter(image, count):
    label = f"Pieces: {count}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
    cv2.rectangle(image, (5, 5), (tw + 20, th + 20), (0, 0, 0), -1)
    cv2.putText(image, label, (10, th + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 80), 2)


def detecter_pieces_watershed(image):
    """
    Algorithme Watershed avec Distance Transform.
    Sépare correctement les pièces qui se touchent ou se chevauchent.
    Chaque région est colorée différemment et numérotée.
    """
    result = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 1. Double seuillage : Otsu + adaptatif (gère les éclairages variables)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thresh_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thresh_adapt = cv2.adaptiveThreshold(blur, 255,
                                         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, 11, 2)
    thresh = cv2.bitwise_or(thresh_otsu, thresh_adapt)

    # 2. Nettoyage morphologique : enlève le bruit, bouche les trous
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 3. Fond certain (dilatation forte)
    sure_bg = cv2.dilate(clean, kernel, iterations=3)

    # 4. Avant-plan certain via Distance Transform
    #    Chaque pixel = distance au bord le plus proche → les centres sont élevés
    dist = cv2.distanceTransform(clean, cv2.DIST_L2, 5)
    cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)

    # Seuil : 40 % du maximum → garde uniquement les centres des objets
    _, sure_fg = cv2.threshold(dist, 0.40 * dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # 5. Zone inconnue (entre fond et avant-plan)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # 6. Marqueurs numérotés par composante connexe
    nb_markers, markers = cv2.connectedComponents(sure_fg)
    markers += 1            # Fond = 1 (pas 0)
    markers[unknown == 255] = 0  # Zone inconnue = 0

    # 7. Watershed sépare les objets collés (frontières → -1)
    markers = cv2.watershed(image, markers)

    # 8. Palette de couleurs distinctes pour numéroter visuellement
    palette = [
        (0, 230, 0), (0, 180, 255), (255, 80, 0),
        (180, 0, 255), (0, 255, 180), (255, 200, 0),
        (0, 100, 255), (255, 0, 150), (100, 255, 0), (255, 130, 130),
    ]

    img_h, img_w = gray.shape
    min_area = max(200, (img_h * img_w) // 2000)   # adaptatif
    max_area = img_h * img_w * 0.70

    nb_pieces = 0
    for label in range(2, nb_markers + 1):
        mask = np.uint8(markers == label) * 255
        area = cv2.countNonZero(mask)

        if min_area < area < max_area:
            nb_pieces += 1
            color = palette[(nb_pieces - 1) % len(palette)]

            # Remplissage translucide
            overlay = result.copy()
            overlay[mask == 255] = color
            cv2.addWeighted(overlay, 0.25, result, 0.75, 0, result)

            # Contour
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(result, cnts, -1, color, 2)

            # Centroïde + numéro
            M = cv2.moments(mask)
            if M["m00"] > 0:
                cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                cv2.circle(result, (cx, cy), 5, color, -1)
                cv2.putText(result, str(nb_pieces), (cx - 8, cy + 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Frontières Watershed en rouge
    result[markers == -1] = [0, 0, 255]

    _draw_counter(result, nb_pieces)
    return result, nb_pieces


def detecter_pieces_hough(image):
    """
    Transformée de Hough circulaire.
    Idéal pour vis, boulons, pièces de monnaie, rondelles.
    Détecte même les cercles qui se touchent légèrement.
    """
    result = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 2)

    h, w = gray.shape
    min_r = max(10, min(h, w) // 40)
    max_r = min(h, w) // 3
    min_dist = max(min_r * 2, 20)

    circles = cv2.HoughCircles(
        blur, cv2.HOUGH_GRADIENT, dp=1.2,
        minDist=min_dist,
        param1=60, param2=28,
        minRadius=min_r, maxRadius=max_r
    )

    nb_pieces = 0
    palette = [
        (0, 230, 0), (0, 180, 255), (255, 80, 0),
        (180, 0, 255), (0, 255, 180), (255, 200, 0),
        (100, 255, 0), (255, 0, 150),
    ]

    if circles is not None:
        circles = np.round(circles[0, :]).astype(int)
        for x, y, r in circles:
            nb_pieces += 1
            color = palette[(nb_pieces - 1) % len(palette)]

            # Cercle rempli translucide
            overlay = result.copy()
            cv2.circle(overlay, (x, y), r, color, -1)
            cv2.addWeighted(overlay, 0.2, result, 0.8, 0, result)

            # Contour et numéro
            cv2.circle(result, (x, y), r, color, 2)
            cv2.circle(result, (x, y), 4, (0, 0, 255), -1)
            cv2.putText(result, str(nb_pieces), (x - 8, y + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    _draw_counter(result, nb_pieces)
    return result, nb_pieces


def detecter_pieces_contours(image):
    """
    Détection par contours externes.
    Efficace quand les pièces sont bien espacées.
    """
    result = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_area = image.shape[0] * image.shape[1]
    min_area = max(200, img_area // 2000)
    palette = [
        (0, 230, 0), (0, 180, 255), (255, 80, 0),
        (180, 0, 255), (0, 255, 180), (255, 200, 0),
    ]
    nb_pieces = 0
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if min_area < area < img_area * 0.70:
            nb_pieces += 1
            color = palette[(nb_pieces - 1) % len(palette)]
            overlay = result.copy()
            cv2.drawContours(overlay, [cnt], -1, color, -1)
            cv2.addWeighted(overlay, 0.2, result, 0.8, 0, result)
            cv2.drawContours(result, [cnt], -1, color, 2)
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                cv2.circle(result, (cx, cy), 5, (0, 0, 255), -1)
                cv2.putText(result, str(nb_pieces), (cx - 8, cy + 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    _draw_counter(result, nb_pieces)
    return result, nb_pieces


def detecter_pieces(image, mode="Watershed (séparation)"):
    if mode == "Watershed (séparation)":
        return detecter_pieces_watershed(image)
    elif mode == "Cercles Hough (pièces rondes)":
        return detecter_pieces_hough(image)
    else:
        return detecter_pieces_contours(image)


# ==================== Utilitaires ====================

def recadrer_selon_ratio(image, ratio):
    if ratio is None:
        return image
    h, w = image.shape[:2]
    if abs(w / h - ratio) < 0.01:
        return image
    if w / h > ratio:
        nw = int(h * ratio)
        return image[:, (w - nw) // 2:(w - nw) // 2 + nw]
    nh = int(w / ratio)
    return image[(h - nh) // 2:(h - nh) // 2 + nh, :]


def base64_to_image(b64):
    return cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_COLOR)


# ==================== GestionnairePieces ====================

class GestionnairePieces:
    def __init__(self):
        self.articles = {}

    def creer_nouvel_article(self, code, libelle="", emplacement=""):
        if code and code not in self.articles:
            dc = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.articles[code] = {'libelle': libelle, 'photos': [], 'emplacement': emplacement, 'date_creation': dc}
            sauvegarder_article(code, libelle, emplacement, dc)
            return True
        return False

    def nettoyer_articles_mal_importes(self):
        a = [c for c, d in self.articles.items()
             if any(k in d.get('libelle', '').upper()
                    for k in ['COLONNE', 'CODE ARTICLE', 'LIBELLÉ', 'EMPLACEMENT'])]
        for c in a:
            supprimer_article_db(c); del self.articles[c]
        return len(a)

    def importer_articles_excel(self, df, col_code, col_libelle, col_emplacement, skip_first_row=True):
        importes = existants = erreurs = 0
        pb = st.progress(0); st_txt = st.empty()
        s = 1 if skip_first_row else 0
        total = len(df) - s
        codes_vus = set()
        for idx in range(s, len(df)):
            pb.progress((idx - s + 1) / total)
            st_txt.text(f"Import… {idx - s + 1}/{total}")
            row = df.iloc[idx]
            try:
                cv_val = row[col_code]
                if pd.isna(cv_val) or str(cv_val).strip() == '': continue
                code = str(cv_val).strip()
                if code.lower() in ['code article', 'code', 'article', 'réf', 'ref']: continue
                if code in codes_vus: continue
                codes_vus.add(code)
                libelle = ""
                if col_libelle and col_libelle in row.index and pd.notna(row[col_libelle]):
                    libelle = str(row[col_libelle]).strip()
                    if libelle.lower() == 'none': libelle = ""
                emplacement = ""
                if col_emplacement and col_emplacement in row.index and pd.notna(row[col_emplacement]):
                    ev = str(row[col_emplacement]).strip()
                    if ev.lower() not in ['none', 'nan', '']: emplacement = ev
                if code not in self.articles:
                    dc = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    self.articles[code] = {'libelle': libelle, 'photos': [], 'emplacement': emplacement, 'date_creation': dc}
                    sauvegarder_article(code, libelle, emplacement, dc)
                    importes += 1
                else:
                    existants += 1
            except Exception:
                erreurs += 1
        pb.empty(); st_txt.empty()
        return importes, existants, erreurs

    def ajouter_photo_article(self, code, frame_orig, frame_analyse, nb_pieces):
        if code in self.articles:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            _, bo = cv2.imencode('.jpg', frame_orig)
            _, ba = cv2.imencode('.jpg', frame_analyse)
            io_b64 = base64.b64encode(bo).decode()
            ia_b64 = base64.b64encode(ba).decode()
            self.articles[code]['photos'].append({
                'timestamp': ts, 'nb_pieces': nb_pieces,
                'image_originale': io_b64, 'image_analyse': ia_b64,
                'id': len(self.articles[code]['photos'])
            })
            sauvegarder_photo(code, ts, nb_pieces, io_b64, ia_b64)
            return True
        return False

    def get_total_article(self, code):
        return sum(p['nb_pieces'] for p in self.articles.get(code, {}).get('photos', []))

    def get_photos_article(self, code):
        return self.articles.get(code, {}).get('photos', [])

    def get_emplacement_article(self, code):
        return self.articles.get(code, {}).get('emplacement', '')

    def get_libelle_article(self, code):
        return self.articles.get(code, {}).get('libelle', '')

    def supprimer_photo(self, code, photo_id):
        if code in self.articles and 0 <= photo_id < len(self.articles[code]['photos']):
            ts = self.articles[code]['photos'][photo_id]['timestamp']
            db_id = get_photo_db_id(code, ts)
            if db_id: supprimer_photo_db(db_id)
            del self.articles[code]['photos'][photo_id]
            for i, p in enumerate(self.articles[code]['photos']): p['id'] = i
            return True
        return False

    def supprimer_article(self, code):
        if code in self.articles:
            supprimer_article_db(code); del self.articles[code]; return True
        return False

    def get_tous_les_totaux(self): return {c: self.get_total_article(c) for c in self.articles}
    def get_tous_emplacements(self): return {c: self.get_emplacement_article(c) for c in self.articles}
    def get_tous_libelles(self): return {c: self.get_libelle_article(c) for c in self.articles}

    def generer_excel(self):
        output = BytesIO()
        wb = openpyxl.Workbook()
        ws = wb.active; ws.title = "Inventaire"
        headers = ["Code Article", "Libellé", "Emplacement", "Quantité totale", "Nombre de photos", "Dernière màj"]
        for col, h in enumerate(headers, 1):
            cell = ws.cell(row=1, column=col, value=h)
            cell.font = Font(color="FFFFFF", bold=True)
            cell.fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
            cell.alignment = Alignment(horizontal="center")
        row = 2
        for code, data in self.articles.items():
            total = sum(p['nb_pieces'] for p in data['photos'])
            derniere = data['photos'][-1]['timestamp'] if data['photos'] else data.get('date_creation', '')
            ws.cell(row=row, column=1, value=code)
            ws.cell(row=row, column=2, value=data.get('libelle', ''))
            ws.cell(row=row, column=3, value=data.get('emplacement', ''))
            ws.cell(row=row, column=4, value=total)
            ws.cell(row=row, column=5, value=len(data['photos']))
            ws.cell(row=row, column=6, value=derniere)
            row += 1
        for col, w in zip('ABCDEF', [20, 40, 20, 15, 15, 22]): ws.column_dimensions[col].width = w
        wd = wb.create_sheet("Détail des photos")
        for col, h in enumerate(["Code Article", "Libellé", "Emplacement", "Photo #", "Date", "Nb pièces"], 1):
            cell = wd.cell(row=1, column=col, value=h)
            cell.font = Font(bold=True)
            cell.fill = PatternFill(start_color="92D050", end_color="92D050", fill_type="solid")
        row = 2
        for code, data in self.articles.items():
            for i, photo in enumerate(data['photos'], 1):
                wd.cell(row=row, column=1, value=code)
                wd.cell(row=row, column=2, value=data.get('libelle', ''))
                wd.cell(row=row, column=3, value=data.get('emplacement', ''))
                wd.cell(row=row, column=4, value=f"Photo {i}")
                wd.cell(row=row, column=5, value=photo['timestamp'])
                wd.cell(row=row, column=6, value=photo['nb_pieces'])
                row += 1
        for col, w in zip('ABCDEF', [20, 40, 20, 12, 22, 18]): wd.column_dimensions[col].width = w
        wb.save(output); output.seek(0)
        return output

    def reinitialiser_tout(self):
        if os.path.exists('inventaire.db'): os.remove('inventaire.db')
        self.articles = {}


# ==================== CSS ====================
st.markdown("""
<style>
    .success-box { background:#d4edda;color:#155724;padding:1rem;border-radius:5px;border-left:5px solid #28a745;margin:1rem 0; }
    .location-badge { background:#17a2b8;color:white;padding:.2rem .5rem;border-radius:5px;font-size:.8rem;margin-left:.5rem; }
    .label-badge { background:#28a745;color:white;padding:.2rem .5rem;border-radius:5px;font-size:.8rem;margin-left:.5rem; }
    .import-section { background:#f8f9fa;padding:1.5rem;border-radius:10px;border:2px dashed #6c757d;margin:1rem 0; }
    .warning-box { background:#fff3cd;color:#856404;padding:1rem;border-radius:5px;border-left:5px solid #ffc107;margin:1rem 0;font-weight:bold; }
    .database-info { background:#d1ecf1;color:#0c5460;padding:.5rem;border-radius:5px;border-left:5px solid #17a2b8;margin:.5rem 0;font-size:.9rem; }
    .algo-card { padding:1rem;border-radius:8px;margin:.5rem 0;font-size:.85rem;line-height:1.5; }
    .algo-watershed { background:#e8f5e9;border-left:4px solid #4caf50; }
    .algo-hough { background:#e3f2fd;border-left:4px solid #2196f3; }
    .algo-contours { background:#fff3e0;border-left:4px solid #ff9800; }
</style>
""", unsafe_allow_html=True)

# ==================== Init session ====================
init_database()
for k, v in [('gestionnaire', None), ('page', 'saisie'), ('article_selectionne', None),
             ('photo_selectionnee', None), ('show_import', False),
             ('photo_temp', None), ('ajout_photo', False), ('search_query', '')]:
    if k not in st.session_state:
        st.session_state[k] = charger_donnees() if k == 'gestionnaire' else v

gestionnaire = st.session_state.gestionnaire

# ==================== Bandeaux ====================
if len(gestionnaire.articles) > 0:
    st.markdown('<div class="warning-box">⚠️ <strong>Attention :</strong> Exportez en Excel avant de quitter !</div>', unsafe_allow_html=True)
st.markdown('<div class="database-info">💾 <strong>Persistance active :</strong> Données sauvegardées automatiquement (SQLite)</div>', unsafe_allow_html=True)

st.title("📦 Gestionnaire d'Inventaire Multi-Pièces")

# ==================== Sidebar ====================
with st.sidebar:
    st.header("📋 Articles")
    if st.button("📥 Importer Excel", use_container_width=True):
        st.session_state.show_import = True; st.rerun()

    if gestionnaire.articles:
        st.write(f"**{len(gestionnaire.articles)} articles**")
        if st.button("🧹 Nettoyer mal importés", use_container_width=True):
            n = gestionnaire.nettoyer_articles_mal_importes()
            st.success(f"{n} supprimés" if n else "Rien à nettoyer")
            if n: st.rerun()

        search_query = st.text_input("🔍 Rechercher", value=st.session_state.search_query,
                                     placeholder="Code, libellé, emplacement…", key="search_input").lower().strip()
        st.session_state.search_query = search_query

        codes = sorted([c for c, d in gestionnaire.articles.items()
                        if not search_query
                        or search_query in c.lower()
                        or search_query in d.get('libelle', '').lower()
                        or search_query in d.get('emplacement', '').lower()])
        if not codes: st.info("Aucun article trouvé")
        for code in codes:
            total = gestionnaire.get_total_article(code)
            libelle = gestionnaire.get_libelle_article(code)
            emplacement = gestionnaire.get_emplacement_article(code)
            c1, c2 = st.columns([3, 1])
            with c1:
                if st.button(f"📦 {code}", key=f"sel_{code}", use_container_width=True):
                    st.session_state.article_selectionne = code
                    st.session_state.page = "details"; st.rerun()
            with c2:
                st.write(f"**{total}**")
            parts = []
            if libelle: parts.append(f"📝 {libelle[:25]}{'…' if len(libelle) > 25 else ''}")
            if emplacement: parts.append(f"📍 {emplacement}")
            if parts: st.caption(" | ".join(parts))
        st.divider()

    if st.button("➕ Nouvel article", use_container_width=True):
        st.session_state.page = "saisie"
        st.session_state.article_selectionne = None; st.rerun()

    st.divider()
    if gestionnaire.articles:
        st.header("📊 Export")
        st.download_button("📥 Télécharger Excel", gestionnaire.generer_excel(),
                           f"inventaire_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           use_container_width=True)
        if st.button("🔄 Réinitialiser tout", type="primary", use_container_width=True):
            gestionnaire.reinitialiser_tout()
            st.session_state.page = "saisie"; st.session_state.article_selectionne = None; st.rerun()
    else:
        st.info("Aucun article")

# ==================== Import Excel ====================
if st.session_state.show_import:
    st.markdown("---")
    st.markdown('<div class="import-section">', unsafe_allow_html=True)
    st.header("📥 Importer depuis Excel")
    uploaded_excel = st.file_uploader("Choisir un fichier Excel", type=['xlsx', 'xls'], key="import_excel")
    if uploaded_excel:
        try:
            df = pd.read_excel(uploaded_excel)
            st.subheader("Aperçu"); st.dataframe(df.head(10))
            cols = df.columns.tolist()
            c1, c2, c3 = st.columns(3)
            with c1: col_code = st.selectbox("📌 CODE *", cols, 0)
            with c2: col_libelle = st.selectbox("📝 LIBELLÉ", ["(Aucune)"] + cols, 2)
            with c3: col_emplacement = st.selectbox("📍 EMPLACEMENT", ["(Aucune)"] + cols, 3)
            skip_first = st.checkbox("Ignorer la 1ère ligne (en-têtes)", True)
            s = 1 if skip_first else 0
            preview = {'Code': df[col_code].iloc[s:].values}
            if col_libelle != "(Aucune)": preview['Libellé'] = df[col_libelle].iloc[s:].values
            if col_emplacement != "(Aucune)": preview['Emplacement'] = df[col_emplacement].iloc[s:].values
            st.subheader("Données à importer"); st.dataframe(pd.DataFrame(preview))
            if st.button("✅ Confirmer l'import", use_container_width=True, type="primary"):
                with st.spinner("Import…"):
                    imp, ex, err = gestionnaire.importer_articles_excel(
                        df, col_code,
                        col_libelle if col_libelle != "(Aucune)" else None,
                        col_emplacement if col_emplacement != "(Aucune)" else None,
                        skip_first)
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("✅ Importés", imp); c2.metric("⚠️ Existants", ex)
                c3.metric("❌ Erreurs", err); c4.metric("📊 Total", len(gestionnaire.articles))
                if imp > 0:
                    st.success(f"✅ {imp} articles importés !"); st.balloons()
                    st.session_state.show_import = False; st.rerun()
        except Exception as e:
            st.error(f"Erreur : {e}")
    if st.button("❌ Fermer", use_container_width=True):
        st.session_state.show_import = False; st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# ==================== Page saisie ====================
elif st.session_state.page == "saisie" and not st.session_state.show_import:
    st.header("➕ Ajouter un article")
    c1, c2, c3 = st.columns([2, 2, 1])
    with c1: code_article = st.text_input("Code article *", placeholder="Obligatoire")
    with c2: libelle = st.text_input("Libellé", placeholder="Description")
    with c3: emplacement = st.text_input("Emplacement", placeholder="Ex: A-12")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("✅ Créer", use_container_width=True):
            if code_article:
                if gestionnaire.creer_nouvel_article(code_article, libelle, emplacement):
                    st.session_state.article_selectionne = code_article
                    st.session_state.page = "details"; st.rerun()
                else: st.error("❌ Code déjà existant")
            else: st.error("❌ Code obligatoire")
    with c2:
        if st.button("❌ Annuler", use_container_width=True): st.rerun()

# ==================== Page détails ====================
elif st.session_state.page == "details" and st.session_state.article_selectionne:
    code_article = st.session_state.article_selectionne
    photos = gestionnaire.get_photos_article(code_article)
    total = gestionnaire.get_total_article(code_article)
    libelle = gestionnaire.get_libelle_article(code_article)
    emplacement = gestionnaire.get_emplacement_article(code_article)

    c1, c2, c3 = st.columns([3, 1, 1])
    with c1:
        st.header(f"📦 {code_article}")
        if libelle: st.markdown(f"<span class='label-badge'>📝 {libelle}</span>", unsafe_allow_html=True)
        if emplacement: st.markdown(f"<span class='location-badge'>📍 {emplacement}</span>", unsafe_allow_html=True)
    with c2: st.metric("Total pièces", total)
    with c3: st.metric("Photos", len(photos))

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("⬅️ Retour", use_container_width=True):
            st.session_state.page = "saisie"; st.rerun()
    with c2:
        if st.button("📸 Ajouter une photo", use_container_width=True):
            st.session_state.ajout_photo = True; st.session_state.photo_temp = None; st.rerun()
    with c3:
        if st.button("🗑️ Supprimer l'article", use_container_width=True, type="primary"):
            if gestionnaire.supprimer_article(code_article):
                st.session_state.page = "saisie"; st.rerun()

    st.divider()

    # ── Ajout de photo ──────────────────────────────────────────────
    if st.session_state.get('ajout_photo', False):
        st.subheader("📸 Ajouter une photo")

        # Guide de sélection d'algorithme
        with st.expander("ℹ️ Quel algorithme choisir ?", expanded=False):
            st.markdown("""
<div class="algo-card algo-watershed">
🌊 <strong>Watershed (séparation)</strong> — <em>Recommandé par défaut</em><br>
Utilise la Distance Transform pour trouver le centre de chaque objet, puis "inonde" depuis ces centres pour séparer les régions.
<b>→ Idéal quand des pièces se touchent ou se chevauchent.</b>
</div>

<div class="algo-card algo-hough">
⭕ <strong>Cercles Hough</strong> — <em>Pièces rondes</em><br>
Vote mathématique sur les gradients d'image pour détecter les cercles.
<b>→ Idéal pour : vis, boulons, rondelles, pièces de monnaie.</b>
</div>

<div class="algo-card algo-contours">
🔷 <strong>Contours</strong> — <em>Objets bien séparés</em><br>
Seuillage + détection des bords externes. Simple et rapide.
<b>→ Idéal quand les pièces ont de l'espace entre elles.</b>
</div>
            """, unsafe_allow_html=True)

        col_left, col_right = st.columns([2, 1])
        with col_right:
            if st.button("❌ Annuler"):
                st.session_state.ajout_photo = False; st.session_state.photo_temp = None; st.rerun()
        with col_left:
            algo_mode = st.selectbox(
                "🔬 Algorithme de détection",
                ["Watershed (séparation)", "Cercles Hough (pièces rondes)", "Contours (objets séparés)"],
                index=0)

        source = st.radio("Source", ["📸 Prendre une photo", "🖼️ Choisir une image"], horizontal=True)
        img_file = (st.camera_input("Prendre une photo", key="cam_photo")
                    if source == "📸 Prendre une photo"
                    else st.file_uploader("Choisir une image", type=['jpg', 'jpeg', 'png'], key="up_photo"))

        if img_file is not None:
            frame_brut = cv2.imdecode(np.frombuffer(img_file.getvalue(), np.uint8), cv2.IMREAD_COLOR)
            st.session_state.photo_temp = {
                'brut': frame_brut, 'format_choisi': "Original",
                'algo': algo_mode, 'recadree': None, 'analyse': None, 'detected': 0
            }

        if st.session_state.photo_temp is not None:
            temp = st.session_state.photo_temp
            frame_brut = temp['brut']

            fmt = st.selectbox("Format d'image", ["Original", "4:3", "16:9"],
                               index=["Original", "4:3", "16:9"].index(temp.get('format_choisi', "Original")))

            # Recalculer si algo, format ou image a changé
            if fmt != temp.get('format_choisi') or algo_mode != temp.get('algo') or temp.get('analyse') is None:
                temp['format_choisi'] = fmt; temp['algo'] = algo_mode
                ratio_map = {"4:3": 4/3, "16:9": 16/9, "Original": None}
                frame_rec = recadrer_selon_ratio(frame_brut, ratio_map[fmt])
                temp['recadree'] = frame_rec
                with st.spinner(f"Analyse avec {algo_mode}…"):
                    res, nb = detecter_pieces(frame_rec, algo_mode)
                temp['analyse'] = res; temp['detected'] = nb

            # Affichage côte à côte
            col_o, col_a = st.columns(2)
            with col_o:
                st.image(cv2.cvtColor(temp['recadree'], cv2.COLOR_BGR2RGB),
                         caption="📷 Image originale", use_container_width=True)
            with col_a:
                st.image(cv2.cvtColor(temp['analyse'], cv2.COLOR_BGR2RGB),
                         caption=f"🔍 Analyse — {temp['detected']} pièce(s) détectée(s)", use_container_width=True)

            st.markdown(f"### 🔢 Résultat : **{temp['detected']} pièce(s)** détectée(s)")

            st.markdown("### ⚙️ Ajustement du comptage")
            col_op1, col_op2, col_op3 = st.columns(3)
            with col_op1:
                operation = st.selectbox("Opération",
                                         ["Utiliser détection", "Remplacer", "Additionner", "Multiplier"])
            with col_op2:
                manuel = st.number_input("Valeur manuelle", min_value=0, value=0, step=1)
            with col_op3:
                st.write(""); st.write("")
                if st.button("✅ Ajouter cette photo", use_container_width=True):
                    d = temp['detected']
                    final = {"Utiliser détection": d, "Remplacer": manuel if manuel > 0 else d,
                             "Additionner": d + manuel, "Multiplier": d * manuel if manuel > 0 else d}[operation]
                    if gestionnaire.ajouter_photo_article(code_article, temp['recadree'], temp['analyse'], final):
                        st.success(f"✅ Photo ajoutée : {final} pièces !")
                        st.session_state.ajout_photo = False; st.session_state.photo_temp = None; st.rerun()

    # ── Photos existantes ───────────────────────────────────────────
    if photos:
        st.subheader("📸 Photos enregistrées")
        tri = st.selectbox("Trier par", ["Plus récente", "Plus ancienne", "Plus de pièces", "Moins de pièces"])
        photos_aff = (list(reversed(photos)) if tri == "Plus récente"
                      else photos if tri == "Plus ancienne"
                      else sorted(photos, key=lambda x: x['nb_pieces'], reverse=(tri == "Plus de pièces")))
        cols = st.columns(3)
        for i, photo in enumerate(photos_aff):
            with cols[i % 3]:
                img = base64_to_image(photo['image_analyse'])
                st.image(cv2.cvtColor(cv2.resize(img, (200, 150)), cv2.COLOR_BGR2RGB), use_container_width=True)
                st.caption(f"📅 {photo['timestamp'][:10]}")
                st.caption(f"🔢 {photo['nb_pieces']} pièces")
                b1, b2 = st.columns(2)
                with b1:
                    if st.button("🔍 Voir", key=f"v_{code_article}_{i}"):
                        st.session_state.photo_selectionnee = photo['id']
                        st.session_state.page = "photo_detail"; st.rerun()
                with b2:
                    if st.button("🗑️", key=f"d_{code_article}_{i}"):
                        gestionnaire.supprimer_photo(code_article, photo['id']); st.rerun()
    else:
        st.info("📸 Aucune photo. Cliquez sur 'Ajouter une photo' pour commencer.")

# ==================== Détail photo ====================
elif st.session_state.page == "photo_detail" and st.session_state.article_selectionne and st.session_state.photo_selectionnee is not None:
    code_article = st.session_state.article_selectionne
    photos = gestionnaire.get_photos_article(code_article)
    photo_id = st.session_state.photo_selectionnee
    if 0 <= photo_id < len(photos):
        photo = photos[photo_id]
        st.header(f"🔍 {code_article} – Photo #{photo_id + 1}")
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("📸 Originale")
            st.image(cv2.cvtColor(base64_to_image(photo['image_originale']), cv2.COLOR_BGR2RGB), use_container_width=True)
        with c2:
            st.subheader(f"🔍 Analyse – {photo['nb_pieces']} pièces")
            st.image(cv2.cvtColor(base64_to_image(photo['image_analyse']), cv2.COLOR_BGR2RGB), use_container_width=True)
        st.metric("Pièces", photo['nb_pieces'])
        st.caption(f"Date : {photo['timestamp']}")
        b1, b2 = st.columns(2)
        with b1:
            if st.button("⬅️ Retour", use_container_width=True):
                st.session_state.page = "details"; st.session_state.photo_selectionnee = None; st.rerun()
        with b2:
            if st.button("🗑️ Supprimer", use_container_width=True, type="primary"):
                gestionnaire.supprimer_photo(code_article, photo_id)
                st.session_state.page = "details"; st.session_state.photo_selectionnee = None; st.rerun()
    else:
        st.error("Photo introuvable")
        if st.button("Retour"):
            st.session_state.page = "details"; st.session_state.photo_selectionnee = None; st.rerun()

# ==================== Footer ====================
st.markdown("---")
c1, c2, c3, c4, c5 = st.columns(5)
c1.caption("📦 Gestionnaire Inventaire")
c2.caption(f"🧩 Total : {sum(gestionnaire.get_tous_les_totaux().values())} pièces")
c3.caption(f"📊 Articles : {len(gestionnaire.articles)}")
c4.caption(f"📍 Emplacements : {sum(1 for e in gestionnaire.get_tous_emplacements().values() if e)}/{len(gestionnaire.articles)}")
c5.caption(f"📝 Libellés : {sum(1 for l in gestionnaire.get_tous_libelles().values() if l)}/{len(gestionnaire.articles)}")
