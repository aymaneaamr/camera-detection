# Contenu principal
if st.session_state.page == "saisie":
    st.header("➕ Ajouter un nouvel article")
    
    # Section scan code-barres (inchangée)
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
                image_annotee, codes = detecter_code_barre(frame)
                if codes:
                    code_trouve = codes[0]['data']
                    st.session_state.code_detecte = code_trouve
                    st.session_state.scan_effectue = True
                    st.image(cv2.cvtColor(image_annotee, cv2.COLOR_BGR2RGB), caption="Code-barres détecté", use_container_width=True)
                    st.markdown(f"""
                    <div class="success-box">
                        <h4>✅ Code-barres détecté !</h4>
                        <div class="code-display">{code_trouve}</div>
                        <p><strong>Type :</strong> {codes[0]['type']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning("❌ Aucun code-barres détecté. Veuillez réessayer avec une image plus claire.")
    else:
        uploaded_barcode = st.file_uploader("Choisir une image de code-barres", type=['jpg', 'jpeg', 'png'], key="upload_barcode")
        if uploaded_barcode:
            with st.spinner("🔍 Analyse du code-barres..."):
                file_bytes = np.asarray(bytearray(uploaded_barcode.read()), dtype=np.uint8)
                frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                image_annotee, codes = detecter_code_barre(frame)
                st.image(cv2.cvtColor(image_annotee, cv2.COLOR_BGR2RGB), caption="Image analysée", use_container_width=True)
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
    
    if st.session_state.scan_effectue:
        if st.button("🔄 Nouveau scan", use_container_width=True):
            st.session_state.scan_effectue = False
            st.session_state.code_detecte = None
            st.rerun()
    
    st.markdown("---")
    
    # ==================== SOLUTION AVEC CALLBACK ====================
    st.markdown("### 📝 Informations de l'article")
    
    # Callback : met à jour le libellé dans session_state dès que le code change
    def on_code_change():
        code = st.session_state.code_article_input
        if code in ARTICLES_PREDEFINIS:
            st.session_state.libelle_input = ARTICLES_PREDEFINIS[code]
        else:
            st.session_state.libelle_input = ""
    
    # Initialisation des clés si elles n'existent pas
    if 'libelle_input' not in st.session_state:
        st.session_state.libelle_input = ""
    if 'code_article_input' not in st.session_state:
        st.session_state.code_article_input = st.session_state.code_detecte if st.session_state.code_detecte else ""
    
    col_code, col_lib, col_emp = st.columns([2, 2, 1])
    
    with col_code:
        code_article = st.text_input(
            "Code article *",
            value=st.session_state.code_article_input,
            placeholder="Code article (obligatoire)",
            key="code_article_input",
            on_change=on_code_change   # ← déclenché à chaque modification
        )
        # Afficher le message si article trouvé
        if code_article and code_article in ARTICLES_PREDEFINIS:
            st.markdown(f"""
            <div class="article-found">
                <strong>📝 Article trouvé :</strong> {ARTICLES_PREDEFINIS[code_article]}
            </div>
            """, unsafe_allow_html=True)
    
    with col_lib:
        libelle = st.text_input(
            "Libellé (optionnel)",
            placeholder="Description de l'article",
            key="libelle_input"   # la valeur est gérée via session_state
        )
    
    with col_emp:
        emplacement = st.text_input(
            "Emplacement (optionnel)",
            placeholder="Ex: A-12, Rayon 3...",
            key="emplacement_input"
        )
    
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
                    # Nettoyer les champs pour le prochain article
                    del st.session_state.code_article_input
                    del st.session_state.libelle_input
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
            if 'code_article_input' in st.session_state:
                del st.session_state.code_article_input
            if 'libelle_input' in st.session_state:
                del st.session_state.libelle_input
            st.rerun()
