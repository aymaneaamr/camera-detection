# Ajoutez cette nouvelle fonction après les fonctions d'import/export existantes
def afficher_tableau_articles(df_articles):
    """Affiche un tableau stylisé des articles"""
    if df_articles.empty:
        return None
    
    # Style du tableau
    st.markdown("""
    <style>
    .dataframe {
        font-size: 14px;
        border-collapse: collapse;
        width: 100%;
    }
    .dataframe th {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        padding: 12px;
        text-align: left;
    }
    .dataframe td {
        padding: 8px;
        border-bottom: 1px solid #ddd;
    }
    .dataframe tr:hover {
        background-color: #f5f5f5;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Afficher le tableau avec des couleurs alternées
    st.dataframe(
        df_articles,
        use_container_width=True,
        height=400,
        column_config={
            "Code": st.column_config.TextColumn("Code Article", width="medium"),
            "Libellé": st.column_config.TextColumn("Libellé", width="large"),
            "Emplacement": st.column_config.TextColumn("Emplacement", width="medium")
        }
    )

# Ajoutez cette section dans la barre latérale, après la section "Base articles prédéfinis"
# et avant le pied de page

with st.sidebar:
    # ... (code existant) ...
    
    # ==================== SECTION TABLEAU DES ARTICLES ====================
    st.divider()
    st.header("📋 Tableau des articles")
    
    # Créer un DataFrame à partir du dictionnaire ARTICLES_PREDEFINIS
    articles_data = []
    for code, infos in ARTICLES_PREDEFINIS.items():
        articles_data.append({
            "Code": code,
            "Libellé": infos["libelle"],
            "Emplacement": infos["emplacement"]
        })
    
    df_articles = pd.DataFrame(articles_data)
    
    if not df_articles.empty:
        # Options d'affichage
        col_view1, col_view2 = st.columns(2)
        with col_view1:
            recherche = st.text_input("🔍 Rechercher", placeholder="Code ou libellé...")
        with col_view2:
            tri = st.selectbox("📊 Trier par", ["Code", "Libellé", "Emplacement"])
        
        # Filtrer les résultats
        if recherche:
            df_filtre = df_articles[
                df_articles['Code'].str.contains(recherche, case=False, na=False) |
                df_articles['Libellé'].str.contains(recherche, case=False, na=False) |
                df_articles['Emplacement'].str.contains(recherche, case=False, na=False)
            ]
        else:
            df_filtre = df_articles
        
        # Trier
        df_filtre = df_filtre.sort_values(by=tri)
        
        # Afficher le compteur
        st.caption(f"📊 {len(df_filtre)} articles sur {len(df_articles)}")
        
        # Afficher le tableau
        if not df_filtre.empty:
            afficher_tableau_articles(df_filtre)
            
            # Option pour télécharger le tableau
            csv = df_filtre.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Télécharger le tableau (CSV)",
                data=csv,
                file_name=f"articles_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.info("Aucun résultat trouvé")
    else:
        st.info("Aucun article dans la base")
    
    # ==================== STATISTIQUES DES ARTICLES ====================
    st.divider()
    st.header("📊 Statistiques")
    
    if not df_articles.empty:
        col_stat1, col_stat2, col_stat3 = st.columns(3)
        with col_stat1:
            st.metric("Total articles", len(df_articles))
        with col_stat2:
            emplacements_renseignes = df_articles['Emplacement'].notna().sum()
            st.metric("Emplacements", f"{emplacements_renseignes}/{len(df_articles)}")
        with col_stat3:
            libelles_renseignes = df_articles['Libellé'].notna().sum()
            st.metric("Libellés", f"{libelles_renseignes}/{len(df_articles)}")
        
        # Top emplacements
        st.subheader("📍 Top emplacements")
        top_emplacements = df_articles['Emplacement'].value_counts().head(5)
        for emp, count in top_emplacements.items():
            if emp and emp != '':
                st.caption(f"{emp}: {count} article{'s' if count > 1 else ''}")
    
    # ==================== ACCÈS RAPIDE ====================
    st.divider()
    st.header("⚡ Accès rapide")
    
    # Sélecteur d'article
    if not df_articles.empty:
        article_selection = st.selectbox(
            "Sélectionner un article",
            options=df_articles['Code'].tolist(),
            format_func=lambda x: f"{x} - {df_articles[df_articles['Code']==x]['Libellé'].values[0][:30]}..."
        )
        
        if article_selection and st.button("📦 Voir les détails", use_container_width=True):
            # Pré-remplir le formulaire avec l'article sélectionné
            st.session_state.code_detecte = article_selection
            st.session_state.page = "saisie"
            st.rerun()
    
    # =====================================================================
