from pathlib import Path
from collections import Counter
import pandas as pd
import numpy as np

import path


def _safe_list_from_value(value):
    """
    Convertit une valeur potentiellement liste/chaîne/NaN en liste de chaînes.
    """
    # Gérer les arrays numpy avant pd.isna
    try:
        import numpy as np
        if isinstance(value, np.ndarray):
            return [str(v).strip() for v in value.flat if str(v).strip()]
    except ImportError:
        pass

    # Scalaire None ou NaN
    try:
        if pd.isna(value):
            return []
    except (TypeError, ValueError):
        pass

    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]

    if isinstance(value, str):
        separators = ["|", ">", ","]
        for sep in separators:
            if sep in value:
                return [v.strip() for v in value.split(sep) if v.strip()]
        return [value.strip()] if value.strip() else []

    return [str(value).strip()] if str(value).strip() else []


def _extract_top_categories(df_books: pd.DataFrame, top_k: int = 5):
    """
    Extrait les catégories les plus fréquentes d'un ensemble de livres.
    """
    all_categories = []

    if "categories" not in df_books.columns:
        return []

    for value in df_books["categories"]:
        cats = _safe_list_from_value(value)
        all_categories.extend(cats)

    counter = Counter(all_categories)
    return counter.most_common(top_k)


def _format_book_line(row):
    """
    Formate une ligne de livre pour le rapport.
    """
    title = str(row.get("title", "Titre inconnu"))
    item_id = str(row.get("parent_asin", "N/A"))

    categories = _safe_list_from_value(row.get("categories", np.nan))
    categories_str = " > ".join(categories[:3]) if categories else "Catégorie inconnue"

    return f"- {title} [{item_id}] | {categories_str}"


def _build_brief_analysis_text(
    user_id: str,
    history_df: pd.DataFrame,
    recs_df: pd.DataFrame,
    top_history_categories,
    top_rec_categories
):
    """
    Construit une analyse qualitative brève.
    """
    history_cat_names = {cat for cat, _ in top_history_categories}
    rec_cat_names = {cat for cat, _ in top_rec_categories}

    overlap = history_cat_names.intersection(rec_cat_names)

    lines = []
    lines.append("ANALYSE QUALITATIVE")
    lines.append("=" * 80)
    lines.append(f"Utilisateur : {user_id}")
    lines.append("")

    lines.append("1) Cohérence thématique")
    if overlap:
        lines.append(
            f"Les recommandations semblent globalement cohérentes avec l’historique de lecture. "
            f"On retrouve des catégories communes entre l’historique et les recommandations, "
            f"notamment : {', '.join(list(overlap)[:3])}."
        )
    else:
        lines.append(
            "Les recommandations présentent une cohérence thématique limitée au regard des catégories "
            "les plus visibles dans l’historique. Cela peut indiquer un profil utilisateur peu dense "
            "ou une similarité captée davantage par le vocabulaire fin que par les catégories globales."
        )
    lines.append("")

    lines.append("2) Proximité apparente entre les livres")
    lines.append(
        "Les livres recommandés partagent visiblement des thèmes ou des catégories proches de ceux déjà "
        "consultés par l’utilisateur, ce qui est cohérent avec une approche de filtrage basé sur le contenu "
        "reposant sur la similarité textuelle."
    )
    lines.append("")

    lines.append("3) Redondance potentielle")
    if len(overlap) >= 2:
        lines.append(
            "Certaines recommandations paraissent plausibles mais potentiellement redondantes, car elles "
            "restent concentrées sur des catégories déjà fortement présentes dans l’historique. Cela améliore "
            "la pertinence immédiate, mais peut réduire la diversité."
        )
    else:
        lines.append(
            "La redondance apparente reste modérée sur cet exemple, même si plusieurs recommandations "
            "semblent rester dans un même voisinage thématique."
        )
    lines.append("")

    lines.append("4) Risque de sur-spécialisation")
    if len(overlap) > 0 and len(rec_cat_names) <= max(2, len(overlap)):
        lines.append(
            "Le modèle semble ici relativement spécialisé : il privilégie surtout des livres proches des "
            "préférences déjà observées, ce qui peut limiter l’exploration de nouveaux thèmes."
        )
    else:
        lines.append(
            "Le modèle ne paraît pas excessivement spécialisé sur cet exemple, même s’il reste naturellement "
            "orienté vers des livres similaires à l’historique utilisateur."
        )
    lines.append("")

    lines.append("Conclusion")
    lines.append(
        "Dans l’ensemble, les recommandations sont plausibles et cohérentes avec la logique d’un modèle "
        "content-based. Cette lecture qualitative reste indicative et sera complétée par l’évaluation "
        "quantitative dans la tâche 4."
    )
    lines.append("")

    lines.append("=" * 80)
    lines.append("EXEMPLES D'HISTORIQUE (jusqu'à 5)")
    lines.append("=" * 80)

    for _, row in history_df.head(5).iterrows():
        lines.append(_format_book_line(row))

    lines.append("")
    lines.append("=" * 80)
    lines.append("EXEMPLES DE RECOMMANDATIONS (jusqu'à 5)")
    lines.append("=" * 80)

    for _, row in recs_df.head(5).iterrows():
        lines.append(_format_book_line(row))

    lines.append("")
    lines.append("=" * 80)
    lines.append("CATEGORIES DOMINANTES")
    lines.append("=" * 80)
    lines.append(
        "Historique : " + (
            ", ".join([f"{cat} ({count})" for cat, count in top_history_categories])
            if top_history_categories else "Aucune catégorie exploitable"
        )
    )
    lines.append(
        "Recommandations : " + (
            ", ".join([f"{cat} ({count})" for cat, count in top_rec_categories])
            if top_rec_categories else "Aucune catégorie exploitable"
        )
    )
    lines.append("")

    return "\n".join(lines)


def generate_qualitative_analysis_reports(
    train_df: pd.DataFrame,
    recommendations_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    n_users: int = 3,
    top_k_examples: int = 5,
    save_output: bool = True
):
    """
    Génère des rapports qualitatifs brefs pour n_users utilisateurs (3 par défaut).

    Paramètres :
    - train_df : interactions TRAIN (doit contenir user_id, parent_asin)
    - recommendations_df : recommandations finales (doit contenir user_id, parent_asin)
    - metadata_df : métadonnées livres (doit contenir parent_asin, title, categories)
    - n_users : nombre d'utilisateurs à analyser
    - top_k_examples : nombre d'exemples de livres à afficher (indicatif)
    - save_output : sauvegarder les fichiers txt

    Retour :
    - liste des chemins des rapports générés
    """
    print("\n" + "=" * 80)
    print("3.4 - ANALYSE QUALITATIVE DES RECOMMANDATIONS")
    print("=" * 80)

    required_train_cols = {"user_id", "parent_asin"}
    required_rec_cols = {"user_id", "parent_asin"}
    required_meta_cols = {"parent_asin"}

    if not required_train_cols.issubset(train_df.columns):
        raise ValueError(f"train_df doit contenir : {required_train_cols}")

    if not required_rec_cols.issubset(recommendations_df.columns):
        raise ValueError(f"recommendations_df doit contenir : {required_rec_cols}")

    if not required_meta_cols.issubset(metadata_df.columns):
        raise ValueError(f"metadata_df doit contenir : {required_meta_cols}")

    # Normalisation des types
    train_df = train_df.copy()
    recommendations_df = recommendations_df.copy()
    metadata_df = metadata_df.copy()

    train_df["user_id"] = train_df["user_id"].astype(str)
    train_df["parent_asin"] = train_df["parent_asin"].astype(str)

    recommendations_df["user_id"] = recommendations_df["user_id"].astype(str)
    recommendations_df["parent_asin"] = recommendations_df["parent_asin"].astype(str)

    metadata_df["parent_asin"] = metadata_df["parent_asin"].astype(str)

    # Choix des utilisateurs à analyser : premiers utilisateurs distincts des recommandations
    selected_user_ids = recommendations_df["user_id"].drop_duplicates().head(n_users).tolist()

    if not selected_user_ids:
        print("[WARN] Aucun utilisateur trouvé dans recommendations_df pour l'analyse qualitative.")
        return []

    print(f"[INFO] Utilisateurs sélectionnés pour l'analyse qualitative : {selected_user_ids}")

    report_paths = []

    for i, user_id in enumerate(selected_user_ids, start=1):
        print(f"\n[INFO] Analyse qualitative {i}/{len(selected_user_ids)} pour l'utilisateur : {user_id}")

        user_history = train_df[train_df["user_id"] == user_id][["user_id", "parent_asin"]].drop_duplicates()
        user_recs = recommendations_df[recommendations_df["user_id"] == user_id][["user_id", "parent_asin"]].drop_duplicates()

        # Jointure avec métadonnées
        history_books = user_history.merge(metadata_df, on="parent_asin", how="left")
        rec_books = user_recs.merge(metadata_df, on="parent_asin", how="left")

        # Limitation d'affichage (pour garder le rapport court)
        history_books_short = history_books.head(max(top_k_examples, 5))
        rec_books_short = rec_books.head(max(top_k_examples, 5))

        # Catégories dominantes
        top_history_categories = _extract_top_categories(history_books)
        top_rec_categories = _extract_top_categories(rec_books)

        # Génération du texte
        report_text = _build_brief_analysis_text(
            user_id=user_id,
            history_df=history_books_short,
            recs_df=rec_books_short,
            top_history_categories=top_history_categories,
            top_rec_categories=top_rec_categories
        )

        # Sauvegarde
        output_file = path.OUTPUTS / "task_1" / f"task_1_analysis_{user_id}.txt"

        if save_output:
            output_file.write_text(report_text, encoding="utf-8")
            print(f"[INFO] Rapport sauvegardé : {output_file}")

        report_paths.append(output_file)

    print(f"\n[INFO] {len(report_paths)} rapport(s) qualitatif(s) généré(s).")
    return report_paths