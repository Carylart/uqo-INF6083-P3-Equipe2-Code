# 3.2.2 Implémentation
# Vous développerez un mécanisme de recommandation top-N permettant, pour chaque utilisateur de test, de produire une liste ordonnée de livres recommandés.
# L’implémentation devra être pensée de manière efficace, en tenant compte de la dimension potentiellement élevée des représentations.

import path
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def recommend_test_items_from_train_scores(
    test_user_ids,
    train_item_ids,
    train_item_tfidf_matrix,
    test_item_ids,
    test_item_tfidf_matrix,
    user_seen_items,
    top_k_train=10,
    top_n=10,
    train_scores_file=None,
    save_output=True
):
    """
    3.2.2 - Implémentation
    ----------------------
    Produit, pour chaque utilisateur de TEST, une liste ordonnée Top-N de livres du catalogue TEST en s'appuyant sur les scores déjà calculés sur le catalogue TRAIN (3.2.1).

    Principe
    --------
    1. Lit les scores TRAIN déjà calculés (task_1_all_users_scores.csv).
    2. Pour chaque utilisateur de TEST :
       - récupère ses Top-K livres recommandés dans TRAIN,
       - calcule la similarité contenu entre les livres TEST et ces Top-K TRAIN,
       - pondère cette similarité par le score TRAIN,
       - agrège avec un max pondéré,
       - exclut les livres déjà vus en TRAIN si présents dans TEST,
       - retourne les Top-N livres TEST.

    Paramètres
    ----------
    test_user_ids : iterable
        Liste des utilisateurs cibles (issus de TEST).

    train_item_ids : list
        Liste ordonnée des identifiants des livres du catalogue TRAIN.

    train_item_tfidf_matrix : sparse matrix
        Matrice TF-IDF des livres TRAIN.

    test_item_ids : list
        Liste ordonnée des identifiants des livres du catalogue TEST.

    test_item_tfidf_matrix : sparse matrix
        Matrice TF-IDF des livres TEST.

    user_seen_items : dict
        Mapping user_id -> set des livres vus en TRAIN.

    top_k_train : int
        Nombre de livres TRAIN les mieux scorés utilisés comme ancres de similarité.

    top_n : int
        Nombre final de recommandations TEST à produire par utilisateur.

    train_scores_file : Path ou str, optionnel
        Chemin du CSV des scores TRAIN.
        Si None, utilise path.REPORTS / "task_1_all_users_scores.csv"

    save_output : bool
        Si True, sauvegarde le CSV final dans outputs/reports.

    Retour
    ------
    pd.DataFrame
        Colonnes :
        - user_id
        - parent_asin
        - score
    """

    if train_scores_file is None:
        train_scores_file = path.OUTPUTS / "task_1" / "task_1_all_users_scores.csv"

    print(f"\n[INFO] Chargement des scores TRAIN depuis : {train_scores_file}")
    scores_df = pd.read_csv(train_scores_file)

    if scores_df.empty:
        print(f"\n[WARNING] Le fichier de scores TRAIN est vide")
        return pd.DataFrame(columns=["user_id","parent_asin", "score"])

    print(f"\n[INFO] Nombre total de lignes de scores TRAIN : {len(scores_df)}")

    # Sécurité : tri global
    scores_df = scores_df.sort_values(by=["user_id", "score"], ascending=[True, False]).reset_index(drop=True)

    # Mapping item TRAIN -> index
    train_item_to_idx = {item_id: idx for idx, item_id in enumerate(train_item_ids)}

    # Similarité TEST x TRAIN (une seule fois pour l'efficacité)
    print(f"\n[INFO] Calcul de la matrice de similarité TEST x TRAIN...")

    # shape = (n_test_items, n_train_items)
    test_train_sim = cosine_similarity(test_item_tfidf_matrix, train_item_tfidf_matrix)
    print(f"\n[INFO] Matrice de similarité calculée : shape={test_train_sim.shape}")

    all_recommendations = []

    print(f"\n[INFO] Début génération Top-{top_n} pour {len(test_user_ids)} utilisateurs de TEST")

    for i, user_id in enumerate(test_user_ids, start=1):
        print(f"\n[INFO] Utilisateur TEST {i}/{len(test_user_ids)} : {user_id}")

        # Scores TRAIN de l'utilisateur
        user_scores = scores_df[scores_df["user_id"] == user_id].copy()

        if user_scores.empty:
            print(f"\n[WARNING] Aucun score TRAIN trouvé pour {user_id} -> utilisateur ignoré")
            continue

        # Top-K TRAIN pour cet utilisateur
        user_topk_train = user_scores.head(top_k_train).copy()

        if user_topk_train.empty:
            print(f"\n[WARNING] Aucun Top-K TRAIN disponible pour {user_id}")
            continue

        print(f"\n[INFO] Top-{top_k_train} TRAIN récupéré pour {user_id}")

        # Garde uniquement les items TRAIN présents dans le mapping
        valid_train_rows = user_topk_train[user_topk_train["parent_asin"].isin(train_item_to_idx)].copy()

        if valid_train_rows.empty:
            print(f"\n[WARNING] Aucun item TRAIN valide dans le Top-{top_k_train} de {user_id}")
            continue

        topk_train_ids = valid_train_rows["parent_asin"].tolist()
        topk_train_scores = valid_train_rows["score"].to_numpy(dtype=float)
        topk_train_indices = [train_item_to_idx[item_id] for item_id in topk_train_ids]

        # Sous-matrice : similarité de tous les items TEST avec les Top-K TRAIN de l'utilisateur
        # shape = (n_test_items, top_k_effectif)
        sim_sub = test_train_sim[:, topk_train_indices]

        # Pondération par les scores TRAIN
        weighted_sim = sim_sub * topk_train_scores  # broadcasting

        # Agrégation : max pondéré
        final_test_scores = weighted_sim.max(axis=1)

        # Construit les résultats candidats TEST
        seen_train_items = user_seen_items.get(user_id, set())

        candidate_rows = []
        for test_idx, test_item_id in enumerate(test_item_ids):
            # Exclure si un item TEST est aussi un item déjà vu en TRAIN
            if test_item_id in seen_train_items:
                continue

            candidate_rows.append((user_id, test_item_id, float(final_test_scores[test_idx])))

        if not candidate_rows:
            print(f"\n[WARNING] Aucun livre TEST candidat pour {user_id}")
            continue

        user_test_scores_df = pd.DataFrame(candidate_rows, columns=["user_id", "parent_asin", "score"])

        # Tri décroissant + Top-N
        user_test_scores_df = user_test_scores_df.sort_values(by="score", ascending=False).reset_index(drop=True)

        # Ajout du rank AVANT le découpage Top-N
        user_test_scores_df["rank"] = user_test_scores_df.index + 1

        # Top-N final
        user_topn = user_test_scores_df.head(top_n).copy()

        print(f"\n[INFO] Top-{top_n} TEST généré pour {user_id}")
        print(user_topn.head(top_n))

        all_recommendations.append(user_topn)

    # Concat finale
    if all_recommendations:
        recommendations_df = pd.concat(all_recommendations, ignore_index=True)
    else:
        recommendations_df = pd.DataFrame(columns=["user_id", "parent_asin", "score"])

    print(f"\n[INFO] Génération des recommandations TEST terminée")
    print(f"\n[INFO] Nombre total de recommandations produites : {len(recommendations_df)}")
    print(f"\n[INFO] Aperçu global :\n")
    print(recommendations_df.head(20))

    # Sauvegarde
    if save_output:
        output_file = path.OUTPUTS / "task_1" / f"task_1_top_{top_n}_test_items_from_train_scores.csv"
        recommendations_df.to_csv(output_file, index=False)
        print(f"\n[INFO] Fichier sauvegardé : {output_file}")

    return recommendations_df