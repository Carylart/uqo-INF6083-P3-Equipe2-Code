# 3.2.1 Principe
# Pour chaque utilisateur cible, vous calculerez un score de pertinence entre son profil et chaque livre non encore vu dans l’ensemble d’entraînement.
# Une mesure de similarité adaptée devra être choisie et justifiée.
# La similarité cosinus est une option naturelle dans le cas de vecteurs TF-IDF, mais d’autres variantes peuvent être considérées si elles sont motivées.
import path
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def compute_candidate_scores_for_user(
    user_id,
    user_profiles_matrix,
    user_to_idx,
    user_seen_items_train,
    item_ids,
    item_tfidf_matrix,
    item_to_idx,
    top_n=-1
):
    """
    3.2.1 - Principe
    ----------------
    Calcule un score de pertinence entre le profil d'un utilisateur et chaque livre non encore vu dans l'ensemble d'entraînement.
    """

    print(f"\n[INFO] Début calcul des scores pour l'utilisateur : {user_id}")

    # Si l'utilisateur n'a pas de profil, on retourne vide
    if user_id not in user_to_idx:
        print(f"[WARNING] Aucun profil trouvé pour l'utilisateur {user_id}")
        return pd.DataFrame(columns=["user_id", "parent_asin", "score"])

    # Récupère le profil utilisateur
    user_idx = user_to_idx[user_id]
    user_profile = user_profiles_matrix[user_idx]
    print(f"[INFO] Profil récupéré pour {user_id} (index={user_idx})")

    # Calcule la similarité cosinus du profil avec tous les items
    print("[INFO] Calcul de la similarité cosinus avec tous les livres...")
    scores = cosine_similarity(user_profile, item_tfidf_matrix).flatten()

    # Exclut les livres déjà vus
    seen_items = user_seen_items_train.get(user_id, set())
    print(f"[INFO] Nombre de livres déjà vus en entraînement : {len(seen_items)}")

    candidate_rows = []
    for item_id in item_ids:
        if item_id not in seen_items:
            item_idx = item_to_idx.get(item_id)
            if item_idx is not None:
                candidate_rows.append((user_id, item_id, scores[item_idx]))

    print(f"[INFO] Nombre de livres candidats non vus : {len(candidate_rows)}")

    # Construit le DataFrame des scores candidats
    scores_df = pd.DataFrame(candidate_rows, columns=["user_id", "parent_asin", "score"])

    # Trie par score décroissant
    scores_df = scores_df.sort_values(by="score", ascending=False).reset_index(drop=True)

    # Ajout du rank
    scores_df["rank"] = scores_df.index + 1

    print("[INFO] Tri des scores terminé")
    print("[INFO] Top 10 recommandations candidates :")
    print(scores_df.head(10))

    # Si top_n > 0 on ne retourne que les N premiers scores
    if top_n > 0:
        return scores_df.head(top_n).copy()

    return scores_df


def compute_candidate_scores_for_all_users(
    user_ids,
    user_profiles_matrix,
    user_to_idx,
    user_seen_items_train,
    item_ids,
    item_tfidf_matrix,
    item_to_idx,
    top_n=-1,
    save_output=True,
    output_filename="task_1_all_users_scores.csv"
):
    """
    Calcule les scores de pertinence pour tous les utilisateurs cibles et sauvegarde un seul fichier global.
    """

    print("\n[INFO] Début calcul global des scores pour tous les utilisateurs...")

    all_scores = []

    # Pour chaque utilisateur, on calcule ses scores
    for i, user_id in enumerate(user_ids, start=1):
        print(f"\n[INFO] ===== Utilisateur {i}/{len(user_ids)} : {user_id} =====")

        user_scores_df = compute_candidate_scores_for_user(
            user_id=user_id,
            user_profiles_matrix=user_profiles_matrix,
            user_to_idx=user_to_idx,
            user_seen_items_train=user_seen_items_train,
            item_ids=item_ids,
            item_tfidf_matrix=item_tfidf_matrix,
            item_to_idx=item_to_idx,
            top_n=top_n
        )

        if not user_scores_df.empty:
            all_scores.append(user_scores_df)

    # Si aucun score n'a été calculé
    if len(all_scores) == 0:
        print("[WARNING] Aucun score calculé pour les utilisateurs fournis.")
        return pd.DataFrame(columns=["user_id", "parent_asin", "score"])

    # Concat tous les résultats
    all_scores_df = pd.concat(all_scores, ignore_index=True)

    # Sauvegarde
    if save_output:
        output_file = path.OUTPUTS / "task_1" / output_filename
        output_file.parent.mkdir(parents=True, exist_ok=True)
        all_scores_df.to_csv(output_file, index=False)
        print(f"[INFO] Fichier global sauvegardé : {output_file}")

    print(f"[INFO] Nombre total de lignes sauvegardées : {len(all_scores_df)}")
    print("[INFO] Aperçu global :")
    print(all_scores_df.head(20))

    return all_scores_df