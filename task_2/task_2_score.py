"""
task_2_score.py - Calcul des scores User-Based Collaborative Filtering

Principe :
-----------
Pour chaque utilisateur cible u :
  1. Calculer la similarité cosinus entre u et tous les autres utilisateurs
     à partir de la matrice R_train (vecteurs de ratings/interactions).
  2. Sélectionner les K voisins les plus proches (plus forte similarité > 0).
  3. Pour chaque item non vu par u, agréger les scores des voisins :
       score(u, i) = Σ [ sim(u, v) * R_train[v, i] ] / Σ |sim(u, v)|
     où la somme porte sur les voisins v qui ont interagi avec l'item i.
  4. Retourner les Top-N items avec les meilleurs scores agrégés.

Justification du choix de la similarité cosinus :
  La matrice R_train est sparse et contient des valeurs binaires ou des ratings.
  La similarité cosinus est invariante à la norme des vecteurs et bien adaptée
  aux matrices sparse, ce qui en fait un choix naturel et performant ici.
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path


def compute_ubcf_scores_for_user(
    target_user_id: str,
    R_train: csr_matrix,
    user_ids_all: np.ndarray,
    item_ids_all: np.ndarray,
    user_to_idx: dict,
    user_seen_items_train: dict,
    k_neighbors: int = 50,
    top_n: int = 20,
    min_common_items: int = 2,
) -> pd.DataFrame:
    """
    Calcule les scores UBCF pour un utilisateur cible.

    Paramètres
    ----------
    target_user_id : str
        Identifiant de l'utilisateur cible.
    R_train : csr_matrix
        Matrice user-item d'entraînement (shape: n_users x n_items).
    user_ids_all : np.ndarray
        Tableau des identifiants de tous les utilisateurs (dans l'ordre des lignes de R_train).
    item_ids_all : np.ndarray
        Tableau des identifiants de tous les items (dans l'ordre des colonnes de R_train).
    user_to_idx : dict
        Mapping user_id -> index dans R_train.
    user_seen_items_train : dict
        Historique des items vus par chaque utilisateur en entraînement.
    k_neighbors : int
        Nombre de voisins à considérer.
    top_n : int
        Nombre de recommandations à retourner.
    min_common_items : int
        Nombre minimum d'items en commun pour qu'un voisin soit valide.

    Retour
    ------
    pd.DataFrame avec colonnes : user_id, parent_asin, score, rank
    """

    if target_user_id not in user_to_idx:
        print(f"[WARNING] Utilisateur {target_user_id} absent de R_train, ignoré.")
        return pd.DataFrame(columns=["user_id", "parent_asin", "score", "rank"])

    target_idx = user_to_idx[target_user_id]
    target_vector = R_train[target_idx]  # (1 x n_items)

    # Items déjà vus par l'utilisateur cible
    seen_items = user_seen_items_train.get(target_user_id, set())

    # ----------------------------------------------------------------
    # Similarité cosinus entre l'utilisateur cible et tous les autres
    # ----------------------------------------------------------------
    # On calcule uniquement la ligne correspondant à l'utilisateur cible
    similarities = cosine_similarity(target_vector, R_train).flatten()  # (n_users,)

    # Exclure l'utilisateur lui-même
    similarities[target_idx] = 0.0

    # Filtrer les voisins sans items en commun (produit scalaire = 0 => pas d'items communs)
    # Si sim > 0, il y a au moins un item en commun (pour matrices binaires/ratings positifs)
    # Pour min_common_items > 1 : vérification explicite via produit scalaire
    if min_common_items > 1:
        # Nombre d'items en commun = produit de la matrice binaire
        target_binary = (target_vector > 0).astype(float)
        all_binary = (R_train > 0).astype(float)
        common_counts = target_binary.dot(all_binary.T).toarray().flatten()
        similarities[common_counts < min_common_items] = 0.0

    # Sélection des K voisins les plus proches (similarité > 0)
    top_k_indices = np.argsort(similarities)[::-1][:k_neighbors]
    top_k_indices = [idx for idx in top_k_indices if similarities[idx] > 0]

    if len(top_k_indices) == 0:
        print(f"[WARNING] Aucun voisin valide trouvé pour {target_user_id}.")
        return pd.DataFrame(columns=["user_id", "parent_asin", "score", "rank"])

    # ----------------------------------------------------------------
    # Agrégation des scores des voisins
    # score(u, i) = Σ sim(u,v)*R[v,i] / Σ |sim(u,v)|
    # ----------------------------------------------------------------
    neighbor_sims = np.array([similarities[idx] for idx in top_k_indices])  # (k,)
    neighbor_matrix = R_train[top_k_indices]  # (k x n_items), sparse

    # Numérateur : somme pondérée
    numerator = neighbor_sims.dot(neighbor_matrix.toarray())  # (n_items,)

    # Dénominateur : somme des similarités (seulement pour les voisins ayant interagi avec l'item)
    neighbor_interaction_mask = (neighbor_matrix > 0).toarray().astype(float)  # (k x n_items)
    denominator = neighbor_sims.dot(neighbor_interaction_mask)  # (n_items,)

    # Score final (éviter division par zéro)
    with np.errstate(divide='ignore', invalid='ignore'):
        item_scores = np.where(denominator > 0, numerator / denominator, 0.0)

    # ----------------------------------------------------------------
    # Construction du DataFrame des candidats (items non vus)
    # ----------------------------------------------------------------
    candidate_rows = []
    for item_idx, item_id in enumerate(item_ids_all):
        if item_id not in seen_items and item_scores[item_idx] > 0:
            candidate_rows.append((target_user_id, item_id, item_scores[item_idx]))

    if len(candidate_rows) == 0:
        return pd.DataFrame(columns=["user_id", "parent_asin", "score", "rank"])

    scores_df = pd.DataFrame(candidate_rows, columns=["user_id", "parent_asin", "score"])
    scores_df = scores_df.sort_values("score", ascending=False).reset_index(drop=True)
    scores_df = scores_df.head(top_n).copy()
    scores_df["rank"] = scores_df.index + 1

    return scores_df


def compute_ubcf_scores_for_all_users(
    target_user_ids: list,
    R_train: csr_matrix,
    user_ids_all: np.ndarray,
    item_ids_all: np.ndarray,
    user_to_idx: dict,
    user_seen_items_train: dict,
    k_neighbors: int = 50,
    top_n: int = 20,
    min_common_items: int = 2,
    output_dir: Path = None,
) -> pd.DataFrame:
    """
    Calcule les scores UBCF pour tous les utilisateurs cibles et sauvegarde les résultats.

    Retour
    ------
    pd.DataFrame avec colonnes : user_id, parent_asin, score, rank
    """

    print(f"\n[INFO] Calcul UBCF pour {len(target_user_ids)} utilisateurs...")
    print(f"[INFO] Paramètres : K_neighbors={k_neighbors}, Top_N={top_n}, min_common={min_common_items}")

    all_scores = []

    for i, user_id in enumerate(target_user_ids, start=1):
        if i % 100 == 0 or i == 1:
            print(f"[INFO] Progression : {i}/{len(target_user_ids)} utilisateurs traités...")

        user_scores_df = compute_ubcf_scores_for_user(
            target_user_id=user_id,
            R_train=R_train,
            user_ids_all=user_ids_all,
            item_ids_all=item_ids_all,
            user_to_idx=user_to_idx,
            user_seen_items_train=user_seen_items_train,
            k_neighbors=k_neighbors,
            top_n=top_n,
            min_common_items=min_common_items,
        )

        if not user_scores_df.empty:
            all_scores.append(user_scores_df)

    if len(all_scores) == 0:
        print("[WARNING] Aucun score calculé pour les utilisateurs fournis.")
        return pd.DataFrame(columns=["user_id", "parent_asin", "score", "rank"])

    all_scores_df = pd.concat(all_scores, ignore_index=True)

    # Sauvegarde
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Fichier complet des scores
        scores_path = output_dir / "task_2_all_users_scores.csv"
        all_scores_df.to_csv(scores_path, index=False)
        print(f"[INFO] Scores sauvegardés : {scores_path}")

        # Fichier Top-N (même convention que Task 1)
        topn_path = output_dir / f"task_2_top_{top_n}_recommendations.csv"
        all_scores_df.to_csv(topn_path, index=False)
        print(f"[INFO] Recommandations Top-{top_n} sauvegardées : {topn_path}")

    print(f"[INFO] Total lignes générées : {len(all_scores_df):,}")
    print(f"[INFO] Utilisateurs avec au moins 1 recommandation : {all_scores_df['user_id'].nunique():,}")

    return all_scores_df