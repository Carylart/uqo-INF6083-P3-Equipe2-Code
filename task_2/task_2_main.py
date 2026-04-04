#!/usr/bin/env python3
"""
Task 2 - Filtrage collaboratif basé sur les utilisateurs (User-Based CF)

Pipeline :
1. Charger les artefacts :
   - R_train (matrice user-item sparse)
   - user_ids / item_ids
   - user_seen_items_train
   - user_to_idx / item_to_idx
2. Calculer la matrice de similarité utilisateur-utilisateur (cosinus)
3. Pour chaque utilisateur cible, identifier ses K voisins les plus proches
4. Agréger les scores des items non vus à partir des voisins
5. Générer les recommandations Top-N
6. Évaluer avec les mêmes métriques que Task 1

Sorties :
- task_2_user_similarities.npz   (similarités inter-utilisateurs, optionnel)
- task_2_all_users_scores.csv
- task_2_top_{N}_recommendations.csv
- task_2_evaluation_global_metrics_top{K}.csv
- task_2_evaluation_per_user_metrics_top{K}.csv
"""

from pathlib import Path
import sys
import pickle

import numpy as np
import pandas as pd
from scipy.sparse import load_npz

# Import depuis la racine du projet
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import path

from task_2_score import compute_ubcf_scores_for_all_users
from task_2_evaluation import task_2_evaluation


def task_2():
    print("=" * 60)
    print("TASK 2 - USER-BASED COLLABORATIVE FILTERING")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Paramètres qui va guider cette partie
    # ------------------------------------------------------------------
    LIMIT_USERS  = 1000   # Même limite que Task 1 -- comparaison équitable (const global intéressante)
    K_NEIGHBORS  = 50     # Nombre de voisins les plus proches à prendre en compte
    TOP_N        = 20     # Top-N recommandations finales
    MIN_COMMON   = 2      # Nombre minimum d'items en commun -- considérer deux utilisateurs similaires (+ elevé + similaires)

    RUN_EVALUATION = True

    output_dir = path.OUTPUTS / "task_2"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Vérifications fichiers requis
    # ------------------------------------------------------------------
    required_files = [
        path.R_TRAIN,
        path.ITEMS,
        path.USERS,
        path.USER_TO_IDX,
        path.ITEM_TO_IDX,
        path.USER_HISTORIES,
        path.JOINING_TEST,
    ]

    for file_path in required_files:
        if not file_path.exists():
            raise FileNotFoundError(f"Fichier requis introuvable : {file_path}")

    # ------------------------------------------------------------------
    # Chargement des artefacts TRAIN
    # ------------------------------------------------------------------
    print("\n[INFO] Chargement R_train (matrice user-item sparse)...")
    R_train = load_npz(path.R_TRAIN).tocsr()
    print(f"[INFO] R_train shape = {R_train.shape}  (users x items)")

    print("\n[INFO] Chargement user_ids...")
    user_ids_all = np.load(path.USERS, allow_pickle=True).astype(str)
    print(f"[INFO] user_ids_all : {len(user_ids_all):,} utilisateurs")

    print("\n[INFO] Chargement item_ids...")
    item_ids_all = np.load(path.ITEMS, allow_pickle=True).astype(str)
    print(f"[INFO] item_ids_all : {len(item_ids_all):,} items")

    print("\n[INFO] Chargement user_to_idx...")
    with open(path.USER_TO_IDX, "rb") as f:
        user_to_idx = pickle.load(f)
    print(f"[INFO] user_to_idx : {len(user_to_idx):,} utilisateurs")

    print("\n[INFO] Chargement item_to_idx...")
    with open(path.ITEM_TO_IDX, "rb") as f:
        item_to_idx = pickle.load(f)
    print(f"[INFO] item_to_idx : {len(item_to_idx):,} items")

    print("\n[INFO] Chargement user_seen_items_train...")
    with open(path.USER_HISTORIES, "rb") as f:
        user_seen_items_train = pickle.load(f)
    print(f"[INFO] user_seen_items_train : {len(user_seen_items_train):,} utilisateurs")

    # ------------------------------------------------------------------
    # Utilisateurs cibles (depuis TEST)
    # ------------------------------------------------------------------
    print("\n[INFO] Chargement du fichier TEST pour récupérer les utilisateurs cibles...")
    df_test = pd.read_parquet(path.JOINING_TEST)
    df_test["user_id"] = df_test["user_id"].astype(str)

    test_user_ids = df_test["user_id"].dropna().unique().tolist()
    print(f"[INFO] Utilisateurs uniques dans TEST : {len(test_user_ids):,}")

    # Filtrer uniquement ceux présents dans R_train
    test_user_ids = [u for u in test_user_ids if u in user_to_idx]
    print(f"[INFO] Utilisateurs présents dans R_train : {len(test_user_ids):,}")

    if LIMIT_USERS is not None and LIMIT_USERS > 0:
        test_user_ids = test_user_ids[:LIMIT_USERS]
        print(f"[INFO] Limitation activée : {len(test_user_ids)} utilisateurs conservés")

    # ------------------------------------------------------------------
    # Calcul des scores UBCF pour tous les utilisateurs cibles
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("2.1 - CALCUL DES SCORES USER-BASED CF")
    print("=" * 60)

    all_scores_df = compute_ubcf_scores_for_all_users(
        target_user_ids=test_user_ids,
        R_train=R_train,
        user_ids_all=user_ids_all,
        item_ids_all=item_ids_all,
        user_to_idx=user_to_idx,
        user_seen_items_train=user_seen_items_train,
        k_neighbors=K_NEIGHBORS,
        top_n=TOP_N,
        min_common_items=MIN_COMMON,
        output_dir=output_dir,
    )

    print(f"\n[INFO] all_scores_df : {len(all_scores_df):,} lignes")

    if all_scores_df.empty:
        print("\n[WARNING] Aucun score calculé. Arrêt du pipeline.")
        return

    print("\n[INFO] Aperçu des recommandations finales :")
    print(all_scores_df.head(20))

    # ------------------------------------------------------------------
    # Évaluation
    # ------------------------------------------------------------------
    if RUN_EVALUATION:
        print("\n" + "#" * 80)
        print("# TASK 2 — ÉVALUATION QUANTITATIVE")
        print("#" * 80)

        task_2_evaluation(k=TOP_N, rating_col="rating", positive_threshold=4.0)

    print("\n[INFO] Pipeline Task 2 terminé avec succès.")


if __name__ == "__main__":
    task_2()