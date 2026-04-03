#!/usr/bin/env python3
"""
Task 1 - Recommandation par similarité de contenu (Content-Based Filtering)

Pipeline :
1. Charger les artefacts :
   - user_profiles_matrix
   - user_to_idx
   - user_seen_items_train
   - item_ids / item_tfidf_matrix (TRAIN)
2. Calculer les scores candidats TRAIN pour les utilisateurs cibles (3.2.1)
3. Construire le catalogue TEST
4. Générer les recommandations Top-N sur TEST à partir des scores TRAIN (3.2.2)

Sorties :
- task_1_all_users_scores.csv
- task_1_top_{N}_test_items_from_train_scores.csv
"""

from pathlib import Path
import sys
import pickle

import numpy as np
import pandas as pd
from scipy.sparse import load_npz

# Pour import path depuis la racine
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import path

from task_1_score import compute_candidate_scores_for_all_users
from task_1_suggestion import recommend_test_items_from_train_scores
from task_1_qualitative_analysis import generate_qualitative_analysis_reports
from task_1_evaluation_items import task_1_evaluation_items


def task_1():
    print("=" * 80)
    print("TASK 1 - CONTENT-BASED FILTERING")
    print("=" * 80)

    # ------------------------------------------------------------------
    # Paramètres de test (important pour éviter des temps trop longs)
    # ------------------------------------------------------------------
    LIMIT_USERS = 1000     # Limite de 1000
    TOP_N_TRAIN = 100      # top-N scores TRAIN conservés par utilisateur (3.2.1)
    TOP_K_TRAIN = 20       # top-K TRAIN utilisés comme ancres pour TEST (3.2.2)
    TOP_N_TEST = 20        # top-N final de recommandations TEST

    RUN_QUALITATIVE_ANALYSIS = True

    output_dir = path.OUTPUTS / "task_1"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Vérifications fichiers requis
    # ------------------------------------------------------------------
    required_files = [
        path.USER_PROFILES,
        path.USER_TO_IDX,
        path.USER_HISTORIES,
        path.ITEMS,
        path.ITEM_TFIDF,
        path.JOINING_TEST,
    ]

    for file_path in required_files:
        if not file_path.exists():
            raise FileNotFoundError(f"Fichier requis introuvable : {file_path}")

    # ------------------------------------------------------------------
    # Chargement des artefacts TRAIN
    # ------------------------------------------------------------------
    print("\n[INFO] Chargement user_profiles_matrix...")
    user_profiles_matrix = load_npz(path.USER_PROFILES).tocsr()
    print(f"[INFO] user_profiles_matrix shape = {user_profiles_matrix.shape}")

    print("\n[INFO] Chargement user_to_idx...")
    with open(path.USER_TO_IDX, "rb") as f:
        user_to_idx = pickle.load(f)
    print(f"[INFO] user_to_idx : {len(user_to_idx):,} utilisateurs")

    print("\n[INFO] Chargement user_seen_items_train...")
    with open(path.USER_HISTORIES, "rb") as f:
        user_seen_items_train = pickle.load(f)
    print(f"[INFO] user_seen_items_train : {len(user_seen_items_train):,} utilisateurs")

    print("\n[INFO] Chargement item_ids TRAIN...")
    train_item_ids = np.load(path.ITEMS, allow_pickle=True).astype(str)
    print(f"[INFO] train_item_ids : {len(train_item_ids):,} items")

    print("\n[INFO] Chargement item_tfidf_matrix TRAIN...")
    train_item_tfidf_matrix = load_npz(path.ITEM_TFIDF).tocsr()
    print(f"[INFO] train_item_tfidf_matrix shape = {train_item_tfidf_matrix.shape}")

    # Mapping item -> idx (plus sûr que de dépendre d’un pickle externe ici)
    train_item_to_idx = {item_id: idx for idx, item_id in enumerate(train_item_ids)}
    print(f"[INFO] train_item_to_idx : {len(train_item_to_idx):,} items")

    # ------------------------------------------------------------------
    # Utilisateurs cibles (depuis TEST)
    # ------------------------------------------------------------------
    print("\n[INFO] Chargement du fichier TEST pour récupérer les utilisateurs cibles...")
    df_test = pd.read_parquet(path.JOINING_TEST)

    if "user_id" not in df_test.columns:
        raise KeyError("La colonne 'user_id' est absente de JOINING_TEST")

    df_test["user_id"] = df_test["user_id"].astype(str)

    test_user_ids = df_test["user_id"].dropna().astype(str).unique().tolist()
    print(f"[INFO] Utilisateurs uniques dans TEST : {len(test_user_ids):,}")

    # Limitation volontaire pour test/debug
    if LIMIT_USERS is not None and LIMIT_USERS > 0:
        test_user_ids = test_user_ids[:LIMIT_USERS]
        print(f"[INFO] Limitation activée : {len(test_user_ids)} utilisateurs conservés")

    # ------------------------------------------------------------------
    # 3.2.1 - Calcul des scores candidats sur le catalogue TRAIN
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("3.2.1 - CALCUL DES SCORES CANDIDATS SUR TRAIN")
    print("=" * 80)

    all_scores_df = compute_candidate_scores_for_all_users(
        user_ids=test_user_ids,
        user_profiles_matrix=user_profiles_matrix,
        user_to_idx=user_to_idx,
        user_seen_items_train=user_seen_items_train,
        item_ids=train_item_ids,
        item_tfidf_matrix=train_item_tfidf_matrix,
        item_to_idx=train_item_to_idx,
        top_n=TOP_N_TRAIN,
        save_output=True,
        output_filename="task_1_all_users_scores.csv"
    )

    print(f"\n[INFO] all_scores_df construit : {len(all_scores_df):,} lignes")

    if all_scores_df.empty:
        print("\n[WARNING] Aucun score TRAIN calculé. Arrêt du pipeline.")
        return

    # ------------------------------------------------------------------
    # Construction du catalogue TEST (items présents dans JOINING_TEST)
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("CONSTRUCTION DU CATALOGUE TEST")
    print("=" * 80)

    if "parent_asin" not in df_test.columns:
        raise KeyError("La colonne 'parent_asin' est absente de JOINING_TEST")

    df_test["parent_asin"] = df_test["parent_asin"].astype(str)

    # On garde uniquement les items TEST qui existent dans le TF-IDF global
    test_item_ids = df_test["parent_asin"].dropna().astype(str).unique().tolist()
    print(f"[INFO] Items uniques dans TEST (brut) : {len(test_item_ids):,}")

    valid_test_item_ids = [item_id for item_id in test_item_ids if item_id in train_item_to_idx]
    print(f"[INFO] Items TEST présents dans item_tfidf_matrix : {len(valid_test_item_ids):,}")

    if len(valid_test_item_ids) == 0:
        print("\n[WARNING] Aucun item TEST exploitable trouvé dans le TF-IDF. Arrêt.")
        return

    test_item_indices = [train_item_to_idx[item_id] for item_id in valid_test_item_ids]
    test_item_tfidf_matrix = train_item_tfidf_matrix[test_item_indices]

    print(f"[INFO] test_item_tfidf_matrix shape = {test_item_tfidf_matrix.shape}")

    # ------------------------------------------------------------------
    # 3.2.2 - Recommandation Top-N TEST à partir des scores TRAIN
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("3.2.2 - RECOMMANDATION TOP-N SUR TEST")
    print("=" * 80)

    recommendations_df = recommend_test_items_from_train_scores(
        test_user_ids=test_user_ids,
        train_item_ids=train_item_ids,
        train_item_tfidf_matrix=train_item_tfidf_matrix,
        test_item_ids=valid_test_item_ids,
        test_item_tfidf_matrix=test_item_tfidf_matrix,
        user_seen_items=user_seen_items_train,
        top_k_train=TOP_K_TRAIN,
        top_n=TOP_N_TEST,
        train_scores_file=path.OUTPUTS / "task_1" / "task_1_all_users_scores.csv",
        save_output=True
    )

    print("\n" + "=" * 80)
    print("RÉSUMÉ FINAL")
    print("=" * 80)
    print(f"[INFO] Utilisateurs traités         : {len(test_user_ids)}")
    print(f"[INFO] Scores TRAIN générés        : {len(all_scores_df):,}")
    print(f"[INFO] Recommandations TEST finales: {len(recommendations_df):,}")

    if not recommendations_df.empty:
        print("\n[INFO] Aperçu final des recommandations :")
        print(recommendations_df.head(20))

    # -------------------------------------------------------------------------
    # 3.2.3 - Analyse qualitative brève
    # -------------------------------------------------------------------------
    if RUN_QUALITATIVE_ANALYSIS:
        print("\n" + "=" * 80)
        print("3.2.3 - ANALYSE QUALITATIVE")
        print("=" * 80)

        print("\n[INFO] Chargement de TRAIN pour l'analyse qualitative...")
        train_df = pd.read_parquet(path.TRAIN)

        print("[INFO] Chargement des métadonnées pour l'analyse qualitative...")
        meta_df = pd.read_parquet(path.ITEM_METADATA_LIGHT)

        qualitative_reports = generate_qualitative_analysis_reports(
            train_df=train_df,
            recommendations_df=recommendations_df,
            metadata_df=meta_df,
            n_users=3,
            top_k_examples=5,
            save_output=True
        )

        print(f"[INFO] Rapports qualitatifs générés : {len(qualitative_reports)}")
    else:
        print("\n[INFO] Analyse qualitative désactivée (RUN_QUALITATIVE_ANALYSIS = False).")

    print("\n[INFO] Pipeline Task 1 terminé avec succès.")

    print("\n" + "#" * 80)
    print("# TASK 1 — ÉVALUATION COMPLÈTE DES RECOMMANDATIONS")
    print("#" * 80)

    print("\n" + "#" * 80)
    print("# ÉVALUATION QUANTITATIVE")
    print("#" * 80)

    task_1_evaluation_items(k=20, rating_col="rating", positive_threshold=4.0,)

if __name__ == "__main__":
    task_1()