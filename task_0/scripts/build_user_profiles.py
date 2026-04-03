#!/usr/bin/env python3
"""
Construction des profils utilisateurs content-based à partir du train.

Ce script :
1. charge les interactions d'entraînement,
2. charge la matrice TF-IDF des livres,
3. construit l'historique des items vus par utilisateur,
4. construit un profil utilisateur = moyenne des vecteurs TF-IDF des livres vus,
5. sauvegarde :
   - user_seen_items_train.pkl
   - user_to_idx.pkl
   - user_profiles_matrix.npz

Sorties :
    path.USER_HISTORIES      -> dict[user_id -> set(parent_asin)]
    path.USER_TO_IDX         -> dict[user_id -> row_idx]
    path.USER_PROFILES       -> sparse matrix (n_users, n_features)
"""

from collections import defaultdict
from pathlib import Path
import sys
import pickle

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, load_npz, save_npz, vstack

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import path

MAX_USERS = 5000


def build_user_profile():
    # Chemins
    train_path = path.JOINING_TRAIN
    user_ids_path = path.USERS
    item_tfidf_path = path.ITEM_TFIDF
    item_to_idx_path = path.ITEM_TO_IDX

    user_seen_out = path.USER_HISTORIES
    user_to_idx_out = path.USER_TO_IDX
    user_profiles_out = path.USER_PROFILES

    # Créer le dossier de sortie
    path.SPLITS.mkdir(parents=True, exist_ok=True)

    # Vérifications
    if not train_path.exists():
        raise FileNotFoundError(f"Le fichier train n'existe pas : {train_path}")

    if not user_ids_path.exists():
        raise FileNotFoundError(f"Le fichier user_ids n'existe pas : {user_ids_path}")

    if not item_tfidf_path.exists():
        raise FileNotFoundError(
            f"Le fichier item_tfidf_matrix n'existe pas : {item_tfidf_path}\n"
            "➡️ Exécute d'abord build_item_tfidf.py"
        )

    if not item_to_idx_path.exists():
        raise FileNotFoundError(
            f"Le fichier item_to_idx n'existe pas : {item_to_idx_path}\n"
            "➡️ Exécute d'abord build_item_tfidf.py"
        )

    # Chargement des données
    print(f"Chargement train depuis {train_path}...")
    df_train = pd.read_parquet(train_path, columns=["user_id", "parent_asin"])
    print(f"Train chargé : {df_train.shape[0]:,} lignes, {df_train.shape[1]} colonnes.")

    required_cols = {"user_id", "parent_asin"}
    missing_cols = required_cols - set(df_train.columns)
    if missing_cols:
        raise KeyError(f"Colonnes manquantes dans train : {missing_cols}")

    print(f"Chargement user_ids depuis {user_ids_path}...")
    user_ids = np.load(user_ids_path, allow_pickle=True).astype(str)
    print(f"{len(user_ids):,} utilisateurs chargés.")

    # Limitation utilisateurs
    if MAX_USERS is not None and len(user_ids) > MAX_USERS:
        user_ids = user_ids[:MAX_USERS]
        print(f"⚠️ Limitation activée : {len(user_ids):,} utilisateurs conservés (MAX_USERS={MAX_USERS:,}).")

    print(f"Chargement item_tfidf_matrix depuis {item_tfidf_path}...")
    item_tfidf_matrix = load_npz(item_tfidf_path).tocsr()
    print(f"item_tfidf_matrix shape = {item_tfidf_matrix.shape}")

    print(f"Chargement item_to_idx depuis {item_to_idx_path}...")
    with open(item_to_idx_path, "rb") as f:
        item_to_idx = pickle.load(f)
    print(f"{len(item_to_idx):,} items dans item_to_idx.")

    # Uniformiser types
    df_train["user_id"] = df_train["user_id"].astype(str)
    df_train["parent_asin"] = df_train["parent_asin"].astype(str)

    # Filtrer train aux users retenus
    user_ids_set = set(user_ids)
    before_train = len(df_train)
    df_train = df_train[df_train["user_id"].isin(user_ids_set)].copy()
    print(f"Train filtré aux utilisateurs retenus : {len(df_train):,} lignes conservées / {before_train:,}")

    # Construction user_seen_items_train
    print("Construction de user_seen_items_train...")
    user_seen_items_train: dict[str, set[str]] = defaultdict(set)

    for user_id, item_id in zip(df_train["user_id"], df_train["parent_asin"]):
        user_seen_items_train[user_id].add(item_id)

    user_seen_items_train = dict(user_seen_items_train)

    print(f"user_seen_items_train construit pour {len(user_seen_items_train):,} utilisateurs.")

    # Construction user_to_idx
    print("Construction de user_to_idx...")
    user_to_idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
    print(f"user_to_idx contient {len(user_to_idx):,} utilisateurs.")

    # Vérifier cohérence
    users_in_train = set(user_seen_items_train.keys())
    users_in_user_ids = set(user_ids)

    missing_in_user_ids = users_in_train - users_in_user_ids
    if missing_in_user_ids:
        print(f"⚠️ {len(missing_in_user_ids):,} utilisateurs du train absents de user_ids.npy")
        print("Ils seront ignorés dans user_profiles_matrix.")

    users_without_history = users_in_user_ids - users_in_train
    if users_without_history:
        print(f"ℹ️ {len(users_without_history):,} utilisateurs sans historique train.")
        print("Ils recevront un profil vide.")

    # Construction user_profiles_matrix
    print("Construction de user_profiles_matrix...")
    n_users = len(user_ids)
    n_features = item_tfidf_matrix.shape[1]

    user_profile_rows = []
    skipped_unknown_items_total = 0

    for idx, user_id in enumerate(user_ids):
        seen_items = user_seen_items_train.get(user_id, set())

        item_indices = []
        skipped_unknown_items = 0

        for item_id in seen_items:
            item_idx = item_to_idx.get(item_id)
            if item_idx is None:
                skipped_unknown_items += 1
            else:
                item_indices.append(item_idx)

        skipped_unknown_items_total += skipped_unknown_items

        if not item_indices:
            user_profile = csr_matrix((1, n_features), dtype=np.float32)
        else:
            item_vectors = item_tfidf_matrix[item_indices]
            user_profile_dense = item_vectors.mean(axis=0)
            user_profile = csr_matrix(np.asarray(user_profile_dense, dtype=np.float32))

        user_profile_rows.append(user_profile)

        if (idx + 1) % 1000 == 0 or (idx + 1) == n_users:
            print(f"  -> {idx + 1:,} / {n_users:,} profils construits")

    user_profiles_matrix = vstack(user_profile_rows).tocsr()

    print(f"user_profiles_matrix shape = {user_profiles_matrix.shape}")
    print(f"nnz = {user_profiles_matrix.nnz:,}")

    if skipped_unknown_items_total > 0:
        print(f"⚠️ {skipped_unknown_items_total:,} interactions ignorées (items absents de item_to_idx).")

    # Sauvegarde
    print(f"Sauvegarde de user_seen_items_train dans {user_seen_out}...")
    with open(user_seen_out, "wb") as f:
        pickle.dump(user_seen_items_train, f)

    print(f"Sauvegarde de user_to_idx dans {user_to_idx_out}...")
    with open(user_to_idx_out, "wb") as f:
        pickle.dump(user_to_idx, f)

    print(f"Sauvegarde de user_profiles_matrix dans {user_profiles_out}...")
    save_npz(user_profiles_out, user_profiles_matrix)

    print("Terminé !")
    print(f"- user_seen_items_train : {user_seen_out}")
    print(f"- user_to_idx          : {user_to_idx_out}")
    print(f"- user_profiles_matrix : {user_profiles_out}")


if __name__ == "__main__":
    build_user_profile()