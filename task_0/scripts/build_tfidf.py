#!/usr/bin/env python3
"""
Construction de la représentation TF-IDF des livres à partir des métadonnées.

Version optimisée mémoire :
- limite optionnelle du nombre d'items
- charge uniquement les colonnes utiles
- filtre les métadonnées sur les parent_asin présents dans item_ids
- aligne exactement les metadata sur l'ordre de item_ids
- sauvegarde :
    * item_ids.npy (réaligné sur le TF-IDF)
    * item_tfidf_matrix.npz
    * item_to_idx.pkl
    * tfidf_vectorizer.pkl
    * item_metadata_light.parquet
"""

from pathlib import Path
import sys
import pickle

import numpy as np
import pandas as pd
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
import pyarrow.dataset as ds

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import path

MAX_ITEMS = None  # mettre None pour tout traiter

META_COLS = [
    "parent_asin",
    "title",
    "subtitle",
    "features",
    "description",
    "categories",
    "author",
]


def safe_join(value) -> str:
    """
    Convertit proprement une valeur metadata en texte exploitable.
    """
    if value is None:
        return ""

    if isinstance(value, float) and pd.isna(value):
        return ""

    if isinstance(value, (list, tuple)):
        return " ".join(str(x) for x in value if x is not None)

    return str(value)


def build_item_text(row: pd.Series) -> str:
    """
    Construit le texte d'un livre à partir de ses métadonnées.
    """
    parts = [
        safe_join(row.get("title", "")),
        safe_join(row.get("subtitle", "")),
        safe_join(row.get("features", "")),
        safe_join(row.get("description", "")),
        safe_join(row.get("categories", "")),
        safe_join(row.get("author", "")),
    ]

    return " ".join(part for part in parts if part).strip().lower()


def load_filtered_metadata(metadata_path: Path, item_ids: np.ndarray) -> pd.DataFrame:
    """
    Charge uniquement les colonnes utiles et filtre les lignes
    dont parent_asin appartient à item_ids.
    """
    print(f"Chargement filtré des metadata depuis {metadata_path}...")

    dataset = ds.dataset(str(metadata_path), format="parquet")

    # Attention : un filtre IN sur trop de valeurs peut être lourd.
    filter_expr = ds.field("parent_asin").isin(item_ids.tolist())

    table = dataset.to_table(columns=META_COLS, filter=filter_expr)
    df_meta = table.to_pandas()

    print(f"Metadata filtrées chargées : {df_meta.shape[0]:,} lignes, {df_meta.shape[1]} colonnes.")
    return df_meta


def build_tfidf():
    # =========================
    # Chemins
    # =========================
    item_ids_path = path.ITEMS
    metadata_path = path.RAW_METABOOK

    item_ids_out = path.ITEMS
    item_tfidf_out = path.ITEM_TFIDF
    item_to_idx_out = path.ITEM_TO_IDX
    tfidf_vectorizer_out = path.TFIDF_VECTORIZER
    item_metadata_light_out = path.ITEM_METADATA_LIGHT

    # Créer le dossier de sortie
    path.SPLITS.mkdir(parents=True, exist_ok=True)

    # =========================
    # Vérifications
    # =========================
    if not item_ids_path.exists():
        raise FileNotFoundError(f"Le fichier item_ids n'existe pas : {item_ids_path}")

    if not metadata_path.exists():
        raise FileNotFoundError(f"Le fichier metadata n'existe pas : {metadata_path}")

    # =========================
    # Chargement item_ids
    # =========================
    print(f"Chargement item_ids depuis {item_ids_path}...")
    item_ids = np.load(item_ids_path, allow_pickle=True).astype(str)
    print(f"{len(item_ids):,} items chargés.")

    # Limitation optionnelle
    if MAX_ITEMS is not None and len(item_ids) > MAX_ITEMS:
        item_ids = item_ids[:MAX_ITEMS]
        print(f"⚠️ Limitation activée : {len(item_ids):,} items conservés (MAX_ITEMS={MAX_ITEMS:,}).")

    # =========================
    # Chargement metadata filtré
    # =========================
    df_meta = load_filtered_metadata(metadata_path, item_ids)

    if "parent_asin" not in df_meta.columns:
        raise KeyError("La colonne 'parent_asin' est absente du fichier metadata.")

    # Uniformiser les IDs
    df_meta["parent_asin"] = df_meta["parent_asin"].astype(str)

    print(f"Items trouvés dans metadata : {df_meta['parent_asin'].nunique():,} / {len(item_ids):,}")

    # =========================
    # Suppression des doublons
    # =========================
    before_dedup = len(df_meta)
    df_meta = df_meta.drop_duplicates(subset="parent_asin", keep="first").copy()
    after_dedup = len(df_meta)

    if before_dedup != after_dedup:
        print(f"Doublons supprimés sur parent_asin : {before_dedup - after_dedup:,}")

    # =========================
    # Réindexation selon l'ordre exact de item_ids
    # =========================
    print("Réindexation selon l'ordre exact de item_ids.npy...")
    df_meta = df_meta.set_index("parent_asin")

    missing_items = [item_id for item_id in item_ids if item_id not in df_meta.index]
    if missing_items:
        print(f"⚠️ {len(missing_items):,} items absents des metadata.")
        print("Ils recevront un texte vide (vecteur TF-IDF nul).")

    # Alignement exact sur item_ids
    df_aligned = df_meta.reindex(item_ids).copy()
    df_aligned["parent_asin"] = item_ids

    # =========================
    # Sauvegarde metadata légère alignée
    # =========================
    print(f"Sauvegarde de item_metadata_light dans {item_metadata_light_out}...")
    df_aligned.reset_index(drop=True).to_parquet(item_metadata_light_out, index=False)

    # =========================
    # Construction des textes
    # =========================
    print("Construction des textes par livre...")
    item_texts = df_aligned.apply(build_item_text, axis=1)

    empty_texts = (item_texts.str.len() == 0).sum()
    if empty_texts:
        print(f"⚠️ {empty_texts:,} items ont un texte vide.")

    # =========================
    # Vectorisation TF-IDF
    # =========================
    print("Vectorisation TF-IDF...")
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=20_000,   # réduit pour économiser RAM
        ngram_range=(1, 1),    # unigrammes uniquement = plus léger
        min_df=2,
        max_df=0.8,
        strip_accents="unicode",
        dtype=np.float32,
    )

    item_tfidf_matrix = vectorizer.fit_transform(item_texts)

    print(f"item_tfidf_matrix shape = {item_tfidf_matrix.shape}")
    print(f"nnz = {item_tfidf_matrix.nnz:,}")

    # =========================
    # Construction de item_to_idx
    # =========================
    print("Construction de item_to_idx...")
    item_to_idx = {item_id: idx for idx, item_id in enumerate(item_ids)}

    # =========================
    # Sauvegardes finales
    # =========================
    print(f"Sauvegarde de item_ids (réaligné TF-IDF) dans {item_ids_out}...")
    np.save(item_ids_out, item_ids)

    print(f"Sauvegarde de item_tfidf_matrix dans {item_tfidf_out}...")
    save_npz(item_tfidf_out, item_tfidf_matrix)

    print(f"Sauvegarde de tfidf_vectorizer dans {tfidf_vectorizer_out}...")
    with open(tfidf_vectorizer_out, "wb") as f:
        pickle.dump(vectorizer, f)

    print(f"Sauvegarde de item_to_idx dans {item_to_idx_out}...")
    with open(item_to_idx_out, "wb") as f:
        pickle.dump(item_to_idx, f)

    # =========================
    # Résumé
    # =========================
    print("\nTerminé !")
    print(f"- item_ids            : {item_ids_out}")
    print(f"- item_tfidf_matrix   : {item_tfidf_out}")
    print(f"- tfidf_vectorizer    : {tfidf_vectorizer_out}")
    print(f"- item_to_idx         : {item_to_idx_out}")
    print(f"- item_metadata_light : {item_metadata_light_out}")


if __name__ == "__main__":
    build_tfidf()