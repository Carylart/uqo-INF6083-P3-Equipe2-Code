"""
task_2_evaluation.py - Évaluation quantitative du filtrage collaboratif (Task 2)

Réutilise les mêmes métriques que Task 1 (Precision@K, Recall@K, F1@K,
HitRate@K, MAP@K, NDCG@K, RMSE, MAE) pour permettre une comparaison directe
entre le filtrage basé sur le contenu (Task 1) et le filtrage collaboratif (Task 2).

Convention d'entrée (identique à Task 1) :
  - recommendations_df : colonnes [user_id, parent_asin, score, rank]
  - test_df            : colonnes [user_id, parent_asin, rating]
"""

import math
import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import path

# Réutilisation directe des fonctions de métriques de Task 1
sys.path.insert(0, str(_ROOT / "task_1"))
from task_1_metric_functions import (
    precision_at_k,
    recall_at_k,
    f1_at_k,
    hit_rate_at_k,
    average_precision_at_k,
    ndcg_at_k,
)


def evaluate_ubcf_recommendations(
    recommendations_df: pd.DataFrame,
    test_df: pd.DataFrame,
    k: int = 20,
    rating_col: str = "rating",
    positive_threshold: float = 4.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Évalue les recommandations UBCF avec les métriques Top-K standard.

    Paramètres
    ----------
    recommendations_df : pd.DataFrame
        Colonnes attendues : user_id, parent_asin, score
    test_df : pd.DataFrame
        Colonnes attendues : user_id, parent_asin, rating_col
    k : int
        Seuil Top-K pour l'évaluation.
    rating_col : str
        Nom de la colonne des notes réelles dans test_df.
    positive_threshold : float
        Seuil à partir duquel un item est considéré pertinent (rating >= threshold).

    Retour
    ------
    (metrics_df, per_user_df)
    """

    print("\n" + "=" * 80)
    print("ÉVALUATION TASK 2 - USER-BASED COLLABORATIVE FILTERING")
    print("=" * 80)

    output_dir = path.OUTPUTS / "task_2"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_prefix = "task_2_evaluation"

    # Vérifications colonnes
    required_rec_cols  = {"user_id", "parent_asin", "score"}
    required_test_cols = {"user_id", "parent_asin", rating_col}

    missing_rec  = required_rec_cols  - set(recommendations_df.columns)
    missing_test = required_test_cols - set(test_df.columns)
    if missing_rec:
        raise ValueError(f"Colonnes manquantes dans recommendations_df : {missing_rec}")
    if missing_test:
        raise ValueError(f"Colonnes manquantes dans test_df : {missing_test}")

    rec_df    = recommendations_df.copy()
    test_copy = test_df.copy()

    # Ground truth : interactions positives dans TEST
    positive_test_df = test_copy[test_copy[rating_col] >= positive_threshold].copy()

    pos_path = output_dir / f"{save_prefix}_positive_test_interactions.csv"
    positive_test_df.to_csv(pos_path, index=False)
    print(f"[INFO] Interactions positives TEST sauvegardées : {pos_path}")

    ground_truth = (
        positive_test_df.groupby("user_id")["parent_asin"]
        .apply(set)
        .to_dict()
    )

    evaluable_users = set(ground_truth.keys())
    print(f"[INFO] Utilisateurs évaluables (>= 1 item positif dans TEST) : {len(evaluable_users):,}")

    # Filtrer et trier les recommandations
    rec_df = rec_df[rec_df["user_id"].isin(evaluable_users)].copy()
    rec_df = rec_df.sort_values(["user_id", "score"], ascending=[True, False])
    rec_df = rec_df.drop_duplicates(subset=["user_id", "parent_asin"], keep="first")
    rec_df["rank"] = rec_df.groupby("user_id").cumcount() + 1

    topk_rec_df = rec_df[rec_df["rank"] <= k].copy()

    topk_path = output_dir / f"{save_prefix}_top{k}_recommendations.csv"
    topk_rec_df.to_csv(topk_path, index=False)
    print(f"[INFO] Recommandations Top-{k} sauvegardées : {topk_path}")

    # Métriques par utilisateur
    per_user_metrics = []

    for user_id in sorted(evaluable_users):
        relevant_items = ground_truth[user_id]
        user_recs = topk_rec_df.loc[
            topk_rec_df["user_id"] == user_id, "parent_asin"
        ].tolist()

        p_at_k  = precision_at_k(user_recs, relevant_items)
        r_at_k  = recall_at_k(user_recs, relevant_items)
        f1_k    = f1_at_k(p_at_k, r_at_k)
        hr_k    = hit_rate_at_k(user_recs, relevant_items)
        ap_k    = average_precision_at_k(user_recs, relevant_items, k=k)
        ndcg_k  = ndcg_at_k(user_recs, relevant_items)

        per_user_metrics.append({
            "user_id":                    user_id,
            "num_relevant_items_test":    len(relevant_items),
            "num_recommended_items":      len(user_recs),
            f"precision@{k}":             p_at_k,
            f"recall@{k}":                r_at_k,
            f"f1@{k}":                    f1_k,
            f"hit_rate@{k}":              hr_k,
            f"ap@{k}":                    ap_k,
            f"ndcg@{k}":                  ndcg_k,
        })

    per_user_df = pd.DataFrame(per_user_metrics)

    per_user_path = output_dir / f"{save_prefix}_per_user_metrics_top{k}.csv"
    per_user_df.to_csv(per_user_path, index=False)
    print(f"[INFO] Métriques par utilisateur sauvegardées : {per_user_path}")

    # RMSE / MAE
    prediction_eval_df = pd.merge(
        rec_df[["user_id", "parent_asin", "score"]],
        test_copy[["user_id", "parent_asin", rating_col]],
        on=["user_id", "parent_asin"],
        how="inner",
    )

    pred_path = output_dir / f"{save_prefix}_prediction_pairs.csv"
    prediction_eval_df.to_csv(pred_path, index=False)
    print(f"[INFO] Paires RMSE/MAE sauvegardées : {pred_path}")

    if prediction_eval_df.empty:
        rmse_value = None
        mae_value  = None
        print("[INFO] Aucune paire commune pour calculer RMSE/MAE.")
    else:
        errors     = prediction_eval_df["score"] - prediction_eval_df[rating_col]
        rmse_value = math.sqrt((errors ** 2).mean())
        mae_value  = errors.abs().mean()
        print(f"[INFO] RMSE = {rmse_value:.6f}")
        print(f"[INFO] MAE  = {mae_value:.6f}")

    # Agrégation globale
    if per_user_df.empty:
        metrics_df = pd.DataFrame([{
            "num_users_evaluated":   0,
            "num_prediction_pairs":  len(prediction_eval_df),
            f"precision@{k}":        0.0,
            f"recall@{k}":           0.0,
            f"f1@{k}":               0.0,
            f"hit_rate@{k}":         0.0,
            f"map@{k}":              0.0,
            f"ndcg@{k}":             0.0,
            "rmse":                  rmse_value,
            "mae":                   mae_value,
        }])
    else:
        metrics_df = pd.DataFrame([{
            "num_users_evaluated":   len(per_user_df),
            "num_prediction_pairs":  len(prediction_eval_df),
            f"precision@{k}":        per_user_df[f"precision@{k}"].mean(),
            f"recall@{k}":           per_user_df[f"recall@{k}"].mean(),
            f"f1@{k}":               per_user_df[f"f1@{k}"].mean(),
            f"hit_rate@{k}":         per_user_df[f"hit_rate@{k}"].mean(),
            f"map@{k}":              per_user_df[f"ap@{k}"].mean(),
            f"ndcg@{k}":             per_user_df[f"ndcg@{k}"].mean(),
            "rmse":                  rmse_value,
            "mae":                   mae_value,
        }])

    metrics_path = output_dir / f"{save_prefix}_global_metrics_top{k}.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"[INFO] Métriques globales sauvegardées : {metrics_path}")

    print("\n===== RÉSULTATS GLOBAUX TASK 2 - UBCF =====")
    print(metrics_df.to_string(index=False))

    return metrics_df, per_user_df


def task_2_evaluation(
    k: int = 20,
    rating_col: str = "rating",
    positive_threshold: float = 4.0,
):
    """
    Point d'entrée de l'évaluation Task 2 (appelé depuis task_2_main.py).
    """

    print("\n" + "=" * 80)
    print("TASK 2 - ÉVALUATION QUANTITATIVE DES RECOMMANDATIONS")
    print("=" * 80)

    # Chargement TEST
    if not path.TEST.exists():
        raise FileNotFoundError(f"Fichier test introuvable : {path.TEST}")

    print(f"[INFO] Chargement TEST depuis {path.TEST}...")
    df_test = pd.read_parquet(path.TEST)
    df_test[rating_col] = pd.to_numeric(df_test[rating_col], errors="coerce")

    # Chargement recommandations Task 2
    task2_rec_path = path.OUTPUTS / "task_2" / f"task_2_top_{k}_recommendations.csv"
    if not task2_rec_path.exists():
        raise FileNotFoundError(f"Recommandations Task 2 introuvables : {task2_rec_path}")

    print(f"[INFO] Chargement recommandations Task 2 depuis {task2_rec_path}...")
    df_rec2 = pd.read_csv(task2_rec_path)

    metrics_t2, per_user_t2 = evaluate_ubcf_recommendations(
        recommendations_df=df_rec2,
        test_df=df_test,
        k=k,
        rating_col=rating_col,
        positive_threshold=positive_threshold,
    )

    # Affichage comparatif
    metrics_t2.insert(0, "task", "T2 - User-Based CF")
    display_cols = [
        "task", f"precision@{k}", f"recall@{k}", f"f1@{k}",
        f"hit_rate@{k}", f"map@{k}", f"ndcg@{k}", "rmse", "mae",
    ]
    available_cols = [c for c in display_cols if c in metrics_t2.columns]
    display_df = metrics_t2[available_cols].reset_index(drop=True)
    display_df.insert(0, "rank", display_df.index + 1)

    print("\n")
    print(display_df.to_string(index=False))

    print("\n" + "=" * 80)
    print("TASK 2 - ÉVALUATION TERMINÉE")
    print("=" * 80)


if __name__ == "__main__":
    task_2_evaluation()