# 3.5.1 Évaluation quantitative
# Vous devrez définir un protocole d'évaluation hors ligne cohérent avec la nature de votre système.
# Les métriques de classement de type top-N sont attendues.
#
# Les métriques calculées :
#   - Precision@K
#   - Recall@K
#   - F1@K
#   - HitRate@K
#   - MAP@K
#   - NDCG@K
#   - RMSE
#   - MAE
#
# Convention :
#   - Les items pertinents dans TEST sont ceux avec rating >= positive_threshold
#   - L'évaluation Top-N se fait par utilisateur
#   - RMSE / MAE se font sur l'intersection entre recommandations et TEST
#
# IMPORTANT :
#   - T1 : format attendu = user_id, parent_asin, score, rank

import os
import math
import numpy as np
import pandas as pd
import sys
from pathlib import Path
from task_1_metric_functions import average_precision_at_k, f1_at_k, hit_rate_at_k, ndcg_at_k, precision_at_k, recall_at_k

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import path

# FONCTION INTERNE COMMUNE (moteur d'évaluation)
def _evaluate_recommendations_common(
    recommendations_df: pd.DataFrame,
    test_df: pd.DataFrame,
    k: int = 10,
    rating_col: str = "rating",
    positive_threshold: float = 4.0,
    save_prefix: str = "task_1_evaluation_items"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fonction interne commune utilisée par Tâche 1, Tâche 2 et Tâche 3.

    Paramètres
    ----------
    recommendations_df : pd.DataFrame
        Format standardisé attendu :
            - user_id
            - parent_asin
            - score

    test_df : pd.DataFrame
        Jeu de test avec au minimum :
            - user_id
            - parent_asin
            - rating_col (par défaut 'rating')

    k : int
        Top-K évalué.

    rating_col : str
        Nom de la colonne des notes réelles dans TEST.

    positive_threshold : float
        Seuil de positivité pour considérer une interaction comme pertinente.

    save_prefix : str
        Préfixe utilisé pour les exports.

    Retour
    ------
    tuple[pd.DataFrame, pd.DataFrame]
        - metrics_df : métriques globales
        - per_user_df : métriques par utilisateur
    """

    # Vérification des colonnes attendues
    required_rec_cols = {"user_id", "parent_asin", "score"}
    required_test_cols = {"user_id", "parent_asin", rating_col}

    missing_rec_cols = required_rec_cols - set(recommendations_df.columns)
    missing_test_cols = required_test_cols - set(test_df.columns)

    if missing_rec_cols:
        raise ValueError(f"Colonnes manquantes dans recommendations_df : {missing_rec_cols}")
    if missing_test_cols:
        raise ValueError(f"Colonnes manquantes dans test_df : {missing_test_cols}")

    # Copies de sécurité
    rec_df = recommendations_df.copy()
    test_copy = test_df.copy()

    # Construction du ground truth depuis TEST, on ne garde que les interactions positives : rating >= positive_threshold
    positive_test_df = test_copy[test_copy[rating_col] >= positive_threshold].copy()

    positive_test_path = os.path.join(path.OUTPUTS / "task_1", f"{save_prefix}_positive_test_interactions.csv")
    positive_test_df.to_csv(positive_test_path, index=False)
    print(f"\n[INFO] Interactions positives TEST sauvegardées : {positive_test_path}")

    # Ground truth = ensemble des items pertinents par utilisateur
    ground_truth = (
        positive_test_df.groupby("user_id")["parent_asin"]
        .apply(set)
        .to_dict()
    )

    evaluable_users = set(ground_truth.keys())

    print(f"\n[INFO] Utilisateurs évaluables (>=1 item positif dans TEST) : {len(evaluable_users)}")

    # Préparation des recommandations Top-K, on ne garde que les utilisateurs évaluables
    rec_df = rec_df[rec_df["user_id"].isin(evaluable_users)].copy()

    # Tri décroissant par score
    rec_df = rec_df.sort_values(["user_id", "score"], ascending=[True, False])

    # Suppression des doublons éventuels (user_id, parent_asin)
    rec_df = rec_df.drop_duplicates(subset=["user_id", "parent_asin"], keep="first")

    # Ajout du rang par utilisateur
    rec_df["rank"] = rec_df.groupby("user_id").cumcount() + 1

    # Garde seulement le Top-K
    topk_rec_df = rec_df[rec_df["rank"] <= k].copy()

    topk_path = os.path.join(path.OUTPUTS / "task_1", f"{save_prefix}_top{k}_recommendations.csv")
    topk_rec_df.to_csv(topk_path, index=False)
    print(f"\n[INFO] Recommandations Top-{k} sauvegardées : {topk_path}")

    # Calcul des métriques par utilisateur
    per_user_metrics = []

    for user_id in sorted(evaluable_users):
        relevant_items = ground_truth[user_id]

        # Liste ordonnée des recommandations pour cet utilisateur
        user_recs = topk_rec_df.loc[
            topk_rec_df["user_id"] == user_id, "parent_asin"
        ].tolist()

        # Calcul des métriques
        p_at_k = precision_at_k(user_recs, relevant_items)
        r_at_k = recall_at_k(user_recs, relevant_items)
        f1_k = f1_at_k(p_at_k, r_at_k)
        hr_k = hit_rate_at_k(user_recs, relevant_items)
        ap_k = average_precision_at_k(user_recs, relevant_items)
        ndcg_k = ndcg_at_k(user_recs, relevant_items)

        per_user_metrics.append({
            "user_id": user_id,
            "num_relevant_items_test": len(relevant_items),
            "num_recommended_items": len(user_recs),
            f"precision@{k}": p_at_k,
            f"recall@{k}": r_at_k,
            f"f1@{k}": f1_k,
            f"hit_rate@{k}": hr_k,
            f"ap@{k}": ap_k,
            f"ndcg@{k}": ndcg_k
        })

    per_user_df = pd.DataFrame(per_user_metrics)

    per_user_path = os.path.join(path.OUTPUTS / "task_1", f"{save_prefix}_per_user_metrics_top{k}.csv")
    per_user_df.to_csv(per_user_path, index=False)
    print(f"\n[INFO] Métriques par utilisateur sauvegardées : {per_user_path}")

    # RMSE / MAE -> intersection entre recommandations et TEST sur (user_id, parent_asin)
    prediction_eval_df = pd.merge(
        rec_df[["user_id", "parent_asin", "score"]],
        test_copy[["user_id", "parent_asin", rating_col]],
        on=["user_id", "parent_asin"],
        how="inner"
    )

    prediction_eval_path = os.path.join(path.OUTPUTS / "task_1", f"{save_prefix}_prediction_pairs.csv")
    prediction_eval_df.to_csv(prediction_eval_path, index=False)
    print(f"\n[INFO] Paires RMSE/MAE sauvegardées : {prediction_eval_path}")

    if prediction_eval_df.empty:
        rmse_value = None
        mae_value = None
        print(f"\n[INFO] Aucune paire commune pour calculer RMSE/MAE.")
    else:
        errors = prediction_eval_df["score"] - prediction_eval_df[rating_col]
        rmse_value = math.sqrt((errors ** 2).mean())
        mae_value = errors.abs().mean()

        print(f"\n[INFO] RMSE = {rmse_value:.6f}")
        print(f"\n[INFO] MAE  = {mae_value:.6f}")

    # Agrégation globale
    if per_user_df.empty:
        metrics_df = pd.DataFrame([{
            "num_users_evaluated": 0,
            "num_prediction_pairs": len(prediction_eval_df),
            f"precision@{k}": 0.0,
            f"recall@{k}": 0.0,
            f"f1@{k}": 0.0,
            f"hit_rate@{k}": 0.0,
            f"map@{k}": 0.0,
            f"ndcg@{k}": 0.0,
            "rmse": rmse_value,
            "mae": mae_value
        }])
    else:
        metrics_df = pd.DataFrame([{
            "num_users_evaluated": len(per_user_df),
            "num_prediction_pairs": len(prediction_eval_df),
            f"precision@{k}": per_user_df[f"precision@{k}"].mean(),
            f"recall@{k}": per_user_df[f"recall@{k}"].mean(),
            f"f1@{k}": per_user_df[f"f1@{k}"].mean(),
            f"hit_rate@{k}": per_user_df[f"hit_rate@{k}"].mean(),
            f"map@{k}": per_user_df[f"ap@{k}"].mean(),
            f"ndcg@{k}": per_user_df[f"ndcg@{k}"].mean(),
            "rmse": rmse_value,
            "mae": mae_value
        }])

    metrics_path = os.path.join(
        path.OUTPUTS / "task_1",
        f"{save_prefix}_global_metrics_top{k}.csv"
    )
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\n[INFO] Métriques globales sauvegardées : {metrics_path}")

    print("\n===== RÉSULTATS GLOBAUX =====")
    print(metrics_df.to_string(index=False))

    return metrics_df, per_user_df


# TÂCHE 1 : ÉVALUATION SIMILARITÉ DIRECTE
def evaluate_task1_recommendations(
    recommendations_df: pd.DataFrame,
    test_df: pd.DataFrame,
    k: int = 10,
    rating_col: str = "rating",
    positive_threshold: float = 4.0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("\n==================================================")
    print("ÉVALUATION TÂCHE 1 - Similarité directe")
    print("==================================================")

    return _evaluate_recommendations_common(
        recommendations_df=recommendations_df,
        test_df=test_df,
        k=k,
        rating_col=rating_col,
        positive_threshold=positive_threshold,
        save_prefix="task_1_task_1_evaluation"
    )

# ─────────────────────────────────────────────────────────────
# FONCTION PRINCIPALE
# ─────────────────────────────────────────────────────────────
def task_1_evaluation_items(
    k: int = 20,
    rating_col: str = "rating",
    positive_threshold: float = 4.0
):
    """
    Évalue les recommandations sur le test set

    Paramètres
    ----------
    k : int
        Top-K pour l'évaluation.
    rating_col : str
        Nom de la colonne des notes réelles dans TEST.
    positive_threshold : float
        Seuil de positivité pour considérer une interaction comme pertinente.
    """

    print("\n" + "=" * 80)
    print("TASK 4 - ÉVALUATION QUANTITATIVE DES RECOMMANDATIONS")
    print("=" * 80)

    task4_dir = path.OUTPUTS / "task_1"
    task4_dir.mkdir(parents=True, exist_ok=True)

    # ── Chargement TEST ───────────────────────────────────
    if not path.TEST.exists():
        raise FileNotFoundError(f"Fichier test introuvable : {path.TEST}")

    print(f"[INFO] Chargement TEST depuis {path.TEST}...")
    df_test = pd.read_parquet(path.TEST)
    df_test[rating_col] = pd.to_numeric(df_test[rating_col], errors="coerce")

    # ── T1 - Similarité directe ──────────────────────────
    if not path.TASK1_REC.exists():
        raise FileNotFoundError(f"Recommandations T1 introuvables : {path.TASK1_REC}")

    print(f"\n[INFO] Chargement recommandations T1 depuis {path.TASK1_REC}...")
    df_rec1 = pd.read_csv(path.TASK1_REC)
    metrics_t1, per_user_t1 = evaluate_task1_recommendations(
        recommendations_df=df_rec1,
        test_df=df_test,
        k=k,
        rating_col=rating_col,
        positive_threshold=positive_threshold
    )

    # Affichage console du tableau trié
    metrics_t1.insert(0, "task", "T1 - Similarité des items")
    display_cols = ["task", f"precision@{k}", f"recall@{k}", f"f1@{k}",
                    f"hit_rate@{k}", f"map@{k}", f"ndcg@{k}", "rmse", "mae"]
    available_cols = [c for c in display_cols if c in metrics_t1.columns]
    display_df = metrics_t1[available_cols].reset_index(drop=True)
    display_df.insert(0, "rank", display_df.index + 1)


    print("\n")
    print(display_df.to_string(index=False))

    print("\n" + "=" * 80)
    print("TASK 4 - ÉVALUATION TERMINÉE")
    print("=" * 80)


if __name__ == "__main__":
    task_1_evaluation_items()