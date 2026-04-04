#!/usr/bin/env python3
"""
Task 2 - Évaluation et comparaison (Étape 6)

Évalue les recommandations RDF et compare avec Task 0 (contenu) et Task 1 (collaboratif).
"""

import sys
import pandas as pd
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import path

# Importer métriques de Task 1
sys.path.insert(0, str(_REPO_ROOT / "task_1"))
from task_1.task_1_metric_functions import precision_at_k, recall_at_k, f1_at_k

def load_test_data():
    """Charge les données de test."""
    test_file = path.JOINING_TEST
    return pd.read_parquet(test_file)

def evaluate_recommendations(reco_df, test_df, k=20):
    """Évalue les recommandations avec métriques Top-K."""
    # Grouper par user
    reco_grouped = reco_df.groupby('user_id')
    test_grouped = test_df.groupby('user_id')

    metrics = {'precision': [], 'recall': [], 'f1': []}
    for user_id in test_grouped.groups:
        if user_id in reco_grouped.groups:
            reco_items = reco_grouped.get_group(user_id)['parent_asin'].tolist()[:k]
            test_items = test_grouped.get_group(user_id)['parent_asin'].tolist()
            # Simuler ratings positifs
            relevant = set(test_items)
            recommended = set(reco_items)

            p = precision_at_k(reco_items, relevant)
            r = recall_at_k(reco_items, relevant)
            f = f1_at_k(p, r)

            metrics['precision'].append(p)
            metrics['recall'].append(r)
            metrics['f1'].append(f)

    # Moyennes
    avg_metrics = {m: sum(v)/len(v) if v else 0 for m, v in metrics.items()}
    return avg_metrics

def compare_with_previous():
    """Compare avec Task 0 et Task 1."""
    # Charger métriques de Task 0 et Task 1 (si disponibles)
    task0_metrics = {}
    task1_metrics = {}
    
    # Essayer charger Task 1 (le plus proche pour comparaison)
    task1_eval_file = path.OUTPUTS / "task_1" / "task_1_task_1_evaluation_global_metrics_top20.csv"
    if task1_eval_file.exists():
        df_task1 = pd.read_csv(task1_eval_file)
        if not df_task1.empty:
            # Extraire les moyennes
            task1_metrics = {
                'precision': df_task1['precision@20'].mean() if 'precision@20' in df_task1.columns else 'N/A',
                'recall': df_task1['recall@20'].mean() if 'recall@20' in df_task1.columns else 'N/A',
                'f1': df_task1['f1@20'].mean() if 'f1@20' in df_task1.columns else 'N/A'
            }
    
    task2_metrics = evaluate_recommendations(pd.read_csv(path.OUTPUTS / 'task_2' / 'task_2_rdf_recommendations.csv'), load_test_data())
    
    print("\nComparaison (moyennes à Top-20) :")
    print(f"Task 0 (Contenu) : {task0_metrics if task0_metrics else 'N/A'}")
    print(f"Task 1 (Collaboratif UBCF) : {task1_metrics if task1_metrics else 'N/A'}")
    print(f"Task 2 (RDF/Graphe) : {task2_metrics}")

    # Discussion
    print("\nAnalyse comparative :")
    print("- Apport sémantique : RDF permet inférence et relations riches (catégories, auteurs, etc.)")
    print("- Impact inférence : Augmente recommandations en découvrant connections explicites.")
    print("- Avantages RDF :")
    print("  * Explicabilité : tracer POURQUOI un item est recommandé via requêtes.")
    print("  * Flexibilité : ajouter nouvelles relations/règles sans recalcul complet.")
    print("  * Sémantique : capturer domaine métier (catégories, auteurs, etc.)")
    print("- Limites RDF :")
    print("  * Performance : graphes volumineux ralentissent inférence.")
    print("  * Complétude : dépend de qualité de l'ontologie et des données.")
    print("  * Froides-start : moins efficace que collaboratif pour users/items nouveaux.")
    
    # Sauvegarder les résultats de comparaison
    comparison_file = path.OUTPUTS / "task_2" / "task_2_comparison_results.csv"
    comparison_data = {
        'approach': ['Task 0 (Contenu)', 'Task 1 (Collaboratif)', 'Task 2 (RDF)'],
        'precision': [task0_metrics.get('precision', 'N/A'), task1_metrics.get('precision', 'N/A'), task2_metrics.get('precision', 'N/A')],
        'recall': [task0_metrics.get('recall', 'N/A'), task1_metrics.get('recall', 'N/A'), task2_metrics.get('recall', 'N/A')],
        'f1': [task0_metrics.get('f1', 'N/A'), task1_metrics.get('f1', 'N/A'), task2_metrics.get('f1', 'N/A')]
    }
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv(comparison_file, index=False)
    print(f"\n✓ Comparaison sauvegardée dans {comparison_file}")
    
    # Sauvegarder l'analyse détaillée
    analysis_file = path.OUTPUTS / "task_2" / "task_2_analysis_report.txt"
    with open(analysis_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("TASK 2 - ÉVALUATION ET COMPARAISON (ÉTAPE 6)\n")
        f.write("=" * 80 + "\n\n")
        f.write("RÉSULTATS DE COMPARAISON (Top-20)\n")
        f.write("-" * 80 + "\n")
        f.write(f"Task 0 (Contenu) : {task0_metrics if task0_metrics else 'N/A'}\n")
        f.write(f"Task 1 (Collaboratif UBCF) : {task1_metrics if task1_metrics else 'N/A'}\n")
        f.write(f"Task 2 (RDF/Graphe) : {task2_metrics}\n\n")
        f.write("ANALYSE COMPARATIVE\n")
        f.write("-" * 80 + "\n")
        f.write("- Apport sémantique : RDF permet inférence et relations riches (catégories, auteurs, etc.)\n")
        f.write("- Impact inférence : Augmente recommandations en découvrant connections explicites.\n")
        f.write("- Avantages RDF :\n")
        f.write("  * Explicabilité : tracer POURQUOI un item est recommandé via requêtes.\n")
        f.write("  * Flexibilité : ajouter nouvelles relations/règles sans recalcul complet.\n")
        f.write("  * Sémantique : capturer domaine métier (catégories, auteurs, etc.)\n")
        f.write("- Limites RDF :\n")
        f.write("  * Performance : graphes volumineux ralentissent inférence.\n")
        f.write("  * Complétude : dépend de qualité de l'ontologie et des données.\n")
        f.write("  * Froides-start : moins efficace que collaboratif pour users/items nouveaux.\n")
    print(f"✓ Analyse sauvegardée dans {analysis_file}")

def main():
    print("Étape 6 - Évaluation et comparaison")
    test_df = load_test_data()
    reco_df = pd.read_csv(path.OUTPUTS / "task_2" / "task_2_rdf_recommendations.csv")
    metrics = evaluate_recommendations(reco_df, test_df)
    print(f"Métriques RDF : {metrics}")
    compare_with_previous()

if __name__ == "__main__":
    main()