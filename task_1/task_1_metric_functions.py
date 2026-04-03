import math

# Fonctions de métriques Top-N
def precision_at_k(recommended_items: list, relevant_items: set) -> float:
    """
    Precision@K :
    proportion d’items pertinents parmi les K premiers résultats retournés.
    """
    if len(recommended_items) == 0:
        return 0.0
    hits = sum(1 for item in recommended_items if item in relevant_items)
    return hits / len(recommended_items)

def recall_at_k(recommended_items: list, relevant_items: set) -> float:
    """
    Recall@K :
    proportion des items pertinents existants retrouvés dans les K premiers résultats.
    """
    if len(relevant_items) == 0:
        return 0.0
    hits = sum(1 for item in recommended_items if item in relevant_items)
    return hits / len(relevant_items)

def f1_at_k(precision_k: float, recall_k: float) -> float:
    """
    F1@K :
    moyenne harmonique entre Precision@K et Recall@K.
    """
    if precision_k + recall_k == 0:
        return 0.0
    return 2 * (precision_k * recall_k) / (precision_k + recall_k)

def hit_rate_at_k(recommended_items: list, relevant_items: set) -> float:
    """
    HitRate@K :
    vaut 1 si au moins un item pertinent apparaît dans le Top-K, sinon 0.
    """
    return 1.0 if any(item in relevant_items for item in recommended_items) else 0.0

def average_precision_at_k(recommended_items: list, relevant_items: set, k = 20) -> float:
    """
    AP@K (Average Precision@K) :
    moyenne de la Precision@k calculée à chaque position k où l’item à la position k est pertinent, jusqu'à K.
    """
    if len(relevant_items) == 0:
        return 0.0

    hits = 0
    precision_sum = 0.0

    for idx, item in enumerate(recommended_items, start=1):
        if item in relevant_items:
            hits += 1
            precision_sum += hits / idx

    denom = min(len(relevant_items), k)
    if denom == 0:
        return 0.0

    return precision_sum / denom

def ndcg_at_k(recommended_items: list, relevant_items: set) -> float:
    """
    NDCG@K :
    mesure la qualité de l'ordre des résultats, en récompensant davantage les items pertinents placés en haut.
    """
    # DCG réel
    dcg = 0.0
    for idx, item in enumerate(recommended_items, start=1):
        if item in relevant_items:
            dcg += 1.0 / math.log2(idx + 1)

    # DCG idéal
    ideal_hits = min(len(relevant_items), len(recommended_items))
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))

    if idcg == 0:
        return 0.0

    return dcg / idcg