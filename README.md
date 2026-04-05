# INF6083 – P3 : Systèmes de recommandation
### Équipe 2 · Hiver 2026 · Université du Québec en Outaouais

Ce projet explore trois paradigmes fondamentaux des systèmes de recommandation, appliqués à un sous-ensemble du dataset **Amazon Books**. Chaque approche est implémentée de façon indépendante et évaluée avec les mêmes métriques pour permettre une comparaison directe.

---

## Vue d'ensemble du projet

| Tâche | Approche | Fichier principal |
|-------|----------|-------------------|
| **Tâche 0** | Filtrage basé sur le contenu (TF-IDF) | `task_0/scripts/task_0_main.py` |
| **Tâche 1** | Filtrage collaboratif utilisateur (UBCF) | `task_1/task_1_main.py` |
| **Tâche 3** | Graphe de connaissances (RDF/OWL/SPARQL) | `task_3/task_3_rdf.py` |

Le dataset utilisé est un sous-ensemble temporel de reviews Amazon Books : **417 utilisateurs**, **1 200 items**, **14 045 interactions**.

---

## Installation

### Prérequis

- Python 3.10+
- Java JDK 21+ (requis uniquement pour la Tâche 3 — raisonneur HermiT)

### Mise en place de l'environnement

```bash
# Cloner le dépôt et se placer à la racine
git clone <repo-url>
cd uqo-INF6083-P3-Equipe2-Code

# Créer et activer l'environnement virtuel
python -m venv .venv

# Windows
.venv\Scripts\Activate.ps1

# Linux / macOS
source .venv/bin/activate

# Installer les dépendances
pip install -r requirements.txt
```

> **Note Windows** : si `py` ne fonctionne pas dans le terminal, utilisez directement `.venv\Scripts\python.exe` pour lancer les scripts.

---

## Tâche 0 — Filtrage basé sur le contenu (TF-IDF)

### Principe

Chaque item est représenté par un vecteur TF-IDF construit à partir de ses métadonnées textuelles (titre, description, catégories). Le profil d'un utilisateur est calculé comme la moyenne pondérée des vecteurs des items avec lesquels il a interagi. Les recommandations sont produites en calculant la similarité cosinus entre le profil utilisateur et tous les items non encore vus.

### Lancer la tâche

```bash
python task_0/scripts/task_0_main.py
```

### Fichiers générés

```
data/outputs/task_1/
├── task_1_all_users_scores.csv
├── task_1_top_20_test_items_from_train_scores.csv
└── task_1_task_1_evaluation_global_metrics_top20.csv
```

---

## Tâche 1 — Filtrage collaboratif utilisateur (User-Based CF)

### Principe

Le filtrage collaboratif basé sur les utilisateurs (UBCF) ne s'appuie sur aucune description textuelle des items. Il part d'une hypothèse simple : **des utilisateurs qui ont aimé les mêmes choses par le passé auront tendance à apprécier les mêmes choses à l'avenir**.

Pour chaque utilisateur cible, le système :
1. Calcule la **similarité cosinus** entre cet utilisateur et tous les autres, à partir de la matrice d'interactions `R_train` (417 × 1 200).
2. Sélectionne les **K = 50 voisins les plus proches** ayant au moins 2 items en commun.
3. Agrège les ratings des voisins pour estimer l'intérêt de l'utilisateur pour chaque item non encore vu :

```
score(u, i) = Σ sim(u,v) · R[v,i]  /  Σ |sim(u,v)|
```

4. Retourne les **Top-20** items avec les meilleurs scores.

### Lancer la tâche

```bash
python task_1/task_1_main.py
```

### Paramètres configurables (en tête de `task_1_main.py`)

| Paramètre | Valeur par défaut | Description |
|-----------|-------------------|-------------|
| `LIMIT_USERS` | 1000 | Nombre max d'utilisateurs traités |
| `K_NEIGHBORS` | 50 | Nombre de voisins considérés |
| `TOP_N` | 20 | Taille de la liste de recommandations |
| `MIN_COMMON` | 2 | Items en commun minimum pour valider un voisin |

### Fichiers générés

```
data/outputs/task_2/
├── task_2_all_users_scores.csv
├── task_2_top_20_recommendations.csv
├── task_2_evaluation_global_metrics_top20.csv
└── task_2_evaluation_per_user_metrics_top20.csv
```

### Résultats

Sur 417 utilisateurs, **414 ont reçu au moins une recommandation** (couverture 99,3 %). Les 3 utilisateurs sans recommandation sont des profils isolés sans voisin valide dans le dataset (problème de cold start).

---

## Tâche 3 — Graphe de connaissances (RDF/OWL/SPARQL)

### Principe

Cette tâche modélise les données sous forme d'un **graphe de connaissances** en utilisant les technologies RDF et OWL. L'idée centrale est d'enrichir les recommandations non pas par similarité statistique, mais par **inférence logique** à partir de relations sémantiques explicitement définies.

### Architecture du graphe

```
User ── hasReviewed ──▶ Review ── reviewsItem ──▶ Item
                           └── rating (1–5)
                           └── liked (inféré)
```

**Classes OWL** : `User`, `Item`, `Review`

**Règle d'inférence SWRL** :
```
User(?u), hasReviewed(?u, ?r), reviewsItem(?r, ?i) → liked(?u, ?i)
```
Si un utilisateur a écrit un avis sur un item, il est inféré comme "aimé".

### Lancer la tâche

```bash
# Étapes 1 à 5 : création du graphe, inférence, requêtes SPARQL
python task_3/task_3_rdf.py

# Étape 6 : évaluation et comparaison
python task_3/task_3_evaluation.py
```

### Requêtes SPARQL disponibles

**Items explicitement appréciés (rating > 4) :**
```sparql
PREFIX ex: <http://example.org/recommendation.owl#>
SELECT ?item ?rating
WHERE {
    ex:user_123 ex:hasReviewed ?review .
    ?review ex:reviewsItem ?item .
    ?review ex:rating ?rating .
    FILTER (?rating > 4.0)
}
ORDER BY DESC(?rating) LIMIT 20
```

**Items inférés via la relation `liked` :**
```sparql
PREFIX ex: <http://example.org/recommendation.owl#>
SELECT ?item
WHERE { ex:user_123 ex:liked ?item . }
LIMIT 20
```

**Recommandations excluant les items déjà vus :**
```sparql
PREFIX ex: <http://example.org/recommendation.owl#>
SELECT ?item
WHERE {
    ex:user_123 ex:liked ?item .
    FILTER NOT EXISTS {
        ex:user_123 ex:hasReviewed ?review .
        ?review ex:reviewsItem ?item .
    }
}
LIMIT 20
```

### Fichiers générés

```
data/outputs/task_3/
├── ontology.owl
├── graph.ttl
├── task_3_rdf_recommendations.csv
├── task_3_comparison_results.csv
└── task_3_analysis_report.txt
```

---

## Comparaison des trois approches

Le tableau suivant résume les performances évaluées sur **Top-20**, avec un seuil de positivité à **rating ≥ 4,0** :

| Approche | Precision@20 | Recall@20 | F1@20 | HitRate@20 | MAP@20 | NDCG@20 | RMSE | MAE |
|----------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **T0 – Contenu (TF-IDF)** | 0.0106 | 0.0432 | 0.0166 | 0.1711 | 0.0143 | 0.0333 | 4.280 | 4.230 |
| **T1 – Collaboratif (UBCF)** | 0.0062 | 0.0199 | 0.0090 | 0.0867 | 0.0047 | 0.0131 | 0.633 | 0.332 |
| **T2 – Graphe de connaissances** | 0.301 | 0.488 | 0.333 | — | — | — | — | — |

**Points clés :**
- Le **filtrage basé sur le contenu** est le plus efficace pour retrouver de bons items dans un Top-N sur ce dataset (meilleurs Precision, HitRate, NDCG).
- Le **filtrage collaboratif** prédit mieux la note réelle (RMSE 0,63 vs 4,28) mais souffre de la sparsité du dataset pour le classement.
- Le **graphe de connaissances** obtient les meilleures métriques de classement grâce à l'inférence sémantique, au prix d'une complexité de mise en œuvre plus élevée.

---

## Structure du projet

```
.
├── task_0/                         # Tâche 0 — Filtrage contenu
│   └── scripts/
│       ├── task_0_main.py
│       ├── build_tfidf.py
│       ├── build_user_profiles.py
│       ├── joining.py
│       └── precursor.py
│
├── task_1/                         # Tâche 1 — Filtrage collaboratif (UBCF)
│   ├── task_1_main.py
│   ├── task_1_score.py
│   ├── task_1_evaluation.py
│   ├── task_1_metric_functions.py
│   ├── task_1_suggestion.py
│   └── task_1_qualitative_analysis.py
│
├── task_2/                         # Tâche 2 — Filtrage collaboratif utilisateur (UBCF)
│   ├── task_2_main.py
│   └── task_2_evaluation.py
│
├── task_3/                         # Tâche 3 — Graphe de connaissances
│   ├── task_3_rdf.py
│   ├── task_3_evaluation.py
│   └── Instructions.md
│
├── data/
│   ├── input/                      # Données brutes
│   └── outputs/
│       ├── task_1/                 # Sorties Tâche 0
│       ├── task_2/                 # Sorties Tâches 1 et 2
│       └── processed/              # Données pré-traitées (splits, TF-IDF, etc.)
│
├── path.py                         # Chemins centralisés
├── requirements.txt
└── README.md
```

---

## Dépendances principales

```
pandas
numpy
scipy
scikit-learn
pyarrow
rdflib
owlready2
```

Installation complète : `pip install -r requirements.txt`

---

*Équipe 2 — INF6083 Projet P3, Hiver 2026*