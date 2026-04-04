# uqo-INF6083-P3-Equipe2-Code

## Projet : Systèmes de recommandation (INF6083 - P3)

Ce projet implémente trois approches de systèmes de recommandation basées sur des données Amazon Books :

1. **Task 0** : Filtrage basé sur le contenu (TF-IDF)
2. **Task 1** : Filtrage collaboratif utilisateur (UBCF)
3. **Task 2** : Système de recommandation basé sur un graphe de connaissances (RDF/OWL)

---

## Task 2 : Graphe de connaissances et recommandation sémantique

### 📋 Vue d'ensemble

Task 2 implémente un **système de recommandation basé sur un graphe de connaissances** exploitant des technologies RDF (Resource Description Framework) et OWL (Web Ontology Language). Le système utilise l'inférence logique pour enrichir les recommandations avec des connaissances déduites automatiquement.

### 🎯 Objectifs

Concevoir et implémenter les 6 étapes décrites dans `Instructions.md` :

1. **Mise en place de l'environnement** : Bibliothèques RDF/OWL
2. **Conception du graphe de connaissances** : Modélisation ontologique
3. **Construction et stockage des triplets** : Transformation en RDF/OWL
4. **Règles d'inférence** : Axiomes OWL et règles SWRL
5. **Exploitation et interrogation avancée** : Requêtes SPARQL
6. **Évaluation et comparaison** : Métriques vs Task 1

### 🏗️ Architecture

#### Entités principales

```
User ← hasReviewed → Review → reviewsItem → Item
```

**Classes OWL** :
- `User` : Utilisateur (ex. : `user_AEZ26WGWJ3EOQ4KWSHG77HJAG4EA`)
- `Item` : Article/Livre (ex. : `item_B09NTKJMWX`)
- `Review` : Avis client (ex. : `review_user_item`)

**Propriétés** :
- `hasReviewed` (ObjectProperty) : Utilisateur → Review
- `reviewsItem` (ObjectProperty) : Review → Item
- `rating` (DataProperty) : Note de 1-5
- `title` (DataProperty) : Titre de l'item
- `liked` (ObjectProperty, inféré) : Utilisateur → Item "aimé"

#### Règles d'inférence SWRL

```sparql
User(?u), hasReviewed(?u, ?r), reviewsItem(?r, ?i) → liked(?u, ?i)
```

**Signification** : Si un utilisateur a écrit une review d'un item, alors cet item est considéré comme "aimé" par l'utilisateur.

### 🚀 Utilisation

#### Installation des dépendances

```bash
pip install rdflib owlready2 pandas
# Java JDK 21+ requis pour HermiT reasoner
```

#### Lancer le pipeline complet

```bash
python task_2/task_2_rdf.py       # Générer ontologie et recommandations
python task_2/task_2_evaluation.py # Évaluer et comparer
```

#### Étapes du pipeline

1. **Chargement des données** : `temporal_filtered.parquet` (14,045 reviews)
2. **Création ontologie** : Définition des classes et propriétés OWL
3. **Peuplement** : Création de ~15,348 individus RDF
4. **Inférence HermiT** : Application des règles SWRL (~3-5s)
5. **Requêtes SPARQL** : Extraction items explicites et inférés
6. **Recommandations** : Génération scores basés sur ratings (Top-20)

### 📊 Résultats (Top-20)

| Métrique | Task 1 (UBCF) | Task 2 (RDF) | Amélioration |
|----------|---|---|---|
| **Precision** | 0.0106 | 0.301 | +2,740% ✓ |
| **Recall** | 0.0432 | 0.488 | +1,030% ✓ |
| **F1** | 0.0166 | 0.333 | +1,906% ✓ |

### 📁 Fichiers générés

```
data/outputs/task_2/
├── ontology.owl                      # Ontologie OWL complète avec inférences
├── task_2_rdf_recommendations.csv    # Recommandations Top-20 par user
├── task_2_comparison_results.csv     # Comparaison métriques Task 0/1/2
├── task_2_analysis_report.txt        # Analyse détaillée
└── graph.ttl                         # Graphe RDF en format Turtle
```

### 🔍 Requêtes SPARQL exemples

#### Trouver items appréciés explicitement (rating > 4)

```sparql
PREFIX ex: <http://example.org/recommendation.owl#>
SELECT ?item ?rating
WHERE {
    ex:user_123 ex:hasReviewed ?review .
    ?review ex:reviewsItem ?item .
    ?review ex:rating ?rating .
    FILTER (?rating > 4.0)
}
ORDER BY DESC(?rating)
LIMIT 20
```

#### Trouver items "likés" via inférence

```sparql
PREFIX ex: <http://example.org/recommendation.owl#>
SELECT ?item
WHERE {
    ex:user_123 ex:liked ?item .
}
LIMIT 20
```

#### Recommandations excluant items déjà vus

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

### 💡 Avantages du graphe de connaissances

✓ **Explicabilité** : Tracer POURQUOI un item est recommandé via SPARQL  
✓ **Flexibilité** : Ajouter nouvelles relations/règles sans recalcul complet  
✓ **Sémantique** : Capturer domaine métier (catégories, auteurs, etc.)  
✓ **Inférence** : Découvrir connections implicites automatiquement  
✓ **Performance** : Task 2 surpasse Task 1 de ~27x en Precision  

### ⚠️ Limitations

✗ **Performance** : Graphes volumineux ralentissent inférence (HermiT)  
✗ **Complétude** : Dépend de qualité de l'ontologie et données  
✗ **Cold-start** : Moins efficace pour users/items nouveaux  
✗ **Complexité** : Courbe d'apprentissage RDF/OWL/SPARQL  

### 📚 Références

- [rdflib documentation](https://rdflib.readthedocs.io/)
- [Owlready2 documentation](https://owlready2.readthedocs.io/)
- [W3C SPARQL specification](https://www.w3.org/TR/sparql11-query/)
- [OWL Web Ontology Language](https://www.w3.org/OWL/)

---

## Structure du projet

```
.
├── task_0/              # Filtrage contenu (TF-IDF)
├── task_1/              # Filtrage collaboratif (UBCF)
├── task_2/              # Graphe connaissances (RDF/OWL)
│   ├── task_2_rdf.py           ✓ Création ontologie + inférence
│   ├── task_2_evaluation.py    ✓ Évaluation et comparaison
│   └── Instructions.md         📋 Spécifications
├── data/
│   ├── input/           # Données brutes
│   └── outputs/
│       ├── task_1/
│       ├── task_2/      ✓ Outputs Task 2 (voir ci-dessus)
│       └── processed/
├── requirements.txt     # Dépendances Python
└── README.md           # Ce fichier
```

---

## Installation et configuration

### Prérequis

- Python 3.10+
- Java JDK 21+ (pour HermiT reasoner)
- pip

### Setup

```bash
# Créer venv
python -m venv .venv
source .venv/Scripts/activate  # Windows
# ou
source .venv/bin/activate      # Linux/macOS

# Installer dépendances
pip install -r requirements.txt

# Ajouter Java au PATH (si nécessaire)
export PATH="$PATH:/usr/local/bin/java"  # Linux/macOS
# ou sur Windows : Paramètres système → Variables d'environnement
```

### Exécution

```bash
# Task 2 complète (étapes 1-5)
python task_2/task_2_rdf.py

# Évaluation et comparaison (étape 6)
python task_2/task_2_evaluation.py
```

---

## Auteurs

Équipe 2 - INF6083 Projet P3 (2026)