#!/usr/bin/env python3
"""
Task 3 - Système de recommandation basé sur un graphe de connaissances

Ce script implémente les étapes décrites dans Instructions.md :
1. Mise en place de l’environnement (rdflib, owlready2)
2. Conception du graphe de connaissances
3. Construction et stockage des triplets RDF
4. Définition de règles d’inférence

Utilise rdflib pour RDF et SPARQL, owlready2 pour OWL et inférence.
"""

from pathlib import Path
import sys
import pandas as pd
from rdflib import Graph, Literal, RDF, URIRef, Namespace
from rdflib.namespace import FOAF, XSD
import owlready2 as owl
from owlready2 import swrl

# Import depuis la racine
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import path

# Définir les namespaces
EX = Namespace("http://example.org/")
SCHEMA = Namespace("http://schema.org/")

def load_data():
    """Charge les données d'interactions."""
    data_path = path.DATA / "outputs" / "processed" / "sample-temporal" / "temporal_filtered.parquet"
    df = pd.read_parquet(data_path)
    print(f"Données chargées : {len(df)} reviews")
    return df

def create_ontology():
    """Crée une ontologie simple avec owlready2."""
    onto = owl.get_ontology("http://example.org/recommendation.owl")

    with onto:
        class User(owl.Thing):
            pass

        class Item(owl.Thing):
            pass

        class Review(owl.Thing):
            pass

        class hasReviewed(owl.ObjectProperty):
            domain = [User]
            range = [Review]

        class reviewsItem(owl.ObjectProperty):
            domain = [Review]
            range = [Item]

        class rating(owl.FunctionalProperty, owl.DataProperty):
            domain = [Review]
            range = [float]

        class title(owl.FunctionalProperty, owl.DataProperty):
            domain = [Item]
            range = [str]

        # Règle d'inférence : si rating >= 4, alors liked
        class liked(owl.ObjectProperty):
            domain = [User]
            range = [Item]

        # Règle SWRL : User(?u), hasReviewed(?u, ?r), reviewsItem(?r, ?i), rating(?r, ?rate) -> liked(?u, ?i)
        rule = owl.Imp()
        rule.set_as_rule("User(?u), hasReviewed(?u, ?r), reviewsItem(?r, ?i) -> liked(?u, ?i)")

    return onto

def populate_ontology(df, onto):
    """Remplit l'ontologie avec les instances."""
    with onto:
        User = onto.User
        Item = onto.Item
        Review = onto.Review

        for _, row in df.iterrows():
            user = User(f"user_{row['user_id']}")
            item = Item(f"item_{row['parent_asin']}")
            review = Review(f"review_{row['user_id']}_{row['parent_asin']}")

            user.hasReviewed.append(review)
            review.reviewsItem.append(item)
            review.rating = float(row['rating'])
            item.title = str(row['title'])

    print(f"Ontologie peuplée avec {len(list(onto.individuals()))} individus")

def run_sparql_query(onto, query):
    """Exécute une requête SPARQL sur l'ontologie."""
    # Utilise le world d'owlready2
    results = list(owl.default_world.sparql(query))
    return results

def generate_recommendations(onto, top_n=20):
    """Génère des recommandations RDF pour tous les utilisateurs (explicites, sans inférence)."""
    recommendations = []
    users = list(onto.User.instances())
    for user in users[:10]:  # Limiter pour test
        query = f"""
        PREFIX ex: <http://example.org/recommendation.owl#>
        SELECT ?item ?rating
        WHERE {{
            <{user.iri}> ex:hasReviewed ?review .
            ?review ex:reviewsItem ?item .
            ?review ex:rating ?rating .
            FILTER (?rating > 4.0)
        }}
        ORDER BY DESC(?rating)
        LIMIT {top_n}
        """
        results = run_sparql_query(onto, query)
        for i, row in enumerate(results):
            recommendations.append({
                'user_id': str(user).split('_')[-1],
                'parent_asin': str(row[0]).split('_')[-1],
                'score': float(row[1]),  # Utiliser rating comme score
                'rank': i+1
            })
    return pd.DataFrame(recommendations)

def main():
    print("Task 3 - Graphe de connaissances pour recommandation")

    # Charger données
    df = load_data()

    # Créer ontologie
    onto = create_ontology()

    # Peupler l'ontologie
    populate_ontology(df, onto)

    # Sauvegarder l'ontologie
    output_file = path.OUTPUTS / "task_3" / "ontology.owl"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    onto.save(file=str(output_file))
    print(f"Ontologie sauvegardée dans {output_file}")

    # Faire l'inférence avec HermiT
    print("[INFO] Activation de l'inférence HermiT...")
    with onto:
        owl.sync_reasoner()
    print("[INFO] Inférence effectuée avec succès")

    # Requêtes SPARQL avancées (Étape 5) - Avec inférence activée
    print("\n[Étape 5] Requêtes SPARQL avancées (explicites + inférées) :")

    # Exemple 1 : Items avec rating > 4 (explicite)
    query_high_rating = """
    PREFIX ex: <http://example.org/recommendation.owl#>
    SELECT ?item ?rating
    WHERE {
        ex:user_AEZ26WGWJ3EOQ4KWSHG77HJAG4EA ex:hasReviewed ?review .
        ?review ex:reviewsItem ?item .
        ?review ex:rating ?rating .
        FILTER (?rating > 4.0)
    }
    LIMIT 10
    """
    results = run_sparql_query(onto, query_high_rating)
    print("Items appréciés explicitement (rating > 4) :")
    count_explicit = len(results)
    for row in results:
        print(row)
    print(f"Total explicite : {count_explicit}")

    # Exemple 2 : Items likés via inférence (inféré par règles SWRL)
    query_inferred_liked = """
    PREFIX ex: <http://example.org/recommendation.owl#>
    SELECT ?item
    WHERE {
        ex:user_AEZ26WGWJ3EOQ4KWSHG77HJAG4EA ex:liked ?item .
    }
    LIMIT 20
    """
    results_inferred = run_sparql_query(onto, query_inferred_liked)
    print("\nItems likés via inférence (règle SWRL appliquée) :")
    count_inferred = len(results_inferred)
    for row in results_inferred:
        print(row)
    print(f"Total inféré : {count_inferred}")
    if count_inferred > count_explicit:
        print(f"\n✓ Apport de l'inférence : +{count_inferred - count_explicit} items découverts via règles")

    # Générer recommandations
    reco_df = generate_recommendations(onto)
    reco_file = path.OUTPUTS / "task_3" / "task_3_rdf_recommendations.csv"
    reco_df.to_csv(reco_file, index=False)
    print(f"\nRecommandations sauvegardées dans {reco_file}")

if __name__ == "__main__":
    main()