# Diagnostic Task 0 — Préparation des données (P2)

- generated_at: 2026-04-03T15:59:50

## A. Réutilisation du sous-ensemble de travail
- note: `P2 réutilise les sous-ensembles P1 (active/temporal, filtered + splits).`
- methodological_note: `Aucun nouvel échantillonnage massif du corpus complet n'est effectué; les jeux issus de P1 sont réexploités pour la fusion avec meta_Books.`

## B. Documentation des sources

### active_pre_split
- stage: `pre_split`
- variant: `active`
- role: `interactions`
- kind: `single`
- exists: `False`
- format: `parquet`
- rows: `0`
- cols: `0`
- size_bytes: `0`
- paths: `['/home/theo/Documents/Cours/INF6083 - Sujets Spéciaux/P3/uqo-INF6083-P3-Equipe2-Code/data/outputs/processed/sample-active-users/active_users_filtered.parquet']`
- columns names: `[]`

### temporal_pre_split
- stage: `pre_split`
- variant: `temporal`
- role: `interactions`
- kind: `single`
- exists: `True`
- format: `parquet`
- rows: `14045`
- cols: `10`
- size_bytes: `8683147`
- paths: `['/home/theo/Documents/Cours/INF6083 - Sujets Spéciaux/P3/uqo-INF6083-P3-Equipe2-Code/data/outputs/processed/sample-temporal/temporal_filtered.parquet']`
- columns names: `['rating', 'title', 'text', 'images', 'asin', 'parent_asin', 'user_id', 'timestamp', 'helpful_vote', 'verified_purchase']`

### metadata
- stage: `raw`
- variant: `meta_books`
- role: `metadata`
- kind: `single`
- exists: `True`
- format: `parquet`
- rows: `4448181`
- cols: `16`
- size_bytes: `4696898230`
- paths: `['/home/theo/Documents/Cours/INF6083 - Sujets Spéciaux/P3/uqo-INF6083-P3-Equipe2-Code/data/outputs/raw/parquet/meta_Books.parquet']`
- columns names: `['main_category', 'title', 'subtitle', 'author', 'average_rating', 'rating_number', 'features', 'description', 'price', 'images', 'videos', 'store', 'categories', 'details', 'parent_asin', 'bought_together']`


### temporal_pre_split
- chemin de sauvegarde: `/home/theo/Documents/Cours/INF6083 - Sujets Spéciaux/P3/uqo-INF6083-P3-Equipe2-Code/data/outputs/joining/temporal_pre_split_clean_joined.parquet`


### temporal_pre_split
- chemin de sauvegarde: `/home/theo/Documents/Cours/INF6083 - Sujets Spéciaux/P3/uqo-INF6083-P3-Equipe2-Code/data/outputs/joining/temporal_pre_split_clean_joined.parquet`

## C. Vérifications schéma et clés (`parent_asin`)

### metadata
- ok: `True`
- missing_required: `[]`
- missing_parent_asin_count: `0`
- missing_parent_asin_pct: `0.0`
- coercion_warning: `None`
- warnings: `[]`

### active_pre_split
- ok: `False`
- missing_required: `[]`
- missing_parent_asin_count: `None`
- missing_parent_asin_pct: `None`
- coercion_warning: `None`
- warnings: `['Chemin(s) manquant(s), vérification Task 3 ignorée']`

### temporal_pre_split
- ok: `True`
- missing_required: `[]`
- missing_parent_asin_count: `0`
- missing_parent_asin_pct: `0.0`
- coercion_warning: `None`
- warnings: `[]`

## C2. Détection de doublons

### metadata
- n_rows: `1200`
- doublons exacts: `0` (0.0%)
- doublons parent_asin: `0` (0.0%)

### temporal_pre_split
- n_rows: `14045`
- doublons exacts: `314` (2.2357%)
- doublons (user_id, parent_asin): `314` (2.2357%)

## C3. Validation des valeurs (rating, timestamp)

### temporal_pre_split
- rating: min=`1.0`, max=`5.0`, mean=`4.2904`, median=`4.0`, hors intervalle=`0` (0.0%), ok=`True`
- timestamp: dtype=`int64`, min=`2022-01-02 07:02:12.046999931`, max=`2022-12-31 20:30:23.150000095`, non convertibles=`0`, ok=`True`

## C3. Validation des valeurs (rating, timestamp)

### temporal_pre_split
- rating: min=`1.0`, max=`5.0`, mean=`4.2904`, median=`4.0`, hors intervalle=`0` (0.0%), ok=`True`
- timestamp: dtype=`int64`, min=`2022-01-02 07:02:12.046999931`, max=`2022-12-31 20:30:23.150000095`, non convertibles=`0`, ok=`True`

## D. Qualité de jointure via `parent_asin`

### temporal_pre_split
- nb_parent_asin_communs: `1200`
- nb_interactions_jointes / nb_interactions_totales: `14045 / 14045`
- ratio_interactions_jointes: `1.0`
- nb_items_avec_meta / nb_items_totaux: `1200 / 1200`
- ratio_items_avec_meta: `1.0`
- interactions_non_jointes_si_inner_join: `0`
- items_sans_meta: `0`

## E. Attributs exploitables

### temporal_pre_split
- interactions_kept: `['user_id', 'parent_asin', 'rating', 'timestamp', 'text']`
- metadata_text_kept: `['title', 'subtitle', 'features', 'description', 'categories', 'author', 'details']`
- metadata_struct_kept: `['average_rating', 'rating_number', 'price']`
- ignored_interactions_cols: `['title', 'images', 'asin', 'helpful_vote', 'verified_purchase']`
- ignored_metadata_cols: `['main_category', 'images', 'videos', 'store', 'bought_together']`

## F. Valeurs manquantes et stratégie

### temporal_pre_split

#### Interactions brutes

| colonne | type | % NaN | % vide | % effectif | stratégie | justification |
|---------|------|-------|--------|------------|-----------|---------------|
| user_id | clé (identifiant) | 0.0% | 0.0% | 0.0% | supprimer lignes incomplètes (clé obligatoire) | Clé primaire interactions — ligne non identifiable sans user_id. |
| parent_asin | clé (identifiant) | 0.0% | 0.0% | 0.0% | supprimer lignes incomplètes (clé obligatoire) | Clé de jointure obligatoire — toute ligne sans parent_asin est inutilisable. |
| rating | numérique | 0.0% | 0.0% | 0.0% | supprimer lignes incomplètes (variable cible) | Variable cible du système de recommandation — ligne sans note exclue. |
| timestamp | numérique | 0.0% | 0.0% | 0.0% | supprimer lignes incomplètes (clé obligatoire) | Nécessaire au split temporel train/test — ligne inutilisable sans date. |
| text | clé (identifiant) | 0.0% | 0.0% | 0.0% | au cas par cas / hors périmètre | — |

#### Métadonnées globales

| colonne | type | % NaN | % vide | % effectif | stratégie | justification |
|---------|------|-------|--------|------------|-----------|---------------|
| parent_asin | clé (identifiant) | 0.0% | 0.0% | 0.0% | supprimer lignes incomplètes (clé obligatoire) | Clé de jointure obligatoire — toute ligne sans parent_asin est inutilisable. |
| title | textuelle (scalaire) | 0.0% | 0.0% | 0.0% | remplacer NaN par chaîne vide | Contenu textuel principal pour TF-IDF/embeddings ; chaîne vide tolérable car concaténé avec description. |
| subtitle | textuelle (scalaire) | 8.4167% | 0.0% | 8.4167% | remplacer NaN par chaîne vide | Complément textuel mineur ; chaîne vide acceptable, faible impact sur la représentation. |
| features | textuelle (liste) | 0.0% | 0.75% | 0.75% | joindre éléments en string, vide si absent | Points clés marketing ; vide tolérable, données complémentaires au content-based. |
| description | textuelle (liste) | 0.0% | 28.5833% | 28.5833% | joindre éléments en string, vide si absent | Contenu sémantique riche pour content-based filtering ; vide tolérable car compensé par title+features. |
| categories | textuelle (liste) | 0.0% | 0.25% | 0.25% | joindre éléments en string, vide si absent | Taxonomie Amazon pour filtrage par genre ; vide tolérable, données complémentaires. |
| author | catégorielle (struct imbriqué) | 5.75% | 0.0% | 5.75% | aplatir struct (extraire champ clé en string) | Structure imbriquée → extraction de author_name ; vide si auteur inconnu, impact limité. |
| details | catégorielle (struct imbriqué) | 0.0% | 0.0% | 0.0% | aplatir struct (extraire champ clé en string) | Structure imbriquée → publisher/language ; vide tolérable, attributs secondaires. |
| average_rating | numérique | 0.0% | 0.0% | 0.0% | imputation médiane (ou exclusion si trop manquant) | Signal de popularité agrégé ; NaN rare, imputation non nécessaire sauf si >5%. |
| rating_number | numérique | 0.0% | 0.0% | 0.0% | imputation médiane (ou exclusion si trop manquant) | Volume de notes ; NaN rare, indicateur de confiance secondaire. |
| price | numérique | 12.0833% | 0.0% | 12.0833% | imputation médiane (ou exclusion si trop manquant) | Distribution asymétrique, ~24% manquant → médiane plus robuste que la moyenne. |

#### Sous-ensemble joint

| colonne | type | % NaN | % vide | % effectif | stratégie | justification |
|---------|------|-------|--------|------------|-----------|---------------|
| user_id | clé (identifiant) | 0.0% | 0.0% | 0.0% | supprimer lignes incomplètes (clé obligatoire) | Clé primaire interactions — ligne non identifiable sans user_id. |
| parent_asin | clé (identifiant) | 0.0% | 0.0% | 0.0% | supprimer lignes incomplètes (clé obligatoire) | Clé de jointure obligatoire — toute ligne sans parent_asin est inutilisable. |
| rating | numérique | 0.0% | 0.0% | 0.0% | supprimer lignes incomplètes (variable cible) | Variable cible du système de recommandation — ligne sans note exclue. |
| timestamp | numérique | 0.0% | 0.0% | 0.0% | supprimer lignes incomplètes (clé obligatoire) | Nécessaire au split temporel train/test — ligne inutilisable sans date. |
| text | clé (identifiant) | 0.0% | 0.0% | 0.0% | au cas par cas / hors périmètre | — |
| title | textuelle (scalaire) | 0.0% | 0.0% | 0.0% | remplacer NaN par chaîne vide | Contenu textuel principal pour TF-IDF/embeddings ; chaîne vide tolérable car concaténé avec description. |
| subtitle | textuelle (scalaire) | 0.0% | 6.9623% | 6.9623% | remplacer NaN par chaîne vide | Complément textuel mineur ; chaîne vide acceptable, faible impact sur la représentation. |
| features | textuelle (liste) | 0.0% | 0.386% | 0.386% | joindre éléments en string, vide si absent | Points clés marketing ; vide tolérable, données complémentaires au content-based. |
| description | textuelle (liste) | 0.0% | 21.4114% | 21.4114% | joindre éléments en string, vide si absent | Contenu sémantique riche pour content-based filtering ; vide tolérable car compensé par title+features. |
| categories | textuelle (liste) | 0.0% | 0.233% | 0.233% | joindre éléments en string, vide si absent | Taxonomie Amazon pour filtrage par genre ; vide tolérable, données complémentaires. |
| average_rating | numérique | 0.0% | 0.0% | 0.0% | imputation médiane (ou exclusion si trop manquant) | Signal de popularité agrégé ; NaN rare, imputation non nécessaire sauf si >5%. |
| rating_number | numérique | 0.0% | 0.0% | 0.0% | imputation médiane (ou exclusion si trop manquant) | Volume de notes ; NaN rare, indicateur de confiance secondaire. |
| price | numérique | 0.0% | 0.0% | 0.0% | imputation médiane (ou exclusion si trop manquant) | Distribution asymétrique, ~24% manquant → médiane plus robuste que la moyenne. |
| author_name | catégorielle (extraite de struct) | 0.0% | 4.3624% | 4.3624% | au cas par cas / hors périmètre | Extrait de struct author ; vide si auteur inconnu, impact limité. |
| details_publisher | catégorielle (extraite de struct) | 0.0% | 0.0% | 0.0% | au cas par cas / hors périmètre | Extrait de struct details ; vide tolérable, attribut catégoriel secondaire. |
| details_language | catégorielle (extraite de struct) | 0.0% | 0.0% | 0.0% | au cas par cas / hors périmètre | Extrait de struct details ; vide tolérable, quasi-constant ('English'). |


## F2. Qualité des champs textuels

### temporal_pre_split
- title: avg_len=`39.2`, median_len=`29`, vides=`0` (0.0%), HTML=`0` (0.0%)
- subtitle: avg_len=`21.8`, median_len=`25`, vides=`972` (6.9206%), HTML=`0` (0.0%)
- description: avg_len=`5606.6`, median_len=`3165`, vides=`3046` (21.6874%), HTML=`19` (0.1353%)
- categories: avg_len=`48.3`, median_len=`46`, vides=`32` (0.2278%), HTML=`0` (0.0%)
- features: avg_len=`1571.5`, median_len=`1459`, vides=`54` (0.3845%), HTML=`0` (0.0%)
- author_name: avg_len=`12.8`, median_len=`13`, vides=`635` (4.5212%), HTML=`0` (0.0%)
- details_publisher: avg_len=`32.8`, median_len=`33`, vides=`0` (0.0%), HTML=`0` (0.0%)
- details_language: avg_len=`6.9`, median_len=`7`, vides=`0` (0.0%), HTML=`0` (0.0%)

## F3. Nettoyage appliqué (avant / après)

### temporal_pre_split

| métrique | avant | après | delta |
|----------|-------|-------|-------|
| lignes | 14045 | 13731 | −314 |
| items | 1200 | 1200 | −0 |
| users | 417 | 417 | −0 |

**Raisons de suppression :**
- `missing_key_cols`: 0 lignes
- `interaction_duplicates`: 314 lignes

## F4. Vérifications post-nettoyage

### temporal_pre_split

- Doublons résiduels `(user_id, parent_asin)`: **0** — OK
- Distribution rating post-nettoyage: min=1.0, max=5.0, mean=4.2888 — OK
- Intégrité parent_asin: 1200 → 1200 items — OK
- NaN résiduel sur clés: {'user_id': 0, 'parent_asin': 0, 'rating': 0, 'timestamp': 0} — OK

## G. Jeux de données finaux

### temporal_pre_split
- path: `/home/theo/Documents/Cours/INF6083 - Sujets Spéciaux/P3/uqo-INF6083-P3-Equipe2-Code/data/outputs/joining/temporal_pre_split_clean_joined.parquet`
- rows: `13731`
- cols: `16`

## H. Usage des colonnes par tâche

### Représentation de contenu (Tâches 0-2)
- `title`: TF-IDF / embeddings — titre du livre
- `description`: TF-IDF / embeddings — description éditoriale
- `categories`: Encodage catégoriel / multi-hot — taxonomie Amazon
- `features`: TF-IDF — points clés marketing
- `author_name`: Encodage catégoriel — filtrage par auteur

### Variables explicatives (Tâche 3)
- `average_rating`: Variable continue — popularité agrégée de l'item
- `rating_number`: Variable continue — volume de notes (confiance)
- `price`: Variable continue — prix (imputé médiane)
- `details_publisher`: Variable catégorielle — éditeur
- `details_language`: Variable catégorielle — langue
- `author_name`: Variable catégorielle — auteur (partagé avec contenu)

## I. Split temporel train / test

### temporal_pre_split

**Méthode** : `temporal_per_user`

**Règle** : n_test = max(1, floor(n_total × 0.2)), borné à n_total − 1. Utilisateurs avec <3 interactions → train uniquement.

**Justification** : Split temporel : on entraîne sur le passé, on évalue sur le futur. Simule un scénario de déploiement réaliste. Utilisateurs avec <3 interactions → train only (pas assez d'historique pour construire un profil ET tester).

| métrique | train | test |
|----------|-------|------|
| interactions | 11,135 | 2,596 |
| utilisateurs | 417 | 417 |
| items | 1,096 | 550 |
| ratio effectif | 81.09% | 18.91% |

- Users train-only (< 3 interactions) : **0**

- Chaque user test ∈ train : **OK** (violateurs : 0)
- Items test-only : **104** (18.9091%)
  - Items test-only ont une représentation metadata (TF-IDF sur title/description) même sans interaction train — acceptable pour un content-based system.

- `/home/theo/Documents/Cours/INF6083 - Sujets Spéciaux/P3/uqo-INF6083-P3-Equipe2-Code/data/outputs/joining/temporal_pre_split/train_interactions.parquet`
- `/home/theo/Documents/Cours/INF6083 - Sujets Spéciaux/P3/uqo-INF6083-P3-Equipe2-Code/data/outputs/joining/temporal_pre_split/test_interactions.parquet`
