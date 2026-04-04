from pathlib import Path

# Racine du projet
ROOT = Path(__file__).resolve().parent

# Point d'entrée des données (dossier `data/`)
DATA = ROOT / "data"

# ===================================================
# GESTION DE L'INPUT/OUTPUTS
# ===================================================
INPUT = DATA / "input"              # Jeu de données brut
OUTPUTS = DATA / "outputs"          # Fichiers intermédiaires générés
JOINING = OUTPUTS / "joining"       # Jointures des données

DATA.mkdir(parents=True, exist_ok=True)
INPUT.mkdir(parents=True, exist_ok=True)
OUTPUTS.mkdir(parents=True, exist_ok=True)
JOINING.mkdir(parents=True, exist_ok=True)

# FICHIERS
RAW_BOOK = OUTPUTS / "raw" / "parquet" / "Books.parquet"
RAW_METABOOK = OUTPUTS / "raw" / "parquet" / "meta_Books.parquet"
TEMPORAL = OUTPUTS / "processed" / "sample-temporal"
SPLITS = TEMPORAL / "splits"
R_TEST = SPLITS / "R_test.npz"
R_TRAIN = SPLITS / "R_train.npz"
TEST = SPLITS / "test.parquet"
TRAIN = SPLITS / "train.parquet"
ITEMS = SPLITS / "item_ids.npy"
USERS = SPLITS / "user_ids.npy"
JOINING_TEST = JOINING / "temporal_pre_split" / "test_interactions.parquet"
JOINING_TRAIN = JOINING / "temporal_pre_split" / "train_interactions.parquet"

# task 0
ITEM_TFIDF = SPLITS / "item_tfidf_matrix.npz"
ITEM_TO_IDX = SPLITS / "item_to_idx.pkl"
TFIDF_VECTORIZER = SPLITS / "tfidf_vectorizer.pkl"
ITEM_METADATA_LIGHT = SPLITS / "item_metadata_light.parquet"
USER_HISTORIES = SPLITS / "user_seen_items_train.pkl"
USER_TO_IDX = SPLITS / "user_to_idx.pkl"
USER_PROFILES = SPLITS / "user_profiles_matrix.npz"

# task 1
TASK1_REC = OUTPUTS / "task_1" / "task_1_top_20_test_items_from_train_scores.csv"

# task 2
TASK2_REC = OUTPUTS / "task_2" / "task_2_top_20_recommendations.csv"