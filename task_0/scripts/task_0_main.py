"""
main.py - Orchestrateur complet du pipeline de préparation des données.

Exécute séquentiellement :
  1. Échantillonnage des utilisateurs actifs  (GPU si disponible, sinon CPU)
  2. Échantillonnage temporel                 (GPU si disponible, sinon CPU)
  3. Nettoyage des échantillons               (CPU / PyArrow)
  4. Filtrage itératif par seuils d'activité  (CPU / PyArrow)
  5. Split train/test + matrices CSR + sauvegarde  (CPU / PyArrow)

Usage :
  python task_0/scripts/main.py              # pipeline complet (50 000 users)
  python task_0/scripts/main.py --quick      # sous-échantillon rapide (2 000 users)
  python task_0/scripts/main.py --users 5000 # nombre d'utilisateurs personnalisé
"""

import argparse
import gc
import os
import time
import sys
import subprocess
import multiprocessing

from pathlib import Path

# Ensure the task_0 directory is on sys.path so `import scripts.*` works when
# running `python task_0/scripts/main.py` from the repo root.
_TASK0_ROOT = Path(__file__).resolve().parents[1]
if str(_TASK0_ROOT) not in sys.path:
    sys.path.insert(0, str(_TASK0_ROOT))

# Also ensure the repository root is on sys.path so `import path` works.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.precursor import (
    RAPIDS_AVAILABLE,
    RAW_BOOKS_PATH,
    SAMPLE_ACTIVE_DIR,
    SAMPLE_TEMPORAL_DIR,
    SAMPLE_GLOB_FILTERED,
    TARGET_YEARS,
    RAW_META_PATH,
    # Dataset preparation (JSONL  to Parquet)
    jsonl_to_parquet_conversion,
    resolve_glob,
    # GPU sampling
    sample_active_users_gpu,
    sample_temporal_gpu,
    # CPU sampling (fallback)
    sample_active_users_cpu,
    sample_temporal_cpu,
    # Post-processing (CPU)
    clean_samples,
    filter_samples,
    split_and_save,
    # Memory helpers
    flush_ram,
    flush_gpu,
)

from scripts.joining import(
    cli_print_md_results,
    cli_print_results,
    run_all
    )

from scripts.build_tfidf import build_tfidf
from scripts.build_user_profiles import build_user_profile


def _final_files_checker() -> bool:

    result = True
    filtered_data_paths = resolve_glob(SAMPLE_GLOB_FILTERED)
    active_splits_dir_path = f"{SAMPLE_ACTIVE_DIR}/splits/"
    temporal_splits_dir_path = f"{SAMPLE_TEMPORAL_DIR}/splits/"


    if len(filtered_data_paths) == 0:
        result = False
    for path in filtered_data_paths:
        if os.path.getsize(path) < 1024:
            result = False

    active_splits_dir = os.listdir(active_splits_dir_path) if os.path.isdir(active_splits_dir_path) else []
    if len(active_splits_dir) == 0:
        result = False

    temporal_splits_dir = os.listdir(temporal_splits_dir_path) if os.path.isdir(temporal_splits_dir_path) else []
    if len(temporal_splits_dir) == 0:
        result = False

    if not os.path.isfile(RAW_META_PATH) or os.path.getsize(RAW_META_PATH) < 1024:
        result = False


    return result



def _joining_files_checker() -> bool:

    print(r"Reutilisation des ensembles P1")

    result = True
    joined_data_paths = resolve_glob("data/joining/*_joined.parquet")

    if len(joined_data_paths) == 0:
        result = False
    for path in joined_data_paths:
        print(f"Jointure chemin : {path}")
        if os.path.getsize(path) < 1024:
            result = False

    return result



def precursor(num_users: int = None, target = "TEMPORAL"):
    t_start = time.time()

    use_gpu = RAPIDS_AVAILABLE
    backend = "GPU (RAPIDS)" if use_gpu else "CPU (PyArrow)"

    # Utilise NUM_USERS par défaut si rien n'est spécifié
    if num_users is None:
        num_users = NUM_USERS

    print(f"Pipeline de préparation - backend : {backend}")
    print(f"Nombre d'utilisateurs cible : {num_users:,}\n")

    # -- 0. Conversion Dataset ----------------------------------------
    result = False
    try:
        result = jsonl_to_parquet_conversion()
    except Exception as e:
        print(f"  Conversion Dataset jsonl_to_parquet_conversion a échoué : {e}")
        sys.exit(1)

    if result:

        # -- 1. Échantillonnage : utilisateurs actifs ---------------------
        if(target == "BOTH" or target == "ACTIVE"):
            print("=" * 70)
            print("  ÉTAPE 1/5 : Échantillonnage des utilisateurs actifs")
            print("=" * 70)

            active_out = f"{SAMPLE_ACTIVE_DIR}/active_users_original.parquet"
            if use_gpu:
                sample_active_users_gpu(RAW_BOOKS_PATH, active_out, num_users=num_users)
            else:
                sample_active_users_cpu(RAW_BOOKS_PATH, active_out, num_users=num_users)

            flush_ram()
            flush_gpu()
            gc.collect()

        # -- 2. Échantillonnage : temporel --------------------------------
        if(target == "BOTH" or target == "TEMPORAL"):
            print("\n" + "=" * 70)
            print("  ÉTAPE 2/5 : Échantillonnage temporel")
            print("=" * 70)

            temporal_out = f"{SAMPLE_TEMPORAL_DIR}/temporal_original.parquet"
            if use_gpu:
                sample_temporal_gpu(
                    RAW_BOOKS_PATH, temporal_out, target_years=[2022],
                    num_users=num_users,
                )
            else:
                sample_temporal_cpu(
                    RAW_BOOKS_PATH, temporal_out, target_years=[2022],
                    num_users=num_users,
                )

            flush_ram()
            flush_gpu()
            gc.collect()

        # -- 3. Nettoyage -------------------------------------------------

        print("\n" + "=" * 70)
        print("  ÉTAPE 3/5 : Nettoyage des échantillons")
        print("=" * 70)

        clean_samples()
        flush_ram()

        # -- 4. Filtrage --------------------------------------------------

        print("\n" + "=" * 70)
        print("  ÉTAPE 4/5 : Filtrage par seuils d'activité")
        print("=" * 70)

        filter_samples()
        flush_ram()

        # -- 5. Split + sauvegarde ----------------------------------------

        print("\n" + "=" * 70)
        print("  ÉTAPE 5/5 : Split train/test + matrices CSR + sauvegarde")
        print("=" * 70)

        split_and_save()
        flush_ram()



    else:
        print(f"   Conversion Dataset jsonl_to_parquet_conversion a échoué  : {result}")

    elapsed = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"  Pipeline complet en {elapsed:.1f}s")
    print(f"{'=' * 70}")


def task_0():

    num_users = 5_000  # valeur par défaut

    # ── Arguments ────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="Pipeline de préparation des données — INF6083 P2"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Sous-échantillon rapide : 2 000 utilisateurs (test / développement)"
    )
    parser.add_argument(
        "--users", type=int, default=None,
        help="Nombre d'utilisateurs à échantillonner (défaut : 50 000)"
    )
    args = parser.parse_args()

    if args.quick:
        num_users = 5_000
        print(f"\n⚡ Mode --quick activé : {num_users:,} utilisateurs\n")
    elif args.users is not None:
        num_users = args.users
        print(f"\n⚡ Mode --users activé : {num_users:,} utilisateurs\n")

    t_start = time.time()

    final_files_checker = _final_files_checker()
    print(f"\n final_files_checker : {final_files_checker}\n")

    # On relance toujours si --quick ou --users est spécifié
    force_rerun = args.quick or (args.users is not None)

    if final_files_checker and not force_rerun:
        print("\n Echantillon present \n")
    else:
        task0_root = str(Path(__file__).resolve().parents[1])
        repo_root  = str(Path(__file__).resolve().parents[2])
        python_code = (
            "import sys; "
            f"sys.path.insert(0, {task0_root!r}); "
            f"sys.path.insert(0, {repo_root!r}); "
            "from scripts.precursor import *; "
            "from scripts.task_0_main import precursor; "
            f"precursor(num_users={num_users!r})"
        )
        subprocess.run([sys.executable, "-c", python_code], check=True)
        flush_ram()
        flush_gpu()
        gc.collect()

    if _joining_files_checker() and os.path.isfile("results/joining/joining_diagnostics.md"):
        cli_print_md_results()
    else:
        print()
        result = run_all(
            verbose=True,
            include_optional_raw=False,
            export_artifacts=True,
            materialize_joined=True)
        cli_print_results(result, t_start)

    build_tfidf()
    build_user_profile()


if __name__ == "__main__":
    task_0()
