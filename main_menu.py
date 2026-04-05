#!/usr/bin/env python3
"""
main_menu.py — Point d'entrée du projet INF6083 P3
Dépose ce fichier à la racine du projet (même niveau que path.py).

Lancement :
    python main_menu.py
"""

import sys
import os
import time
from pathlib import Path

# ── Racine du projet ───────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent

# Ajouter la racine et les dossiers de tâches au sys.path une seule fois
def _setup_paths():
    dirs = [
        str(ROOT),
        str(ROOT / "task_0" / "scripts"),
        str(ROOT / "task_1"),
        str(ROOT / "task_2"),
        str(ROOT / "task_3"),
    ]
    for d in dirs:
        if d not in sys.path:
            sys.path.insert(0, d)

_setup_paths()


# ══════════════════════════════════════════════════════════════════════════════
# Couleurs terminal
# ══════════════════════════════════════════════════════════════════════════════
_USE_COLOR = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()

def _c(code): return code if _USE_COLOR else ""

RESET  = _c("\033[0m")
BOLD   = _c("\033[1m")
DIM    = _c("\033[2m")
CYAN   = _c("\033[96m")
GREEN  = _c("\033[92m")
YELLOW = _c("\033[93m")
RED    = _c("\033[91m")
BLUE   = _c("\033[94m")
PURPLE = _c("\033[95m")


# ══════════════════════════════════════════════════════════════════════════════
# Helpers d'affichage
# ══════════════════════════════════════════════════════════════════════════════
def clear():
    os.system("cls" if os.name == "nt" else "clear")

def banner():
    print(f"""
{CYAN}{BOLD}╔══════════════════════════════════════════════════════════════╗
║        INF6083 — Systèmes de recommandation  •  P3          ║
║                       Équipe 2  •  Hiver 2026                ║
╚══════════════════════════════════════════════════════════════╝{RESET}
""")

def section(title, color=BLUE):
    print(f"\n{color}{BOLD}{'─' * 64}")
    print(f"  {title}")
    print(f"{'─' * 64}{RESET}\n")

def ok(msg):   print(f"  {GREEN}✓{RESET}  {msg}")
def warn(msg): print(f"  {YELLOW}⚠{RESET}  {msg}")
def err(msg):  print(f"  {RED}✗{RESET}  {msg}")
def info(msg): print(f"  {DIM}{msg}{RESET}")

def ask(prompt):
    return input(f"\n{BOLD}{YELLOW}  ▶  {prompt}{RESET} ").strip()

def pause():
    input(f"\n{DIM}  Appuie sur Entrée pour revenir au menu...{RESET}")

def confirm(msg="Lancer ?"):
    r = ask(f"{msg} [o/N]")
    return r.lower() in ("o", "oui", "y", "yes")


# ══════════════════════════════════════════════════════════════════════════════
# Vérification de l'environnement
# ══════════════════════════════════════════════════════════════════════════════
def action_check_env():
    section("Vérification de l'environnement", CYAN)

    required = [
        ("numpy",     "numpy"),
        ("pandas",    "pandas"),
        ("scipy",     "scipy"),
        ("sklearn",   "scikit-learn"),
        ("pyarrow",   "pyarrow"),
    ]
    optional = [
        ("rdflib",    "rdflib        (Tâche 2 — graphe RDF)"),
        ("owlready2", "owlready2     (Tâche 2 — inférence OWL)"),
    ]

    all_ok = True
    print(f"  {BOLD}Dépendances requises :{RESET}")
    for module, label in required:
        try:
            __import__(module)
            ok(label)
        except ImportError:
            err(f"{label}  →  pip install {module}")
            all_ok = False

    print(f"\n  {BOLD}Dépendances optionnelles :{RESET}")
    for module, label in optional:
        try:
            __import__(module)
            ok(f"{label}")
        except ImportError:
            warn(f"{label}  →  pip install {module}")

    print(f"\n  {BOLD}Fichiers de données :{RESET}")
    import path as p
    checks = [
        (p.R_TRAIN,        "R_train.npz              (matrice interactions TRAIN)"),
        (p.ITEMS,          "item_ids.npy              (catalogue items)"),
        (p.USERS,          "user_ids.npy              (liste utilisateurs)"),
        (p.USER_PROFILES,  "user_profiles_matrix.npz  (profils TF-IDF)"),
        (p.USER_HISTORIES, "user_seen_items_train.pkl (historiques train)"),
        (p.JOINING_TEST,   "test_interactions.parquet (jeu de test)"),
        (p.TEST,           "test.parquet              (splits test)"),
        (p.TRAIN,          "train.parquet             (splits train)"),
    ]
    for fpath, label in checks:
        if fpath.exists():
            ok(label)
        else:
            warn(f"{label}  {DIM}→ introuvable{RESET}")

    print()
    if all_ok:
        ok("Environnement prêt.")
    else:
        warn("Certaines dépendances sont manquantes (voir ci-dessus).")
    pause()


# ══════════════════════════════════════════════════════════════════════════════
# Tâche 0 — Précurseur (préparation des données)
# ══════════════════════════════════════════════════════════════════════════════
def action_precursor():
    section("Tâche 0 — Précurseur : préparation des données", GREEN)
    info("Conversion JSONL→Parquet, échantillonnage, nettoyage, filtrage, split train/test.")
    info("Nécessite : data/outputs/raw/parquet/Books.parquet et meta_Books.parquet\n")

    if not confirm("Lancer le précurseur ?"):
        warn("Annulé.")
        pause()
        return

    try:
        from task_0_main import precursor
        t0 = time.time()
        precursor()
        print(f"\n{GREEN}{BOLD}  Précurseur terminé en {time.time()-t0:.1f}s.{RESET}")
    except FileNotFoundError as e:
        err(f"Fichier introuvable : {e}")
        warn("Assure-toi que Books.parquet et meta_Books.parquet sont dans data/outputs/raw/parquet/")
    except Exception as e:
        err(f"Erreur : {e}")
    pause()


# ══════════════════════════════════════════════════════════════════════════════
# Tâche 0 — Filtrage basé sur le contenu (TF-IDF)
# ══════════════════════════════════════════════════════════════════════════════
def action_task0():
    section("Tâche 0 — Filtrage basé sur le contenu (TF-IDF)", GREEN)
    info("Construit les profils utilisateurs TF-IDF et génère les recommandations.")
    info("Sorties → data/outputs/task_1/\n")

    if not confirm("Lancer la Tâche 0 ?"):
        warn("Annulé.")
        pause()
        return

    try:
        from task_0_main import task_0
        t0 = time.time()
        task_0()
        print(f"\n{GREEN}{BOLD}  Tâche 0 terminée en {time.time()-t0:.1f}s.{RESET}")
    except Exception as e:
        err(f"Erreur : {e}")
    pause()


# ══════════════════════════════════════════════════════════════════════════════
# Tâche 1 — Filtrage collaboratif UBCF
# ══════════════════════════════════════════════════════════════════════════════
def action_task1():
    section("Tâche 1 — Filtrage collaboratif utilisateur (UBCF)", PURPLE)
    info("Calcule les voisins les plus proches et génère les recommandations.")
    info("Sorties → data/outputs/task_2/\n")

    if not confirm("Lancer la Tâche 1 ?"):
        warn("Annulé.")
        pause()
        return

    try:
        from task_1_main import task_1
        t0 = time.time()
        task_1()
        print(f"\n{GREEN}{BOLD}  Tâche 1 terminée en {time.time()-t0:.1f}s.{RESET}")
    except Exception as e:
        err(f"Erreur : {e}")
    pause()


# ══════════════════════════════════════════════════════════════════════════════
# Tâche 1 — Filtrage collaboratif UBCF (notre task_2_main.py)
# ══════════════════════════════════════════════════════════════════════════════
def action_ubcf():
    section("Tâche 1 — UBCF (task_2_main.py)", PURPLE)
    info("Similarité cosinus entre utilisateurs, agrégation pondérée des ratings.")
    info("Sorties → data/outputs/task_2/task_2_top_20_recommendations.csv\n")

    if not confirm("Lancer l'UBCF ?"):
        warn("Annulé.")
        pause()
        return

    try:
        from task_2_main import task_2
        t0 = time.time()
        task_2()
        print(f"\n{GREEN}{BOLD}  UBCF terminé en {time.time()-t0:.1f}s.{RESET}")
    except ImportError as e:
        err(f"Import manquant : {e}")
        warn("Vérifie que task_2_score.py est bien dans le dossier task_2/")
    except Exception as e:
        err(f"Erreur : {e}")
    pause()


# ══════════════════════════════════════════════════════════════════════════════
# Tâche 3 — Graphe de connaissances RDF/OWL
# ══════════════════════════════════════════════════════════════════════════════
def action_task3_rdf():
    section("Tâche 3 — Graphe de connaissances (RDF/OWL/SPARQL)", BLUE)
    info("Crée l'ontologie OWL, peuple les triplets RDF, applique les règles SWRL.")
    info("Sorties → data/outputs/task_3/ (ontology.owl, graph.ttl, recommandations)\n")

    # Vérifier rdflib et owlready2 avant de lancer
    missing = []
    for lib in ("rdflib", "owlready2"):
        try:
            __import__(lib)
        except ImportError:
            missing.append(lib)
    if missing:
        warn(f"Dépendances manquantes : {', '.join(missing)}")
        info(f"Installe-les avec : pip install {' '.join(missing)}")
        pause()
        return

    if not confirm("Lancer la construction du graphe ?"):
        warn("Annulé.")
        pause()
        return

    try:
        from task_3.task_3_rdf import main
        t0 = time.time()
        main()
        print(f"\n{GREEN}{BOLD}  Graphe RDF construit en {time.time()-t0:.1f}s.{RESET}")
    except Exception as e:
        err(f"Erreur : {e}")
    pause()


# ══════════════════════════════════════════════════════════════════════════════
# Tâche 3 — Évaluation comparative
# ══════════════════════════════════════════════════════════════════════════════
def action_task3_eval():
    section("Tâche 3 — Évaluation et comparaison des trois approches", BLUE)
    info("Compare Tâche 0 (contenu), Tâche 1 (UBCF) et Tâche 3 (RDF).")
    info("Sorties → data/outputs/task_3/task_3_comparison_results.csv\n")

    if not confirm("Lancer l'évaluation comparative ?"):
        warn("Annulé.")
        pause()
        return

    try:
        from task_3.task_3_evaluation import main
        t0 = time.time()
        main()
        print(f"\n{GREEN}{BOLD}  Évaluation terminée en {time.time()-t0:.1f}s.{RESET}")
    except Exception as e:
        err(f"Erreur : {e}")
    pause()


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline complet
# ══════════════════════════════════════════════════════════════════════════════
def action_pipeline_complet():
    section("Pipeline complet — Tâches 0 → UBCF → RDF → Évaluation", YELLOW)
    info("Séquence : Tâche 0 (TF-IDF) → UBCF → Graphe RDF → Évaluation comparative")
    warn("Cette séquence peut prendre plusieurs minutes.\n")

    if not confirm("Lancer le pipeline complet ?"):
        warn("Annulé.")
        pause()
        return

    etapes = [
        ("Tâche 0  — Contenu TF-IDF",           _run("task_0_main",   "task_0")),
        ("Tâche 1  — UBCF",                      _run("task_2_main",   "task_2")),
        ("Tâche 3  — Graphe RDF",                _run("task_3.task_3_rdf",    "main")),
        ("Tâche 3  — Évaluation comparative",    _run("task_3.task_3_evaluation", "main")),
    ]

    t_total = time.time()
    for i, (label, fn) in enumerate(etapes, 1):
        print(f"\n  {BOLD}[{i}/{len(etapes)}] {label}...{RESET}")
        t0 = time.time()
        try:
            fn()
            ok(f"Terminé en {time.time()-t0:.1f}s")
        except Exception as e:
            err(f"Échec : {e}")
            warn("Pipeline interrompu.")
            pause()
            return

    print(f"\n{GREEN}{BOLD}  ✓ Pipeline complet terminé en {time.time()-t_total:.1f}s.{RESET}")
    pause()


def _run(module_name, func_name):
    """Retourne une fonction qui importe et appelle module.func_name()."""
    def fn():
        import importlib
        mod = importlib.import_module(module_name)
        mod = importlib.reload(mod)
        getattr(mod, func_name)()
    return fn


# ══════════════════════════════════════════════════════════════════════════════
# Menu principal
# ══════════════════════════════════════════════════════════════════════════════
MENU = [
    # (touche, label affiché, fonction)
    ("0", f"{DIM}Vérifier l'environnement{RESET}",                              action_check_env),
    (None, None, None),  # séparateur
    ("1", f"{GREEN}Tâche 0  —  Précurseur : préparation des données{RESET}",    action_precursor),
    ("2", f"{GREEN}Tâche 0  —  Filtrage basé sur le contenu (TF-IDF){RESET}",   action_task0),
    (None, None, None),
    ("3", f"{PURPLE}Tâche 1  —  Filtrage collaboratif (task_1_main.py){RESET}", action_task1),
    ("4", f"{PURPLE}Tâche 1  —  UBCF (task_2_main.py){RESET}",                 action_ubcf),
    (None, None, None),
    ("5", f"{BLUE}Tâche 3  —  Graphe de connaissances (RDF/OWL){RESET}",       action_task3_rdf),
    ("6", f"{BLUE}Tâche 3  —  Évaluation et comparaison{RESET}",               action_task3_eval),
    (None, None, None),
    ("7", f"{YELLOW}Pipeline complet  (0 → UBCF → RDF → évaluation){RESET}",   action_pipeline_complet),
    (None, None, None),
    ("q", f"{DIM}Quitter{RESET}",                                               None),
]


def print_menu():
    banner()
    print(f"  {BOLD}Que veux-tu lancer ?{RESET}\n")
    for key, label, _ in MENU:
        if key is None:
            print()
        else:
            print(f"    {CYAN}{BOLD}[{key}]{RESET}  {label}")
    print()


def main():
    while True:
        clear()
        print_menu()

        choice = ask("Ton choix :").lower()

        if choice in ("q", "quit", "exit"):
            clear()
            print(f"\n  {DIM}Au revoir !{RESET}\n")
            sys.exit(0)

        action = None
        for key, _, fn in MENU:
            if key == choice:
                action = fn
                break

        if action is None:
            clear()
            print_menu()
            warn(f"Choix invalide : « {choice} »")
            time.sleep(1.2)
            continue

        clear()
        action()


if __name__ == "__main__":
    main()