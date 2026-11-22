"""
============================================================
MATHS COACH — Générateur d'exercices (prépa ECG)
============================================================

Application web mono-fichier (backend Flask + API Gemini + frontend
intégré) qui génère des exercices de mathématiques sur le chapitre
de ton choix, avec une difficulté de 1 à 5 étoiles.

Installation :

    pip install flask google-genai

Clé API (gratuite sur https://aistudio.google.com/apikey), au choix :

    - variable d'environnement :  GEMINI_API_KEY=...
    - ou saisie dans l'interface (stockée uniquement dans ton navigateur)

Lancement :

    python maths_coach.py       puis ouvrir http://127.0.0.1:5001

Fonctionnement :

    1. On saisit le chapitre étudié et on règle la difficulté.
         1 étoile  : exercice de TD accessible, découverte des outils
         5 étoiles : problème de recherche en plusieurs parties
    2. L'énoncé s'affiche au centre de la page (formules en LaTeX
       rendues par KaTeX, chargé depuis un CDN : connexion requise).
    3. "Nouvel exercice" en génère un autre. La difficulté est un
       simple paramètre, modifiable à tout moment sans question.

Réserve d'exercices :

    Tout exercice généré est stocké dans une base SQLite
    (maths_coach.db, créée à côté du script). "Nouvel exercice"
    sert d'abord un exercice jamais vu de la réserve (instantané),
    et n'appelle Gemini que s'il n'y en a plus. En arrière-plan,
    l'application prépare 2 exercices d'avance pour le chapitre et
    la difficulté en cours, afin que le suivant soit déjà prêt.

Les énoncés sont écrits par une IA : une erreur d'énoncé reste
possible, surtout en difficulté 4-5. Les indications de correction
(repliées par défaut) aident à repérer un exercice mal posé.

Variable optionnelle : GEMINI_MODEL (par défaut gemini-3.8-flash).
============================================================
"""

import json
import os
import random
import re
import sqlite3
import threading
import time
import traceback
import unicodedata
import webbrowser
from contextlib import closing
from datetime import datetime

from flask import Flask, jsonify, request, Response
from google import genai
from google.genai import types


# ============================================================
# CONFIGURATION
# ============================================================

PRIMARY_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
FALLBACK_MODELS = ["gemini-flash-latest", "gemini-3.7-flash", "gemini-3.6-flash", "gemini-3.5-flash"]

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "maths_coach.db")
PREFETCH_TARGET = 2  # exercices non vus gardés d'avance par (chapitre, difficulté)

CHAPTERS = {
    "Première année": [
        "Logique et raisonnement",
        "Récurrence, sommes et produits",
        "Ensembles et applications",
        "Dénombrement",
        "Suites numériques",
        "Limites et continuité",
        "Dérivation et convexité",
        "Intégration sur un segment",
        "Séries numériques",
        "Systèmes linéaires et matrices",
        "Espaces vectoriels et applications linéaires",
        "Probabilités discrètes",
        "Variables aléatoires discrètes",
    ],
    "Deuxième année": [
        "Compléments d'algèbre linéaire",
        "Réduction des matrices (éléments propres)",
        "Algèbre bilinéaire et produit scalaire",
        "Intégrales généralisées",
        "Fonctions de deux variables",
        "Variables aléatoires à densité",
        "Convergences et approximations",
        "Estimation",
        "Informatique : Python",
    ],
}

DIFFICULTY = {
    1: {
        "label": "Découverte",
        "short": "Exercice de TD très direct : une consigne, trois petits calculs du cours.",
        "spec": "Exercice de TD très court et direct, exactement comme dans une feuille de TD de prépa : UNE consigne d'une ligne (« Étudier la parité des fonctions suivantes », « Déterminer le degré de... », « Factoriser le polynôme... », « Résoudre l'équation... », « Simplifier... ») suivie de 2 à 4 sous-questions a), b), c) indépendantes et du même type, chacune étant une application directe d'une formule ou d'une méthode du cours. Aucun contexte, aucun récit, aucune modélisation, aucune idée à trouver, calculs simples. Tout tient dans une seule question @@Q (deux au maximum) ; laisse @@CONTEXTE vide.",
    },
    2: {
        "label": "Application",
        "short": "Exercice de TD classique : méthodes incontournables, courtes démonstrations.",
        "spec": "Exercice de TD classique et court, exactement comme dans une feuille de TD de prépa : 1 à 3 questions, soit une consigne avec sous-questions a), b), c) un peu plus techniques, soit une petite démonstration guidée en 2 ou 3 étapes (« Montrer que... », « En déduire que... »). Il travaille les méthodes incontournables du chapitre, celles qu'il faut savoir refaire sans hésiter. Aucun contexte, aucun récit, aucune modélisation : uniquement des objets mathématiques donnés directement par leurs formules. Énoncé bref ; laisse @@CONTEXTE vide.",
    },
    3: {
        "label": "Question difficile",
        "short": "Question difficile de TD : énoncé court mais qui demande une vraie idée.",
        "spec": "Question difficile de TD, comme les derniers exercices d'une feuille de TD de prépa : énoncé court et sec (2 à 5 lignes) mais qui demande une vraie idée ou une méthode non immédiate (démontrer une propriété surprenante, déterminer tous les objets vérifiant une condition, montrer une divisibilité, une inégalité ou une identité qui ne se voit pas tout de suite). 1 à 3 questions, peu ou pas de guidage ; une brève indication est possible. Ce n'est PAS un problème : pas de contexte concret, pas de parties A/B, pas de long enchaînement de questions.",
    },
    4: {
        "label": "Problème",
        "short": "Problème de concours en 1 ou 2 parties, questions enchaînées.",
        "spec": "Problème de type écrit de concours (EDHEC, ECRICOME) en 1 ou 2 parties : 5 à 8 questions enchaînées combinant plusieurs notions du chapitre (voire d'un chapitre voisin), avec des résultats intermédiaires fournis (« en déduire que... »), questions moins guidées, au moins une idée à trouver et une question finale difficile. Notations et définitions posées dans @@CONTEXTE ; une situation concrète ou une modélisation est bienvenue lorsque le chapitre s'y prête.",
    },
    5: {
        "label": "Recherche",
        "short": "Problème de recherche en plusieurs parties, peu guidé.",
        "spec": "Problème de recherche d'environ 1h30 à 2h. 8 à 12 questions réparties en 2 ou 3 parties (Partie A, B, C) qui construisent progressivement un résultat non trivial, éventuellement à partir d'une situation concrète ou d'un modèle. Questions ouvertes (conjecturer, étudier, déterminer tous les..., montrer qu'il existe...), peu de guidage ; la dernière partie demande synthèse et initiative.",
    },
}

# Exemples de style (tirés de vraies feuilles de TD) glissés dans le prompt
EXEMPLES = {
    1: r"""- Étudier la parité des fonctions suivantes définies sur $\mathbb{R}$ : a) $f(x)=e^x-e^{-x}$ b) $g(x)=\dfrac{e^{2x}-1}{e^{2x}+1}$ c) $h(x)=\dfrac{e^x}{(e^x+1)^2}$
- Soit $P=3X^4-3X+1$ et $Q=-3X^4+X^3+2X^2$. Déterminer le degré de $P+Q$, de $P\circ Q$, de $PQ$ et de $P^2-Q^2$.
- a) Factoriser $P=X^4+X^3-17X^2+15X$. b) Factoriser $Q=3X^3-X-2$.
- Résoudre l'équation $e^x=1+6e^{-x}$ d'inconnue $x\in\mathbb{R}$.""",
    2: r"""- Soit $f(x)=\ln\left(x+\sqrt{1+x^2}\right)$. a) Montrer que pour tout $x\in\mathbb{R}$, $x+\sqrt{1+x^2}>0$. En déduire que $f$ est bien définie sur $\mathbb{R}$. b) Montrer que $f$ est impaire.
- Calculer la dérivée de $x\mapsto\arctan x+\arctan\dfrac1x$. Que peut-on en déduire ?
- a) Déterminer le quotient et le reste de la division euclidienne de $X^5-X^2+2$ par $X^2+1$.
- Démontrer que : $\forall x\in\mathbb{R}_+,\ x-\dfrac{x^2}{2}\leqslant\ln(1+x)\leqslant x$.""",
    3: r"""- Soit $f:\mathbb{R}\to\mathbb{R}$ une fonction croissante et périodique de période $T>0$. Montrer que $f$ est constante.
- Déterminer tous les polynômes $P\in\mathbb{R}[X]$ tels que $P(X^2)=(X^2+1)P$.
- Soit $P\in\mathbb{R}[X]$ tel que pour tout $x\in\mathbb{R}$, $P(x)=P(\sin x)$. Montrer que $P$ est constant.
- Soit $n\geqslant 2$. On pose $Q=X^2-3X+2$ et $P=X^n-2X^{n-1}-X+2$. Montrer que $Q$ divise $P$.""",
}


# ============================================================
# OUTILS GEMINI
# ============================================================

class UserError(Exception):
    """Erreur dont le message peut être montré tel quel à l'utilisateur."""


def friendly(ex):
    traceback.print_exc()  # détail complet dans le terminal pour le débogage
    if isinstance(ex, UserError):
        return str(ex)
    m = str(ex)
    if "API_KEY_INVALID" in m or "API key not valid" in m or "PERMISSION_DENIED" in m:
        return "Clé API refusée par Google. Vérifie qu'elle est correcte et active."
    if "429" in m or "RESOURCE_EXHAUSTED" in m:
        return "Quota Gemini atteint. Attends quelques minutes ou utilise une autre clé."
    if "503" in m or "UNAVAILABLE" in m or "overloaded" in m.lower():
        return "Le service Gemini est surchargé. Réessaie dans un instant."
    if "timed out" in m.lower() or "timeout" in m.lower() or "deadline" in m.lower():
        return "Gemini met trop de temps à répondre. Réessaie dans un instant."
    return "Erreur inattendue : " + m[:240]


def new_client(key, timeout=120):
    """Client Gemini avec un délai maximum (en secondes) pour ne jamais rester bloqué."""
    return genai.Client(api_key=key, http_options=types.HttpOptions(timeout=timeout * 1000))


def get_key():
    key = (
        request.headers.get("X-Api-Key")
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
    )
    if not key:
        raise UserError("Aucune clé API. Ajoute-la dans « Clé API Gemini » sur la page d'accueil.")
    return key.strip()


# Réglages de vitesse : c'est surtout la "réflexion" interne du modèle qui rend
# la génération lente (30 à 90 s). On la limite selon la difficulté.
THINKING = {1: ("minimal", 0), 2: ("minimal", 0), 3: ("low", 1024), 4: ("low", 2048), 5: ("medium", 4096)}
ATTEMPT_TIMEOUT = 70   # secondes max pour UN appel (au-delà, il est considéré comme bloqué)
GEN_DEADLINE = 120     # secondes max pour l'ensemble des tentatives d'un exercice


def thinking_cfg(diff):
    """Réglage de réflexion compatible avec la version du SDK / du modèle (ou None)."""
    if diff not in THINKING:
        return None
    level, budget = THINKING[diff]
    try:
        return types.ThinkingConfig(thinking_level=level.upper())
    except Exception:  # noqa: BLE001
        pass
    try:
        return types.ThinkingConfig(thinking_budget=budget)
    except Exception:  # noqa: BLE001
        return None


def gen(client, contents, system=None, temperature=0.8, diff=None):
    t_end = time.monotonic() + GEN_DEADLINE
    think = thinking_cfg(diff)
    errors = []
    for model in [PRIMARY_MODEL] + FALLBACK_MODELS:
        for attempt in range(3):
            remaining = t_end - time.monotonic()
            if remaining < 8:
                break
            cfg = {
                "temperature": temperature,
                "http_options": types.HttpOptions(timeout=int(min(remaining, ATTEMPT_TIMEOUT) * 1000)),
            }
            if system:
                cfg["system_instruction"] = system
            if think is not None:
                cfg["thinking_config"] = think
            try:
                return client.models.generate_content(
                    model=model, contents=contents,
                    config=types.GenerateContentConfig(**cfg),
                )
            except Exception as ex:  # noqa: BLE001
                m = str(ex)
                low = m.lower()
                if "API_KEY_INVALID" in m or "API key not valid" in m:
                    raise
                if think is not None and "thinking" in low:
                    think = None  # réglage refusé par ce modèle : on réessaie sans
                    errors.append((False, ex))
                    continue
                if "404" in m or "NOT_FOUND" in m:
                    errors.append((True, ex))
                    break  # modèle indisponible : on passe au suivant
                if "429" in m or "RESOURCE_EXHAUSTED" in m:
                    errors.append((False, ex))
                    break  # quota de ce modèle atteint : on essaie un autre modèle
                if "503" in m or "UNAVAILABLE" in m or "overloaded" in low:
                    errors.append((False, ex))
                    time.sleep(min(2 * (attempt + 1), max(0, t_end - time.monotonic() - 8)))
                    continue  # surcharge temporaire : on réessaie
                if "timed out" in low or "timeout" in low or "deadline" in low or "504" in m:
                    errors.append((False, ex))
                    continue  # appel bloqué : on le relance au lieu d'attendre
                raise
    real = [e for is_404, e in errors if not is_404]
    if real or errors:
        raise (real or [e for _, e in errors])[0]
    raise UserError("Gemini met trop de temps à répondre. Réessaie dans un instant.")


# ============================================================
# GÉNÉRATION ET LECTURE DES EXERCICES
# ============================================================

SYSTEM_PROF = """Tu es professeur de mathématiques en classe préparatoire ECG (voie générale) et auteur de sujets de concours (EDHEC, ECRICOME, ESSEC, HEC, emlyon).
Tu écris des énoncés rigoureux, autonomes et sans erreur, strictement dans le programme officiel de mathématiques ECG : n'utilise aucune notion ni aucun théorème hors programme. En particulier, AUCUN nombre complexe (ni i, ni module, ni argument, ni exponentielle complexe) : tout se passe dans R, et les polynômes sont à coefficients réels. Privilégie les questions types qui aident à maîtriser le chapitre. Ton style est celui des feuilles de TD de prépa : consignes brèves et directes (« Déterminer... », « Montrer que... », « Factoriser... »), sans narration ni mise en situation, sauf pour les problèmes de difficulté 4 et 5.
Tu soignes les notations : tout objet est défini avant d'être utilisé, les hypothèses sont explicites (ensembles de définition, valeurs de n, conditions sur les paramètres)."""


ANGLES_TECH = [
    "application directe d'une formule du cours",
    "calcul ou simplification technique",
    "démonstration classique courte",
    "méthode type du chapitre sur un cas précis",
    "vérification d'une propriété à partir de la définition",
]
ANGLES_PROBLEM = [
    "situation concrète modélisée (économie, jeu, tirages, file d'attente...)",
    "conjecture puis preuve",
    "problème combinant deux notions du chapitre",
    "étude complète d'un objet avec résultat final non trivial",
]


def build_prompt(chapter, diff, avoid_titles):
    d = DIFFICULTY[diff]
    avoid = ""
    if avoid_titles:
        avoid = "\nÉvite de reprendre les thèmes de ces exercices déjà générés : " + " ; ".join(avoid_titles) + "."
    python_hint = (
        "\nSi le chapitre s'y prête, une question peut demander d'écrire ou de compléter un court script Python (bloc de code écrit tel quel dans l'énoncé)."
        if diff >= 4 else ""
    )
    ex = EXEMPLES.get(diff, "")
    examples = (
        "\n\nEXEMPLES DU STYLE ATTENDU, tirés de vraies feuilles de TD (ne les recopie pas : invente un exercice nouveau "
        "sur le chapitre demandé, dans le même style, avec la même longueur et la même difficulté) :\n" + ex
    ) if ex else ""
    return f"""Génère un exercice original de mathématiques sur le chapitre : « {chapter} ».
Niveau de difficulté : {diff}/5 ({d['label']}).
{d['spec']}{python_hint}{examples}{avoid}
Angle suggéré pour varier : {random.choice(ANGLES_PROBLEM if diff >= 4 else ANGLES_TECH)}.

Vérifie rapidement que l'exercice est correct, bien posé, et que tout résultat annoncé dans l'énoncé est vrai. Sois efficace : n'écris rien en dehors du format demandé.

FORMAT DE SORTIE STRICT : texte brut, pas de JSON, pas de bloc de code, pas de Markdown (pas de **, pas de #, pas de puces). Chaque marqueur @@ est seul sur sa ligne.
@@TITRE
titre court de l'exercice
@@CONTEXTE
définitions, notations et hypothèses communes (laisse vide si inutile)
@@PARTIE titre de la partie (uniquement s'il y a plusieurs parties, par exemple : Partie A : étude d'une fonction)
@@INTRO
définitions propres à cette partie (facultatif)
@@Q
énoncé de la première question
@@Q
énoncé de la deuxième question
(autant de @@Q que nécessaire ; ne numérote pas les questions, c'est l'interface qui s'en charge ; les sous-questions a), b) restent à l'intérieur du texte de la question)
@@CORRECTION
@@C
indication de correction de la question 1 : idée clé et résultat final, 1 à 4 lignes
@@C
indication de correction de la question 2
(exactement une section @@C par question, dans le même ordre)

FORMULES : écris les mathématiques en LaTeX avec $...$ pour les formules dans le texte et $$...$$ pour les formules mises en valeur. Utilise uniquement des commandes compatibles KaTeX. Pas de \\begin{{align}} ni de \\begin{{enumerate}} : pour aligner des lignes, utilise \\begin{{aligned}} à l'intérieur de $$...$$. Matrices : \\begin{{pmatrix}}. Ensembles : \\mathbb{{R}}, \\mathbb{{N}}. Intervalles d'entiers : \\llbracket 1, n \\rrbracket.
Le texte est en français."""


def parse_exercise(raw):
    raw = re.sub(r"^```\w*\s*$", "", (raw or "").strip(), flags=re.M)
    sections, cur = [], None
    for line in raw.splitlines():
        m = re.match(r"^\s*@@(\w+)\s*(.*)$", line)
        if m:
            cur = [m.group(1).upper(), [m.group(2)] if m.group(2).strip() else []]
            sections.append(cur)
        elif cur is not None:
            cur[1].append(line)

    def clean(lines):
        return re.sub(r"\n{3,}", "\n\n", "\n".join(lines).strip())

    titre, contexte, corrections = "", "", []
    parties = [{"titre": "", "intro": "", "questions": []}]
    for kind, lines in sections:
        txt = clean(lines)
        if kind == "TITRE":
            titre = txt
        elif kind == "CONTEXTE":
            contexte = txt
        elif kind == "PARTIE":
            parties.append({"titre": txt, "intro": "", "questions": []})
        elif kind == "INTRO":
            parties[-1]["intro"] = txt
        elif kind == "Q" and txt:
            parties[-1]["questions"].append(txt)
        elif kind == "C" and txt:
            corrections.append(txt)
    parties = [p for p in parties if p["questions"]]
    if not parties:
        raise UserError("L'exercice généré est illisible. Réessaie.")
    return {"titre": titre or "Exercice", "contexte": contexte, "parties": parties, "corrections": corrections}


def generate_exercise(client, chapter, diff, ckey):
    prompt = build_prompt(chapter, diff, recent_titles(ckey, diff))
    last = None
    for _ in range(2):
        try:
            return parse_exercise(gen(client, prompt, system=SYSTEM_PROF, diff=diff).text)
        except UserError as ex:
            last = ex
    raise last


# ============================================================
# RÉSERVE (SQLite)
# ============================================================

def norm_key(chapter):
    s = unicodedata.normalize("NFD", chapter.lower())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s)).strip()[:80]


def db():
    con = sqlite3.connect(DB_PATH, timeout=10)
    con.row_factory = sqlite3.Row
    return con


def init_db():
    with closing(db()) as con, con:
        con.execute("""CREATE TABLE IF NOT EXISTS exercises (
            id INTEGER PRIMARY KEY AUTOINCREMENT, ckey TEXT NOT NULL, chapter TEXT NOT NULL,
            difficulty INTEGER NOT NULL, created TEXT NOT NULL, seen INTEGER NOT NULL DEFAULT 0, data TEXT NOT NULL)""")
        con.execute("CREATE INDEX IF NOT EXISTS idx_ex ON exercises (ckey, difficulty, seen)")


def save_exercise(ckey, chapter, diff, data, seen):
    with closing(db()) as con, con:
        con.execute("INSERT INTO exercises (ckey, chapter, difficulty, created, seen, data) VALUES (?,?,?,?,?,?)",
                    (ckey, chapter, diff, datetime.now().isoformat(timespec="seconds"), int(seen),
                     json.dumps(data, ensure_ascii=False)))


def count_unseen(ckey, diff):
    with closing(db()) as con:
        return con.execute("SELECT COUNT(*) FROM exercises WHERE ckey=? AND difficulty=? AND seen=0",
                           (ckey, diff)).fetchone()[0]


def take_unseen(ckey, diff):
    """Renvoie un exercice jamais vu de la réserve (et le marque comme vu), ou None."""
    with closing(db()) as con, con:
        rows = con.execute("SELECT id, data FROM exercises WHERE ckey=? AND difficulty=? AND seen=0",
                           (ckey, diff)).fetchall()
        if not rows:
            return None
        row = random.choice(rows)
        con.execute("UPDATE exercises SET seen=1 WHERE id=?", (row["id"],))
        return json.loads(row["data"])


def recent_titles(ckey, diff, limit=10):
    with closing(db()) as con:
        rows = con.execute("SELECT data FROM exercises WHERE ckey=? AND difficulty=? ORDER BY id DESC LIMIT ?",
                           (ckey, diff, limit)).fetchall()
    return [json.loads(r["data"]).get("titre", "") for r in rows]


# préparation d'exercices d'avance en arrière-plan (générés EN PARALLÈLE)
INFLIGHT = {}  # (ckey, difficulté) -> nombre de générations en cours
INFLIGHT_LOCK = threading.Lock()


def _prefetch_one(key, chapter, diff, ckey):
    try:
        client = new_client(key, timeout=ATTEMPT_TIMEOUT)
        save_exercise(ckey, chapter, diff, generate_exercise(client, chapter, diff, ckey), seen=False)
    except Exception:  # noqa: BLE001
        traceback.print_exc()  # une réserve qui échoue ne doit jamais gêner l'utilisateur
    finally:
        with INFLIGHT_LOCK:
            INFLIGHT[(ckey, diff)] = max(0, INFLIGHT.get((ckey, diff), 1) - 1)


def prefetch(key, chapter, diff, ckey):
    token = (ckey, diff)
    with INFLIGHT_LOCK:
        need = PREFETCH_TARGET - count_unseen(ckey, diff) - INFLIGHT.get(token, 0)
        if need <= 0:
            return
        INFLIGHT[token] = INFLIGHT.get(token, 0) + need
    for _ in range(need):
        threading.Thread(target=_prefetch_one, args=(key, chapter, diff, ckey), daemon=True).start()


def wait_for_inflight(ckey, diff, timeout=90):
    """Si un exercice est déjà en cours de préparation, on l'attend au lieu d'en lancer un second."""
    t_end = time.monotonic() + timeout
    while time.monotonic() < t_end:
        with INFLIGHT_LOCK:
            busy = INFLIGHT.get((ckey, diff), 0) > 0
        data = take_unseen(ckey, diff)
        if data is not None:
            return data
        if not busy:
            return None
        time.sleep(0.4)
    return None


# ============================================================
# API
# ============================================================

app = Flask(__name__)


def _params(src):
    chapter = re.sub(r"\s+", " ", str(src.get("chapter", ""))).strip()[:100]
    try:
        diff = int(src.get("difficulty", 3))
    except (TypeError, ValueError):
        diff = 3
    if not chapter:
        raise UserError("Indique un chapitre.")
    return chapter, max(1, min(5, diff))


@app.get("/api/config")
def api_config():
    return jsonify({
        "has_env_key": bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")),
        "chapters": CHAPTERS,
        "difficulty": {str(k): {"label": v["label"], "short": v["short"]} for k, v in DIFFICULTY.items()},
    })


@app.post("/api/exercise")
def api_exercise():
    try:
        chapter, diff = _params(request.get_json(force=True))
        key = get_key()
        ckey = norm_key(chapter)
        data = take_unseen(ckey, diff) or wait_for_inflight(ckey, diff)
        from_stock = data is not None
        if data is None:
            data = generate_exercise(new_client(key, timeout=ATTEMPT_TIMEOUT), chapter, diff, ckey)
            save_exercise(ckey, chapter, diff, data, seen=True)
        remaining = count_unseen(ckey, diff)
        if remaining < PREFETCH_TARGET:
            prefetch(key, chapter, diff, ckey)
        return jsonify({"ok": True, "exercise": data, "from_stock": from_stock, "remaining": remaining})
    except Exception as ex:  # noqa: BLE001
        return jsonify({"ok": False, "message": friendly(ex)}), 400


@app.get("/api/stock")
def api_stock():
    try:
        chapter, diff = _params(request.args)
        return jsonify({"ok": True, "remaining": count_unseen(norm_key(chapter), diff)})
    except Exception:  # noqa: BLE001
        return jsonify({"ok": True, "remaining": 0})


@app.get("/")
def index():
    return Response(INDEX_HTML, mimetype="text/html")


# ============================================================
# FRONTEND
# ============================================================

INDEX_HTML = r"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Maths Coach</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;500;600&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.css">
<style>
:root{
  --bg:#FFFFFF; --surface:#FFFFFF; --ink:#000000; --muted:#4B5563; --line:#D1D5DB;
  --accent:#1D3BC4; --on-accent:#FFFFFF; --pen:#C62828; --r:6px;
}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 "Instrument Sans",system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:740px;min-height:100vh;margin:0 auto;padding:40px 24px;display:flex;flex-direction:column;justify-content:center}
h1{font-size:30px;line-height:1.15;margin:0 0 10px;font-weight:600;letter-spacing:-.01em}
h3{font-size:16px;margin:30px 0 8px;font-weight:600}
p{margin:0 0 12px}
.muted{color:var(--muted)}
.small{font-size:14px}
[hidden]{display:none!important}
button{font:inherit;color:inherit;cursor:pointer}
:focus-visible{outline:2px solid var(--accent);outline-offset:2px}

.field{margin:26px 0 0}
.field>label,.field>.lab{display:block;font-weight:500;margin-bottom:8px}
input[type=text],input[type=password]{width:100%;font:inherit;color:inherit;background:var(--surface);border:1px solid var(--line);border-radius:var(--r);padding:10px 12px}
.chips{margin-top:12px}
.chips .grp{font-size:14px;color:var(--muted);margin:12px 0 6px}
.chips .row{display:flex;flex-wrap:wrap;gap:6px 8px}
.chips button{border:1px solid var(--line);background:var(--surface);border-radius:999px;padding:4px 12px;font-size:14px;color:var(--muted)}
.chips button:hover{color:var(--ink);border-color:var(--muted)}
.chips button[aria-pressed=true]{color:var(--on-accent);background:var(--accent);border-color:var(--accent)}
.stars{display:inline-flex;gap:2px;vertical-align:middle}
.stars button{border:0;background:none;font-size:28px;line-height:1;padding:0 2px;color:color-mix(in srgb,var(--muted) 38%,transparent)}
.stars button[aria-pressed=true]{color:var(--accent)}
.stars.sm button{font-size:20px}
.dlabel{margin-left:12px;color:var(--muted);font-size:15px}
details{margin-top:22px}
summary{cursor:pointer;color:var(--muted)}
.btn{border:1px solid var(--accent);background:var(--accent);color:var(--on-accent);border-radius:var(--r);padding:11px 22px;font-weight:500}
.btn:disabled{opacity:.55;cursor:default}
.btn.ghost{background:none;color:var(--accent)}
.err{color:var(--pen);margin-top:14px}
.actions{margin-top:30px;display:flex;gap:12px;flex-wrap:wrap;align-items:center}

.ex .where{color:var(--muted);font-size:14px;margin-bottom:6px}
.ex.busy .body{opacity:.45}
.ex .ctx,.ex .intro{margin:14px 0 4px}
.ex .q{display:flex;gap:12px;margin:16px 0}
.ex .n{flex:none;min-width:1.6em;font-weight:600;color:var(--accent)}
.qt,.ctx,.intro{white-space:pre-line;font-size:17px;line-height:1.7}
.katex{font-size:1.08em}
.katex-display{margin:.9em 0;overflow-x:auto;overflow-y:hidden}
.controls{margin-top:36px;padding-top:20px;border-top:1px solid var(--line);display:flex;flex-wrap:wrap;gap:14px 24px;align-items:center;justify-content:space-between}
.controls .left{display:flex;align-items:center;gap:8px;flex-wrap:wrap}
.controls .right{display:flex;align-items:center;gap:14px}
.linkbtn{border:0;background:none;color:var(--muted);text-decoration:underline;text-underline-offset:3px;padding:0}
.corr summary{color:var(--accent)}
.corr .q{margin:10px 0}
.corr .qt{font-size:16px}
@media (max-width:560px){h1{font-size:26px}.controls{justify-content:flex-start}}
</style>
</head>
<body>
<main>

<!-- ACCUEIL -->
<section id="s-setup">
  <h1>Exercices de maths</h1>
  <p class="muted">Indique le chapitre que tu travailles et le niveau voulu. L'énoncé s'affiche, et tu peux en demander un nouveau autant de fois que tu veux.</p>

  <div class="field"><label for="chapter">Chapitre</label>
    <input type="text" id="chapter" placeholder="Par exemple : suites numériques" maxlength="100" autocomplete="off">
    <div class="chips" id="chips"></div>
  </div>

  <div class="field"><span class="lab">Difficulté</span>
    <div class="stars" id="stars-setup" role="group" aria-label="Difficulté"></div><span class="dlabel" id="dlabel-setup"></span>
  </div>

  <details id="key-box"><summary>Clé API Gemini</summary>
    <div style="margin-top:10px"><input type="password" id="key" placeholder="Colle ta clé ici" autocomplete="off">
    <p class="small muted" style="margin-top:6px">Gratuite sur aistudio.google.com/apikey. Elle reste dans ton navigateur et n'est envoyée qu'à ton serveur local.</p></div>
  </details>

  <div class="actions"><button class="btn" id="go">Générer un exercice</button><span id="setup-err" class="err" hidden></span></div>
</section>

<!-- CHARGEMENT -->
<section id="s-load" hidden>
  <h1>Génération de l'exercice</h1>
  <p class="muted" id="load-msg">Compte 5 à 30 secondes selon la difficulté.</p>
  <p class="muted small" id="load-clock">0 s</p>
  <div class="actions"><button class="btn ghost" id="load-cancel">Annuler</button></div>
</section>

<!-- EXERCICE -->
<section id="s-ex" class="ex" hidden>
  <div class="where" id="where"></div>
  <div class="body" id="body"></div>
  <div class="controls">
    <div class="left"><div class="stars sm" id="stars-ex" role="group" aria-label="Difficulté"></div><span class="dlabel" id="dlabel-ex" style="margin-left:4px"></span></div>
    <div class="right"><button class="linkbtn" id="change">Changer de chapitre</button>
      <button class="btn" id="next">Nouvel exercice</button></div>
  </div>
  <div class="small muted" id="stock" style="margin-top:10px;text-align:right"></div>
  <div class="err" id="ex-err" hidden></div>
</section>

</main>

<script src="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/contrib/auto-render.min.js"></script>
<script>
(() => {
  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
  const S = { chapter: "", diff: 3, cfg: null, abort: null, poll: null, busy: false };
  const KEY = "mc-key", PREF = "mc-prefs";

  const getKey = () => { try { return localStorage.getItem(KEY) || ""; } catch (e) { return ""; } };
  function loadPrefs() {
    try { const p = JSON.parse(localStorage.getItem(PREF) || "{}"); if (p.chapter) S.chapter = p.chapter; if (p.diff) S.diff = p.diff; } catch (e) {}
  }
  function savePrefs() { try { localStorage.setItem(PREF, JSON.stringify({ chapter: S.chapter, diff: S.diff })); } catch (e) {} }
  function show(id) { ["s-setup", "s-load", "s-ex"].forEach((s) => ($(s).hidden = s !== id)); window.scrollTo(0, 0); }

  function typeset(el) {
    if (typeof renderMathInElement !== "function") return;
    renderMathInElement(el, {
      delimiters: [{ left: "$$", right: "$$", display: true }, { left: "\\[", right: "\\]", display: true },
                   { left: "$", right: "$", display: false }, { left: "\\(", right: "\\)", display: false }],
      throwOnError: false,
    });
  }

  // ---------- difficulté (paramètre persistant) ----------
  function drawStars(el, labelEl) {
    el.innerHTML = [1, 2, 3, 4, 5].map((i) => `<button type="button" data-i="${i}" aria-label="${i} étoile${i > 1 ? "s" : ""}" aria-pressed="${i <= S.diff}">★</button>`).join("");
    const d = S.cfg.difficulty[S.diff];
    labelEl.textContent = d.label;
    labelEl.title = d.short;
  }
  function syncStars() { drawStars($("stars-setup"), $("dlabel-setup")); drawStars($("stars-ex"), $("dlabel-ex")); $("dlabel-setup").textContent = S.cfg.difficulty[S.diff].label + " : " + S.cfg.difficulty[S.diff].short; }
  function onStars(e) {
    const b = e.target.closest("button"); if (!b) return;
    S.diff = +b.dataset.i; savePrefs(); syncStars(); refreshStock();
  }
  $("stars-setup").addEventListener("click", onStars);
  $("stars-ex").addEventListener("click", onStars);

  // ---------- accueil ----------
  function buildSetup() {
    $("chips").innerHTML = Object.entries(S.cfg.chapters).map(([g, list]) =>
      `<div class="grp">${esc(g)}</div><div class="row">${list.map((c) => `<button type="button" aria-pressed="false">${esc(c)}</button>`).join("")}</div>`).join("");
    $("chips").onclick = (e) => { const b = e.target.closest("button"); if (!b) return; $("chapter").value = b.textContent; markChip(); };
    $("chapter").addEventListener("input", markChip);
    $("chapter").value = S.chapter; markChip();
    $("key").value = getKey();
    if (S.cfg.has_env_key) $("key-box").hidden = true; else if (!getKey()) $("key-box").open = true;
    syncStars();
  }
  function markChip() {
    const v = $("chapter").value.trim();
    $("chips").querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", b.textContent === v));
  }
  $("key").addEventListener("change", () => { try { localStorage.setItem(KEY, $("key").value.trim()); } catch (e) {} });

  // ---------- appel API ----------
  async function requestExercise(signal) {
    const res = await fetch("/api/exercise", {
      method: "POST", signal,
      headers: Object.assign({ "Content-Type": "application/json" }, getKey() ? { "X-Api-Key": getKey() } : {}),
      body: JSON.stringify({ chapter: S.chapter, difficulty: S.diff }),
    });
    const data = await res.json().catch(() => ({ ok: false, message: "Réponse du serveur illisible." }));
    if (!data.ok) throw new Error(data.message || "Erreur.");
    return data;
  }

  $("go").addEventListener("click", async () => {
    const err = $("setup-err"); err.hidden = true;
    S.chapter = $("chapter").value.trim();
    if (!S.chapter) { err.textContent = "Indique un chapitre."; err.hidden = false; return; }
    try { localStorage.setItem(KEY, $("key").value.trim()); } catch (e) {}
    savePrefs();
    show("s-load");
    const ctl = new AbortController(); S.abort = ctl;
    const t0 = Date.now(), clk = setInterval(() => { $("load-clock").textContent = Math.round((Date.now() - t0) / 1000) + " s"; }, 500);
    try {
      const data = await requestExercise(ctl.signal);
      clearInterval(clk);
      renderExercise(data); show("s-ex"); typeset($("body")); startPoll();
    } catch (e) {
      clearInterval(clk); show("s-setup");
      if (e.name !== "AbortError") { err.textContent = e.message; err.hidden = false; }
    }
  });
  $("load-cancel").addEventListener("click", () => { if (S.abort) S.abort.abort(); });

  // ---------- exercice ----------
  function renderExercise(data) {
    const d = data.exercise;
    let n = 0;
    const parts = d.parties.map((p) =>
      (p.titre ? `<h3>${esc(p.titre)}</h3>` : "") +
      (p.intro ? `<div class="intro">${esc(p.intro)}</div>` : "") +
      p.questions.map((q) => { n++; return `<div class="q"><span class="n">${n}.</span><div class="qt">${esc(q)}</div></div>`; }).join("")
    ).join("");
    const corr = (d.corrections || []).length
      ? `<details class="corr"><summary>Indications de correction</summary>
           ${d.corrections.map((c, i) => `<div class="q"><span class="n">${i + 1}.</span><div class="qt">${esc(c)}</div></div>`).join("")}
           <p class="small muted">Rédigées par une IA : vérifie-les, elles peuvent contenir des erreurs.</p></details>` : "";
    $("where").textContent = S.chapter;
    $("body").innerHTML = `<h1>${esc(d.titre)}</h1>${d.contexte ? `<div class="ctx">${esc(d.contexte)}</div>` : ""}${parts}${corr}`;
    $("ex-err").hidden = true;
    setStock(data.remaining, data.from_stock);
  }
  function setStock(n, fromStock) {
    $("stock").textContent = n > 0 ? `${n} exercice${n > 1 ? "s" : ""} déjà prêt${n > 1 ? "s" : ""}` : (fromStock ? "" : "");
  }

  $("next").addEventListener("click", async () => {
    if (S.busy) return;
    S.busy = true; $("next").disabled = true; $("next").textContent = "Génération en cours";
    $("s-ex").classList.add("busy"); $("ex-err").hidden = true;
    try {
      const data = await requestExercise();
      renderExercise(data); typeset($("body")); window.scrollTo(0, 0);
    } catch (e) {
      $("ex-err").textContent = e.message; $("ex-err").hidden = false;
    } finally {
      S.busy = false; $("next").disabled = false; $("next").textContent = "Nouvel exercice"; $("s-ex").classList.remove("busy");
    }
  });
  $("change").addEventListener("click", () => { stopPoll(); $("chapter").value = S.chapter; markChip(); show("s-setup"); });

  // état de la réserve, rafraîchi régulièrement pendant qu'on lit l'exercice
  async function refreshStock() {
    if ($("s-ex").hidden || !S.chapter) return;
    try {
      const r = await fetch(`/api/stock?chapter=${encodeURIComponent(S.chapter)}&difficulty=${S.diff}`);
      const j = await r.json(); setStock(j.remaining);
    } catch (e) {}
  }
  function startPoll() { stopPoll(); S.poll = setInterval(refreshStock, 5000); }
  function stopPoll() { clearInterval(S.poll); }

  // ---------- INIT ----------
  loadPrefs();
  fetch("/api/config").then((r) => r.json()).then((c) => { S.cfg = c; buildSetup(); });
})();
</script>
</body>
</html>
"""


# ============================================================
# LANCEMENT
# ============================================================

if __name__ == "__main__":
    init_db()
    url = "http://127.0.0.1:5001"
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    print(f"Maths Coach : {url}")
    app.run(host="127.0.0.1", port=5001, debug=False, threaded=True)
