"""Deterministic DJ mixing rules: Camelot key compatibility, BPM matching
(including half/double time) and energy flow.

These run in plain Python before the LLM is involved, so the model only ever
chooses between tracks that are already known to mix well, and every number
shown to the user comes from here rather than from model-written text.
"""

import re

CAMELOT_RE = re.compile(r"^\s*(1[0-2]|[1-9])\s*([ABab])\s*$")

# Weights for the overall transition score. Key and tempo matter most for a
# clean mix; energy shapes the set.
WEIGHTS = {"key": 0.45, "bpm": 0.40, "energy": 0.15}

# Tempo difference (as a fraction) that a DJ can comfortably pitch across.
BPM_IDEAL = 0.03
BPM_MAX = 0.08

# Furthest the in-browser mixer will stretch a track to sync it (keep in step
# with MAX_PITCH in templates/index.html). Picks beyond this can't be
# beat-matched at all, so they're only offered when nothing closer exists.
SYNC_MAX = 0.16


def parse_camelot(key):
    """'8A' -> (8, 'A'). Returns None for anything that isn't a Camelot key."""
    if not isinstance(key, str):
        return None
    m = CAMELOT_RE.match(key)
    if not m:
        return None
    return int(m.group(1)), m.group(2).upper()


def _wheel_steps(a, b):
    """Shortest distance between two positions on the 12-step wheel."""
    d = abs(a - b) % 12
    return min(d, 12 - d)


def key_relation(key_a, key_b):
    """Describes the harmonic relationship between two Camelot keys.

    Returns (score 0-1, label). Standard harmonic-mixing moves:
      same key                         -> perfect
      same number, A<->B               -> relative major/minor
      +/-1 on the wheel, same letter   -> adjacent (smooth)
      +/-2 on the wheel, same letter   -> energy boost (works, more noticeable)
      +/-1 and switch letter           -> diagonal (usable, riskier)
    Anything else clashes.
    """
    a, b = parse_camelot(key_a), parse_camelot(key_b)
    if a is None or b is None:
        return 0.3, "unknown key"
    steps = _wheel_steps(a[0], b[0])
    same_letter = a[1] == b[1]
    if steps == 0 and same_letter:
        return 1.0, "same key"
    if steps == 0:
        return 0.9, "relative major/minor"
    if steps == 1 and same_letter:
        return 0.85, "adjacent key"
    if steps == 2 and same_letter:
        return 0.5, "energy boost (+2)"
    if steps == 1:
        return 0.4, "diagonal move"
    return 0.0, "key clash"


def bpm_relation(bpm_a, bpm_b):
    """Compares tempos, allowing half/double time (e.g. 75 vs 150).

    Returns (score 0-1, percentage difference, label).
    """
    try:
        a, b = float(bpm_a), float(bpm_b)
    except (TypeError, ValueError):
        return 0.3, None, "unknown BPM"
    if a <= 0 or b <= 0:
        return 0.3, None, "unknown BPM"

    options = [(b, ""), (b * 2, " (half time)"), (b / 2, " (double time)")]
    target, note = min(options, key=lambda o: abs(o[0] - a) / a)
    diff = abs(target - a) / a

    if diff <= BPM_IDEAL:
        score = 1.0
    elif diff >= BPM_MAX:
        score = 0.0
    else:
        score = 1.0 - (diff - BPM_IDEAL) / (BPM_MAX - BPM_IDEAL)
    return score, round(diff * 100, 1), f"{diff * 100:.1f}% tempo change{note}"


def energy_relation(energy_a, energy_b):
    """Holding or gently lifting energy scores best; big drops score worst."""
    try:
        delta = int(energy_b) - int(energy_a)
    except (TypeError, ValueError):
        return 0.5, "unknown energy"
    if 0 <= delta <= 2:
        return 1.0, "holds/lifts energy" if delta else "holds energy"
    if delta == -1:
        return 0.8, "slight energy dip"
    if delta > 2:
        return 0.6, "big energy jump"
    return max(0.0, 0.6 + 0.15 * (delta + 2)), "energy drop"


def transition(current, candidate):
    """Full scored breakdown for mixing from `current` into `candidate`."""
    k_score, k_label = key_relation(current.get("key"), candidate.get("key"))
    b_score, b_pct, b_label = bpm_relation(current.get("bpm"), candidate.get("bpm"))
    e_score, e_label = energy_relation(current.get("energy"), candidate.get("energy"))
    total = (
        WEIGHTS["key"] * k_score
        + WEIGHTS["bpm"] * b_score
        + WEIGHTS["energy"] * e_score
    )
    return {
        "score": round(total, 3),
        "key_score": k_score,
        "key_label": k_label,
        "bpm_score": b_score,
        "bpm_pct": b_pct,
        "bpm_label": b_label,
        "energy_score": e_score,
        "energy_label": e_label,
    }


def describe_transition(current, candidate):
    """Plain-English one-liner built from real numbers, used when the LLM is
    unavailable or its answer had to be replaced."""
    t = transition(current, candidate)
    text = f"{t['key_label']}, {t['bpm_label']}."
    return text[0].upper() + text[1:]


def same_track(a, b):
    return (
        a.get("title", "").strip().lower() == b.get("title", "").strip().lower()
        and a.get("artist", "").strip().lower() == b.get("artist", "").strip().lower()
    )


def can_sync(t):
    """True unless the tempos are known to be too far apart to beat-match."""
    return t["bpm_pct"] is None or t["bpm_pct"] <= SYNC_MAX * 100


def smooth_step(t):
    """A set-chain step a DJ would actually play: no key clash, and the tempo
    within comfortable pitch range (BPM_MAX) rather than merely syncable."""
    return t["key_score"] > 0.0 and (t["bpm_pct"] is None or t["bpm_pct"] < BPM_MAX * 100)


# A recommendation plus 3 alternatives.
MIN_PICKS = 4


def shortlist(current, library, limit=15):
    """Ranks library tracks by how well they follow `current`.

    Returns a list of (index_in_library, track, transition_dict), best first.
    The current track itself is excluded. Tracks whose tempo can't be synced
    are left out, unless there aren't enough syncable ones for a full set of
    picks, in which case they're added at the end.
    """
    ranked = []
    for i, track in enumerate(library):
        if same_track(current, track):
            continue
        ranked.append((i, track, transition(current, track)))
    ranked.sort(key=lambda r: r[2]["score"], reverse=True)
    syncable = [r for r in ranked if can_sync(r[2])]
    if len(syncable) >= MIN_PICKS:
        return syncable[:limit]
    return (syncable + [r for r in ranked if not can_sync(r[2])])[:limit]


MIN_CHAIN_SCORE = 0.45


def build_chain(start, library, exclude_indices, length=3, min_score=MIN_CHAIN_SCORE):
    """Greedy set builder: from `start`, repeatedly pick the best-scoring
    unused track. Used as a fallback when the LLM's chain is invalid.
    Stops early rather than adding a transition that clashes."""
    used = set(exclude_indices)
    chain = []
    prev = start
    for _ in range(length):
        best = None
        for i, track in enumerate(library):
            if i in used or same_track(prev, track) or same_track(start, track):
                continue
            t = transition(prev, track)
            if not smooth_step(t) or t["score"] < min_score:
                continue
            if best is None or t["score"] > best[2]["score"]:
                best = (i, track, t)
        if best is None:
            break
        chain.append(best)
        used.add(best[0])
        prev = best[1]
    return chain
