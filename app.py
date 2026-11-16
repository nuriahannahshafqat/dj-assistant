import fcntl
import json
import logging
import os
import re
import secrets
import tempfile
from datetime import timedelta

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv
from flask import Flask, redirect, render_template, request, session, url_for

import dj_logic
from audio_analysis import AudioAnalysisError, analyse_preview

load_dotenv()

app = Flask(__name__)

_secret = os.getenv("SECRET_KEY")
if not _secret:
    raise RuntimeError("SECRET_KEY environment variable must be set before running the app.")
app.secret_key = _secret
app.permanent_session_lifetime = timedelta(days=365)

logging.basicConfig(level=logging.ERROR)

MUSICBRAINZ_BASE = "https://musicbrainz.org/ws/2"
MB_HEADERS = {"User-Agent": "dj-assistant/1.0 (nuriashafqat711@gmail.com)"}
ITUNES_SEARCH_URL = "https://itunes.apple.com/search"
SEED_LIBRARY_PATH = "library.json"
VISITOR_LIBRARIES_PATH = "visitor_libraries.json"
MB_TIMEOUT = 10
CLAUDE_TIMEOUT = 60
SHORTLIST_SIZE = 15

_retry = Retry(total=3, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
_session = requests.Session()
_session.mount("https://", HTTPAdapter(max_retries=_retry))


def _mb_artist_name(recording: dict) -> str:
    """Builds the joint artist credit, preferring each artist's canonical
    registered name over the (sometimes inconsistently cased) per-recording
    credited-as name, and normalizing ft/feat join phrases to include a period.
    """
    credits = recording.get("artist-credit", [])
    parts = []
    for c in credits:
        if isinstance(c, dict):
            parts.append(c.get("artist", {}).get("name") or c.get("name", ""))
            parts.append(c.get("joinphrase", ""))
        elif isinstance(c, str):
            parts.append(c)
    name = "".join(parts).strip()
    name = re.sub(r"\bft\b(?!\.)", "ft.", name)
    name = re.sub(r"\bfeat\b(?!\.)", "feat.", name)
    return name


def _mb_fetch_details(rec_id: str, track: str, artist: str) -> dict:
    detail = _session.get(
        f"{MUSICBRAINZ_BASE}/recording/{rec_id}",
        params={"inc": "tags+genres+artist-credits", "fmt": "json"},
        headers=MB_HEADERS,
        timeout=MB_TIMEOUT,
    )
    detail.raise_for_status()
    data = detail.json()
    return {
        "title": data.get("title", track),
        "artist": _mb_artist_name(data) or artist,
        "mbid": rec_id,
        "tags": [t["name"] for t in data.get("tags", [])],
        "genres": [g["name"] for g in data.get("genres", [])],
    }


def _mb_search(query: str, limit: int = 5) -> list:
    resp = _session.get(
        f"{MUSICBRAINZ_BASE}/recording",
        params={"query": query, "fmt": "json", "limit": limit},
        headers=MB_HEADERS,
        timeout=MB_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json().get("recordings", [])


def _pick_recording(results: list, typed_track: str) -> dict | None:
    """Chooses the best confident MusicBrainz match.

    MusicBrainz often scores an instrumental/remix/live recording as highly
    as the original, so unless the user actually asked for one of those,
    prefer a confident match whose title isn't an alternate version.
    """
    confident = [r for r in results if r.get("score", 0) >= 85]
    if not confident:
        return None
    if _is_alt_version(typed_track):
        return confident[0]
    originals = [r for r in confident if not _is_alt_version(r.get("title", ""))]
    return (originals or confident)[0]


def search_musicbrainz(track: str, artist: str) -> tuple:
    """Returns (best_match_dict, suggestions_list).

    Tries a strict quoted query first; falls back to an unquoted fuzzy query
    to populate suggestions when the strict search returns nothing or low confidence.
    """
    strict = _mb_search(f'recording:"{track}" AND artist:"{artist}"', limit=10)
    best = _pick_recording(strict, track)
    if best:
        return _mb_fetch_details(best["id"], track, artist), []

    # Strict query failed or low confidence — try fuzzy (no quotes)
    fuzzy = _mb_search(f"recording:{track} artist:{artist}")

    # Deduplicate by (title, artist) preserving order
    seen = set()
    suggestions = []
    for r in (strict + fuzzy):
        if r.get("score", 0) < 40:
            continue
        key = (r.get("title", "").lower(), _mb_artist_name(r).lower())
        if key not in seen:
            seen.add(key)
            suggestions.append({"title": r.get("title", ""), "artist": _mb_artist_name(r)})
        if len(suggestions) == 5:
            break

    return {}, suggestions


_FEAT_PATTERN = re.compile(r"\s*[\(\[](feat\.|with)[^\)\]]*[\)\]]", re.IGNORECASE)
_ALT_VERSION_KEYWORDS = (
    "remix", "instrumental", "mix)", "megamix", "sped up", "slowed",
    "live", "acoustic", "karaoke", "cover", "mixed", "extended", "version)",
)


def _normalize_title(s: str) -> str:
    s = _FEAT_PATTERN.sub("", s)
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _is_alt_version(track_name: str) -> bool:
    lowered = track_name.lower()
    return any(kw in lowered for kw in _ALT_VERSION_KEYWORDS)


def get_preview_url(track: str, artist: str) -> str | None:
    try:
        resp = _session.get(
            ITUNES_SEARCH_URL,
            params={"term": f"{track} {artist}", "entity": "song", "limit": 10},
            timeout=MB_TIMEOUT,
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
        if not results:
            return None

        candidates = [r for r in results if not _is_alt_version(r.get("trackName", ""))] or results

        target_title = _normalize_title(track)
        target_artist = artist.lower()
        exact = [
            r for r in candidates
            if _normalize_title(r.get("trackName", "")) == target_title
            and target_artist in r.get("artistName", "").lower()
        ]

        best = exact[0] if exact else candidates[0]
        return best.get("previewUrl")
    except Exception:
        logging.exception("iTunes preview lookup failed")
        return None


def get_visitor_id() -> str:
    if "visitor_id" not in session:
        session["visitor_id"] = secrets.token_urlsafe(16)
        session.permanent = True
    return session["visitor_id"]


class _FileLock:
    """Cross-process lock so concurrent requests (or multiple gunicorn
    workers) can't interleave a read-modify-write of the libraries file."""

    def __init__(self, path):
        self.path = path + ".lock"

    def __enter__(self):
        self._fh = open(self.path, "w")
        fcntl.flock(self._fh, fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc):
        fcntl.flock(self._fh, fcntl.LOCK_UN)
        self._fh.close()


def _load_all_visitor_libraries() -> dict:
    if not os.path.exists(VISITOR_LIBRARIES_PATH):
        return {}
    with open(VISITOR_LIBRARIES_PATH) as f:
        return json.load(f)


def _save_all_visitor_libraries(data: dict) -> None:
    # Write to a temp file then atomically swap it in, so a crash mid-write
    # can never leave a half-written (corrupt) JSON file behind.
    directory = os.path.dirname(os.path.abspath(VISITOR_LIBRARIES_PATH))
    fd, tmp_path = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, VISITOR_LIBRARIES_PATH)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def load_library(visitor_id: str) -> list:
    with _FileLock(VISITOR_LIBRARIES_PATH):
        data = _load_all_visitor_libraries()
        if visitor_id not in data:
            with open(SEED_LIBRARY_PATH) as f:
                data[visitor_id] = json.load(f)
            _save_all_visitor_libraries(data)
        return data[visitor_id]


def save_library(visitor_id: str, library: list) -> None:
    with _FileLock(VISITOR_LIBRARIES_PATH):
        data = _load_all_visitor_libraries()
        data[visitor_id] = library
        _save_all_visitor_libraries(data)


def call_claude(prompt: str, max_tokens: int = 512) -> str:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set")
    resp = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": "claude-sonnet-4-6",
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=CLAUDE_TIMEOUT,
    )
    if not resp.ok:
        raise RuntimeError(f"Anthropic API error {resp.status_code}: {resp.text}")
    return resp.json()["content"][0]["text"]


def parse_json_response(raw: str) -> dict:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise RuntimeError("Could not parse JSON from API response.")
    return json.loads(match.group())


def _safe_tags(mb_info: dict) -> str:
    tags = [re.sub(r"[^\w\s\-&']", "", t)[:50] for t in mb_info.get("tags", [])[:20]]
    return ", ".join(t for t in tags if t) or "none"


def get_track_metadata(track: str, artist: str, mb_info: dict) -> dict:
    prompt = f"""What is the BPM, Camelot Wheel key, energy level, and genre of "{track}" by {artist}?
MusicBrainz tags for reference: {_safe_tags(mb_info)}

Before answering, decide honestly: do you actually recognize this specific recording and recall its real,
documented tempo/key — or are you estimating from genre/artist conventions because you don't have specific
knowledge of this exact track? Rate your confidence accordingly; don't default to "high" just because you
produced a specific-sounding number.

Respond ONLY with a JSON object, no other text:
{{"bpm": <integer>, "key": "<Camelot key e.g. 8A or 5B>", "energy": <integer 1-10>, "genre": "<primary genre e.g. R&B, Hip-Hop, Pop>", "confidence": "<high|medium|low>", "confidence_note": "<one short sentence: recalled fact or estimate, and why>"}}"""
    return parse_json_response(call_claude(prompt, max_tokens=300))


def build_track_metadata(track: str, artist: str, mb_info: dict, preview_url: str | None) -> dict:
    """BPM and key are measured from the audio preview when possible; the LLM
    supplies the subjective fields (energy, genre) and is the fallback for
    BPM/key when there's no preview or analysis fails.

    The LLM's own BPM/key guesses are kept alongside (`llm_bpm`, `llm_key`) so
    the two sources can be compared — see scripts/analyse_library.py.
    """
    llm = get_track_metadata(track, artist, mb_info)
    meta = {
        "bpm": llm.get("bpm") or 0,
        "key": llm.get("key") or "1A",
        "genre": llm.get("genre") or "Unknown",
        "energy": llm.get("energy") or 5,
        "llm_bpm": llm.get("bpm"),
        "llm_key": llm.get("key"),
        "source": "llm",
        "confidence": llm.get("confidence") or "low",
        "confidence_note": llm.get("confidence_note") or "",
    }

    if preview_url:
        try:
            audio = analyse_preview(preview_url, session=_session)
            meta.update({
                "bpm": audio["bpm"],
                "key": audio["key"],
                "source": "audio",
                "confidence": audio["confidence"],
                "confidence_note": (
                    f"Measured from the audio preview: {audio['bpm']} BPM "
                    f"({audio['bpm_confidence']} confidence), {audio['key_name']} "
                    f"({audio['key_confidence']} confidence)."
                ),
            })
        except AudioAnalysisError:
            logging.exception("Audio analysis failed; falling back to LLM estimate")
    return meta


def find_in_library(track: str, artist: str, library: list) -> dict | None:
    probe = {"title": track, "artist": artist}
    return next((t for t in library if dj_logic.same_track(probe, t)), None)


def _track_view(t: dict) -> dict:
    return {
        "title": t.get("title", ""),
        "artist": t.get("artist", ""),
        "bpm": t.get("bpm"),
        "key": t.get("key"),
        "energy": t.get("energy"),
        "genre": t.get("genre"),
    }


def recommend_next_track(current: dict, mb_info: dict, library: list) -> dict:
    """Two-stage recommender.

    1. Python ranks the library by Camelot/BPM/energy compatibility and keeps
       the top SHORTLIST_SIZE candidates.
    2. Claude picks from that shortlist *by ID* and writes the explanation,
       weighing vibe and genre, which rules can't capture.

    The model's answer is then validated: unknown or duplicate IDs are
    replaced with the best-scoring valid track, and BPM/key figures shown to
    the user always come from the library, never from model output. If the
    API call fails entirely, a fully rule-based recommendation is returned.
    """
    candidates = dj_logic.shortlist(current, library, limit=SHORTLIST_SIZE)
    if not candidates:
        raise RuntimeError("Your library needs at least one other track to recommend from.")

    by_id = {f"t{n + 1}": c for n, c in enumerate(candidates)}
    lines = []
    for cid, (_, t, tr) in by_id.items():
        lines.append(
            f'{cid}: "{t.get("title")}" by {t.get("artist")} | {t.get("bpm")} BPM | '
            f'{t.get("key")} | energy {t.get("energy")} | {t.get("genre")} | '
            f'match {tr["score"]:.2f} ({tr["key_label"]}; {tr["bpm_label"]}; {tr["energy_label"]})'
        )

    prompt = f"""You are an expert DJ assistant. Choose what to play after the current track.

Current track: "{current.get('title')}" by {current.get('artist')}
- {current.get('bpm')} BPM, key {current.get('key')} (Camelot), energy {current.get('energy')}/10
- MusicBrainz tags: {_safe_tags(mb_info)}
- Genres: {", ".join(g[:50] for g in mb_info.get("genres", [])[:10]) or "none found"}

These candidates have already been checked for harmonic and tempo compatibility
and ranked by a match score (key, BPM incl. half/double time, energy flow).
You may ONLY choose from these IDs:
{chr(10).join(lines)}

Use the match scores as a strong guide, but use your judgement on vibe, mood
and genre flow — e.g. prefer a slightly lower score if the vibe is much better.

Respond with ONLY a JSON object, no other text:
{{
  "recommended_id": "t?",
  "bpm_compatibility": "2-3 sentences on the BPM relationship and whether it works",
  "energy_vibes": "2-3 sentences on energy level, mood, and genre fit",
  "summary": "1 punchy paragraph — why this is the right next track",
  "alternatives": [{{"id": "t?", "reason": "one sentence"}}, {{"id": "t?", "reason": "one sentence"}}, {{"id": "t?", "reason": "one sentence"}}],
  "chain": [{{"id": "t?", "reason": "one sentence on the transition from the recommended track"}}, {{"id": "t?", "reason": "transition from chain track 1"}}, {{"id": "t?", "reason": "transition from chain track 2"}}]
}}

Alternatives: 3 other IDs, not the recommended one. Chain: 3 IDs that follow the
recommended track in order to build a coherent set; don't repeat any ID used above."""

    try:
        raw = parse_json_response(call_claude(prompt, max_tokens=1500))
    except Exception:
        logging.exception("LLM recommendation failed; using rule-based fallback")
        raw = {}

    return _assemble_recommendation(current, library, by_id, raw)


def _assemble_recommendation(current: dict, library: list, by_id: dict, raw: dict) -> dict:
    """Turns the model's ID-based answer into the template's format, fixing
    anything invalid so the user never sees a track that isn't in their library."""
    ranked_ids = list(by_id)  # already best-first
    used = set()
    replaced = 0

    def take(cid):
        nonlocal replaced
        if cid in by_id and cid not in used:
            used.add(cid)
            return cid, False
        for fallback in ranked_ids:
            if fallback not in used:
                used.add(fallback)
                replaced += 1
                return fallback, True
        return None, True

    rec_id, rec_replaced = take(raw.get("recommended_id"))
    _, rec_track, rec_tr = by_id[rec_id]

    result = {
        "recommended_track": rec_track.get("title"),
        "recommended_artist": rec_track.get("artist"),
        "recommended_bpm": rec_track.get("bpm"),
        "recommended_key": rec_track.get("key"),
        "match_score": round(rec_tr["score"] * 100),
        "bpm_compatibility": raw.get("bpm_compatibility") if not rec_replaced else None,
        "energy_vibes": raw.get("energy_vibes") if not rec_replaced else None,
        "summary": raw.get("summary") if not rec_replaced else None,
        "alternatives": [],
        "chain": [],
    }
    if not result["bpm_compatibility"]:
        result["bpm_compatibility"] = (
            f"{current.get('bpm')} → {rec_track.get('bpm')} BPM: {rec_tr['bpm_label']}."
        )
    if not result["energy_vibes"]:
        result["energy_vibes"] = (
            f"Energy {current.get('energy')} → {rec_track.get('energy')}: {rec_tr['energy_label']}."
        )
    if not result["summary"]:
        result["summary"] = (
            f"Top-ranked match from your library: {dj_logic.describe_transition(current, rec_track)}"
        )

    # Alternatives: must come from the shortlist.
    for alt in (raw.get("alternatives") or [])[:3]:
        cid, was_replaced = take(alt.get("id") if isinstance(alt, dict) else None)
        if cid is None:
            break
        _, t, _ = by_id[cid]
        reason = alt.get("reason") if not was_replaced else None
        result["alternatives"].append({
            **_track_view(t),
            "reason": reason or dj_logic.describe_transition(current, t),
        })
    while len(result["alternatives"]) < 3:
        cid, _ = take(None)
        if cid is None:
            break
        _, t, _ = by_id[cid]
        result["alternatives"].append({**_track_view(t), "reason": dj_logic.describe_transition(current, t)})

    # Chain: each step must be a real track AND mix well from the previous one.
    prev = rec_track
    for step in (raw.get("chain") or [])[:3]:
        cid = step.get("id") if isinstance(step, dict) else None
        if cid not in by_id or cid in used:
            break
        _, t, _ = by_id[cid]
        if dj_logic.transition(prev, t)["key_score"] == 0.0:
            break  # key clash — stop trusting the model's chain from here
        used.add(cid)
        result["chain"].append({**_track_view(t), "reason": step.get("reason") or dj_logic.describe_transition(prev, t)})
        prev = t

    if len(result["chain"]) < 3:
        exclude = {by_id[c][0] for c in used}
        exclude |= {i for i, t in enumerate(library) if dj_logic.same_track(t, current)}
        for _, t, _ in dj_logic.build_chain(prev, library, exclude, length=3 - len(result["chain"])):
            result["chain"].append({**_track_view(t), "reason": dj_logic.describe_transition(prev, t)})
            replaced += 1
            prev = t

    result["llm_corrections"] = replaced
    return result


@app.route("/", methods=["GET", "POST"])
def index():
    visitor_id = get_visitor_id()
    rec = None
    search_error = None
    suggestions = []
    track = ""
    artist = ""

    add_success = request.args.get("add_success")
    add_error = request.args.get("add_error")
    add_warning = request.args.get("add_warning")
    add_suggestions = session.pop("add_suggestions", [])
    library = list(reversed(load_library(visitor_id)))

    if request.method == "POST":
        track = request.form.get("track", "").strip()
        artist = request.form.get("artist", "").strip()

        if not track or not artist:
            search_error = "Please enter both a track name and artist."
        elif len(track) > 200 or len(artist) > 200:
            search_error = "Track and artist names must be under 200 characters."
        else:
            try:
                mb_info, suggestions = search_musicbrainz(track, artist)
                if not mb_info:
                    search_error = f'We couldn\'t find "{track}" by {artist}. Please check the spelling or try one of the suggestions below.'
                else:
                    mb_artist = mb_info.get("artist") or artist
                    mb_title = mb_info.get("title") or track
                    lib = load_library(visitor_id)
                    # Use stored BPM/key if the track is already in the library;
                    # otherwise measure it so the shortlist has real numbers.
                    current = find_in_library(mb_title, mb_artist, lib) or find_in_library(track, mb_artist, lib)
                    if current is None:
                        preview = get_preview_url(track, mb_artist)
                        meta = build_track_metadata(track, mb_artist, mb_info, preview)
                        current = {"title": mb_title, "artist": mb_artist, **meta}
                    rec = recommend_next_track(current, mb_info, lib)
            except Exception as e:
                logging.exception("Search error")
                search_error = "Something went wrong. Please try again."

    return render_template(
        "index.html",
        track=track,
        artist=artist,
        rec=rec,
        search_error=search_error,
        suggestions=suggestions,
        add_success=add_success,
        add_error=add_error,
        add_warning=add_warning,
        add_suggestions=add_suggestions,
        library=library,
        active_page="home",
    )


@app.route("/add", methods=["POST"])
def add_track():
    visitor_id = get_visitor_id()
    track = request.form.get("track", "").strip()
    artist = request.form.get("artist", "").strip()

    if not track or not artist:
        return redirect(url_for("index", add_error="Please enter both a track name and artist."))

    if len(track) > 200 or len(artist) > 200:
        return redirect(url_for("index", add_error="Track and artist names must be under 200 characters."))

    try:
        mb_info, add_suggestions = search_musicbrainz(track, artist)
        if not mb_info:
            session["add_suggestions"] = add_suggestions
            return redirect(url_for("index", add_error=f'Couldn\'t find "{track}" by {artist}. Did you mean one of these?'))

        mb_artist = mb_info.get("artist") or artist

        library = load_library(visitor_id)
        mb_title = mb_info.get("title", track).lower()
        for t in library:
            stored_title = t["title"].lower()
            stored_artist = t["artist"].lower()
            if stored_artist == mb_artist.lower() and (stored_title == track.lower() or stored_title == mb_title):
                return redirect(url_for("index", add_warning=f'"{t["title"]}" by {t["artist"]} is already in your library.'))

        preview_url = get_preview_url(track, mb_artist)
        metadata = build_track_metadata(track, mb_artist, mb_info, preview_url)
        confidence = metadata["confidence"]

        library.append({
            "title": mb_info.get("title", track),
            "artist": mb_artist,
            **metadata,
            "preview_url": preview_url,
        })
        save_library(visitor_id, library)

        add_success = f'"{mb_info.get("title", track)}" by {mb_artist} added to your library!'
        if confidence in ("low", "medium"):
            note = metadata.get("confidence_note", "")
            suffix = f" — {note}" if note else ""
            return redirect(url_for(
                "index",
                add_success=add_success,
                add_warning=f"BPM/key/energy confidence for this track is {confidence}{suffix}. Worth double-checking manually.",
            ))
        return redirect(url_for("index", add_success=add_success))

    except Exception:
        logging.exception("Add track error")
        return redirect(url_for("index", add_error="Something went wrong adding that track. Please try again."))


@app.route("/remove", methods=["POST"])
def remove_track():
    visitor_id = get_visitor_id()
    title = request.form.get("title", "").strip()
    artist = request.form.get("artist", "").strip()
    next_page = request.form.get("next", "index")
    target = "library_page" if next_page == "library" else "index"

    library = load_library(visitor_id)
    filtered = [t for t in library if not (t["title"] == title and t["artist"] == artist)]

    if len(filtered) == len(library):
        return redirect(url_for(target, add_error=f'Couldn\'t find "{title}" by {artist} to remove.'))

    save_library(visitor_id, filtered)
    return redirect(url_for(target, add_success=f'Removed "{title}" by {artist} from your library.'))


@app.route("/library")
def library_page():
    visitor_id = get_visitor_id()
    library = list(reversed(load_library(visitor_id)))
    add_success = request.args.get("add_success")
    add_error = request.args.get("add_error")
    return render_template(
        "library.html",
        library=library,
        add_success=add_success,
        add_error=add_error,
        active_page="library",
    )


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", 5000)),
        debug=os.getenv("FLASK_DEBUG", "false").lower() == "true",
    )
