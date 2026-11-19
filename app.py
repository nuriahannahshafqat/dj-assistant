import difflib
import fcntl
import json
import logging
import os
import re
import secrets
import tempfile
import threading
import time
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
# MusicBrainz asks clients to identify themselves with a way to get in touch.
# Set MB_CONTACT (an email or URL) in .env to use your own; it defaults to the repo.
MB_CONTACT = os.getenv("MB_CONTACT") or "https://github.com/nuriahannahshafqat/dj-assistant"
MB_HEADERS = {"User-Agent": f"dj-assistant/1.0 ( {MB_CONTACT} )"}
ITUNES_SEARCH_URL = "https://itunes.apple.com/search"
SEED_LIBRARY_PATH = "library.json"
VISITOR_LIBRARIES_PATH = "visitor_libraries.json"
MB_TIMEOUT = 10
CLAUDE_TIMEOUT = 60
SHORTLIST_SIZE = 15

# Retry brief server hiccups, but never a timed-out read: a server that
# isn't answering shouldn't keep the user waiting (search falls back to iTunes).
_retry = Retry(total=3, read=0, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
MB_SEARCH_TIMEOUT = (4, 6)   # (connect, read) seconds
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
    _mb_wait_turn()
    detail = _session.get(
        f"{MUSICBRAINZ_BASE}/recording/{rec_id}",
        params={"inc": "tags+genres+artist-credits", "fmt": "json"},
        headers=MB_HEADERS,
        timeout=MB_SEARCH_TIMEOUT,
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


_MB_MIN_INTERVAL = 1.1      # MusicBrainz allows ~1 request/second per client
_mb_last_call = [0.0]
_mb_lock = threading.Lock()


def _mb_wait_turn():
    """Spaces out MusicBrainz calls; back-to-back queries otherwise get 503s."""
    with _mb_lock:
        wait = _mb_last_call[0] + _MB_MIN_INTERVAL - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _mb_last_call[0] = time.monotonic()


def _mb_search(query: str, limit: int = 5) -> list:
    _mb_wait_turn()
    resp = _session.get(
        f"{MUSICBRAINZ_BASE}/recording",
        params={"query": query, "fmt": "json", "limit": limit},
        headers=MB_HEADERS,
        timeout=MB_SEARCH_TIMEOUT,
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


_LUCENE_SPECIAL = re.compile(r'[+\-&|!(){}\[\]^"~*?:\\/]')


def _fuzzy_terms(text: str) -> str:
    """'save ur tears' -> 'save~ ur~ tears~' (each word may be up to 2 edits off,
    so slang/typos like 'ur' -> 'your' still match)."""
    words = _LUCENE_SPECIAL.sub(" ", text).split()
    return " ".join(f"{w}~" for w in words)


def _words(s: str) -> str:
    s = _FEAT_PATTERN.sub("", s or "")
    return " ".join(re.sub(r"[^a-z0-9]+", " ", s.lower()).split())


def _similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, _words(a), _words(b)).ratio()


MAX_SUGGESTIONS = 5
_BRACKETS = re.compile(r"\s*[\(\[][^\)\]]*[\)\]]")


def _rank_suggestions(candidates: list, track: str, artist: str) -> list:
    """Turns raw (title, artist) search hits into a short "Did you mean?" list.

    Search engines score loosely (a cover of the song, or 'Save Ur Breath',
    can outrank the real thing), so hits are re-scored by how close the title
    and artist are to what was typed. Covers/remixes/karaoke are dropped unless
    the user asked for one.
    """
    wants_alt = _is_alt_version(track)
    target_artist = _ARTIST_SPLIT.split(artist.strip(), maxsplit=1)[0]
    scored = []
    for title, cand_artist in candidates:
        if not title or not cand_artist:
            continue
        if not wants_alt and _is_alt_version(title):
            continue
        if " / " in title and "/" not in track:      # medleys, e.g. "The Morning / Save Your Tears"
            continue
        cand_main = _ARTIST_SPLIT.split(cand_artist.strip(), maxsplit=1)[0]
        artist_sim = max(_similarity(target_artist, cand_main), _similarity(artist, cand_artist))
        title_sim = _similarity(track, title)
        if artist_sim < 0.6 or title_sim < 0.6:
            continue
        # Prefer the exact artist spelling and the plain title over "(edit)" etc.
        scored.append((title_sim + artist_sim + (0.05 if title == _BRACKETS.sub("", title) else 0),
                       title, cand_main, cand_artist))
    scored.sort(key=lambda s: s[0], reverse=True)

    # One suggestion per song: "Kill Bill (vocals)", "kill bill [nightcore]" and a
    # mis-tagged "The Weekend" upload are all the same song as the top hit.
    kept = []
    for _, title, cand_main, cand_artist in scored:
        base = _words(_BRACKETS.sub("", title))
        if any(base == _words(_BRACKETS.sub("", k_title)) and _similarity(cand_main, k_main) >= 0.8
               for k_title, k_main, _ in kept):
            continue
        kept.append((title, cand_main, cand_artist))
        if len(kept) == MAX_SUGGESTIONS:
            break
    return [{"title": t, "artist": a} for t, _, a in kept]


def _suggest(track: str, artist: str) -> list:
    """Spelling-tolerant "Did you mean?" options for a search with no exact match."""
    candidates = []
    if _mb_available():
        try:
            fuzzy = _mb_search(
                f"recording:({_fuzzy_terms(track)}) AND artist:({_fuzzy_terms(artist)})", limit=25
            )
            candidates = [(r.get("title", ""), _mb_artist_name(r)) for r in fuzzy]
        except requests.RequestException:
            logging.warning("MusicBrainz fuzzy search failed", exc_info=True)
            _mark_mb_down()

    suggestions = _rank_suggestions(candidates, track, artist)
    if suggestions:
        return suggestions

    # iTunes search copes well with slang and partial titles, so try it too.
    try:
        hits = _itunes_search(f"{track} {artist}", 25)
        candidates = [(h.get("trackName", ""), h.get("artistName", "")) for h in hits]
    except requests.RequestException:
        logging.exception("iTunes suggestion search failed")
        return []
    return _rank_suggestions(candidates, track, artist)


class SearchUnavailable(Exception):
    """Neither MusicBrainz nor iTunes could be reached."""


# When MusicBrainz is down (it returns 502s or hangs), don't make every search
# wait on it again: skip straight to iTunes for a minute.
_MB_COOLDOWN = 60
_mb_down_until = [0.0]


def _mb_available() -> bool:
    return time.monotonic() >= _mb_down_until[0]


def _mark_mb_down():
    _mb_down_until[0] = time.monotonic() + _MB_COOLDOWN


def _search_itunes(track: str, artist: str) -> tuple:
    """Same contract as search_musicbrainz, using iTunes only (fallback)."""
    try:
        hits = _itunes_search(f"{track} {artist}", 25)
    except requests.RequestException as e:
        raise SearchUnavailable() from e
    best = _pick_itunes_result(hits, track, artist, require_preview=False)
    if best:
        genre = best.get("primaryGenreName")
        return {"title": best.get("trackName", track), "artist": best.get("artistName", artist),
                "mbid": None, "tags": [], "genres": [genre] if genre else []}, []
    candidates = [(h.get("trackName", ""), h.get("artistName", "")) for h in hits]
    return {}, _rank_suggestions(candidates, track, artist)


def search_musicbrainz(track: str, artist: str) -> tuple:
    """Returns (best_match_dict, suggestions_list).

    Only an exact (quoted) title + artist match is used directly. Anything
    looser - misspellings, slang like 'ur' for 'your' - comes back as
    suggestions for the user to pick from, so we never silently search for a
    different song than the one they meant.

    Falls back to iTunes when MusicBrainz is down; raises SearchUnavailable
    if iTunes can't be reached either.
    """
    if not _mb_available():
        return _search_itunes(track, artist)
    try:
        strict = _mb_search(f'recording:"{track}" AND artist:"{artist}"', limit=10)
        best = _pick_recording(strict, track)
        if best:
            return _mb_fetch_details(best["id"], track, artist), []
    except requests.RequestException:
        logging.warning("MusicBrainz unavailable, falling back to iTunes", exc_info=True)
        _mark_mb_down()
        return _search_itunes(track, artist)
    return {}, _suggest(track, artist)


SEARCH_UNAVAILABLE_MSG = ("The music databases we search (MusicBrainz and iTunes) aren't responding "
                          "right now, so we can't look songs up. Please try again in a few minutes.")


def not_found_message(track: str, artist: str, suggestions: list) -> str:
    if suggestions:
        return f'Couldn\'t find an exact match for "{track}" by {artist}. Did you mean one of these?'
    # The page lists tips underneath (SEARCH_TIPS) when there are no suggestions.
    return f'Unable to find "{track}" by {artist}. A few things to try:'


SEARCH_TIPS = [
    'Write words out in full, e.g. "your" instead of "ur"',
    "Check the spelling of the song title and the artist",
    "Make sure the artist is the one who actually released that song",
    "Some songs aren't listed on MusicBrainz or iTunes (the music databases we search), so they can't be found",
]


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


_ARTIST_SPLIT = re.compile(r"\s+(?:feat\.?|ft\.?|featuring|with|x|&|and)\s+|,\s*", re.IGNORECASE)


def _primary_artist(artist: str) -> str:
    """'SZA feat. Travis Scott' -> 'sza' (alphanumerics only)."""
    first = _ARTIST_SPLIT.split(artist.strip(), maxsplit=1)[0]
    return re.sub(r"[^a-z0-9]", "", first.lower())


def _pick_itunes_result(results: list, track: str, artist: str, require_preview: bool = True) -> dict | None:
    """Returns the iTunes result that is actually this song, or None.

    Both the title and the main artist must match. There's deliberately no
    fallback to the top result: iTunes often ranks a *different* song by the
    same artist first (e.g. "No Idea" for "No Pole" by Don Toliver, which
    isn't on iTunes at all), and playing the wrong song is worse than
    playing nothing.
    """
    target_title = _normalize_title(track)
    target_artist = _primary_artist(artist)
    if not target_title or not target_artist:
        return None
    wants_alt = _is_alt_version(track)

    matches = [
        r for r in results
        if (r.get("previewUrl") or not require_preview)
        and _normalize_title(r.get("trackName", "")) == target_title
        and target_artist in re.sub(r"[^a-z0-9]", "", r.get("artistName", "").lower())
        and (wants_alt or not _is_alt_version(r.get("trackName", "")))
    ]
    return matches[0] if matches else None


def _itunes_search(term: str, limit: int) -> list:
    resp = _session.get(
        ITUNES_SEARCH_URL,
        params={"term": term, "entity": "song", "limit": limit},
        timeout=MB_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json().get("results", [])


def get_preview_url(track: str, artist: str) -> str | None:
    """Preview clip for exactly this song, or None if iTunes doesn't have it.

    Searches "title artist" first. iTunes ranking sometimes buries the
    original under covers and karaoke versions (e.g. Redbone), so if that
    finds nothing, search the main artist's catalogue as well."""
    try:
        best = _pick_itunes_result(_itunes_search(f"{track} {artist}", 25), track, artist)
        if best is None:
            first_artist = _ARTIST_SPLIT.split(artist.strip(), maxsplit=1)[0]
            best = _pick_itunes_result(_itunes_search(first_artist, 200), track, artist)
        return best.get("previewUrl") if best else None
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
        "preview_url": t.get("preview_url"),
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
  "summary": "ONE punchy sentence, max 15 words — why this is the right next track",
  "alternatives": [{{"id": "t?", "reason": "max 10 words"}}, {{"id": "t?", "reason": "max 10 words"}}, {{"id": "t?", "reason": "max 10 words"}}],
  "chain": [{{"id": "t?", "reason": "max 10 words on the transition from the recommended track"}}, {{"id": "t?", "reason": "max 10 words"}}, {{"id": "t?", "reason": "max 10 words"}}]
}}

The page already shows the BPM, key and energy numbers, so don't repeat them —
talk about vibe, mood, and how the mix will feel. Keep it short: DJs skim.
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
        "recommended_energy": rec_track.get("energy"),
        "recommended_preview_url": rec_track.get("preview_url"),
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

    # Alternatives: must come from the shortlist, and mustn't clash in key
    # (the model sometimes picks a same-artist track for "continuity" even
    # though it's a key clash). A clash is only allowed if nothing else is left.
    def clashes(cid):
        return by_id[cid][2]["key_score"] == 0.0

    def best_unused_non_clash():
        return next((c for c in ranked_ids if c not in used and not clashes(c)), None)

    for alt in (raw.get("alternatives") or [])[:3]:
        wanted = alt.get("id") if isinstance(alt, dict) else None
        if wanted in by_id and wanted not in used and clashes(wanted) and best_unused_non_clash():
            wanted = None
        cid, fell_back = take(wanted if wanted is not None else best_unused_non_clash())
        if cid is None:
            break
        was_replaced = fell_back or cid != (alt.get("id") if isinstance(alt, dict) else None)
        if was_replaced and not fell_back:   # take() already counted its own fallbacks
            replaced += 1
        _, t, _ = by_id[cid]
        reason = alt.get("reason") if not was_replaced else None
        result["alternatives"].append({
            **_track_view(t),
            "reason": reason or dj_logic.describe_transition(current, t),
        })
    while len(result["alternatives"]) < 3:
        cid, _ = take(best_unused_non_clash())
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
        if not dj_logic.smooth_step(dj_logic.transition(prev, t)):
            break  # key clash or too big a tempo jump: stop trusting the model's chain here
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


def _deck_view(t: dict) -> dict:
    """What the mixer needs for one deck. Older library entries may predate
    stored preview URLs, so look one up on the fly when it's missing. A stored
    None means "already checked: not on iTunes", so don't search again."""
    if "preview_url" in t:
        url = t["preview_url"]
    else:
        url = get_preview_url(t.get("title", ""), t.get("artist", ""))
    return {
        "title": t.get("title", ""),
        "artist": t.get("artist", ""),
        "bpm": t.get("bpm"),
        "key": t.get("key"),
        "energy": t.get("energy"),
        "preview_url": url,
        # False when BPM/key are only the LLM's estimate (e.g. no iTunes clip to measure)
        "measured": t.get("source") in ("audio", "published"),
    }


def _grade(score: float) -> str:
    return "good" if score >= 0.8 else "ok" if score >= 0.4 else "bad"


def _quick_stats(current: dict, track: dict) -> dict:
    """Short, glanceable numbers for the stat tiles (instead of paragraphs)."""
    tr = dj_logic.transition(current, track)
    show = lambda v: "?" if v in (None, "") else v

    pct = tr["bpm_pct"]
    if pct is None:
        tempo_sub = "unknown"
    elif tr["bpm_score"] == 1.0:
        tempo_sub = "easy blend"
    elif tr["bpm_score"] > 0:
        tempo_sub = f"{pct:g}% pitch"
    else:
        tempo_sub = f"{pct:g}% apart"
    if "half time" in tr["bpm_label"]:
        tempo_sub += " · half-time"
    elif "double time" in tr["bpm_label"]:
        tempo_sub += " · double-time"

    match = round(tr["score"] * 100)
    return {
        "match_value": f"{match}%",
        "match_sub": "great mix" if match >= 80 else "solid" if match >= 60 else "risky",
        "match_grade": _grade(tr["score"]),
        "tempo_value": f"{show(current.get('bpm'))} → {show(track.get('bpm'))}",
        "tempo_sub": tempo_sub,
        "tempo_grade": _grade(tr["bpm_score"]),
        "key_value": f"{show(current.get('key'))} → {show(track.get('key'))}",
        "key_sub": tr["key_label"],
        "key_grade": _grade(tr["key_score"]),
        "energy_value": f"{show(current.get('energy'))} → {show(track.get('energy'))}",
        "energy_sub": tr["energy_label"],
        "energy_grade": _grade(tr["energy_score"]),
    }


def _build_options(current: dict, rec: dict, library: list) -> list:
    """The recommended track plus each alternative, each with everything the
    page needs to swap it into Deck B: stats, a one-line reason, and a set chain."""
    picks = [({
        "title": rec["recommended_track"],
        "artist": rec["recommended_artist"],
        "bpm": rec["recommended_bpm"],
        "key": rec["recommended_key"],
        "energy": rec["recommended_energy"],
        "preview_url": rec["recommended_preview_url"],
    }, rec["summary"], rec["chain"])]

    for alt in rec["alternatives"]:
        exclude = {
            i for i, t in enumerate(library)
            if dj_logic.same_track(t, current) or dj_logic.same_track(t, alt)
        }
        chain, prev = [], alt
        for _, t, _ in dj_logic.build_chain(alt, library, exclude, length=3):
            chain.append({**_track_view(t), "reason": dj_logic.describe_transition(prev, t)})
            prev = t
        picks.append((alt, alt["reason"], chain))

    return [
        {**_deck_view(t), "reason": reason, "stats": _quick_stats(current, t), "chain": chain}
        for t, reason, chain in picks
    ]


# Status messages after an add/remove ride along in the session for exactly one
# page view. (They used to go in the URL, so they stuck around: the Find Next
# Track form posted back to that same URL and showed the old message again.)
_MSG_KINDS = ("add_success", "add_warning", "add_error", "add_tips")


def _redirect_with(endpoint: str, **messages):
    session["page_messages"] = {k: v for k, v in messages.items() if k in _MSG_KINDS and v}
    return redirect(url_for(endpoint))


def _pop_messages() -> dict:
    msgs = session.pop("page_messages", None) or {}
    return {k: msgs.get(k) for k in _MSG_KINDS}


@app.route("/", methods=["GET", "POST"])
def index():
    visitor_id = get_visitor_id()
    rec = None
    search_error = None
    search_tips = False
    suggestions = []
    track = ""
    artist = ""

    msgs = _pop_messages()
    add_success, add_error, add_warning = msgs["add_success"], msgs["add_error"], msgs["add_warning"]
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
                    search_error = not_found_message(track, artist, suggestions)
                    search_tips = not suggestions
                else:
                    mb_artist = mb_info.get("artist") or artist
                    mb_title = mb_info.get("title") or track
                    lib = load_library(visitor_id)
                    # Use stored BPM/key if the track is already in the library;
                    # otherwise measure it so the shortlist has real numbers.
                    current = find_in_library(mb_title, mb_artist, lib) or find_in_library(track, mb_artist, lib)
                    if current is None:
                        preview = get_preview_url(mb_title, mb_artist)
                        meta = build_track_metadata(track, mb_artist, mb_info, preview)
                        current = {"title": mb_title, "artist": mb_artist, **meta, "preview_url": preview}
                    rec = recommend_next_track(current, mb_info, lib)
                    rec["deck_a"] = _deck_view(current)
                    rec["options"] = _build_options(current, rec, lib)
                    rec["deck_b"] = rec["options"][0]
            except SearchUnavailable:
                logging.exception("Song lookup services unavailable")
                search_error = SEARCH_UNAVAILABLE_MSG
            except Exception as e:
                logging.exception("Search error")
                search_error = "Something went wrong. Please try again."

    return render_template(
        "index.html",
        track=track,
        artist=artist,
        rec=rec,
        search_error=search_error,
        search_tips=search_tips,
        tips=SEARCH_TIPS,
        suggestions=suggestions,
        add_success=add_success,
        add_error=add_error,
        add_warning=add_warning,
        add_tips=msgs["add_tips"],
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
        return _redirect_with("index", add_error="Please enter both a track name and artist.")

    if len(track) > 200 or len(artist) > 200:
        return _redirect_with("index", add_error="Track and artist names must be under 200 characters.")

    try:
        mb_info, add_suggestions = search_musicbrainz(track, artist)
        if not mb_info:
            session["add_suggestions"] = add_suggestions
            return _redirect_with("index", add_error=not_found_message(track, artist, add_suggestions),
                                  add_tips=not add_suggestions)

        mb_artist = mb_info.get("artist") or artist

        library = load_library(visitor_id)
        mb_title = mb_info.get("title", track).lower()
        for t in library:
            stored_title = t["title"].lower()
            stored_artist = t["artist"].lower()
            if stored_artist == mb_artist.lower() and (stored_title == track.lower() or stored_title == mb_title):
                return _redirect_with("index", add_warning=f'"{t["title"]}" by {t["artist"]} is already in your library.')

        preview_url = get_preview_url(mb_info.get("title", track), mb_artist)
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
            return _redirect_with(
                "index",
                add_success=add_success,
                add_warning=f"BPM/key/energy confidence for this track is {confidence}{suffix}. Worth double-checking manually.",
            )
        return _redirect_with("index", add_success=add_success)

    except SearchUnavailable:
        logging.exception("Song lookup services unavailable")
        return _redirect_with("index", add_error=SEARCH_UNAVAILABLE_MSG)
    except Exception:
        logging.exception("Add track error")
        return _redirect_with("index", add_error="Something went wrong adding that track. Please try again.")


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
        return _redirect_with(target, add_error=f'Couldn\'t find "{title}" by {artist} to remove.')

    save_library(visitor_id, filtered)
    return _redirect_with(target, add_success=f'Removed "{title}" by {artist} from your library.')


@app.route("/library")
def library_page():
    visitor_id = get_visitor_id()
    library = list(reversed(load_library(visitor_id)))
    msgs = _pop_messages()
    add_success, add_error = msgs["add_success"], msgs["add_error"]
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
