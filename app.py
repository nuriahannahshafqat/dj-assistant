import json
import logging
import os
import re
import secrets
from datetime import timedelta

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv
from flask import Flask, redirect, render_template, request, session, url_for

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


def search_musicbrainz(track: str, artist: str) -> tuple:
    """Returns (best_match_dict, suggestions_list).

    Tries a strict quoted query first; falls back to an unquoted fuzzy query
    to populate suggestions when the strict search returns nothing or low confidence.
    """
    strict = _mb_search(f'recording:"{track}" AND artist:"{artist}"')
    if strict and strict[0].get("score", 0) >= 85:
        return _mb_fetch_details(strict[0]["id"], track, artist), []

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


def _load_all_visitor_libraries() -> dict:
    if not os.path.exists(VISITOR_LIBRARIES_PATH):
        return {}
    with open(VISITOR_LIBRARIES_PATH) as f:
        return json.load(f)


def _save_all_visitor_libraries(data: dict) -> None:
    with open(VISITOR_LIBRARIES_PATH, "w") as f:
        json.dump(data, f, indent=2)


def load_library(visitor_id: str) -> list:
    data = _load_all_visitor_libraries()
    if visitor_id not in data:
        with open(SEED_LIBRARY_PATH) as f:
            data[visitor_id] = json.load(f)
        _save_all_visitor_libraries(data)
    return data[visitor_id]


def save_library(visitor_id: str, library: list) -> None:
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


def recommend_next_track(track: str, artist: str, mb_info: dict, library: list) -> dict:
    prompt = f"""You are an expert DJ assistant. Recommend the best next track to play after the current one.

Current track:
- Title: {track}
- Artist: {artist}
- MusicBrainz tags: {_safe_tags(mb_info)}
- Genres: {", ".join(g[:50] for g in mb_info.get("genres", [])[:10]) or "none found"}

DJ Library (title / artist / bpm / key in Camelot notation):
{json.dumps(library, indent=2)}

Selection criteria (in priority order):
1. Camelot Wheel key compatibility — same key, ±1 step, or relative major/minor (+3 or -3)
2. BPM compatibility — ideally within ±3 BPM; double/half-time also works
3. Energy and vibe — consider genre tags to maintain or build energy naturally

Respond with ONLY a JSON object in this exact format, no other text:
{{
  "recommended_track": "track title",
  "recommended_artist": "artist name",
  "bpm_compatibility": "2-3 sentences on BPM delta and whether it works",
  "energy_vibes": "2-3 sentences on energy level, mood, and genre fit",
  "summary": "1 punchy paragraph tying it all together — why this is the perfect next track",
  "alternatives": [
    {{"title": "track title", "artist": "artist name", "reason": "one sentence on why this also works"}},
    {{"title": "track title", "artist": "artist name", "reason": "one sentence on why this also works"}},
    {{"title": "track title", "artist": "artist name", "reason": "one sentence on why this also works"}}
  ],
  "chain": [
    {{"title": "track title", "artist": "artist name", "bpm": 0, "key": "0A", "reason": "one sentence on the transition from the recommended track"}},
    {{"title": "track title", "artist": "artist name", "bpm": 0, "key": "0A", "reason": "one sentence on the transition from track 1"}},
    {{"title": "track title", "artist": "artist name", "bpm": 0, "key": "0A", "reason": "one sentence on the transition from track 2"}}
  ]
}}

For alternatives: pick 3 other good options from the library (not the recommended track).
For chain: pick 3 tracks from the library that should follow the recommended track in order, building a coherent set. Do not repeat the recommended track or any alternative."""

    return parse_json_response(call_claude(prompt, max_tokens=2048))


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
                    rec = recommend_next_track(track, mb_artist, mb_info, load_library(visitor_id))
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

        metadata = get_track_metadata(track, mb_artist, mb_info)
        confidence = metadata.get("confidence") or "low"
        preview_url = get_preview_url(track, mb_artist)

        library.append({
            "title": mb_info.get("title", track),
            "artist": mb_artist,
            "bpm": metadata.get("bpm") or 0,
            "key": metadata.get("key") or "1A",
            "genre": metadata.get("genre") or "Unknown",
            "energy": metadata.get("energy") or 5,
            "confidence": confidence,
            "confidence_note": metadata.get("confidence_note") or "",
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
