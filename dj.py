import argparse
import json
import os

import requests
from dotenv import load_dotenv

load_dotenv()

MUSICBRAINZ_BASE = "https://musicbrainz.org/ws/2"
MB_HEADERS = {"User-Agent": "dj-assistant/1.0 (nuriashafqat711@gmail.com)"}


def search_musicbrainz(track: str, artist: str) -> dict:
    params = {
        "query": f'recording:"{track}" AND artist:"{artist}"',
        "fmt": "json",
        "limit": 1,
    }
    resp = requests.get(f"{MUSICBRAINZ_BASE}/recording", params=params, headers=MB_HEADERS)
    resp.raise_for_status()
    recordings = resp.json().get("recordings", [])
    if not recordings:
        return {}

    rec_id = recordings[0]["id"]
    detail = requests.get(
        f"{MUSICBRAINZ_BASE}/recording/{rec_id}",
        params={"inc": "tags+genres", "fmt": "json"},
        headers=MB_HEADERS,
    )
    detail.raise_for_status()
    data = detail.json()
    return {
        "title": data.get("title", track),
        "mbid": rec_id,
        "tags": [t["name"] for t in data.get("tags", [])],
        "genres": [g["name"] for g in data.get("genres", [])],
    }


def load_library(path: str = "library.json") -> list:
    with open(path) as f:
        return json.load(f)


def recommend_next_track(track: str, artist: str, mb_info: dict, library: list) -> str:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set")

    prompt = f"""You are an expert DJ assistant. Recommend the single best next track to play after the current one.

Current track:
- Title: {track}
- Artist: {artist}
- MusicBrainz tags: {", ".join(mb_info.get("tags", [])) or "none found"}
- Genres: {", ".join(mb_info.get("genres", [])) or "none found"}

DJ Library (title / artist / bpm / key in Camelot notation):
{json.dumps(library, indent=2)}

Selection criteria (in priority order):
1. Camelot Wheel key compatibility — same key, ±1 step, or relative major/minor (+3 or -3)
2. BPM compatibility — ideally within ±3 BPM; double/half-time also works
3. Energy and vibe — consider genre tags to maintain or build energy naturally

Pick ONE track and explain: which Camelot keys are involved, the BPM delta, and why the energy fits."""

    resp = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": "claude-sonnet-4-6",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": prompt}],
        },
    )
    if not resp.ok:
        print("API error:", resp.status_code, resp.text)
        resp.raise_for_status()
    return resp.json()["content"][0]["text"]


def main():
    parser = argparse.ArgumentParser(description="DJ next-track recommendation")
    parser.add_argument("--track", required=True, help="Current track title")
    parser.add_argument("--artist", required=True, help="Current track artist")
    args = parser.parse_args()

    print(f"Searching MusicBrainz for '{args.track}' by {args.artist}...")
    mb_info = search_musicbrainz(args.track, args.artist)
    if mb_info:
        print(f"Found MBID {mb_info['mbid']} | tags: {', '.join(mb_info['tags']) or 'none'}")
    else:
        print("No MusicBrainz match found — proceeding without tags.")

    library = load_library()
    print(f"Loaded {len(library)} tracks from library.json\n")

    result = recommend_next_track(args.track, args.artist, mb_info, library)
    print("Recommendation\n" + "-" * 40)
    print(result)


if __name__ == "__main__":
    main()
