"""Measures every track in a library from its audio preview and compares the
result with the LLM's BPM/key guesses.

Usage:
    python scripts/analyse_library.py library.json
    python scripts/analyse_library.py library.json --write   # also save measured values

Prints accuracy figures (useful for the README / CV) and writes a CSV of
every track to analysis_report.csv.

With --write, the library file is updated in place (a .bak copy is kept):
`bpm`/`key` become the measured values and the old guesses are preserved as
`llm_bpm`/`llm_key`.
"""

import argparse
import csv
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dj_logic  # noqa: E402
from audio_analysis import AudioAnalysisError, analyse_preview  # noqa: E402


def bpm_matches(guess, measured, tolerance=0.03):
    """Same tempo within tolerance, counting half/double time as a match."""
    try:
        g, m = float(guess), float(measured)
    except (TypeError, ValueError):
        return False, False
    exact = abs(g - m) / m <= tolerance
    octave = any(abs(g - m * f) / (m * f) <= tolerance for f in (0.5, 2))
    return exact, exact or octave


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("library", help="library.json (a list of tracks)")
    parser.add_argument("--write", action="store_true", help="save measured BPM/key back to the file")
    parser.add_argument("--report", default="analysis_report.csv")
    args = parser.parse_args()

    with open(args.library) as f:
        library = json.load(f)
    if not isinstance(library, list):
        sys.exit("Expected a JSON list of tracks (e.g. library.json).")

    rows = []
    for t in library:
        title, artist = t.get("title"), t.get("artist")
        guess_bpm = t.get("llm_bpm", t.get("bpm"))
        guess_key = t.get("llm_key", t.get("key"))
        url = t.get("preview_url")
        if not url:
            print(f"  skip  {title} — {artist}: no preview")
            continue
        try:
            audio = analyse_preview(url)
        except AudioAnalysisError as e:
            print(f"  fail  {title} — {artist}: {e}")
            continue

        bpm_exact, bpm_octave = bpm_matches(guess_bpm, audio["bpm"])
        key_score, key_label = dj_logic.key_relation(guess_key, audio["key"])
        rows.append({
            "title": title, "artist": artist,
            "llm_bpm": guess_bpm, "audio_bpm": audio["bpm"], "bpm_confidence": audio["bpm_confidence"],
            "bpm_match": bpm_exact, "bpm_match_incl_half_double": bpm_octave,
            "llm_key": guess_key, "audio_key": audio["key"], "audio_key_name": audio["key_name"],
            "key_confidence": audio["key_confidence"],
            "key_exact": key_label == "same key", "key_relation": key_label,
        })
        print(f"  ok    {title} — {artist}: LLM {guess_bpm} BPM/{guess_key} | "
              f"audio {audio['bpm']} BPM/{audio['key']} ({key_label})")

        if args.write:
            t.setdefault("llm_bpm", guess_bpm)
            t.setdefault("llm_key", guess_key)
            t.update({"bpm": audio["bpm"], "key": audio["key"], "source": "audio",
                      "confidence": audio["confidence"],
                      "confidence_note": f"Measured from the audio preview: {audio['key_name']}."})

    if not rows:
        sys.exit("No tracks could be analysed.")

    n = len(rows)
    pct = lambda k: 100 * sum(r[k] for r in rows) / n  # noqa: E731
    compatible = 100 * sum(r["key_relation"] in ("same key", "relative major/minor", "adjacent key") for r in rows) / n

    print(f"\nAnalysed {n} of {len(library)} tracks")
    print(f"  LLM BPM within ±3% of measured:            {pct('bpm_match'):.0f}%")
    print(f"  ...counting half/double time as correct:   {pct('bpm_match_incl_half_double'):.0f}%")
    print(f"  LLM key exactly right:                     {pct('key_exact'):.0f}%")
    print(f"  LLM key at least mix-compatible:           {compatible:.0f}%")

    with open(args.report, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nPer-track results: {args.report}")

    if args.write:
        shutil.copy(args.library, args.library + ".bak")
        with open(args.library, "w") as f:
            json.dump(library, f, indent=2, ensure_ascii=False)
        print(f"Updated {args.library} (backup at {args.library}.bak)")


if __name__ == "__main__":
    main()
