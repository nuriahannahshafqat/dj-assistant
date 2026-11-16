# What changed

## 1. BPM and key are measured from audio, not guessed
`audio_analysis.py` downloads each track's 30-second iTunes preview, decodes it
with ffmpeg and analyses it with librosa:

- **Tempo:** onset-strength tempogram for a first estimate, then beat tracking
  and a line fit through the beat times for a precise BPM. Confidence comes from
  checking that both halves of the clip agree.
- **Key:** chromagram of the harmonic part of the signal, correlated against all
  24 Krumhansl-Kessler key profiles, then converted to Camelot notation.
  Confidence is the margin between the best and second-best key.

The LLM still supplies energy and genre (subjective), and is the fallback for
BPM/key if a track has no preview. Its guesses are kept as `llm_bpm` / `llm_key`.

## 2. Two-stage recommender
`dj_logic.py` scores every library track against the current one in plain
Python (Camelot key relationship, BPM incl. half/double time, energy flow) and
keeps the top 15. Claude then chooses from that shortlist **by ID** and writes
the explanation. The answer is validated: unknown or duplicate IDs are replaced
by the best-scoring real track, chain steps can't clash in key, and all BPM/key
numbers shown come from the library. If the API is down, a rule-based
recommendation is returned instead of an error.

Also fixed: the old prompt described ±3 on the Camelot wheel as "relative
major/minor". Relative major/minor is the same number with A↔B (8A ↔ 8B).

## 3. Smaller fixes
- 60s timeout on the Claude API call (it previously had none)
- Library file writes are atomic and locked, so concurrent requests can't corrupt it
- The current track's BPM/key is now measured too (before, only its name was sent)

## Running it
```bash
pip install -r requirements.txt     # also needs ffmpeg installed (brew install ffmpeg)
python -m pytest                    # 22 tests, ~12s
python scripts/analyse_library.py library.json            # LLM vs audio accuracy report
python scripts/analyse_library.py library.json --write    # replace guesses with measured values
```

## Deploying on Railway
The `Dockerfile` installs ffmpeg and runs the app with gunicorn. Railway uses a
Dockerfile automatically when one is in the repo root. Keep `SECRET_KEY` and
`ANTHROPIC_API_KEY` set as variables.

Note: Railway's filesystem resets on each deploy, so `visitor_libraries.json`
is wiped. To keep user libraries, attach a Railway volume and point
`VISITOR_LIBRARIES_PATH` at it, or move storage to Railway Postgres.
