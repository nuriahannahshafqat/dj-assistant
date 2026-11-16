"""Measures BPM and musical key from a track's 30-second audio preview.

Pipeline: download preview -> decode with ffmpeg to mono PCM -> librosa
onset envelope for tempo, chroma + Krumhansl-Kessler key profiles for key ->
convert to Camelot notation.

Requires the `ffmpeg` binary on the server (see Dockerfile).
"""

import logging
import subprocess

import numpy as np
import requests

SAMPLE_RATE = 22050
MAX_PREVIEW_BYTES = 5 * 1024 * 1024
DOWNLOAD_TIMEOUT = 15

PITCH_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]

# Krumhansl-Kessler key profiles: how strongly each scale degree is
# perceived as belonging to a major / minor key (index 0 = tonic).
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


class AudioAnalysisError(Exception):
    pass


def to_camelot(pitch_class, mode):
    """(pitch class 0-11 with C=0, 'major'|'minor') -> Camelot key, e.g. (9, 'minor') -> '8A'.

    Moving up a fifth (7 semitones) is one step clockwise on the wheel, and
    C major sits at 8B. A minor key shares a number with its relative major
    (3 semitones above its tonic).
    """
    if mode == "minor":
        pitch_class = (pitch_class + 3) % 12
        letter = "A"
    else:
        letter = "B"
    number = (7 * pitch_class + 7) % 12 + 1
    return f"{number}{letter}"


def decode_audio(data: bytes) -> np.ndarray:
    """Decodes any format ffmpeg understands (iTunes previews are AAC/.m4a)."""
    try:
        proc = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", "pipe:0", "-f", "f32le",
             "-ac", "1", "-ar", str(SAMPLE_RATE), "pipe:1"],
            input=data,
            capture_output=True,
            timeout=30,
            check=True,
        )
    except FileNotFoundError as e:
        raise AudioAnalysisError("ffmpeg is not installed") from e
    except subprocess.CalledProcessError as e:
        raise AudioAnalysisError(f"ffmpeg failed: {e.stderr.decode(errors='ignore')[:200]}") from e
    except subprocess.TimeoutExpired as e:
        raise AudioAnalysisError("ffmpeg timed out") from e

    y = np.frombuffer(proc.stdout, dtype=np.float32)
    if y.size < SAMPLE_RATE * 5:
        raise AudioAnalysisError("decoded audio is shorter than 5 seconds")
    return y


def download_preview(url: str, session=None) -> bytes:
    http = session or requests
    resp = http.get(url, timeout=DOWNLOAD_TIMEOUT, stream=True)
    resp.raise_for_status()
    chunks, total = [], 0
    for chunk in resp.iter_content(64 * 1024):
        total += len(chunk)
        if total > MAX_PREVIEW_BYTES:
            raise AudioAnalysisError("preview file is unexpectedly large")
        chunks.append(chunk)
    return b"".join(chunks)


def estimate_tempo(y: np.ndarray, sr: int = SAMPLE_RATE):
    """Returns (bpm, confidence).

    Confidence comes from a stability check: the tempo is estimated on each
    half of the clip separately; if both halves agree (allowing for half/double
    time) the estimate is trustworthy.
    """
    import librosa

    hop = 256  # finer time resolution than librosa's default (512) -> more precise BPM

    def _tempo(sig):
        onset_env = librosa.onset.onset_strength(y=sig, sr=sr, hop_length=hop)
        rough = float(np.atleast_1d(
            librosa.feature.tempo(onset_envelope=onset_env, sr=sr, hop_length=hop))[0])
        # Refine: track individual beats near the rough tempo and fit a line
        # through their times. The slope is the beat period, averaged over the
        # whole clip, so it isn't stuck to tempogram bins or frame boundaries.
        _, beats = librosa.beat.beat_track(
            onset_envelope=onset_env, sr=sr, hop_length=hop, start_bpm=rough, units="time")
        if len(beats) >= 8:
            period = np.polyfit(np.arange(len(beats)), beats, 1)[0]
            refined = 60.0 / float(period)
            if abs(refined - rough) / rough < 0.06:
                return refined
        return rough

    bpm = _tempo(y)
    half = y.size // 2
    first, second = _tempo(y[:half]), _tempo(y[half:])

    def _close(a, b):
        return any(abs(a - b * m) / a <= 0.02 for m in (1, 2, 0.5))

    if _close(first, second) and _close(bpm, first):
        confidence = "high"
    elif _close(first, second) or _close(bpm, first) or _close(bpm, second):
        confidence = "medium"
    else:
        confidence = "low"
    return round(bpm), confidence


def estimate_key(y: np.ndarray, sr: int = SAMPLE_RATE):
    """Returns (camelot_key, readable_name, confidence).

    Averages a chromagram over the clip and correlates it against all 24
    rotated major/minor key profiles. Confidence reflects how clearly the best
    key beats the runner-up (relative major/minor is a common near-tie).
    """
    import librosa

    # Harmonic part only, so drums don't smear the pitch content.
    harmonic = librosa.effects.harmonic(y)
    chroma = librosa.feature.chroma_cqt(y=harmonic, sr=sr).mean(axis=1)
    if not np.any(chroma):
        raise AudioAnalysisError("no pitched content detected")

    scores = []
    for pc in range(12):
        for mode, profile in (("major", MAJOR_PROFILE), ("minor", MINOR_PROFILE)):
            r = np.corrcoef(chroma, np.roll(profile, pc))[0, 1]
            scores.append((r, pc, mode))
    scores.sort(reverse=True)

    best_r, pc, mode = scores[0]
    margin = best_r - scores[1][0]
    confidence = "high" if margin >= 0.08 else "medium" if margin >= 0.03 else "low"
    name = f"{PITCH_NAMES[pc]} {'minor' if mode == 'minor' else 'major'}"
    return to_camelot(pc, mode), name, confidence


def analyse_audio(y: np.ndarray, sr: int = SAMPLE_RATE) -> dict:
    bpm, bpm_conf = estimate_tempo(y, sr)
    key, key_name, key_conf = estimate_key(y, sr)
    order = {"low": 0, "medium": 1, "high": 2}
    overall = min(bpm_conf, key_conf, key=order.get)
    return {
        "bpm": bpm,
        "key": key,
        "key_name": key_name,
        "bpm_confidence": bpm_conf,
        "key_confidence": key_conf,
        "confidence": overall,
    }


def analyse_preview(url: str, session=None) -> dict:
    """Download + decode + analyse. Raises AudioAnalysisError on failure."""
    try:
        data = download_preview(url, session=session)
    except requests.RequestException as e:
        raise AudioAnalysisError(f"could not download preview: {e}") from e
    result = analyse_audio(decode_audio(data))
    logging.info("Audio analysis for %s: %s", url, result)
    return result
