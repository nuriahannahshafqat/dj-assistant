import subprocess

import numpy as np
import pytest

from audio_analysis import SAMPLE_RATE as SR, analyse_audio, decode_audio, to_camelot

# Standard Camelot wheel, (pitch class, mode) -> key.
CAMELOT_TABLE = {
    (0, "major"): "8B", (7, "major"): "9B", (2, "major"): "10B", (9, "major"): "11B",
    (4, "major"): "12B", (11, "major"): "1B", (6, "major"): "2B", (1, "major"): "3B",
    (8, "major"): "4B", (3, "major"): "5B", (10, "major"): "6B", (5, "major"): "7B",
    (9, "minor"): "8A", (4, "minor"): "9A", (11, "minor"): "10A", (6, "minor"): "11A",
    (1, "minor"): "12A", (8, "minor"): "1A", (3, "minor"): "2A", (10, "minor"): "3A",
    (5, "minor"): "4A", (0, "minor"): "5A", (7, "minor"): "6A", (2, "minor"): "7A",
}


def test_to_camelot_matches_the_wheel():
    for (pc, mode), expected in CAMELOT_TABLE.items():
        assert to_camelot(pc, mode) == expected, (pc, mode)


def _synth(bpm, notes_hz, seconds=30):
    """Kick drum on every beat plus a sustained chord."""
    t = np.arange(int(SR * seconds)) / SR
    pad = sum(np.sin(2 * np.pi * f * t) for f in notes_hz) * 0.15
    kick = np.zeros_like(t)
    for start in np.arange(0, seconds, 60 / bpm):
        i, n = int(start * SR), int(0.12 * SR)
        seg = np.sin(2 * np.pi * 60 * np.arange(n) / SR) * np.exp(-np.linspace(0, 8, n))
        kick[i:i + n] += seg[: len(kick[i:i + n])]
    return (pad + kick * 0.8).astype(np.float32)


def _as_aac(y):
    """Round-trip through AAC so the test exercises the same decode path as
    real iTunes previews."""
    return subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "f32le", "-ar", str(SR), "-ac", "1",
         "-i", "pipe:0", "-c:a", "aac", "-f", "adts", "pipe:1"],
        input=y.tobytes(), capture_output=True, check=True,
    ).stdout


@pytest.mark.slow
@pytest.mark.parametrize("bpm,notes,key", [
    (100, [110, 220, 261.63, 329.63], "8A"),       # A minor
    (124, [130.81, 261.63, 329.63, 392.0], "8B"),  # C major
    (140, [87.31, 174.61, 207.65, 261.63], "4A"),  # F minor
])
def test_detects_known_bpm_and_key(bpm, notes, key):
    result = analyse_audio(decode_audio(_as_aac(_synth(bpm, notes))))
    assert abs(result["bpm"] - bpm) <= 1
    assert result["key"] == key
