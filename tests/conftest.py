import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("SECRET_KEY", "test-secret")


def track(title, bpm, key, energy=5, artist="Test Artist", genre="R&B"):
    return {"title": title, "artist": artist, "bpm": bpm, "key": key,
            "energy": energy, "genre": genre}


@pytest.fixture
def library():
    return [
        track("Same Key", 98, "8A", 6),
        track("Adjacent", 100, "9A", 6),
        track("Relative", 97, "8B", 7),
        track("Half Time", 49, "8A", 5),
        track("Clash Key", 98, "2A", 6),
        track("Too Fast", 125, "8A", 6),
        track("Chain Next", 101, "10A", 7),
        track("Chain Next 2", 102, "11A", 7),
    ]
