import json

import pytest

import app as dj_app
import dj_logic
from conftest import track


@pytest.fixture
def current():
    return track("Now Playing", 98, "8A", 6, artist="Someone Else")


def _titles(rec):
    return (
        [rec["recommended_track"]]
        + [a["title"] for a in rec["alternatives"]]
        + [c["title"] for c in rec["chain"]]
    )


def test_valid_llm_answer_is_used(monkeypatch, current, library):
    def fake_claude(prompt, max_tokens=512):
        return json.dumps({
            "recommended_id": "t2",
            "bpm_compatibility": "LLM bpm text",
            "energy_vibes": "LLM vibe text",
            "summary": "LLM summary",
            "alternatives": [{"id": "t1", "reason": "a"}, {"id": "t3", "reason": "b"}, {"id": "t4", "reason": "c"}],
            "chain": [{"id": "t5", "reason": "x"}],
        })
    monkeypatch.setattr(dj_app, "call_claude", fake_claude)

    rec = dj_app.recommend_next_track(current, {}, library)
    assert rec["summary"] == "LLM summary"
    assert len(rec["alternatives"]) == 3
    assert 1 <= len(rec["chain"]) <= 3
    prev = {"key": next(t["key"] for t in library if t["title"] == rec["recommended_track"])}
    for step in rec["chain"]:
        assert dj_logic.key_relation(prev["key"], step["key"])[0] > 0, "chain must not clash"
        prev = step


def test_hallucinated_ids_are_replaced_with_real_tracks(monkeypatch, current, library):
    def fake_claude(prompt, max_tokens=512):
        return json.dumps({
            "recommended_id": "t999",  # doesn't exist
            "summary": "Play 'A Song I Made Up'",
            "alternatives": [{"id": "t1"}, {"id": "t1"}, {"id": "nope"}],  # duplicate + bogus
            "chain": [{"id": "t42"}],
        })
    monkeypatch.setattr(dj_app, "call_claude", fake_claude)

    rec = dj_app.recommend_next_track(current, {}, library)
    library_titles = {t["title"] for t in library}
    titles = _titles(rec)
    assert all(t in library_titles for t in titles)
    assert len(titles) == len(set(titles)), "no track should appear twice"
    assert "Made Up" not in rec["summary"], "text for a replaced pick must not be shown"
    assert rec["llm_corrections"] > 0


def test_numbers_come_from_library_not_model(monkeypatch, current, library):
    def fake_claude(prompt, max_tokens=512):
        return json.dumps({"recommended_id": "t1", "alternatives": [], "chain": []})
    monkeypatch.setattr(dj_app, "call_claude", fake_claude)

    rec = dj_app.recommend_next_track(current, {}, library)
    by_title = {t["title"]: t for t in library}
    for item in rec["alternatives"] + rec["chain"]:
        assert item["bpm"] == by_title[item["title"]]["bpm"]
        assert item["key"] == by_title[item["title"]]["key"]


def test_llm_outage_falls_back_to_rules(monkeypatch, current, library):
    def broken(prompt, max_tokens=512):
        raise RuntimeError("API down")
    monkeypatch.setattr(dj_app, "call_claude", broken)

    rec = dj_app.recommend_next_track(current, {}, library)
    assert rec["recommended_track"] in {"Same Key", "Adjacent", "Relative", "Half Time"}
    assert rec["summary"]
    assert len(rec["alternatives"]) == 3


def test_clashing_key_never_recommended_first(monkeypatch, current, library):
    monkeypatch.setattr(dj_app, "call_claude", lambda *a, **k: (_ for _ in ()).throw(RuntimeError()))
    rec = dj_app.recommend_next_track(current, {}, library)
    assert rec["recommended_track"] != "Clash Key"


def test_prompt_only_contains_shortlisted_tracks(monkeypatch, current, library):
    seen = {}

    def capture(prompt, max_tokens=512):
        seen["prompt"] = prompt
        raise RuntimeError("stop")
    monkeypatch.setattr(dj_app, "SHORTLIST_SIZE", 3)
    monkeypatch.setattr(dj_app, "call_claude", capture)
    dj_app.recommend_next_track(current, {}, library)
    assert "t3:" in seen["prompt"] and "t4:" not in seen["prompt"]


def test_storage_roundtrip_is_atomic(tmp_path, monkeypatch):
    seed = tmp_path / "library.json"
    seed.write_text(json.dumps([track("Seed", 100, "8A")]))
    monkeypatch.setattr(dj_app, "SEED_LIBRARY_PATH", str(seed))
    monkeypatch.setattr(dj_app, "VISITOR_LIBRARIES_PATH", str(tmp_path / "visitors.json"))

    lib = dj_app.load_library("v1")
    assert lib[0]["title"] == "Seed"
    lib.append(track("Added", 101, "9A"))
    dj_app.save_library("v1", lib)
    assert [t["title"] for t in dj_app.load_library("v1")] == ["Seed", "Added"]
    assert not list(tmp_path.glob("*.tmp")), "temp files should be cleaned up"


def test_musicbrainz_prefers_original_over_instrumental():
    results = [
        {"id": "inst", "title": "Shut It Down (instrumental)", "score": 100},
        {"id": "orig", "title": "Shut It Down", "score": 100},
    ]
    assert dj_app._pick_recording(results, "Shut It Down")["id"] == "orig"
    # ...unless the user actually typed "instrumental"
    assert dj_app._pick_recording(results, "Shut It Down instrumental")["id"] == "inst"
    assert dj_app._pick_recording([{"id": "x", "title": "A", "score": 50}], "A") is None


def _id_of(current, library, title):
    ranked = dj_logic.shortlist(current, library, limit=dj_app.SHORTLIST_SIZE)
    return next(f"t{n + 1}" for n, (_, t, _) in enumerate(ranked) if t["title"] == title)


def test_key_clash_alternative_is_swapped_for_a_compatible_one(monkeypatch, current, library):
    clash_id = _id_of(current, library, "Clash Key")
    rec_id = _id_of(current, library, "Adjacent")
    def fake_claude(prompt, max_tokens=512):
        return json.dumps({
            "recommended_id": rec_id, "summary": "s",
            "alternatives": [{"id": clash_id, "reason": "same artist continuity"}],
            "chain": [],
        })
    monkeypatch.setattr(dj_app, "call_claude", fake_claude)
    rec = dj_app.recommend_next_track(current, {}, library)
    alt_titles = [a["title"] for a in rec["alternatives"]]
    assert "Clash Key" not in alt_titles
    assert "same artist continuity" not in [a["reason"] for a in rec["alternatives"]]


def test_chain_never_jumps_more_than_comfortable_pitch_range():
    start = track("Start", 100, "8A", 6)
    lib = [start, track("Big Jump", 115, "8A", 6), track("Close", 103, "9A", 6)]
    chain = dj_logic.build_chain(start, lib, exclude_indices={0}, length=3)
    assert [t["title"] for _, t, _ in chain] == ["Close"]
