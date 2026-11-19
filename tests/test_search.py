import pytest
import requests

import app


@pytest.fixture(autouse=True)
def musicbrainz_up():
    app._mb_down_until[0] = 0.0
    yield
    app._mb_down_until[0] = 0.0


def _mb_outage(*a, **k):
    raise requests.exceptions.RetryError("too many 502 error responses")


ITUNES_TIMELESS = [
    {"trackName": "Timeless (with Playboi Carti)", "artistName": "The Weeknd & Playboi Carti", "primaryGenreName": "R&B/Soul"},
    {"trackName": "Timeless", "artistName": "Some Cover Band"},
]


def test_musicbrainz_outage_falls_back_to_itunes(monkeypatch):
    monkeypatch.setattr(app, "_mb_search", _mb_outage)
    monkeypatch.setattr(app, "_itunes_search", lambda term, limit: ITUNES_TIMELESS)
    found, suggestions = app.search_musicbrainz("timeless", "the weeknd")
    assert found["title"].startswith("Timeless") and "Weeknd" in found["artist"]
    assert found["genres"] == ["R&B/Soul"] and suggestions == []


def test_after_an_outage_musicbrainz_is_skipped_for_a_while(monkeypatch):
    calls = []
    def mb(*a, **k):
        calls.append(1)
        return _mb_outage()
    monkeypatch.setattr(app, "_mb_search", mb)
    monkeypatch.setattr(app, "_itunes_search", lambda term, limit: ITUNES_TIMELESS)
    app.search_musicbrainz("timeless", "the weeknd")
    app.search_musicbrainz("timeless", "the weeknd")
    assert len(calls) == 1          # second search went straight to iTunes


def test_both_services_down_raises_search_unavailable(monkeypatch):
    monkeypatch.setattr(app, "_mb_search", _mb_outage)
    def itunes_down(term, limit):
        raise requests.exceptions.ConnectionError("down")
    monkeypatch.setattr(app, "_itunes_search", itunes_down)
    with pytest.raises(app.SearchUnavailable):
        app.search_musicbrainz("timeless", "the weeknd")


def test_fuzzy_terms_allow_typos_and_strip_query_syntax():
    assert app._fuzzy_terms("save ur tears") == "save~ ur~ tears~"
    # Quotes/brackets can't break out of the query.
    assert app._fuzzy_terms('kill "bill" (live)') == "kill~ bill~ live~"


# Real MusicBrainz hits for the old, malformed fallback query for
# "save ur tears" by The Weeknd: none of them is the song.
OLD_JUNK = [
    ("Save your tears (The Weeknd cover)", "Marco Montagnini"),
    ("Save Ur", "Hndacvc"),
    ("Ur Tears", "LIL SAD"),
    ("Save Your Tears (8-Bit The Weeknd Emulation)", "8-Bit Arcade"),
    ("Save Ur Breath", "Drongomala"),
    ("Weeknd", "Nocturnal Me"),
]


def test_junk_hits_are_not_suggested():
    assert app._rank_suggestions(OLD_JUNK, "save ur tears", "the weeknd") == []


def test_real_song_is_suggested_once_from_duplicates():
    hits = OLD_JUNK + [("Save Your Tears", "The Weeknd")] * 5 + [
        ("Save Your Tears (Remix)", "The Weeknd & Ariana Grande"),
        ("The Morning / Save Your Tears", "The Weeknd"),
    ]
    assert app._rank_suggestions(hits, "save ur tears", "the weeknd") == [
        {"title": "Save Your Tears", "artist": "The Weeknd"}
    ]


def test_artist_typo_still_suggests():
    hits = [("Save Your Tears", "The Weeknd")]
    assert app._rank_suggestions(hits, "save your tears", "the weekend")


def test_closest_titles_come_first():
    hits = [("Blinding Lights", "The Weeknd"), ("Save Your Tears", "The Weeknd"), ("Save Me", "The Weeknd")]
    titles = [s["title"] for s in app._rank_suggestions(hits, "save ur tears", "weeknd")]
    assert titles[0] == "Save Your Tears"
    assert "Blinding Lights" not in titles


def test_versions_and_mistagged_artists_collapse_into_one_suggestion():
    # Real MusicBrainz hits for "blindin lights" / "kil bill".
    hits = [
        ("Blinding Lights", "The Weekend"),
        ("Blinding Lights (acapella)", "The Weeknd"),
        ("Blinding Lights", "The Weeknd"),
        ("Blinding Lights (BAN DELLO edit)", "The Weeknd"),
    ]
    assert app._rank_suggestions(hits, "blindin lights", "weeknd") == [
        {"title": "Blinding Lights", "artist": "The Weeknd"}
    ]
    hits = [("Kill Bill (vocals)", "SZA"), ("kill bill [nightcore]", "SZA"), ("Kill Bill", "SZA")]
    assert app._rank_suggestions(hits, "kil bill", "sza") == [{"title": "Kill Bill", "artist": "SZA"}]


def test_falls_back_to_itunes_when_musicbrainz_has_nothing(monkeypatch):
    monkeypatch.setattr(app, "_mb_search", lambda q, limit=5: [])
    monkeypatch.setattr(app, "_itunes_search", lambda term, limit: [
        {"trackName": "Save Your Tears", "artistName": "The Weeknd"},
        {"trackName": "Save Your Tears", "artistName": "KIDZ BOP Kids"},
    ])
    found, suggestions = app.search_musicbrainz("save ur tears", "the weeknd")
    assert found == {}
    assert suggestions == [{"title": "Save Your Tears", "artist": "The Weeknd"}]


def test_messages_explain_what_to_do():
    with_options = app.not_found_message("save ur tears", "the weeknd", [{"title": "x", "artist": "y"}])
    assert "Did you mean" in with_options
    none = app.not_found_message("asdfgh", "nobody", [])
    assert 'Unable to find "asdfgh"' in none
    assert any("ur" in tip for tip in app.SEARCH_TIPS) and any("iTunes" in tip for tip in app.SEARCH_TIPS)
