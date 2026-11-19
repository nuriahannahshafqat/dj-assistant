import pytest

import app


@pytest.fixture
def client(monkeypatch):
    libs = {}
    monkeypatch.setattr(app, "load_library", lambda vid: list(libs.get(vid, [])))
    monkeypatch.setattr(app, "save_library", lambda vid, lib: libs.__setitem__(vid, list(lib)))
    monkeypatch.setattr(app, "search_musicbrainz", lambda track, artist: (
        ({"title": "Money Trees", "artist": "Kendrick Lamar"}, []) if track == "Money Trees" else ({}, [])
    ))
    monkeypatch.setattr(app, "get_preview_url", lambda track, artist: None)
    monkeypatch.setattr(app, "build_track_metadata", lambda *a: {
        "bpm": 72, "key": "4A", "energy": 5, "genre": "Hip-Hop", "confidence": "high", "confidence_note": ""})
    app.app.config["TESTING"] = True
    return app.app.test_client()


ADDED = "added to your library"


def test_add_message_shows_once_and_not_in_url(client):
    resp = client.post("/add", data={"track": "Money Trees", "artist": "Kendrick Lamar"})
    assert resp.status_code == 302
    assert resp.headers["Location"] == "/"          # nothing in the query string

    assert ADDED in client.get("/").get_data(as_text=True)
    assert ADDED not in client.get("/").get_data(as_text=True)


def test_searching_after_adding_doesnt_bring_message_back(client):
    client.post("/add", data={"track": "Money Trees", "artist": "Kendrick Lamar"})
    client.get("/")
    page = client.post("/", data={"track": "zzz", "artist": "nobody"}).get_data(as_text=True)
    assert ADDED not in page
    assert "Unable to find &#34;zzz&#34;" in page      # Jinja escapes the quotes
    assert 'class="toast-tips"' in page and "actually released that song" in page


def test_outage_shows_a_clear_message(client, monkeypatch):
    def down(track, artist):
        raise app.SearchUnavailable()
    monkeypatch.setattr(app, "search_musicbrainz", down)
    page = client.post("/", data={"track": "Timeless", "artist": "The Weeknd"}).get_data(as_text=True)
    assert "aren&#39;t responding" in page and "Something went wrong" not in page


def test_remove_message_goes_to_library_page_once(client):
    client.post("/add", data={"track": "Money Trees", "artist": "Kendrick Lamar"})
    client.get("/")
    resp = client.post("/remove", data={"title": "Money Trees", "artist": "Kendrick Lamar", "next": "library"})
    assert resp.headers["Location"] == "/library"
    assert "Removed" in client.get("/library").get_data(as_text=True)
    assert "Removed" not in client.get("/library").get_data(as_text=True)
