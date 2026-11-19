import app


def r(title, artist, preview=True):
    return {"trackName": title, "artistName": artist,
            "previewUrl": f"https://audio/{title}-{artist}.m4a" if preview else None}


# Real iTunes results for "no pole Don Toliver": the song itself isn't there.
NO_POLE_RESULTS = [
    r("No Pole - Slowed and Reverb", "rufflws"),
    r("No Idea", "Don Toliver"),
    r("TORE UP", "Don Toliver"),
    r("No Pole", "David Shannon"),
    r("no pole (slowed + reverb)", "Slo & Twilight"),
]


def test_never_falls_back_to_a_different_song():
    # Used to return "No Idea" (the top Don Toliver result).
    assert app._pick_itunes_result(NO_POLE_RESULTS, "No Pole", "Don Toliver") is None


def test_same_title_by_another_artist_is_rejected():
    assert app._pick_itunes_result([r("No Pole", "David Shannon")], "No Pole", "Don Toliver") is None


def test_exact_match_is_found_below_other_results():
    results = NO_POLE_RESULTS + [r("No Pole", "Don Toliver")]
    best = app._pick_itunes_result(results, "no pole", "Don Toliver")
    assert best["trackName"] == "No Pole" and best["artistName"] == "Don Toliver"


def test_featured_artists_on_either_side_still_match():
    results = [r("Open Arms (feat. Travis Scott)", "SZA")]
    assert app._pick_itunes_result(results, "Open Arms", "SZA feat. Travis Scott")
    results = [r("Rockstar", "Post Malone & 21 Savage")]
    assert app._pick_itunes_result(results, "rockstar", "Post Malone feat. 21 Savage")


def test_alt_versions_skipped_unless_asked_for():
    results = [r("Kill Bill (Sped Up)", "SZA"), r("Kill Bill", "SZA")]
    assert app._pick_itunes_result(results, "Kill Bill", "SZA")["trackName"] == "Kill Bill"
    results = [r("Kill Bill (Sped Up)", "SZA")]
    assert app._pick_itunes_result(results, "Kill Bill", "SZA") is None


def test_results_without_a_preview_are_skipped():
    results = [r("Kill Bill", "SZA", preview=False)]
    assert app._pick_itunes_result(results, "Kill Bill", "SZA") is None
