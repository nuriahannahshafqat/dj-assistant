import dj_logic
from conftest import track


def test_parse_camelot():
    assert dj_logic.parse_camelot("8A") == (8, "A")
    assert dj_logic.parse_camelot(" 12b ") == (12, "B")
    assert dj_logic.parse_camelot("13A") is None
    assert dj_logic.parse_camelot("C minor") is None
    assert dj_logic.parse_camelot(None) is None


def test_key_relations():
    assert dj_logic.key_relation("8A", "8A")[1] == "same key"
    assert dj_logic.key_relation("8A", "8B")[1] == "relative major/minor"
    assert dj_logic.key_relation("8A", "9A")[1] == "adjacent key"
    assert dj_logic.key_relation("8A", "7A")[1] == "adjacent key"
    assert dj_logic.key_relation("8A", "10A")[1] == "energy boost (+2)"
    assert dj_logic.key_relation("8A", "2A")[0] == 0.0


def test_wheel_wraps_around():
    # 12 and 1 are neighbours on the Camelot wheel.
    assert dj_logic.key_relation("12A", "1A")[1] == "adjacent key"


def test_old_prompt_rule_plus_three_is_not_compatible():
    # The original prompt called +/-3 "relative major/minor" - it isn't.
    assert dj_logic.key_relation("8A", "11A")[0] == 0.0


def test_bpm_exact_and_close():
    assert dj_logic.bpm_relation(100, 100)[0] == 1.0
    assert dj_logic.bpm_relation(100, 102)[0] == 1.0  # within 3%
    assert dj_logic.bpm_relation(100, 125)[0] == 0.0


def test_bpm_half_and_double_time():
    score, pct, label = dj_logic.bpm_relation(150, 75)
    assert score == 1.0 and "half time" in label
    score, pct, label = dj_logic.bpm_relation(74, 150)
    assert score == 1.0 and "double time" in label


def test_bpm_missing_values_dont_crash():
    assert dj_logic.bpm_relation(None, 100)[2] == "unknown BPM"
    assert dj_logic.bpm_relation(0, 100)[2] == "unknown BPM"


def test_energy_prefers_hold_or_lift():
    hold = dj_logic.energy_relation(6, 6)[0]
    lift = dj_logic.energy_relation(6, 8)[0]
    drop = dj_logic.energy_relation(6, 2)[0]
    assert hold == lift == 1.0
    assert drop < 0.5


def test_shortlist_ranks_and_excludes_current(library):
    current = track("Same Key", 98, "8A", 6)  # also in library
    ranked = dj_logic.shortlist(current, library, limit=10)
    titles = [t["title"] for _, t, _ in ranked]
    assert "Same Key" not in titles
    assert titles[0] in {"Adjacent", "Relative", "Half Time"}
    assert titles.index("Clash Key") > titles.index("Adjacent")
    # 125 vs 98 BPM is too far apart to sync, and there are enough closer picks.
    assert "Too Fast" not in titles


def test_shortlist_falls_back_to_unsyncable_when_needed():
    current = track("Now", 98, "8A", 6)
    lib = [
        track("Close", 100, "9A", 6),
        track("Close 2", 97, "8B", 6),
        track("Far", 125, "8A", 6),
        track("Far 2", 140, "8A", 6),
    ]
    titles = [t["title"] for _, t, _ in dj_logic.shortlist(current, lib)]
    # Not enough syncable tracks for a full set of picks: far ones go last.
    assert set(titles[:2]) == {"Close", "Close 2"}
    assert set(titles[2:]) == {"Far", "Far 2"}


def test_build_chain_never_adds_a_clash():
    lib = [track("Start", 100, "8A"), track("Clash", 100, "2A"), track("Way Off", 140, "5B")]
    assert dj_logic.build_chain(lib[0], lib, exclude_indices={0}) == []


def test_build_chain_has_no_repeats(library):
    start = library[0]
    chain = dj_logic.build_chain(start, library, exclude_indices={0}, length=3)
    idx = [i for i, _, _ in chain]
    assert len(idx) == 3 and len(set(idx)) == 3 and 0 not in idx
