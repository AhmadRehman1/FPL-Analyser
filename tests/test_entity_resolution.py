from fpl_quant import entity_resolution as er


def test_normalize_strips_accents():
    assert er.normalize_name("Aarón Anselmino") == er.normalize_name("Aaron Anselmino")


def test_normalize_case_and_whitespace_insensitive():
    assert er.normalize_name("  Erling  Haaland ") == er.normalize_name("erling haaland")


def test_normalize_strips_punctuation():
    assert er.normalize_name("O'Neill") == "o neill"


def test_normalize_none_is_empty_string():
    assert er.normalize_name(None) == ""


def test_normalize_german_sharp_s_matches_double_s():
    # real bug this project hit: a research-pull source spelled "Pascal Gross" with a plain
    # double-s while the registered name uses the German sharp-s character -- NFKD doesn't
    # decompose it (not a diacritic), so this needs casefold(), not lower().
    assert er.normalize_name("Pascal Groß") == er.normalize_name("Pascal Gross")


def test_team_uid_deterministic():
    assert er.team_uid_for("Arsenal") == er.team_uid_for("Arsenal")


def test_team_uid_differs_for_different_teams():
    assert er.team_uid_for("Arsenal") != er.team_uid_for("Chelsea")


def test_player_uid_matches_across_accent_variants():
    # the real collision found in FPL-Core-Insights data: same person, two spellings
    assert er.player_uid_for("Aarón Anselmino") == er.player_uid_for("Aaron Anselmino")


def test_player_uid_differs_for_different_names():
    assert er.player_uid_for("Erling Haaland") != er.player_uid_for("Bukayo Saka")
