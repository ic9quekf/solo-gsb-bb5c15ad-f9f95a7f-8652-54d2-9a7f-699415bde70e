"""Tests for the identifier normalization endpoint.

Covers: Python lexical extraction (comments/strings/keywords skipped),
Unicode NFKC collision reporting (distinct raw spellings only; plain
repeats and case differences never collide), occurrence positions and
code-point offsets in stable source order, keyword vs. soft-keyword
handling, report persistence and retrieval, listing/filters, and 422 on
lexical failure with no audit record written.
"""

import pytest

# ﬀ is U+FB00 LATIN SMALL LIGATURE FF -> NFKC "ff"
LIG = "ﬀ"
# fullwidth lowercase f,o,o -> NFKC "foo"
FW = "ｆｏｏ"
# U+210E PLANCK CONSTANT -> NFKC "h", and it is a legal XID_Start char
PLANCK = "ℎ"

PATH = "/api/v1/identifiers/normalization"


def post(client, source, filename=None):
    body = {"source": source}
    if filename is not None:
        body["filename"] = filename
    return client.post(PATH, json=body)


# ------------------------------------------------------------ extraction ---

def test_basic_collision_fullwidth(client):
    source = "foo = 1\nbar = " + FW + "\n"
    resp = post(client, source, filename="m.py")
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["stored"] is True
    assert body["identifier_count"] == 3  # foo, bar, fullwidth
    assert body["normalized_count"] == 2
    assert body["collision_count"] == 1

    collision = body["collisions"][0]
    assert collision["normalized"] == "foo"
    assert collision["spelling_count"] == 2
    by_spelling = {s["spelling"]: s for s in collision["spellings"]}
    assert set(by_spelling) == {"foo", FW}

    ascii_occ = by_spelling["foo"]["occurrences"][0]
    assert ascii_occ["offset"] == 0
    assert ascii_occ["end_offset"] == 3
    assert ascii_occ["position"] == {"line": 1, "column": 1}
    assert ascii_occ["end_position"] == {"line": 1, "column": 4}

    fw_occ = by_spelling[FW]["occurrences"][0]
    # source offsets are code points; fullwidth chars each occupy one
    assert source[fw_occ["offset"]:fw_occ["end_offset"]] == FW
    assert fw_occ["offset"] == 14
    assert fw_occ["position"] == {"line": 2, "column": 7}


def test_ligature_collision(client):
    source = f"x = ff + {LIG}\n"
    resp = post(client, source)
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["collision_count"] == 1
    collision = body["collisions"][0]
    assert collision["normalized"] == "ff"
    assert {s["spelling"] for s in collision["spellings"]} == {"ff", LIG}


def test_square_symbol_collision(client):
    source = f"h = 1\nx = {PLANCK}\n"
    resp = post(client, source)
    assert resp.status_code == 201, resp.text
    normalized = {c["normalized"] for c in resp.json()["collisions"]}
    assert "h" in normalized


def test_non_identifier_unicode_is_a_lexical_error(client):
    # U+339A SQUARE NM normalizes to "nm" but is not an XID_Start code
    # point, so Python's lexer rejects it (extraction follows Python's
    # own lexical rules, not generic Unicode word splitting).
    resp = post(client, "㎚ = 1\n")
    assert resp.status_code == 422


def test_comments_and_strings_are_skipped(client):
    source = (
        "foo = 1\n"
        f"# {FW} lives only in a comment\n"
        "s = 'foo and " + FW + " inside a string'\n"
        't = f"foo {FW} interpolation"\n'
        f"x = {FW}\n"
    )
    resp = post(client, source)
    assert resp.status_code == 201, resp.text
    body = resp.json()
    # only foo(assignment), s, t, x, fullwidth(final line) are names;
    # string/comment contents never participate
    collision = body["collisions"][0]
    spellings = {s["spelling"]: s["occurrences"] for s in collision["spellings"]}
    assert set(spellings) == {"foo", FW}
    assert len(spellings["foo"]) == 1
    assert len(spellings[FW]) == 1
    # the single fullwidth occurrence is the one in code, at the last line
    occ = spellings[FW][0]
    assert occ["position"]["line"] == 5
    assert source[occ["offset"]:occ["end_offset"]] == FW


def test_keywords_are_skipped_but_soft_keywords_kept(client):
    # `if`, `def`, `return` are hard keywords and must not count as
    # identifiers; `match`/`case` are soft keywords and stay name tokens.
    source = (
        "def if_like():\n"
        "    foo = 0\n"
        "    if True:\n"
        "        match = 1\n"
        f"        return match + {FW}\n"
        "    case = 2\n"
        "    return case\n"
    )
    resp = post(client, source)
    assert resp.status_code == 201, resp.text
    normalized_words = {c["normalized"] for c in resp.json()["collisions"]}
    # hard keywords never participate
    assert "def" not in normalized_words
    assert "if" not in normalized_words
    assert "return" not in normalized_words
    assert "true" not in normalized_words
    # soft keywords are ordinary identifier tokens; same spelling repeated
    # never collides; the fullwidth/ASCII foo pair does collide
    assert "match" not in normalized_words
    assert "case" not in normalized_words
    assert "foo" in normalized_words


def test_plain_repetition_is_not_a_collision(client):
    source = "foo = foo + foo\n"
    resp = post(client, source)
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["identifier_count"] == 3
    assert body["normalized_count"] == 1
    assert body["collision_count"] == 0
    assert body["collisions"] == []


def test_case_difference_is_not_a_collision(client):
    source = "foo = 1\nFoo = 2\nFOO = 3\n"
    resp = post(client, source)
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["identifier_count"] == 3
    assert body["normalized_count"] == 3
    assert body["collision_count"] == 0


def test_three_distinct_spellings_one_normalized_form(client):
    # fullwidth-f + ASCII 'oo' is a third raw spelling of "foo"
    source = f"foo = 1\nx = {FW}\nｆoo = 3\n"
    resp = post(client, source)
    assert resp.status_code == 201, resp.text
    collision = resp.json()["collisions"][0]
    assert collision["normalized"] == "foo"
    assert collision["spelling_count"] == 3
    # spellings ordered by first appearance in source
    assert [s["spelling"] for s in collision["spellings"]] == [
        "foo", FW, "ｆoo",
    ]


def test_occurrences_stable_in_source_order(client):
    source = (
        f"{FW}\nfoo\n{FW}\nfoo\n"
    )
    resp = post(client, source)
    assert resp.status_code == 201, resp.text
    collision = resp.json()["collisions"][0]
    spellings = {s["spelling"]: s["occurrences"] for s in collision["spellings"]}
    fw_lines = [o["position"]["line"] for o in spellings[FW]]
    ascii_lines = [o["position"]["line"] for o in spellings["foo"]]
    assert fw_lines == [1, 3]
    assert ascii_lines == [2, 4]
    # collisions ordered by first appearance: this is the only collision,
    # starting at offset 0
    assert spellings[FW][0]["offset"] == 0


def test_collisions_ordered_by_first_appearance(client):
    # collision "h" appears before collision "foo"
    source = f"{PLANCK} = 1\nh = 2\n{FW} = 3\nfoo = 4\n"
    resp = post(client, source)
    assert resp.status_code == 201, resp.text
    assert [c["normalized"] for c in resp.json()["collisions"]] == ["h", "foo"]


def test_offsets_are_code_points_with_astral_chars(client):
    # astral character (😀 = 2 UTF-16 units, 1 code point) precedes a name;
    # every occurrence's offset slice must equal its raw spelling
    source = "# 😀 comment\nfoo = 1\nx = " + FW + "\n"
    resp = post(client, source)
    assert resp.status_code == 201, resp.text
    collision = resp.json()["collisions"][0]
    for spelling in collision["spellings"]:
        for occ in spelling["occurrences"]:
            assert source[occ["offset"]:occ["end_offset"]] == \
                spelling["spelling"]


def test_multiline_and_crlf_offsets(client):
    source = "foo = 1\r\nx = " + FW + "\r\n"
    resp = post(client, source)
    assert resp.status_code == 201, resp.text
    collision = resp.json()["collisions"][0]
    for spelling in collision["spellings"]:
        for occ in spelling["occurrences"]:
            assert source[occ["offset"]:occ["end_offset"]] == spelling["spelling"]
    fw = next(s for s in collision["spellings"] if s["spelling"] == FW)
    assert fw["occurrences"][0]["position"] == {"line": 2, "column": 5}


def test_empty_and_comment_only_source(client):
    for source in ("", "# just a comment\n", "\n\n"):
        resp = post(client, source)
        assert resp.status_code == 201, resp.text
        body = resp.json()
        assert body["identifier_count"] == 0
        assert body["normalized_count"] == 0
        assert body["collision_count"] == 0


def test_non_identifier_unicode_still_scans(client):
    # operators/numbers using unicode-aware tokens: ° is an ERRORTOKEN?
    # Actually non-name stray characters are lexical errors (see 422 tests);
    # a valid identifier with combining marks normalizes as expected.
    source = "café = 1\ncafe = 2\n"  # é U+00E9 does not compose away
    resp = post(client, source)
    assert resp.status_code == 201, resp.text
    assert resp.json()["collision_count"] == 0


# ------------------------------------------------------------- persistence ---

def test_report_persisted_and_readable_by_id(client):
    source = f"foo = 1\nx = {FW}\n"
    resp = post(client, source, filename="pkg/mod.py")
    assert resp.status_code == 201, resp.text
    report_id = resp.json()["id"]

    got = client.get(f"{PATH}/{report_id}")
    assert got.status_code == 200
    record = got.json()
    assert record["id"] == report_id
    assert record["filename"] == "pkg/mod.py"
    assert record["source_length"] == len(source)
    assert record["identifier_count"] == 3
    assert record["collision_count"] == 1
    assert record["raw_source"] == source
    assert record["report"]["collisions"][0]["normalized"] == "foo"


def test_zero_collision_report_is_persisted(client):
    resp = post(client, "a = 1\nb = 2\n")
    assert resp.status_code == 201, resp.text
    report_id = resp.json()["id"]
    got = client.get(f"{PATH}/{report_id}")
    assert got.status_code == 200
    assert got.json()["collision_count"] == 0


def test_get_unknown_report_404(client):
    resp = client.get(f"{PATH}/9999")
    assert resp.status_code == 404
    assert resp.json()["detail"] == "normalization report not found"


def test_list_reports_filters_and_pagination(client):
    post(client, "a = 1\n", filename="one.py")
    post(client, f"foo = 1\n{FW}\n", filename="two.py")
    post(client, "b = 2\n", filename="three.py")

    listing = client.get(PATH)
    assert listing.status_code == 200
    body = listing.json()
    assert body["total"] == 3
    assert len(body["items"]) == 3
    # newest first
    assert [i["filename"] for i in body["items"]] == [
        "three.py", "two.py", "one.py",
    ]
    # summaries must not leak raw source/report
    assert "raw_source" not in body["items"][0]
    assert "report" not in body["items"][0]

    only_col = client.get(PATH, params={"only_collisions": "true"})
    assert only_col.json()["total"] == 1
    assert only_col.json()["items"][0]["filename"] == "two.py"
    assert only_col.json()["items"][0]["collision_count"] == 1

    by_name = client.get(PATH, params={"filename": "TWO"})
    assert by_name.json()["total"] == 1

    page = client.get(PATH, params={"limit": 2, "offset": 0})
    assert [i["filename"] for i in page.json()["items"]] == [
        "three.py", "two.py",
    ]
    page2 = client.get(PATH, params={"limit": 2, "offset": 2})
    assert [i["filename"] for i in page2.json()["items"]] == ["one.py"]


# --------------------------------------------------------------- 422 paths ---

def test_lexical_error_422_writes_nothing(client):
    resp = post(client, "x = 'unterminated\n")
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["error"] == "invalid_python_source"
    # quote sits at offset 4 (0-based), line 1 column 5
    assert detail["offset"] == 4
    assert detail["line"] == 1
    assert detail["column"] == 5
    assert detail["message"]

    listing = client.get(PATH)
    assert listing.json()["total"] == 0


def test_stray_character_422_position(client):
    source = "x = 1\n$\n"
    resp = post(client, source)
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["offset"] == 6
    assert detail["line"] == 2
    assert detail["column"] == 1


def test_nul_character_422(client):
    resp = post(client, "x = 1\n\x00\n")
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["line"] == 2
    assert detail["offset"] == 6


def test_unterminated_triple_string_422(client):
    resp = post(client, 'x = """abc\n')
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert "multi-line" in detail["message"]
    assert detail["line"] == 1


def test_indentation_error_422_position(client):
    source = "def f():\n  x = 1\n y = 2\n"
    resp = post(client, source)
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["line"] == 3
    assert detail["column"] == 1
    assert detail["offset"] == 17


def test_lexical_error_after_valid_prefix_does_not_persist(client):
    # earlier valid content must still not be stored when a later line fails
    source = f"foo = 1\n{FW} = 2\n$\n"
    resp = post(client, source)
    assert resp.status_code == 422
    assert client.get(PATH).json()["total"] == 0


def test_missing_source_field_422(client):
    resp = client.post(PATH, json={"filename": "x.py"})
    assert resp.status_code == 422


def test_syntax_but_not_lexical_error_is_accepted(client):
    # invalid *syntax* (bare @, leading-zero number) is still lexically
    # scannable: the endpoint performs lexical extraction only
    resp = post(client, "x = 09\n@\n")
    assert resp.status_code == 201, resp.text


# ------------------------------------------------------------------ module ---

def test_module_error_attributes():
    from app.identifier_norm import IdentifierLexError

    err = IdentifierLexError(3, 2, 4, "boom")
    assert err.offset == 3
    assert err.line == 2
    assert err.column == 4
    assert err.message == "boom"
    assert "line 2" in str(err)
