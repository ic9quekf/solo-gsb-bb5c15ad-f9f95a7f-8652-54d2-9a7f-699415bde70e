"""Python identifier extraction and Unicode NFKC collision detection.

Python source code is scanned with the standard-library :mod:`tokenize`
lexer, so only genuine *name* tokens are collected: comments, string
literals (including their prefixes) and every keyword (``keyword.kwlist``;
soft keywords such as ``match``/``case`` remain usable as identifiers and
are kept) are skipped.

Each identifier spelling is mapped through Unicode NFKC
(:func:`unicodedata.normalize`).  A *collision* is reported only when two
or more **distinct raw spellings** share one normalized form.  A name that
merely repeats, or spellings that differ only in case (ASCII case is
preserved by NFKC), never collide.

The lexer works on the raw text; a lexical failure (an
:class:`IndentationError`, :class:`TokenError`, or any ``ERRORTOKEN`` the
lexer emits for stray characters such as ``$``, NUL or an unterminated
string) raises :class:`IdentifierLexError` carrying the offending
position in the original source before any result is produced.
"""

from __future__ import annotations

import io
import keyword
import tokenize
import unicodedata
from dataclasses import dataclass, field


class IdentifierLexError(ValueError):
    """The source could not be scanned by the Python lexer.

    ``offset`` is the 0-based code-point offset in the original source
    where the problem was detected; ``line``/``column`` are 1-based.
    """

    def __init__(self, offset: int, line: int, column: int, message: str):
        super().__init__(f"line {line}, column {column}: {message}")
        self.offset = offset
        self.line = line
        self.column = column
        self.message = message


@dataclass(frozen=True)
class Occurrence:
    """One raw occurrence of an identifier spelling in the source."""

    offset: int                       # code-point offset of the first char
    line: int                         # 1-based source line
    column: int                       # 1-based source column (code points)
    end_offset: int                   # code-point offset just past the name
    end_line: int
    end_column: int


@dataclass
class Collision:
    """Two or more distinct raw spellings sharing one NFKC form."""

    normalized: str
    spellings: dict[str, list[Occurrence]] = field(default_factory=dict)


@dataclass
class NormalizationReport:
    identifier_count: int
    normalized_count: int
    collisions: list[Collision]

    def to_dict(self) -> dict:
        return {
            "identifier_count": self.identifier_count,
            "normalized_count": self.normalized_count,
            "collision_count": len(self.collisions),
            "collisions": [
                {
                    "normalized": collision.normalized,
                    "spelling_count": len(collision.spellings),
                    "spellings": [
                        {
                            "spelling": spelling,
                            "occurrence_count": len(occurrences),
                            "occurrences": [
                                {
                                    "offset": occ.offset,
                                    "position": {
                                        "line": occ.line,
                                        "column": occ.column,
                                    },
                                    "end_offset": occ.end_offset,
                                    "end_position": {
                                        "line": occ.end_line,
                                        "column": occ.end_column,
                                    },
                                }
                                for occ in occurrences
                            ],
                        }
                        for spelling, occurrences in collision.spellings.items()
                    ],
                }
                for collision in self.collisions
            ],
        }


def _line_starts(source: str) -> list[int]:
    """Code-point offset of the first character of every line."""
    starts = [0]
    for index, char in enumerate(source):
        if char == "\n":
            starts.append(index + 1)
    return starts


def _offset_of(starts: list[int], line: int, column: int) -> int:
    """Convert tokenize (line, 0-based column) to an absolute offset.

    ``line`` is 1-based; continuation tokens on virtual lines never
    outrun the source because they always carry a real ``line`` member.
    """
    return starts[line - 1] + column


def _lex_error(starts: list[int], line: int, column: int, message: str) -> IdentifierLexError:
    # tokenize columns are 0-based; the API reports 1-based columns.
    offset = _offset_of(starts, line, column)
    return IdentifierLexError(offset, line, column + 1, message)


def scan_identifier_collisions(source: str) -> NormalizationReport:
    """Extract identifiers and find NFKC-normalization collisions.

    Raises :class:`IdentifierLexError` on any lexical failure.
    """
    starts = _line_starts(source)

    # normalized form -> raw spelling -> occurrences in source order.
    groups: dict[str, dict[str, list[Occurrence]]] = {}
    identifier_count = 0

    try:
        token_stream = tokenize.generate_tokens(
            io.StringIO(source, newline=None).readline
        )
        tokens = list(token_stream)
    except tokenize.TokenError as exc:
        message, (line, column) = exc.args
        raise _lex_error(starts, line, column, message.lower()) from exc
    except IndentationError as exc:
        # TabError is a subclass; same attributes apply.
        line = exc.lineno
        column = exc.offset or 1
        raise _lex_error(starts, line, column - 1, exc.msg) from exc

    for tok in tokens:
        if tok.type == tokenize.ERRORTOKEN and tok.string.strip():
            # Whitespace is sometimes emitted as a pseudo-ERRORTOKEN when an
            # unterminated string leaves the lexer mid-line; the first real
            # stray token (a quote, '$', NUL ...) is the offending point.
            raise _lex_error(
                starts, tok.start[0], tok.start[1],
                f"unexpected character {tok.string!r}",
            )
        if tok.type != tokenize.NAME:
            continue
        if keyword.iskeyword(tok.string):
            continue

        identifier_count += 1
        normalized = unicodedata.normalize("NFKC", tok.string)
        occurrence = Occurrence(
            offset=_offset_of(starts, tok.start[0], tok.start[1]),
            line=tok.start[0],
            column=tok.start[1] + 1,
            end_offset=_offset_of(starts, tok.end[0], tok.end[1]),
            end_line=tok.end[0],
            end_column=tok.end[1] + 1,
        )
        groups.setdefault(normalized, {}).setdefault(tok.string, []).append(
            occurrence
        )

    collisions = [
        Collision(normalized=normalized, spellings=spellings)
        for normalized, spellings in groups.items()
        if len(spellings) >= 2
    ]
    # Stable order: by first source position of the collision (the earliest
    # occurrence across its spellings); normalized form breaks ties.
    def first_offset(collision: Collision) -> int:
        return min(
            occurrences[0].offset
            for occurrences in collision.spellings.values()
        )

    collisions.sort(key=lambda c: (first_offset(c), c.normalized))
    # Within a collision list spellings by first appearance, and occurrence
    # lists are already in scan order (hence source order).
    for collision in collisions:
        ordered = sorted(
            collision.spellings.items(),
            key=lambda item: item[1][0].offset,
        )
        collision.spellings = dict(ordered)

    return NormalizationReport(
        identifier_count=identifier_count,
        normalized_count=len(groups),
        collisions=collisions,
    )
