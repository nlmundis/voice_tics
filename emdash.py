"""The em-dash rule: a lone dash is a violation, a matched pair is not.

The rule this implements — an em dash is allowed only as a MATCHED PAIR
bracketing a mid-sentence aside, and a lone dash should be a colon, a comma,
a semicolon or a new sentence — is ONE AUTHOR'S, not a measured model tic.
That is why ``[lint.emdash] enabled`` is false by default.

WHY IT SHIPS ANYWAY
    The em dash is the single most-cited "AI tell" in public advice, which
    makes it the best illustration of this repo's argument. On the reference
    corpus the model's em-dash rate and the baseline author's were 1.0x apart:
    identical. A wordlist tool would have gated on it and corrected the wrong
    writer; the matched baseline said do not.

    So the useful thing this module does for most people is not failing their
    build. It is putting a number on the tell before they act on it. Run
    ``voice_tics.py`` first, read the ``em_dash_aside`` row, and turn this on
    only if YOUR ratio says something.

WHAT IT DOES NOT CATCH
    A matched pair used where a colon would read better. That is style, not
    the stated rule, and counting dashes cannot tell the difference.

TWO EXEMPTIONS, BOTH NARROW ON PURPOSE
    A leading ``term — definition`` separator in a TABLE CELL is a label
    rather than a clause join, and is never counted. Table cells only: see
    ``LEAD_TERM_RE`` for the 1,334 bogus findings that taught that lesson.

    ``[lint.emdash] exempt`` is an optional regex for a citation or record
    format that uses a lone dash as a separator by construction. Off unless
    you set it. Both exemptions are measured against the raw block and applied
    by blanking rather than deletion, so neither can see text the other
    rewrote and reported offsets still index into the original.

STDLIB ONLY, NO PACKAGE-RELATIVE IMPORTS
    So a git hook or a CI step can load this file by path under the system
    interpreter, with no virtualenv and no install.
"""

from __future__ import annotations

import re
from typing import Iterator, List, Tuple

EM_DASH = "—"

# Anchored to line starts like voice_tics.CODE_FENCE_RE, and for the same
# two reasons: a mid-line ``` run is literal text per CommonMark, and the
# two modules' renderings must agree about what a fence is — prose_lint
# partitions lines into body and table passes using THIS module's
# stripping, so a fence definition that disagrees with scrub's reopens
# the double-lint/silent-drop seam (2026-08-13, PR #46 verification).
FENCE_RE = re.compile(r"^[ \t]{0,3}```.*?\n[ \t]{0,3}```[ \t]*$",
                      re.S | re.M)
# Nonempty spans only, matching voice_tics.INLINE_CODE_RE exactly. "``" is
# literal text in CommonMark, not a code span, and stripping it here while
# voice_tics kept it made the two renderings disagree about whether a
# "``|"-led line is a table row — prose_lint then linted that line twice
# (2026-08-13). The two definitions must stay byte-identical for that
# reason. No dash-counting change either way: an empty pair holds no dash.
INLINE_CODE_RE = re.compile(r"`[^`\n]+`")
HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s")

# A leading "term — definition" separator is a label rather than a clause join,
# with a 60-character ceiling on the term.
#
# CRITICALLY, it is applied ONLY to table cells, where
# "AGC — Atlassian Government Cloud" is a genuine glossary row.
# Applying the same exception to ordinary prose inverts
# the check: it strips the single dash out of "Read the SOW first — it changes
# the severity", making the count even and the violation invisible, while
# leaving "Every deliverable — discovery, apps — follows" at an odd count and
# reporting a correct matched pair as a violation. That inversion produced 1334
# bogus findings on the first run of the scanner (2026-08-02) and is the reason
# the flag is threaded through blocks() rather than applied everywhere.
LEAD_TERM_RE = re.compile(rf"^[^.;:!?]{{1,60}}?{EM_DASH}")


def blank(block: str, spans: List[Tuple[int, int]]) -> str:
    """``block`` with each span replaced by as many spaces as it held.

    Blanking rather than deleting keeps every surviving character at its
    original offset. Every exemption in this module goes through here for that
    reason: two exemptions that both delete would each see a string the other
    had already shifted, and the reported position would index into neither.

    Args:
        block: The text to blank spans out of.
        spans: Half-open (start, end) offsets into ``block``.

    Returns:
        ``block`` with those spans spaced out, same length as it started.
    """
    out = block
    for start, end in spans:
        out = out[:start] + " " * (end - start) + out[end:]
    return out


def exempt_spans(block: str, exempt: str = "") -> List[Tuple[int, int]]:
    """The spans of ``block`` the configured exemption pattern covers.

    Args:
        block: One block as ``blocks()`` yields it, code already stripped.
        exempt: The ``[lint.emdash] exempt`` regex, or "" for no exemption,
            which is the default and the case for almost everyone.

    Returns:
        Ascending, non-overlapping spans; empty when nothing is exempt and
        empty when the pattern cannot be applied.

    FAILURE POLICY: NO SPANS, NEVER AN EXCEPTION
        A bad pattern returns no spans rather than raising. No spans means
        nothing is exempt, so every dash is counted and the caller gets its
        ordinary noisy verdict — loud, and wrong in the direction someone
        notices. Letting the exception escape would instead reach a caller's
        top-level handler, exit 0 and print nothing, silently disarming the
        ENTIRE em-dash check rather than just this exemption. In the private
        original the exemption was a sibling module loaded by file path, and a
        zero-byte copy of it did exactly that: loaded cleanly, exempted
        everything, and was found doing it in review (2026-08-14).

    WRITE THE NARROWEST PATTERN THAT WORKS
        Every span this returns is prose whose dashes stop being counted, so
        an over-broad pattern is a hole in the check, not a convenience. The
        original's exemption covered a citation separator and used the
        SHORTEST parse — prefix through the FIRST closing quote — because its
        first version used a greedy parse, and one quotation mark anywhere in
        trailing prose extended the span to the end of the line and took a
        real lone dash with it, silently, through an error-tier lint. Prefer
        a non-greedy pattern anchored to something structural.
    """
    if not exempt:
        return []
    try:
        return [m.span() for m in re.finditer(exempt, block)]
    except re.error:
        return []


def strip_code(text: str) -> str:
    """Blank out fenced blocks and inline code spans, preserving line count.

    A rule about prose must not fire on a code sample that legitimately
    contains the character it is looking for. Newlines are preserved so
    paragraph splitting downstream still lines up.

    Args:
        text: Raw markdown or chat text.

    Returns:
        The text with code blanked and its line count unchanged.
    """
    text = FENCE_RE.sub(lambda m: "\n" * m.group(0).count("\n"), text)
    return INLINE_CODE_RE.sub(" ", text)


def is_table_row(line: str) -> bool:
    """Whether a line is treated as a table row: a leading pipe.

    THE single definition of "table row" for every caller that must agree
    with the cell walk below — prose_lint masks exactly these lines out of
    its body-prose pass. Two definitions (this one and a trailing-pipe
    regex) linted the gap rows twice (found 2026-08-13, PR #46 review).
    """
    return line.lstrip().startswith("|")


def numbered_blocks(text: str) -> Iterator[Tuple[int, str, bool]]:
    """The blocks of ``blocks()``, each carrying its 1-based line number.

    The single walk both block iterators are built on. Carrying the line
    number here, where the split happens, is what lets a linter report
    exact locations; reverse-locating a block by substring search
    mis-attributed a violation to an earlier line that merely contained
    the same text (found 2026-08-13, prose_lint review).

    Args:
        text: Text with code already stripped.

    Yields:
        (line_number, block_text, is_table_cell) triples, one per line or
        per table cell; cells of one row share the row's line number.
    """
    for i, line in enumerate(text.split("\n"), start=1):
        if HEADING_RE.match(line):
            continue
        if is_table_row(line):
            # Split on unescaped pipes only: GFM and Obsidian read "\|"
            # as a literal pipe INSIDE one cell, and "\|" is Obsidian's
            # required alias syntax for a wikilink in a table cell, so a
            # naive split cut such cells — and any sentence in them — in
            # half (2026-08-13, PR #46 verification, round six).
            for cell in re.split(r"(?<!\\)\|", line):
                if cell.strip():
                    yield i, cell, True
            continue
        yield i, line, False


def blocks(text: str) -> Iterator[Tuple[str, bool]]:
    """Split markdown/chat text into the units an em dash pair may span.

    A matched pair must sit inside one sentence, so the unit is a line, a list
    item, or a single table cell. Headings are skipped for the same reason
    lint.py skips them.

    Args:
        text: Text with code already stripped.

    Yields:
        (block_text, is_table_cell) pairs. The flag matters because the
        "term — definition" label exception is only valid inside a table cell;
        applying it to ordinary prose inverts the whole check.
    """
    for _line, blk, is_cell in numbered_blocks(text):
        yield blk, is_cell


def lone_dash_lines(text: str, exempt: str = ""
                    ) -> List[Tuple[int, str, int]]:
    """Lone-dash violations with the 1-based line each one sits on.

    The rule permits em dashes only as a matched pair around a mid-sentence
    aside, so an odd count within one block is the signal. Counting rather than
    pattern-matching is what makes this cheap and unambiguous.

    Args:
        text: Raw markdown or chat text. Code is stripped here, so callers do
            not have to remember to; ``strip_code`` preserves line count, so
            the numbers refer to the caller's original text.

    Returns:
        A list of (line number, block, index of the first COUNTED em dash in
        that block). Positions rather than formatted excerpts, so a caller that
        must redact before displaying can do so. The index skips dashes the
        exemptions removed, so it points at the dash the caller is being asked
        about rather than at an exempt one earlier in the line.
    """
    out: List[Tuple[int, str, int]] = []
    for line_no, blk, is_cell in numbered_blocks(strip_code(text)):
        # BOTH exemptions are measured against the RAW block and applied by
        # blanking, so neither can see text the other rewrote and the surviving
        # offsets still index into blk. Measuring the label exception after
        # masking is not a refactor detail: masking blanks the colon in
        # a citation label's colon, which is the character that kept
        # LEAD_TERM_RE from matching a cell opening with a citation, so the
        # label exception then swallowed up to 60 further characters of the
        # writer's own prose, dash included (2026-08-14 review).
        spans = exempt_spans(blk, exempt)
        if is_cell:
            label = LEAD_TERM_RE.match(blk)
            if label:
                spans = spans + [(label.start(), label.end())]
        counted = blank(blk, spans)
        if counted.count(EM_DASH) % 2 == 1:
            out.append((line_no, blk, max(counted.find(EM_DASH), 0)))
    return out


def lone_dash_blocks(text: str, exempt: str = "") -> List[Tuple[str, int]]:
    """``lone_dash_lines`` without the line numbers, for callers that judge
    text that never had them (chat messages, single table cells).

    Returns:
        A list of (block, index of the first COUNTED em dash in that block).
        Counted, not first: a dash an exemption removed is not the one the
        caller is being asked about, and this is the API the deployed hook
        renders its excerpt from.
    """
    return [(blk, pos) for _line, blk, pos in lone_dash_lines(text, exempt)]
