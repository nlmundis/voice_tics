#!/usr/bin/env python3
"""Tests for the em-dash rule in emdash.py.

The rule: an em dash is allowed only as a MATCHED PAIR bracketing a
mid-sentence aside. A lone dash is the violation.

The cases that matter most here are the two inversions this check has already
suffered. Applying the "term — definition" label exception outside table cells
strips the dash out of "Read the SOW first — it changes the severity", hiding a
real violation, while flagging "Every deliverable — discovery, apps — follows",
a correct matched pair. That inversion produced 1334 bogus findings on the
scanner's first run, so both directions are asserted below rather than just
the happy path.

Run:
  make test
"""

from __future__ import annotations

import os
import sys
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import emdash  # noqa: E402

# The exemption pattern these tests exercise: a citation whose separator is a
# lone em dash by construction. SHORTEST parse — prefix through the FIRST
# closing quote — which is the narrowness the tests below pin. A greedy parse
# ([\s\S]*") would let one quotation mark anywhere in trailing prose extend
# the span to end of line and swallow a real lone dash, silently.
CITE_RE = r'EVIDENCE:[^"\n]*"[^"\n]*"'


class LoneDashTest(unittest.TestCase):
    """The core check, both directions."""

    def test_a_lone_dash_in_prose_is_flagged(self) -> None:
        found = emdash.lone_dash_blocks("Read the SOW first — it changes the severity.", CITE_RE)
        self.assertEqual(len(found), 1)

    def test_a_matched_pair_is_clean(self) -> None:
        self.assertEqual(
            emdash.lone_dash_blocks("Every deliverable — discovery, apps — follows.", CITE_RE), [])

    def test_the_label_exception_applies_only_inside_table_cells(self) -> None:
        """The inversion that cost 1334 bogus findings. Both halves asserted."""
        self.assertEqual(
            emdash.lone_dash_blocks("| AGC — Atlassian Government Cloud | yes |", CITE_RE), [])
        # The same shape in prose is a real violation and must NOT be excused.
        self.assertEqual(
            len(emdash.lone_dash_blocks("AGC — Atlassian Government Cloud is the target.", CITE_RE)), 1)

    def test_code_is_exempt(self) -> None:
        self.assertEqual(emdash.lone_dash_blocks("```\nx = a — b\n```", CITE_RE), [])
        self.assertEqual(emdash.lone_dash_blocks("Use `a — b` here.", CITE_RE), [])

    def test_headings_are_skipped(self) -> None:
        self.assertEqual(emdash.lone_dash_blocks("## A heading — with a dash", CITE_RE), [])

    def test_a_pair_may_not_span_two_lines(self) -> None:
        """Each dash on its own line is two lone dashes, not one pair."""
        self.assertEqual(len(emdash.lone_dash_blocks("First — line\nSecond — line", CITE_RE)), 2)

    def test_position_points_at_the_first_dash(self) -> None:
        block, at = emdash.lone_dash_blocks("abc — def", CITE_RE)[0]
        self.assertEqual(block[at], emdash.EM_DASH)

    def test_no_dashes_is_clean(self) -> None:
        self.assertEqual(emdash.lone_dash_blocks("Plain prose, no dashes at all.", CITE_RE), [])


class EvidenceCitationExemptionTest(unittest.TestCase):
    """The carve-out for the citation separator, and its four boundaries.

    The exemption exists because ``EVIDENCE: [[Note]] — "quote"`` carries a
    lone dash by construction. It is narrow on purpose: every case below that
    ends in a violation is a case where a wider rule would have gone silent.
    """

    CITE = 'EVIDENCE: [[Note A]] — "a quote long enough to count"'

    def test_a_citation_alone_is_not_a_violation(self) -> None:
        self.assertEqual(emdash.lone_dash_blocks(self.CITE, CITE_RE), [])

    def test_dashes_inside_the_quote_are_the_cited_authors(self) -> None:
        """Three dashes without the carve-out: separator plus the pair the
        source wrote. A verbatim quote is not the citing writer's to edit."""
        self.assertEqual(
            emdash.lone_dash_blocks(
                'EVIDENCE: [[Note]] — "he said — plainly — that it held"',
                CITE_RE), [])

    def test_prose_after_the_quote_still_counts(self) -> None:
        line = f"{self.CITE} and then — a lone dash"
        found = emdash.lone_dash_blocks(line, CITE_RE)
        self.assertEqual(len(found), 1)
        _blk, at = found[0]
        self.assertEqual(line[at], emdash.EM_DASH)
        self.assertGreater(at, len(self.CITE),
                           "the reported dash must be the trailing one, not "
                           "the exempt separator")

    def test_prose_between_two_citations_still_counts(self) -> None:
        """A greedy match over the whole line would swallow this dash."""
        line = (f"{self.CITE} and — separately "
                'EVIDENCE: [[Note B]] — "a second quote of real length"')
        self.assertEqual(len(emdash.lone_dash_blocks(line, CITE_RE)), 1)

    def test_a_prefix_that_does_not_parse_is_not_exempt(self) -> None:
        """Shape is the whole test, so text merely dressed as a citation buys
        nothing; a citation checker would report it malformed."""
        for line in ("EVIDENCE: no brackets — so this is just prose",
                     "EVIDENCE: [[Note]] — no quote, so this is just prose"):
            with self.subTest(line=line):
                self.assertEqual(len(emdash.lone_dash_blocks(line, CITE_RE)), 1)

    def test_a_quote_in_trailing_prose_does_not_buy_silence(self) -> None:
        """The 2026-08-14 defect. Greedy, the exempt span ran to the last quote
        on the line and took this dash with it, through an ERROR-tier lint."""
        line = f'{self.CITE} — the entry calls it a "convention", not a rule'
        found = emdash.lone_dash_blocks(line, CITE_RE)
        self.assertEqual(len(found), 1)
        _blk, at = found[0]
        self.assertGreater(at, len(self.CITE))

    def test_a_quote_in_the_gap_between_citations_does_not_buy_silence(self) -> None:
        """The same defect, two-citation form, and a regression against the
        behavior this module had before the exemption existed."""
        line = (f'{self.CITE} — the SOW says "x" '
                'EVIDENCE: [[Note B]] — "a second quote of real length"')
        self.assertEqual(len(emdash.lone_dash_blocks(line, CITE_RE)), 1)

    def test_a_prefix_without_a_target_is_not_exempt_even_with_a_quote(self) -> None:
        self.assertEqual(
            len(emdash.lone_dash_blocks(
                'EVIDENCE: no wikilink at all — "a quoted clause of real length"')), 1, CITE_RE)

    def test_prose_before_a_citation_still_counts(self) -> None:
        """The left edge, which nothing pinned until review."""
        self.assertEqual(
            len(emdash.lone_dash_blocks(f"As argued — at length, {self.CITE}", CITE_RE)), 1)

    def test_a_citation_in_a_table_cell_is_exempt(self) -> None:
        self.assertEqual(emdash.lone_dash_blocks(f"| claim | {self.CITE} |", CITE_RE), [])

    def test_a_cell_citation_does_not_arm_the_label_exception(self) -> None:
        """Blanking the citation blanks the colon in EVIDENCE:, which is the
        character that kept LEAD_TERM_RE from matching a cell opening with one.
        Measured against the raw block, it still cannot (2026-08-14 review)."""
        cell = f'| {self.CITE} — the SOW changes it |'
        prose = f'{self.CITE} — the SOW changes it'
        self.assertEqual(len(emdash.lone_dash_blocks(cell, CITE_RE)), 1,
                         "a cell must not excuse prose a paragraph would not")
        self.assertEqual(len(emdash.lone_dash_blocks(prose, CITE_RE)), 1)

    def test_the_reported_position_skips_an_exempt_dash(self) -> None:
        """The docstring promises the index points at a COUNTED dash; in a cell
        carrying both an exempt dash and a real one, that is the second.

        This once used a label cell, "AGC — Atlassian Government Cloud, which
        — note". Counting cannot tell that from a matched pair, and the label
        exemption now applies only to an odd count, so the cell is clean: the
        undercount this module prefers to failing a build on a correct pair.
        """
        cell = '| EVIDENCE: "a — b" and then a lone — dash |'
        found = emdash.lone_dash_lines(cell, CITE_RE)
        self.assertEqual(len(found), 1)
        _line, blk, at = found[0]
        self.assertEqual(blk[at], emdash.EM_DASH)
        self.assertGreater(at, blk.index(emdash.EM_DASH),
                           "the cited dash was exempted, so it must not be the "
                           "one reported")

    def test_a_matched_pair_in_a_cell_is_clean(self) -> None:
        """The label exemption blanked the first dash of ANY cell, so a
        correct pair became odd and failed the build."""
        self.assertEqual(emdash.lone_dash_lines(
            "| Every deliverable — discovery, apps — follows |"), [])

    def test_the_label_ceiling_is_sixty_characters(self) -> None:
        """Past sixty characters a leading dash is a clause, not a label."""
        # The cell text includes its padding: " " + term + " " precedes the dash.
        self.assertEqual(emdash.lone_dash_lines(f"| {'x' * 58} — definition |"), [])
        self.assertEqual(
            len(emdash.lone_dash_lines(f"| {'x' * 59} — definition |")), 1)

    def test_the_label_exception_still_works_beside_the_carve_out(self) -> None:
        """The two exemptions are independent; neither may disarm the other."""
        self.assertEqual(
            emdash.lone_dash_blocks("| AGC — Atlassian Government Cloud | yes |", CITE_RE), [])
        self.assertEqual(
            len(emdash.lone_dash_blocks("AGC — Atlassian Government Cloud is it.", CITE_RE)), 1)

    def test_an_unmatchable_pattern_disarms_only_the_exemption(self) -> None:
        """A pattern that matches nothing must cost the exemption, nothing else.

        The failing direction matters: no spans means every dash is counted,
        so the caller gets its ordinary noisy verdict. The alternative — an
        exception escaping into a caller's top-level handler, exiting 0 and
        printing nothing — would disarm the WHOLE check silently.
        """
        self.assertEqual(emdash.exempt_spans(self.CITE, r"ZZZ_NO_MATCH"), [])
        self.assertEqual(len(emdash.lone_dash_blocks(self.CITE, r"ZZZ_NO_MATCH")), 1)

    def test_an_invalid_pattern_disarms_only_the_exemption(self) -> None:
        """A malformed regex must not raise out of the walker."""
        self.assertEqual(emdash.exempt_spans(self.CITE, r"([unclosed"), [])
        self.assertEqual(len(emdash.lone_dash_blocks(self.CITE, r"([unclosed")), 1)

    def test_no_pattern_means_no_exemption(self) -> None:
        """The default: a citation separator is counted like any other dash."""
        self.assertEqual(emdash.exempt_spans(self.CITE), [])
        self.assertEqual(len(emdash.lone_dash_blocks(self.CITE)), 1)


if __name__ == "__main__":
    unittest.main()
