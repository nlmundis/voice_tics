"""Tests for prose_lint.py, the linter over the measured structure tics.

The case that matters most is the one the re-aim exists for: a long,
passive, "However"-led academic sentence must produce ZERO findings. The
predecessor failed builds at mean sentence length over 16, and a length
threshold measures register, not authorship (see the README). If anyone
re-adds a length gate, the academic-register test here goes red — that is its
job, not decoration.

The second case that carries weight is line numbers across a code fence. The
corpus scanner's ``scrub`` collapses a multi-line fence to one space, which
is fine for counting and fatal for a linter; ``keep_lines=True`` preserves
geometry, and the fence tests pin both the ignored-inside-code behaviour
and the still-correct line number after the fence.

Run:
    python3 -m unittest discover tests -v
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import unittest
from unittest import mock


def _emdash_cfg(exempt: str = ""):
    """A config with the em-dash rule on, as a user opts into it.

    The rule ships OFF: it is one author's punctuation preference, not a
    measured model tic. These tests therefore enable it explicitly rather
    than relying on a default, which also pins that the default is off.
    """
    import vt_config as config_mod
    return config_mod.Config(emdash_enabled=True, emdash_exempt=exempt)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import prose_lint  # noqa: E402
import voice_tics  # noqa: E402
from tests import support  # noqa: E402


def setUpModule() -> None:
    support.enter_hermetic_cwd()


def tearDownModule() -> None:
    support.leave_hermetic_cwd()


def _rules(found):
    return [f.rule for f in found]


class TicDetection(unittest.TestCase):
    def test_method_defence_is_an_error_with_its_line(self):
        text = "First line is fine.\nThe approach is deliberate here.\n"
        found, _ = prose_lint.lint_text(text)
        self.assertIn("method_defence", _rules(found))
        hit = next(f for f in found if f.rule == "method_defence")
        self.assertEqual(hit.tier, "error")
        self.assertEqual(hit.line, 2)

    def test_not_x_but_y_contracted_form_is_a_warning(self):
        # The contracted surface form is 81% of the model's measured uses;
        # a linter that only saw "is not" would miss the tic it exists for.
        # A warning: in one author's papers the "not A, but B" contrast was
        # ordinary academic prose and did not separate from the model's.
        text = "The problem isn't the code, it's the config.\n"
        found, _ = prose_lint.lint_text(text)
        hit = next(f for f in found if f.rule == "not_x_but_y")
        self.assertEqual(hit.tier, "warning")
        self.assertEqual(hit.line, 1)

    def test_curly_apostrophe_still_caught(self):
        # macOS emits U+2019 by default; SMART_PUNCT folding must reach
        # this caller too, or the linter is dark on most real drafts.
        text = "The problem isn’t the code, it’s the config.\n"
        found, _ = prose_lint.lint_text(text)
        self.assertIn("not_x_but_y", _rules(found))

    def test_appositive_negation_is_an_error(self):
        text = "We chose it by measuring, not guessing.\n"
        found, _ = prose_lint.lint_text(text)
        hit = next(f for f in found if f.rule == "appositive_negation")
        self.assertEqual(hit.tier, "error")

    def test_hedge_adverbs_are_a_warning(self):
        found, _ = prose_lint.lint_text("It simply works.\n")
        hit = next(f for f in found if f.rule == "hedge_adverbs")
        self.assertEqual(hit.tier, "warning")
        self.assertEqual(hit.line, 1)

    def test_lone_em_dash_is_an_error_matched_pair_is_not(self):
        lone = "Read the SOW first — it changes the severity.\n"
        pair = ("Every deliverable — discovery, apps, sign-off — "
                "follows the template.\n")
        found_lone, _ = prose_lint.lint_text(lone, _emdash_cfg())
        found_pair, _ = prose_lint.lint_text(pair, _emdash_cfg())
        self.assertIn("lone_em_dash", _rules(found_lone))
        self.assertEqual(_rules(found_pair), [])

    def test_duplicate_lone_dash_lines_get_distinct_line_numbers(self):
        line = "We shipped it — the client signed off."
        found, _ = prose_lint.lint_text(f"{line}\n{line}\n", _emdash_cfg())
        dashes = [f for f in found if f.rule == "lone_em_dash"]
        self.assertEqual([f.line for f in dashes], [1, 2])
        self.assertEqual({f.tier for f in dashes}, {"error"})

    def test_lone_dash_not_attributed_to_an_earlier_containing_line(self):
        # 'x — y' is a substring of the legal pair on line 1; the reviewed
        # cursor walk pointed the author at the correct line and hid the
        # real one.
        found, _ = prose_lint.lint_text("x — y — z\nx — y\n", _emdash_cfg())
        dashes = [f for f in found if f.rule == "lone_em_dash"]
        self.assertEqual([f.line for f in dashes], [2])

    def test_identical_violating_table_rows_get_their_own_lines(self):
        row = "| Done. shipped — signed |"
        found, _ = prose_lint.lint_text(f"{row}\n{row}\n", _emdash_cfg())
        dashes = [f for f in found if f.rule == "lone_em_dash"]
        self.assertEqual([f.line for f in dashes], [1, 2])

    def test_table_cell_prose_is_linted_for_tics(self):
        # scrub() drops table rows as layout, but a model-drafted
        # assessment carries its findings in tables; a linter blind to
        # cell prose ships the tic in the deliverable.
        text = "| item | The problem isn't the migration, it's the licensing |\n"
        found, _ = prose_lint.lint_text(text)
        hit = next(f for f in found if f.rule == "not_x_but_y")
        self.assertEqual(hit.line, 1)
        self.assertEqual(hit.tier, "warning")

    def test_emphasis_in_a_table_cell_does_not_hide_the_tic(self):
        # Cells go through the same scrub as body prose: "**isn't**"
        # leaves an asterisk where the pattern needs whitespace, so an
        # unscrubbed cell pass linted the bolded tic clean while the
        # identical body sentence failed the run.
        text = "| item | The problem **isn't** the code, it's the config |\n"
        found, _ = prose_lint.lint_text(text)
        self.assertIn("not_x_but_y", _rules(found))

    def test_curly_apostrophe_in_a_table_cell_still_caught(self):
        text = "| item | The problem isn’t the code, it’s the config |\n"
        found, _ = prose_lint.lint_text(text)
        self.assertIn("not_x_but_y", _rules(found))

    def test_table_cell_on_a_later_line_gets_that_line(self):
        text = ("Intro prose.\n"
                "| a | b |\n"
                "| c | The problem isn't the code, it's the config |\n")
        found, _ = prose_lint.lint_text(text)
        hit = next(f for f in found if f.rule == "not_x_but_y")
        self.assertEqual(hit.line, 3)

    def test_row_without_a_trailing_pipe_is_linted_exactly_once(self):
        # Legal GFM. The body and cell passes must be disjoint by ONE
        # definition of "table row" (lib/emdash.is_table_row); splitting
        # on two definitions reported every tic in such rows twice.
        text = "| item | The problem isn't the migration, it's the licensing\n"
        found, _ = prose_lint.lint_text(text)
        hits = [f for f in found if f.rule == "not_x_but_y"]
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].line, 1)

    def test_inline_code_before_the_pipe_still_lints_exactly_once(self):
        # The mask must classify the SAME rendering the cell walk sees
        # (the code-stripped line). Classifying the raw line let both
        # passes lint a row whose leading pipe hides behind inline code.
        text = "`x` | The problem isn't the code, it's the config\n"
        found, _ = prose_lint.lint_text(text)
        hits = [f for f in found if f.rule == "not_x_but_y"]
        self.assertEqual(len(hits), 1)

    def test_empty_backtick_pair_row_lints_exactly_once(self):
        # "``" is literal text in CommonMark, not a code span; the two
        # inline-code regexes disagreeing about it made the renderings
        # disagree about what a table row is.
        text = "``| a | The problem isn't the code, it's the config |\n"
        found, _ = prose_lint.lint_text(text)
        hits = [f for f in found if f.rule == "not_x_but_y"]
        self.assertEqual(len(hits), 1)

    def test_empty_backtick_pair_cell_and_body_verdicts_agree(self):
        # Parity is the cell pass's contract: identical prose must get
        # the identical verdict inside and outside a table.
        prose = "It's not ``, it's b.\n"
        body_found, _ = prose_lint.lint_text(prose)
        cell_found, _ = prose_lint.lint_text(f"| {prose.strip()} |\n")
        self.assertEqual(
            "not_x_but_y" in _rules(body_found),
            "not_x_but_y" in _rules(cell_found))

    def test_masked_row_cannot_splice_a_link_across_it(self):
        # Blanking a row deletes the ")" that blocked a link match in the
        # raw text; with a newline-admitting target class, MD_LINK_RE then
        # stitched line 1 to line 3's first ")" and swallowed the tic
        # between — a silent error drop. Single-line link regexes close
        # the splice class by construction: no scrub pass except the
        # fence pass can cross a line boundary at all.
        text = ("See [t](http://example.com/a\n"
                "| b) | c |\n"
                "The fix isn't luck, it's design (really) and more.\n")
        found, _ = prose_lint.lint_text(text)
        hits = [f for f in found if f.rule == "not_x_but_y"]
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].line, 3)

    def test_masked_row_cannot_splice_a_wikilink_across_it(self):
        # The wikilink twin: the "]" inside the row blocked the match on
        # raw text; masking spliced "[[tic...\n\nb|d]]" together and the
        # tic sat in the discarded target half. It must be reported as
        # the literal prose it renders as, on its own line.
        dropped = ("[[The problem isn't the code, it's the config\n"
                   "| x] |\n"
                   "b|d]]\n")
        found, _ = prose_lint.lint_text(dropped)
        hits = [f for f in found if f.rule == "not_x_but_y"]
        self.assertEqual([f.line for f in hits], [1])
        relocated = ("[[a\n"
                     "| x] |\n"
                     "b|The problem isn't the code, it's the config]]\n")
        found, _ = prose_lint.lint_text(relocated)
        hits = [f for f in found if f.rule == "not_x_but_y"]
        self.assertEqual([f.line for f in hits], [3])

    def test_escaped_pipe_does_not_split_a_cell(self):
        # "\|" is a literal pipe inside one GFM cell and Obsidian's
        # required alias syntax for a wikilink in a table cell; a naive
        # split cut the sentence in half and lost the tic entirely.
        text = ("| Finding |\n"
                "| ------- |\n"
                "| The problem isn't the API \\| v2 shim, it's the config\n")
        found, _ = prose_lint.lint_text(text)
        hits = [f for f in found if f.rule == "not_x_but_y"]
        self.assertEqual([f.line for f in hits], [3])

    def test_aliased_wikilink_in_a_cell_keeps_its_sentence_whole(self):
        text = ("| Finding |\n"
                "| ------- |\n"
                "| The problem isn't [[Shim Notes\\|the shim]], "
                "it's the config\n")
        found, _ = prose_lint.lint_text(text)
        hits = [f for f in found if f.rule == "not_x_but_y"]
        self.assertEqual([f.line for f in hits], [3])

    def test_masked_row_cannot_stitch_a_match_across_itself(self):
        # A masked row must be OPAQUE, not transparent: with the row
        # blanked to an empty line, patterns whose classes admit
        # newlines stitched the lines around it into a finding the
        # rendered document does not contain — a manufactured
        # build-failing error. The "." terminator blocks the bridge.
        text = ("The problem is not the code,\n"
                "| filler. filler\n"
                "it's the config.\n")
        found, _ = prose_lint.lint_text(text)
        self.assertEqual([f for f in found if f.rule == "not_x_but_y"], [])

    def test_masked_row_inside_unclosed_fence_cannot_manufacture(self):
        # Renders as one code block to EOF (no closing fence): the
        # reader sees no prose at all, so nothing may be reported.
        text = ("```\n"
                "x is not the code,\n"
                "| filler. filler\n"
                "it's the config.\n")
        found, _ = prose_lint.lint_text(text)
        self.assertEqual([f for f in found if f.rule == "not_x_but_y"], [])

    def test_pipe_led_lazy_continuation_is_a_documented_undercount(self):
        # PINS THE DOCUMENTED IMPERFECTION, not desired behaviour: this
        # renders as one paragraph (no delimiter row, so no table), and
        # the wrap-spanning tic goes uncounted because the line-based
        # walk cannot tell a lazy continuation from a delimiter-less
        # table without paragraph context. If this test starts failing
        # because the tic IS found, the imperfection is fixed — delete
        # this test and the docstring sentence together.
        text = "The problem isn't\n| the config, it's the code.\n"
        found, _ = prose_lint.lint_text(text)
        self.assertEqual([f for f in found if f.rule == "not_x_but_y"], [])

    def test_multiline_wikilink_opening_on_a_row_cannot_drop_a_finding(self):
        # Masking the scrubbed OUTPUT lost this one: the wikilink's
        # display prose relocated onto the masked row line and the
        # error vanished. Masking the input leaves line 2's prose in
        # the body pass. Fail toward visible: the finding must exist.
        text = ("| c [[x\n"
                "y|The problem isn't the code, it's the config]] |\n")
        found, _ = prose_lint.lint_text(text)
        hits = [f for f in found if f.rule == "not_x_but_y"]
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].line, 2)


class LineNumbersOnRealMarkdown(unittest.TestCase):
    """Line numbers must survive layout scrubbing, not just code fences.

    The ^\\s* layout regexes (heading, bullet, table row, blockquote) all
    consume a preceding blank line's newline because \\s matches \\n; the
    first keep_lines implementation preserved only the fence pass, so every
    finding after a blank-line-preceded heading drifted one line per
    occurrence. Each case here places the tic AFTER the construct.
    """

    TIC = "The approach is deliberate here.\n"

    def _line_of_tic(self, text):
        found, _ = prose_lint.lint_text(text)
        return next(f for f in found if f.rule == "method_defence").line

    def test_after_heading(self):
        self.assertEqual(
            self._line_of_tic("Intro.\n\n# Section\n\nMore prose here.\n"
                              + self.TIC), 6)

    def test_after_list(self):
        self.assertEqual(
            self._line_of_tic("Intro.\n\n- one\n- two\n\n" + self.TIC), 6)

    def test_after_table(self):
        self.assertEqual(
            self._line_of_tic("Intro.\n\n| a | b |\n| c | d |\n\n"
                              + self.TIC), 6)

    def test_after_blockquote(self):
        self.assertEqual(
            self._line_of_tic("Intro.\n\n> quoted\n>\n> more\n\n"
                              + self.TIC), 7)

    def test_after_hard_wrapped_link_target(self):
        self.assertEqual(
            self._line_of_tic("See [ref](http://x\n.example) for detail.\n"
                              + self.TIC), 3)

    def test_crlf_line_endings(self):
        text = "Intro.\r\n\r\n# Section\r\n\r\n" + self.TIC
        self.assertEqual(self._line_of_tic(text), 5)


class RegisterIsNeverLinted(unittest.TestCase):
    ACADEMIC = (
        "However, it should be noted that the convergence behaviour of the "
        "second-order scheme, when applied to the transonic flow field "
        "described previously, remains sensitive to the choice of limiter, "
        "and the resulting solutions were therefore examined across a range "
        "of Courant numbers before any conclusion was drawn.\n"
    )

    def test_academic_register_produces_zero_findings(self):
        found, stats = prose_lint.lint_text(self.ACADEMIC)
        self.assertEqual(found, [])
        # The sentence is far past the predecessor's threshold of 16; the
        # number must be visible in stats yet cause nothing.
        self.assertGreater(stats["mean_words_per_sentence"], 16)

    def test_stats_are_info_only_in_the_exit_code(self):
        code, _out = _run_main([_tmp(self.ACADEMIC)])
        self.assertEqual(code, 0)


class CodeFences(unittest.TestCase):
    FENCED = (
        "Intro line.\n"
        "```py\n"
        "x = 1  # which means that this is code\n"
        "y = 2\n"
        "```\n"
        "The problem isn't the code, it's the config.\n"
    )

    def test_tic_inside_a_fence_is_ignored(self):
        found, _ = prose_lint.lint_text(self.FENCED)
        self.assertNotIn("method_defence", _rules(found))

    def test_line_number_after_a_fence_is_exact(self):
        found, _ = prose_lint.lint_text(self.FENCED)
        hit = next(f for f in found if f.rule == "not_x_but_y")
        self.assertEqual(hit.line, 6)

    def test_midline_backtick_runs_are_not_fence_delimiters(self):
        # Per CommonMark a fence delimiter must start its line; a
        # mid-line ``` run is literal text. Unanchored, a stray pair
        # swallowed the prose between them.
        text = ("A ``` x.\n"
                "The approach is deliberate here.\n"
                "C ``` y.\n")
        found, _ = prose_lint.lint_text(text)
        hit = next(f for f in found if f.rule == "method_defence")
        self.assertEqual(hit.line, 2)

    def test_masked_row_cannot_splice_a_fence_across_it(self):
        # Round-five counterexample: masking row 5 deleted the mid-line
        # ``` that closed a spurious fence match, stitching line 1's run
        # to line 9's and swallowing the line-7 error. Rendered, this
        # document has no fence at all; the tic must be reported.
        text = ("A ``` x.\n"
                "\n"
                "| h | b |\n"
                "| - | - |\n"
                "| c | ``` |\n"
                "\n"
                "The approach is deliberate here.\n"
                "\n"
                "C ``` y.\n")
        found, _ = prose_lint.lint_text(text)
        hit = next(f for f in found if f.rule == "method_defence")
        self.assertEqual(hit.line, 7)

    def test_tic_in_a_row_between_midline_backtick_runs_is_found(self):
        # Pins lib/emdash's fence anchoring specifically: an unanchored
        # strip_code would blank this row as spurious fence interior, so
        # it would be a row to neither pass and the error would vanish.
        text = ("x ``` y\n"
                "| a | The problem isn't the code, it's the config |\n"
                "z ``` w\n")
        found, _ = prose_lint.lint_text(text)
        hits = [f for f in found if f.rule == "not_x_but_y"]
        self.assertEqual([f.line for f in hits], [2])

    def test_scrub_keep_lines_preserves_line_count(self):
        self.assertEqual(
            voice_tics.scrub(self.FENCED, keep_lines=True).count("\n"),
            self.FENCED.count("\n"))

    def test_scrub_default_is_unchanged(self):
        self.assertEqual(voice_tics.scrub("a\n```\nb\n```\nc"),
                         voice_tics.scrub("a\n```\nb\n```\nc", keep_lines=False))


class Wiring(unittest.TestCase):
    def test_no_tier_pattern_can_anchor_on_the_mask_terminator(self):
        # The body pass writes a bare "." for each masked table row, and
        # the opaque-mask design rests on no tier pattern being able to
        # match on or across it. That is true of the current tiers by
        # inspection of their atoms, but voice_tics also carries
        # patterns it is NOT true of — rhetorical_self_question's
        # "[.!?]\s+" lead-in would anchor on the mask. This test exists
        # for the day someone promotes such a key into a tier: it fails
        # then, pointing here instead of at a phantom finding.
        probe = "Done.\n.\nthe result? fine either way\n.\n."
        import vt_config as config_mod
        cfg = config_mod.Config.default()
        for key, _tier, rx, _why in prose_lint.tier_patterns(
                cfg.error_tics, cfg.warn_tics):
            self.assertIsNone(
                rx.search(probe),
                f"tier pattern {key!r} can anchor on the mask terminator")

    def test_missing_pattern_key_raises_loudly(self):
        with self.assertRaises(KeyError):
            prose_lint._compiled("no_such_tic")

    def test_every_default_tier_key_exists_in_voice_tics(self):
        # The loud-failure contract, asserted directly: every key the
        # shipped defaults aim at still exists upstream.
        import vt_config as config_mod
        for key in config_mod.DEFAULT_ERROR_TICS + config_mod.DEFAULT_WARN_TICS:
            _rx, why = prose_lint._compiled(key)
            self.assertTrue(why)

    def test_every_default_tier_key_has_recorded_provenance(self):
        """The tiers must stay next to the numbers that set them.

        TIC_PROVENANCE is documentation and config owns membership, so the
        two can drift. A default key with no recorded ratio is a tier nobody
        can audit, which is how a measured threshold decays into a taste.
        """
        import vt_config as config_mod
        for key in config_mod.DEFAULT_ERROR_TICS + config_mod.DEFAULT_WARN_TICS:
            self.assertIn(key, prose_lint.TIC_PROVENANCE)

    def test_emdash_rule_is_off_by_default(self):
        """Shipping it on would impose one author's punctuation on everyone."""
        import vt_config as config_mod
        self.assertFalse(config_mod.Config.default().emdash_enabled)
        found, _ = prose_lint.lint_text("We shipped it — they signed.\n")
        self.assertEqual([f.rule for f in found], [])


class BannedPhrases(unittest.TestCase):
    """House style: the reader's own phrases, not a claim about any model."""

    def _cfg(self, *entries):
        import vt_config as config_mod
        return config_mod.Config(banned=entries)

    BAN = ("load bearing", "does real work", "error")

    def test_a_banned_phrase_is_an_error(self) -> None:
        found, _ = prose_lint.lint_text("The lock is load bearing.\n",
                                        self._cfg(self.BAN))
        self.assertEqual([f.tier for f in found], ["error"])
        self.assertIn("does real work", found[0].why)

    def test_a_hyphen_and_a_space_are_the_same_phrase(self) -> None:
        """Someone banning 'load bearing' means the hyphenated form too.

        The hyphen is a typesetting choice, not a different phrase, and
        requiring both entries is how half the bans silently miss.
        """
        for text in ("a load bearing wall", "a load-bearing wall",
                     "a load  bearing wall"):
            with self.subTest(text=text):
                found, _ = prose_lint.lint_text(text, self._cfg(self.BAN))
                self.assertEqual(len(found), 1, text)

    def test_matching_is_case_insensitive(self) -> None:
        found, _ = prose_lint.lint_text("LOAD-BEARING and Load-Bearing",
                                        self._cfg(self.BAN))
        self.assertEqual(len(found), 2)

    def test_a_word_boundary_is_required_at_both_ends(self) -> None:
        """A ban must not fire inside a longer word at either end.

        Both directions are needed, and the leading one is easy to lose: in
        "overload bearingly" the TRAILING boundary already blocks the match,
        so that string alone cannot tell a leading boundary from none. The
        first case below is the one that can.
        """
        for text in ("an overload bearing down on us",   # leading only
                     "load bearingly odd",                # trailing only
                     "overload bearingly"):               # both
            with self.subTest(text=text):
                found, _ = prose_lint.lint_text(text, self._cfg(self.BAN))
                self.assertEqual(found, [], text)

    def test_punctuation_in_a_phrase_is_literal_not_regex(self) -> None:
        """A phrase is typed by someone sick of it, not by a regex author.

        'C++' as a regex is a repetition error; taken literally it is a
        phrase. Escaping is the tool's job, not the reader's.
        """
        found, _ = prose_lint.lint_text("we still write C++ here",
                                        self._cfg(("C++", "", "error")))
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].excerpt, "C++")

    def test_code_is_not_prose(self) -> None:
        """A banned phrase inside a code sample is not the author's writing."""
        found, _ = prose_lint.lint_text(
            "```\nload bearing\n```\n\nUse `load-bearing` as a term.\n",
            self._cfg(self.BAN))
        self.assertEqual(found, [])

    def test_the_tier_is_configurable(self) -> None:
        found, _ = prose_lint.lint_text(
            "load bearing", self._cfg(("load bearing", "x", "warning")))
        self.assertEqual([f.tier for f in found], ["warning"])

    def test_nothing_is_banned_by_default(self) -> None:
        """A shipped ban list would be one person's taste imposed on everyone.

        This is the same rule as the empty signature table and the em-dash
        default: the repo ships measurements, and preferences are the
        reader's to add.
        """
        import vt_config as config_mod
        self.assertEqual(config_mod.Config.default().banned, ())
        found, _ = prose_lint.lint_text("This is load-bearing.\n")
        self.assertEqual(found, [])

    def test_line_numbers_survive_a_code_fence(self) -> None:
        text = "intro\n\n```\nx\ny\n```\n\nload bearing here\n"
        found, _ = prose_lint.lint_text(text, self._cfg(self.BAN))
        self.assertEqual([f.line for f in found], [8])

    def test_a_banned_finding_is_named_after_its_phrase(self) -> None:
        """So a JSON consumer can tell two bans apart, and tell them from tics."""
        found, _ = prose_lint.lint_text("load-bearing", self._cfg(self.BAN))
        self.assertEqual(found[0].rule, "banned:load bearing")


class OneEngineForEveryRule(unittest.TestCase):
    """Tics and banned phrases read a document the same way."""

    def _cfg(self, **kwargs):
        import vt_config as config_mod
        return config_mod.Config(**kwargs)

    def test_a_banned_phrase_in_a_well_formed_table_row_is_found(self) -> None:
        """scrub blanks complete rows, so the old whole-document pass never
        saw one; a row missing its trailing pipe was caught instead."""
        cfg = self._cfg(banned=(("load bearing", "", "error"),))
        found, _ = prose_lint.lint_text(
            "| a | b |\n|---|---|\n| this cell is load bearing | x |\n", cfg)
        self.assertEqual([(f.line, f.rule) for f in found],
                         [(3, "banned:load bearing")])

    def test_no_finding_is_placed_on_a_table_row_it_does_not_hold(self) -> None:
        """The "." row mask was matched as a sentence terminator."""
        cfg = self._cfg(error_tics=("rhetorical_self_question",), warn_tics=())
        found, _ = prose_lint.lint_text(
            "| col | why |\n|---|---|\n| alpha | beta |\n"
            "The result? Nothing separated.\n", cfg)
        self.assertEqual([(f.line, f.excerpt) for f in found],
                         [(4, "The result?")])

    def test_a_finding_is_on_the_line_of_its_words(self) -> None:
        """Not the line whose full stop the pattern opened with."""
        cfg = self._cfg(error_tics=("rhetorical_self_question",), warn_tics=())
        found, _ = prose_lint.lint_text(
            "This approach was tested carefully.\nThe result? Nothing.\n", cfg)
        self.assertEqual([(f.line, f.excerpt) for f in found],
                         [(2, "The result?")])

    def test_no_match_crosses_a_table(self) -> None:
        """Wrapped across a plain line break this IS the tic; across a table
        it is two unrelated lines, and an empty-line mask stitched them."""
        cfg = self._cfg(error_tics=("not_x_but_y",), warn_tics=())
        wrapped, _ = prose_lint.lint_text(
            "It is not the code,\nit's the config.\n", cfg)
        self.assertEqual(len(wrapped), 1)
        found, _ = prose_lint.lint_text(
            "It is not the code,\n| a | b |\nit's the config.\n", cfg)
        self.assertEqual(found, [])

    def test_code_in_a_list_item_fence_is_not_prose(self) -> None:
        found, _ = prose_lint.lint_text(
            "- Install it:\n\n    ```bash\n    # which means that this is code\n"
            "    ```\n\n- Done.\n")
        self.assertEqual(found, [])

    def test_code_in_a_no_break_space_indented_fence_is_not_a_cell(self) -> None:
        found, _ = prose_lint.lint_text(
            "Intro.\n\n\u00a0```\n| a which means that b | c |\n```\n\nEnd.\n")
        self.assertEqual(found, [])

    def test_crlf_on_stdin_is_linted_like_a_file(self) -> None:
        cfg = _tmp_cfg("[lint.emdash]\nenabled = true\n")
        with mock.patch("sys.stdin", io.StringIO(
                "Intro.\r\n```\r\nx = a — b\r\n```\r\nTail.\r\n")):
            code, out = _run_main(["--config", cfg, "-"])
        self.assertEqual(code, 0, out)


class BannedPhraseConfig(unittest.TestCase):
    """Validation of [lint] banned."""

    def _parse(self, banned):
        import vt_config as config_mod
        return config_mod.Config.parse({"lint": {"banned": banned}})

    def test_a_well_formed_entry_round_trips(self) -> None:
        cfg = self._parse([{"phrase": "blast radius", "instead": "scope"}])
        self.assertEqual(cfg.banned, (("blast radius", "scope", "error"),))

    def test_instead_is_optional(self) -> None:
        cfg = self._parse([{"phrase": "blast radius"}])
        self.assertEqual(cfg.banned[0][1], "")

    def test_an_empty_phrase_is_rejected(self) -> None:
        import vt_config as config_mod
        with self.assertRaises(config_mod.ConfigError):
            self._parse([{"phrase": "   "}])

    def test_an_unknown_tier_is_rejected(self) -> None:
        import vt_config as config_mod
        with self.assertRaises(config_mod.ConfigError) as ctx:
            self._parse([{"phrase": "x", "tier": "fatal"}])
        self.assertIn("error", str(ctx.exception))

    def test_a_repeated_phrase_is_rejected(self) -> None:
        """Differing only in spacing still collides once normalised."""
        import vt_config as config_mod
        with self.assertRaises(config_mod.ConfigError):
            self._parse([{"phrase": "load bearing"},
                         {"phrase": "Load  Bearing"}])

    def test_an_unknown_key_is_rejected(self) -> None:
        import vt_config as config_mod
        with self.assertRaises(config_mod.ConfigError):
            self._parse([{"phrase": "x", "replace": "y"}])


    def test_a_phrase_with_no_words_is_rejected(self) -> None:
        """"--" compiled to an empty pattern: an error at every character."""
        import vt_config as config_mod
        for phrase in ("-", "--", " - ", "- -"):
            with self.subTest(phrase=phrase):
                with self.assertRaises(config_mod.ConfigError):
                    self._parse([{"phrase": phrase}])

    def test_hyphen_and_space_spellings_are_one_phrase(self) -> None:
        """They compile to one regex, so both would report every hit twice,
        and the default-error twin overrode a warning the user asked for."""
        import vt_config as config_mod
        with self.assertRaises(config_mod.ConfigError):
            self._parse([{"phrase": "load-bearing"},
                         {"phrase": "load bearing", "tier": "warning"}])

    def test_the_rule_name_does_not_depend_on_the_spelling(self) -> None:
        cfg = self._parse([{"phrase": "Load-Bearing"}])
        found, _ = prose_lint.lint_text("A load bearing wall.\n", cfg)
        self.assertEqual([f.rule for f in found], ["banned:load bearing"])


class TierConfig(unittest.TestCase):
    """[lint] error and [lint] warn: a rule has exactly one tier."""

    def _parse(self, lint):
        import vt_config as config_mod
        return config_mod.Config.parse({"lint": lint})

    def test_a_key_in_both_tiers_is_rejected(self) -> None:
        import vt_config as config_mod
        with self.assertRaises(config_mod.ConfigError) as caught:
            self._parse({"error": ["not_x_but_y"], "warn": ["not_x_but_y"]})
        self.assertIn("not_x_but_y", str(caught.exception))

    def test_a_key_repeated_in_one_tier_is_rejected(self) -> None:
        import vt_config as config_mod
        for tier in ("error", "warn"):
            with self.subTest(tier=tier):
                with self.assertRaises(config_mod.ConfigError):
                    self._parse({tier: ["let_me", "let_me"]})

    def test_a_default_tier_key_listed_in_the_other_is_rejected(self) -> None:
        """Moving a default error key to warn without also setting error."""
        import vt_config as config_mod
        with self.assertRaises(config_mod.ConfigError):
            self._parse({"warn": ["method_defence"]})

    def test_disjoint_tiers_load(self) -> None:
        cfg = self._parse({"error": ["let_me"], "warn": ["not_x_but_y"]})
        self.assertEqual((cfg.error_tics, cfg.warn_tics),
                         (("let_me",), ("not_x_but_y",)))


class ParentConfigNote(unittest.TestCase):
    """Config is read from the working directory only, and says so."""

    def setUp(self) -> None:
        import tempfile
        self._tmp = tempfile.TemporaryDirectory()
        self.root = os.path.realpath(self._tmp.name)
        self.sub = os.path.join(self.root, "a", "b")
        os.makedirs(self.sub)
        with open(os.path.join(self.root, "voice_tics.toml"), "w") as fh:
            fh.write('[lint]\nbanned = [ { phrase = "blast radius" } ]\n')

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_a_parent_config_is_named(self) -> None:
        import vt_config as config_mod
        note = config_mod.unread_parent_config("", start=self.sub)
        self.assertIn(os.path.join(self.root, "voice_tics.toml"), note)
        self.assertIn("--config", note)

    def test_no_note_when_a_config_is_named_or_local(self) -> None:
        import vt_config as config_mod
        self.assertEqual(config_mod.unread_parent_config("x.toml", start=self.sub), "")
        self.assertEqual(config_mod.unread_parent_config("", start=self.root), "")

    def test_the_linter_prints_the_note_and_still_uses_defaults(self) -> None:
        cwd = os.getcwd()
        err = io.StringIO()
        doc = os.path.join(self.sub, "doc.md")
        with open(doc, "w") as fh:
            fh.write("The blast radius is small.\n")
        try:
            os.chdir(self.sub)
            with contextlib.redirect_stderr(err):
                code, out = _run_main([doc])
        finally:
            os.chdir(cwd)
        self.assertEqual(code, 0)
        self.assertNotIn("banned", out)
        self.assertIn("was not read", err.getvalue())


def _tmp_cfg(text: str) -> str:
    """Write a throwaway config file and return its path."""
    return support.scratch_file(text, suffix=".toml")


def _tmp(text: str) -> str:
    """Write a throwaway markdown file and return its path."""
    return support.scratch_file(text, suffix=".md")


def _run_main(argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = prose_lint.main(argv)
    return code, buf.getvalue()


class Cli(unittest.TestCase):
    CLEAN = "A perfectly ordinary sentence.\n"
    WARN_ONLY = "The problem isn't the code, it's the config.\n"
    ERRORED = "The approach is deliberate here.\n"

    def test_exit_codes(self):
        self.assertEqual(_run_main([_tmp(self.CLEAN)])[0], 0)
        self.assertEqual(_run_main([_tmp(self.WARN_ONLY)])[0], 0)
        self.assertEqual(_run_main([_tmp(self.ERRORED)])[0], 1)

    def test_strict_promotes_warnings(self):
        self.assertEqual(_run_main(["--strict", _tmp(self.WARN_ONLY)])[0], 1)

    def test_lone_em_dash_alone_fails_the_run(self):
        # The dash rule reaches the exit code through the same tier
        # machinery as the pattern tics; if a future edit demotes it to a
        # warning, this is the test that goes red.
        cfg = _tmp_cfg("[lint.emdash]\nenabled = true\n")
        code, _ = _run_main([_tmp("We shipped — the client signed.\n"),
                             "--config", cfg])
        self.assertEqual(code, 1)

    def test_multi_file_worst_of_aggregation(self):
        # A clean last file must not overwrite an earlier error.
        self.assertEqual(
            _run_main([_tmp(self.ERRORED), _tmp(self.CLEAN)])[0], 1)
        self.assertEqual(
            _run_main([_tmp(self.CLEAN), _tmp(self.CLEAN)])[0], 0)

    def test_stdin_dash_reads_stdin(self):
        real = sys.stdin
        sys.stdin = io.StringIO(self.ERRORED)
        try:
            code, out = _run_main(["-"])
        finally:
            sys.stdin = real
        self.assertEqual(code, 1)
        self.assertIn("method_defence", out)

    def test_a_retired_tic_key_is_a_usage_error_before_any_file(self):
        """Exit 2 with a message, not a KeyError traceback from the first
        file that reaches the rule."""
        cfg = _tmp_cfg('[lint]\nerror = ["no_such_detector"]\nwarn = []\n')
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            code, _ = _run_main(["--config", cfg, _tmp(self.CLEAN)])
        self.assertEqual(code, 2)
        self.assertIn("no_such_detector", err.getvalue())

    def test_missing_file_is_a_usage_error(self):
        with contextlib.redirect_stderr(io.StringIO()):
            code, _ = _run_main(["/nonexistent/definitely_not_here.md"])
        self.assertEqual(code, 2)

    def test_undecodable_file_is_a_usage_error_not_a_lint_failure(self):
        # Exit 1 means "the prose has tics"; a latin-1 file must not be
        # recorded as a prose violation by a pipeline keyed on that code.
        path = support.scratch_file(b"caf\xe9 test\n")  # latin-1 e-acute
        with contextlib.redirect_stderr(io.StringIO()):
            code, _ = _run_main([path])
        self.assertEqual(code, 2)

    def test_json_shape(self):
        code, out = _run_main(["--json", _tmp(self.ERRORED)])
        payload = json.loads(out)
        self.assertEqual(payload["exit"], 1)
        self.assertEqual(payload["exit"], code)
        (entry,) = payload["files"]
        self.assertEqual(entry["findings"][0]["rule"], "method_defence")
        self.assertIn("mean_words_per_sentence", entry["stats"])

    def test_json_exit_field_agrees_with_process_exit_when_clean(self):
        code, out = _run_main(["--json", _tmp(self.CLEAN)])
        self.assertEqual(json.loads(out)["exit"], code)
        self.assertEqual(code, 0)

    def test_report_names_the_info_line_as_never_failing(self):
        _code, out = _run_main([_tmp(self.CLEAN)])
        self.assertIn("never a failure", out)


if __name__ == "__main__":
    unittest.main()
