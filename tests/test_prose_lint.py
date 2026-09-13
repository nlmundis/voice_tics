"""Tests for prose_lint.py, the linter over the measured structure tics.

The case that matters most is the one the re-aim exists for: a long,
passive, "However"-led academic sentence must produce ZERO findings. The
predecessor failed builds at mean sentence length over 16, and the matched
baseline showed that threshold detects the author's register, not the model
(13.89 vs 13.85 words/sentence). If anyone re-adds a length gate, the
academic-register test here goes red — that is its job, not decoration.

The second load-bearing case is line numbers across a code fence. The
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
import tempfile
import unittest


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

    def test_not_x_but_y_contracted_form(self):
        # The contracted surface form is 81% of the model's measured uses;
        # a linter that only saw "is not" would miss the tic it exists for.
        text = "The problem isn't the code, it's the config.\n"
        found, _ = prose_lint.lint_text(text)
        hit = next(f for f in found if f.rule == "not_x_but_y")
        self.assertEqual(hit.tier, "error")
        self.assertEqual(hit.line, 1)

    def test_curly_apostrophe_still_caught(self):
        # macOS emits U+2019 by default; SMART_PUNCT folding must reach
        # this caller too, or the linter is dark on most real drafts.
        text = "The problem isn’t the code, it’s the config.\n"
        found, _ = prose_lint.lint_text(text)
        self.assertIn("not_x_but_y", _rules(found))

    def test_appositive_negation_is_a_warning(self):
        text = "We chose it by measuring, not guessing.\n"
        found, _ = prose_lint.lint_text(text)
        hit = next(f for f in found if f.rule == "appositive_negation")
        self.assertEqual(hit.tier, "warning")

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
        self.assertEqual(hit.tier, "error")

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


def _tmp_cfg(text: str) -> str:
    """Write a throwaway voice_tics.toml and return its path."""
    fh = tempfile.NamedTemporaryFile(
        "w", suffix=".toml", delete=False, encoding="utf-8")
    fh.write(text)
    fh.close()
    return fh.name


def _tmp(text: str) -> str:
    fh = tempfile.NamedTemporaryFile(
        "w", suffix=".md", delete=False, encoding="utf-8")
    fh.write(text)
    fh.close()
    return fh.name


def _run_main(argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = prose_lint.main(argv)
    return code, buf.getvalue()


class Cli(unittest.TestCase):
    CLEAN = "A perfectly ordinary sentence.\n"
    WARN_ONLY = "We chose it by measuring, not guessing.\n"
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

    def test_missing_file_is_a_usage_error(self):
        with contextlib.redirect_stderr(io.StringIO()):
            code, _ = _run_main(["/nonexistent/definitely_not_here.md"])
        self.assertEqual(code, 2)

    def test_undecodable_file_is_a_usage_error_not_a_lint_failure(self):
        # Exit 1 means "the prose has tics"; a latin-1 file must not be
        # recorded as a prose violation by a pipeline keyed on that code.
        fh = tempfile.NamedTemporaryFile("wb", suffix=".md", delete=False)
        fh.write(b"caf\xe9 test\n")  # latin-1 e-acute: invalid as UTF-8
        fh.close()
        with contextlib.redirect_stderr(io.StringIO()):
            code, _ = _run_main([fh.name])
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
