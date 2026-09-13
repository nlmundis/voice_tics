"""The sample-baseline path: reading your own writing instead of transcripts.

Covers the reader, the contamination guard, and the config keys that reach
them. The guard gets both directions on purpose — a guard only ever tested on
the quiet case is a guard nobody has seen work.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import vt_config as config_mod  # noqa: E402
import voice_tics as vt  # noqa: E402


class ReadSamplesTest(unittest.TestCase):
    """What counts as a sample paragraph, and what is skipped."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _write(self, name: str, text: str) -> None:
        (self.tmp / name).write_text(text, encoding="utf-8")

    def test_paragraphs_are_units_not_files(self) -> None:
        """Each blank-line-separated paragraph is one unit.

        Whole-file units would give one opener per document, and an opener
        distribution built from a handful of observations is not one.
        """
        self._write("a.md", "First para here.\n\nSecond para here.\n\nThird.")
        got = list(vt.read_samples(str(self.tmp / "*.md")))
        self.assertEqual(len(got), 3)

    def test_front_matter_is_not_prose(self) -> None:
        """YAML keys would otherwise rank as phrases."""
        self._write("a.md", "---\ntags: [alpha]\ndate: 2026-01-01\n---\n"
                            "The actual sentence.")
        got = list(vt.read_samples(str(self.tmp / "*.md")))
        self.assertEqual(len(got), 1)
        self.assertNotIn("tags", got[0])
        self.assertIn("actual", got[0])

    def test_front_matter_only_at_the_very_start(self) -> None:
        """A horizontal rule mid-document is not front matter."""
        self._write("a.md", "Opening line.\n\n---\n\nStill my prose.")
        joined = " ".join(vt.read_samples(str(self.tmp / "*.md")))
        self.assertIn("Opening", joined)
        self.assertIn("Still", joined)

    def test_code_is_stripped_like_every_other_path(self) -> None:
        self._write("a.md", "Prose here.\n\n```\ndef f(): return 1\n```\n")
        joined = " ".join(vt.read_samples(str(self.tmp / "*.md")))
        self.assertNotIn("return", joined)

    def test_unreadable_file_is_counted_not_fatal(self) -> None:
        """One bad file must not cost the whole measurement."""
        self._write("good.md", "Real prose in here.")
        (self.tmp / "bad.md").write_bytes(b"\xff\xfe\x00 not utf-8 \xff")
        stats: dict = {}
        got = list(vt.read_samples(str(self.tmp / "*.md"), stats=stats))
        self.assertEqual(stats.get("sample_unreadable"), 1)
        self.assertEqual(stats.get("sample_files"), 1)
        self.assertTrue(got)

    def test_stats_count_files_and_paragraphs(self) -> None:
        self._write("a.md", "One.\n\nTwo.")
        self._write("b.md", "Three.")
        stats: dict = {}
        list(vt.read_samples(str(self.tmp / "*.md"), stats=stats))
        self.assertEqual(stats["sample_files"], 2)
        self.assertEqual(stats["sample_paragraphs"], 3)

    def test_empty_and_whitespace_paragraphs_are_not_units(self) -> None:
        self._write("a.md", "Real prose.\n\n   \n\n\n\nMore prose.")
        self.assertEqual(len(list(vt.read_samples(str(self.tmp / "*.md")))), 2)


class ContaminationWarningTest(unittest.TestCase):
    """The guard must fire on a parity-heavy corpus and stay quiet otherwise."""

    def _corpora(self, words: int = 10000):
        mine, theirs = vt.Corpus("model"), vt.Corpus("baseline")
        mine.total_words = words
        theirs.total_words = words
        return mine, theirs

    def test_fires_when_most_structures_sit_at_parity(self) -> None:
        """Equal rates across the board is what a shared author looks like."""
        mine, theirs = self._corpora()
        structures = {f"s{i}": (100, 100) for i in range(10)}
        got = vt.contamination_warning(structures, mine, theirs)
        self.assertIsNotNone(got)
        assert got is not None
        self.assertIn("10 of 10", got)

    def test_quiet_when_the_corpora_separate(self) -> None:
        mine, theirs = self._corpora()
        structures = {f"s{i}": (800, 100) for i in range(10)}
        self.assertIsNone(vt.contamination_warning(structures, mine, theirs))

    def test_quiet_below_the_share_threshold(self) -> None:
        """Half at parity is not enough; the guard is deliberately reluctant."""
        mine, theirs = self._corpora()
        structures = {f"near{i}": (100, 100) for i in range(5)}
        structures.update({f"far{i}": (900, 100) for i in range(5)})
        self.assertIsNone(vt.contamination_warning(structures, mine, theirs))

    def test_quiet_on_too_little_data(self) -> None:
        """Four comparable structures cannot support the claim."""
        mine, theirs = self._corpora()
        structures = {f"s{i}": (100, 100) for i in range(4)}
        self.assertIsNone(vt.contamination_warning(structures, mine, theirs))

    def test_one_sided_structures_are_not_comparable(self) -> None:
        """A zero on either side is the floor correction, not the author.

        Ten structures the model alone uses must not be read as ten
        agreements; they are not comparable at all, so the guard has too
        little data and says nothing.
        """
        mine, theirs = self._corpora()
        structures = {f"s{i}": (100, 0) for i in range(10)}
        self.assertIsNone(vt.contamination_warning(structures, mine, theirs))

    def test_structures_only_the_baseline_uses_are_not_comparable(self) -> None:
        """A zero on the MODEL side must be excluded, not scored as 0.0x.

        Counting it would drag the parity share down and silence the guard on
        a corpus that genuinely is contaminated. The mirror case (zero on the
        baseline side) is skipped anyway by the zero-rate check, so this is
        the direction that actually distinguishes the two.
        """
        mine, theirs = self._corpora()
        structures = {f"near{i}": (100, 100) for i in range(8)}
        structures.update({f"baseline_only{i}": (0, 100) for i in range(8)})
        got = vt.contamination_warning(structures, mine, theirs)
        self.assertIsNotNone(got, "one-sided structures must not dilute the share")
        assert got is not None
        self.assertIn("8 of 8", got)

    def test_band_is_symmetric_in_log_space(self) -> None:
        """0.8 and 1.25 are the same distance from parity, inverted."""
        low, high = vt.INDISTINGUISHABLE
        self.assertAlmostEqual(low * high, 1.0, places=6)


class SamplesConfigTest(unittest.TestCase):
    """The [baseline] samples key and its validation."""

    def test_samples_default_empty_means_use_transcripts(self) -> None:
        self.assertEqual(config_mod.Config.default().samples, "")

    def test_samples_is_expanded(self) -> None:
        cfg = config_mod.Config.parse({"baseline": {"samples": "~/w/*.md"}})
        self.assertTrue(cfg.samples.startswith(os.path.expanduser("~")))

    def test_samples_must_be_a_string(self) -> None:
        with self.assertRaises(config_mod.ConfigError):
            config_mod.Config.parse({"baseline": {"samples": ["a", "b"]}})

    def test_duplicate_signature_names_are_rejected(self) -> None:
        """Counts are keyed by name, so a repeat would silently replace."""
        with self.assertRaises(config_mod.ConfigError) as ctx:
            config_mod.Config.parse({"baseline": {"signatures": [
                {"name": "however", "pattern": r"\bhowever\b"},
                {"name": "however", "pattern": r"\bHowever\b"},
            ]}})
        self.assertIn("reuses the name", str(ctx.exception))

    def test_a_named_config_file_that_is_missing_is_an_error(self) -> None:
        """A --config the user typed and the tool ignored is worse than a crash."""
        with self.assertRaises(config_mod.ConfigError):
            config_mod.load("/nonexistent/voice_tics.toml")

    def test_no_config_file_at_all_is_not_an_error(self) -> None:
        """The zero-config case: defaults, not a crash."""
        import tempfile
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as empty:
            try:
                os.chdir(empty)
                self.assertEqual(config_mod.load().signatures, ())
            finally:
                os.chdir(cwd)

    def test_unknown_baseline_key_is_rejected_with_a_suggestion(self) -> None:
        with self.assertRaises(config_mod.ConfigError) as ctx:
            config_mod.Config.parse({"baseline": {"sample": "x"}})
        self.assertIn("samples", str(ctx.exception))


class SampleBaselineEndToEndTest(unittest.TestCase):
    """--baseline-from must replace the transcript turns, not blend with them."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        import json
        recs = []
        for role, text in (("user", "check it"),
                           ("assistant", "Let me check. Let me verify.")):
            body = ({"role": "user", "content": text} if role == "user"
                    else {"role": "assistant",
                          "content": [{"type": "text", "text": text}],
                          "model": "claude-test"})
            recs.append(json.dumps({
                "type": role, "message": body,
                "timestamp": "2026-09-01T00:00:00Z", "sessionId": "s1"}))
        (self.tmp / "s1.jsonl").write_text("\n".join(recs), encoding="utf-8")
        (self.tmp / "mine.md").write_text(
            "I wrote this paragraph myself and it is entirely my own prose.\n\n"
            "Here is a second paragraph of my writing for the corpus.\n",
            encoding="utf-8")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _run(self, extra):
        import io
        import json
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = vt.main(["--glob", str(self.tmp / "*.jsonl"), "--json",
                          "--min-count", "1"] + extra)
        self.assertEqual(rc, 0)
        return json.loads(buf.getvalue())

    def test_transcript_baseline_is_the_default_and_is_labelled(self) -> None:
        got = self._run([])
        self.assertEqual(got["baseline_source"], "transcripts")

    def test_sample_baseline_replaces_the_user_turns(self) -> None:
        """The user turn must not also land in the baseline.

        Blending would build a corpus that is neither register and weight it
        by whichever side happened to be larger — a number nobody could read.
        The user turn is "check it"; the samples never say it.
        """
        got = self._run(["--baseline-from", str(self.tmp / "*.md")])
        self.assertEqual(got["baseline_source"], str(self.tmp / "*.md"))
        self.assertEqual(got["stats"]["sample_files"], 1)
        self.assertEqual(got["stats"]["sample_paragraphs"], 2)
        # Two sample paragraphs and nothing else: the user turn is excluded.
        self.assertEqual(got["baseline"]["turns"], 2)

    def test_a_glob_matching_no_prose_fails_loudly(self) -> None:
        """Silently measuring against an empty baseline would be worse."""
        import contextlib
        import io
        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            rc = vt.main(["--glob", str(self.tmp / "*.jsonl"),
                          "--baseline-from", str(self.tmp / "nothing*.md")])
        self.assertEqual(rc, 1)
        self.assertIn("no sample prose matched", err.getvalue())


if __name__ == "__main__":
    unittest.main()
