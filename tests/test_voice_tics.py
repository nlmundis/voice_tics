"""Tests for voice_tics.py.

Run with:
    python3 -m unittest discover tests -v

The contamination cases here are regression tests, not hypotheticals. Both were
found by running the scan and reading a result that was obviously wrong:

* The first model-direction run ranked "you've hit your" as a top-30 tic at
  113x and put it second in the opener table. It is the harness's rate-limit
  notice stored in an assistant record, not prose the model composed.
* The first baseline-direction run ranked "execute autonomously without asking
  clarifying questions", "is an automated run" and "task the user is not
  present" as the human's top tics at ~364x. Those come from recurring
  scheduled-task prompts, logged as user records. 130 of 222 sessions in the
  45-day window turned out to be scheduled runs.

Both would have been reported as findings with a number attached, which is the
failure this suite exists to prevent: a measured-looking number sourced from the
harness rather than from a speaker.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import unittest
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import emdash  # noqa: E402
import voice_tics as vt  # noqa: E402
from tests import support  # noqa: E402


def setUpModule() -> None:
    support.enter_hermetic_cwd()


def tearDownModule() -> None:
    support.leave_hermetic_cwd()


def _rec(kind: str, text: str, *, sidechain: bool = False,
         blocks: Any = None, meta: bool = False,
         model: str | None = None,
         ts: str = "2026-08-12T10:00:00.000Z",
         uuid: str | None = None) -> str:
    """One transcript line as the harness writes it."""
    content: Any
    if blocks is not None:
        content = blocks
    elif kind == "assistant":
        content = [{"type": "text", "text": text}]
    else:
        content = text
    rec: Dict[str, Any] = {
        "type": kind,
        "isSidechain": sidechain,
        "timestamp": ts,
        "message": {"role": kind, "content": content},
    }
    if meta:
        rec["isMeta"] = True
    if model is not None:
        rec["message"]["model"] = model
    if uuid is not None:
        rec["uuid"] = uuid
    return json.dumps(rec)


def _replayed(line: str) -> str:
    """A copy of a transcript line as a resume writes it back into the file.

    The uuid, timestamp and message stay the original's. The fields observed
    to differ in real replays are rewritten: ``slug`` and ``cwd`` to their
    resume-time values, and ``parentUuid``, ``promptId`` and
    ``toolUseResult``, which a replay re-links or re-records.
    """
    rec = json.loads(line)
    rec["slug"] = "serialized-swinging-flute"
    rec["cwd"] = "/resumed/from/here"
    rec["parentUuid"] = "relinked-on-resume"
    rec["promptId"] = "prompt-after-resume"
    rec["toolUseResult"] = {"rerecorded": True}
    return json.dumps(rec)


def _session(lines: List[str], tmp: pathlib.Path, name: str = "s.jsonl") -> str:
    """Write transcript lines to a file and return its path."""
    path = tmp / name
    path.write_text("\n".join(lines) + "\n")
    return str(path)


class ScrubTest(unittest.TestCase):
    """scrub() must leave prose and remove everything that is not prose."""

    def test_strips_fenced_code(self) -> None:
        got = vt.scrub("before\n```python\nlet me = 1\n```\nafter")
        self.assertNotIn("let me", got)
        self.assertIn("before", got)
        self.assertIn("after", got)

    def test_strips_inline_code_and_urls_and_paths(self) -> None:
        got = vt.scrub("see `let me` at https://x.com/let/me and ~/a/b/let_me.py")
        self.assertNotIn("let me", got)
        self.assertNotIn("x.com", got)
        self.assertNotIn("let_me.py", got)

    def test_keeps_link_text_drops_target(self) -> None:
        got = vt.scrub("read [the findings](https://example.com/let/me)")
        self.assertIn("the findings", got)
        self.assertNotIn("example.com", got)

    def test_keeps_wikilink_text(self) -> None:
        self.assertIn("Writing Style", vt.scrub("see [[Writing Style]]"))
        self.assertIn("Writing Style", vt.scrub("see [[Resources|Writing Style]]"))

    def test_strips_markdown_layout_not_words(self) -> None:
        got = vt.scrub("## Heading\n- **bold** item\n> quoted\n")
        for token in ("#", "-", "**", ">"):
            self.assertNotIn(token, got)
        for word in ("Heading", "bold", "item", "quoted"):
            self.assertIn(word, got)


class TokeniseTest(unittest.TestCase):
    """words() and sentences() define every rate in the report."""

    def test_words_are_lowercased_alpha(self) -> None:
        self.assertEqual(vt.words("Let ME check 42 times!"),
                         ["let", "me", "check", "times"])

    def test_sentences_split_on_terminals_and_newlines(self) -> None:
        self.assertEqual(len(vt.sentences("One. Two! Three?")), 3)
        self.assertEqual(len(vt.sentences("One\nTwo\n\nThree")), 3)

    def test_blank_input_yields_nothing(self) -> None:
        self.assertEqual(vt.sentences("   \n  "), [])
        self.assertEqual(vt.words(""), [])


class UserTextTest(unittest.TestCase):
    """Only prose the human actually typed may enter the baseline corpus."""

    def test_plain_string_passes(self) -> None:
        self.assertEqual(vt._user_text("just do it"), "just do it")

    def test_harness_furniture_is_dropped(self) -> None:
        for noisy in ("<system-reminder>stuff</system-reminder>",
                      "<command-name>/model</command-name>",
                      "UserPromptSubmit hook success: ...",
                      "[Request interrupted by user]"):
            self.assertIsNone(vt._user_text(noisy), noisy)

    def test_tool_result_blocks_are_dropped_wholesale(self) -> None:
        blocks = [{"type": "tool_result", "content": "output"}]
        self.assertIsNone(vt._user_text(blocks))

    def test_list_of_text_blocks_passes(self) -> None:
        """The majority shape of real user records: typed prose arrives as
        [{'type': 'text', ...}] blocks, not a bare string. This branch was
        previously covered by no test at all."""
        blocks = [{"type": "text", "text": "one"},
                  {"type": "text", "text": "two"}]
        self.assertEqual(vt._user_text(blocks), "one\ntwo")

    def test_image_with_caption_keeps_the_typed_text(self) -> None:
        """Regression: ANY non-text block used to drop the whole record, so
        the caption the human typed beside a screenshot was silently lost from
        his baseline."""
        blocks = [{"type": "image", "source": {"type": "base64"}},
                  {"type": "text", "text": "here are the modes"}]
        self.assertEqual(vt._user_text(blocks), "here are the modes")

    def test_image_only_record_yields_nothing(self) -> None:
        self.assertIsNone(vt._user_text([{"type": "image", "source": {}}]))

    def test_tool_result_beside_text_still_drops_wholesale(self) -> None:
        """A tool round-trip is machine output end to end even when a text
        block rides along; only non-tool blocks get the skip treatment."""
        blocks = [{"type": "tool_result", "content": "output"},
                  {"type": "text", "text": "words"}]
        self.assertIsNone(vt._user_text(blocks))

    def test_empty_is_dropped(self) -> None:
        self.assertIsNone(vt._user_text("   "))
        self.assertIsNone(vt._user_text(None))


    def test_a_system_reminder_is_cut_out_and_the_prompt_kept(self) -> None:
        """The harness appends reminders to the record holding the prompt, so
        dropping the whole record deleted what was typed beside it."""
        got = vt._user_text("Please rewrite the intro; it reads like a "
                            "brochure.\n<system-reminder>Contents of a "
                            "config file</system-reminder>")
        self.assertIsNotNone(got)
        self.assertIn("rewrite the intro", got)
        self.assertNotIn("config file", got)

    def test_an_unclosed_system_reminder_still_drops_the_record(self) -> None:
        self.assertIsNone(vt._user_text("typed words <system-reminder>and "
                                        "the rest of it"))


class AssistantTextTest(unittest.TestCase):
    """Only prose the model composed may enter its corpus."""

    def test_text_blocks_are_joined(self) -> None:
        blocks = [{"type": "text", "text": "one"}, {"type": "text", "text": "two"}]
        self.assertEqual(vt._assistant_text(blocks), "one\ntwo")

    def test_thinking_is_excluded(self) -> None:
        blocks = [{"type": "thinking", "thinking": "hidden reasoning"},
                  {"type": "text", "text": "visible"}]
        self.assertEqual(vt._assistant_text(blocks), "visible")

    def test_thinking_only_record_yields_nothing(self) -> None:
        self.assertIsNone(vt._assistant_text([{"type": "thinking",
                                               "thinking": "x"}]))

    def test_rate_limit_notice_is_not_voice(self) -> None:
        """Regression: 'you've hit your' ranked as a top-30 tic at 113x."""
        blocks = [{"type": "text",
                   "text": "You've hit your usage limit. Resets at 3pm."}]
        self.assertIsNone(vt._assistant_text(blocks))


    def test_the_model_writing_about_rate_limits_is_voice(self) -> None:
        """A notice is the whole record, so markers match only at its start;
        as substrings they deleted the model's own explanations."""
        for prose in ("The client waits out 429 rate limits before retrying.",
                      "That table row quotes an API Error: 529 verbatim.",
                      "Earlier you've hit your quota twice this week."):
            with self.subTest(prose=prose):
                self.assertEqual(vt._assistant_text(
                    [{"type": "text", "text": prose}]), prose)

    def test_notices_that_open_the_record_are_still_dropped(self) -> None:
        for notice in ("API Error: 529 overloaded", "No response requested.",
                       "  [Request interrupted by user]"):
            with self.subTest(notice=notice):
                self.assertIsNone(vt._assistant_text(
                    [{"type": "text", "text": notice}]))


class ReadTurnsTest(unittest.TestCase):
    """Session-level filtering, which is where both contaminations were fixed."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = pathlib.Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_reads_both_roles(self) -> None:
        path = _session([_rec("user", "do the thing"),
                         _rec("assistant", "Let me check.")], self.tmp)
        turns = list(vt.read_turns([path]))
        self.assertEqual([t.role for t in turns], ["user", "assistant"])

    def test_sidechain_is_skipped(self) -> None:
        path = _session([_rec("assistant", "subagent prose", sidechain=True),
                         _rec("assistant", "main prose")], self.tmp)
        turns = list(vt.read_turns([path]))
        self.assertEqual(len(turns), 1)
        self.assertIn("main prose", turns[0].text)

    def test_scheduled_run_drops_the_whole_session(self) -> None:
        """Regression: scheduled prompts ranked as the human's top tics at 364x.

        The model's replies in that session must go too. They are addressed to
        a log rather than to a reader, so keeping them would clean his corpus
        by corrupting the model's.
        """
        path = _session([
            _rec("user", "This is an automated run; the user is not present."),
            _rec("assistant", "Now the brief is written."),
        ], self.tmp)
        stats: Dict[str, int] = {}
        turns = list(vt.read_turns([path], stats=stats))
        self.assertEqual(turns, [])
        self.assertEqual(stats["automated_sessions"], 1)
        self.assertEqual(stats["sessions"], 1)

    def test_interactive_session_is_kept_and_counted(self) -> None:
        path = _session([_rec("user", "go"), _rec("assistant", "Done.")],
                        self.tmp)
        stats: Dict[str, int] = {}
        list(vt.read_turns([path], stats=stats))
        self.assertEqual(stats.get("automated_sessions", 0), 0)
        self.assertEqual(stats["turns"], 2)

    def test_unreadable_and_malformed_lines_do_not_raise(self) -> None:
        path = _session(["{not json", _rec("user", "ok")], self.tmp)
        self.assertEqual(len(list(vt.read_turns([path, "/nope/missing.jsonl"]))),
                         1)

    def test_non_object_json_line_does_not_kill_the_session(self) -> None:
        """Regression: '"a string"' is valid JSON, and .get() on it raised
        AttributeError, silently losing every session after the bad line."""
        path = _session(['"just a string"', "[1, 2, 3]", "17",
                         _rec("user", "still here")], self.tmp)
        turns = list(vt.read_turns([path]))
        self.assertEqual(len(turns), 1)
        self.assertIn("still here", turns[0].text)

    def test_meta_user_record_is_excluded_and_counted(self) -> None:
        """Regression: isMeta records were 55% of the human's baseline by word
        count, compressing every ratio toward 1."""
        path = _session([
            _rec("user", "expanded slash command boilerplate", meta=True),
            _rec("user", "the words he actually typed"),
        ], self.tmp)
        stats: Dict[str, int] = {}
        turns = list(vt.read_turns([path], stats=stats))
        self.assertEqual(len(turns), 1)
        self.assertIn("actually typed", turns[0].text)
        self.assertEqual(stats["meta_records"], 1)

    def test_synthetic_assistant_record_is_always_dropped(self) -> None:
        path = _session([
            _rec("assistant", "harness stand-in text", model="<synthetic>"),
            _rec("assistant", "real prose", model="claude-fable-5"),
        ], self.tmp)
        stats: Dict[str, int] = {}
        turns = list(vt.read_turns([path], stats=stats))
        self.assertEqual([t.text for t in turns], ["real prose"])
        self.assertEqual(stats["synthetic_records"], 1)

    def test_model_filter_keeps_matching_assistant_and_every_user(self) -> None:
        """--model filters the model's side only: the baseline is the human's,
        whichever model he was talking to."""
        path = _session([
            _rec("user", "go"),
            _rec("assistant", "opus prose", model="claude-opus-5"),
            _rec("assistant", "fable prose", model="claude-fable-5"),
        ], self.tmp)
        stats: Dict[str, int] = {}
        turns = list(vt.read_turns([path], stats=stats, model="fable"))
        self.assertEqual([t.text for t in turns], ["go", "fable prose"])
        self.assertEqual(stats["model_filtered"], 1)

    def test_list_shaped_user_record_is_read(self) -> None:
        """Most real user records carry their prose as a block list, not a
        bare string; the fixture's bare-string default masked that shape."""
        path = _session([_rec("user", "", blocks=[
            {"type": "text", "text": "typed as a block list"}])], self.tmp)
        turns = list(vt.read_turns([path]))
        self.assertEqual(len(turns), 1)
        self.assertIn("typed as a block list", turns[0].text)

    def test_since_cutoff_keeps_only_newer_turns(self) -> None:
        """Regression gap: inverting the cutoff comparison — keeping only
        turns OLDER than the window — previously survived the whole suite."""
        path = _session([
            _rec("user", "the old words", ts="2026-08-01T10:00:00.000Z"),
            _rec("user", "the new words", ts="2026-08-12T10:00:00.000Z"),
        ], self.tmp)
        since = dt.datetime(2026, 8, 5, tzinfo=dt.timezone.utc)
        turns = list(vt.read_turns([path], since=since))
        self.assertEqual([t.text for t in turns], ["the new words"])

    def test_parse_when_reads_iso_z_and_rejects_garbage(self) -> None:
        self.assertEqual(
            vt._parse_when({"timestamp": "2026-08-12T10:00:00.000Z"}),
            dt.datetime(2026, 8, 12, 10, 0, tzinfo=dt.timezone.utc))
        self.assertIsNone(vt._parse_when({"timestamp": "yesterday"}))
        self.assertIsNone(vt._parse_when({"timestamp": 1723456789}))
        self.assertIsNone(vt._parse_when({}))

    def test_transcripts_decode_as_utf8_whatever_the_locale(self) -> None:
        """Transcripts are UTF-8 by contract, not by locale. Without an
        explicit encoding, a non-UTF-8 locale decodes the curly apostrophe
        to replacement garbage, SMART_PUNCT never folds it, and the
        contraction detectors silently go dark — the same regression class
        the smart-punctuation fold itself fixed. The subprocess is the only
        honest probe: macOS auto-enables UTF-8 mode under a C locale, so
        PYTHONUTF8=0 must be set explicitly to reproduce the bad locale."""
        rec = {
            "type": "assistant",
            "isSidechain": False,
            "timestamp": "2026-08-12T10:00:00.000Z",
            "message": {"role": "assistant", "model": "claude-fable-5",
                        "content": [{"type": "text",
                                     "text": "It’s not a bug, it’s a flaw."}]},
        }
        path = self.tmp / "curly.jsonl"
        # ensure_ascii=False so the raw curly-quote BYTES land in the file;
        # json.dumps' default \\u2019 escape is plain ASCII and decodes
        # identically under every codec, which would prove nothing.
        path.write_text(json.dumps(rec, ensure_ascii=False) + "\n",
                        encoding="utf-8")
        code = (
            "import sys; sys.path.insert(0, sys.argv[1]); "
            "import voice_tics as vt; "
            "turns = list(vt.read_turns([sys.argv[2]])); "
            "print(vt.structure_counts(turns[0].text)['not_x_but_y'])"
        )
        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env = dict(os.environ, PYTHONUTF8="0", LC_ALL="C", LANG="C")
        got = subprocess.run([sys.executable, "-c", code, repo, str(path)],
                             env=env, capture_output=True, text=True)
        self.assertEqual(got.stdout.strip(), "1", got.stderr)


    def _stats(self, lines: List[str], **kwargs: Any) -> Dict[str, int]:
        stats: Dict[str, int] = {}
        path = _session(lines, self.tmp)
        self.turns = list(vt.read_turns([path], stats=stats, **kwargs))
        return stats

    def test_a_marker_behind_a_system_reminder_still_drops_the_session(self) -> None:
        stats = self._stats([
            _rec("user", "<system-reminder>hook output</system-reminder>\n"
                         "This is a scheduled task. Do the thing."),
            _rec("assistant", "Alpha beta gamma delta."),
        ])
        self.assertEqual(self.turns, [])
        self.assertEqual(stats.get("automated_sessions"), 1)

    def test_a_days_cutoff_after_the_marker_still_drops_the_session(self) -> None:
        """--days used to skip the marker record as too old and then keep the
        rest of the scheduled run, reporting nothing dropped."""
        stats = self._stats([
            _rec("user", "This is a scheduled task. Do the thing.",
                 ts="2026-01-01T10:00:00Z"),
            _rec("assistant", "Alpha beta gamma delta.",
                 ts="2026-01-03T10:00:00Z"),
        ], since=dt.datetime(2026, 1, 2, tzinfo=dt.timezone.utc))
        self.assertEqual(self.turns, [])
        self.assertEqual(stats.get("automated_sessions"), 1)

    def test_every_meta_record_dropped_is_counted(self) -> None:
        stats = self._stats([
            _rec("user", "<command-name>/x</command-name> expansion", meta=True),
            _rec("user", "a plain expansion", meta=True),
        ])
        self.assertEqual(stats.get("meta_records"), 2)

    def test_a_compaction_summary_is_not_the_authors_prose(self) -> None:
        """The harness stores the model's compaction summary as a USER
        record. Counted as the author, summaries held half of one 45-day
        baseline's words. The fixture sets isCompactSummary alone, so a
        filter keyed on any other flag fails here."""
        summary = json.loads(_rec(
            "user", "This session is being continued from a previous "
                    "conversation. The work so far — summarised."))
        summary["isCompactSummary"] = True
        stats = self._stats([
            _rec("user", "please carry on"),
            json.dumps(summary),
            _rec("assistant", "Carrying on."),
        ])
        self.assertEqual([t.role for t in self.turns], ["user", "assistant"])
        self.assertNotIn("summarised", " ".join(t.text for t in self.turns))
        self.assertEqual(stats.get("compact_summaries"), 1)

    def test_the_report_says_how_many_summaries_were_dropped(self) -> None:
        mine, theirs = vt.Corpus("m"), vt.Corpus("t")
        structures = {n: (0, 0) for n, _, _ in vt.STRUCTURE_PATTERNS}
        report = vt.render(mine, theirs, [], structures, 0, False,
                           stats={"sessions": 1, "compact_summaries": 3})
        self.assertIn("3 compaction summaries", report)

    def test_a_summary_still_marks_a_scheduled_session(self) -> None:
        summary = json.loads(_rec("user", "This is a scheduled task. Summary."))
        summary["isCompactSummary"] = True
        stats = self._stats([json.dumps(summary),
                             _rec("assistant", "Alpha beta.")])
        self.assertEqual(self.turns, [])
        self.assertEqual(stats.get("automated_sessions"), 1)

    def test_a_dropped_notice_is_counted(self) -> None:
        stats = self._stats([_rec("assistant", "API Error: 500", model="m")])
        self.assertEqual(stats.get("noise_records"), 1)

    def test_a_harness_notice_in_a_user_record_is_counted(self) -> None:
        stats = self._stats([_rec("user", "<task-notification>done"
                                          "</task-notification>"),
                             _rec("user", "the words he typed")])
        self.assertEqual([t.text for t in self.turns], ["the words he typed"])
        self.assertEqual(stats.get("noise_records"), 1)

    CROSS = ('<cross-session-message from="a1b2" name="Other session">'
             "Here is the finding — reply when done.</cross-session-message>")

    def test_a_cross_session_message_is_not_the_authors_prose(self) -> None:
        """Another session's model wrote it; the harness delivers it as a
        user record with no flag."""
        stats = self._stats([_rec("user", self.CROSS),
                             _rec("user", "please carry on")])
        self.assertEqual([t.text for t in self.turns], ["please carry on"])
        self.assertEqual(stats.get("cross_session_messages"), 1)

    def test_text_typed_beside_a_cross_session_message_is_kept(self) -> None:
        stats = self._stats([_rec("user", self.CROSS + "\nthanks, go ahead")])
        self.assertEqual(len(self.turns), 1)
        self.assertIn("go ahead", self.turns[0].text)
        self.assertNotIn("finding", self.turns[0].text)
        self.assertEqual(stats.get("cross_session_messages"), 1)

    def test_a_typed_mention_of_the_tag_is_the_authors_prose(self) -> None:
        """Only the harness writes the from attribute; an author discussing
        the tag is still the author."""
        stats = self._stats([_rec("user", "why does <cross-session-message> "
                                          "reach my baseline at all")])
        self.assertEqual(len(self.turns), 1)
        self.assertNotIn("cross_session_messages", stats)

    def test_a_quoted_close_tag_does_not_leak_the_rest_of_a_message(self) -> None:
        """A message quoting the close tag ends the non-greedy match early;
        what follows is still the other session's text."""
        quoting = ('<cross-session-message from="a1b2">The tag ends with '
                   "</cross-session-message> and then more of the message."
                   "</cross-session-message>")
        stats = self._stats([_rec("user", quoting)])
        self.assertEqual(self.turns, [])
        self.assertEqual(stats.get("cross_session_messages"), 1)

    def test_text_typed_between_two_messages_is_kept(self) -> None:
        stats = self._stats([_rec("user", self.CROSS + "\nyes, merge it\n"
                                          + self.CROSS)])
        self.assertEqual([t.text.strip() for t in self.turns], ["yes, merge it"])
        self.assertEqual(stats.get("cross_session_messages"), 1)

    def test_a_notice_beside_a_cross_session_message_drops_the_record(self) -> None:
        stats = self._stats([_rec("user", self.CROSS
                                  + "\n[Request interrupted by user]")])
        self.assertEqual(self.turns, [])
        self.assertEqual(stats.get("cross_session_messages"), 1)
        self.assertNotIn("noise_records", stats)

    def test_a_marker_quoted_by_another_session_keeps_yours(self) -> None:
        """The marker test reads your text after the message is cut out."""
        quoting = ('<cross-session-message from="a1b2">This is a scheduled '
                   "task.</cross-session-message>\nplease look at this")
        stats = self._stats([_rec("user", quoting),
                             _rec("assistant", "Looking.")])
        self.assertEqual(stats.get("automated_sessions", 0), 0)
        self.assertEqual([t.role for t in self.turns], ["user", "assistant"])

    def test_a_record_with_no_timestamp_is_inside_the_window(self) -> None:
        undated = json.loads(_rec("user", "x", meta=True))
        del undated["timestamp"]
        kept = json.loads(_rec("user", "undated words"))
        del kept["timestamp"]
        stats = self._stats([json.dumps(undated), json.dumps(kept)],
                            since=dt.datetime(2026, 8, 1, tzinfo=dt.timezone.utc))
        self.assertEqual([t.text for t in self.turns], ["undated words"])
        self.assertEqual(stats.get("meta_records"), 1)

    def test_an_unclosed_cross_session_message_drops_the_record(self) -> None:
        stats = self._stats([_rec("user", self.CROSS.split("</")[0])])
        self.assertEqual(self.turns, [])
        self.assertEqual(stats.get("cross_session_messages"), 1)

    def test_a_flagged_record_is_counted_once_as_flagged(self) -> None:
        stats = self._stats([_rec("user", self.CROSS, meta=True)])
        self.assertEqual(stats.get("meta_records"), 1)
        self.assertNotIn("cross_session_messages", stats)

    def test_exclusions_before_the_window_are_not_counted(self) -> None:
        """A record the date cutoff drops anyway was never in the window, so
        counting it reported exclusions the window did not contain."""
        old = "2026-01-01T10:00:00.000Z"
        summary = json.loads(_rec("user", "Summary.", ts=old))
        summary["isCompactSummary"] = True
        stats = self._stats(
            [json.dumps(summary), _rec("user", "x", meta=True, ts=old),
             _rec("user", self.CROSS, ts=old),
             _rec("assistant", "Alpha.", model="<synthetic>", ts=old),
             _rec("assistant", "Beta.", sidechain=True, ts=old),
             _rec("assistant", "Gamma.", model="other", ts=old),
             _rec("assistant", "API Error: 500", model="m", ts=old),
             _rec("user", "<task-notification>x</task-notification>", ts=old),
             _rec("user", "old words", uuid="u0", ts=old),
             _replayed(_rec("user", "old words", uuid="u0", ts=old)),
             _rec("user", "in the window")],
            since=dt.datetime(2026, 8, 1, tzinfo=dt.timezone.utc), model="m")
        self.assertEqual([t.text for t in self.turns], ["in the window"])
        self.assertEqual(stats.get("turns"), 1)
        for key in ("compact_summaries", "meta_records", "synthetic_records",
                    "cross_session_messages", "sidechain_records",
                    "model_filtered", "noise_records", "replayed_records"):
            self.assertNotIn(key, stats)

    def test_a_subagent_record_is_counted(self) -> None:
        stats = self._stats([_rec("assistant", "Alpha.", sidechain=True),
                             _rec("user", "a prompt", sidechain=True)])
        self.assertEqual(self.turns, [])
        self.assertEqual(stats.get("sidechain_records"), 2)

    def test_a_dropped_scheduled_run_is_not_counted_twice(self) -> None:
        """Its records are reported as one automated session, not also as
        exclusions, copies included."""
        earlier = _rec("user", "earlier words", uuid="u5")
        stats = self._stats([_rec("user", "x", meta=True), earlier,
                             _replayed(earlier),
                             _rec("user", "This is a scheduled task. Go.")])
        self.assertEqual(stats.get("automated_sessions"), 1)
        self.assertNotIn("meta_records", stats)
        self.assertNotIn("replayed_records", stats)

    def test_the_report_names_every_exclusion_counter(self) -> None:
        mine, theirs = vt.Corpus("m"), vt.Corpus("t")
        structures = {n: (0, 0) for n, _, _ in vt.STRUCTURE_PATTERNS}
        counts = {"meta_records": 11, "compact_summaries": 12,
                  "cross_session_messages": 13, "synthetic_records": 14,
                  "noise_records": 15, "sidechain_records": 16,
                  "model_filtered": 17, "replayed_records": 18}
        report = vt.render(mine, theirs, [], structures, 0, False,
                           stats={"sessions": 1, **counts})
        line = next(l for l in report.splitlines()
                    if l.startswith("Excluded records:"))
        for phrase in ("11 harness-composed (isMeta)", "12 compaction summaries",
                       "13 with cross-session messages cut out",
                       "14 synthetic", "15 harness notices", "16 subagent",
                       "17 other-model", "18 copies of an earlier record"):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, line)

    def test_a_timestamp_without_an_offset_is_utc_not_a_crash(self) -> None:
        stats = self._stats(
            [_rec("assistant", "Alpha beta.", ts="2026-01-03T10:00:00")],
            since=dt.datetime(2026, 1, 2, tzinfo=dt.timezone.utc))
        self.assertEqual(stats.get("turns"), 1)
        self.assertEqual(self.turns[0].when.tzinfo, dt.timezone.utc)

    def test_a_non_object_message_costs_one_record_not_the_scan(self) -> None:
        bad = _session([json.dumps({"type": "assistant", "message": "oops"}),
                        json.dumps({"type": "user", "message": ["x"]})],
                       self.tmp, "bad.jsonl")
        good = _session([_rec("assistant", "Alpha beta.")], self.tmp, "good.jsonl")
        self.assertEqual(len(list(vt.read_turns([bad, good]))), 1)

    def test_an_undecodable_byte_costs_one_line_not_the_scan(self) -> None:
        path = self.tmp / "bytes.jsonl"
        path.write_bytes(b'{"type": "user", "message": {"content": "caf\xe9"}}\n'
                         + _rec("assistant", "Alpha beta.").encode() + b"\n")
        self.assertEqual([t.role for t in vt.read_turns([str(path)])],
                         ["user", "assistant"])

    def test_only_the_first_record_of_a_response_opens_it(self) -> None:
        """A response that calls a tool is several assistant records; the
        continuation after the tool result opens nothing."""
        tool_use = [{"type": "text", "text": "Let me read the parser."},
                    {"type": "tool_use", "id": "t1", "name": "Read", "input": {}}]
        result = [{"type": "tool_result", "tool_use_id": "t1", "content": "x"}]
        self._stats([
            _rec("user", "fix the parser"),
            _rec("assistant", "", blocks=tool_use),
            _rec("user", "", blocks=result),
            _rec("assistant", "The bug is on line twelve."),
            _rec("user", "thanks"),
            _rec("assistant", "Glad it helped."),
        ])
        self.assertEqual([(t.role, t.opens) for t in self.turns],
                         [("user", True), ("assistant", True),
                          ("assistant", False), ("user", True),
                          ("assistant", True)])

    def test_a_replayed_copy_is_counted_once(self) -> None:
        """A transcript can hold a run of its earlier records written again,
        directly after a resume's session metadata, under the same uuid,
        timestamp and message. Read as new records, every turn in that run
        was held twice."""
        prompt = _rec("user", "fix the parser", uuid="u1")
        reply = _rec("assistant", "The bug is on line twelve.", uuid="a1")
        resume = json.dumps({"type": "custom-title", "customTitle": "Parser"})
        stats = self._stats([
            prompt, reply, resume, _replayed(prompt), _replayed(reply),
            _rec("user", "now the tests", uuid="u2"),
            _rec("assistant", "Adding them.", uuid="a2"),
        ])
        self.assertEqual([t.text for t in self.turns],
                         ["fix the parser", "The bug is on line twelve.",
                          "now the tests", "Adding them."])
        self.assertEqual(stats.get("turns"), 4)
        self.assertEqual(stats.get("replayed_records"), 2)

    def test_a_record_without_a_uuid_is_never_collapsed(self) -> None:
        """Without a uuid a copy cannot be told from the same words sent
        again, so both are kept."""
        for uuid in (None, ""):
            with self.subTest(uuid=uuid):
                stats = self._stats([_rec("user", "go on", uuid=uuid),
                                     _rec("user", "go on", uuid=uuid)])
                self.assertEqual([t.text for t in self.turns],
                                 ["go on", "go on"])
                self.assertNotIn("replayed_records", stats)

    def test_a_reused_uuid_with_a_different_message_is_kept(self) -> None:
        """Dropping it would hide text no earlier record carried."""
        stats = self._stats([_rec("assistant", "First draft.", uuid="a1"),
                             _rec("assistant", "Second draft.", uuid="a1")])
        self.assertEqual([t.text for t in self.turns],
                         ["First draft.", "Second draft."])
        self.assertNotIn("replayed_records", stats)

    def test_the_same_message_under_a_new_uuid_is_kept(self) -> None:
        """The same words sent again get a new uuid; only a copy keeps its
        original's, so both halves of the key are needed."""
        stats = self._stats([_rec("user", "go on", uuid="u1"),
                             _rec("user", "go on", uuid="u2")])
        self.assertEqual([t.text for t in self.turns], ["go on", "go on"])
        self.assertNotIn("replayed_records", stats)

    def test_a_copy_with_no_text_is_dropped_but_not_counted(self) -> None:
        """A copy of a tool call, tool result or thinking block was never a
        turn, so counting it overstated what the dedupe removed."""
        prompt = _rec("user", "run the tests", uuid="u1")
        call = _rec("assistant", "", uuid="a1", blocks=[
            {"type": "tool_use", "id": "t1", "name": "Bash", "input": {}}])
        result = _rec("user", "", uuid="u2", blocks=[
            {"type": "tool_result", "tool_use_id": "t1", "content": "ok"}])
        thinking = _rec("assistant", "", uuid="a2", blocks=[
            {"type": "thinking", "thinking": "weighing it"}])
        reply = _rec("assistant", "All green.", uuid="a3")
        records = [prompt, call, result, thinking, reply]
        stats = self._stats(records + [_replayed(r) for r in records])
        self.assertEqual([t.text for t in self.turns],
                         ["run the tests", "All green."])
        self.assertEqual(stats.get("replayed_records"), 2)

    def test_a_copied_prompt_does_not_mark_the_next_reply_an_opener(self) -> None:
        """The opener table counts responses to something you sent; a copy
        of a prompt was not sent again."""
        prompt = _rec("user", "fix the parser", uuid="u1")
        self._stats([prompt, _rec("assistant", "Fixed.", uuid="a1"),
                     _replayed(prompt),
                     _rec("assistant", "Also added a test.", uuid="a2")])
        self.assertEqual([(t.role, t.opens) for t in self.turns],
                         [("user", True), ("assistant", True),
                          ("assistant", False)])

    def test_the_same_record_in_two_transcripts_is_kept_in_each(self) -> None:
        line = _rec("user", "the same words", uuid="u1")
        paths = [_session([line], self.tmp, "a.jsonl"),
                 _session([_replayed(line)], self.tmp, "b.jsonl")]
        stats: Dict[str, int] = {}
        self.assertEqual(len(list(vt.read_turns(paths, stats=stats))), 2)
        self.assertNotIn("replayed_records", stats)

    def test_a_replayed_exclusion_is_counted_once(self) -> None:
        """A copy is reported as a copy, not as its original's exclusion
        again, so each record lands in one counter."""
        meta = _rec("user", "expanded slash command", meta=True, uuid="u1")
        side = _rec("assistant", "subagent prose", sidechain=True, uuid="a1")
        stats = self._stats([meta, side, _replayed(meta), _replayed(side)])
        self.assertEqual(stats.get("meta_records"), 1)
        self.assertEqual(stats.get("sidechain_records"), 1)
        self.assertEqual(stats.get("replayed_records"), 2)

    def test_a_record_the_scan_does_not_read_marks_no_copy(self) -> None:
        """The key is recorded only once a record is admitted as a user or
        assistant record, so another record type carrying the same uuid and
        message hides nothing."""
        typed = json.loads(_rec("user", "the words typed", uuid="u1"))
        other = dict(typed, type="attachment")
        self._stats([json.dumps(other), json.dumps(typed)])
        self.assertEqual([t.text for t in self.turns], ["the words typed"])

    def test_a_lone_surrogate_in_a_message_costs_nothing(self) -> None:
        """JSON can escape half a surrogate pair, which strict UTF-8 cannot
        encode; hashing that message must not end the scan."""
        lone = json.loads(_rec("user", "placeholder", uuid="u1"))
        lone["message"]["content"] = "caf\ud800 words"
        self._stats([json.dumps(lone), _rec("assistant", "Alpha beta.",
                                            uuid="a1")])
        self.assertEqual([t.role for t in self.turns], ["user", "assistant"])


class NgramTest(unittest.TestCase):
    """Phrase ranking mechanics."""

    def test_ngrams_windows(self) -> None:
        self.assertEqual(list(vt.ngrams(["a", "b", "c"], 2)),
                         [("a", "b"), ("b", "c")])

    def test_ngrams_shorter_than_width_yields_nothing(self) -> None:
        self.assertEqual(list(vt.ngrams(["a"], 3)), [])

    def test_subsumed_phrases_are_dropped(self) -> None:
        """'let me check' and 'me check' are one habit, not two findings."""
        long_f = vt.Finding("let me check", 3, 100, 10.0, 0, 0.1, 100.0)
        short_f = vt.Finding("me check", 2, 100, 10.0, 0, 0.1, 100.0)
        kept = vt._drop_subsumed([long_f, short_f])
        self.assertEqual([f.phrase for f in kept], ["let me check"])

    def test_distinct_phrase_survives(self) -> None:
        a = vt.Finding("let me check", 3, 100, 10.0, 0, 0.1, 100.0)
        b = vt.Finding("say the word", 3, 90, 9.0, 0, 0.1, 90.0)
        self.assertEqual(len(vt._drop_subsumed([a, b])), 2)

    def test_containment_respects_word_boundaries(self) -> None:
        """Regression: raw substring matching dropped 'own the' as
        "contained" in 'down the', and 'in the' inside 'within the' —
        different habits, silently merged."""
        down = vt.Finding("down the", 2, 30, 10.0, 0, 0.1, 120.0)
        own = vt.Finding("own the", 2, 35, 9.0, 0, 0.1, 110.0)
        self.assertEqual([f.phrase for f in vt._drop_subsumed([down, own])],
                         ["down the", "own the"])
        within = vt.Finding("within the", 2, 30, 10.0, 0, 0.1, 120.0)
        in_the = vt.Finding("in the", 2, 30, 9.0, 0, 0.1, 110.0)
        self.assertEqual(
            [f.phrase for f in vt._drop_subsumed([within, in_the])],
            ["within the", "in the"])

    def test_much_more_frequent_shorter_phrase_survives(self) -> None:
        """The 1.35 count guard: a contained phrase used far more often than
        its container is its own habit, not a window onto it."""
        long_f = vt.Finding("let me check", 3, 100, 10.0, 0, 0.1, 100.0)
        short_hot = vt.Finding("let me", 2, 200, 20.0, 0, 0.1, 90.0)
        self.assertEqual(
            [f.phrase for f in vt._drop_subsumed([long_f, short_hot])],
            ["let me check", "let me"])

    def test_equal_ratio_ladder_collapses_to_the_longest_phrase(self) -> None:
        """Regression: a fixed phrase against a zero baseline produces every
        window of itself at the SAME ratio and count. The stable sort left
        the width-2 windows ranked first, and _drop_subsumed never drops a
        longer phrase into a shorter kept one, so one habit filled ten of
        the top-N rows. Ties must rank longest-first so the windows collapse
        into the phrase that contains them."""
        mine, theirs = vt.Corpus("m"), vt.Corpus("n")
        fillers = ("ax bx cx dx ex fx gx hx ix jx "
                   "kx lx mx nx ox px qx rx sx tx").split()
        mine.add(" ".join(f"let me verify the counts {w}" for w in fillers))
        theirs.add("an unrelated baseline entirely.")
        got = vt.rank_phrases(mine, theirs, min_count=5, top=50)
        self.assertEqual([f.phrase for f in got],
                         ["let me verify the counts"])

    def test_min_count_excludes_rare_phrases(self) -> None:
        mine, theirs = vt.Corpus("m"), vt.Corpus("n")
        mine.add("alpha beta. " * 3)
        theirs.add("something else entirely.")
        self.assertEqual(vt.rank_phrases(mine, theirs, min_count=10, top=20), [])
        self.assertTrue(vt.rank_phrases(mine, theirs, min_count=2, top=20))


    def test_a_string_longer_than_the_widest_ngram_reports_once(self) -> None:
        """Its widest windows are siblings, not containers, so containment
        alone kept every one and a second habit fell off the top rows."""
        mine, theirs = vt.Corpus("m"), vt.Corpus("t")
        for _ in range(10):
            mine.add("Alpha bravo charlie delta echo foxtrot golf hotel india juliet.")
            mine.add("Quebec romeo sierra.")
        for i in range(40):
            theirs.add(f"unrelated words number {i}")
        top = [f.phrase for f in vt.rank_phrases(mine, theirs, 5, 5)]
        self.assertEqual(top, ["alpha bravo charlie delta echo foxtrot",
                               "quebec romeo sierra"])

    def test_no_ngram_crosses_a_sentence_or_a_scrubbed_span(self) -> None:
        corpus = vt.Corpus("m")
        corpus.add(vt.scrub("The test passed.\nRead the file next.",
                            gap=vt.SCRUB_GAP))
        corpus.add(vt.scrub("Run `git status` then look.", gap=vt.SCRUB_GAP))
        self.assertNotIn(("passed", "read"), corpus.grams[2])
        self.assertNotIn(("run", "then"), corpus.grams[2])
        self.assertIn(("then", "look"), corpus.grams[2])
        self.assertEqual(corpus.total_words, 10)

    def test_the_gap_is_invisible_to_the_structure_detectors(self) -> None:
        """Counts must not move because a removed span is marked differently."""
        text = "It's not the code `x`, it's the config https://e.io and more."
        self.assertEqual(vt.structure_counts(vt.scrub(text)),
                         vt.structure_counts(vt.scrub(text, gap=vt.SCRUB_GAP)))


class CorpusTest(unittest.TestCase):
    """Rates and openers."""

    def test_rate_is_per_ten_thousand_words(self) -> None:
        c = vt.Corpus("x")
        c.add(" ".join(["word"] * 1000))
        self.assertAlmostEqual(c.rate(1), 10.0, places=6)

    def test_empty_corpus_rates_zero_not_error(self) -> None:
        c = vt.Corpus("x")
        self.assertEqual(c.rate(5), 0.0)
        self.assertEqual(c.mean_sentence_len(), 0.0)

    def test_opener_is_first_three_words_of_first_sentence(self) -> None:
        c = vt.Corpus("x")
        c.add("Let me check the file. Then something else.")
        self.assertEqual(c.openers.most_common(1)[0][0], "let me check")


    def test_a_continuation_counts_words_but_opens_nothing(self) -> None:
        corpus = vt.Corpus("m")
        corpus.add("The bug is on line twelve.", opens=False)
        self.assertEqual(sum(corpus.openers.values()), 0)
        self.assertEqual(corpus.total_words, 6)


class FenceScanTest(unittest.TestCase):
    """emdash.fence_spans: one fence definition, linear, and CommonMark-shaped.

    The regex it replaced is kept here as a spec for the subset where the two
    must still agree: three-backtick fences indented up to three spaces or a
    tab. Beyond that subset the scan deliberately knows more, and the cases
    below pin each difference.
    """

    LEGACY_SPEC = re.compile(r"^[ \t]{0,3}```.*?\n[ \t]{0,3}```[ \t]*$",
                             re.DOTALL | re.MULTILINE)

    TRICKY = [
        "",
        "no fences here",
        "```\ncode\n```",
        "```py\ncode\n```",
        "   ```\ncode\n   ```",
        "```\ncode\n```   ",
        # A closer-shaped line that is really the next opener.
        "```a\n```b\n```",
        # Unclosed: the regex matches nothing, so neither may the scan.
        "```a\nstill open",
        "```a\n```b\nstill open",
        # Two complete fences back to back.
        "```\none\n```\n```\ntwo\n```",
        # A fence line mid-line is literal text, not a fence.
        "text ``` more\n``` \n",
        "\t```\ncode\n\t```",
        "```\n\n```",
        "```\n```\n```\n```",
    ]

    def test_agrees_with_the_legacy_spec_on_its_subset(self) -> None:
        for text in self.TRICKY:
            with self.subTest(text=text):
                want = [m.span() for m in self.LEGACY_SPEC.finditer(text)]
                self.assertEqual(vt.fence_spans(text), want, text)

    def test_agrees_with_the_legacy_spec_on_random_subset_input(self) -> None:
        """Randomised, because the tricky list is only what someone thought of."""
        import random
        lines = ["```", "```py", "   ```", "```x", "text", "", "  ",
                 "a ``` b", "\t```", "```   "]
        rng = random.Random(7)
        for _ in range(600):
            text = "\n".join(rng.choice(lines) for _ in range(rng.randint(0, 12)))
            if rng.random() < 0.5:
                text += "\n"
            want = [m.span() for m in self.LEGACY_SPEC.finditer(text)]
            self.assertEqual(vt.fence_spans(text), want, repr(text))

    def _code_is_gone(self, text: str) -> None:
        self.assertNotIn("secret", vt.scrub(text))
        self.assertNotIn("secret", emdash.strip_code(text))

    def test_a_tilde_fence_is_code(self) -> None:
        self._code_is_gone("~~~\nsecret — code\n~~~\nprose")

    def test_a_backtick_in_the_info_string_is_not_a_fence(self) -> None:
        """"```x``` is the inline form" is a paragraph in CommonMark; as an
        opener it swallowed the prose after it and desynchronised the next
        real fence."""
        text = "```x``` is inline — here.\nReal prose.\n\n```\nsecret\n```\n"
        self.assertEqual(vt.fence_spans(text), [(text.index("```\nsecret"),
                                                 len(text) - 1)])

    def test_a_longer_fence_shows_a_shorter_one_inside_it(self) -> None:
        self._code_is_gone("````\n```\nsecret\n```\n````\nprose")
        self.assertIn("prose", vt.scrub("````\n```\nsecret\n```\n````\nprose"))

    def test_a_crlf_document_is_fenced_like_an_lf_one(self) -> None:
        """Piped on stdin, CRLF is not translated, so the verdict depended on
        how the document reached the tool."""
        self._code_is_gone("Intro.\r\n```\r\nsecret\r\n```\r\nTail.\r\n")

    def test_a_fence_indented_inside_a_list_item_is_code(self) -> None:
        self._code_is_gone("- Install it:\n\n    ```bash\n    secret\n    ```\n")

    def test_scrub_and_strip_code_agree_about_a_no_break_space(self) -> None:
        """scrub folds U+00A0 to a space before scanning; strip_code does not.
        The scan must give both the same answer."""
        text = "Intro.\n\n\u00a0```\n| a secret b | c |\n```\n\nEnd.\n"
        folded = text.translate(vt.SMART_PUNCT)
        self.assertEqual(vt.fence_spans(folded), emdash.fence_spans(text))
        self._code_is_gone(text)

    def test_scrub_still_removes_fenced_code(self) -> None:
        self.assertNotIn("secret", vt.scrub("before\n```\nsecret\n```\nafter"))
        self.assertIn("before", vt.scrub("before\n```\nsecret\n```\nafter"))

    def test_an_unclosed_fence_is_not_stripped(self) -> None:
        """An unterminated fence is literal text, so a stray opener cannot
        delete the rest of a document from the lint."""
        text = "```\nthis is never closed"
        self.assertEqual(vt.fence_spans(text), [])
        self.assertIn("never closed", vt.scrub(text))

    def test_keep_lines_preserves_the_line_count(self) -> None:
        text = "a\n```\nx\ny\n```\nb"
        self.assertEqual(vt.strip_fences(text, " ", True).count("\n"),
                         text.count("\n"))

    def test_unclosed_fences_do_not_blow_up(self) -> None:
        """The reason fence_spans exists, pinned as a budget.

        The regex this replaced was lazy and multiline, so it restarted at
        every opening line and scanned to end of input: 3.16s for this input. The
        scan does it in about a millisecond. The budget is generous by three
        orders of magnitude so this cannot flake on a loaded machine, while
        still failing outright if the quadratic path ever comes back.
        """
        import time
        text = "\n".join("```unclosed %d" % i for i in range(8000))
        start = time.perf_counter()
        spans = vt.fence_spans(text)
        elapsed = time.perf_counter() - start
        self.assertEqual(spans, [])
        self.assertLess(elapsed, 0.5, f"fence scan took {elapsed:.3f}s")


class PatternTest(unittest.TestCase):
    """The named structure and signature detectors."""

    def test_let_me_structure_fires(self) -> None:
        self.assertEqual(vt.structure_counts("Let me check that.")["let_me"], 1)
        self.assertEqual(vt.structure_counts("Now updating it.")["let_me"], 0)

    def test_not_x_but_y_fires(self) -> None:
        got = vt.structure_counts("It's not just a bug, it's a design flaw.")
        self.assertGreaterEqual(got["not_x_but_y"], 1)

    def test_curly_apostrophe_counts_same_as_straight(self) -> None:
        """Regression: not_x_but_y scored 0 on the U+2019 form of a sentence
        it scored 1 on with a straight apostrophe, and macOS emits curly by
        default. scrub() folds smart punctuation before any pattern runs."""
        curly = "It’s not a bug, it’s a flaw."
        got = vt.structure_counts(vt.scrub(curly))
        self.assertEqual(got["not_x_but_y"], 1)

    def test_i_should_note_matches_contraction_and_will(self) -> None:
        """Regression: '\\bi\\s+'?ll' could never match "I'll" (no space
        before the contraction), and "will" was missing entirely."""
        for phrase in ("I'll flag the risk.", "I will note the gap.",
                       "I should mention one thing."):
            self.assertEqual(
                vt.structure_counts(phrase)["i_should_note"], 1, phrase)
        self.assertEqual(
            vt.structure_counts("I will do it now.")["i_should_note"], 0)

    def test_ultimately_fires_signposted_conclusion(self) -> None:
        """Regression: ',\\b' after 'ultimately' never matched — a word
        boundary after a comma needs a word character next, and prose puts a
        space there."""
        got = vt.structure_counts("Ultimately, the fix held.")
        self.assertEqual(got["signposted_conclusion"], 1)
        self.assertEqual(
            vt.structure_counts("In conclusion, done.")["signposted_conclusion"],
            1)

    def test_appositive_negation_ignores_participial_clauses(self) -> None:
        """Regression: 'I asked, not knowing why' is grammar, not the tic."""
        counts = vt.structure_counts("I asked, not knowing why.")
        self.assertEqual(counts["appositive_negation"], 0)
        self.assertGreaterEqual(
            vt.structure_counts("It's a flaw, not a bug.")
            ["appositive_negation"], 1)
        self.assertGreaterEqual(
            vt.structure_counts("We got there by measuring, not guessing.")
            ["appositive_negation"], 1)

    def test_not_x_but_y_matches_negative_contractions(self) -> None:
        """Round-two regression: 'is\\s+not' cannot match inside "isn't", so
        the contracted reveal — likely its majority surface form — scored
        zero until the leading group spelled the contractions out."""
        for s in ("The problem isn't the code, it's the config.",
                  "These aren't bugs, they're features."):
            self.assertEqual(vt.structure_counts(s)["not_x_but_y"], 1, s)

    def test_appositive_dash_branch_ignores_compound_hyphens(self) -> None:
        """Round-two regression: [—-] matched the hyphen INSIDE a compound
        word, so 'not merely absent-minded' counted as the dash form."""
        got = vt.structure_counts(
            "The professor was not merely absent-minded; he forgot it.")
        self.assertEqual(got["appositive_negation"], 0)
        self.assertGreaterEqual(
            vt.structure_counts("It is not just a bug — a design flaw.")
            ["appositive_negation"], 1)

    def test_appositive_negation_complement_heuristic(self) -> None:
        """Round-two regression: a gerund WITH a complement is a participial
        clause whatever the pre-comma word ends in (round one miscounted
        'this morning, not knowing…'), and a -thing pronoun is a noun
        whatever follows it."""
        for clause in ("I woke up this morning, not knowing where I was.",
                       "She kept running, not looking back even once.",
                       "He said something, not realizing the mic was on."):
            self.assertEqual(
                vt.structure_counts(clause)["appositive_negation"], 0, clause)
        for tic in ("This is deliberate, not something to fix.",
                    "It was measured, not guessed."):
            self.assertGreaterEqual(
                vt.structure_counts(tic)["appositive_negation"], 1, tic)

    def test_rule_of_three_counts_multiword_items(self) -> None:
        """Regression: items were capped at one to two words, so a genuine
        three-item list of short clauses did not count."""
        got = vt.structure_counts(
            "Read the file, run the query, and check the result.")
        self.assertEqual(got["rule_of_three"], 1)
        self.assertEqual(
            vt.structure_counts("Just one, and two.")["rule_of_three"], 0)

    def test_rule_of_three_comma_styles_count_in_separate_rows(self) -> None:
        """Round three: both comma styles count, in disjoint rows. The
        non-Oxford surface form measured ~90% clause coordination on the
        live corpus, so widening rule_of_three itself would have poisoned a
        precise count; the noisy form gets its own labelled row instead."""
        ox = vt.structure_counts("It was fast, cheap, and good.")
        no = vt.structure_counts("It was fast, cheap and good.")
        self.assertEqual(ox["rule_of_three"], 1)
        self.assertEqual(ox["rule_of_three_no_oxford"], 0)
        self.assertEqual(no["rule_of_three"], 0)
        self.assertEqual(no["rule_of_three_no_oxford"], 1)

    def test_method_defence_requires_the_verb_not_the_noun(self) -> None:
        """Regression: 'matters?' with the -s optional matched the NOUN in
        'discuss this matter', counting a formal register as the tic,
        inflating his baseline and understating the ratio."""
        for noun in ("We will discuss this matter tomorrow.",
                     "The committee reviewed this matter in June."):
            self.assertEqual(
                vt.structure_counts(noun)["method_defence"], 0, noun)
        for verb in ("The gate is offline, which means that nothing runs.",
                     "This matters because the baseline is his."):
            self.assertEqual(
                vt.structure_counts(verb)["method_defence"], 1, verb)

    def test_signature_is_counted_case_insensitively(self) -> None:
        got = vt.signature_counts("However, it works. however it is slow.",
                                  [("however", r"\bhowever\b")])
        self.assertEqual(got["however"], 2)

    def test_signature_absent_scores_zero(self) -> None:
        got = vt.signature_counts("plain text", [("herein", r"\bherein\b")])
        self.assertEqual(got["herein"], 0)

    def test_signatures_default_to_empty_not_to_someone_elses_habits(self) -> None:
        """An unconfigured run must measure NO signatures.

        The shipped table is deliberately empty: signature phrases are a claim
        about one person's writing, and a default would present a stranger's
        habits as yours. Pinned because a well-meaning "sensible default" here
        is the exact regression that would make the tool wrong for everyone
        who did not write it.
        """
        self.assertEqual(vt.BASELINE_SIGNATURES, ())
        self.assertEqual(vt.signature_counts("However, herein, thus."), {})

    def test_every_pattern_compiles_and_reports(self) -> None:
        counts = vt.structure_counts("some prose")
        self.assertEqual(set(counts), {n for n, _, _ in vt.STRUCTURE_PATTERNS})
        table = [("alpha", r"\balpha\b"), ("beta", r"\bbeta\b")]
        self.assertEqual(set(vt.signature_counts("some prose", table)),
                         {"alpha", "beta"})



class BaselineFloorTest(unittest.TestCase):
    """The continuity floor keeps a never-used phrase finite AND ranked."""

    def _structure_ratio(self, baseline_count: int) -> float:
        mine, theirs = vt.Corpus("m"), vt.Corpus("t")
        mine.total_words = theirs.total_words = 10000
        structures = {n: (0, 0) for n, _, _ in vt.STRUCTURE_PATTERNS}
        structures["not_x_but_y"] = (10, baseline_count)
        report = vt.render(mine, theirs, [], structures, 0, False)
        row = next(line for line in report.splitlines()
                   if line.rstrip().endswith(" not_x_but_y"))
        return float(row.split("x")[0])

    def test_a_structure_the_baseline_never_uses_is_not_zero(self) -> None:
        """Without the floor the strongest possible tic printed a zero ratio."""
        self.assertEqual(self._structure_ratio(0), 20.0)

    def test_never_said_ranks_above_said_once(self) -> None:
        """The reason for half the smallest count rather than one of it."""
        self.assertGreater(self._structure_ratio(0), self._structure_ratio(1))


class SourceTextRedactionTest(unittest.TestCase):
    """Every source-text string that prints goes through ``redact``.

    Today the tokeniser strips the characters redaction keys on, so no real
    corpus can reach these paths with an address in it. The belt exists for
    the day that changes, which is exactly when nobody would notice one of
    the four calls had gone. So the address is planted past the tokeniser.
    """

    ADDRESS = "sam@vendor.io"

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = pathlib.Path(self._tmp.name)
        _session([_rec("user", "a plain request"),
                  _rec("assistant", "A plain answer.")], self.tmp)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _run(self, *extra: str) -> str:
        import contextlib
        import io
        from unittest import mock
        planted = vt.Finding(phrase=self.ADDRESS, width=1, mine=5,
                             mine_rate=1.0, theirs=0, theirs_rate=1.0,
                             ratio=1.0)
        real_add = vt.Corpus.add

        def add(corpus, text, *args, **kwargs):
            real_add(corpus, text, *args, **kwargs)
            corpus.openers[self.ADDRESS] += 1

        buf = io.StringIO()
        with mock.patch.object(vt, "rank_phrases", return_value=[planted]), \
                mock.patch.object(vt.Corpus, "add", add), \
                contextlib.redirect_stdout(buf):
            rc = vt.main(["--glob", str(self.tmp / "*.jsonl"),
                          "--include-source-text", *extra])
        self.assertEqual(rc, 0)
        return buf.getvalue()

    def test_the_text_report_redacts_phrases_and_openers(self) -> None:
        out = self._run()
        self.assertNotIn(self.ADDRESS, out)
        self.assertEqual(out.count("[REDACTED-EMAIL]"), 2)

    def test_the_json_report_redacts_phrases_and_openers(self) -> None:
        out = self._run("--json")
        self.assertNotIn(self.ADDRESS, out)
        doc = json.loads(out)
        self.assertEqual(doc["phrases"][0]["phrase"], "[REDACTED-EMAIL]")
        self.assertIn("[REDACTED-EMAIL]", [o for o, _ in doc["openers"]])


class MainTest(unittest.TestCase):
    """End-to-end, including the flip that --speaker performs."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = pathlib.Path(self._tmp.name)
        _session([
            _rec("user", "check the numbers however you like"),
            _rec("assistant", "Let me check. Let me verify the counts."),
        ], self.tmp)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_json_output_is_valid_and_labels_the_speaker(self) -> None:
        import io
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = vt.main(["--glob", str(self.tmp / "*.jsonl"), "--json",
                          "--min-count", "1"])
        self.assertEqual(rc, 0)
        got = json.loads(buf.getvalue())
        self.assertEqual(got["speaker"], "model")
        self.assertIn("let_me", got["structures"])
        self.assertEqual(got["structures"]["let_me"]["model"], 2)
        # No config file, so no signatures are configured and the table is
        # empty rather than absent: a consumer can tell "none configured"
        # from "key withheld".
        self.assertEqual(got["signatures"], {})

    def test_configured_signatures_reach_the_json_table(self) -> None:
        """The [baseline] signatures table must survive the whole pipeline."""
        import io
        import contextlib
        cfg = self.tmp / "voice_tics.toml"
        cfg.write_text('[baseline]\n'
                       'signatures = [{name = "however", '
                       'pattern = "\\\\bhowever\\\\b"}]\n',
                       encoding="utf-8")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = vt.main(["--glob", str(self.tmp / "*.jsonl"), "--json",
                          "--min-count", "1", "--config", str(cfg)])
        self.assertEqual(rc, 0)
        got = json.loads(buf.getvalue())
        self.assertEqual(got["signatures"]["however"]["baseline"], 1)

    def test_speaker_flip_keeps_structure_attribution(self) -> None:
        """--speaker must swap which corpus is examined, not relabel counts.

        The source-text gate is asserted per speaker too: the baseline
        direction is the one where a leak would quote his own typed prose.
        """
        import io
        import contextlib
        for speaker in ("model", "baseline"):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                vt.main(["--glob", str(self.tmp / "*.jsonl"), "--json",
                         "--speaker", speaker, "--min-count", "1"])
            got = json.loads(buf.getvalue())
            self.assertEqual(got["structures"]["let_me"]["model"], 2, speaker)
            self.assertEqual(got["structures"]["let_me"]["baseline"], 0, speaker)
            self.assertNotIn("phrases", got, speaker)
            self.assertNotIn("openers", got, speaker)

    def test_speaker_flip_swaps_the_examined_corpus(self) -> None:
        """Regression gap: with the flip made a no-op the structure
        assertions above hold vacuously, because the structures table is
        speaker-independent. The phrase table is built from ``mine``, so
        under --speaker baseline it must quote HIS prose, not the model's."""
        import io
        import contextlib
        expected = {"model": ("verify the counts", "the numbers"),
                    "baseline": ("the numbers", "verify the counts")}
        for speaker, (present, absent) in expected.items():
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                vt.main(["--glob", str(self.tmp / "*.jsonl"), "--json",
                         "--speaker", speaker, "--min-count", "1",
                         "--include-source-text"])
            got = json.loads(buf.getvalue())
            phrases = [f["phrase"] for f in got["phrases"]]
            self.assertTrue(any(present in p for p in phrases), speaker)
            self.assertFalse(any(absent in p for p in phrases), speaker)

    def test_json_withholds_source_text_by_default(self) -> None:
        """Fail safe: the keys carrying transcript fragments must be absent,
        not empty, so 'withheld' cannot read as 'measured as zero'."""
        import io
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = vt.main(["--glob", str(self.tmp / "*.jsonl"), "--json",
                          "--min-count", "1"])
        self.assertEqual(rc, 0)
        got = json.loads(buf.getvalue())
        self.assertFalse(got["source_text_included"])
        self.assertNotIn("phrases", got)
        self.assertNotIn("openers", got)
        self.assertIn("structures", got)

    def test_json_includes_source_text_only_on_opt_in(self) -> None:
        import io
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = vt.main(["--glob", str(self.tmp / "*.jsonl"), "--json",
                          "--min-count", "1", "--include-source-text"])
        self.assertEqual(rc, 0)
        got = json.loads(buf.getvalue())
        self.assertTrue(got["source_text_included"])
        self.assertIn("phrases", got)
        self.assertIn("openers", got)
        self.assertTrue(any("let me" in f["phrase"] for f in got["phrases"]))

    def test_text_report_withholds_source_text_by_default(self) -> None:
        import io
        import contextlib
        for flag, expect_tables in (([], False), (["--include-source-text"],
                                                  True)):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                vt.main(["--glob", str(self.tmp / "*.jsonl"),
                         "--min-count", "1"] + flag)
            report = buf.getvalue()
            self.assertEqual("PHRASES" in report, expect_tables, flag)
            # A fragment only the phrase/opener tables can reproduce — the
            # let_me structure DESCRIPTION legitimately says "let me check",
            # so that string cannot serve as the probe.
            self.assertEqual("verify the counts" in report, expect_tables,
                             flag)
            # The counts-only concentration line prints either way.
            self.assertIn("distinct openers", report)

    def test_model_filter_via_cli(self) -> None:
        _session([
            _rec("user", "hi"),
            _rec("assistant", "opus words here", model="claude-opus-5"),
            _rec("assistant", "fable words here", model="claude-fable-5"),
        ], self.tmp, name="mixed.jsonl")
        import io
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = vt.main(["--glob", str(self.tmp / "mixed.jsonl"), "--json",
                          "--min-count", "1", "--model", "fable"])
        self.assertEqual(rc, 0)
        got = json.loads(buf.getvalue())
        self.assertEqual(got["model"]["turns"], 1)
        self.assertEqual(got["model_filter"], "fable")
        self.assertEqual(got["stats"]["model_filtered"], 1)

    def test_missing_transcripts_exit_nonzero(self) -> None:
        import io
        import contextlib
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            rc = vt.main(["--glob", str(self.tmp / "none-*.jsonl")])
        self.assertEqual(rc, 1)


    def test_a_recursive_transcript_glob_reads_every_level(self) -> None:
        import io
        import contextlib
        deep = self.tmp / "nested" / "deeper"
        deep.mkdir(parents=True)
        _session([_rec("assistant", "Deep words here.")], deep, "d.jsonl")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = vt.main(["--glob", str(self.tmp / "**" / "*.jsonl"), "--json"])
        self.assertEqual(rc, 0)
        self.assertEqual(json.loads(buf.getvalue())["stats"]["sessions"], 2)


if __name__ == "__main__":
    unittest.main()
