#!/usr/bin/env python3
"""Voice-tic discovery: which phrases and structures does the model overuse?

This is the *discovery* step. It answers the question before any rule exists:
of everything the model writes, what does it repeat far more than you would?

WHY A MATCHED BASELINE, NOT A WORD LIST
    "Phrases an AI overuses" is a folk category, and a hand-written list of them
    is an assertion with no source. Worse, such a list is actively wrong for
    anyone whose register overlaps it. The author this was built for leans on
    "However", "It should be noted that" and "Thus" in his own writing, so a
    generic AI-slop detector flags his genuine academic register as
    machine-written. Run one against a lawyer, an academic or a technical
    writer and it corrects the wrong author — deleting the voice it was
    installed to protect. His words, after one such tool had been at his
    drafts: "we essentially made me sound simpler so I don't sound like AI."

    Instead the baseline is YOU, in the same transcripts. Every Claude Code
    session .jsonl holds both sides of the conversation: the model's prose and
    yours. Same medium, same topics, same weeks, same technical vocabulary.
    That match is the whole point — it is what lets a ratio mean something.
    A phrase that appears at 40x your rate is a tic; a phrase both sides use
    at similar rates is just the subject matter.

    The register caveat is real and is reported rather than hidden: your turns
    are short and imperative, the model's are long and explanatory, so
    explanatory connectives are over-represented by construction. Ratios are
    evidence for review, never a verdict.

WHAT IT MEASURES
    Three families, deliberately kept separate because they need different
    remedies:

    * **Phrases** — n-grams (2..6) ranked by rate-per-10k-words against the
      matched baseline. Remedy: say it another way.
    * **Openers** — how the first sentence of a response begins, and how
      concentrated that distribution is. Remedy: vary the entry.
    * **Structures** — templated constructions ("not X, but Y", rule-of-three,
      the method-defence move). Remedy: restructure. These are the ones that
      read as machine-written even when every phrase is fresh.

DETERMINISTIC ONLY
    Every number here is a count over text already on disk. There is no grader
    and no model in the loop — a model grading a model is the expensive option
    and the least trustworthy one. The output ranks candidates for YOUR
    judgment; it does not decide what a tic is, and it cannot.

PROSE ONLY
    Code fences, inline code, URLs, file paths and markdown link targets are
    stripped before counting. Without that, shared vocabulary from pasted
    tracebacks and paths dominates every n-gram table.

SOURCE TEXT IS OPT-IN, AND THE GATE IS THE ONLY PROTECTION
    The structure and signature tables carry only counts against named
    patterns. The phrase and opener tables are reconstructed fragments of the
    transcripts themselves, and 64% of sessions name a client or engagement,
    so by default neither prints. ``--include-source-text`` opts in, and the
    gate applies to ``--json`` identically. Opt-in output does pass through
    ``redact``, but that is a belt kept for the day ``WORD_RE`` widens:
    tokenisation has already stripped the ``@``, dots and digits that email
    and phone redaction key on, so today redaction cannot fire on this
    output (verified 2026-08-13: a planted address printed as its adjacent
    alpha tokens). Treat opt-in output as quoting the transcripts.

Usage:
    python3 voice_tics.py --days 30
    python3 voice_tics.py --model claude-fable-5
    python3 voice_tics.py --include-source-text --top 40 --min-count 8
    python3 voice_tics.py --json > tics.json
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import datetime as dt
import glob
import json
import os
import re
import sys
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import vt_config as config_mod  # noqa: E402
from vt_redact import redact  # noqa: E402

PROJECTS_GLOB = config_mod.DEFAULT_TRANSCRIPTS

# n-gram widths scanned. 2 catches "clean signal"; 6 catches whole templated
# clauses like "that is not a bug it is".
NGRAM_WIDTHS = (2, 3, 4, 5, 6)

# A phrase must occur at least this often in the model's prose before its ratio
# is worth reading. Ratios over tiny counts are noise: one use against a zero
# baseline is an infinite ratio and means nothing.
DEFAULT_MIN_COUNT = 5

# Rate assumed for a phrase the baseline never uses, so the ratio stays finite
# and sortable. Expressed as a count, converted to a rate alongside the real
# ones. Chosen as 0.5 -- half the smallest observable count -- which is the
# standard continuity correction and keeps "never said it once" ranked above
# "said it once".
BASELINE_FLOOR_COUNT = 0.5

# ---------------------------------------------------------------- text scrubs
# Both delimiters anchored to line starts (up to 3 spaces indent), per
# CommonMark: a mid-line ``` run is literal text, not a fence. Unanchored,
# a stray mid-line pair swallowed the prose between — and prose_lint's
# row-masking could delete the run that closed such a match, stitching it
# to the next one lines away (2026-08-13, PR #46 verification, round
# five). The closer line must hold nothing but the backticks.
CODE_FENCE_RE = re.compile(r"^[ \t]{0,3}```.*?\n[ \t]{0,3}```[ \t]*$",
                           re.DOTALL | re.MULTILINE)
# The two halves of that pattern, used by strip_fences() to do the same job in
# one left-to-right pass. CODE_FENCE_RE stays as the SPEC the scan is tested
# against -- tests/test_voice_tics.py asserts the two agree -- because a regex
# is the clearer statement of what a fence IS, and it is still correct, just
# quadratic on input nobody sane writes.
FENCE_OPEN_RE = re.compile(r"^[ \t]{0,3}```", re.MULTILINE)
FENCE_CLOSE_RE = re.compile(r"^[ \t]{0,3}```[ \t]*$", re.MULTILINE)
INLINE_CODE_RE = re.compile(r"`[^`\n]+`")
URL_RE = re.compile(r"https?://\S+")
# A path-looking token: two or more slash-separated segments, or a leading ~/ .
PATH_RE = re.compile(r"(?:~|\.{0,2})?/[\w.\-]+(?:/[\w.\-]+)+/?")
# Single-line on purpose, matching what actually renders: a CommonMark link
# destination and an Obsidian wikilink cannot contain a newline, so a
# "[x](..." whose ")" sits lines away is literal prose, not a link. The
# newline-admitting classes these carried were also a splice hazard: they
# let a match stitch across a line prose_lint had blanked, swallowing —
# and silently dropping — the prose between (2026-08-13, PR #46
# verification, round four). With them single-line, no scrub pass except
# the fence pass can cross a line boundary at all.
MD_LINK_RE = re.compile(r"\[([^\]\n]*)\]\([^)\n]*\)")
# The displayed prose of a wikilink is the alias when there is one, so the
# optional group swallows the target and the capture takes what a reader sees.
# Capturing before the pipe instead kept "Resources" and dropped "Writing
# Style", which is the half that is actually written English.
WIKILINK_RE = re.compile(r"\[\[(?:[^\]|\n]*\|)?([^\]\n]*)\]\]")
HTML_TAG_RE = re.compile(r"<[^>\n]{1,80}>")
# Markdown structure that is layout, not voice.
LIST_BULLET_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+", re.MULTILINE)
HEADING_RE = re.compile(r"^\s*#{1,6}\s+", re.MULTILINE)
TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$", re.MULTILINE)
BLOCKQUOTE_RE = re.compile(r"^\s*>\s?", re.MULTILINE)
EMPHASIS_RE = re.compile(r"(\*\*|\*|__|_)")

WORD_RE = re.compile(r"[A-Za-z][A-Za-z'’\-]*")

# Curly punctuation folded to straight before any pattern runs. Every detector
# written with a straight apostrophe -- not_x_but_y, isnt_about_its_about,
# heres_the_thing, hedge_adverbs -- scored ZERO on the curly form, and macOS
# emits curly by default, so those detectors were dark on a large share of real
# prose. Measured: not_x_but_y counted 1 on "it's not a bug, it's a flaw" and 0
# on the identical sentence with U+2019. Normalising here rather than widening
# a dozen character classes keeps one definition of what an apostrophe is.
SMART_PUNCT = str.maketrans({
    "’": "'", "‘": "'", "ʼ": "'", "“": '"',
    "”": '"', "′": "'", " ": " ", "‑": "-",
})
# Sentence split on terminal punctuation followed by whitespace. Deliberately
# simple: abbreviations produce a handful of short false splits, which moves
# mean sentence length by well under the precision anyone acts on.
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# Harness furniture that appears inside user records but was never typed by
# you. Counting it as your prose would poison the baseline with the model's
# own vocabulary, since several of these are written by hooks.
#
# A system reminder is the exception to dropping the whole record: the harness
# APPENDS it to the record holding what you typed, so it is cut out and the
# rest kept (see ``_user_text``). The markers below mean the record as a whole
# is harness output.
SYSTEM_REMINDER_RE = re.compile(r"<system-reminder>.*?</system-reminder>",
                                re.DOTALL)
USER_NOISE_MARKERS = (
    "<system-reminder>",
    "<local-command-caveat>",
    "<command-name>",
    "<command-message>",
    "<local-command-stdout>",
    "UserPromptSubmit hook",
    "Caveat: The messages below were generated",
    "[Request interrupted",
    "<task-notification>",
    "tool_use_error",
)

# Boilerplate carried by every scheduled-task prompt. A scheduled run logs its
# prompt as a user record, so without this the automation's own wording is
# scored as your voice: the first run of the baseline direction ranked
# "execute autonomously without asking clarifying questions" and "is an
# automated run" as the human's top tics at ~364x. Both came from recurring
# scheduled-task prompts, not from anyone's keyboard.
#
# A session carrying any of these is dropped ENTIRELY, both sides, rather than
# turn by turn. The model's replies in an unattended run are addressed to a log
# rather than to a reader, which is a different register; leaving them in would
# fix his corpus and quietly corrupt the model's.
AUTOMATED_SESSION_MARKERS = (
    "is an automated run",
    "the user is not present",
    "execute autonomously without asking clarifying questions",
    "This is a scheduled task",
    "scheduled task run",
)

# Text stored in an assistant record that the model never composed. The first
# run of this scan ranked "you've hit your" as a top-30 tic at 113x and put it
# second in the opener table; it is the harness's rate-limit notice. A usage
# notice counted as voice is the same class of error as counting a hook's
# output as your prose, and it is why both sides get a filter.
#
# A marker matches only at the START of the record, because a harness notice
# is the whole record. As substrings anywhere, "rate limit" and "API Error"
# deleted the model's own explanations of rate limiting, and "rate limit" is
# gone altogether: no notice opens with it. Measured on the reference corpus
# (554 transcripts, 2026-09-13): of 194 assistant records the substring
# markers matched, 174 were ``<synthetic>`` notices, which the synthetic
# filter drops first anyway, and 18 of the remaining 20 were the model's own
# prose with the marker mid-text.
ASSISTANT_NOISE_MARKERS = (
    "you've hit your",
    "You've hit your",
    "No response requested",
    "Claude usage limit reached",
    "[Request interrupted",
    "API Error",
)

# Stands in for a span ``scrub`` removed (code, a URL, a path, a tag) when the
# text is headed for the n-gram tables. It is whitespace to every regex, so
# the structure detectors and the sentence splitter see exactly what a space
# would give them, but it is not a space, so ``Corpus.add`` can refuse to
# build an n-gram across it. U+2029 PARAGRAPH SEPARATOR: never typed, and not
# a newline, so line-based passes are unaffected.
SCRUB_GAP = "\u2029"

# YOUR OWN signature phrases, checked by name rather than discovered, and
# therefore empty until you fill in ``[baseline] signatures`` in
# voice_tics.toml. See voice_tics.toml.example for a worked set.
#
# Empty is the honest default. Every other table in this module is a claim
# about how the MODEL writes, which is the same for everyone running this and
# belongs in version control. This one is a claim about how ONE PERSON writes,
# and a shipped default would be a stranger's habits presented as yours.
#
# WHY DISCOVERY CANNOT FILL THIS IN FOR YOU
#     Discovery ranks a speaker against the OTHER speaker, so pointed at your
#     turns it surfaces your REGISTER — short, imperative, conversational,
#     because that is what a person typing instructions to an agent sounds
#     like — rather than your HABITS. The phrases you actually reach for in
#     long-form writing barely appear in a chat transcript at all, so no
#     threshold over this corpus will ever promote them. Naming them is the
#     only way they get counted, and knowing them is the part only you can do.
#
#     The reference author named "However" himself, as the word he actively
#     works to cut. No n-gram ranking would have found it: he writes it in
#     documents, not in chat.
#
# KEEP THE PROVENANCE SPLIT WHEN YOU FILL IT IN
#     In the reference set, twelve entries came from a profiled writing-style
#     source and five were untested conversational-filler hypotheses. Those
#     are different kinds of claim — one is a fact about the author's voice,
#     the other is a thing to measure — and a flat list loses the difference.
#     A ratio near 1.0 on a hypothesis entry means the hypothesis was wrong;
#     that is the entry doing its job, not a bug.
BASELINE_SIGNATURES: Tuple[Tuple[str, str], ...] = ()


def fence_spans(text: str) -> List[Tuple[int, int]]:
    """The spans ``CODE_FENCE_RE`` would match, found in ONE pass.

    Same answer as the regex, without its worst case. ``CODE_FENCE_RE`` is
    lazy and multiline, so on a document whose fences are never closed it
    restarts at every opening line and scans to end of input each time: O(n^2),
    measured at 0.047 / 0.196 / 0.788 / 3.160 seconds for 1,000 / 2,000 /
    4,000 / 8,000 unclosed fence lines. A document demonstrating markdown
    syntax is enough to trigger it, and ``scrub`` runs over every record in a
    corpus.

    It is linear because openers and closers are each found once, and the
    closer index only ever moves FORWARD: no opener rescans text an earlier
    opener already passed, which is precisely what the regex does. The
    ``break`` is a small saving on top, not the reason -- with it replaced by
    ``continue`` the answer and the complexity are both unchanged, which is
    why no mutant guards it.

    Args:
        text: Raw markdown or chat text.

    Returns:
        Ascending, non-overlapping (start, end) offsets, exactly as
        ``CODE_FENCE_RE.finditer`` would yield them.
    """
    closers = [(m.start(), m.end()) for m in FENCE_CLOSE_RE.finditer(text)]
    spans: List[Tuple[int, int]] = []
    index, after = 0, 0
    for opener in FENCE_OPEN_RE.finditer(text):
        if opener.start() < after:
            continue
        # A fence needs the newline between opener and closer, so the closer
        # must start strictly after the opener does.
        while index < len(closers) and closers[index][0] <= opener.start():
            index += 1
        if index >= len(closers):
            break
        spans.append((opener.start(), closers[index][1]))
        after = closers[index][1]
    return spans


def strip_fences(text: str, repl: str = " ", keep_lines: bool = False) -> str:
    """``text`` with every fenced block replaced by ``repl``.

    The fence pass of ``scrub``, split out so it can use ``fence_spans``.

    Args:
        text: Raw markdown or chat text.
        repl: Literal replacement for each block. Not a template; the fence
            pattern has no groups to expand.
        keep_lines: Re-append the newlines each block held, so a caller that
            maps offsets back to line numbers stays exact.
    """
    out: List[str] = []
    last = 0
    for start, end in fence_spans(text):
        out.append(text[last:start])
        out.append(repl + "\n" * (text.count("\n", start, end)
                                  - repl.count("\n")) if keep_lines else repl)
        last = end
    out.append(text[last:])
    return "".join(out)


def scrub(text: str, keep_lines: bool = False, gap: str = " ") -> str:
    """Strip everything that is not authored prose.

    Removes code, URLs, paths, markdown link targets and markdown layout marks,
    leaving the sentences a reader would actually hear. Link and wikilink text
    is kept because it is written prose; the target is dropped because it is an
    address.

    Args:
        text: Raw markdown or chat text.
        keep_lines: Preserve the line count through EVERY pass, so a caller
            that maps match offsets back to line numbers — prose_lint.py —
            stays exact. Not just the fence pass: the ``^\\s*`` layout
            regexes (heading, bullet, table row, blockquote) all consume a
            preceding blank line's newline because ``\\s`` matches ``\\n``,
            so each substitution re-appends the newlines its match swallowed
            (2026-08-13 review: the fence-only version drifted one line per
            blank-line-preceded heading on real markdown; link and wikilink
            passes are single-line by design, see MD_LINK_RE). Corpus
            counting does not care about geometry, hence the default.
        gap: What replaces a removed span. The corpus readers pass
            ``SCRUB_GAP`` so no n-gram joins the words either side of
            deleted code; everything else keeps a space.
    """

    def _keep(repl: str):
        """A sub() replacement that re-appends the newlines the match ate."""
        def f(m: "re.Match[str]") -> str:
            out = m.expand(repl)
            return out + "\n" * (m.group(0).count("\n") - out.count("\n"))
        return f

    passes: Tuple[Tuple[re.Pattern[str], str], ...] = (
        (INLINE_CODE_RE, gap),
        (MD_LINK_RE, r"\1"),
        (WIKILINK_RE, r"\1"),
        (URL_RE, gap),
        (PATH_RE, gap),
        (HTML_TAG_RE, gap),
        (TABLE_ROW_RE, gap),
        (HEADING_RE, ""),
        (LIST_BULLET_RE, ""),
        (BLOCKQUOTE_RE, ""),
        (EMPHASIS_RE, ""),
    )
    text = text.translate(SMART_PUNCT)
    # The fence pass runs first and separately: it is the only one that must
    # cross line boundaries, and the only one with a quadratic regex.
    text = strip_fences(text, gap, keep_lines)
    for rx, repl in passes:
        text = rx.sub(_keep(repl) if keep_lines else repl, text)
    return text


def words(text: str) -> List[str]:
    """Lowercased word tokens of a prose string."""
    return [w.lower() for w in WORD_RE.findall(text)]


def sentences(text: str) -> List[str]:
    """Non-empty sentences of a prose string."""
    out = []
    for chunk in text.split("\n"):
        chunk = chunk.strip()
        if not chunk:
            continue
        for s in SENT_SPLIT_RE.split(chunk):
            s = s.strip()
            if s:
                out.append(s)
    return out


# ------------------------------------------------------------------- corpora
@dataclasses.dataclass
class Turn:
    """One authored message: who wrote it, when, and its scrubbed prose."""

    role: str
    when: Optional[dt.datetime]
    text: str
    session: str
    # Whether this record begins a response. A response that calls a tool is
    # stored as several assistant records, and only the first one after
    # something you sent is an opening; the rest continue after a tool result.
    opens: bool = True


def _user_text(content: object) -> Optional[str]:
    """Prose you actually typed, or None for harness-generated records.

    A user record may be a bare string or a block list. A record carrying a
    tool_result block is a tool round-trip, machine output end to end, so it
    is dropped wholesale. Any other non-text block -- an image, mostly -- is
    merely skipped: the text blocks beside it are a caption you typed,
    and dropping the whole record silently lost that prose from your baseline
    (16 such records in the live corpus when this was measured).

    System reminders get the same treatment for the same reason: the harness
    appends them to the record that holds your prompt, so each closed
    ``<system-reminder>`` span is cut out and what remains is judged on its
    own. An unclosed one still drops the record, because the rest of it cannot
    be told apart from the reminder.
    """
    if isinstance(content, str):
        raw = content
    elif isinstance(content, list):
        parts = []
        for b in content:
            if not isinstance(b, dict):
                continue
            if b.get("type") == "tool_result":
                return None
            if b.get("type") == "text":
                parts.append(b.get("text") or "")
        raw = "\n".join(parts)
    else:
        return None
    raw = SYSTEM_REMINDER_RE.sub("\n", raw)
    if not raw.strip():
        return None
    if any(marker in raw for marker in USER_NOISE_MARKERS):
        return None
    return raw


def _is_tool_result(content: object) -> bool:
    """Whether a user record is a tool round-trip rather than something sent."""
    return isinstance(content, list) and any(
        isinstance(b, dict) and b.get("type") == "tool_result" for b in content)


def _assistant_text(content: object) -> Optional[str]:
    """The user-facing prose of an assistant record, or None.

    Only ``text`` blocks count. ``thinking`` is excluded deliberately: it is a
    different register, nobody reads it as the model's voice, and including it
    would swamp the counts.
    """
    if not isinstance(content, list):
        return None
    parts = [b.get("text") or "" for b in content
             if isinstance(b, dict) and b.get("type") == "text"]
    raw = "\n".join(parts)
    if not raw.strip():
        return None
    if raw.lstrip().startswith(ASSISTANT_NOISE_MARKERS):
        return None
    return raw


def _parse_when(rec: dict) -> Optional[dt.datetime]:
    """The record timestamp as an aware datetime, or None if unparseable.

    A timestamp with no UTC offset is read as UTC, which is what the harness
    writes. Left naive, it could not be compared with the aware ``since``
    cutoff, and one such record aborted the whole scan, but only under
    ``--days``.
    """
    ts = rec.get("timestamp")
    if not isinstance(ts, str):
        return None
    try:
        when = dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None
    return when if when.tzinfo else when.replace(tzinfo=dt.timezone.utc)


def read_turns(paths: Sequence[str],
               since: Optional[dt.datetime] = None,
               stats: Optional[Dict[str, int]] = None,
               model: Optional[str] = None) -> Iterator[Turn]:
    """Yield authored turns from interactive session transcripts.

    Sidechain (subagent) records are skipped: they are a different job with a
    different prompt, and mixing them in would attribute a subagent's register
    to the main voice. Whole sessions that turn out to be scheduled runs are
    skipped too, for the reason given at ``AUTOMATED_SESSION_MARKERS``.

    Assistant records whose ``message.model`` is ``<synthetic>`` are always
    dropped: those are harness-composed stand-ins, not any model's prose.

    Args:
        paths: Session .jsonl paths. One file is one session, which is what
            makes the session-level automated test possible in a single pass.
        since: Drop turns older than this instant.
        stats: Optional counter dict, updated in place with
            ``sessions``/``automated_sessions``/``turns`` (and the exclusion
            counters ``meta_records``/``synthetic_records``/
            ``model_filtered``/``noise_records``) so the caller can report
            what was excluded instead of excluding it silently.
        model: Substring an assistant record's ``message.model`` must contain,
            or None for all models. Without it every figure downstream is a
            blend across every model that ever wrote a transcript here, not
            the voice of any one of them. Your turns are never filtered:
            your baseline is yours, whichever model you were talking to.

    Yields:
        One ``Turn`` per authored message, prose already scrubbed.
    """
    for path in paths:
        session = os.path.basename(path)[:8]
        buffered: List[Turn] = []
        automated = False
        # True until the first assistant prose after something you sent, so
        # the opener table counts responses rather than records.
        opens = True
        try:
            fh = open(path, encoding="utf-8", errors="replace")
        except OSError:
            continue
        with fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except (ValueError, TypeError):
                    continue
                # A line can be valid JSON without being an object. A bare
                # string or array reached .get() and killed the whole run with
                # AttributeError, losing every session after the bad line --
                # a scan that silently reports fewer sessions is worse than one
                # that reports none, so this is a guard rather than a try.
                # The same holds one level down, for ``message``.
                if not isinstance(rec, dict):
                    continue
                kind = rec.get("type")
                if kind not in ("user", "assistant"):
                    continue
                if rec.get("isSidechain"):
                    continue
                message = rec.get("message")
                if not isinstance(message, dict):
                    message = {}
                content = message.get("content")
                if kind == "assistant":
                    rec_model = message.get("model") or ""
                    if not isinstance(rec_model, str):
                        rec_model = ""
                    if rec_model == "<synthetic>":
                        _count(stats, "synthetic_records")
                        continue
                    if model is not None and model not in rec_model:
                        _count(stats, "model_filtered")
                        continue
                    text = _assistant_text(content)
                    if text is None and _has_text(content):
                        _count(stats, "noise_records")
                else:
                    if not _is_tool_result(content):
                        opens = True
                    text = _user_text(content)
                    # The session test runs BEFORE the date cutoff and before
                    # isMeta, so neither can hide the one record that marks a
                    # scheduled run: a --days cutoff falling after the prompt
                    # re-admitted the rest of that session, and the report
                    # then said nothing had been dropped.
                    if text and any(m in text for m in AUTOMATED_SESSION_MARKERS):
                        automated = True
                        break
                    # isMeta marks a user record the harness composed rather
                    # than you: slash-command expansions and similar. It is
                    # the single largest contaminant found so far -- 55% of
                    # your baseline by word count (120,755 words down to
                    # 54,688) -- and it compressed every ratio toward 1
                    # because so much of "his" corpus was machine text.
                    # Dropping it changed appositive_negation from 1.1x to
                    # 2.4x, which is the difference between contradicting
                    # your own observation about your writing and confirming
                    # it. Counted before the noise filter, so every isMeta
                    # record dropped is a record reported.
                    if rec.get("isMeta"):
                        _count(stats, "meta_records")
                        continue
                when = _parse_when(rec)
                if since is not None and when is not None and when < since:
                    if kind == "assistant" and text:
                        opens = False
                    continue
                if not text:
                    continue
                turn_opens = opens if kind == "assistant" else True
                if kind == "assistant":
                    opens = False
                buffered.append(Turn(role=kind, when=when,
                                     text=scrub(text, gap=SCRUB_GAP),
                                     session=session, opens=turn_opens))
        if stats is not None:
            stats["sessions"] = stats.get("sessions", 0) + 1
            if automated:
                stats["automated_sessions"] = (
                    stats.get("automated_sessions", 0) + 1)
        if automated:
            continue
        if stats is not None:
            stats["turns"] = stats.get("turns", 0) + len(buffered)
        yield from buffered


def _count(stats: Optional[Dict[str, int]], key: str) -> None:
    """Add one to ``stats[key]`` when the caller asked for counters."""
    if stats is not None:
        stats[key] = stats.get(key, 0) + 1


def _has_text(content: object) -> bool:
    """Whether an assistant record carries any non-blank text block."""
    return isinstance(content, list) and any(
        isinstance(b, dict) and b.get("type") == "text"
        and str(b.get("text") or "").strip() for b in content)


# ------------------------------------------------------------------ counting
# YAML front matter: a "---" first line, then non-blank lines, then a closing
# "---" or "...". The closer must come before the first blank line. Without
# that bound, a file that merely OPENS with a thematic break, or whose closing
# fence carries a trailing space, lost everything up to the next rule anywhere
# in the file, silently; with it such a file is left whole, which is the safe
# way to be wrong. Each line starts ``[ \t]*\S`` so a failed match cannot
# backtrack through the ways of splitting a line.
FRONT_MATTER_RE = re.compile(
    r"\A---[ \t]*\n(?:[ \t]*\S[^\n]*\n)*?(?:---|\.\.\.)[ \t]*(?:\n|\Z)")


def read_samples(pattern: str,
                 stats: Optional[Dict[str, int]] = None) -> Iterator[str]:
    """Scrubbed paragraphs from a glob of your own writing.

    The alternative baseline. Instead of your turns in the transcripts, point
    this at documents you wrote — posts, reports, papers, anything in markdown
    or plain text — and the model is ranked against your WRITING register
    rather than your CHAT register.

    WHICH BASELINE TO USE, AND WHAT EACH ONE COSTS
        They fail in opposite directions, so pick by what you are linting.

        Transcripts (the default) are matched on medium, topic and week — the
        same conversation, the same technical vocabulary, the same fortnight.
        That match is what lets a ratio mean something. What they are NOT
        matched on is register: instructions typed to an agent are short and
        imperative, so the constructions you reach for in long-form prose
        barely occur, and no threshold over this corpus will ever surface
        them. Measured on the reference author's transcripts over a 14-day
        window on 2026-09-12: 8.9 words per sentence on the chat side against
        12.2 on the model's, from only 249 baseline words, so read it as the
        direction of the gap rather than its size. (The 13.89 / 13.85 figures
        elsewhere in this repo are a different measurement: a 45-day window
        on 2026-08-13, after harness-composed records were removed.)

        Samples are matched on register and on author, which is what you
        actually want when the thing being linted is a document. What they
        give up is the topic and time match: a ratio against three-year-old
        blog posts about other subjects partly measures the subject change,
        not the writer. Shared technical vocabulary inflates on both sides.

        Use samples when you are linting documents, which is most of the time.
        Use transcripts when you want to know how the model writes to YOU,
        specifically, right now. Neither is the "real" answer; they answer
        different questions, and the report names which one it used.

    A SAMPLE CORPUS CAN ALSO BE A TARGET RATHER THAN A MIRROR
        Nothing here requires the samples to be yours. Point it at prose you
        want to sound more like and the ratios become distance-from-target
        instead of tic-over-baseline — the same arithmetic, a different claim.
        Be deliberate about which you are doing: a target corpus makes the
        report advice about a voice you are adopting, not evidence about a
        voice you have. The tool cannot tell the two apart and does not try.

    PARAGRAPHS, NOT WHOLE FILES
        Each blank-line-separated paragraph counts as one unit, so the opener
        table has something to measure. Counting a whole file as one unit
        would yield one opener per document and an opener distribution built
        from a handful of observations, which is not a distribution.

    Args:
        pattern: A glob, ``~`` expanded. ``**`` recurses only when the glob
            itself says so (``docs/**/*.md``).
        stats: Optional counter dict; ``sample_files`` and ``sample_paragraphs``
            are added to it.

    Yields:
        Scrubbed paragraph prose, ready for ``Corpus.add``. Files that cannot
        be read are skipped and counted in ``stats['sample_unreadable']``
        rather than killing the run: one unreadable file in a corpus of
        hundreds should not cost you the measurement.
    """
    counts = stats if stats is not None else {}
    for path in sorted(glob.glob(os.path.expanduser(pattern), recursive=True)):
        if not os.path.isfile(path):
            continue
        try:
            # utf-8-sig: a byte-order mark would otherwise sit in front of the
            # front matter fence and hide it.
            with open(path, encoding="utf-8-sig") as fh:
                raw = fh.read()
        except (OSError, UnicodeDecodeError):
            counts["sample_unreadable"] = counts.get("sample_unreadable", 0) + 1
            continue
        counts["sample_files"] = counts.get("sample_files", 0) + 1
        # Front matter is metadata, not prose, and its keys would otherwise
        # rank as phrases. See FRONT_MATTER_RE for what counts as front matter.
        raw = FRONT_MATTER_RE.sub("", raw, count=1)
        # Fences go BEFORE the paragraph split. Split first, a code block
        # with a blank line inside it arrived in two pieces, neither holding
        # both fence lines, so its code was counted as your prose.
        raw = strip_fences(raw, "", keep_lines=True)
        for para in re.split(r"\n\s*\n", raw):
            prose = scrub(para, gap=SCRUB_GAP)
            if words(prose):
                counts["sample_paragraphs"] = counts.get("sample_paragraphs", 0) + 1
                yield prose


# A ratio this close to 1.0 says the two corpora are indistinguishable on that
# construction. Band chosen as a quarter in either direction (0.8 = 1/1.25), so
# it is symmetric in log space rather than arithmetically lopsided.
INDISTINGUISHABLE = (0.8, 1.25)

# A structure the model uses at least this often while the baseline never uses
# it counts as evidence the corpora DIFFER. Fewer uses than this against a zero
# is too thin to say anything, the same floor ``--min-count`` applies to
# phrases.
ONE_SIDED_MIN_COUNT = DEFAULT_MIN_COUNT

# Fraction of comparable structures inside that band above which a sample
# baseline is reported as suspect. Set high on purpose: an author who genuinely
# writes like the model is a real outcome and should not be second-guessed on
# thin evidence. Two thirds of every measured construction landing inside a
# quarter of parity is not that; it is the corpora sharing an author.
CONTAMINATION_SHARE = 0.66


def contamination_warning(structures: Dict[str, Tuple[int, int]],
                          mine: "Corpus", theirs: "Corpus") -> Optional[str]:
    """A warning when a sample baseline looks like it contains model prose.

    THE FAILURE THIS CATCHES
        A sample corpus is only a baseline if YOU wrote it. Point this at
        documents an assistant drafted or edited and the model's prose is on
        both sides of the ratio, so every tic divides by itself and lands near
        1.0. The report then says "no tics here" — silently, and in the
        direction that passes. That is the worst way for a measurement tool to
        be wrong, and nothing about the output would otherwise reveal it.

        The mechanism is arithmetic rather than conjecture: a construction
        the model produces at rate r appears on both sides at about r, and
        r/r is 1. How much assistance it takes to matter has not been
        measured, so this warns rather than fails, and the threshold below is
        deliberately reluctant.

    WHAT IT DOES NOT CATCH: PARTIAL CONTAMINATION
        Dilution moves every ratio smoothly toward 1.0, and this only notices
        once most constructions are inside the band. On a synthetic sweep
        (2026-09-13) it stayed quiet up to 70% model-written paragraphs and
        fired from 78%, while a tic that read 593x against clean samples read
        3.2x at 25% contamination and 1.3x at 70%, in silence both times. So
        a quiet guard is not evidence the samples are clean; only knowing
        where the samples came from is.

    WHY A COUNT AND NOT A CLASSIFIER
        Detecting "was this written by a model" per document is the thing this
        whole repo argues cannot be done by wordlist and should not be done by
        a model in the loop. So it is not attempted. Instead: count how many
        constructions land inside a quarter of parity. That number is cheap,
        deterministic, interpretable, and it is evidence for the reader rather
        than a verdict — which is the same contract as every other number here.

    Args:
        structures: name -> (mine_count, theirs_count).
        mine: The corpus under examination.
        theirs: The baseline corpus.

    Returns:
        A warning string, or None when the corpora look distinguishable or
        when there is too little data to say.
    """
    ratios = []
    separated = 0
    for m_count, t_count in structures.values():
        # Constructions BOTH corpora use are compared by ratio. One the model
        # uses and the baseline never does is not a ratio (a zero there is
        # the floor correction talking) but it is not silence either: it is
        # the strongest evidence the baseline holds no model prose. Dropping
        # it, as this once did, left only the constructions humans and models
        # genuinely share, which sit near parity by nature, so the guard fired
        # on corpora that were entirely the author's own. A construction only
        # the BASELINE uses says nothing about contamination and is skipped.
        if m_count and t_count:
            m_rate, t_rate = mine.rate(m_count), theirs.rate(t_count)
            if t_rate:
                ratios.append(m_rate / t_rate)
        elif m_count >= ONE_SIDED_MIN_COUNT and not t_count:
            separated += 1
    comparable = len(ratios) + separated
    if comparable < 5:
        return None
    low, high = INDISTINGUISHABLE
    near = sum(1 for r in ratios if low <= r <= high)
    share = near / comparable
    if share < CONTAMINATION_SHARE:
        return None
    return (f"WARNING: {near} of {comparable} comparable structures sit "
            f"within a quarter of parity.\n"
            "That is what a baseline containing model-written prose looks "
            "like. If any\n"
            "of your samples were drafted or edited by an assistant, the "
            "model is on both\n"
            "sides of every ratio and the tics divide out. Narrow the glob "
            "to prose you\n"
            "wrote unaided, then re-run. If the samples really are all "
            "yours, this is\n"
            "instead a finding: you and the model write alike.")


def ngrams(tokens: Sequence[str], width: int) -> Iterator[Tuple[str, ...]]:
    """Every contiguous ``width``-token window."""
    for i in range(len(tokens) - width + 1):
        yield tuple(tokens[i:i + width])


@dataclasses.dataclass
class Corpus:
    """Token counts and n-gram tables for one speaker."""

    label: str
    total_words: int = 0
    total_sentences: int = 0
    turns: int = 0
    grams: Dict[int, collections.Counter] = dataclasses.field(
        default_factory=lambda: collections.defaultdict(collections.Counter))
    openers: collections.Counter = dataclasses.field(
        default_factory=collections.Counter)
    sentence_lengths: List[int] = dataclasses.field(default_factory=list)

    def add(self, text: str, opens: bool = True) -> None:
        """Fold one turn's prose into the tables.

        N-grams are taken within one sentence and within one run of text
        ``scrub`` left intact (split at ``SCRUB_GAP``). Taken over the whole
        turn's token stream, the table ranked strings nobody wrote: the last
        words of one sentence welded to the first of the next, and the words
        either side of a deleted code span. Both effects grow with turn
        length and code density, which are far higher on the model's side,
        so the phantom grams landed in the ratio's numerator.

        Args:
            text: Scrubbed prose of one turn, paragraph or record.
            opens: Whether this text begins a response. Only then does its
                first sentence count toward the opener table; a record that
                continues a response after a tool result opens nothing.
        """
        self.turns += 1
        sents = sentences(text)
        self.total_sentences += len(sents)
        for s in sents:
            self.sentence_lengths.append(len(words(s)))
        self.total_words += len(words(text))
        for s in sents:
            for run in s.split(SCRUB_GAP):
                toks = words(run)
                for width in NGRAM_WIDTHS:
                    self.grams[width].update(ngrams(toks, width))
        if sents and opens:
            first = words(sents[0])[:3]
            if first:
                self.openers[" ".join(first)] += 1

    def rate(self, count: float) -> float:
        """Occurrences per 10,000 words."""
        if not self.total_words:
            return 0.0
        return count * 10000.0 / self.total_words

    def mean_sentence_len(self) -> float:
        """Mean words per sentence."""
        if not self.sentence_lengths:
            return 0.0
        return sum(self.sentence_lengths) / len(self.sentence_lengths)


# --------------------------------------------------------------- structures
# Templated constructions that read as machine-written even when the individual
# words are fresh. Each is a named, deterministic pattern; the report gives a
# rate per 10k words for both speakers so a shared habit is visible as shared.
STRUCTURE_PATTERNS: Tuple[Tuple[str, str, str], ...] = (
    (
        "not_x_but_y",
        # The leading group needs the contracted negations spelled out:
        # "is\s+not" cannot match inside "isn't", so "The problem isn't the
        # code, it's the config" -- likely the majority surface form of the
        # reveal -- scored zero until the contractions were added.
        r"\b(?:(?:is|are|was|were|it'?s)\s+not|(?:isn|aren|wasn|weren)'?t)\s+"
        r"(?:just\s+)?[^.;:!?]{2,40}?[,—-]\s*"
        r"(?:it'?s|they'?re|but)\b",
        "'X is not just A, it's B' — the explicit reveal template",
    ),
    (
        "appositive_negation",
        # The baseline author named this on 2026-08-12 and the first detector
        # scored it at 9
        # occurrences, which was obviously wrong -- the model had used it four
        # times in that same conversation. The reason: the explicit "it's not
        # X, it's Y" form above is the *rare* variant. The common one is the
        # trailing appositive, "Y, not X", which the reveal pattern cannot see
        # because there is no second copula.
        #
        # What separates the tic from grammar is the COMPLEMENT: a participial
        # clause's gerund takes one ("I asked, not knowing WHY") while the
        # tic's negated alternative ends there ("measuring, not guessing").
        # So an -ing word followed by another word is treated as a clause and
        # skipped -- whatever the word before the comma ends in, which is what
        # retired the round-one parallel-gerund branch that miscounted
        # "this morning, not knowing where I was". The -thing pronouns are
        # carved out because they are nouns whatever follows them
        # ("deliberate, not something to fix"). Two accepted imperfections,
        # both rare and both toward under-counting, the safe direction: a
        # NOUN in -ing with a complement is skipped as if it were a clause
        # ("a rule, not string matching"), and a qualifier lets a clause
        # through by backtracking (", not just knowing why" counts). The
        # dash form requires the dash to be a separator -- an em dash, or a
        # hyphen spaced on both sides -- so the hyphen inside "absent-minded"
        # no longer counts.
        r",\s+not\s+(?:just\s+|only\s+|merely\s+|simply\s+)?"
        r"(?:(?:some|any|every|no)thing\b[^.;:!?]{0,60}"
        r"|(?!\w+ing\s+\w)[^.;:!?]{2,60})"
        r"|\bnot\s+(?:just|only|merely|simply)\s+[^.;:!?]{2,50}?"
        r"(?:\s*—\s*|\s-\s)",
        "'Y, not X' trailing negated alternative",
    ),
    (
        "isnt_about_its_about",
        r"\b(?:is|isn'?t|not)\s+about\s+[^.;:!?]{2,40}?[,.]?\s*"
        r"(?:it'?s|but)\s+about\b",
        "'not about X, it's about Y'",
    ),
    (
        "heres_the_thing",
        r"\b(?:here'?s|that'?s)\s+(?:the\s+)?(?:thing|key|catch|kicker|point|"
        r"problem|rub)\b",
        "'Here's the thing' / 'that's the catch' scene-setting",
    ),
    (
        "the_real_x_is",
        r"\bthe\s+(?:real|actual|deeper|underlying|bigger)\s+"
        r"(?:question|issue|problem|point|answer|risk|reason)\s+(?:here\s+)?is\b",
        "'the real question is' escalation",
    ),
    (
        "method_defence",
        # The first branch takes the VERB forms only: with the -s optional,
        # "matters?" matched the noun in "we will discuss this matter", which
        # is the baseline author's formal register rather than the tic, so the
        # baseline
        # was inflated and the ratio understated. Plural agreement ("results
        # which matter") is forfeited -- undercounting, the safe direction.
        # The old "(?:is\s+)?" was dead weight ("which is means" is not
        # English) and went with it.
        r"\b(?:which|this)\s+(?:means|matters)\s+(?:that\s+)?"
        r"|\bthe\s+(?:measurement|method|approach|script|check)\s+"
        r"(?:is|was|does|deliberately)\b"
        r"|\bwhich is (?:exactly )?(?:why|what|the point)\b"
        r"|\bthat is (?:the point|deliberate|by design)\b",
        "defending the method instead of stating the finding",
    ),
    (
        "worth_noting",
        r"\b(?:it'?s|it is)\s+worth\s+(?:noting|pointing out|remembering|"
        r"flagging)\b|\bnote\s+that\b|\bkeep\s+in\s+mind\b",
        "'worth noting that' hedge-aside",
    ),
    (
        "let_me",
        # Any verb, not a list of them. The first version enumerated
        # check/look/see/start/... and silently missed "let me verify", which
        # was 158 uses and a top-five tic in its own right -- the detector for
        # the largest measured habit was undercounting it. Classify positively
        # and carve out the one different sense: "let me know" is addressed to
        # the reader, not narration of the model's own next move.
        r"\blet\s+me\s+(?!know\b)(?:just\s+)?\w+",
        "'let me check / verify / start' narration of own next move",
    ),
    (
        "i_should_note",
        # The contraction has no space, so "\bi\s+'?ll" could never match
        # "I'll flag" -- the commonest surface form of the habit -- and "will"
        # was missing outright, so "I will note" scored zero too. The
        # alternation now splits contraction from spaced modal.
        r"\bi(?:\s+(?:should|want\s+to|need\s+to|will)|'ll)\s+"
        r"(?:note|flag|point\s+out|mention|be\s+clear|caveat)\b",
        "'I should flag' self-narration",
    ),
    (
        "rule_of_three",
        # Was rule_of_three_dash, a misleading name: the pattern never had a
        # dash in it. It also capped the first item at one word and the rest
        # at two, so "read the file, run the query, and check the result"
        # -- a genuine three-item list -- did not count. Items are now up to
        # three words each, and "or" lists count alongside "and".
        r"\b\w+(?:\s+\w+){0,2},\s+\w+(?:\s+\w+){0,2},\s+(?:and|or)\s+\w+",
        "three-item list (only a tic when the rate is far above baseline)",
    ),
    (
        "rule_of_three_no_oxford",
        # The non-Oxford surface form "A, B and C" is regex-indistinguishable
        # from clause coordination. Measured on the live corpus 2026-08-13:
        # the form would have added 369 model / 39 baseline matches, and the
        # samples were overwhelmingly "clause, clause and clause" ("that's
        # done, let me know and I'll go") or an intro phrase plus a pair
        # ("in fact, can we schedule and auto..."), with genuine lists rare.
        # So the form gets its own labelled row instead of widening
        # rule_of_three, which would have poisoned a precise count with ~44%
        # noise. Same design as the not_x_but_y / appositive_negation split:
        # both comma styles are counted, separately.
        r"\b\w+(?:\s+\w+){0,2},\s+\w+(?:\s+\w+){0,2}\s+(?:and|or)\s+\w+",
        "non-Oxford 'A, B and C' (LOW PRECISION: mostly clause coordination)",
    ),
    (
        "em_dash_aside",
        r"—",
        "em dash — the most-cited AI tell; measure it before you gate on it",
    ),
    # ---- crowd-sourced tells, collected 2026-08-12 from the Wikipedia
    # WikiProject AI Cleanup "signs of AI writing" page, tropes.fyi, and
    # stephenturner/skill-deslop. These are HYPOTHESES, not rules. The whole
    # point of measuring them against a matched baseline is that the folk list
    # is not automatically true of this model: the lexicon below scores 0.6x
    # on the cleaned reference corpus (model 1.47 vs baseline 2.38 per 10k,
    # 2026-08-13), meaning the baseline author used these words 1.6 times as
    # often as the model did.
    # (The 0.3x this comment first carried was measured against the isMeta-
    # contaminated corpus.) A rule adopted from the list unmeasured would have
    # "corrected" the wrong writer.
    (
        "delve_ecosystem",
        r"\b(?:delve|leverage|robust|seamless(?:ly)?|holistic|nuanced|"
        r"multifaceted|underscor\w+|pivotal|crucial|realm|landscape|tapestry|"
        r"testament|myriad|plethora|meticulous\w*|vibrant|intricate\w*|"
        r"paradigm|synergy|cornerstone|embark|garner|bolster\w*|foster\w*|"
        r"showcase|streamline|utilize|groundbreaking|transformative|"
        r"cutting-edge|game-changer|interplay|navigate the|deep dive)\b",
        "the flagged-slop lexicon (crowd list)",
    ),
    (
        "copula_avoidance",
        r"\b(?:serves? as|stands? as|represents? a|marks? a|boasts)\b",
        "'serves as' instead of 'is' (crowd list)",
    ),
    (
        "vague_attribution",
        r"\b(?:experts?|observers?|critics?|analysts?|industry reports?|"
        r"studies|researchers?)\s+(?:argue|suggest|say|note|have|cite|claim)\b",
        "unnamed authorities in place of a source (crowd list)",
    ),
    (
        "signposted_conclusion",
        # "ultimately," sits outside the \b-closed group: a trailing \b after
        # a comma only matches when the NEXT character is a word character,
        # and "Ultimately, the ..." always has a space there, so inside the
        # group it could never match anything at all.
        r"\b(?:in conclusion|to sum up|in summary|at the end of the day|"
        r"all in all)\b|\bultimately,",
        "announcing the ending (crowd list)",
    ),
    (
        "false_range",
        r"\bfrom\s+\w+(?:\s+\w+){0,2}\s+to\s+\w+(?:\s+\w+){0,2}\b",
        "'from X to Y' as a dressed-up two-item list (crowd list)",
    ),
    (
        "rhetorical_self_question",
        r"(?:^|[.!?]\s+)(?:the\s+\w+\?|so\s+what|why\?|the\s+result\?|"
        r"the\s+catch\?|what\s+does\s+that\s+mean\?)",
        "posing a question nobody asked (crowd list)",
    ),
    (
        "throat_clearing_opener",
        r"\b(?:let'?s\s+(?:dive|unpack|break\s+this|take\s+a\s+step)|"
        r"great\s+question|i\s+hope\s+this\s+helps|without\s+further\s+ado|"
        r"in\s+today'?s\s+fast-paced|make\s+no\s+mistake|let\s+that\s+sink\s+in|"
        r"picture\s+this|ever\s+wondered)\b",
        "throat-clearing openers and emphasis crutches (crowd list)",
    ),
    (
        "hedge_adverbs",
        r"\b(?:really|just|literally|genuinely|honestly|simply|actually|"
        r"deeply|truly|fundamentally|inherently|inevitably|interestingly|"
        r"importantly|crucially|quietly|remarkably|arguably)\b",
        "hedge adverbs from the popular AI-slop kill-lists",
    ),
    (
        "not_only_but_also",
        r"\bnot\s+only\s+[^.;:!?]{2,50}?\s+but\s+(?:also\s+)?\b",
        "'not only X but also Y'",
    ),
    (
        "trailing_participle",
        r",\s+(?:making|allowing|ensuring|providing|enabling|highlighting|"
        r"reflecting|underscoring|demonstrating)\s+\w+",
        "trailing '-ing' summary clause bolted onto a sentence",
    ),
)


def structure_counts(text: str) -> Dict[str, int]:
    """Occurrences of each named structure in one prose string."""
    return {name: len(re.findall(pat, text, re.IGNORECASE))
            for name, pat, _ in STRUCTURE_PATTERNS}


def signature_counts(text: str,
                     signatures: Sequence[Tuple[str, str]] = BASELINE_SIGNATURES
                     ) -> Dict[str, int]:
    """Occurrences of each configured signature phrase in one prose string.

    Args:
        text: Scrubbed prose from one turn.
        signatures: (name, pattern) pairs from ``[baseline] signatures``.
            Defaults to the empty module table, so an unconfigured run
            returns ``{}`` and the SIGNATURES section simply does not print.
    """
    return {name: len(re.findall(pat, text, re.IGNORECASE))
            for name, pat in signatures}


# ------------------------------------------------------------------ analysis
@dataclasses.dataclass
class Finding:
    """One candidate tic with the evidence behind it."""

    phrase: str
    width: int
    mine: int
    mine_rate: float
    theirs: int
    theirs_rate: float
    ratio: float


def rank_phrases(mine: Corpus, theirs: Corpus, min_count: int,
                 top: int) -> List[Finding]:
    """Candidate phrase tics, most over-represented first.

    A phrase qualifies when the model used it at least ``min_count`` times.
    Shorter n-grams that are merely a window onto a better-ranked longer one
    are dropped, so a single habit reports once rather than five times.
    """
    found: List[Finding] = []
    for width in NGRAM_WIDTHS:
        for gram, count in mine.grams[width].items():
            if count < min_count:
                continue
            phrase = " ".join(gram)
            theirs_count = theirs.grams[width].get(gram, 0)
            theirs_rate = theirs.rate(max(theirs_count, BASELINE_FLOOR_COUNT))
            mine_rate = mine.rate(count)
            found.append(Finding(
                phrase=phrase, width=width, mine=count, mine_rate=mine_rate,
                theirs=theirs_count, theirs_rate=theirs_rate,
                ratio=mine_rate / theirs_rate if theirs_rate else 0.0))
    # Ties break longest-first. A fixed phrase with a zero baseline -- the
    # canonical single-habit case -- produces every window of itself at the
    # SAME ratio and count, and the stable sort used to leave the width-2
    # windows ranked first; _drop_subsumed only drops a phrase contained in
    # an already-kept one, so the whole ladder survived and one habit filled
    # ten of the top-N rows. Longest-first puts the containing phrase ahead
    # of its windows, and the windows collapse into it.
    found.sort(key=lambda f: (-f.ratio, -f.mine, -f.width))
    return _drop_subsumed(found)[:top]


def _drop_subsumed(found: Sequence[Finding]) -> List[Finding]:
    """Remove shorter phrases wholly contained in a better-ranked longer one.

    "is not just" and "it is not just a" are one habit. Keeping both would
    triple-count it and push genuinely separate tics off the report.
    """
    kept: List[Finding] = []
    # Widest-width findings already seen, kept or absorbed; a narrower window
    # of an absorbed one is contained in the same string. A repeated string
    # longer than the widest n-gram has no single row that contains it: its
    # widest windows are siblings, each shifted one token along, and
    # containment alone kept every one, so one sentence filled as many rows
    # as it had words and pushed distinct tics off the report. A widest
    # window that overlaps a seen one by all but one token, at a comparable
    # count, is that same string continuing.
    widest: List[Finding] = []
    top = max(NGRAM_WIDTHS)
    for f in found:
        # Containment is judged on whole tokens, so both sides are padded
        # with spaces. Raw substring matching dropped "own the" as
        # "contained" in "down the" -- two different habits, and the loss
        # was invisible because the dropped row simply never printed.
        padded = f" {f.phrase} "
        if any(padded in f" {k.phrase} " and f.mine <= k.mine * 1.35
               for k in kept + widest):
            continue
        if f.width == top:
            toks = f.phrase.split()
            if any(_shifted_by_one(toks, w.phrase.split())
                   and f.mine <= w.mine * 1.35 for w in widest):
                widest.append(f)
                continue
            widest.append(f)
        kept.append(f)
    return kept


def _shifted_by_one(a: Sequence[str], b: Sequence[str]) -> bool:
    """Whether two equal-length token windows overlap in all but one token."""
    return list(a[1:]) == list(b[:-1]) or list(a[:-1]) == list(b[1:])


# -------------------------------------------------------------------- report
def _fmt_rate(r: float) -> str:
    """A rate rendered at a precision that does not overstate it."""
    return f"{r:6.2f}"


def render(mine: Corpus, theirs: Corpus, phrases: Sequence[Finding],
           structures: Dict[str, Tuple[int, int]], opener_top: int,
           show_source_text: bool,
           signatures: Optional[Dict[str, Tuple[int, int]]] = None,
           speaker: str = "model",
           stats: Optional[Dict[str, int]] = None,
           keep_domains: Sequence[str] = (),
           baseline_source: str = "") -> str:
    """The human-readable report.

    ``mine`` is whichever speaker is under examination and ``theirs`` is the
    matched baseline, so the same renderer serves both directions.

    ``show_source_text`` gates the two tables that reproduce transcript
    fragments, phrases and opener rows. The gate is the protection; the
    ``redact`` pass on what prints cannot currently fire (see the module
    docstring). The opener concentration line is counts only, so it prints
    either way.
    """
    other = "baseline" if speaker == "model" else "model"
    out: List[str] = []
    out.append("=" * 78)
    out.append(f"VOICE TICS — {speaker} prose measured against {other}'s")
    # Which baseline produced these ratios is not a detail: the two answer
    # different questions and a report that does not say which one it used
    # cannot be read at all.
    out.append(f"Baseline: {baseline_source}" if baseline_source
               else "Baseline: your turns in the same transcripts")
    out.append("=" * 78)
    out.append("")
    out.append(f"{'':22}{speaker:>12}{other:>12}")
    out.append(f"{'turns':22}{mine.turns:>12,}{theirs.turns:>12,}")
    out.append(f"{'words':22}{mine.total_words:>12,}{theirs.total_words:>12,}")
    out.append(f"{'sentences':22}{mine.total_sentences:>12,}"
               f"{theirs.total_sentences:>12,}")
    out.append(f"{'mean words/sentence':22}{mine.mean_sentence_len():>12.1f}"
               f"{theirs.mean_sentence_len():>12.1f}")
    out.append("")
    if baseline_source:
        out.append("Caveat: a sample baseline matches your REGISTER but not")
        out.append("the model's topics or dates, so shared subject vocabulary")
        out.append("inflates both sides. Ratios rank candidates; they do not")
        out.append("convict.")
    else:
        out.append("Register caveat: your turns are short and directive, the")
        out.append("model's are long and explanatory, so explanatory")
        out.append("connectives are over-represented by construction. Ratios")
        out.append("rank candidates; they do not convict.")
    if stats:
        set_aside = stats.get("turns_set_aside", 0)
        out.append(f"Corpus: {stats.get('sessions', 0):,} sessions read, "
                   f"{stats.get('automated_sessions', 0):,} dropped as "
                   f"scheduled runs, {stats.get('turns', 0):,} turns kept"
                   + (f", {set_aside:,} of your turns set aside for the "
                      "sample baseline." if set_aside else "."))
        if baseline_source:
            # The counter that makes skipping an unreadable sample safe has
            # to reach the reader: losing files shrinks the baseline and
            # raises every ratio, and nothing else in the report shows it.
            out.append(f"Samples: {stats.get('sample_files', 0):,} files read, "
                       f"{stats.get('sample_paragraphs', 0):,} paragraphs "
                       f"kept, {stats.get('sample_unreadable', 0):,} "
                       "unreadable and skipped.")
        out.append(f"Excluded records: {stats.get('meta_records', 0):,} "
                   f"harness-composed (isMeta), "
                   f"{stats.get('synthetic_records', 0):,} synthetic, "
                   f"{stats.get('noise_records', 0):,} harness notices, "
                   f"{stats.get('model_filtered', 0):,} other-model.")
    out.append("")

    if show_source_text:
        out.append("-" * 78)
        out.append("PHRASES — rate per 10k words, ranked by over-representation")
        out.append("-" * 78)
        out.append(f"{'ratio':>8} {speaker:>8} {'rate':>7} {other:>8} {'rate':>7}  phrase")
        for f in phrases:
            ratio_txt = f"{f.ratio:7.1f}x" if f.ratio else "     inf"
            out.append(f"{ratio_txt} {f.mine:8d} {_fmt_rate(f.mine_rate)} "
                       f"{f.theirs:8d} {_fmt_rate(f.theirs_rate)}  "
                       f"{redact(f.phrase, keep_domains)}")
        out.append("")
    else:
        out.append("Phrase and opener text withheld: those tables reproduce")
        out.append("transcript fragments. Re-run with --include-source-text.")
        out.append("")

    out.append("-" * 78)
    out.append("STRUCTURES — templated constructions, rate per 10k words")
    out.append("-" * 78)
    out.append(f"{'ratio':>8} {speaker:>8} {'rate':>7} {other:>8} {'rate':>7}  name")
    rows = []
    for name, pat, why in STRUCTURE_PATTERNS:
        m, t = structures[name]
        mr, tr = mine.rate(m), theirs.rate(max(t, BASELINE_FLOOR_COUNT))
        rows.append((mr / tr if tr else 0.0, name, m, mr, t, tr, why))
    rows.sort(key=lambda r: -r[0])
    for ratio, name, m, mr, t, tr, why in rows:
        out.append(f"{ratio:7.1f}x {m:8d} {_fmt_rate(mr)} {t:8d} "
                   f"{_fmt_rate(tr)}  {name}")
        out.append(f"{'':>40}  {why}")
    out.append("")

    if signatures:
        out.append("-" * 78)
        out.append("SIGNATURES — your configured phrases, rate per 10k words")
        out.append("-" * 78)
        out.append(f"{'baseline':>14} {'rate':>7} {'model':>7} {'rate':>7}  phrase")
        sig_rows = sorted(signatures.items(),
                          key=lambda kv: -kv[1][0])
        for name, (n_count, m_count) in sig_rows:
            n_rate = (theirs.rate(n_count) if speaker == "model"
                      else mine.rate(n_count))
            m_rate = (mine.rate(m_count) if speaker == "model"
                      else theirs.rate(m_count))
            out.append(f"{n_count:14d} {_fmt_rate(n_rate)} {m_count:7d} "
                       f"{_fmt_rate(m_rate)}  {name}")
        out.append("")

    out.append("-" * 78)
    out.append(f"OPENERS — first three words of a response, {speaker} only")
    out.append("-" * 78)
    total = sum(mine.openers.values()) or 1
    if show_source_text:
        for opener, count in mine.openers.most_common(opener_top):
            out.append(f"{count:6d}  {100.0 * count / total:5.1f}%  "
                       f"{redact(opener, keep_domains)}")
    distinct = len(mine.openers)
    out.append("")
    out.append(f"{distinct:,} distinct openers across {total:,} responses; "
               f"top {opener_top} cover "
               f"{100.0 * sum(c for _, c in mine.openers.most_common(opener_top)) / total:.1f}%.")
    return "\n".join(out)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Measure and report the model's over-used phrases and structures."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="",
                    help=f"config file (default: ./{config_mod.CONFIG_NAME} "
                         "if present, else built-in defaults)")
    ap.add_argument("--glob", default=None,
                    help="glob for session .jsonl transcripts; overrides "
                         "[corpus] transcripts")
    ap.add_argument("--baseline-from", default=None, metavar="GLOB",
                    help="use your own writing as the baseline instead of "
                         "your transcript turns (e.g. 'posts/**/*.md'); "
                         "overrides [baseline] samples")
    ap.add_argument("--days", type=int, default=None,
                    help="only turns from the last N days")
    ap.add_argument("--top", type=int, default=35,
                    help="phrase rows to show")
    ap.add_argument("--opener-top", type=int, default=20,
                    help="opener rows to show")
    ap.add_argument("--min-count", type=int, default=DEFAULT_MIN_COUNT,
                    help="minimum model occurrences for a phrase to rank")
    ap.add_argument("--include-source-text", action="store_true",
                    help="opt in to the phrase and opener tables, which "
                         "quote transcript fragments with no effective "
                         "redaction; applies to --json too")
    ap.add_argument("--model", default=None,
                    help="only count assistant records whose model id "
                         "contains this substring (e.g. fable); without it "
                         "figures blend every model in the corpus")
    ap.add_argument("--speaker", choices=("model", "baseline"), default="model",
                    help="whose tics to rank; the other side is the baseline")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    a = ap.parse_args(argv)

    try:
        cfg = config_mod.load(a.config)
    except config_mod.ConfigError as exc:
        print(f"config: {exc}", file=sys.stderr)
        return 2
    # An explicit --glob wins over the config file, which wins over the
    # built-in default. Stated here rather than in argparse defaults so the
    # precedence is one readable line instead of an interaction.
    pattern = a.glob or cfg.transcripts
    samples = a.baseline_from if a.baseline_from is not None else cfg.samples

    since = None
    if a.days:
        since = (dt.datetime.now(dt.timezone.utc)
                 - dt.timedelta(days=a.days))

    # Recursive like the samples glob, so a ``**`` pattern means what it says
    # instead of silently reading one level and reporting success.
    paths = sorted(p for p in glob.glob(os.path.expanduser(pattern),
                                        recursive=True) if os.path.isfile(p))
    if not paths:
        print(f"no transcripts matched {pattern}", file=sys.stderr)
        return 1

    model, baseline = Corpus("model"), Corpus("baseline")
    struct: Dict[str, List[int]] = {n: [0, 0] for n, _, _ in STRUCTURE_PATTERNS}
    # [baseline, model] -- your signatures are yours by definition, so your
    # count leads and the model's is shown beside it to expose bleed between
    # the two of you.
    sigs: Dict[str, List[int]] = {n: [0, 0] for n, _ in cfg.signatures}
    stats: Dict[str, int] = {}
    set_aside = 0
    for turn in read_turns(paths, since=since, stats=stats, model=a.model):
        is_model = turn.role == "assistant"
        # With a sample baseline the user turns are dropped entirely rather
        # than blended in. Mixing the two would build a baseline that is
        # neither register and silently weight it by whichever side happened
        # to be larger, which is a number nobody could interpret.
        if not is_model and samples:
            set_aside += 1
            continue
        (model if is_model else baseline).add(turn.text, opens=turn.opens)
        idx = 0 if is_model else 1
        for name, count in structure_counts(turn.text).items():
            struct[name][idx] += count
        for name, count in signature_counts(turn.text, cfg.signatures).items():
            sigs[name][1 if is_model else 0] += count
    if set_aside:
        # "turns" then counts only turns that reached a corpus, so the report
        # does not credit the transcript with turns main() discarded.
        stats["turns"] = stats.get("turns", 0) - set_aside
        stats["turns_set_aside"] = set_aside
    if samples:
        for para in read_samples(samples, stats=stats):
            baseline.add(para)
            for name, count in structure_counts(para).items():
                struct[name][1] += count
            for name, count in signature_counts(para, cfg.signatures).items():
                sigs[name][0] += count
        if not stats.get("sample_paragraphs"):
            print(f"no sample prose matched {samples}", file=sys.stderr)
            return 1

    # The renderer always compares ``mine`` against ``theirs``; which corpus is
    # which is the only thing --speaker changes.
    mine, theirs = ((model, baseline) if a.speaker == "model"
                    else (baseline, model))
    structures = ({k: (v[0], v[1]) for k, v in struct.items()}
                  if a.speaker == "model"
                  else {k: (v[1], v[0]) for k, v in struct.items()})
    signatures = {k: (v[0], v[1]) for k, v in sigs.items()}
    phrases = rank_phrases(mine, theirs, a.min_count, a.top)
    # Only meaningful for a sample baseline: transcript user turns are yours
    # by construction, so there is nothing there to be contaminated BY.
    warning = (contamination_warning(structures, mine, theirs)
               if samples else None)

    if a.json:
        # The source-text gate applies here identically: the keys that carry
        # transcript fragments are absent unless opted into, rather than
        # present-but-empty, so a consumer cannot mistake "withheld" for
        # "measured as zero".
        doc: Dict[str, object] = {
            "speaker": a.speaker,
            "baseline_source": samples or "transcripts",
            "baseline_contamination_warning": warning,
            "model_filter": a.model,
            "source_text_included": bool(a.include_source_text),
            "stats": stats,
            "model": {"turns": model.turns, "words": model.total_words,
                      "sentences": model.total_sentences,
                      "mean_sentence_len": model.mean_sentence_len()},
            "baseline": {"turns": baseline.turns, "words": baseline.total_words,
                       "sentences": baseline.total_sentences,
                       "mean_sentence_len": baseline.mean_sentence_len()},
            "structures": {k: {"model": v[0], "baseline": v[1]}
                           for k, v in struct.items()},
            "signatures": {k: {"baseline": v[0], "model": v[1]}
                           for k, v in sigs.items()},
        }
        if a.include_source_text:
            doc["phrases"] = [
                dict(dataclasses.asdict(f), phrase=redact(f.phrase, cfg.keep_domains))
                for f in phrases]
            doc["openers"] = [(redact(o, cfg.keep_domains), c) for o, c
                              in mine.openers.most_common(a.opener_top)]
        print(json.dumps(doc, indent=2))
        return 0

    if warning:
        print(warning, file=sys.stderr)
    print(render(mine, theirs, phrases, structures, a.opener_top,
                 a.include_source_text, signatures=signatures,
                 speaker=a.speaker, stats=stats,
                 keep_domains=cfg.keep_domains,
                 baseline_source=(f"your writing, {samples}" if samples else "")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
