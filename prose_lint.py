#!/usr/bin/env python3
"""Lint a document for model tics chosen by measuring against a matched baseline.

Every rule here was chosen because it separated the model from a human author
on a matched corpus, not because it sounds robotic. Those measurements have
since been withdrawn (see the README: the transcript baseline counted some of
the model's writing as the author's), so the defaults are a starting point.
Measure your own before you rely on them.

WHAT IS DELIBERATELY ABSENT, AND WHY
    * **Any sentence-length or readability threshold.** A length threshold
      measures register, not authorship. This linter runs on documents, and
      a chat baseline says nothing about how long the author's document
      sentences are. A length
      gate lints an ACADEMIC REGISTER — long compound-complex sentences,
      passive voice, "However" — and deletes it. The tool that did exactly
      that to the reference author drew this summary from him: "we
      essentially made me sound simpler so I don't sound like AI." Sentence
      length is REPORTED here and can never fail a run.

    * **The slop lexicon** ("delve", "crucial", …): a list of words somebody
      decided sound robotic. For an author whose register overlaps it, a
      lexicon rule flags the author as readily as the model, and only a
      matched baseline can say which of the two you are.

    * **The chat-register family** ("let me check", "want me to"): useless
      here, because none of them can appear in a document. A document linter
      carrying them would report clean runs as a property of the register
      rather than of the prose.

WHAT FAILS A RUN INSTEAD
    Structures. ``TIC_PROVENANCE`` below records why each default is in its
    tier. Which keys are in force is config; these are the shipped defaults:

    * ``method_defence``: defending the method instead of stating the
      finding.
    * ``not_x_but_y``: the explicit "it's not A, it's B" reveal.
    * ``appositive_negation``: a WARNING rather than an error, so an
      occurrence is a prompt to look, not proof of a draft. ``--strict``
      promotes warnings to failures.

    Each was put in its tier on ratios that have since been withdrawn, so
    the tiers are a starting point, not a finding about who writes what.

    A lone em dash is available as a rule and ships OFF. It is one author's
    punctuation preference about LONE dashes, not a measurement of a rate.
    Turn it on if your own measurement and your own style both say so.

    Every rule here, not only a length threshold, was measured against a chat
    baseline and runs on documents. ``--baseline-from`` in voice_tics.py
    measures against documents instead, which is the better test of a rule
    you mean to enforce on documents.

RE-MEASURE BEFORE YOU TRUST THE DEFAULTS
    The defaults were chosen on one author's corpus. Run ``voice_tics.py``
    against your own transcripts or your own writing, read the STRUCTURES
    table, and set ``[lint] error`` from what separates for YOU. Shipping this
    file's defaults unexamined is the same mistake as shipping a wordlist,
    only with better provenance.

SINGLE SOURCE OF TRUTH
    Patterns come from ``voice_tics.STRUCTURE_PATTERNS`` by key, so a
    sharpened detector there sharpens this linter with no change here. A key
    that no longer exists raises before any file is read, rather than
    silently narrowing the lint partway through a batch.

Exit codes: 0 clean or warnings only; 1 any error, or any warning under
``--strict``; 2 usage or config error.

Usage:
    python3 prose_lint.py draft.md
    python3 prose_lint.py --strict draft.md other.md
    cat draft.md | python3 prose_lint.py -
    python3 prose_lint.py --json draft.md
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import sys
from typing import Dict, List, Optional, Pattern, Sequence, Tuple

import vt_config as config_mod
import emdash
import voice_tics

# Why each DEFAULT tier key is in the tier it is in. The tiers were chosen on
# ratios measured against the reference author's transcripts, and those ratios
# are withdrawn (README): the baseline counted some of the model's writing as
# the author's. So no figure is kept here. A ratio in this table would have to
# be a dated measurement file committed under docs, which tests/test_repo.py
# recomputes; until a baseline is trusted, there is none to cite.
#
# Which keys are actually in force is ``[lint] error`` and ``[lint] warn``,
# defaulting to config.DEFAULT_ERROR_TICS and DEFAULT_WARN_TICS. This table is
# documentation, not membership; ``tests/test_prose_lint.py`` pins that every
# default key appears here, so the two cannot drift apart silently.
TIC_PROVENANCE: Dict[str, str] = {
    "method_defence":
        "error: defending the method instead of stating the finding; chosen "
        "on withdrawn ratios, so re-measure before relying on the tier",
    "not_x_but_y":
        "error: the explicit reveal template; chosen on withdrawn ratios "
        "that rested on one author use or none, so re-measure before "
        "relying on the tier",
    "appositive_negation":
        "warning: a prompt to look rather than an error; chosen on withdrawn "
        "ratios, so re-measure before relying on the tier",
}

EXCERPT_CAP = 60


def _compiled(key: str) -> Tuple[Pattern[str], str]:
    """The compiled voice_tics structure pattern for ``key``, plus its label.

    Raises:
        KeyError: ``voice_tics.STRUCTURE_PATTERNS`` no longer defines the
            key. Loud on purpose: a renamed or retired pattern must force
            this linter to be re-aimed, never let it lint with a rule
            silently missing.
    """
    for name, pattern, why in voice_tics.STRUCTURE_PATTERNS:
        if name == key:
            # IGNORECASE matches voice_tics.structure_counts: the ratios
            # that put a key in a tier were measured case-insensitively,
            # so the lint must fire on the same set of strings.
            return re.compile(pattern, re.IGNORECASE), why
    raise KeyError(
        f"voice_tics.STRUCTURE_PATTERNS no longer defines {key!r}; "
        "re-aim prose_lint.py rather than letting the rule vanish silently"
    )


@dataclasses.dataclass
class Finding:
    """One lint hit: which rule, where, what matched, and why it is a rule."""

    rule: str
    tier: str  # "error" or "warning"
    line: Optional[int]  # 1-based; None when the block could not be located
    excerpt: str
    why: str


def _excerpt(matched: str) -> str:
    """Whitespace-normalised match text, capped for one-line display."""
    flat = " ".join(matched.split())
    return flat if len(flat) <= EXCERPT_CAP else flat[: EXCERPT_CAP - 1] + "…"


def _line_of(offset: int, text: str) -> int:
    """1-based line number of a character offset in line-preserved text."""
    return text.count("\n", 0, offset) + 1


def tier_patterns(error_tics: Sequence[str] = (),
                  warn_tics: Sequence[str] = (),
                  ) -> Tuple[Tuple[str, str, Pattern[str], str], ...]:
    """Compile the configured tic keys into (key, tier, pattern, why) rows.

    Built once per run, before any file is read, so a key that no longer
    exists upstream fails the WHOLE run rather than the first lint call
    partway through a batch. That is the loud-failure contract: a rule that
    quietly stopped being checked is the worst outcome for a linter, worse
    than a crash, because the build goes green.

    Args:
        error_tics: STRUCTURE_PATTERNS keys whose matches fail a run.
        warn_tics: Keys whose matches warn, and fail only under --strict.

    Raises:
        KeyError: A key ``voice_tics.STRUCTURE_PATTERNS`` no longer defines.
    """
    return tuple(
        (key, tier) + _compiled(key)
        for tier, keys in (("error", error_tics), ("warning", warn_tics))
        for key in keys
    )


Row = Tuple[str, str, Pattern[str], str]  # rule, tier, pattern, why


def _pattern_findings(text: str, rows: Sequence[Row]) -> List[Finding]:
    """Findings for every (rule, tier, pattern, why) row, measured tics and
    banned phrases alike: one pass over body prose, one over table cells.

    ONE ENGINE FOR BOTH KINDS OF RULE
        Banned phrases once had their own pass, a scrub of the whole document,
        and scrub blanks every complete table row, so a banned phrase in a
        well-formed table was never reported while a measured tic in the same
        cell was. A model-drafted assessment carries its findings in tables.
        Both kinds of rule now go through here; they stay apart in their rule
        names and their config, not in how a document is read.

    THE PARTITION
        The two passes partition the document's LINES by one test on one
        rendering: ``emdash.is_table_row`` of the code-stripped line, the same
        rendering the cell walk classifies. Rows are blanked out of the INPUT
        before body scrubbing, not out of the scrubbed output, where a
        multi-line link opening on a row can relocate a following line's prose
        onto the masked row and silently drop its finding (found 2026-08-13,
        PR #46 verification; earlier, classifying the mask on the raw line
        while the walk saw the stripped one linted markup-prefixed rows
        twice).

        A masked row becomes a bare "." rather than an empty line. Every tier
        pattern's bridging classes exclude ".", and "." is not whitespace, so
        no match can CROSS a masked row: an empty line was regex-transparent,
        and patterns whose classes admit newlines stitched the lines around a
        masked row into a finding the rendered document does not contain
        (2026-08-13, round seven). The "." is not inert, though, and this
        comment once said it was: a pattern that opens at a sentence
        terminator, ``rhetorical_self_question``, can START on it. That match
        is right, since a table does end the sentence before the prose after
        it, but it was reported on the row's line with the "." in its excerpt.
        The lead-in rule below is what fixes that.

    WHERE A FINDING IS REPORTED
        A pattern may begin with the terminator and whitespace that end the
        PREVIOUS sentence (``rhetorical_self_question`` does). The line and
        excerpt start after that lead-in, so the finding lands on the line
        holding the flagged words, not the line before.

    Cell prose goes through the same ``scrub`` as body prose, so
    ``**isn't**`` in a cell is caught exactly as in a paragraph. Two accepted
    imperfections, both toward under-counting, the safe direction: a
    trailing-pipe row prefixed by an HTML tag is a row to neither pass (the
    tag survives ``strip_code``) yet still blanked by scrub's TABLE_ROW_RE
    once the tag is stripped, so its cells go unlinted; and a pipe-led lazy
    continuation line of a paragraph, which CommonMark renders as prose,
    there being no delimiter row, is treated as a table row, so a tic
    spanning the wrap, or cut in half by that line's own unescaped pipes,
    goes uncounted. Telling that line from a delimiter-less table needs
    paragraph context the line-based walk deliberately does not model
    (modelling it would change the em-dash hook's verdicts on chat-message
    tables).
    """
    raw_lines = text.split("\n")
    stripped = emdash.strip_code(text)
    body_input = "\n".join(
        "." if emdash.is_table_row(plain) else raw
        for raw, plain in zip(raw_lines, stripped.split("\n")))
    body = voice_tics.scrub(body_input, keep_lines=True)
    out: List[Finding] = []
    for rule, tier, rx, why in rows:
        for m in rx.finditer(body):
            lead = _lead_in(m.group(0))
            out.append(Finding(
                rule=rule,
                tier=tier,
                line=_line_of(m.start() + lead, body),
                excerpt=_excerpt(m.group(0)[lead:]),
                why=why,
            ))
    for row_line, cell, is_cell in emdash.numbered_blocks(stripped):
        if not is_cell:
            continue
        prose = voice_tics.scrub(cell)
        for rule, tier, rx, why in rows:
            for m in rx.finditer(prose):
                lead = _lead_in(m.group(0))
                out.append(Finding(
                    rule=rule,
                    tier=tier,
                    line=row_line,
                    excerpt=_excerpt(m.group(0)[lead:]),
                    why=why,
                ))
    return out


def _lead_in(matched: str) -> int:
    """Length of the sentence terminators and whitespace a match opens with.

    Zero when that would be the whole match, so a rule made of punctuation
    still reports what it matched.
    """
    lead = len(matched) - len(matched.lstrip(".!?" + " \t\n\r\f\v\u2029"))
    return lead if lead < len(matched) else 0


def _emdash_findings(text: str, exempt: str = "") -> List[Finding]:
    """Lone-em-dash findings, each on the line ``emdash`` says it is on.

    The walker carries line numbers natively (``lone_dash_lines``); the
    earlier reverse-location by substring search mis-attributed a violation
    to a preceding line that merely contained the same text.
    """
    return [
        Finding(
            rule="lone_em_dash",
            tier="error",
            line=line_no,
            excerpt=_excerpt(blk),
            why="em dash outside a matched pair — use a colon, comma, "
                "semicolon, or a new sentence",
        )
        for line_no, blk, _pos in emdash.lone_dash_lines(text, exempt)
    ]


def banned_pattern(phrase: str) -> Pattern[str]:
    """A literal phrase compiled so hyphens and spaces are interchangeable.

    Someone who bans "load bearing" means "load-bearing" too; the hyphen is a
    typesetting choice, not a different phrase. Each word is escaped, so a
    phrase containing regex punctuation is still matched literally.

    Word boundaries are applied only where the phrase actually starts and ends
    with a word character. Without that check, banning "C++" would compile to
    a boundary that can never match after the final "+".
    """
    words = [re.escape(w) for w in config_mod.banned_words(phrase)]
    body = r"[-\s]+".join(words)
    lead = r"\b" if re.match(r"\w", phrase.strip()) else ""
    tail = r"\b" if re.search(r"\w$", phrase.strip()) else ""
    return re.compile(lead + body + tail, re.IGNORECASE)


def banned_rows(banned: Sequence[Tuple[str, str, str]]) -> List[Row]:
    """The reader's own banned phrases as rows for ``_pattern_findings``.

    Separate rules from the measured tics on purpose. These carry no ratio and
    claim nothing about how a model writes; they are one person saying they do
    not want to read a phrase, which is a perfectly good reason and a
    different kind of claim. Keeping them apart, by rule name and by config
    key, is what stops a preference from being read later as evidence.
    """
    rows: List[Row] = []
    for phrase, instead, tier in banned:
        why = f"banned phrase; instead: {instead}" if instead else "banned phrase"
        rule = f"banned:{' '.join(config_mod.banned_words(phrase.lower()))}"
        rows.append((rule, tier, banned_pattern(phrase), why))
    return rows


def _stats(text: str) -> Dict[str, float]:
    """Register numbers for the info line: reported, never a failure."""
    prose = voice_tics.scrub(text)
    n_words = len(voice_tics.words(prose))
    sents = voice_tics.sentences(prose)
    mean = (n_words / len(sents)) if sents else 0.0
    return {
        "words": float(n_words),
        "sentences": float(len(sents)),
        "mean_words_per_sentence": round(mean, 2),
    }


def lint_text(text: str, cfg: Optional[config_mod.Config] = None
              ) -> Tuple[List[Finding], Dict[str, float]]:
    """All findings for one document, in line order, plus its register stats.

    Args:
        text: The document. Markdown structure is understood; plain text works.
        cfg: The installation's config. Defaults to the built-in tiers with
            the em-dash rule off, which is what an unconfigured run gets.
    """
    cfg = cfg or config_mod.Config.default()
    rows = list(tier_patterns(cfg.error_tics, cfg.warn_tics))
    found = _pattern_findings(text, rows + banned_rows(cfg.banned))
    if cfg.emdash_enabled:
        found += _emdash_findings(text, cfg.emdash_exempt)
    found.sort(key=lambda f: (f.line is None, f.line or 0, f.rule))
    return found, _stats(text)


INFO_NOTE = (
    "reported, never a failure: a sentence-length threshold measures "
    "register, not authorship, so it would lint the author's register, not "
    "the machine's (see the README)"
)


def _render(path: str, found: Sequence[Finding],
            stats: Dict[str, float]) -> List[str]:
    """Human-readable report lines for one file."""
    out = []
    for f in found:
        where = f"{path}:{f.line}" if f.line is not None else f"{path}:?"
        out.append(f'{where}: {f.tier} {f.rule}: "{f.excerpt}" — {f.why}')
    out.append(
        f"{path}: info: {stats['mean_words_per_sentence']} words/sentence "
        f"over {int(stats['sentences'])} sentences — {INFO_NOTE}"
    )
    return out


def _read(path: str) -> str:
    """Contents of ``path``, with ``-`` meaning stdin."""
    if path == "-":
        return sys.stdin.read()
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point; returns the process exit code."""
    ap = argparse.ArgumentParser(
        description="Lint a document for measured model tics "
                    "(no sentence-length threshold: length measures register)")
    ap.add_argument("files", nargs="+", metavar="FILE",
                    help="markdown/text files to lint, or - for stdin")
    ap.add_argument("--strict", action="store_true",
                    help="warnings also fail the run")
    ap.add_argument("--json", action="store_true", dest="as_json",
                    help="machine-readable output")
    ap.add_argument("--config", default="",
                    help=f"config file (default: ./{config_mod.CONFIG_NAME} "
                         "if present, else built-in defaults)")
    args = ap.parse_args(argv)

    try:
        cfg = config_mod.load(args.config)
    except config_mod.ConfigError as exc:
        print(f"prose_lint: config: {exc}", file=sys.stderr)
        return 2
    note = config_mod.unread_parent_config(args.config)
    if note:
        print(f"prose_lint: {note}", file=sys.stderr)
    try:
        # Compiled once, before any file is read, so an unknown tic key fails
        # the run instead of the first file that happens to reach it.
        tier_patterns(cfg.error_tics, cfg.warn_tics)
    except KeyError as exc:
        print(f"prose_lint: {exc.args[0]}", file=sys.stderr)
        return 2

    reports = []
    worst = 0
    for path in args.files:
        try:
            text = _read(path)
        except (OSError, UnicodeDecodeError) as exc:
            print(f"prose_lint: cannot read {path}: {exc}", file=sys.stderr)
            return 2
        found, stats = lint_text(text, cfg)
        tiers = {f.tier for f in found}
        if "error" in tiers or (args.strict and "warning" in tiers):
            worst = 1
        reports.append((path, found, stats))

    if args.as_json:
        print(json.dumps({
            "files": [
                {"path": p, "stats": s,
                 "findings": [dataclasses.asdict(f) for f in fs]}
                for p, fs, s in reports
            ],
            "exit": worst,
        }, indent=2))
    else:
        for p, fs, s in reports:
            for line in _render(p, fs, s):
                print(line)
    return worst


if __name__ == "__main__":
    sys.exit(main())
