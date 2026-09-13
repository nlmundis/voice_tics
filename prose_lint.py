#!/usr/bin/env python3
"""Lint a document for the model tics that measurement actually confirmed.

Every rule here earned its place by separating the model from a human author
on a matched corpus. Nothing is here because it sounds robotic.

WHAT IS DELIBERATELY ABSENT, AND WHY
    * **Any sentence-length or readability threshold.** The measurement that
      motivated this tool found mean sentence length does not separate the two
      writers at all: 13.89 words for the model against 13.85 for the author
      (45-day corpus). A length gate therefore lints an ACADEMIC REGISTER —
      long compound-complex sentences, passive voice, "However" — and deletes
      it. The tool that did exactly that to the reference author drew this
      summary from him: "we essentially made me sound simpler so I don't sound
      like AI." Sentence length is REPORTED here and can never fail a run.

    * **The slop lexicon** ("delve", "crucial", …): measured at 0.6x, meaning
      the AUTHOR used those words more often than the model did. A lexicon
      rule would correct the wrong writer.

    * **The chat-register family** ("let me check", "want me to"): the largest
      measured tics by far, and useless here, because none of them can appear
      in a document. A document linter carrying them would report clean runs
      as a property of the register rather than of the prose.

WHAT FAILS A RUN INSTEAD
    Structures, at the model's measured ratios (per 10k words, model over
    baseline, 45-day matched corpus). Which keys are in force is config;
    these are the shipped defaults:

    * ``method_defence`` — 10.0x. Defending the method instead of stating the
      finding. It reaches documents through model drafts rather than through
      the author's own writing, so flagging it corrects the right author.
    * ``not_x_but_y`` — 7.7x. The explicit "it's not A, it's B" reveal.
    * ``appositive_negation`` — 2.4x, a WARNING rather than an error: it is
      model-heavier but genuinely shared, so an occurrence is a prompt to
      look, not proof of a draft. ``--strict`` promotes warnings to failures.

    A lone em dash is available as a rule and ships OFF. It is one author's
    punctuation preference, and on the reference corpus the em-dash rates were
    identical (1.0x) — the most-cited AI tell in public advice, and it did not
    survive measurement. Turn it on only if your own ratio says something.

RE-MEASURE BEFORE YOU TRUST THE DEFAULTS
    Those ratios come from one author and one corpus. Run ``voice_tics.py``
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

# Why each DEFAULT tier key is in the tier it is in, as a rate ratio of
# model-over-baseline on the reference corpus (45 days, 2026-08-13). Kept
# beside the keys so a future re-tiering starts from the number that put them
# there rather than from taste, and so anyone can see the defaults are a
# measurement someone took and not a preference someone had.
#
# Which keys are actually in force is ``[lint] error`` and ``[lint] warn``,
# defaulting to config.DEFAULT_ERROR_TICS and DEFAULT_WARN_TICS. This table is
# documentation, not membership; ``tests/test_prose_lint.py`` pins that every
# default key appears here, so the two cannot drift apart silently.
TIC_PROVENANCE: Dict[str, str] = {
    "method_defence": "10.0x model/baseline, 2026-08-13",
    "not_x_but_y": "7.7x model/baseline, 2026-08-13",
    "appositive_negation":
        "2.4x model/baseline, 2026-08-13 — shared, model-heavier, so a "
        "warning rather than an error",
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


def _tic_findings(text: str,
                  tiers: Sequence[Tuple[str, str, Pattern[str], str]]
                  ) -> List[Finding]:
    """Pattern-tic findings: one pass over body prose, one over table cells.

    The two passes partition the document's LINES by one test on one
    rendering: ``emdash.is_table_row`` of the code-stripped line, the
    same rendering the cell walk classifies. Rows are blanked out of the
    INPUT before body scrubbing — not out of the scrubbed output, where a
    multi-line link opening on a row can relocate a following line's prose
    onto the masked row and silently drop its finding (found 2026-08-13,
    PR #46 verification; earlier, classifying the mask on the raw line
    while the walk saw the stripped one linted markup-prefixed rows twice).

    Cell prose then goes through the same ``scrub`` as body prose, so
    ``**isn't**`` in a cell is caught exactly as in a paragraph — a
    model-drafted assessment carries its findings in tables. Two accepted
    imperfections, both toward under-counting, the safe direction: a
    trailing-pipe row prefixed by an HTML tag is a row to neither pass
    (the tag survives ``strip_code``) yet still blanked by scrub's
    TABLE_ROW_RE once the tag is stripped, so its cells go unlinted; and
    a pipe-led lazy continuation line of a paragraph — which CommonMark
    renders as prose, there being no delimiter row — is treated as a
    table row, so a tic spanning the wrap, or cut in half by that line's
    own unescaped pipes, goes uncounted. Telling that line from a
    delimiter-less table needs paragraph context the line-based walk
    deliberately does not model (modelling it would change the em-dash
    hook's verdicts on chat-message tables).
    """
    raw_lines = text.split("\n")
    stripped_lines = emdash.strip_code(text).split("\n")
    # A masked row becomes a bare terminator, not an empty line: every
    # tier pattern's bridging classes exclude ".", and "." is not
    # whitespace, so no match can cross a masked row in either direction
    # — an empty line was regex-transparent, and patterns whose classes
    # admit newlines stitched the lines around a masked row into a
    # finding the rendered document does not contain (2026-08-13, PR #46
    # verification, round seven). "." can neither start nor complete any
    # pattern, so it can appear in no excerpt.
    body_input = "\n".join(
        "." if emdash.is_table_row(stripped) else raw
        for raw, stripped in zip(raw_lines, stripped_lines))
    body = voice_tics.scrub(body_input, keep_lines=True)
    out: List[Finding] = []
    for key, tier, rx, why in tiers:
        for m in rx.finditer(body):
            out.append(Finding(
                rule=key,
                tier=tier,
                line=_line_of(m.start(), body),
                excerpt=_excerpt(m.group(0)),
                why=why,
            ))
    for row_line, cell, is_cell in emdash.numbered_blocks(
            emdash.strip_code(text)):
        if not is_cell:
            continue
        prose = voice_tics.scrub(cell)
        for key, tier, rx, why in tiers:
            for m in rx.finditer(prose):
                out.append(Finding(
                    rule=key,
                    tier=tier,
                    line=row_line,
                    excerpt=_excerpt(m.group(0)),
                    why=why,
                ))
    return out


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
    tiers = tier_patterns(cfg.error_tics, cfg.warn_tics)
    found = _tic_findings(text, tiers)
    if cfg.emdash_enabled:
        found += _emdash_findings(text, cfg.emdash_exempt)
    found.sort(key=lambda f: (f.line is None, f.line or 0, f.rule))
    return found, _stats(text)


INFO_NOTE = (
    "reported, never a failure: mean sentence length does not separate "
    "model from author (13.89 vs 13.85, 2026-08-13), so a threshold here "
    "would lint the author's register, not the machine's"
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
                    "(no sentence-length threshold, by measurement)")
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
