"""Everything about this installation that is not about the method.

WHAT BELONGS IN CONFIG AND WHAT BELONGS IN CODE
    The line is whose fact it is.

    The structure detectors in ``voice_tics.STRUCTURE_PATTERNS`` are the
    tool's substance — claims about how the model writes, each with its
    measured ratio recorded beside it — so they live in code, in version
    control, where a change to one is a change somebody can review.

    Your signature phrases, your transcript location, the domains you do not
    need redacted, and which detectors are allowed to fail your build are
    facts about YOU. They live here. The private original this was extracted
    from had all four hardcoded, which is exactly why it could not be
    published without this file existing first.

IT RUNS WITH NO CONFIG AT ALL
    Every field has a default and ``voice_tics.toml`` is optional. The
    defaults are the conservative reading: the standard Claude Code transcript
    location, no signature phrases, no allowlisted domains, and the em-dash
    rule off. A tool that demands a config file before it will say anything is
    a tool nobody evaluates.

UNKNOWN KEYS ARE AN ERROR
    A misspelled key that is silently ignored is a setting the user believes
    is in force and is not. For a linter that can fail a build, and for a
    redaction allowlist, that failure is silent in the direction that hurts.
    Every table is checked against its known keys and an unrecognised one
    raises ``ConfigError`` naming the nearest match.
"""

from __future__ import annotations

import difflib
import os
import re
from typing import Any, Dict, List, Mapping, Sequence, Tuple

CONFIG_NAME = "voice_tics.toml"


def _toml_module() -> Any:
    """The TOML parser module, imported only when a config file exists.

    WHY THE IMPORT IS NOT AT MODULE SCOPE
        ``tomllib`` arrived in Python 3.11. At module scope, a missing parser
        on 3.9 or 3.10 killed both entry points before argparse ran, config
        file or not, which made the README's "stdlib only, Python 3.9+" false
        for the zero-config case the tool is built around. Importing here
        means only a run that actually has a config file needs a parser, and
        that run gets a ConfigError saying what to install instead of a
        traceback.

    Raises:
        ConfigError: Neither ``tomllib`` nor ``tomli`` can be imported.
    """
    try:
        import tomllib
        return tomllib
    except ImportError:  # Python 3.10 and earlier
        pass
    try:
        import tomli
        return tomli
    except ImportError:
        raise ConfigError(
            "reading a config file on Python 3.10 or earlier needs tomli: "
            "pip install tomli") from None


# The standard Claude Code transcript location. Every session writes a .jsonl
# here holding both sides of the conversation, which is what makes a matched
# baseline possible without asking anyone to collect anything.
DEFAULT_TRANSCRIPTS = "~/.claude/projects/*/*.jsonl"

# Which detectors may fail a lint run, by key into STRUCTURE_PATTERNS. These
# two are the defaults because they are the ones whose measured separation was
# large enough to act on. The ratios themselves live in the README table and
# ``prose_lint.TIC_PROVENANCE``, which tests/test_repo.py holds cell for cell,
# and are deliberately not repeated here: an unchecked copy of a measurement is
# how one of them goes stale (this comment once carried method_defence's figure
# for the detector before it was sharpened, after every other copy had moved
# on). They are still only a
# default: the whole argument of this tool is that you should measure your own.
DEFAULT_ERROR_TICS: Tuple[str, ...] = ("method_defence", "not_x_but_y")

# Model-heavier but genuinely shared, so an occurrence is a prompt to look
# rather than proof of a draft. Warnings never fail a run without --strict.
DEFAULT_WARN_TICS: Tuple[str, ...] = ("appositive_negation",)

TOP_KEYS = frozenset({"corpus", "baseline", "lint", "output"})
CORPUS_KEYS = frozenset({"transcripts"})
BASELINE_KEYS = frozenset({"signatures", "samples"})
LINT_KEYS = frozenset({"error", "warn", "emdash", "banned"})
BANNED_KEYS = frozenset({"phrase", "instead", "tier"})
EMDASH_KEYS = frozenset({"enabled", "exempt"})
OUTPUT_KEYS = frozenset({"keep_domains"})
SIGNATURE_KEYS = frozenset({"name", "pattern"})


class ConfigError(Exception):
    """A config file that cannot be trusted to mean what it says.

    Raised for unreadable files, malformed TOML, unknown keys, wrong types and
    uncompilable patterns. Never raised for a MISSING file, which is the
    supported zero-config case and yields ``Config.default()`` instead.
    """


def _reject_unknown(raw: Mapping[str, Any], known: frozenset, where: str) -> None:
    """Raise on the first key ``where`` does not define, suggesting a fix."""
    for key in raw:
        if key not in known:
            near = difflib.get_close_matches(key, sorted(known), n=1)
            hint = f"; did you mean {near[0]!r}?" if near else ""
            raise ConfigError(f"unknown key {key!r} in {where}{hint}")


def _as_str_list(value: Any, where: str) -> List[str]:
    """A TOML array of strings, or a ConfigError naming what arrived instead."""
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ConfigError(f"{where} must be an array of strings")
    return list(value)


def _parse_tiers(lint: Mapping[str, Any]) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """``[lint] error`` and ``[lint] warn``, each key in at most one place.

    A key listed in both tiers produced two findings for every match, one of
    each tier, which no one can have meant: it is what moving a key between
    tiers and forgetting the old line looks like. A key listed twice in one
    tier doubled every finding. Both are rejected rather than resolved,
    because picking a tier on the user's behalf is a setting they did not
    choose.
    """
    error = (tuple(_as_str_list(lint["error"], "[lint] error"))
             if "error" in lint else DEFAULT_ERROR_TICS)
    warn = (tuple(_as_str_list(lint["warn"], "[lint] warn"))
            if "warn" in lint else DEFAULT_WARN_TICS)
    for name, keys in (("[lint] error", error), ("[lint] warn", warn)):
        repeated = sorted({k for k in keys if keys.count(k) > 1})
        if repeated:
            raise ConfigError(f"{name} lists {', '.join(repeated)} more than once")
    both = sorted(set(error) & set(warn))
    if both:
        # Name the default when it is the default that collides: someone who
        # writes only `warn = ["method_defence"]` never typed an error list.
        error_name = ("[lint] error" if "error" in lint
                      else "[lint] error (its default, since error is unset)")
        warn_name = ("[lint] warn" if "warn" in lint
                     else "[lint] warn (its default, since warn is unset)")
        raise ConfigError(
            f"{', '.join(both)} is in both {error_name} and {warn_name}; "
            "a rule has one tier")
    return error, warn


def _parse_keep_domains(value: Any) -> Tuple[str, ...]:
    """Validate ``[output] keep_domains`` into bare, lowercased domains.

    An entry spelled ``.example.com``, ``example.com.`` or with a stray space
    matched nothing at all: the allowlist compares against the parsed domain,
    so the entry was inert and every address it was meant to keep was
    redacted anyway, with no sign the setting was not in force. Rejected here
    rather than normalised, because a silently rewritten setting is the same
    class of surprise as a silently ignored one.

    Raises:
        ConfigError: A non-string, an empty entry, or one carrying whitespace,
            an ``@``, or a leading or trailing dot.
    """
    out: List[str] = []
    for entry in _as_str_list(value, "[output] keep_domains"):
        bare = entry.strip().strip(".")
        if (not bare or bare != entry or "@" in entry
                or any(ch.isspace() for ch in entry)):
            hint = f'; write it as "{bare}", which covers its subdomains too' \
                if bare and "@" not in bare and not any(
                    ch.isspace() for ch in bare) else ""
            raise ConfigError(
                f"[output] keep_domains entry {entry!r} is not a bare "
                f"domain{hint}")
        out.append(entry.lower())
    return tuple(out)


def banned_words(phrase: str) -> List[str]:
    """The words of a banned phrase, split where the matcher joins them.

    THE single normalisation of a banned phrase. ``prose_lint.banned_pattern``
    joins these words with "any run of spaces or hyphens", so the duplicate
    check and the finding's rule name must split on exactly that too. When
    the duplicate check folded case and spaces but not hyphens,
    ``load-bearing`` and ``load bearing`` compiled to one regex and were both
    accepted, and every hit was reported twice at two tiers.
    """
    return [w for w in re.split(r"[-\s]+", phrase.strip()) if w]


def _parse_banned(value: Any) -> Tuple[Tuple[str, str, str], ...]:
    """Validate ``[lint] banned`` into (phrase, instead, tier) triples.

    WHY THESE ARE LITERAL PHRASES AND NOT REGEXES
        ``[baseline] signatures`` takes regexes because it is a measurement
        instrument, aimed by someone who has already decided what to count.
        This one is aimed by someone who is simply sick of a phrase, and
        making them escape it is a way to be wrong quietly: ``C++`` compiles
        to a repetition error and ``(beta)`` to an empty group that matches
        everywhere.

        So the phrase is taken literally, matched case-insensitively, and the
        gaps between its words match any run of spaces or hyphens. "load
        bearing" therefore also catches "load-bearing", which is the whole
        reason someone typed it.

    Raises:
        ConfigError: A malformed entry, a phrase with no words once spaces
            and hyphens are removed, a tier that is not "error" or
            "warning", or a phrase that repeats an earlier one.
    """
    if not isinstance(value, list):
        raise ConfigError("[lint] banned must be an array of tables")
    out: List[Tuple[str, str, str]] = []
    seen: Dict[str, int] = {}
    for i, entry in enumerate(value):
        where = f"[lint] banned entry {i + 1}"
        if not isinstance(entry, dict):
            raise ConfigError(f"{where} must be a table with phrase and instead")
        _reject_unknown(entry, BANNED_KEYS, where)
        phrase = entry.get("phrase")
        if not isinstance(phrase, str) or not phrase.strip():
            raise ConfigError(f"{where} needs a non-empty string phrase")
        if not banned_words(phrase):
            # "--" passed the emptiness check and compiled to an empty
            # pattern, which reported an error at every character offset.
            raise ConfigError(
                f"{where} ({phrase!r}) has nothing to match: spaces and "
                "hyphens are the gaps between words, not words")
        instead = entry.get("instead", "")
        if not isinstance(instead, str):
            raise ConfigError(f"{where} ({phrase}) instead must be a string")
        tier = entry.get("tier", "error")
        if tier not in ("error", "warning"):
            raise ConfigError(
                f"{where} ({phrase}) tier must be \"error\" or \"warning\", "
                f"not {tier!r}")
        key = " ".join(banned_words(phrase.lower()))
        if key in seen:
            raise ConfigError(
                f"{where} repeats the phrase {phrase!r} from entry {seen[key]}")
        seen[key] = i + 1
        out.append((phrase, instead, tier))
    return tuple(out)


def _parse_signatures(value: Any) -> Tuple[Tuple[str, str], ...]:
    """Validate ``[baseline] signatures`` into (name, pattern) pairs.

    Each entry is a table with ``name`` and ``pattern``. The pattern is
    compiled here rather than at first use so a typo fails at startup, next to
    the line that caused it, instead of halfway through a scan.
    """
    if not isinstance(value, list):
        raise ConfigError("[baseline] signatures must be an array of tables")
    out: List[Tuple[str, str]] = []
    seen: Dict[str, int] = {}
    for i, entry in enumerate(value):
        where = f"[baseline] signatures entry {i + 1}"
        if not isinstance(entry, dict):
            raise ConfigError(f"{where} must be a table with name and pattern")
        _reject_unknown(entry, SIGNATURE_KEYS, where)
        name, pattern = entry.get("name"), entry.get("pattern")
        if not isinstance(name, str) or not name.strip():
            raise ConfigError(f"{where} needs a non-empty string name")
        if not isinstance(pattern, str) or not pattern:
            raise ConfigError(f"{where} ({name}) needs a non-empty string pattern")
        if name in seen:
            raise ConfigError(
                f"{where} reuses the name {name!r} from entry {seen[name]}; "
                "counts are keyed by name, so the later entry would silently "
                "replace the earlier one")
        try:
            re.compile(pattern)
        except re.error as exc:
            raise ConfigError(f"{where} ({name}) has an invalid pattern: {exc}") from None
        seen[name] = i + 1
        out.append((name, pattern))
    return tuple(out)


class Config:
    """One installation's answers, with a default for every field.

    Attributes:
        transcripts: Glob for the session files to read, ``~`` expanded.
        signatures: The baseline author's own phrases, checked by name. Empty
            by default; see ``voice_tics.toml.example`` for why you would fill
            it in and why discovery cannot find these for you.
        samples: Optional glob of your own writing to use as the baseline
            instead of your transcript turns. Empty means use the transcripts.
            See ``voice_tics.read_samples`` for what each baseline costs.
        error_tics: Structure keys that fail a lint run.
        warn_tics: Structure keys that warn, and fail only under ``--strict``.
        emdash_enabled: Whether a lone em dash is an error. Off by default:
            it is one author's punctuation rule, not a measured model tic.
        emdash_exempt: Optional regex whose matches are blanked before em
            dashes are counted, for a citation format that uses a lone dash
            as a separator by construction.
        banned: (phrase, instead, tier) triples from ``[lint] banned``.
            Your own house style, matched literally. Empty by default.
        keep_domains: Domains left in cleartext by ``redact``.
    """

    def __init__(self,
                 transcripts: str = DEFAULT_TRANSCRIPTS,
                 samples: str = "",
                 signatures: Sequence[Tuple[str, str]] = (),
                 error_tics: Sequence[str] = DEFAULT_ERROR_TICS,
                 warn_tics: Sequence[str] = DEFAULT_WARN_TICS,
                 emdash_enabled: bool = False,
                 emdash_exempt: str = "",
                 banned: Sequence[Tuple[str, str, str]] = (),
                 keep_domains: Sequence[str] = ()) -> None:
        self.transcripts = os.path.expanduser(transcripts)
        self.samples = os.path.expanduser(samples) if samples else ""
        self.signatures = tuple(signatures)
        self.error_tics = tuple(error_tics)
        self.warn_tics = tuple(warn_tics)
        self.emdash_enabled = emdash_enabled
        self.emdash_exempt = emdash_exempt
        self.banned = tuple(banned)
        self.keep_domains = tuple(keep_domains)

    @classmethod
    def default(cls) -> "Config":
        """The zero-config installation: standard paths, no personal facts."""
        return cls()

    @classmethod
    def parse(cls, raw: Mapping[str, Any]) -> "Config":
        """Build a Config from decoded TOML; every problem is a ConfigError."""
        _reject_unknown(raw, TOP_KEYS, "the config file")
        corpus = raw.get("corpus", {})
        baseline = raw.get("baseline", {})
        lint = raw.get("lint", {})
        output = raw.get("output", {})
        for table, keys, name in ((corpus, CORPUS_KEYS, "[corpus]"),
                                  (baseline, BASELINE_KEYS, "[baseline]"),
                                  (lint, LINT_KEYS, "[lint]"),
                                  (output, OUTPUT_KEYS, "[output]")):
            if not isinstance(table, dict):
                raise ConfigError(f"{name} must be a table")
            _reject_unknown(table, keys, name)

        emdash = lint.get("emdash", {})
        if not isinstance(emdash, dict):
            raise ConfigError("[lint.emdash] must be a table")
        _reject_unknown(emdash, EMDASH_KEYS, "[lint.emdash]")
        enabled = emdash.get("enabled", False)
        if not isinstance(enabled, bool):
            raise ConfigError("[lint.emdash] enabled must be true or false")
        exempt = emdash.get("exempt", "")
        if not isinstance(exempt, str):
            raise ConfigError("[lint.emdash] exempt must be a string")
        if exempt:
            try:
                re.compile(exempt)
            except re.error as exc:
                raise ConfigError(f"[lint.emdash] exempt is not a valid regex: {exc}") from None

        transcripts = corpus.get("transcripts", DEFAULT_TRANSCRIPTS)
        if not isinstance(transcripts, str) or not transcripts:
            raise ConfigError("[corpus] transcripts must be a non-empty string")

        samples = baseline.get("samples", "")
        if not isinstance(samples, str):
            raise ConfigError("[baseline] samples must be a string glob")

        error_tics, warn_tics = _parse_tiers(lint)
        return cls(
            transcripts=transcripts,
            samples=samples,
            signatures=_parse_signatures(baseline["signatures"])
            if "signatures" in baseline else (),
            error_tics=error_tics,
            warn_tics=warn_tics,
            emdash_enabled=enabled,
            emdash_exempt=exempt,
            banned=_parse_banned(lint["banned"]) if "banned" in lint else (),
            keep_domains=_parse_keep_domains(output["keep_domains"])
            if "keep_domains" in output else (),
        )


def load(path: str = "") -> Config:
    """The config at ``path``, or ``./voice_tics.toml``, or defaults.

    Only the working directory is searched, never its parents: which file
    governs a run should not depend on where above you someone left one.
    ``unread_parent_config`` exists so a caller can say when that choice
    skipped a file.

    Args:
        path: An explicit config file. Missing it is an error, because a
            ``--config`` the user typed and this tool ignored is worse than a
            crash.

    Returns:
        A ``Config``. With no ``path`` and no ``voice_tics.toml`` in the
        working directory, the defaults: the tool runs out of the box.

    Raises:
        ConfigError: The file is unreadable, is not valid TOML, or says
            something this version does not understand.
    """
    named = bool(path)
    target = path or CONFIG_NAME
    try:
        with open(target, "rb") as fh:
            data = fh.read()
    except FileNotFoundError:
        if named:
            raise ConfigError(f"no config file at {target}") from None
        return Config.default()
    except OSError as exc:
        raise ConfigError(f"cannot read {target}: {exc}") from None
    toml = _toml_module()
    try:
        raw = toml.loads(data.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise ConfigError(f"cannot read {target}: {exc}") from None
    except toml.TOMLDecodeError as exc:
        raise ConfigError(f"{target}: {exc}") from None
    return Config.parse(raw)


def unread_parent_config(path: str = "", start: str = "") -> str:
    """A note naming a parent directory's config that ``load`` did not read.

    Run from a subdirectory of a project, the project's banned phrases and
    tiers silently did not apply, and a lint gate went green. Discovery stays
    in the working directory (see ``load``); this makes the skip visible.

    Args:
        path: The ``--config`` the user passed, if any. A named config means
            nothing was skipped.
        start: Directory to search from; the working directory by default.

    Returns:
        A one-line note for stderr, or "" when a config was named, one exists
        in the working directory, or no parent has one.
    """
    if path:
        return ""
    here = os.path.abspath(start or os.getcwd())
    if os.path.exists(os.path.join(here, CONFIG_NAME)):
        return ""
    parent = os.path.dirname(here)
    while parent != here:
        candidate = os.path.join(parent, CONFIG_NAME)
        if os.path.isfile(candidate):
            return (f"note: {candidate} was not read; config is read from "
                    "the working directory only. Pass --config to use it.")
        here, parent = parent, os.path.dirname(parent)
    return ""
