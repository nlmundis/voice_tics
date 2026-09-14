"""Repo invariants and redaction: the things nothing else would notice.

Docs drift, example configs rot, and a redaction default flips the wrong way
in a one-line diff. None of those break a unit test of the measurement, so
they are pinned here.
"""

from __future__ import annotations

import ast
import fnmatch
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import tokenize
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import prose_lint  # noqa: E402
import vt_config as config_mod  # noqa: E402
import vt_redact  # noqa: E402
from tests import support  # noqa: E402


def _read(path: str) -> str:
    """File contents, with the handle closed."""
    with open(path, encoding="utf-8") as fh:
        return fh.read()


class RedactionTest(unittest.TestCase):
    """The allowlist must fail toward over-redacting, never toward leaking."""

    def test_an_address_is_redacted_by_default(self) -> None:
        """No allowlist means every address goes. This is the shipped default.

        Both TLDs on purpose: a default allowlist of ("com",) would still
        redact the .io one, so a .io-only test cannot see a default that has
        quietly been opened up.
        """
        self.assertEqual(vt_redact.redact("write to sam@vendor.io now"),
                         "write to [REDACTED-EMAIL] now")
        self.assertEqual(vt_redact.redact("write to sam@vendor.com now"),
                         "write to [REDACTED-EMAIL] now")
        self.assertEqual(vt_redact.redact("sam@mail.vendor.com"),
                         "[REDACTED-EMAIL]")

    def test_phone_numbers_go_too(self) -> None:
        self.assertIn("[REDACTED-PHONE]", vt_redact.redact("call 555-123-4567"))

    def test_an_allowlisted_domain_survives(self) -> None:
        self.assertEqual(
            vt_redact.redact("sam@mine.com", keep_domains=["mine.com"]),
            "sam@mine.com")

    def test_a_subdomain_of_an_allowlisted_domain_survives(self) -> None:
        """The intent of the rule, and the reason for the leading dot."""
        self.assertEqual(
            vt_redact.redact("jira@mail.mine.com", keep_domains=["mine.com"]),
            "jira@mail.mine.com")

    def test_the_three_ways_a_substring_allowlist_fails_open(self) -> None:
        """Each one leaves a real external address in cleartext.

        These are the cases that turned the allowlist from a substring test
        into a parse of the domain. All three must redact.
        """
        for address in ("sam@mine.com.evil.net",   # merely CONTAINS the entry
                        "sam@notmine.com",          # suffix, no dot boundary
                        "mine.com@evil.net"):       # match was in the LOCAL part
            with self.subTest(address=address):
                self.assertEqual(
                    vt_redact.redact(address, keep_domains=["mine.com"]),
                    "[REDACTED-EMAIL]")

    def test_a_non_ascii_local_part_is_redacted_whole(self) -> None:
        """The ASCII-only pattern published the head of the name in cleartext.

        It matched the ASCII TAIL of a non-ASCII local part, so
        'mariagit@vendor.io' with an accent redacted to 'mar[REDACTED-EMAIL]'
        while looking like the guard had fired.
        """
        got = vt_redact.redact("maríagit@vendor.io")
        self.assertEqual(got, "[REDACTED-EMAIL]")

    def test_a_decomposed_name_is_redacted(self) -> None:
        """NFD 'jose' is 'jose' + U+0301; \\w alone does not span the mark."""
        self.assertEqual(vt_redact.redact("josé@vendor.io"),
                         "[REDACTED-EMAIL]")

    def test_a_version_number_is_not_an_address(self) -> None:
        text = "see build 1.2.3 and host:8080"
        self.assertEqual(vt_redact.redact(text), text)

    def test_a_package_pin_is_not_an_address(self) -> None:
        self.assertEqual(vt_redact.redact("pin pkg@1.2.3 here"),
                         "pin pkg@1.2.3 here")

    def test_no_character_in_a_name_publishes_its_head(self) -> None:
        """Any allowlist of name characters forgets one, and each one it
        forgets prints the head of the name while the guard appears to fire.
        An apostrophe, the zero-width non-joiner of Persian orthography, and
        an Adlam combining mark from outside the Basic Multilingual Plane."""
        for address in ("mary.o'donnell@evil.net",
                        "\u0646\u0627\u0635\u0631\u200c\u0645\u062d@vendor.io",
                        "\U0001e922\U0001e945\U0001e926@vendor.io"):
            with self.subTest(address=ascii(address)):
                self.assertEqual(vt_redact.redact("to " + address),
                                 "to [REDACTED-EMAIL]")

    def test_a_mark_outside_the_bmp_in_the_domain_does_not_hide_it(self) -> None:
        self.assertEqual(
            vt_redact.redact("sam@\U0001e922\U0001e945\U0001e926.io"),
            "[REDACTED-EMAIL]")

    def test_an_address_in_punctuation_keeps_the_punctuation(self) -> None:
        self.assertEqual(vt_redact.redact("(sam@vendor.io)"),
                         "([REDACTED-EMAIL])")
        self.assertEqual(vt_redact.redact("mailto:sam@vendor.io"),
                         "mailto:[REDACTED-EMAIL]")

    def test_a_long_run_without_an_address_is_linear(self) -> None:
        """The start-at-a-delimiter guard: without it, every offset of a long
        token is rescanned to the end, and 100,000 characters take minutes."""
        import time
        started = time.perf_counter()
        vt_redact.redact("a" * 100000)
        self.assertLess(time.perf_counter() - started, 2.0)

    def test_every_common_phone_rendering_is_redacted(self) -> None:
        """One separator between every group was required, so the two most
        common US renderings and every international one printed in the clear."""
        for number in ("555-123-4567", "(555) 123-4567", "(555)123-4567",
                       "5551234567", "555.123.4567", "555\u2013123\u20134567",
                       "+1 555 123 4567", "+44 20 7946 0958", "020 7946 0958",
                       "+33 6 12 34 56 78", "+49 30 901820"):
            with self.subTest(number=number):
                self.assertEqual(vt_redact.redact(f"call {number} today"),
                                 "call [REDACTED-PHONE] today")

    def test_short_numbers_are_not_phones(self) -> None:
        for text in ("in 2026-09-12", "count 123456789", "ext 4567"):
            with self.subTest(text=text):
                self.assertEqual(vt_redact.redact(text), text)


class KeepDomainsConfigTest(unittest.TestCase):
    """An allowlist entry that can match nothing must not load silently."""

    def _parse(self, domains):
        return config_mod.Config.parse({"output": {"keep_domains": domains}})

    def test_a_bare_domain_is_accepted_lowercased(self) -> None:
        self.assertEqual(self._parse(["Mine.com"]).keep_domains, ("mine.com",))

    def test_entries_that_would_match_nothing_are_rejected(self) -> None:
        for entry in (".mine.com", "mine.com.", " mine.com", "mine .com",
                      "sam@mine.com", ""):
            with self.subTest(entry=entry):
                with self.assertRaises(config_mod.ConfigError):
                    self._parse([entry])

    def test_the_error_names_the_spelling_that_works(self) -> None:
        with self.assertRaises(config_mod.ConfigError) as caught:
            self._parse([".mine.com"])
        self.assertIn('"mine.com"', str(caught.exception))


class ExampleConfigTest(unittest.TestCase):
    """The shipped examples must parse, or they are worse than absent."""

    def test_the_annotated_example_parses(self) -> None:
        cfg = config_mod.load(os.path.join(REPO, "voice_tics.toml.example"))
        # Everything optional is commented out, so it must equal the defaults.
        self.assertEqual(cfg.signatures, ())
        self.assertFalse(cfg.emdash_enabled)

    def test_the_full_example_exercises_every_documented_key(self) -> None:
        """examples/full.toml exists so the parser and the docs cannot drift."""
        cfg = config_mod.load(os.path.join(REPO, "examples", "full.toml"))
        self.assertTrue(cfg.samples)
        self.assertTrue(cfg.signatures)
        self.assertTrue(cfg.emdash_enabled)
        self.assertTrue(cfg.emdash_exempt)
        self.assertTrue(cfg.keep_domains)
        self.assertTrue(cfg.error_tics)
        self.assertTrue(cfg.warn_tics)

    def test_every_config_table_key_appears_in_the_annotated_example(self) -> None:
        """A key the parser accepts but the example never mentions is a key
        nobody will discover, and a key the example mentions but the parser
        rejects is a broken copy-paste. Both directions are checked."""
        text = _read(os.path.join(REPO, "voice_tics.toml.example"))
        known = (config_mod.CORPUS_KEYS | config_mod.BASELINE_KEYS
                 | config_mod.LINT_KEYS | config_mod.EMDASH_KEYS
                 | config_mod.OUTPUT_KEYS) - {"emdash"}
        for key in known:
            with self.subTest(key=key):
                self.assertRegex(text, rf"(?m)^#?\s*{re.escape(key)}\s*=",
                                 msg=f"{key} is undocumented")


class MutationSpecTest(unittest.TestCase):
    """The mutation gate's spec must name real files and real suites."""

    def _spec(self) -> dict:
        try:
            import tomllib
        except ModuleNotFoundError:
            import tomli as tomllib  # type: ignore[no-redef]
        with open(os.path.join(REPO, "mutt_check.toml"), "rb") as fh:
            return tomllib.load(fh)

    def test_every_mutant_names_a_file_that_exists(self) -> None:
        for mutant in self._spec()["mutant"]:
            with self.subTest(mutant=mutant["name"]):
                self.assertTrue(
                    os.path.isfile(os.path.join(REPO, mutant["file"])),
                    f"{mutant['file']} does not exist")

    def test_every_mutant_anchor_appears_exactly_once(self) -> None:
        """A stale or ambiguous anchor tests nothing.

        mutt_check reports this as STALE at run time, but that needs the whole
        gate; this catches it in the unit suite, in under a second.
        """
        for mutant in self._spec()["mutant"]:
            with self.subTest(mutant=mutant["name"]):
                text = _read(os.path.join(REPO, mutant["file"]))
                self.assertEqual(
                    text.count(mutant["find"]), 1,
                    f"anchor for {mutant['name']} is stale or ambiguous")

    def test_every_mutant_says_why_it_matters(self) -> None:
        for mutant in self._spec()["mutant"]:
            with self.subTest(mutant=mutant["name"]):
                self.assertTrue(mutant.get("why", "").strip())

    def test_every_named_suite_is_importable(self) -> None:
        import importlib
        spec = self._spec()
        names = set(spec["run"]["suites"])
        for mutant in spec["mutant"]:
            names.update(mutant.get("suites", []))
        for dotted in sorted(names):
            with self.subTest(suite=dotted):
                parts = dotted.split(".")
                # tests.test_x or tests.test_x.ClassName
                module = importlib.import_module(".".join(parts[:2]))
                if len(parts) > 2:
                    self.assertTrue(hasattr(module, parts[2]),
                                    f"{parts[2]} not in {module.__name__}")


# --- published measurements ------------------------------------------------
# No measured figure is published today: the ratios were withdrawn because
# the transcript baseline counted some of the model's writing as the
# author's. These checks are plain functions returning problems, so
# MeasurementBindingTest can run them on a made-up table and measurement file
# and keep them working while the real README has nothing for them to check.

MEASUREMENTS = os.path.join(REPO, "docs", "measurements")
DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
SEPARATOR = re.compile(r"\|(?:\s*:?-+:?\s*\|)+")
# Number words a multiple can be spelled with.
_NUMBER_WORDS = (r"(?:two|three|four|five|six|seven|eight|nine|ten|eleven"
                 r"|twelve|thirteen|fourteen|fifteen|sixteen|seventeen"
                 r"|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy"
                 r"|eighty|ninety|hundred|thousand|several|many)")
# A multiple: digits with x, the times sign, fold or times, spaced or
# hyphenated or not; a number word with fold; twice, thrice or a number word
# and times, before as, more, less or the; or a power of ten said in words.
# Markdown emphasis around a number does not hide it. Not shares, which are sizes
# rather than ratios, and not colon ratios, which nothing here has used.
# MeasurementBindingTest lists the spellings either way.
RATIO = re.compile(
    r"(?i)(?<![A-Za-z0-9.%\\])\d*[.,]?\d+(?:\s?[x×]|\s?-?(?:fold|times))"
    r"(?![A-Za-z0-9])"
    rf"|\b{_NUMBER_WORDS}-?fold\b"
    rf"|\b(?:twice|thrice|{_NUMBER_WORDS}[\s-]+times)\s+(?:as|more|less|the)\b"
    r"|\borders?\s+of\s+magnitude\b")
# One measured figure in TIC_PROVENANCE: a ratio, the date it was measured,
# and the model / author uses behind it.
PROVENANCE_FIGURE = re.compile(
    r"(?P<ratio>\d+\.\d+x)\b[^;]*?(?P<date>\d{4}-\d{2}-\d{2})\s*"
    r"\((?P<counts>[\d,]+ / [\d,]+) uses\)")
# Table rows that do not name their detector in backticks.
ROW_KEYS = {"Mean sentence length": None, "Em dash": "em_dash_aside",
            "The slop lexicon": "delve_ecosystem"}


def load_measurements(folder: str) -> "tuple[dict, list[str]]":
    """Measurement files keyed by date, and any file misnamed for its date."""
    out: dict = {}
    problems: "list[str]" = []
    if not os.path.isdir(folder):
        return out, problems
    for name in sorted(os.listdir(folder)):
        if name.endswith(".json"):
            with open(os.path.join(folder, name), encoding="utf-8") as fh:
                doc = json.load(fh)
            if name != f"{doc.get('date')}.json":
                problems.append(f"{name} records the date {doc.get('date')!r}")
            out[doc.get("date")] = doc
    return out, problems


def figure(doc: dict, key: "str | None") -> str:
    """A table cell as the README must print it, from the raw counts."""
    if key is None:
        return (f"{doc['model']['mean_sentence_len']:.2f} model, "
                f"{doc['author']['mean_sentence_len']:.2f} author")
    model, author = doc["structures"][key]["model"], doc["structures"][key]["author"]
    ratio = ((model / doc["model"]["words"])
             / (max(author, 0.5) / doc["author"]["words"]))
    return f"{ratio:.1f}x ({model:,} / {author:,})"


def _cells(line: str) -> "list[str]":
    return [c.strip() for c in line.strip().strip("|").split("|")]


def headline_table(readme: str) -> "dict | None":
    """The README's first table whose header row holds a date, or None.

    Every header cell after the label that is a bare ISO date is a
    measurement column, wherever it sits. Taking the columns by position
    exempted a trailing date column from every check.
    """
    lines = readme.splitlines()
    start = next((i for i, line in enumerate(lines)
                  if line.startswith("|") and DATE.search(line)), None)
    if start is None:
        return None
    header = _cells(lines[start])
    end = start + 2
    rows = []
    while end < len(lines) and lines[end].startswith("|"):
        rows.append(_cells(lines[end]))
        end += 1
    columns = [i for i, cell in enumerate(header) if i and DATE.fullmatch(cell)]
    return {"lines": lines, "start": start, "end": end, "header": header,
            "rows": rows, "columns": columns}


def _row_key(label: str) -> "tuple[bool, str | None]":
    """(whether the row names a known measurement, its structure key)."""
    tick = re.match(r"`(\w+)`", label)
    if tick:
        return True, tick.group(1)
    for prefix, key in ROW_KEYS.items():
        if label.startswith(prefix):
            return True, key
    return False, None


def table_problems(readme: str, measured: dict) -> "list[str]":
    """Why the README's dated table and the measurement files disagree."""
    table = headline_table(readme)
    if table is None:
        return ["measurement files but no table"] if measured else []
    problems = []
    lines, start, header = table["lines"], table["start"], table["header"]
    if start + 1 >= len(lines) or not SEPARATOR.fullmatch(lines[start + 1].strip()):
        problems.append("the dated header row has no |---| row under it")
    last = len(header) - 1
    for i, cell in enumerate(header):
        if i == 0 or i in table["columns"]:
            continue
        if DATE.search(cell):
            problems.append(f"header cell {cell!r} holds a date but is not one")
        elif i != last:
            problems.append(f"header cell {cell!r} sits among the measurement "
                            "columns but is not a date")
    dates = [header[i] for i in table["columns"]]
    if sorted(dates) != sorted(measured):
        problems.append(f"table dates {sorted(dates)} but measurement files "
                        f"{sorted(measured)}")
    keyed = []
    for cells in table["rows"]:
        known, key = _row_key(cells[0])
        if known:
            keyed.append((cells, key))
        else:
            problems.append(f"row {cells[0]!r} names no known measurement")
    keys = [key for _cells_, key in keyed]
    if len(keys) != len(set(keys)):
        problems.append("a measurement has two rows")
    for date, doc in measured.items():
        if set(doc["structures"]) != set(keys) - {None}:
            problems.append(f"{date} measures other structures than the rows")
    for cells, key in keyed:
        for i in table["columns"]:
            doc = measured.get(header[i])
            if doc is None or (key is not None and key not in doc["structures"]):
                continue
            got = cells[i] if i < len(cells) else ""
            if got != figure(doc, key):
                problems.append(f"{cells[0][:30]} on {header[i]}: {got!r}, "
                                f"counts give {figure(doc, key)!r}")
    return problems


def provenance_problems(provenance: "dict[str, str]", measured: dict) -> "list[str]":
    """Why a TIC_PROVENANCE note's figures are not the measured ones.

    Each ratio must be a dated figure with its counts, equal to what those
    counts give; a bare ratio is checked by nothing.
    """
    problems = []
    for key, note in provenance.items():
        figures = list(PROVENANCE_FIGURE.finditer(note))
        if len(RATIO.findall(note)) != len(figures):
            problems.append(f"{key}'s note carries a ratio with no date and counts")
        for fig in figures:
            doc = measured.get(fig.group("date"))
            if doc is None or key not in doc["structures"]:
                problems.append(f"{key} cites {fig.group('date')}, which has "
                                "no measurement of it")
                continue
            got = f"{fig.group('ratio')} ({fig.group('counts')})"
            if got != figure(doc, key):
                problems.append(f"{key}: {got}, counts give {figure(doc, key)}")
    return problems


def readme_without_recomputed_cells(readme: str) -> str:
    """The README with only the cells table_problems recomputes blanked.

    Blanking the whole table span let a ratio in its prose column past the
    copy guard.
    """
    table = headline_table(readme)
    if table is None:
        return readme
    lines = list(table["lines"])
    for n in range(table["start"] + 2, table["end"]):
        cells = _cells(lines[n])
        for i in table["columns"]:
            if i < len(cells):
                cells[i] = ""
        lines[n] = "| " + " | ".join(cells) + " |"
    return "\n".join(lines)


def _gitignore_patterns(root: str) -> "list[tuple[str, bool]]":
    """The .gitignore patterns at root, as (glob, folders only).

    Only the forms this repo uses are understood: a name or glob matched
    against a file's or folder's own name, with a trailing slash for folders
    only. Any other form raises, so a pattern the walk would misread fails the
    suite rather than scanning or skipping the wrong files.
    """
    path = os.path.join(root, ".gitignore")
    if not os.path.isfile(path):
        return []
    out = []
    for raw in _read(path).splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        pattern = line.rstrip("/")
        if pattern.startswith("!") or "/" in pattern or "**" in pattern:
            raise ValueError(f".gitignore pattern {raw!r} is not a form "
                             "shipped_files understands")
        out.append((pattern, line.endswith("/")))
    return out


def shipped_files(root: str = REPO) -> "list[str]":
    """Every text file under root that could publish a figure, by walking it.

    Walked rather than listed, so a new file or folder is read without anyone
    remembering to add it, and chosen by whether it decodes as UTF-8 rather
    than by suffix, so a CHANGELOG or a new example format is read too.
    Skipped: whatever .gitignore names, since a developer's own config must
    not fail their build; hidden folders other than .github; caches; and any
    folder holding a virtual environment.
    """
    patterns = _gitignore_patterns(root)

    def ignored(name: str, folder: bool) -> bool:
        return any(fnmatch.fnmatchcase(name, glob)
                   for glob, folders_only in patterns if folder or not folders_only)

    out = []
    for folder, dirs, files in os.walk(root):
        dirs[:] = sorted(
            d for d in dirs
            if not ((d.startswith(".") and d != ".github") or d == "__pycache__"
                    or os.path.isfile(os.path.join(folder, d, "pyvenv.cfg"))
                    or ignored(d, True)))
        for name in sorted(files):
            if name == ".git" or ignored(name, False):
                continue
            path = os.path.join(folder, name)
            try:
                _read(path)
            except (UnicodeDecodeError, OSError):
                continue
            out.append(os.path.relpath(path, root).replace(os.sep, "/"))
    return out


def _blank_provenance_strings(source: str) -> str:
    """prose_lint's source with the string literals of TIC_PROVENANCE blanked.

    provenance_problems checks those strings. A comment beside them, even one
    between two parts of an implicitly joined string, is checked by nothing
    else, so it stays. The assignment is found by parsing, and a module
    without exactly one top-level assignment to TIC_PROVENANCE raises rather
    than blanking the wrong strings or none.
    """
    assigns = [
        node for node in ast.parse(source).body
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and any(isinstance(target, ast.Name) and target.id == "TIC_PROVENANCE"
                for target in (node.targets if isinstance(node, ast.Assign)
                               else [node.target]))]
    if len(assigns) != 1:
        raise ValueError("expected one top-level TIC_PROVENANCE assignment, "
                         f"found {len(assigns)}")
    first, last = assigns[0].lineno, assigns[0].end_lineno
    offsets = [0]
    for line in source.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    spans = [(offsets[tok.start[0] - 1] + tok.start[1],
              offsets[tok.end[0] - 1] + tok.end[1])
             for tok in tokenize.generate_tokens(io.StringIO(source).readline)
             if tok.type == tokenize.STRING and first <= tok.start[0]
             and tok.end[0] <= last]
    for a, b in reversed(spans):
        source = source[:a] + " " + source[b:]
    return source


def _comments_and_docstrings(source: str) -> str:
    """A test module's prose: its comments and docstrings.

    Its other string literals are fixtures and are not read, assertion
    messages included; a figure written into one would pass.
    """
    parts = [tok.string for tok in
             tokenize.generate_tokens(io.StringIO(source).readline)
             if tok.type == tokenize.COMMENT]
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)):
            doc = ast.get_docstring(node, clean=False)
            if doc:
                parts.append(doc)
    return "\n".join(parts)


def _toml_loads(text: str) -> dict:
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib  # type: ignore[no-redef]
    return tomllib.loads(text)


def _mutation_spec_prose(text: str) -> str:
    """A mutation spec's claims: its comment lines and every string value
    except a mutant's find and replace, which are fixtures.

    Read with a TOML parser, because a pattern over the raw text cannot tell a
    find key from a line inside another string.
    """
    parts = [line for line in text.splitlines() if line.lstrip().startswith("#")]

    def strings(value: object, key: "str | None" = None) -> None:
        if isinstance(value, dict):
            for k, v in value.items():
                strings(v, k)
        elif isinstance(value, list):
            for v in value:
                strings(v, key)
        elif isinstance(value, str) and key not in ("find", "replace"):
            parts.append(value)

    strings(_toml_loads(text))
    return "\n".join(parts)


def scanned_text(name: str, text: str) -> str:
    """What the copy guard reads of a shipped file, as one line of prose.

    Comment markers that open a line, and the quotes between implicitly joined
    string literals, are removed first, so a figure split across lines still
    reads as one phrase.
    """
    if name == "prose_lint.py":
        text = _blank_provenance_strings(text)
    elif name == "README.md":
        text = readme_without_recomputed_cells(text)
    elif name == "mutt_check.toml":
        text = _mutation_spec_prose(text)
    elif name.startswith("tests/") and name.endswith(".py"):
        text = _comments_and_docstrings(text)
    text = re.sub(r"\n[ \t]*#+[ \t]?", "\n", text)
    text = re.sub(r"[\"'][ \t]*\n[ \t]*[\"']", " ", text)
    return re.sub(r"\s+", " ", text)


def copied_ratios(name: str, text: str,
                  allowed: "dict[tuple[str, str], str]") -> "tuple[list, set]":
    """Ratios in a file's scanned text outside an allowed phrase, and the
    allowed phrases for that file that were found."""
    spans, found = [], set()
    for (file, phrase) in allowed:
        if file == name:
            for m in re.finditer(re.escape(phrase), text):
                spans.append(m.span())
                found.add((file, phrase))
    copies = [(name, text[max(0, m.start() - 40):m.end() + 10])
              for m in RATIO.finditer(text)
              if not any(a <= m.start() and m.end() <= b for a, b in spans)]
    return copies, found


def copy_guard(root: str,
               allowed: "dict[tuple[str, str], str]") -> "tuple[list, list]":
    """Ratios copied into the prose under root, and allowances gone stale.

    A stale allowance is a key of ``allowed`` whose phrase no longer appears
    in its file, so a permission cannot outlive the text it was granted for.
    """
    copies, found = [], set()
    for name in shipped_files(root):
        text = scanned_text(name, _read(os.path.join(root, name)))
        file_copies, file_found = copied_ratios(name, text, allowed)
        copies += file_copies
        found |= file_found
    return copies, sorted(set(allowed) - found)


class DocsTest(unittest.TestCase):
    """The README must not promise commands or files that are not there."""

    def _readme(self) -> str:
        return _read(os.path.join(REPO, "README.md"))

    def test_every_python_script_the_readme_invokes_exists(self) -> None:
        for script in set(re.findall(r"python3 (\w+\.py)", self._readme())):
            with self.subTest(script=script):
                self.assertTrue(os.path.isfile(os.path.join(REPO, script)))

    # Snake_case identifiers the README backticks that are NOT detectors:
    # modules and functions. Listed rather than pattern-matched so adding one
    # is a deliberate edit.
    NOT_DETECTORS = frozenset({"read_turns", "prose_lint", "voice_tics",
                               "mutt_check", "lint_text", "read_samples"})

    def test_every_detector_name_in_the_readme_is_real(self) -> None:
        """A README naming a detector that does not exist is a broken promise."""
        import voice_tics
        known = {n for n, _, _ in voice_tics.STRUCTURE_PATTERNS}
        found = set(re.findall(r"`([a-z][a-z0-9]*(?:_[a-z0-9]+)+)`",
                               self._readme()))
        self.assertTrue(found, "the README should name some detectors")
        for token in sorted(found - self.NOT_DETECTORS):
            with self.subTest(token=token):
                self.assertIn(token, known,
                              f"README names {token}, which is not a detector")

    def test_the_readme_files_referenced_by_name_exist(self) -> None:
        for name in ("voice_tics.toml.example", "LICENSE", "Makefile"):
            with self.subTest(name=name):
                self.assertTrue(os.path.isfile(os.path.join(REPO, name)))

    def test_the_headline_table_is_computed_from_the_measurements(self) -> None:
        """Every cell equals what the raw counts give; no table, no files.

        The August em dash sat in a table here as 1.0x, a figure no
        measurement ever produced. A cell a test recomputes cannot do that,
        and a measurement file with no table, or a table with no files, is a
        claim with half its evidence missing.
        """
        measured, misnamed = load_measurements(MEASUREMENTS)
        self.assertEqual(misnamed, [])
        self.assertEqual(table_problems(self._readme(), measured), [])

    def test_the_readme_corpus_sizes_are_the_measurements(self) -> None:
        readme = self._readme()
        for date, doc in load_measurements(MEASUREMENTS)[0].items():
            with self.subTest(date=date):
                self.assertIn(f"{doc['model']['words']:,} words of model prose", readme)
                self.assertIn(f"{doc['author']['words']:,} of the author's", readme)
                if "transcripts_read" in doc:
                    self.assertIn(f"{doc['transcripts_read']:,} transcripts read, "
                                  f"{doc['dropped_as_scheduled_runs']:,} dropped",
                                  readme)

    def test_tic_provenance_is_computed_from_the_measurements(self) -> None:
        """Each dated figure in TIC_PROVENANCE equals the one the counts give,
        and a note carries no ratio that is not such a figure.

        Matching a ratio anywhere in the README stopped meaning anything once
        the table carried two dates, and one ratio's digits can contain
        another's.
        """
        measured = load_measurements(MEASUREMENTS)[0]
        self.assertEqual(
            provenance_problems(prose_lint.TIC_PROVENANCE, measured), [])

    # Ratios that may stay in shipped prose because they are not measurements
    # of anyone's writing, keyed by the phrase that carries each one so the
    # same number elsewhere in the file is still caught. Anything else must be
    # a recomputed table cell or a checked TIC_PROVENANCE figure.
    ALLOWED_RATIOS = {
        ("README.md", "A phrase at 40x your rate"):
            "illustration of what a tic ratio means",
        ("README.md", "a tic that read 593x"):
            "synthetic contamination sweep, not a corpus measurement",
        ("README.md", "samples read 3.2x at 25% contamination"):
            "synthetic contamination sweep, not a corpus measurement",
        ("voice_tics.py", "appears at 40x your rate"):
            "illustration of what a tic ratio means",
        ("voice_tics.py", "a tic that read 593x against clean samples read "
                          "3.2x at 25% contamination and 1.3x at 70%"):
            "synthetic contamination sweep, not a corpus measurement",
        ("voice_tics.py", "top tics at ~364x"):
            "harness text the first, unfiltered scan ranked as the author's: "
            "the incident behind a filter, not a finding about any writer",
        ("voice_tics.py", "top-30 tic at 113x"):
            "a harness notice the first, unfiltered scan ranked as a tic: "
            "the incident behind a filter, not a finding about any writer",
        ("tests/test_voice_tics.py", "top tics at ~364x"):
            "the same scheduled-run incident, in the regression's docstring",
        ("tests/test_voice_tics.py", "top tics at 364x"):
            "the same scheduled-run incident, in the regression's docstring",
        ("tests/test_voice_tics.py", "top-30 tic at 113x"):
            "the same harness-notice incident, in the regression's docstring",
        ("tests/test_repo.py", "sat in a table here as 1.0x"):
            "the incident this guard exists for: a figure no run produced",
        ("tests/test_repo.py", "sat in four files as 1.0x"):
            "the incident this guard exists for: a figure no run produced",
        ("tests/test_voice_tics.py", "generous by three orders of magnitude"):
            "a timing budget in a performance test, not a measurement of writing",
        ("mutt_check.toml", "how 1.0x reached the README"):
            "the incident this guard exists for: a figure no run produced",
        ("mutt_check.toml", "how 1.0x sat in four files"):
            "the incident this guard exists for: a figure no run produced",
    }

    def test_measured_ratios_are_not_copied_into_other_files(self) -> None:
        """No file under the repo carries a ratio in its prose outside a
        table the tests recompute, the TIC_PROVENANCE strings they check, and
        the phrases ALLOWED_RATIOS names with a reason.

        The August em dash sat in four files as 1.0x, a figure no measurement
        ever produced, and each copy looked like corroboration for the others.
        shipped_files says which files are read and scanned_text which part
        of each; RATIO says which spellings count.
        """
        copies, stale = copy_guard(REPO, self.ALLOWED_RATIOS)
        self.assertEqual(copies, [])
        self.assertEqual(stale, [], "an allowed phrase no longer appears where named")

    def test_the_readme_passes_its_own_linter(self) -> None:
        """A linter whose README fails its shipped defaults argues against
        itself. A reflow once joined 'that is' and 'deliberate' onto one line
        and the README began failing method_defence. Warnings may stay."""
        found, _info = prose_lint.lint_text(self._readme())
        self.assertEqual([(f.line, f.rule) for f in found if f.tier == "error"], [])


class MeasurementBindingTest(unittest.TestCase):
    """The measurement checks catch what they exist for, on made-up figures.

    Nothing is published for them to check today, so without these a broken
    check would pass the suite until the day a table returned.
    """

    DOC = {"date": "2026-10-01",
           "model": {"words": 1000, "mean_sentence_len": 12.0},
           "author": {"words": 1000, "mean_sentence_len": 10.0},
           "structures": {"em_dash_aside": {"model": 10, "author": 2}}}
    TABLE = ("intro\n\n| Measured | 2026-10-01 | What a wordlist does |\n"
             "|---|---|---|\n| Em dash | 5.0x (10 / 2) | Flags it |\n\nafter\n")

    def _measured(self, *docs: dict) -> dict:
        return {doc["date"]: doc for doc in docs}

    def test_a_table_that_matches_its_counts_passes(self) -> None:
        self.assertEqual(table_problems(self.TABLE, self._measured(self.DOC)), [])

    def test_a_wrong_cell_is_caught(self) -> None:
        readme = self.TABLE.replace("5.0x (10 / 2)", "9.9x (10 / 2)")
        self.assertTrue(table_problems(readme, self._measured(self.DOC)))

    def test_a_table_with_no_files_is_caught(self) -> None:
        self.assertTrue(table_problems(self.TABLE, {}))

    def test_files_with_no_table_are_caught(self) -> None:
        self.assertTrue(table_problems("no table\n", self._measured(self.DOC)))

    LATER = {"date": "2026-10-05",
             "model": {"words": 1000, "mean_sentence_len": 12.0},
             "author": {"words": 1000, "mean_sentence_len": 10.0},
             "structures": {"em_dash_aside": {"model": 5, "author": 1}}}

    def test_a_trailing_date_column_is_a_measurement_column(self) -> None:
        readme = ("| Measured | 2026-10-01 | 2026-10-05 |\n|---|---|---|\n"
                  "| Em dash | 5.0x (10 / 2) | 9.9x (5 / 1) |\n")
        problems = table_problems(readme, self._measured(self.DOC, self.LATER))
        self.assertEqual(len(problems), 1, problems)
        self.assertIn("2026-10-05", problems[0])

    def test_a_correct_trailing_date_column_passes(self) -> None:
        readme = ("| Measured | 2026-10-01 | 2026-10-05 |\n|---|---|---|\n"
                  "| Em dash | 5.0x (10 / 2) | 5.0x (5 / 1) |\n")
        self.assertEqual(
            table_problems(readme, self._measured(self.DOC, self.LATER)), [])

    def _only_problem(self, readme: str, measured: dict, needle: str) -> None:
        problems = table_problems(readme, measured)
        self.assertEqual(len(problems), 1, problems)
        self.assertIn(needle, problems[0])

    def test_a_header_that_is_not_a_bare_date_is_caught(self) -> None:
        readme = self.TABLE.replace("| 2026-10-01 |", "| 2026-10-01 rebuilt |")
        self._only_problem(readme, {}, "holds a date but is not one")

    def test_a_middle_column_that_is_not_a_date_is_caught(self) -> None:
        readme = ("| Measured | Notes | 2026-10-01 | What a wordlist does |\n"
                  "|---|---|---|---|\n"
                  "| Em dash | seen | 5.0x (10 / 2) | Flags it |\n")
        self._only_problem(readme, self._measured(self.DOC), "is not a date")

    def test_a_missing_separator_row_is_caught(self) -> None:
        readme = self.TABLE.replace("|---|---|---|\n", "")
        self.assertTrue(any("|---|" in p for p in
                            table_problems(readme, self._measured(self.DOC))))

    def test_a_row_naming_no_measurement_is_caught(self) -> None:
        readme = self.TABLE.replace(
            "| Em dash | 5.0x (10 / 2) | Flags it |\n",
            "| Em dash | 5.0x (10 / 2) | Flags it |\n| Colons | seen | x |\n")
        self._only_problem(readme, self._measured(self.DOC),
                           "names no known measurement")

    def test_a_file_measuring_other_structures_is_caught(self) -> None:
        doc = json.loads(json.dumps(self.DOC))
        doc["structures"]["not_x_but_y"] = {"model": 1, "author": 1}
        self._only_problem(self.TABLE, self._measured(doc),
                           "measures other structures")

    def test_a_measurement_with_two_rows_is_caught(self) -> None:
        readme = self.TABLE.replace(
            "| Em dash | 5.0x (10 / 2) | Flags it |\n",
            "| Em dash | 5.0x (10 / 2) | Flags it |\n"
            "| Em dash again | 5.0x (10 / 2) | Flags it |\n")
        self._only_problem(readme, self._measured(self.DOC), "two rows")

    def test_a_misnamed_measurement_file_is_caught(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            with open(os.path.join(folder, "2026-10-02.json"), "w",
                      encoding="utf-8") as fh:
                json.dump(self.DOC, fh)
            self.assertTrue(load_measurements(folder)[1])

    def test_provenance_figures_must_be_the_counts(self) -> None:
        measured = self._measured(self.DOC)
        good = {"em_dash_aside": "5.0x model/baseline, 2026-10-01 (10 / 2 uses)"}
        self.assertEqual(provenance_problems(good, measured), [])
        for note in ("9.9x model/baseline, 2026-10-01 (10 / 2 uses)",
                     "5.0x model/baseline, 2026-10-09 (10 / 2 uses)",
                     "error: once 5.0x, now withdrawn"):
            with self.subTest(note=note):
                self.assertTrue(provenance_problems({"em_dash_aside": note},
                                                    measured))

    def _ratios(self, text: str) -> "list[str]":
        return [m.group(0) for m in RATIO.finditer(text)]

    def test_only_recomputed_cells_escape_the_copy_guard(self) -> None:
        readme = self.TABLE.replace("Flags it", "Calls it 3.1x worse")
        self.assertEqual(self._ratios(scanned_text("README.md", readme)), ["3.1x"])

    def test_every_ratio_spelling_is_caught(self) -> None:
        for text in ("7.7x", "8×", "7.7 ×", "2.3-fold", "2.3 fold", "3 times",
                     "7-times", "_7.7x_", "**7.7x**", ".5x", "sevenfold",
                     "seven-fold", "twice as often", "seven times more",
                     "twenty times more", "twice the rate",
                     "an order of magnitude"):
            with self.subTest(text=text):
                self.assertTrue(RATIO.search(f"it ran {text} here"))
        for text in ("52%", "x86", "0x1f", "3x3", "half the words",
                     "ran it twice", "several times"):
            with self.subTest(text=text):
                self.assertIsNone(RATIO.search(f"it ran {text} here"))

    def test_a_figure_split_across_lines_is_read_as_one(self) -> None:
        for source in ("# it ran seven\n# times more\n",
                       'NOTE = ("it ran seven "\n        "times more")\n'):
            with self.subTest(source=source):
                self.assertEqual(
                    self._ratios(scanned_text("voice_tics.py", source)),
                    ["seven times more"])

    def test_a_comment_inside_the_provenance_literal_is_scanned(self) -> None:
        source = ('TIC_PROVENANCE: Dict[str, str] = {\n'
                  '    # once 12.4x\n    "k": "note 9.9x "\n'
                  '        # then 11.1x\n        "more",\n}\n'
                  'AFTER = "later 3.3x"\n')
        self.assertEqual(self._ratios(scanned_text("prose_lint.py", source)),
                         ["12.4x", "11.1x", "3.3x"])

    def test_a_missing_provenance_assignment_fails_loudly(self) -> None:
        with self.assertRaises(ValueError):
            scanned_text("prose_lint.py", "OTHER = {}\n")

    def test_test_prose_is_scanned_and_test_fixtures_are_not(self) -> None:
        source = ('"""Module 7.7x."""\n# comment 8x\n'
                  'def f():\n    """Doc 2.3-fold."""\n    return "fixture 9.9x"\n')
        text = scanned_text("tests/support.py", source)
        self.assertEqual(sorted(self._ratios(text)), ["2.3-fold", "7.7x", "8x"])

    def test_a_mutant_why_is_scanned_and_its_fixtures_are_not(self) -> None:
        spec = ('# section 6.6x\n[[mutant]]\nwhy = """it read 7.7x\n'
                'find = "8.8x"\n"""\nfind = "a 8x"\n'
                "replace = '''b\n9.9x'''\n")
        self.assertEqual(self._ratios(scanned_text("mutt_check.toml", spec)),
                         ["6.6x", "7.7x", "8.8x"])

    def test_an_allowed_value_is_allowed_only_in_its_phrase(self) -> None:
        allowed = {("README.md", "A phrase at 40x your rate"): "illustration"}
        copies, found = copied_ratios(
            "README.md", "A phrase at 40x your rate. The em dash ran 40x.",
            allowed)
        self.assertEqual(len(copies), 1)
        self.assertEqual(found, set(allowed))

    def _tree(self, root: str, files: "dict[str, bytes]") -> None:
        for rel, data in files.items():
            path = os.path.join(root, *rel.split("|"))
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as fh:
                fh.write(data)

    def test_the_walk_reads_text_and_skips_what_is_not_shipped(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            self._tree(root, {
                ".gitignore": b"venv/\nvoice_tics.toml\n",
                "README.md": b"hello", "NOTES": b"no suffix",
                ".github|w.yml": b"on: push", "examples|x.json": b"{}",
                "img.bin": b"\xff\xfe\x00\x81",
                "voice_tics.toml": b"# ran 7.7x", "venv|a.py": b"",
                "env|pyvenv.cfg": b"", "env|lib.py": b"",
                "__pycache__|a.py": b"", ".hidden|a.md": b""})
            self.assertEqual(sorted(shipped_files(root)), sorted([
                ".github/w.yml", ".gitignore", "NOTES", "README.md",
                "/".join(["examples", "x.json"])]))

    def test_an_unreadable_gitignore_pattern_fails_loudly(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            self._tree(root, {".gitignore": b"/build\n"})
            with self.assertRaises(ValueError):
                shipped_files(root)

    def test_a_copy_and_a_stale_allowance_are_both_reported(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            self._tree(root, {"NOTES": b"the em dash ran 7.7x"})
            allowed = {("README.md", "A phrase at 40x your rate"): "gone"}
            copies, stale = copy_guard(root, allowed)
        self.assertEqual([name for name, _context in copies], ["NOTES"])
        self.assertEqual(stale, sorted(allowed))

    def test_the_real_repo_walk_reaches_nested_folders(self) -> None:
        files = shipped_files()
        self.assertIn(".github/workflows/check.yml", files)
        self.assertIn("/".join(["tests", "test_repo.py"]), files)


class StdlibOnlyTest(unittest.TestCase):
    """"Stdlib only, Python 3.9+" must be true with no TOML parser installed.

    On 3.9 and 3.10 there is no ``tomllib``. A module-scope import made both
    entry points die before argparse, config file or not.
    """

    BLOCK = ("import sys; sys.modules['tomllib'] = None; "
             "sys.modules['tomli'] = None; sys.path.insert(0, %r); " % REPO)

    def _python(self, code: str) -> "subprocess.CompletedProcess[str]":
        with tempfile.TemporaryDirectory() as cwd:
            return subprocess.run([sys.executable, "-c", self.BLOCK + code],
                                  capture_output=True, text=True, cwd=cwd)

    def test_both_entry_points_import_without_a_toml_parser(self) -> None:
        run = self._python("import voice_tics, prose_lint; "
                           "print(prose_lint.lint_text('It is fine.')[0])")
        self.assertEqual(run.returncode, 0, run.stderr)

    def test_no_config_file_means_no_parser_is_needed(self) -> None:
        run = self._python("import vt_config; "
                           "print(vt_config.load().error_tics)")
        self.assertEqual(run.returncode, 0, run.stderr)

    def test_a_config_file_without_a_parser_names_what_to_install(self) -> None:
        path = os.path.join(REPO, "voice_tics.toml.example")
        run = self._python(
            "import vt_config\n"
            "try:\n"
            "    vt_config.load(%r)\n"
            "except vt_config.ConfigError as exc:\n"
            "    print(exc)\n" % path)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn("pip install tomli", run.stdout)

    def test_the_readme_says_when_tomli_is_needed(self) -> None:
        readme = _read(os.path.join(REPO, "README.md"))
        self.assertIn("Stdlib only, Python 3.9+", readme)
        self.assertIn("pip install tomli", readme)


class VersionSupportTest(unittest.TestCase):
    """The README's version floor and the CI matrix are one claim."""

    def test_ci_runs_every_version_from_the_readme_floor_up(self) -> None:
        readme = _read(os.path.join(REPO, "README.md"))
        floor = re.search(r"Python 3\.(\d+)\+", readme)
        self.assertIsNotNone(floor, "the README states no version floor")
        workflow = _read(os.path.join(REPO, ".github", "workflows",
                                      "check.yml"))
        matrix = re.search(r"python: \[(.+?)\]", workflow)
        self.assertIsNotNone(matrix, "the workflow has no python matrix")
        minors = sorted(int(m) for m in re.findall(r"3\.(\d+)",
                                                   matrix.group(1)))
        self.assertEqual(minors[0], int(floor.group(1)))
        self.assertEqual(minors, list(range(minors[0], minors[-1] + 1)),
                         "the matrix skips a version between floor and top")

    def test_ci_runs_the_whole_gate(self) -> None:
        workflow = _read(os.path.join(REPO, ".github", "workflows",
                                      "check.yml"))
        self.assertIn("make check", workflow)

    def test_ci_and_readme_pin_the_same_mutt_check(self) -> None:
        pin = r"mutt_check@(v[\d.]+)"
        readme = re.findall(pin, _read(os.path.join(REPO, "README.md")))
        workflow = re.findall(pin, _read(os.path.join(
            REPO, ".github", "workflows", "check.yml")))
        makefile = re.findall(pin, _read(os.path.join(REPO, "Makefile")))
        self.assertTrue(readme and workflow and makefile)
        self.assertEqual(set(readme) | set(workflow) | set(makefile),
                         {readme[0]})


class GateTest(unittest.TestCase):
    """`make check` must fail loudly, never skip, when the gate is absent."""

    def test_a_missing_mutt_check_stops_the_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = subprocess.run(
                ["make", "-s", "-C", REPO, "mutants",
                 "MUTT_CHECK=" + os.path.join(tmp, "absent")],
                capture_output=True, text=True)
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("pip install", run.stdout)


class HermeticSuiteTest(unittest.TestCase):
    """A developer's own voice_tics.toml must not reach the suite."""

    def test_scratch_files_are_removed_when_the_module_ends(self) -> None:
        support.enter_hermetic_cwd()
        try:
            path = support.scratch_file("x")
            cwd = os.getcwd()
            self.assertNotEqual(os.path.realpath(cwd), os.path.realpath(REPO))
        finally:
            support.leave_hermetic_cwd()
        self.assertFalse(os.path.exists(path))
        self.assertFalse(os.path.exists(cwd))


class ShippedCommandsTest(unittest.TestCase):
    """Every command the README tells a reader to run must exist after a clone."""

    STANDARD = frozenset({"python3", "make", "git", "cd", "cp", "cat", "pip"})

    def test_every_readme_command_is_shipped_or_standard(self) -> None:
        readme = _read(os.path.join(REPO, "README.md"))
        for block in re.findall(r"```bash\n(.*?)```", readme, re.S):
            for line in block.splitlines():
                command = line.split("#", 1)[0].strip()
                for stage in filter(None, (c.strip() for c in command.split("|"))):
                    first = stage.split()[0]
                    with self.subTest(command=stage):
                        self.assertTrue(
                            first in self.STANDARD
                            or os.path.isfile(os.path.join(REPO, first)),
                            f"README runs {first!r}, which is neither shipped "
                            "nor a standard tool")

    def test_the_scripts_are_executable_so_their_shebangs_work(self) -> None:
        for script in ("voice_tics.py", "prose_lint.py"):
            with self.subTest(script=script):
                self.assertTrue(os.access(os.path.join(REPO, script), os.X_OK))

    def test_every_repo_path_a_file_cites_exists(self) -> None:
        """A comment citing a measurement doc that never shipped is how
        a reader learns to distrust every other citation."""
        cited = re.compile(r"\b(?:docs|examples|tests)/[\w./-]+")
        for name in sorted(os.listdir(REPO)) + [
                os.path.join("tests", n) for n in os.listdir(
                    os.path.join(REPO, "tests"))]:
            if not name.endswith((".py", ".md", ".toml", ".example")):
                continue
            for ref in set(cited.findall(_read(os.path.join(REPO, name)))):
                ref = ref.rstrip(".")
                with self.subTest(file=name, ref=ref):
                    self.assertTrue(os.path.exists(os.path.join(REPO, ref)),
                                    f"{name} cites {ref}, which does not exist")


if __name__ == "__main__":
    unittest.main()
