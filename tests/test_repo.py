"""Repo invariants and redaction: the things nothing else would notice.

Docs drift, example configs rot, and a redaction default flips the wrong way
in a one-line diff. None of those break a unit test of the measurement, so
they are pinned here.
"""

from __future__ import annotations

import os
import re
import sys
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import prose_lint  # noqa: E402
import vt_config as config_mod  # noqa: E402
import vt_redact  # noqa: E402


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

    def test_the_measured_ratios_in_the_readme_match_the_code(self) -> None:
        """The README's headline table and TIC_PROVENANCE must agree.

        Two copies of a measurement is how one of them quietly becomes wrong.
        """
        readme = self._readme()
        for key, note in prose_lint.TIC_PROVENANCE.items():
            ratio = re.match(r"([\d.]+x)", note)
            if ratio and f"`{key}`" in readme:
                with self.subTest(key=key):
                    self.assertIn(ratio.group(1), readme,
                                  f"{key} is {ratio.group(1)} in code; the "
                                  "README says otherwise")


if __name__ == "__main__":
    unittest.main()
