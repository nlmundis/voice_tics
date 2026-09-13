"""Repo invariants and redaction: the things nothing else would notice.

Docs drift, example configs rot, and a redaction default flips the wrong way
in a one-line diff. None of those break a unit test of the measurement, so
they are pinned here.
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
import tempfile
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
