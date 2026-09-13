"""Email and phone redaction for anything this tool prints from transcripts.

WHY IT IS HERE AT ALL
    The structure and signature tables carry only counts against named
    patterns, so they are safe by construction. The phrase and opener tables
    are reconstructed fragments of the transcripts themselves, and a working
    transcript names clients, colleagues and addresses. Those tables are
    opt-in for that reason (``--include-source-text``); this module is the
    second layer under the opt-in, not the first.

    Be clear-eyed about how much it buys: by the time an n-gram reaches the
    output, tokenisation has already stripped the ``@``, the dots and the
    digits that these patterns key on, so on today's ``WORD_RE`` redaction
    cannot fire on phrase output at all. It is a belt kept for the day that
    tokeniser widens. Treat opt-in output as quoting the transcripts.

THE DEFAULT IS TO REDACT
    ``keep_domains`` is empty unless configured, so every address is redacted
    out of the box. That is the opposite of the private original this was
    extracted from, which allowlisted the author's own employer. An allowlist
    that ships with somebody else's domains in it is a leak with a default
    value, and the person who would be hurt by it is never the person who
    installed the tool.

TWO BUGS WORTH CARRYING FORWARD
    Both are the reason the patterns look the way they do, and both failed in
    the direction that leaks rather than the direction that over-redacts.

    1. The address pattern was ASCII-only, so ``josé@vendor.io`` was never
       redacted. Worse than a clean miss: the regex still matched the ASCII
       TAIL of a non-ASCII local part, so ``maríagit@vendor.io`` matched
       ``agit@vendor.io`` and redacted to ``marí[REDACTED-EMAIL]`` —
       publishing the head of the name in cleartext while looking like the
       guard had fired. Widened rather than replaced: every character the old
       pattern accepted is still accepted, so this can only ever match MORE.
       For a redactor that is the safe direction — an over-match costs one
       over-redacted string, visible in the diff, where an under-match is a
       leak nobody sees.

    2. The allowlist test was a plain substring check, which failed open three
       ways, each leaving a real external address unredacted::

           someone@example.com.evil.net   domain merely CONTAINS the entry
           someone@notexample.com         suffix match with no dot boundary
           example.com@evil.net           the match was in the LOCAL part

       It is anchored to the parsed domain now. Subdomains stay allowlisted
       (``jira@mail.example.com`` under ``example.com``), which is the intent
       of the rule and the reason for the leading dot rather than a bare
       ``endswith``.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Sequence, Tuple

EMAIL_TOKEN = "[REDACTED-EMAIL]"
PHONE_TOKEN = "[REDACTED-PHONE]"


def _combining_mark_class() -> str:
    """Character-class body covering every combining mark in the BMP.

    Returns:
        A string of ``\\uXXXX`` escapes and ranges, safe to splice between the
        brackets of a character class. Never empty.

    Computed rather than written out because Python's ``re`` has no ``\\p{M}``
    and ``\\w`` EXCLUDES combining marks — they are not ``str.isalnum()``.
    Without them a decomposed name fails to match at all: NFD "josé" is
    ``jose`` + U+0301, and every Devanagari or Arabic mailbox carries vowel
    signs or points in the middle of the word. A hardcoded range list would go
    stale against Python's Unicode version somewhere nobody looks; this costs
    about five milliseconds at import and cannot.

    Scanned over the BMP only. The supplementary planes carry marks for
    historic scripts and musical notation, which is not where mailbox names
    live, and scanning them costs fifteen times as much.
    """
    marks = [c for c in range(0x300, 0x10000)
             if unicodedata.category(chr(c)).startswith("M")]
    parts, start, previous = [], marks[0], marks[0]
    for code in marks[1:] + [-1]:
        if code != previous + 1:
            parts.append("\\u%04x" % start if start == previous
                         else "\\u%04x-\\u%04x" % (start, previous))
            start = code
        previous = code
    return "".join(parts)


_MARKS = _combining_mark_class()
# A local part or a domain label. ``\w`` is Unicode-aware for str patterns, so
# it already spans every script's letters and digits; the marks are what it
# misses.
_LOCAL_CHAR = r"[\w%s.%%+-]" % _MARKS
_LABEL_CHAR = r"[\w%s.-]" % _MARKS
# The trailing label, kept to letters — ``[^\W\d_]`` is ``\w`` minus digits and
# underscore, i.e. a Unicode letter — so a bare host:port or a dotted version
# number is still not an address. Marks belong here too: the second character
# of the IDN TLD ``.संगठन`` is one.
_TLD_CHAR = r"(?:[^\W\d_]|[%s])" % _MARKS

EMAIL_RE = re.compile(r"%s+@%s+\.%s{2,}" % (_LOCAL_CHAR, _LABEL_CHAR, _TLD_CHAR))
PHONE_RE = re.compile(r"\(?\b\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b")


def is_allowlisted(email: str, keep_domains: Sequence[str]) -> bool:
    """True when the address's DOMAIN is one the caller chose not to redact.

    Anchored to the parsed domain, never a substring of the whole address; see
    the three fail-open cases in the module docstring. A subdomain of an
    allowlisted domain is allowlisted too.

    Args:
        email: One address, as matched by ``EMAIL_RE``.
        keep_domains: Domains to leave in cleartext. Empty means redact
            everything, which is the default everywhere in this tool.
    """
    domain = email.lower().strip().rpartition("@")[2]
    return any(domain == d.lower() or domain.endswith("." + d.lower())
               for d in keep_domains)


def redact(text: str, keep_domains: Sequence[str] = ()) -> str:
    """``text`` with every address and phone number replaced by a token.

    Addresses whose domain appears in ``keep_domains`` are left alone; with
    the default empty allowlist, every address is redacted.

    Args:
        text: Any string bound for the terminal or for ``--json``.
        keep_domains: Domains to leave in cleartext, from
            ``[output] keep_domains`` in the config file.

    Returns:
        The same string with ``[REDACTED-EMAIL]`` and ``[REDACTED-PHONE]``
        substituted. Never raises; a string with nothing to redact comes back
        unchanged.
    """
    def _email_sub(m: "re.Match[str]") -> str:
        address = m.group(0)
        return address if is_allowlisted(address, keep_domains) else EMAIL_TOKEN

    return PHONE_RE.sub(PHONE_TOKEN, EMAIL_RE.sub(_email_sub, text))
