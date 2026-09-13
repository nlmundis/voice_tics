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

THREE BUGS WORTH CARRYING FORWARD
    All are the reason the patterns look the way they do, and all failed in
    the direction that leaks rather than the direction that over-redacts.

    1. The address pattern was ASCII-only, so ``josé@vendor.io`` was never
       redacted. Worse than a clean miss: the regex still matched the ASCII
       TAIL of a non-ASCII local part, so ``maríagit@vendor.io`` matched
       ``agit@vendor.io`` and redacted to ``marí[REDACTED-EMAIL]`` —
       publishing the head of the name in cleartext while looking like the
       guard had fired.

       Widening the character class fixed that name and not the bug. Any
       allowlist of characters has a character it forgot, and each one
       reopens the same leak: an apostrophe (``mary.o'donnell@``), the
       zero-width non-joiner of Persian orthography, a combining mark outside
       the Basic Multilingual Plane (Adlam, a living script). So the local
       part is no longer a list of what an address may contain. It is
       everything back to the previous DELIMITER — whitespace or one of the
       RFC 5322 specials — so a tail-only match cannot happen whatever the
       name holds. (A match is also only allowed to START at a delimiter.
       That changes no answer, since the leftmost match already begins
       there; it keeps a long run of non-delimiters, a base64 blob say, from
       being rescanned from every offset in it.) For a redactor
       that is the safe direction: an over-match costs one over-redacted
       string, visible in the diff, where an under-match is a leak nobody
       sees.

    2. The allowlist test was a plain substring check, which failed open three
       ways, each leaving a real external address unredacted::

           someone@example.com.evil.net   domain merely CONTAINS the entry
           someone@notexample.com         suffix match with no dot boundary
           example.com@evil.net           the match was in the LOCAL part

       It is anchored to the parsed domain now. Subdomains stay allowlisted
       (``jira@mail.example.com`` under ``example.com``), which is the intent
       of the rule and the reason the comparison adds a dot rather than using
       a bare ``endswith``. Write allowlist entries bare, ``example.com``: the
       config rejects ``.example.com``, which would otherwise match nothing.

    3. The phone pattern demanded one separator between every digit group,
       so ``(555)123-4567``, a bare ``5551234567`` and every international
       rendering printed in cleartext. A phone number is now any run of ten
       to fifteen digits (the E.164 range), optionally led by ``+`` or a
       parenthesis, with up to three separators between digits. That
       over-redacts a long digit run that is not a phone, which is the
       direction this module is allowed to fail in. Seven-digit local
       numbers are not matched: at that length a phone is indistinguishable
       from an ordinary count or identifier.
"""

from __future__ import annotations

import functools
import re
import unicodedata
from typing import Pattern, Sequence

EMAIL_TOKEN = "[REDACTED-EMAIL]"
PHONE_TOKEN = "[REDACTED-PHONE]"


def _combining_mark_class() -> str:
    """Character-class body covering every combining mark in Unicode.

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

    Scanned over every plane. It once stopped at the BMP on the reasoning
    that the supplementary planes hold only historic scripts, which is false:
    Adlam, a living script for Fulani, has its marks at U+1E944 and above.
    The full scan costs about a tenth of a second, which is why the patterns
    are compiled on the first ``redact`` call rather than at import: the
    linter imports this module and never redacts.
    """
    marks = [c for c in range(0x300, 0x110000)
             if unicodedata.category(chr(c)).startswith("M")]
    parts, start, previous = [], marks[0], marks[0]
    for code in marks[1:] + [-1]:
        if code != previous + 1:
            parts.append("\\U%08x" % start if start == previous
                         else "\\U%08x-\\U%08x" % (start, previous))
            start = code
        previous = code
    return "".join(parts)


# What ends a local part: whitespace and the RFC 5322 specials, plus "@".
# Everything else, apostrophes and joiners included, belongs to the name.
_DELIMITERS = r"\s@<>()\[\]\\,;:\""


@functools.lru_cache(maxsize=None)
def email_pattern() -> Pattern[str]:
    """The compiled address pattern, built on first use.

    The local part is a negated class (see bug 1 in the module docstring) and
    the match must begin at a delimiter or at the start of the text. A domain
    label is a Unicode word character, a combining mark from any plane, a
    zero-width joiner or non-joiner, a dot or a hyphen. The trailing label is
    kept to letters and marks, so a bare host:port or a dotted version number
    such as ``pkg@1.2.3`` is still not an address; marks belong there too,
    since the second character of the IDN TLD ``.संगठन`` is one.
    """
    marks = _combining_mark_class()
    label = r"[\w%s\u200c\u200d.-]" % marks
    tld = r"(?:[^\W\d_]|[%s])" % marks
    local = r"(?<![^%s])[^%s]+" % (_DELIMITERS, _DELIMITERS)
    return re.compile(r"%s@%s+\.%s{2,}" % (local, label, tld))


# Separators a phone number is written with: space, tab, no-break space,
# parentheses, dot, hyphen and the Unicode dashes U+2010 to U+2015.
_PHONE_SEP = "[ \t\u00a0().\\-\u2010-\u2015]"
PHONE_RE = re.compile(r"(?<!\d)\+?\(?\d(?:%s{0,3}\d){9,14}(?!\d)" % _PHONE_SEP)


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

    return PHONE_RE.sub(PHONE_TOKEN, email_pattern().sub(_email_sub, text))
