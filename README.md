# voice_tics

Find the phrases and structures an AI model overuses, by measuring them
against **your own writing**, not against a list of words somebody decided
sound robotic.

Then lint documents for the ones your own measurement separates.

```bash
python3 voice_tics.py --days 30                 # what does the model repeat?
python3 prose_lint.py draft.md                  # does this draft do it?
```

---

## The argument

"Phrases an AI overuses" is a folk category. Every public AI-slop detector is
a hand-written list — *delve*, *crucial*, *tapestry*, the em dash — and a
hand-written list is an assertion with no source.

It is also **wrong for a specific, predictable group of people**: anyone whose
natural register overlaps it. Academics, lawyers, technical writers, anyone
formally trained. Run a wordlist detector over their drafts and it flags their
genuine voice as machine-written, and then they edit it out.

The author this tool was built for writes *However*, *It should be noted
that*, *Thus*. After a wordlist linter had been at his drafts, his summary was:

> we essentially made me sound simpler so I don't sound like AI

That is the failure this repo exists to prevent. The fix is not a better list.
It is a **baseline**.

## The method

Every Claude Code session transcript holds both sides of a conversation: the
model's prose and yours. Same medium, same topics, same weeks, same technical
vocabulary. So rank the model's n-grams, openers and structures against *your*
rates in the same files.

A phrase at 40x your rate is a tic. A phrase you both use at similar rates is
just the subject matter.

Nothing here is a judgement call. Every number is a count over text already on
disk: no grader, no model in the loop. The output ranks candidates for you;
it does not decide what a tic is, and it cannot.

### Why there is no results table

Earlier versions of this README published ratios measured on the reference
author's transcripts. They are withdrawn, because reviews kept finding more of
the model's own writing counted as the author's.

A transcript records who delivered each message to the session, not who wrote
it. Claude Code flags two kinds of user record it wrote itself: slash-command
expansions and similar (`isMeta`), and the summaries it writes when it compacts
a conversation (`isCompactSummary`). This tool drops both, and drops records it
recognises by their content too: harness notices, hook output, messages another
session's model sent, and whole scheduled-run sessions. Whatever it does not
recognise counts as the author. On the reference author's 45-day window ending
2026-09-13, after every exclusion the tool makes, about a quarter of the words
left as "the author" were prompts the model had drafted for sessions the author
launched, and nearly a fifth more were in turns whose opening text recurred at
the start of five or more sessions, which the author may or may not have typed
each time. Each correction moved the headline ratios, some of them several
times over.

So the method stands and the published numbers do not. Measure your own
corpus, and read a transcript baseline as the most it could be yours, not as
what is: the report counts what it recognised as someone else's writing and
dropped, and cannot count what it did not recognise. Where you can, use
`--baseline-from` on documents you wrote unaided (below).

Sentence length is **reported and can never fail a run** in `prose_lint`: a
length threshold measures register, a linter runs on documents, and a chat
baseline says nothing about the register of your documents. The same objection
reaches every default rule, since each was chosen on a chat baseline and runs
on documents; `--baseline-from` is the better test of a rule you mean to
enforce. The em-dash rule
ships **off**: it is one author's rule about lone dashes, not a measurement.

## Two baselines, and what each one costs

The transcript baseline has a real weakness, and it is worth stating plainly:
instructions you type to an agent are short and imperative, so the
constructions you reach for in *long-form prose* barely occur in it. No
threshold over that corpus will ever surface them.

So you can point the tool at your own documents instead:

```bash
python3 voice_tics.py --baseline-from 'posts/**/*.md'
```

|  | Transcripts (default) | Your documents |
|---|---|---|
| Matched on | medium, topic, week | register, author |
| Not matched on | **register** | topic, date |
| Use it when | you want to know how the model writes *to you, now* | you are linting **documents** |

Neither is the real answer; they answer different questions, and the report
says which one it used.

The same mechanism takes a corpus that is not yours at all. Point it at prose
you want to sound more like and the ratios become distance-from-target rather
than tic-over-baseline: the same arithmetic, a different claim. The tool
cannot tell the two apart and does not try, so be deliberate about which you
are doing.

> **The one real trap.** A sample corpus is only a baseline if *you* wrote it.
> Point it at documents an assistant drafted or edited and the model's prose is
> on both sides of the ratio: every tic divides by itself and lands near 1.0,
> and the report says "nothing here", silently, in the direction that passes.
> The tool counts how many constructions sit within a quarter of parity and
> warns when most of them do, but that guard is deliberately reluctant, and it
> does not see partial contamination at all: on a synthetic sweep it stayed
> quiet up to 70% model-written paragraphs, while a tic that read 593x against
> clean samples read 3.2x at 25% contamination. A quiet guard is not evidence
> the samples are clean. Narrow the glob to prose you wrote unaided.

## Install

Nothing to install. Stdlib only, Python 3.9+.

```bash
git clone https://github.com/nlmundis/voice_tics
cd voice_tics
python3 voice_tics.py --days 30
```

The one exception is reading a config file on Python 3.10 or earlier, where
the standard library has no TOML parser: `pip install tomli`. Without a config
file nothing is imported beyond the standard library, on any supported
version.

## Configure

Optional. With no config file the tool reads the standard Claude Code
transcript location, ranks against your own turns, and lints with the two
default error tics.

```bash
cp voice_tics.toml.example voice_tics.toml
```

The file is read from the working directory only, never from a parent. Run
from a subdirectory, the tool uses its defaults and prints a note naming the
parent config it did not read; pass `--config` to use that file.

What lives in the config is what is a fact about **you**: your signature
phrases, where your writing is, which detectors may fail your build, the
domains you do not need redacted. What lives in the code is what is a claim
about the **model**: the structure detectors, in version control where a
change is reviewable. The default lint rules were chosen on the measurements
this README has withdrawn, so treat them as a starting point, not a finding.

Unknown keys are an error, not a shrug. A misspelled key that gets silently
ignored is a setting you believe is in force and is not.

## Linting

```bash
python3 prose_lint.py draft.md              # exit 1 on any error
python3 prose_lint.py --strict draft.md     # warnings fail too
cat draft.md | python3 prose_lint.py -
python3 prose_lint.py --json draft.md
```

Exit codes: `0` clean or warnings only, `1` any error (or any warning under
`--strict`), `2` usage or config error.

### Banning a phrase you are sick of

The rules above were chosen by measuring, on figures since withdrawn (see
above). This one is your own list, with no measurement behind it:

```toml
[lint]
banned = [
  { phrase = "load bearing", instead = "does real work, carries weight" },
  { phrase = "blast radius", instead = "what it touches", tier = "warning" },
]
```

Nothing is banned by default. This is you saying you do not want to read a
phrase, which needs no ratio behind it, and it is kept in its own config key
and its own finding type so a preference is never later read back as evidence.

The phrase is **literal, not a regex** — you are banning something you are
tired of, not authoring a pattern, and making you escape it is a way to be
wrong quietly (`C++` as a regex is a repetition error). Matching ignores case,
and the gaps between words match any run of spaces or hyphens, so
`load bearing` also catches `load-bearing`.

A tic key that no longer exists upstream fails the **whole run** before any
file is read, rather than the first file that reaches it. A rule that quietly
stopped being checked is worse than a crash, because the build goes green.

## Privacy

The structure and signature tables carry only counts against named patterns.
The phrase and opener tables reconstruct fragments of your transcripts, so
**they do not print by default**. `--include-source-text` opts in, and the
gate applies to `--json` identically.

That gate is the protection. Opt-in output does pass through a redaction pass,
but tokenisation has already stripped the `@`, dots and digits that email and
phone patterns key on, so today it cannot fire on phrase output at all. It is
a belt kept for the day the tokeniser widens. **Treat opt-in output as quoting
your transcripts.**

## Limits

- POSIX. Not tested on Windows.
- Transcripts are read in Claude Code's `.jsonl` format. Other agent CLIs
  would need a different reader; `read_turns` is the seam.
- Sample files are read as UTF-8 markdown or plain text. One unreadable file
  is counted and skipped, never fatal.
- The transcript baseline is only as clean as the harness's own flags and the
  markers this tool knows. Records flagged `isMeta` or `isCompactSummary` and
  harness notices are dropped and counted; cross-session messages are cut out,
  or dropped with their record when a tag is left over, and counted. A user
  record it does not recognise is counted as yours whoever wrote it: a session
  prompt a model drafted and you launched, a template you run repeatedly, text
  you pasted. On the reference author's 45-day window ending 2026-09-13,
  spawned-session prompts alone were 10,643 of the 44,314 author words the
  report kept.
- `not_x_but_y` matches contracted closings (`it's B`, `they're B`, `but B`)
  and misses the fully uncontracted `it is not A, it is B`. Undercounting is
  the deliberate direction throughout: a missed tic costs one unflagged
  sentence, a false one costs trust in every other row. Widening it would also
  change every ratio anyone has measured with it.
- The `rule_of_three_no_oxford` detector is low precision and is labelled as
  such in the output. It mostly catches clause coordination.
- `prose_lint` misses a table row that opens with an HTML tag and ends with a
  pipe (`<b>| … |`): the tag hides the row from the table pass, and the body
  pass blanks it once the tag is stripped. Telling it apart needs markup
  context the line-based walk does not model. The miss is toward
  undercounting.
- Fenced code (backticks or tildes, at any indentation, LF or CRLF) is
  skipped. An *indented* code block, four spaces after a blank line with no
  fence, is linted as prose: inside a list item the same indentation is an
  ordinary continuation paragraph, and telling them apart needs list context
  the line-based walk does not model.
- Table rows are recognised by a leading pipe. A GFM table written without
  outer pipes is linted as prose, so with the em-dash rule on, a glossary row
  like `AGC — Atlassian Government Cloud | yes` in such a table is reported.
- With the em-dash rule on, a lone dash in a table cell is excused when the
  text before it looks like a label (up to sixty characters, no sentence
  punctuation). `Read the SOW first — it changes the severity` in a cell has
  that shape, and counting dashes cannot tell a term from a clause. A matched
  pair in a cell is never reported.
- Chat-register tics (`let_me`, `i_should_note`) are measured but are useless
  for linting documents, because they cannot appear in one. `prose_lint` does
  not carry them: a linter that did would report clean runs as a property of
  the register rather than of the prose.

## Development

```bash
make test       # the unit suite; standard library only
make check      # the unit suite, then the mutation gate
```

`make check` also runs [mutt_check](https://github.com/nlmundis/mutt_check),
a curated mutation gate that reverts each design decision the code rests on
and requires the tests to go red. It is a separate tool, so install it first:

```bash
pip install "mutt_check @ git+https://github.com/nlmundis/mutt_check@v0.0.2"
```

Without it, `make check` stops with that instruction rather than skipping the
gate: a check that quietly did not run is a green build nobody earned. To use
a checkout instead of an install, `make check MUTT_CHECK="python3 path/to/mutt_check.py"`.

## License

MIT.
