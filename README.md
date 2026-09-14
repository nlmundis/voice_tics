# voice_tics

Find the phrases and structures an AI model overuses, by measuring them
against **your own writing**, not against a list of words somebody decided
sound robotic.

Then lint documents for the ones that actually separated.

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

### What the method found

One author, measured twice a month apart. Both runs take a 45-day window of
Claude Code transcripts, rank the model against the author's own turns in the
same files, and blend every model in the window.

- **2026-08-13**, the private tool this repo was extracted from: 233
  transcripts read, 134 of them dropped whole as scheduled runs, leaving
  381,274 words of model prose against 56,715 of the author's.
- **2026-09-13**, this repo: 563 transcripts read, 257 dropped as scheduled
  runs, leaving 885,083 words against 96,897.

Ratios are the model's rate over the author's. In brackets are the uses each
ratio rests on, model / author.

| Measured | 2026-08-13 | 2026-09-13 | What a wordlist does |
|---|---|---|---|
| Mean sentence length (words) | 14.00 model, 13.92 author | 13.61 model, 12.34 author | Fails builds over 16 |
| Em dash | 2.3x (7,872 / 505) | 1.8x (14,747 / 922) | Treats it as the signature tell |
| The slop lexicon (*delve*, *crucial*, …) | 0.6x (54 / 13) | 1.0x (45 / 5) | Flags them as machine-written |
| `appositive_negation` ("Y, not X") | 2.3x (1,146 / 74) | 1.7x (2,462 / 157) | Not on any list |
| `not_x_but_y` ("it's not A, it's B") | 7.7x (52 / 1) | 12.4x (113 / 1) | Not on any list |
| `method_defence` (defending the method instead of stating the finding) | 10.0x (134 / 2) | 3.8x (345 / 10) | Not on any list |

The two rows that separated most are on no public list, on both dates. Read
them with their counts: they rest on one and two of the author's uses in
August and on one and ten in September, so a single further use by the author
would halve `not_x_but_y` on either date. They are candidates the ratio
surfaced, not constants.

Two of the folk tells did not separate. The slop lexicon ran the author's way
in August and sat at parity in September, on 13 and then 5 of the author's
uses. Mean sentence length stayed inside the tool's own parity band, a quarter
either way, on both dates.

The em dash did separate: 2.3x, then 1.8x, on hundreds of the author's uses,
about as much as `appositive_negation`, which ships only as a warning. The
em-dash rule ships off anyway, because it is one author's punctuation rule
rather than a measured tic: it flags lone dashes, not a rate, and on the
September window it would have raised 803 findings across 68 of the author's
1,883 turns.

Every row moved between the dates, and the code is not why: the private tool,
run over the September transcripts, gives ratios within 7% of this repo's. The
transcripts changed, and the two windows share only two weeks. That is the
best argument this repo has for measuring your own corpus rather than
trusting anyone's table, this one included.

Sentence length is **reported and can never fail a run** in `prose_lint`. The
em-dash rule ships **off**.

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
about the **model**: the structure detectors, each with its measured ratio
recorded beside it, in version control where a change is reviewable.

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

Every rule above earned its place by measuring. This one does not, and that is
deliberate:

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
- `not_x_but_y` matches contracted closings (`it's B`, `they're B`, `but B`)
  and misses the fully uncontracted `it is not A, it is B`. Undercounting is
  the deliberate direction throughout: a missed tic costs one unflagged
  sentence, a false one costs trust in every other row. Widening it would also
  invalidate the measurements recorded beside it.
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
