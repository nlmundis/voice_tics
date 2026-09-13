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

Three results from a 45-day reference corpus, all of which a wordlist would
have got backwards:

| Measured | Result | What a wordlist does |
|---|---|---|
| Mean sentence length | 13.89 model vs **13.85** author | Fails builds over 16 |
| Em dash | **1.0x**, identical rates | Treats it as the signature tell |
| The slop lexicon (*delve*, *crucial*, …) | **0.6x**: the *author* used them more | Flags them as machine-written |
| `not_x_but_y` ("it's not A, it's B") | **7.7x** | Not on any list |
| `method_defence` (defending the method instead of stating the finding) | **10.0x** | Not on any list |

The two that separated are not in any public list. The three everyone gates on
did not separate at all. That asymmetry is the whole point.

Sentence length is therefore **reported and can never fail a run** in
`prose_lint`. The em-dash rule ships **off**.

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
> warns when most of them do, but that guard is deliberately reluctant. Narrow
> the glob to prose you wrote unaided.

## Install

Nothing to install. Stdlib only, Python 3.9+.

```bash
git clone https://github.com/nlmundis/voice_tics
cd voice_tics
python3 voice_tics.py --days 30
```

On Python 3.10 and earlier, `pip install tomli` if you want a config file.

## Configure

Optional. With no config file the tool reads the standard Claude Code
transcript location, ranks against your own turns, and lints with the two
default error tics.

```bash
cp voice_tics.toml.example voice_tics.toml
```

What lives in the config is what is a fact about **you**: your signature
phrases, where your writing is, which detectors may fail your build, the
domains you do not need redacted. What lives in the code is what is a claim
about the **model**: the structure detectors, each with its measured ratio
recorded beside it, in version control where a change is reviewable.

Unknown keys are an error, not a shrug. A misspelled key that gets silently
ignored is a setting you believe is in force and is not.

## Linting

```bash
prose_lint draft.md                 # exit 1 on any error
prose_lint --strict draft.md        # warnings fail too
cat draft.md | prose_lint -
prose_lint --json draft.md
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
  invalidate the 7.7x measurement recorded beside it.
- The `rule_of_three_no_oxford` detector is low precision and is labelled as
  such in the output. It mostly catches clause coordination.
- Chat-register tics (`let_me`, `i_should_note`) are measured but are useless
  for linting documents, because they cannot appear in one. `prose_lint` does
  not carry them: a linter that did would report clean runs as a property of
  the register rather than of the prose.

## Development

```bash
make check
```

Runs the unit suite, then `mutt_check` against it: a curated mutation gate
that reverts each design decision the code rests on and requires the tests
to go red. See [mutt_check](https://github.com/nlmundis/mutt_check).

## License

MIT.
