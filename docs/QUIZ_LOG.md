# Quiz log

The real record of whether Kalyan is interview-ready, module by module. Code passing
`cargo test` means the implementation is correct. An entry here saying **PASS** means
Kalyan can defend it unaided. Those are different claims — this file tracks the
second one.

See the working agreement in [../CLAUDE.md](../CLAUDE.md) (D10 in
[DECISIONS.md](DECISIONS.md)) for the loop this log is part of: explain first, Claude
codes, run the test yourself, closed-book quiz, gate on the quiz not the test.

## Format

One entry per module, appended in order. Re-quizzes get a new entry, not an edit to
the old one — the history of getting something wrong and then right is itself useful
before an interview.

```
### <module> — <date>
**Explained first:** yes/no — one line on how it went
**Quiz result:** PASS / SHAKY / FAIL
**Questions asked:**
1. ...
2. ...
**Missed / shaky on:** ... (or "none")
**Action:** none / re-explain / re-derive snippet / re-quiz scheduled
```

## Pre-interview spot check

Before any real interview, re-open this file and cold-quiz every module marked
**SHAKY** at any point in its history, even if it later passed — shaky-then-passed
modules are the ones most likely to have quietly decayed. Two or more SHAKY/FAIL
entries across the whole log on the same *type* of question (not just the same
module) is the trigger named in [DECISIONS.md](DECISIONS.md) D10 to reconsider the
loop itself.

---

*(entries begin once week 1 starts)*
