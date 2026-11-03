# Documentation claim auditor

Offline static checking of documentation promises against Python call evidence.
Source files are parsed without executing them. No runtime dependencies are needed.
Requires Python 3.9 or newer.

```sh
python3 -m pip install .
speech-emotion-recognition-claim-audit /path/to/project
python3 -m speech_emotion_recognition_claim_audit /path/to/project --json
python3 -m speech_emotion_recognition_claim_audit --version
```

The auditor reads the root README.md and Markdown files beneath any docs
component. Fenced blocks are excluded; inline code counts. Matching folds case,
expands negative contractions, replaces hyphens, and collapses whitespace within
paragraphs. Negative spans are removed before affirmative matching.

Evidence is module local. Recognized operations are named calls, specialized
class construction paired with prediction, and a bound library factory call.
Imports alone and unrelated audio or text operations do not count. This static
heuristic cannot prove runtime correctness, data flow, or model quality.

Symlinks are skipped. Source under fixtures, docs, or examples is excluded.
Test evidence is reported separately and cannot establish production support.
Invalid syntax supplies no evidence. A promise without source is flagged;
a denial without source passes.

```text
documented-speech-emotion-recognition-claim-false   HIGH (MEDIUM for tests only)
documented-no-speech-emotion-recognition-claim-false   HIGH
```

Each family produces at most one finding. JSON includes verdict, findings with
id, severity and message, evidence flags, and inspected paths. The verdict is
FAIL whenever findings exist, independently of the failure threshold.

Options: --json emits structured output; --verbose lists evidence and paths;
--ignore excludes a relative path prefix and may be repeated; --fail-on accepts
LOW, MEDIUM, or HIGH (default HIGH). --version prints 0.1.0.
Exit status is 0 below threshold, 2 at or above threshold, and 1 for usage or I/O
errors. The audit reads local files without networking or modifying the target.

```sh
python3 -m unittest discover
python3 -m build
```

MIT licensed. Both wheel and source archives are supported.
