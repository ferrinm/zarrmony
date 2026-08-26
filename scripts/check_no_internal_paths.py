#!/usr/bin/env python3
"""Block internal share paths and dataset identifiers from reaching a public commit.

zarrmony is a public repo; the datasets it is tested against are not. Absolute
paths to lab shares, sample/accession IDs and collaborator names leak
experimental design (what is being stained, in what model, for whom) even when
the surrounding text is purely technical.

Run as a pre-commit hook over staged files. Use placeholders instead:
``/mnt/readonly/<dataset>``, ``$SRC``, ``metadata_<dataset>.json``.

Add a trailing ``# allow-internal-path`` comment on a line that is a genuine
false positive. Keep those rare — the point is that the reviewer has to look.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ALLOW_MARKER = "allow-internal-path"

# Patterns kept here are *structural* — they describe the shape of an internal
# path, never the name of a specific lab, collaborator or study. A blocklist
# naming the things it protects would publish them itself, which is exactly the
# disclosure this hook exists to prevent.
#
# Each entry is (compiled pattern, what to write instead).
RULES: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"/Volumes/[A-Za-z0-9._-]*-ro\b"),
        "internal read-only mount; use /mnt/readonly/<dataset>",
    ),
    (
        re.compile(r"/data/microscopy/"),
        "internal cluster share; use $SRC or /mnt/readonly/<dataset>",
    ),
    (
        # No leading \b: these show up as `..._Trial#1234`, and `_` is a word
        # character, so a word boundary would never match there.
        re.compile(r"Trial[#_ ]?\d{3,}\b", re.IGNORECASE),
        "trial number identifying a specific experiment",
    ),
]

# Site-specific names (lab, collaborator, instrument share) belong in an
# untracked file so they never reach the public repo. One regex per line;
# blank lines and `#` comments ignored. Absent by default — see CONTRIBUTING.md.
LOCAL_PATTERNS = Path(__file__).resolve().parent.parent / ".internal-patterns"


def load_rules() -> list[tuple[re.Pattern[str], str]]:
    """``RULES`` plus any site-local patterns, if the untracked file exists."""
    rules = list(RULES)
    if not LOCAL_PATTERNS.exists():
        return rules
    for lineno, raw in enumerate(
        LOCAL_PATTERNS.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            rules.append((re.compile(line, re.IGNORECASE), "site-local identifier"))
        except re.error as exc:
            print(
                f"warning: {LOCAL_PATTERNS.name}:{lineno}: bad regex ({exc}); skipped",
                file=sys.stderr,
            )
    return rules


# What this cannot catch: a bare dataset name carrying a sample ID and a marker
# panel (`<line>_<marker>-<marker>_<marker>`) with no path or trial number
# around it. There is no safe general pattern for that — a biology-term
# blocklist is unbounded and would fire on legitimate channel names. Add such
# names to the local pattern file; CONTRIBUTING.md and review cover the rest.

# File suffixes worth scanning. Everything else (lockfiles, images, binaries)
# is skipped — uv.lock in particular is huge and full of hashes.
SCANNED_SUFFIXES = {
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
    ".json",
    ".org",
    ".cfg",
    ".ini",
    ".sh",
}


def scan(path: Path, rules: list[tuple[re.Pattern[str], str]]) -> list[str]:
    """Return one message per offending line in ``path``."""
    if path.suffix.lower() not in SCANNED_SUFFIXES:
        return []
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []

    problems: list[str] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if ALLOW_MARKER in line:
            continue
        for pattern, advice in rules:
            match = pattern.search(line)
            if match:
                problems.append(f"{path}:{lineno}: {match.group(0)!r} — {advice}")
                break
    return problems


def main(argv: list[str]) -> int:
    rules = load_rules()
    problems: list[str] = []
    for name in argv:
        problems.extend(scan(Path(name), rules))

    if not problems:
        return 0

    print("Internal dataset paths or identifiers found in staged changes:\n")
    for problem in problems:
        print(f"  {problem}")
    print(
        "\nThis repo is public. Replace these with placeholders, or append "
        f"'# {ALLOW_MARKER}' to the line if it is genuinely safe."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
