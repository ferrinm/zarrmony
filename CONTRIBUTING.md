# Contributing to zarrmony

## Never name internal datasets

**This repository is public. The datasets zarrmony is developed and tested
against are not.**

Absolute paths to lab shares, sample and accession IDs, trial numbers, marker
panels and collaborator names leak experimental design — what is being
stained, in what model, for whom — even when the surrounding text is purely
technical about chunk shapes and dtypes. A dataset directory name that encodes
a cell line plus an antibody panel plus a trial number describes an
unpublished experiment.

This applies to **issues and pull request descriptions** exactly as much as to
committed files. GitHub retains the full edit history of every issue and
comment and serves it over the public API, so redacting after the fact does
not undo the disclosure. Get it right the first time.

### Use placeholders

| Instead of                          | Write                                     |
| ----------------------------------- | ----------------------------------------- |
| an absolute path to a lab share     | `/mnt/readonly/<dataset>` or `$SRC`       |
| a sidecar named after a sample      | `metadata_<dataset>.json`                 |
| a sample, accession or trial number | `<dataset>`, or `slide A` / `B` / `C`     |
| a collaborator or lab name          | "the internal share", "an external group" |

Keep the technical facts — array shapes, dtypes, voxel spacings, channel
_filter_ names (`DAPI`, `FITC`, `Cy5`), instrument models, timings. Those are
what make an issue useful to work from, and none of them identify a study on
their own. It is the combination of sample identity and biological target that
has to stay out.

Where a runbook genuinely needs a real path to be executable, take it from an
environment variable set by the person running it, and say in the document
that the value is tracked internally.

### The pre-commit hook

`scripts/check_no_internal_paths.py` runs as a pre-commit hook and blocks known
share prefixes and identifier shapes. Install hooks once with:

```bash
uv run pre-commit install
```

The patterns committed to the repo are deliberately **structural** — the shape
of an internal mount (`/Volumes/<share>-ro`), a cluster path, a trial number.
They never name a lab, collaborator or study, because a blocklist naming those
would publish them itself.

Site-specific names go in `.internal-patterns` at the repo root, which is
gitignored. One regex per line, matched case-insensitively, `#` for comments:

```
PD_example\w*
Some[ _]?Lab
```

The file is absent from a fresh clone and the hook works without it — create
one locally so the guard also catches your site's names in prose, not just in
paths.

It is a backstop, not a substitute for judgement: it cannot see issue or PR
text, and it only knows the patterns it is given. In particular it cannot
recognise a bare dataset name that encodes a sample plus a marker panel — add
those to `.internal-patterns`.

For a genuine false positive, append `# allow-internal-path` to the line.
Keep those rare; the marker exists so a reviewer has to look at the line.

## Development

```bash
uv sync --extra dev
uv run pytest
uv run pre-commit run --all-files
```

See [`docs/writing-a-reader-plugin.md`](docs/writing-a-reader-plugin.md) for
adding support for a new format, and [`docs/adr/`](docs/adr/) for the
architectural decisions behind the current design.
