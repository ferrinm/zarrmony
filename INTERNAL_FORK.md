# Internal Fork

This document describes the relationship between this public repository and
its internal Calico fork. It exists so that when the private fork is created
(or re-synced), there's a single source of truth for what differs and why.

## Repos

- **Public** (this repo): [`github.com/ferrinm/zarrmony`](https://github.com/ferrinm/zarrmony) — distributed as `zarrmony` on [PyPI](https://pypi.org/project/zarrmony/).
- **Private** (planned): `github.com/calico/calicolabs-zarrmony` — distributed as `calicolabs-zarrmony` on Calico-PyPI (internal index).

The public repo is the source of truth; the private repo is a downstream
overlay maintained for internal Calico deployment.

## Expected Diff at Fork Time

The private fork applies a small overlay on top of every public commit it
inherits. Files that differ:

### `pyproject.toml`

| Field            | Public                        | Private                                                   |
| ---------------- | ----------------------------- | --------------------------------------------------------- |
| `name`           | `zarrmony`                    | `calicolabs-zarrmony`                                     |
| `authors`        | Max Ferrin                    | Calico Data Engineering Team (or both)                    |
| `[project.urls]` | `github.com/ferrinm/zarrmony` | `github.com/calico/calicolabs-zarrmony`                   |
| dependencies     | as listed                     | optionally adds `calicolabs-calico-logging` if integrated |

### `.github/workflows/`

Public workflows use uv-based GitHub Actions (`astral-sh/setup-uv`,
`actions/setup-node`, etc.) and publish releases to public PyPI. Private
workflows use `calico/calico-github-actions/.github/workflows/*` wrappers
(see template) and publish to Calico-PyPI via
`release-new-version-calico-pypi.yml`. These are wholly separate files;
do not attempt to merge them.

### `.github/CODEOWNERS`

Public: `* @ferrinm`
Private: `* @calico/sweng-dev @calico/data-eng-dev`

### `.github/pull_request_template.md`

Not present in public. Private should add the template's JIRA-referencing
PR template.

### Repo name

The Calico template's bootstrap script enforces a `calicolabs-` prefix on
every repo created from it. The private repo MUST be named
`calicolabs-zarrmony`.

## Deliberate Template Deviations

These are choices where this repo intentionally diverges from
`github-template-python-library` and that the private fork should preserve
rather than "correct":

- **Tests at `tests/`, not `src/tests/`** — avoids shipping test code in
  the wheel; matches Python ecosystem norm.
- **License: Apache-2.0** — the template defaults to MIT; Calico's
  external-release policy explicitly allows either.

## Sync Procedure (Public → Private)

Manual periodic merge from public into private. Run from a clone of the
private repo:

```bash
# One-time setup:
git remote add public git@github.com:ferrinm/zarrmony.git

# Each sync:
git fetch public
git merge public/main
# Resolve conflicts in pyproject.toml, .github/workflows/, CODEOWNERS, etc.
# (The overlay is small; conflicts only happen when public touches
# overlaid files — usually pyproject deps or workflow tweaks.)
git push origin main
```

Conflicts on overlay files are useful signals — they mean the public side
changed something that has internal-side implications worth a manual look.

When sync becomes painful, escalate to a script in `scripts/sync.sh` (or
similar) and ultimately to a scheduled GitHub Action that auto-PRs the
sync. Don't pre-build that automation; wait until the manual flow actually
hurts.

## What Doesn't Sync

These are configured per-repo in GitHub Settings and never travel via git:

- **Repo secrets** (PyPI tokens, Calico-PyPI credentials, signing keys).
- **Branch protection rules**.
- **Repo settings** (Issues/Projects/Wiki enabled, default branch, merge
  strategies, etc.).
- **GitHub Actions enabled/disabled status**.
- **Webhooks and integrations**.

Configure these by hand on first creation of the private repo, then leave
them alone.
