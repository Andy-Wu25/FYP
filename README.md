# Code Similarity Tool

A semantic code similarity detector that compares the code in your working change against two corpora: a **private** organisation index of your team's own repositories, and a **public** index of open-source repositories you have explicitly loaded. We report nearest-neighbour matches by cosine distance over Tree-sitter-parsed declaration embeddings.

It runs as a CLI for day-to-day developer use, as a Git hook for automatic checks before commit or push, and as a FastAPI webhook server that posts findings as review comments on GitHub pull requests.

## Why

Code review can catch obvious copy-paste duplication, but it cannot reasonably detect a function that has been refactored, renamed, or paraphrased from another part of the codebase or from a public source. This tool answers a single question on every change: *is anything in my working set semantically similar to code that has been seen before?* It is intended for engineers who want a fast feedback loop on accidental duplication and for teams that need to flag potential license-relevant matches against open-source corpora.

## Features

- **Two indexes, one store.** Private (organisation-scoped) and public (operator-loaded) collections live in the same ChromaDB instance but are queried independently.
- **Tree-sitter parsing.** Functions, methods, and constructors are extracted as discrete elements; files in languages without a grammar fall back to whole-file embedding.
- **vLLM embeddings.** OpenAI-compatible `/v1/embeddings` endpoint, default model `Octen/Octen-Embedding-8B`. Long elements are handled either by chunking-and-averaging or by truncation, configurable per index.
- **Three query scopes.** `staged` (the Git index), `files` (explicit paths), or `repo` (the entire working tree).
- **License filtering on public results** by SPDX identifier, with fuzzy suggestions for misspelled keywords.
- **Permalinks.** Public-corpus matches link directly to the matched file and line range on GitHub at the indexed commit.
- **Three automation surfaces.** A coloured terminal dashboard with an optional curses browser, a JSON output mode for CI, and a GitHub App webhook server that comments on pull requests.

## Quick Start

```bash
git clone https://github.com/Andy-Wu25/FYP.git
cd FYP
python3.11 -m venv venv && source venv/bin/activate
pip install -e .

# Index your repository once, then check a staged change
code-sim-update
git add src/parser.py
code-sim-check-private
```

`code-sim-check-private` prints a framed dashboard ranked by cosine distance, with hits in red below 0.15 (strong near-duplicate), yellow below 0.35 (moderate), and plain otherwise.

## Requirements

- Python 3.11+
- Git on `PATH`
- A vLLM server exposing the OpenAI-compatible `/v1/embeddings` route, with the configured model loaded (default: `Octen/Octen-Embedding-8B`)

The runtime dependency set is intentionally small:

| Package | Version | Role |
|---|---|---|
| `chromadb` | `==1.1.1` | Vector store and HNSW index |
| `tree-sitter-language-pack` | `==0.9.1` | Parser grammars |
| `identify` | `>=2.6.0` | File-type detection fallback |
| `requests` | `>=2.31.0` | HTTP client for vLLM and GitHub |

## Installation

```bash
pip install -e .              # core CLI
pip install -e '.[bot]'       # + FastAPI / uvicorn for the PR bot
pip install -e '.[dev]'       # + pytest
```

Twenty-four shell entry points are registered after install. Run `code-sim-config` to print the resolved configuration (database path, organisation, model, limits) annotated with the source of every value — this is the fastest way to confirm the install succeeded.

## Commands

| Group | Command | Purpose |
|---|---|---|
| **Check** | `code-sim-check-private` | Compare staged code against the private index |
| | `code-sim-check-public` | Compare against the public index, optional `--license` filter |
| | `code-sim-check-self` | Restrict to matches from the current repository only |
| | `code-sim-check` | Alias for `check-private` |
| **Index** | `code-sim-update` | Sync the current repo into the private index |
| | `code-sim-index` | Same as `update`, or `--url <github>` for public |
| | `code-sim-index-public <url>` | Operator-mode public indexing with license auto-detection |
| | `code-sim-remove-public <url>` | Remove all indexed entries for a public repo |
| **Tooling** | `code-sim-config` | Print resolved configuration |
| | `code-sim-stats` | Database stats dashboard with per-language and license breakdowns |
| | `code-sim-ignore` | Create a `.code-simignore` template |
| | `code-sim-install-hook` / `code-sim-delete-hook` | Manage Git hooks |
| **Snapshots** | `code-sim-snapshot-{save,list,load,delete}` | Freeze and restore the entire ChromaDB directory |
| **Bot** | `code-sim-bot-serve` | Run the FastAPI webhook server for PR review comments |
| **Evaluation** | `code-sim-batch-eval` | Run the full evaluation matrix across snapshots |
| | `code-sim-batch-index` | Batch-index a list of repositories into the public collection |
| | `code-sim-eval`, `code-sim-eval-private`, `code-sim-eval-public` | Evaluation helpers |

All check commands accept `--scope staged|files|repo`, `--top-k N`, `--max-distance X`, `--min-lines N`, `--no-color`, `--no-interactive`, and `--json FILE`.

## Architecture

```
code_similarity_tool/
├── core/        # Tree-sitter parsing, language detection, source filtering, .code-simignore matcher, runtime context
├── infra/       # CodeVectorStore (ChromaDB wrapper), EmbeddingClient (vLLM), public_links (GitHub permalinks)
├── indexing/    # Private and public write paths, update, removal
├── checking/    # Private, public, and self read paths, shared formatters, optional curses browser
├── tools/       # Auxiliary CLIs: config, stats, snapshots, evaluation suite
└── bot/         # FastAPI webhook server, GitHub App client, PR analyser, comment formatter
```

The package is split by responsibility rather than by command. Every CLI entry point lives in `[project.scripts]` in `pyproject.toml` and dispatches to a `main()` function in the appropriate subpackage.

### Data layout

All persistent state lives under `CODE_SIM_DB_PATH` (default `code_similarity_tool/.code-sim/shared-db`) and consists of two ChromaDB collections (`project_code_private` and `project_code_public`) plus two `_embedding_config_*.json` files that record the model, chunking mode, and character budget active at index time. Query-time embeddings are produced with the same parameters, so a model or pipeline change surfaces as a warning rather than as silent retrieval failure.

Private data is partitioned by `org_id` and `repo_id`; public data is partitioned by `public_source_id` derived from `url + commit`. Multiple repositories on the same host can share a private query scope by exporting the same `CODE_SIM_ORG_ID` and `CODE_SIM_DB_PATH`.

## Configuration

Every configurable parameter has an environment variable with a sensible default; the most commonly tuned ones are below. Run `code-sim-config` for the full resolved view at any time.

| Variable | Default | Purpose |
|---|---|---|
| `CODE_SIM_ORG_ID` | `my-org` | Scopes the private index across repositories |
| `CODE_SIM_DB_PATH` | `code_similarity_tool/.code-sim/shared-db` | On-disk location of the ChromaDB store |
| `VLLM_INPUT_PREFIX` | `\n` | Prepended to every embedding input for consistency with model training |
| `CODE_SIM_MAX_FILE_BYTES` | `250000` | File-size ceiling for indexing and queries |
| `VLLM_BASE_URL` | `http://127.0.0.1:8000` | Embedding server endpoint |
| `VLLM_MODEL` | `Octen/Octen-Embedding-8B` | Embedding model identifier |
| `VLLM_LONG_TEXT_MODE` | `chunk` | `chunk` (split + average) or `truncate` |
| `VLLM_CHUNK_OVERLAP` | `512` | Chunk overlap in characters for `chunk` mode |

## Automation

### Git hook

```bash
code-sim-install-hook                    # default: pre-push runs code-sim-update
code-sim-install-hook --stage pre-commit # alternative: pre-commit runs code-sim-check
code-sim-delete-hook                     # surgical removal of the code-sim block
```

The installer is idempotent and leaves any unrelated hook content intact.

### GitHub pull request bot

A FastAPI webhook server posts similarity findings directly as PR review comments. It requires a GitHub App with `pull_requests: write` and `contents: read`, and three secrets in the environment:

```bash
export GITHUB_APP_ID=...
export GITHUB_APP_PRIVATE_KEY="$(cat /path/to/private-key.pem)"
export GITHUB_WEBHOOK_SECRET=...

code-sim-bot-serve --host 0.0.0.0 --port 3000
```

Tuning flags (`--max-distance`, `--top-k`, `--min-lines`, `--allowed-orgs`, `--max-files`, `--max-elements`) fall back to `GITHUB_BOT_*` environment variables when the command line is silent. On every pull request event the bot verifies the HMAC signature, downloads the changed files through the GitHub API, runs the same extract–embed–query–filter pipeline that the CLI uses, and posts (or edits in place) a single summary review comment.

A live sandbox installation is configured at https://github.com/Andy-Wu25/code-sim-bot-test for end-to-end demonstration.

## Workflows

**Solo developer.** Run `code-sim-update` after a meaningful change so the local index stays current; run `code-sim-check-private` before committing if you suspect duplication.

**Team / organisation.** Export a shared `CODE_SIM_ORG_ID` and `CODE_SIM_DB_PATH` in every contributor's shell profile, install the pre-push hook in each repository, and let the GitHub bot handle pull request reviews automatically.

**Operator (public corpus curation).** Use `code-sim-index-public <url>` to load open-source repositories into the public collection; license metadata is auto-detected and persisted alongside each element. Run `code-sim-remove-public <url>` to retire a repository.

## Tests

```bash
pip install -e '.[dev]'
pytest                                 # full suite
pytest tests/test_staged_hunks.py      # single module
pytest tests/test_scope_modes.py -k repo
```

The pytest suite covers the parser fallback, the source filter, the staged-hunk parser, the CLI entry points, the ChromaDB client against an in-process store, public URL validation, license filtering, and the permalink builder. Embedding calls are stubbed at the boundary, so the suite does not require a running vLLM server. Git on `PATH` is required because the runtime context uses `git rev-parse` for repository discovery.

## Project Status

This project was developed as the final-year project for an MEng Computer Science degree at UCL. It is functional and tested but not maintained as a production product; bug reports and patches are welcome via the GitHub issue tracker.
