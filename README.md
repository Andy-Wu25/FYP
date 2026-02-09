# Code Similarity Tool

Code Similarity Tool detects similar functions/methods across repositories in the same organization.

## Architecture Goals

- `code-sim-check` is **read-only** and never mutates the vector database.
- `code-sim-update` and `code-sim-index` are the only write paths.
- All repositories in the same organization (`org_id`) share one ChromaDB.
- Each repository has a deterministic `repo_id` so results are traceable.
- Similarity checks in one repo can surface matches from other repos in the same org.

## Design Choices

1. Parsing with Tree-sitter (AST-based)
- Language support: Python and Java.
- Similarity is computed at function/method granularity.

2. Embeddings with vLLM + Octen
- Backend: vLLM OpenAI-compatible API.
- Model: `Octen/Octen-Embedding-8B`.
- Client endpoint: `/v1/embeddings`.

3. Vector storage with ChromaDB
- Single shared database path across repos.
- Metadata includes `org_id`, `repo_id`, `repo_name`, `file_path`, and code element attributes.
- Queries are filtered by `org_id`.

4. Workflow separation
- Pre-commit/manual check: `code-sim-check` (read-only).
- Push-time sync: `code-sim-update` (write-only sync for current repo).
- Full rebuild for repo: `code-sim-index` (delete+reindex repo scope).

## Required Technologies

- Python 3.11+
- Git
- ChromaDB (`chromadb`)
- Tree-sitter language pack (`tree-sitter-language-pack`)
- Requests (`requests`)
- vLLM server
- `Octen/Octen-Embedding-8B` loaded in vLLM

No `pre-commit` Python package is required.

## Install

Local editable install:

```bash
pip install -e .
```

or pipx install from a path/repo:

```bash
pipx install /path/to/code-similarity-tool
```

## Commands

- `code-sim-ignore`
  - Create `.code-simignore` in repo root.
- `code-sim-check`
  - Read staged Python/Java code from Git index.
  - Embed staged functions/methods.
  - Query org-scoped DB and print nearest matches.
  - **No DB updates.**
- `code-sim-update`
  - Sync current repository into shared DB.
  - Deletes prior entries for this repo, then upserts current code.
- `code-sim-index`
  - Same sync behavior as update; used for explicit full rebuilds.
- `code-sim-install-hook --stage pre-push`
  - Install pre-push hook that runs `code-sim-update`.

## Environment Variables

- `CODE_SIM_ORG_ID`
  - Organization scope key. Repos with same value share search scope.
  - Default: `local`.
- `CODE_SIM_REPO_ID`
  - Optional override for repository id.
  - Default: derived from `remote.origin.url` if available, else repo path hash.
- `CODE_SIM_DB_PATH`
  - Shared ChromaDB directory.
  - Default: `~/.code-sim/chroma`.
- `CODE_SIM_COLLECTION`
  - Chroma collection name. Default: `project_code`.
- `CODE_SIM_LOG_LEVEL`
  - `INFO` or `DEBUG`. Default: `INFO`.
- `VLLM_BASE_URL`
  - Default: `http://127.0.0.1:8000`.
- `VLLM_API_KEY`
  - Optional bearer token.
- `VLLM_MODEL`
  - Default: `Octen/Octen-Embedding-8B`.
- `VLLM_TIMEOUT_S`
  - Default: `60`.
- `VLLM_VERIFY_MODELS`
  - `1` to probe `/v1/models`, `0` to skip.

## Multi-Repo Organization Example (Local)

This reproduces cross-repo matching on one machine.

1. Set shared org and DB:

```bash
export CODE_SIM_ORG_ID=my-org
export CODE_SIM_DB_PATH=/tmp/code-sim-shared-db
export VLLM_BASE_URL=http://127.0.0.1:8000
export VLLM_MODEL=Octen/Octen-Embedding-8B
```

2. Repo A (already indexed source):

```bash
cd /path/to/repo-a
git init
code-sim-ignore
code-sim-index
```

3. Repo B (new repo to compare against repo A):

```bash
cd /path/to/repo-b
git init
code-sim-ignore
# add code files here
git add .
code-sim-check
```

`code-sim-check` in repo B will query the shared org DB and can return similar code from repo A.

## Recommended Daily Workflow

1. Write code.
2. `git add <files>`
3. `code-sim-check` (review similarity output)
4. `git commit`
5. `code-sim-update`
6. `git push`

## Notes

- Ignore rules in `.code-simignore` apply to both checking and indexing.
- Check uses staged content, not working-tree-only changes.
- If you need strict push enforcement, install pre-push hook:

```bash
code-sim-install-hook --stage pre-push
```
