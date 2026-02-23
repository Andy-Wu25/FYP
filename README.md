# Code Similarity Tool

Code Similarity Tool compares code elements against two datasets:

- private organization code (stored in private collection)
- central public GNU-licensed code (stored in public collection)

## Architecture

1. Read-only checks
- `code-sim-check` and `code-sim-check-private` query private org data only.
- `code-sim-check-self` queries the current repository scope only (within private data).
- `code-sim-check-public` queries public GNU index only.
- Query scope is configurable with `--scope`:
  - `staged` (default): code elements overlapping staged hunks.
  - `files`: whole provided file(s), staged or unstaged.
  - `repo`: whole current repository.
- These commands never update the database.

2. Write paths
- `code-sim-update` syncs the current repository into private org scope.
- `code-sim-index` syncs private scope by default.
- `code-sim-index --url <github-url>` (or `code-sim-index-public <github-url>`) indexes a public GitHub repo into public scope, after GNU license validation.

3. Shared DB and identity
- All repos share one ChromaDB path (`CODE_SIM_DB_PATH`).
- Private and public data are stored in separate collections.
- Private data is partitioned by `org_id` and `repo_id`.
- Public data is partitioned by `public_source_id` derived from `url + commit`.

4. Parsing and Embeddings
- Uses all grammars available in installed `tree-sitter-language-pack`.
- For files where no tree-sitter grammar is available (or no declaration-like elements are extracted), the full file is embedded as one element.
- Built-in noise filters skip docs, lockfiles, generated/vendor/cache folders, media/binaries, secrets/certs, and minified bundles.
- vLLM OpenAI-compatible API (`/v1/embeddings`)
- default model: `Octen/Octen-Embedding-8B`

## Required Technologies

- Python 3.11+
- Git
- ChromaDB (`chromadb`)
- Tree-sitter language pack (`tree-sitter-language-pack`)
- Requests (`requests`)
- vLLM server with `Octen/Octen-Embedding-8B`

No `pre-commit` package is required.

## Install

```bash
pipx install /path/to/code-similarity-tool
```

or

```bash
pip install -e .
```

## Commands

- `code-sim-ignore`
  - Create `.code-simignore` template.
- `code-sim-check`
  - Alias of `code-sim-check-private`.
- `code-sim-check-private`
  - Compare code against private org index (`--scope staged|files|repo`).
- `code-sim-check-self`
  - Compare code against only the current repository's private index entries (`--scope staged|files|repo`).
- `code-sim-check-public`
  - Compare code against public GNU index (`--scope staged|files|repo`).
  - Output includes commit-pinned GitHub permalink to the matched file/line and commit URL.
- `code-sim-update`
  - Sync current repo to private org index.
- `code-sim-index`
  - Without URL: same behavior as private sync.
  - With URL (`code-sim-index --url ...` or positional URL): public GNU indexing path.
- `code-sim-index-public <url>`
  - Explicit operator command for public indexing.
  - `--debug-element-index N` prints the Nth collected public element and exits (for embedding failure diagnosis).
- `code-sim-install-hook --stage pre-push`
  - Install pre-push hook to run `code-sim-update`.

## Environment Variables

- `CODE_SIM_ORG_ID`
  - Private organization key. Repos with same value share private query scope.
  - Default: `local`
- `CODE_SIM_REPO_ID`
  - Optional override for private repo id.
  - Default: hash(remote origin URL) else hash(repo path)
- `CODE_SIM_DB_PATH`
  - Shared ChromaDB path.
  - Default: `~/.code-sim/chroma`
- `CODE_SIM_COLLECTION`
  - Base collection prefix. Default: `project_code`
- `CODE_SIM_PRIVATE_COLLECTION`
  - Private collection name. Default: `<CODE_SIM_COLLECTION>_private`
- `CODE_SIM_PUBLIC_COLLECTION`
  - Public collection name. Default: `<CODE_SIM_COLLECTION>_public`
- `CODE_SIM_LOG_LEVEL`
  - `INFO` / `DEBUG`
- `CODE_SIM_MAX_FILE_BYTES`
  - Max file size for private/public indexing and query file scans. Default: `250000`
- `VLLM_BASE_URL`
  - Default: `http://127.0.0.1:8000`
- `VLLM_API_KEY`
  - Optional token
- `VLLM_MODEL`
  - Default: `Octen/Octen-Embedding-8B`
- `VLLM_TIMEOUT_S`
  - Default: `60`
- `VLLM_VERIFY_MODELS`
  - `1` to probe `/v1/models`, `0` to skip

## Recommended Workflows

### Private org workflow (repo developer)

1. stage changes

```bash
git add .
```

2. check privately

```bash
code-sim-check-private
```

or check unstaged whole file(s):

```bash
code-sim-check-private --scope files path/to/file.ext
```

or check the whole repository:

```bash
code-sim-check-private --scope repo
```

3. commit

```bash
git commit -m "..."
```

4. sync before push

```bash
code-sim-update
git push
```

### Public central indexing workflow (service operator)

1. index a GNU-licensed GitHub repo

```bash
code-sim-index-public https://github.com/<owner>/<repo>
```

or

```bash
code-sim-index --url https://github.com/<owner>/<repo>
```

2. optional specific ref

```bash
code-sim-index-public https://github.com/<owner>/<repo> --ref v1.2.3
```

Validation performed before indexing:
- URL must be a valid GitHub repository URL
- git ref must resolve
- license must be detected and in GNU allowlist (`GPL-*`, `LGPL-*`, `AGPL-*`)

## Multi-Repo Private Scope Example

Use identical values in repo A and repo B:

```bash
export CODE_SIM_ORG_ID=my-org
export CODE_SIM_DB_PATH=/Users/andywu/.code-sim/shared-db
export VLLM_BASE_URL=http://127.0.0.1:8000
export VLLM_MODEL=Octen/Octen-Embedding-8B
```

Then:
- repo A runs `code-sim-update` (or `code-sim-index`)
- repo B runs `code-sim-check-private`

repo B can now retrieve matches from repo A.

## Migration Note

If you previously used a single mixed collection, run:

```bash
code-sim-update
```

for each private repository you want indexed, and re-run public indexing commands for any public sources. New versions use separate private/public collections by design.
