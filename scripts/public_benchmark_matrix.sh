#!/usr/bin/env bash
set -euo pipefail

# Nested benchmark ladder:
# - first 5 repos are also included in 10/20/50
# - first 10 repos are also included in 20/50
# - first 20 repos are also included in 50
#
# This list intentionally avoids the known heavy outliers from your current run
# (for example sqlalchemy/alembic, pallets/flask, pallets/click) and biases
# toward small library / utility repos across all tiers.
#
# Important: "balanced" here is by curation, not by a hard measured guarantee.
# Exact code element counts depend on your tree-sitter coverage, fallback file
# embedding, ignore rules, and the current HEAD of each repo.

readonly DEFAULT_SNAPSHOT_PREFIX="public_benchmark"
readonly TOTAL_REPOS=50

readonly -a REPOS=(
  "https://github.com/developit/mitt"
  "https://github.com/pallets/itsdangerous"
  "https://github.com/gorilla/mux"
  "https://github.com/SwiftyJSON/SwiftyJSON"
  "https://github.com/filp/whoops"

  "https://github.com/rails/jbuilder"
  "https://github.com/chalk/chalk"
  "https://github.com/serde-rs/json"
  "https://github.com/symfony/yaml"
  "https://github.com/vercel/ms"

  "https://github.com/node-fetch/node-fetch"
  "https://github.com/pallets/markupsafe"
  "https://github.com/dbader/schedule"
  "https://github.com/debug-js/debug"
  "https://github.com/google/uuid"
  "https://github.com/joho/godotenv"
  "https://github.com/dtolnay/anyhow"
  "https://github.com/rust-lang/log"
  "https://github.com/jwt/ruby-jwt"
  "https://github.com/ramsey/uuid"

  "https://github.com/go-chi/chi"
  "https://github.com/vlucas/phpdotenv"
  "https://github.com/nikic/FastRoute"
  "https://github.com/lukeed/clsx"
  "https://github.com/lukeed/uid"
  "https://github.com/sindresorhus/ky"
  "https://github.com/pointfreeco/swift-parsing"
  "https://github.com/kishikawakatsumi/KeychainAccess"
  "https://github.com/dart-lang/http"
  "https://github.com/antirez/linenoise"

  "https://github.com/madler/zlib"
  "https://github.com/jbogard/MediatR"
  "https://github.com/DapperLib/Dapper"
  "https://github.com/google/gson"
  "https://github.com/square/moshi"
  "https://github.com/elixir-lang/plug"
  "https://github.com/SnapKit/SnapKit"
  "https://github.com/Quick/Quick"
  "https://github.com/vuejs/petite-vue"
  "https://github.com/spf13/cobra"

  "https://github.com/expressjs/express"
  "https://github.com/sinatra/sinatra"
  "https://github.com/fmtlib/fmt"
  "https://github.com/gabime/spdlog"
  "https://github.com/serde-rs/yaml"
  "https://github.com/sirupsen/logrus"
  "https://github.com/urfave/cli"
  "https://github.com/encode/httpx"
  "https://github.com/markedjs/marked"
  "https://github.com/cashapp/molecule"
)

usage() {
  cat <<'EOF'
Usage:
  bash scripts/public_benchmark_matrix.sh list
  bash scripts/public_benchmark_matrix.sh 5
  bash scripts/public_benchmark_matrix.sh 10
  bash scripts/public_benchmark_matrix.sh 20
  bash scripts/public_benchmark_matrix.sh 50
  bash scripts/public_benchmark_matrix.sh add-10
  bash scripts/public_benchmark_matrix.sh add-20
  bash scripts/public_benchmark_matrix.sh add-50
  bash scripts/public_benchmark_matrix.sh all-with-snapshots

Modes:
  list
    Print the ordered repo ladder.

  5 | 10 | 20 | 50
    Index the first N repos from the ordered ladder.
    Use these when starting from a clean public index.

  add-10 | add-20 | add-50
    Incrementally extend an existing run:
    - add-10 indexes repos 6-10
    - add-20 indexes repos 11-20
    - add-50 indexes repos 21-50
    This is the safest manual snapshot workflow.

  all-with-snapshots
    Index all 50 repos in one pass and automatically save snapshots after
    5, 10, 20, and 50 repos, using code-sim-snapshot-save.

Environment variables:
  SNAPSHOT_PREFIX
    Prefix for snapshot names in all-with-snapshots mode.
    Default: public_benchmark

  SHOW_STATS
    If set to 1, run code-sim-stats after each snapshot in all-with-snapshots mode.

Notes:
  - For strict reproducibility, prefer all-with-snapshots in one session so the
    5/10/20/50 snapshots are all built from one continuous indexing run.
  - This ladder is intentionally biased toward small and medium libraries.
    It avoids the dominant repos you already observed in your stats output.
  - If you want manual checkpoints, use:
      5 -> snapshot -> add-10 -> snapshot -> add-20 -> snapshot -> add-50 -> snapshot
  - If you rerun 10/20/50 later from scratch, HEAD may move on some repos
    between runs unless you pin refs yourself.
EOF
}

print_repos() {
  local i
  for ((i = 0; i < ${#REPOS[@]}; i++)); do
    printf "%2d. %s\n" "$((i + 1))" "${REPOS[i]}"
  done
}

save_snapshot() {
  local count="$1"
  local prefix="${SNAPSHOT_PREFIX:-$DEFAULT_SNAPSHOT_PREFIX}"
  local name="${prefix}_${count}_repos"

  echo
  echo "==> Saving snapshot: ${name}"
  code-sim-snapshot-save "${name}" --force

  if [[ "${SHOW_STATS:-0}" == "1" ]]; then
    echo
    echo "==> Stats after snapshot ${name}"
    code-sim-stats
  fi
}

index_range() {
  local start="$1"
  local end="$2"
  local i

  for ((i = start; i < end; i++)); do
    echo
    echo "==> [$((i + 1))/${TOTAL_REPOS}] ${REPOS[i]}"
    code-sim-index-public "${REPOS[i]}"
  done
}

index_first_n() {
  local count="$1"
  index_range 0 "${count}"
}

run_all_with_snapshots() {
  index_range 0 5
  save_snapshot 5

  index_range 5 10
  save_snapshot 10

  index_range 10 20
  save_snapshot 20

  index_range 20 50
  save_snapshot 50
}

main() {
  local mode="${1:-}"

  case "${mode}" in
    list)
      print_repos
      ;;
    5|10|20|50)
      index_first_n "${mode}"
      ;;
    add-10)
      index_range 5 10
      ;;
    add-20)
      index_range 10 20
      ;;
    add-50)
      index_range 20 50
      ;;
    all-with-snapshots)
      run_all_with_snapshots
      ;;
    *)
      usage
      exit 1
      ;;
  esac
}

main "$@"
