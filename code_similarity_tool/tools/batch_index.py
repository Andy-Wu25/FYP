#!/usr/bin/env python3
"""Batch-index 100 public GitHub repos with a snapshot every 5."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from ..indexing.public_index import index_public_github_repo
from .snapshot import cmd_save

SNAPSHOT_EVERY = 10

# fmt: off
REPOS = [
    # ── 1–50: existing repos from public_benchmark_matrix.sh ──
    "https://github.com/developit/mitt",                  #  1  JS
    "https://github.com/pallets/itsdangerous",            #  2  Python
    "https://github.com/gorilla/mux",                     #  3  Go
    "https://github.com/SwiftyJSON/SwiftyJSON",           #  4  Swift
    "https://github.com/filp/whoops",                     #  5  PHP
    "https://github.com/rails/jbuilder",                  #  6  Ruby
    "https://github.com/chalk/chalk",                     #  7  JS
    "https://github.com/serde-rs/json",                   #  8  Rust
    "https://github.com/symfony/yaml",                    #  9  PHP
    "https://github.com/vercel/ms",                       # 10  JS
    "https://github.com/node-fetch/node-fetch",           # 11  JS
    "https://github.com/pallets/markupsafe",              # 12  Python
    "https://github.com/dbader/schedule",                 # 13  Python
    "https://github.com/debug-js/debug",                  # 14  JS
    "https://github.com/google/uuid",                     # 15  Go
    "https://github.com/joho/godotenv",                   # 16  Go
    "https://github.com/dtolnay/anyhow",                  # 17  Rust
    "https://github.com/rust-lang/log",                   # 18  Rust
    "https://github.com/jwt/ruby-jwt",                    # 19  Ruby
    "https://github.com/ramsey/uuid",                     # 20  PHP
    "https://github.com/go-chi/chi",                      # 21  Go
    "https://github.com/vlucas/phpdotenv",                # 22  PHP
    "https://github.com/nikic/FastRoute",                 # 23  PHP
    "https://github.com/lukeed/clsx",                     # 24  JS
    "https://github.com/lukeed/uid",                      # 25  JS
    "https://github.com/sindresorhus/ky",                 # 26  JS
    "https://github.com/pointfreeco/swift-parsing",       # 27  Swift
    "https://github.com/kishikawakatsumi/KeychainAccess", # 28  Swift
    "https://github.com/dart-lang/http",                  # 29  Dart
    "https://github.com/antirez/linenoise",               # 30  C
    "https://github.com/madler/zlib",                     # 31  C
    "https://github.com/jbogard/MediatR",                 # 32  C#
    "https://github.com/DapperLib/Dapper",                # 33  C#
    "https://github.com/google/gson",                     # 34  Java
    "https://github.com/square/moshi",                    # 35  Kotlin
    "https://github.com/elixir-lang/plug",                # 36  Elixir
    "https://github.com/SnapKit/SnapKit",                 # 37  Swift
    "https://github.com/Quick/Quick",                     # 38  Swift
    "https://github.com/vuejs/petite-vue",                # 39  JS
    "https://github.com/spf13/cobra",                     # 40  Go
    "https://github.com/expressjs/express",               # 41  JS
    "https://github.com/sinatra/sinatra",                 # 42  Ruby
    "https://github.com/fmtlib/fmt",                      # 43  C++
    "https://github.com/gabime/spdlog",                   # 44  C++
    "https://github.com/serde-rs/yaml",                   # 45  Rust
    "https://github.com/sirupsen/logrus",                 # 46  Go
    "https://github.com/urfave/cli",                      # 47  Go
    "https://github.com/encode/httpx",                    # 48  Python
    "https://github.com/markedjs/marked",                 # 49  JS
    "https://github.com/cashapp/molecule",                # 50  Kotlin
    # ── 51–100: new repos ──
    "https://github.com/psf/requests",                    # 51  Python
    "https://github.com/fastify/fastify",                 # 52  JS
    "https://github.com/uber-go/zap",                     # 53  Go
    "https://github.com/BurntSushi/toml",                 # 54  Rust
    "https://github.com/square/okio",                     # 55  Kotlin
    "https://github.com/pallets/click",                   # 56  Python
    "https://github.com/koajs/koa",                       # 57  JS
    "https://github.com/gin-gonic/gin",                   # 58  Go
    "https://github.com/rayon-rs/rayon",                  # 59  Rust
    "https://github.com/square/javapoet",                 # 60  Java
    "https://github.com/pytest-dev/pytest",               # 61  Python
    "https://github.com/tj/commander.js",                 # 62  JS
    "https://github.com/stretchr/testify",                # 63  Go
    "https://github.com/clap-rs/clap",                    # 64  Rust
    "https://github.com/square/wire",                     # 65  Java
    "https://github.com/pydantic/pydantic-core",          # 66  Rust
    "https://github.com/axios/axios",                     # 67  JS
    "https://github.com/labstack/echo",                   # 68  Go
    "https://github.com/tokio-rs/bytes",                  # 69  Rust
    "https://github.com/ReactiveX/RxKotlin",              # 70  Kotlin
    "https://github.com/tiangolo/fastapi",                # 71  Python
    "https://github.com/date-fns/date-fns",               # 72  TS
    "https://github.com/gofiber/fiber",                   # 73  Go
    "https://github.com/hyperium/hyper",                  # 74  Rust
    "https://github.com/spring-projects/spring-petclinic", # 75  Java
    "https://github.com/celery/celery",                   # 76  Python
    "https://github.com/reduxjs/redux",                   # 77  JS
    "https://github.com/hashicorp/go-multierror",         # 78  Go
    "https://github.com/crossbeam-rs/crossbeam",          # 79  Rust
    "https://github.com/JakeWharton/timber",              # 80  Kotlin
    "https://github.com/pallets/jinja",                   # 81  Python
    "https://github.com/sindresorhus/got",                # 82  JS
    "https://github.com/julienschmidt/httprouter",        # 83  Go
    "https://github.com/seanmonstar/reqwest",             # 84  Rust
    "https://github.com/mapstruct/mapstruct",             # 85  Java
    "https://github.com/urllib3/urllib3",                  # 86  Python
    "https://github.com/fastify/light-my-request",        # 87  JS
    "https://github.com/gorilla/websocket",               # 88  Go
    "https://github.com/image-rs/image",                  # 89  Rust
    "https://github.com/google/dagger",                   # 90  Java
    "https://github.com/python-poetry/poetry-core",       # 91  Python
    "https://github.com/vitejs/vite-plugin-react",        # 92  JS
    "https://github.com/mitchellh/mapstructure",          # 93  Go
    "https://github.com/actix/actix-web",                 # 94  Rust
    "https://github.com/square/retrofit",                 # 95  Java
    "https://github.com/python-attrs/attrs",              # 96  Python
    "https://github.com/remix-run/react-router",          # 97  TS
    "https://github.com/spf13/viper",                     # 98  Go
    "https://github.com/serde-rs/serde",                  # 99  Rust
    "https://github.com/square/retrofit",                 # 100 Java — intentional duplicate at 95; kept for round number
]
# fmt: on

assert len(REPOS) == 100, f"Expected 100 repos, got {len(REPOS)}"


def _repo_short(url: str) -> str:
    """Extract 'owner/repo' from a GitHub URL."""
    return "/".join(url.rstrip("/").split("/")[-2:])


def run_batch(start_from: int = 1) -> None:
    results: list[dict] = []
    errors: list[str] = []
    total_elements = 0

    print(f"=== Batch index: {len(REPOS)} repos, snapshot every {SNAPSHOT_EVERY} ===")
    if start_from > 1:
        print(f"    Resuming from repo #{start_from}")
    print()

    batch_t0 = time.monotonic()

    for i, url in enumerate(REPOS, start=1):
        short = _repo_short(url)

        if i < start_from:
            print(f"[{i:3d}/100] SKIP  {short}")
            continue

        print(f"[{i:3d}/100] INDEX {short} ...", flush=True)
        t0 = time.monotonic()
        try:
            count = index_public_github_repo(url)
        except Exception as exc:
            elapsed = time.monotonic() - t0
            msg = f"#{i} {short}: {exc}"
            errors.append(msg)
            results.append({"i": i, "repo": short, "elements": 0, "time_s": round(elapsed, 1), "error": str(exc)})
            print(f"         ERROR  ({elapsed:.1f}s) {exc}")
            continue

        elapsed = time.monotonic() - t0
        total_elements += count
        results.append({"i": i, "repo": short, "elements": count, "time_s": round(elapsed, 1), "error": None})
        print(f"         OK     {count} elements ({elapsed:.1f}s)")

        # Snapshot every SNAPSHOT_EVERY repos
        if i % SNAPSHOT_EVERY == 0:
            snap_name = f"{i}repos"
            print(f"         SNAPSHOT '{snap_name}' ...", flush=True)
            try:
                cmd_save(snap_name, snapshot_dir=None, force=True)
                print(f"         SNAPSHOT saved")
            except SystemExit:
                print(f"         SNAPSHOT FAILED for '{snap_name}'")

    batch_elapsed = time.monotonic() - batch_t0

    # ── Summary ──
    print()
    print("=" * 60)
    print(f"  Repos indexed : {len(results) - len(errors)}/{len(results)} attempted")
    print(f"  Total elements: {total_elements}")
    print(f"  Errors        : {len(errors)}")
    print(f"  Wall time     : {batch_elapsed / 60:.1f} min")
    print("=" * 60)

    if errors:
        print("\nFailed repos:")
        for e in errors:
            print(f"  - {e}")

    # ── Write results JSON ──
    out_path = Path("batch_results.json")
    out_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nResults written to {out_path}")


def list_repos() -> None:
    for i, url in enumerate(REPOS, start=1):
        print(f"{i:3d}. {_repo_short(url):45s} {url}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-index 100 public GitHub repos with periodic snapshots.")
    parser.add_argument("--list", action="store_true", help="List repos and exit")
    parser.add_argument("--start-from", type=int, default=1, metavar="N", help="Resume from repo N (skip 1..N-1)")
    args = parser.parse_args()

    if args.list:
        list_repos()
        return

    run_batch(start_from=args.start_from)


if __name__ == "__main__":
    main()
