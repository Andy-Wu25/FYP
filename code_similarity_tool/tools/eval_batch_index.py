#!/usr/bin/env python3
"""Batch-index 256 public GitHub repos with snapshots at powers of two.

Designed for scalability evaluation: saves a database snapshot after
indexing 1, 2, 4, 8, 16, 32, 64, 128, and 256 repositories so that
retrieval quality can be measured against progressively larger corpora.

Repos are curated for code-clone / code-similarity evaluation:
  - 15+ programming languages
  - clustered by functional domain (web frameworks, HTTP clients, logging,
    CLI parsers, JSON/serialisation, testing, error handling, etc.)
  - professional, well-maintained, medium-sized open-source projects

Usage:
    code-sim-batch-index --list
    code-sim-batch-index [--start-from N] [--store-code] [--truncate]
"""
from __future__ import annotations

import argparse
import json
import socket
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

from ..indexing.public_index import index_public_github_repo
from ..infra.embeddings import add_embedding_args
from .snapshot import cmd_save


# ── Embedding server health check / retry ────────────────────────────
_EMBED_URL = None  # resolved lazily from OPENAI_API_BASE


def _get_embed_host_port() -> tuple[str, int]:
    """Return (host, port) of the embedding server from OPENAI_API_BASE."""
    import os
    base = os.environ.get("OPENAI_API_BASE", "http://127.0.0.1:8000")
    parsed = urlparse(base)
    return (parsed.hostname or "127.0.0.1", parsed.port or 8000)


def _is_server_healthy(timeout: float = 15.0) -> bool:
    """Check the embedding server responds to an HTTP request (not just TCP)."""
    import os
    base = os.environ.get("OPENAI_API_BASE", "http://127.0.0.1:8000")
    try:
        resp = requests.get(f"{base}/v1/models", timeout=timeout)
        return resp.status_code == 200
    except Exception:
        return False


def _is_server_error(exc: Exception) -> bool:
    """Check whether an exception is caused by a server connectivity or responsiveness issue.

    Catches both hard failures (connection refused, broken pipe) and soft
    failures (read timeout — server is reachable but unresponsive).
    """
    msg = str(exc).lower()
    error_keywords = [
        "connection refused", "connectionrefusederror",
        "connectionerror", "connection reset",
        "broken pipe", "brokenpipeerror",
        "timed out", "timeout", "timeouterror",
        "readtimeout", "read timed out",
        "connecttimeout", "connect timed out",
        "no route to host", "network is unreachable",
    ]
    if any(kw in msg for kw in error_keywords):
        return True
    # Walk the exception chain
    cause = exc.__cause__
    while cause is not None:
        if isinstance(cause, (ConnectionError, ConnectionRefusedError,
                              ConnectionResetError, BrokenPipeError,
                              TimeoutError, OSError, socket.timeout)):
            return True
        if isinstance(cause, (requests.exceptions.ConnectionError,
                              requests.exceptions.ReadTimeout,
                              requests.exceptions.Timeout)):
            return True
        cause = cause.__cause__
    return False


def _wait_for_server(max_wait: int = 300, poll_interval: int = 10) -> bool:
    """Block until the embedding server is healthy, or until max_wait seconds.

    Uses an actual HTTP health check (/v1/models), not just a TCP connect,
    so it detects both dead tunnels and unresponsive/stuck servers.

    Returns True if the server came back, False if we gave up.
    """
    host, port = _get_embed_host_port()
    print(f"\n         ⏳ Embedding server {host}:{port} not responding. "
          f"Waiting up to {max_wait}s...", flush=True)
    print(f"         (Restart your SSH tunnel if needed)", flush=True)

    waited = 0
    while waited < max_wait:
        time.sleep(poll_interval)
        waited += poll_interval
        if _is_server_healthy():
            print(f"         ✓ Server healthy after {waited}s wait.", flush=True)
            return True
        dots = "." * min(waited // poll_interval, 30)
        print(f"         ... still waiting ({waited}s){dots}", flush=True)

    print(f"         ✗ Server still not responding after {max_wait}s. "
          f"Continuing (remaining repos will likely fail).", flush=True)
    return False

SNAPSHOT_POINTS = {1, 2, 4, 8, 16, 32, 64, 128, 256}

# fmt: off
REPOS = [
    # ── 1–8: diverse base (one per major ecosystem) ─────────────
    "https://github.com/expressjs/express",                  #   1  JS   – web framework
    "https://github.com/pallets/flask",                      #   2  Py   – web framework
    "https://github.com/gin-gonic/gin",                      #   3  Go   – web framework
    "https://github.com/serde-rs/json",                      #   4  Rust – JSON
    "https://github.com/google/gson",                        #   5  Java – JSON
    "https://github.com/gabime/spdlog",                      #   6  C++  – logging
    "https://github.com/sinatra/sinatra",                    #   7  Ruby – web framework
    "https://github.com/filp/whoops",                        #   8  PHP  – error handling

    # ── 9–16: second wave – broaden language & domain ───────────
    "https://github.com/fastify/fastify",                    #   9  JS   – web framework
    "https://github.com/tiangolo/fastapi",                   #  10  Py   – web framework
    "https://github.com/labstack/echo",                      #  11  Go   – web framework
    "https://github.com/actix/actix-web",                    #  12  Rust – web framework
    "https://github.com/square/moshi",                       #  13  Kt   – JSON
    "https://github.com/DapperLib/Dapper",                   #  14  C#   – ORM
    "https://github.com/dart-lang/http",                     #  15  Dart – HTTP
    "https://github.com/elixir-lang/plug",                   #  16  Ex   – web middleware

    # ── 17–32: HTTP clients, routing, more frameworks ───────────
    "https://github.com/koajs/koa",                          #  17  JS   – web framework
    "https://github.com/encode/httpx",                       #  18  Py   – HTTP client
    "https://github.com/gofiber/fiber",                      #  19  Go   – web framework
    "https://github.com/seanmonstar/reqwest",                #  20  Rust – HTTP client
    "https://github.com/spring-projects/spring-petclinic",   #  21  Java – web app
    "https://github.com/nlohmann/json",                      #  22  C++  – JSON
    "https://github.com/jbogard/MediatR",                    #  23  C#   – mediator
    "https://github.com/hapijs/hapi",                        #  24  JS   – web framework
    "https://github.com/psf/requests",                       #  25  Py   – HTTP client
    "https://github.com/go-chi/chi",                         #  26  Go   – routing
    "https://github.com/hyperium/hyper",                     #  27  Rust – HTTP
    "https://github.com/square/javapoet",                    #  28  Java – code gen
    "https://github.com/SwiftyJSON/SwiftyJSON",              #  29  Swift– JSON
    "https://github.com/rack/rack",                          #  30  Ruby – web interface
    "https://github.com/symfony/yaml",                       #  31  PHP  – YAML
    "https://github.com/fmtlib/fmt",                         #  32  C++  – formatting

    # ── 33–64: CLI, HTTP, routing, templating ───────────────────
    "https://github.com/node-fetch/node-fetch",              #  33  JS   – HTTP client
    "https://github.com/pallets/werkzeug",                   #  34  Py   – WSGI toolkit
    "https://github.com/gorilla/mux",                        #  35  Go   – routing
    "https://github.com/tokio-rs/axum",                      #  36  Rust – web framework
    "https://github.com/JamesNK/Newtonsoft.Json",             #  37  C#   – JSON
    "https://github.com/AutoMapper/AutoMapper",              #  38  C#   – object mapping
    "https://github.com/jwt/ruby-jwt",                       #  39  Ruby – JWT
    "https://github.com/nikic/FastRoute",                    #  40  PHP  – routing
    "https://github.com/axios/axios",                        #  41  JS   – HTTP client
    "https://github.com/encode/starlette",                   #  42  Py   – ASGI framework
    "https://github.com/julienschmidt/httprouter",           #  43  Go   – routing
    "https://github.com/seanmonstar/warp",                   #  44  Rust – web framework
    "https://github.com/square/wire",                        #  45  Java – protobuf
    "https://github.com/SnapKit/SnapKit",                    #  46  Swift– layout
    "https://github.com/rails/jbuilder",                     #  47  Ruby – JSON builder
    "https://github.com/vlucas/phpdotenv",                   #  48  PHP  – env
    "https://github.com/tj/commander.js",                    #  49  JS   – CLI parser
    "https://github.com/pallets/click",                      #  50  Py   – CLI parser
    "https://github.com/spf13/cobra",                        #  51  Go   – CLI parser
    "https://github.com/clap-rs/clap",                       #  52  Rust – CLI parser
    "https://github.com/yargs/yargs",                        #  53  JS   – CLI parser
    "https://github.com/urfave/cli",                         #  54  Go   – CLI parser
    "https://github.com/apple/swift-argument-parser",        #  55  Swift– CLI parser
    "https://github.com/debug-js/debug",                     #  56  JS   – logging
    "https://github.com/winstonjs/winston",                  #  57  JS   – logging
    "https://github.com/pinojs/pino",                        #  58  JS   – logging
    "https://github.com/sirupsen/logrus",                    #  59  Go   – logging
    "https://github.com/uber-go/zap",                        #  60  Go   – logging
    "https://github.com/rs/zerolog",                         #  61  Go   – logging
    "https://github.com/rust-lang/log",                      #  62  Rust – logging
    "https://github.com/serilog/serilog",                    #  63  C#   – logging
    "https://github.com/Seldaek/monolog",                    #  64  PHP  – logging

    # ── 65–128: testing, errors, utilities, serialisation ───────
    "https://github.com/pytest-dev/pytest",                  #  65  Py   – testing
    "https://github.com/stretchr/testify",                   #  66  Go   – testing
    "https://github.com/mochajs/mocha",                      #  67  JS   – testing
    "https://github.com/slimphp/Slim",                       #  68  PHP  – web framework
    "https://github.com/dtolnay/anyhow",                     #  69  Rust – error handling
    "https://github.com/dtolnay/thiserror",                  #  70  Rust – error handling
    "https://github.com/hashicorp/go-multierror",            #  71  Go   – error handling
    "https://github.com/tqdm/tqdm",                           #  72  Py   – progress bars
    "https://github.com/fatih/color",                        #  73  Go   – terminal colour
    "https://github.com/antirez/linenoise",                  #  74  C    – line editing
    "https://github.com/developit/mitt",                     #  75  JS   – event emitter
    "https://github.com/pallets/itsdangerous",               #  76  Py   – signing
    "https://github.com/pallets/markupsafe",                 #  77  Py   – HTML escaping
    "https://github.com/pallets/jinja",                      #  78  Py   – templating
    "https://github.com/janl/mustache.js",                   #  79  JS   – templating
    "https://github.com/markedjs/marked",                    #  80  JS   – markdown
    "https://github.com/dbader/schedule",                    #  81  Py   – scheduling
    "https://github.com/robfig/cron",                        #  82  Go   – scheduling
    "https://github.com/celery/celery",                      #  83  Py   – task queue
    "https://github.com/vercel/ms",                          #  84  JS   – time parsing
    "https://github.com/arrow-py/arrow",                     #  85  Py   – dates
    "https://github.com/google/uuid",                        #  86  Go   – UUID
    "https://github.com/ramsey/uuid",                        #  87  PHP  – UUID
    "https://github.com/lukeed/uid",                         #  88  JS   – ID generation
    "https://github.com/lukeed/clsx",                        #  89  JS   – classnames
    "https://github.com/vuejs/petite-vue",                   #  90  JS   – reactive UI
    "https://github.com/more-itertools/more-itertools",      #  91  Py   – itertools
    "https://github.com/joho/godotenv",                      #  92  Go   – env loading
    "https://github.com/spf13/viper",                        #  93  Go   – configuration
    "https://github.com/mitchellh/mapstructure",             #  94  Go   – struct mapping
    "https://github.com/psf/black",                          #  95  Py   – code formatter
    "https://github.com/gorilla/websocket",                  #  96  Go   – WebSocket
    "https://github.com/tidwall/gjson",                      #  97  Go   – JSON reading
    "https://github.com/cashapp/molecule",                   #  98  Kt   – Compose runtime
    "https://github.com/JakeWharton/timber",                 #  99  Kt   – logging
    "https://github.com/square/okio",                        # 100  Kt   – I/O
    "https://github.com/ReactiveX/RxKotlin",                 # 101  Kt   – reactive
    "https://github.com/google/dagger",                      # 102  Java – DI
    "https://github.com/PyCQA/isort",                        # 103  Py   – import sorting
    "https://github.com/mapstruct/mapstruct",                # 104  Java – object mapping
    "https://github.com/keleshev/schema",                    # 105  Py   – validation
    "https://github.com/python-poetry/poetry-core",          # 106  Py   – packaging
    "https://github.com/python-attrs/attrs",                 # 107  Py   – classes
    "https://github.com/crossbeam-rs/crossbeam",             # 108  Rust – concurrency
    "https://github.com/tokio-rs/bytes",                     # 109  Rust – byte buffers
    "https://github.com/serde-rs/serde",                     # 110  Rust – serialisation
    "https://github.com/jaraco/path",                        # 111  Py   – path utilities
    "https://github.com/BurntSushi/toml",                    # 112  Rust – TOML
    "https://github.com/rayon-rs/rayon",                     # 113  Rust – parallelism
    "https://github.com/image-rs/image",                     # 114  Rust – image proc
    "https://github.com/madler/zlib",                        # 115  C    – compression
    "https://github.com/shopspring/decimal",                 # 116  Go   – decimal math
    "https://github.com/charmbracelet/bubbletea",            # 117  Go   – TUI framework
    "https://github.com/charmbracelet/lipgloss",             # 118  Go   – TUI styling
    "https://github.com/mattn/go-isatty",                    # 119  Go   – terminal detect
    "https://github.com/aio-libs/aiohttp",                   # 120  Py   – async HTTP
    "https://github.com/valyala/fasthttp",                   # 121  Go   – HTTP server
    "https://github.com/urllib3/urllib3",                     # 122  Py   – HTTP client
    "https://github.com/go-yaml/yaml",                       # 123  Go   – YAML
    "https://github.com/sindresorhus/got",                   # 124  JS   – HTTP client
    "https://github.com/fastify/light-my-request",           # 125  JS   – HTTP testing
    "https://github.com/vitejs/vite-plugin-react",           # 126  JS   – build plugin
    "https://github.com/pelletier/go-toml",                  # 127  Go   – TOML
    "https://github.com/pkg/errors",                         # 128  Go   – errors

    # ── 129–160: databases, caching, data stores ──────────────
    "https://github.com/redis/go-redis",                     # 129  Go   – Redis client
    "https://github.com/redis/redis-py",                     # 130  Py   – Redis client
    "https://github.com/redis/node-redis",                   # 131  JS   – Redis client
    "https://github.com/go-sql-driver/mysql",                # 132  Go   – MySQL driver
    "https://github.com/jackc/pgx",                          # 133  Go   – PostgreSQL
    "https://github.com/mattn/go-sqlite3",                   # 134  Go   – SQLite driver
    "https://github.com/jmoiron/sqlx",                       # 135  Go   – SQL extensions
    "https://github.com/sequelize/sequelize",                  # 136  JS   – ORM
    "https://github.com/tortoise/tortoise-orm",              # 137  Py   – async ORM
    "https://github.com/go-gorm/gorm",                       # 138  Go   – ORM
    "https://github.com/uptrace/bun",                        # 139  Go   – SQL client
    "https://github.com/Masterminds/squirrel",               # 140  Go   – SQL builder
    "https://github.com/patrickmn/go-cache",                 # 141  Go   – in-memory cache
    "https://github.com/allegro/bigcache",                   # 142  Go   – concurrent cache
    "https://github.com/dgraph-io/ristretto",               # 143  Go   – cache
    "https://github.com/lib/pq",                             # 144  Go   – PostgreSQL driver

    # ── 145–160: auth, JWT, crypto, WebSocket ─────────────────
    "https://github.com/golang-jwt/jwt",                     # 145  Go   – JWT
    "https://github.com/auth0/node-jsonwebtoken",            # 146  JS   – JWT
    "https://github.com/jpadilla/pyjwt",                     # 147  Py   – JWT
    "https://github.com/lestrrat-go/jwx",                    # 148  Go   – JWX
    "https://github.com/FiloSottile/age",                    # 149  Go   – encryption
    "https://github.com/caddyserver/certmagic",              # 150  Go   – auto HTTPS
    "https://github.com/nhooyr/websocket",                   # 151  Go   – WebSocket
    "https://github.com/websockets/ws",                      # 152  JS   – WebSocket
    "https://github.com/gorilla/handlers",                   # 153  Go   – HTTP middleware
    "https://github.com/rs/cors",                            # 154  Go   – CORS
    "https://github.com/graphql-go/graphql",                 # 155  Go   – GraphQL
    "https://github.com/99designs/gqlgen",                   # 156  Go   – GraphQL codegen
    "https://github.com/graphql/graphql-js",                 # 157  JS   – GraphQL ref impl
    "https://github.com/twitchtv/twirp",                     # 158  Go   – RPC framework
    "https://github.com/hashicorp/go-retryablehttp",         # 159  Go   – retryable HTTP
    "https://github.com/cenkalti/backoff",                   # 160  Go   – exponential backoff

    # ── 161–192: utilities, filesystem, data structures ───────
    "https://github.com/spf13/afero",                        # 161  Go   – filesystem abstraction
    "https://github.com/spf13/pflag",                        # 162  Go   – POSIX flags
    "https://github.com/spf13/cast",                         # 163  Go   – safe type casting
    "https://github.com/hashicorp/hcl",                      # 164  Go   – config language
    "https://github.com/gofrs/uuid",                         # 165  Go   – UUID
    "https://github.com/mitchellh/go-homedir",               # 166  Go   – home directory
    "https://github.com/colinhacks/zod",                     # 167  TS   – schema validation
    "https://github.com/hapijs/joi",                         # 168  JS   – validation
    "https://github.com/ajv-validator/ajv",                  # 169  JS   – JSON schema validator
    "https://github.com/ljharb/qs",                          # 170  JS   – query string
    "https://github.com/sindresorhus/execa",                 # 171  JS   – child process
    "https://github.com/sindresorhus/p-queue",               # 172  JS   – promise queue
    "https://github.com/sindresorhus/ora",                   # 173  JS   – terminal spinner
    "https://github.com/enquirer/enquirer",                    # 174  JS   – interactive CLI
    "https://github.com/isaacs/minimatch",                   # 175  JS   – glob matching
    "https://github.com/mrmlnc/fast-glob",                    # 176  JS   – glob matching
    "https://github.com/marshmallow-code/marshmallow",       # 177  Py   – serialisation
    "https://github.com/pydantic/pydantic",                  # 178  Py   – data validation
    "https://github.com/tiangolo/typer",                     # 179  Py   – CLI framework
    "https://github.com/Textualize/rich",                    # 180  Py   – rich terminal
    "https://github.com/hynek/structlog",                    # 181  Py   – structured logging
    "https://github.com/sdispater/pendulum",                 # 182  Py   – datetime
    "https://github.com/ijl/orjson",                         # 183  Py/Rust – fast JSON
    "https://github.com/msgpack/msgpack-python",             # 184  Py   – MessagePack
    "https://github.com/google/wire",                        # 185  Go   – DI code gen
    "https://github.com/uber-go/fx",                         # 186  Go   – DI framework
    "https://github.com/uber-go/dig",                        # 187  Go   – DI container
    "https://github.com/google/go-cmp",                      # 188  Go   – value comparison
    "https://github.com/google/go-querystring",              # 189  Go   – URL query encoding
    "https://github.com/Masterminds/semver",                 # 190  Go   – semantic versioning
    "https://github.com/hashicorp/go-version",               # 191  Go   – version parsing
    "https://github.com/dustin/go-humanize",                 # 192  Go   – human-friendly units

    # ── 193–224: testing, mocking, assertions ─────────────────
    "https://github.com/jarcoal/httpmock",                   # 193  Go   – HTTP mocking
    "https://github.com/vektra/mockery",                     # 194  Go   – mock generator
    "https://github.com/DATA-DOG/go-sqlmock",                # 195  Go   – SQL mocking
    "https://github.com/onsi/gomega",                        # 196  Go   – assertion
    "https://github.com/onsi/ginkgo",                        # 197  Go   – BDD testing
    "https://github.com/chaijs/chai",                        # 198  JS   – assertion
    "https://github.com/sinonjs/sinon",                      # 199  JS   – mocking
    "https://github.com/ladjs/supertest",                    # 200  JS   – HTTP testing
    "https://github.com/nock/nock",                          # 201  JS   – HTTP mocking
    "https://github.com/FasterXML/jackson-core",             # 202  Java – JSON streaming
    "https://github.com/assertj/assertj",                    # 203  Java – fluent assertions
    "https://github.com/wiremock/wiremock",                    # 204  Java – HTTP mocking
    "https://github.com/google/truth",                       # 205  Java – assertion
    "https://github.com/InsertKoinIO/koin",                  # 206  Kt   – DI
    "https://github.com/mockk/mockk",                        # 207  Kt   – mocking
    "https://github.com/Kotlin/kotlinx.serialization",       # 208  Kt   – serialisation

    # ── 225–256: compression, crypto, I/O, more languages ─────
    "https://github.com/lz4/lz4",                            # 209  C    – compression
    "https://github.com/google/snappy",                      # 210  C++  – compression
    "https://github.com/google/benchmark",                   # 211  C++  – benchmarking
    "https://github.com/google/leveldb",                     # 212  C++  – KV store
    "https://github.com/redis/hiredis",                      # 213  C    – Redis client
    "https://github.com/libuv/libuv",                        # 214  C    – async I/O
    "https://github.com/nothings/stb",                       # 215  C    – single-file libs
    "https://github.com/onevcat/Kingfisher",                 # 216  Swift– image loading
    "https://github.com/Moya/Moya",                          # 217  Swift– network abstraction
    "https://github.com/vapor/vapor",                        # 218  Swift– web framework
    "https://github.com/FluentValidation/FluentValidation",  # 219  C#   – validation
    "https://github.com/moq/moq",                            # 220  C#   – mocking
    "https://github.com/App-vNext/Polly",                    # 221  C#   – resilience
    "https://github.com/xunit/xunit",                        # 222  C#   – testing
    "https://github.com/guzzle/guzzle",                      # 223  PHP  – HTTP client
    "https://github.com/PHPMailer/PHPMailer",                # 224  PHP  – email
    "https://github.com/thephpleague/flysystem",             # 225  PHP  – filesystem
    "https://github.com/composer/semver",                    # 226  PHP  – versioning
    "https://github.com/ruby-grape/grape",                   # 227  Ruby – REST API framework
    "https://github.com/rspec/rspec-core",                   # 228  Ruby – testing
    "https://github.com/dry-rb/dry-validation",              # 229  Ruby – validation
    "https://github.com/dry-rb/dry-types",                   # 230  Ruby – type system
    "https://github.com/bitflags/bitflags",                  # 231  Rust – bitflags
    "https://github.com/rust-itertools/itertools",           # 232  Rust – iterator tools
    "https://github.com/indexmap-rs/indexmap",               # 233  Rust – ordered map
    "https://github.com/DaveGamble/cJSON",                   # 234  C    – JSON
    "https://github.com/BurntSushi/memchr",                  # 235  Rust – byte search
    "https://github.com/mitsuhiko/insta",                    # 236  Rust – snapshot testing
    "https://github.com/console-rs/indicatif",               # 237  Rust – progress bars
    "https://github.com/console-rs/console",                 # 238  Rust – terminal utilities
    "https://github.com/dirs-dev/dirs-rs",                   # 239  Rust – directory paths
    "https://github.com/dtolnay/proc-macro2",                # 240  Rust – proc macros
    "https://github.com/dtolnay/quote",                      # 241  Rust – quasi-quoting
    "https://github.com/dtolnay/syn",                        # 242  Rust – Rust parser
    "https://github.com/chronotope/chrono",                  # 243  Rust – datetime
    "https://github.com/uuid-rs/uuid",                       # 244  Rust – UUID
    "https://github.com/tokio-rs/tracing",                   # 245  Rust – diagnostics
    "https://github.com/jbeder/yaml-cpp",                    # 246  C++  – YAML
    "https://github.com/diesel-rs/diesel",                   # 247  Rust – ORM
    "https://github.com/launchbadge/sqlx",                   # 248  Rust – async SQL
    "https://github.com/hyperium/tonic",                     # 249  Rust – gRPC
    "https://github.com/tower-rs/tower",                     # 250  Rust – service abstractions
    "https://github.com/tokio-rs/mio",                       # 251  Rust – async I/O
    "https://github.com/rust-lang/hashbrown",                # 252  Rust – hash map
    "https://github.com/bluss/indexmap",                     # 253  Rust – (alias, dedup will catch)
    "https://github.com/dart-lang/args",                     # 254  Dart – CLI args
    "https://github.com/dart-lang/shelf",                    # 255  Dart – web server middleware
    "https://github.com/elixir-lang/gen_stage",              # 256  Ex   – data pipeline
]
# fmt: on

assert len(REPOS) == 256, f"Expected 256 repos, got {len(REPOS)}"

# Verify no duplicates in primary list
_unique = set(r.rstrip("/").lower() for r in REPOS)
assert len(_unique) == 256, f"Duplicate repo URLs detected ({256 - len(_unique)} dupes)"


def _repo_short(url: str) -> str:
    """Extract 'owner/repo' from a GitHub URL."""
    return "/".join(url.rstrip("/").split("/")[-2:])


def run_batch(start_from: int = 1, store_code: bool = False,
              truncate_tokens: int | None = None) -> None:
    total = len(REPOS)
    results: list[dict] = []
    errors: list[str] = []
    total_elements = 0

    print(f"=== Batch index: {total} repos ===")
    print(f"    Snapshots at: {sorted(SNAPSHOT_POINTS)}")
    print(f"    Store code  : {'yes' if store_code else 'no'}")
    print(f"    Truncate    : {f'{truncate_tokens} tokens' if truncate_tokens else 'off (auto-chunk on overflow)'}")
    if start_from > 1:
        print(f"    Resuming from repo #{start_from}")
    print()

    batch_t0 = time.monotonic()

    for i, url in enumerate(REPOS, start=1):
        short = _repo_short(url)

        if i < start_from:
            print(f"[{i:3d}/{total}] SKIP  {short}")
            continue

        print(f"[{i:3d}/{total}] INDEX {short} ...", flush=True)
        max_retries = 2
        for attempt in range(1 + max_retries):
            t0 = time.monotonic()
            try:
                count = index_public_github_repo(url, store_code=store_code, truncate_tokens=truncate_tokens)
            except Exception as exc:
                elapsed = time.monotonic() - t0

                # If this looks like a connection failure, wait for the server
                if _is_server_error(exc) and attempt < max_retries:
                    print(f"         CONN ERROR ({elapsed:.1f}s) {exc}")
                    recovered = _wait_for_server(max_wait=300, poll_interval=10)
                    if recovered:
                        print(f"         RETRY  {short} (attempt {attempt + 2}/{max_retries + 1})", flush=True)
                        continue
                    # Server didn't come back — fall through to record error

                msg = f"#{i} {short}: {exc}"
                errors.append(msg)
                results.append({"i": i, "repo": short, "elements": 0, "time_s": round(elapsed, 1), "error": str(exc)})
                print(f"         ERROR  ({elapsed:.1f}s) {exc}")
                break
            else:
                elapsed = time.monotonic() - t0
                total_elements += count
                results.append({"i": i, "repo": short, "elements": count, "time_s": round(elapsed, 1), "error": None})
                print(f"         OK     {count} elements ({elapsed:.1f}s)")
                break

        # Snapshot at powers of two
        if i in SNAPSHOT_POINTS:
            prefix = f"trunc{truncate_tokens}-" if truncate_tokens else ""
            snap_name = f"{prefix}{i}repos"

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
    print(f"  Store code    : {'yes' if store_code else 'no'}")
    print(f"  Snapshots     : {sorted(p for p in SNAPSHOT_POINTS if p <= total)}")
    print(f"  Wall time     : {batch_elapsed / 60:.1f} min")
    print("=" * 60)

    if errors:
        print("\nFailed repos:")
        for e in errors:
            print(f"  - {e}")

    # ── Write results JSON ──
    out_path = Path("eval_batch_results.json")
    out_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nResults written to {out_path}")


def list_repos() -> None:
    total = len(REPOS)
    for i, url in enumerate(REPOS, start=1):
        marker = " *" if i in SNAPSHOT_POINTS else "  "
        print(f"{i:3d}.{marker} {_repo_short(url):45s} {url}")
    print(f"\n  * = snapshot point  ({total} repos total)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch-index 256 public GitHub repos with snapshots at powers of two for evaluation."
    )
    parser.add_argument("--list", action="store_true", help="List repos and exit")
    parser.add_argument("--start-from", type=int, default=1, metavar="N", help="Resume from repo N (skip 1..N-1)")
    parser.add_argument("--store-code", action="store_true", help="Store source code text in the vector database.")
    add_embedding_args(parser)
    args = parser.parse_args()

    if args.list:
        list_repos()
        return

    run_batch(
        start_from=args.start_from,
        store_code=args.store_code,
        truncate_tokens=args.truncate,
    )


if __name__ == "__main__":
    main()
