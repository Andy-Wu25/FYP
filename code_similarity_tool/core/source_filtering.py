from __future__ import annotations

from pathlib import Path


EXCLUDED_DIR_NAMES = {
    # Version control
    ".git",
    # IDE / editor
    ".idea",
    ".vs",
    ".vscode",
    # Framework build caches
    ".next",
    ".nuxt",
    ".svelte-kit",
    # Language caches
    ".mypy_cache",
    ".pytest_cache",
    ".cache",
    "__pycache__",
    # Build outputs
    "build",
    "cache",
    "coverage",
    "dist",
    "out",
    "target",
    # CI / CD
    ".circleci",
    ".github",
    ".gitlab",
    # Documentation
    "doc",
    "docs",
    "documentation",
    # Benchmarks
    "bench",
    "benchmarks",
    # Test data / fixtures / snapshots
    "fixtures",
    "testdata",
    "__snapshots__",
    # Dependencies / vendored
    "node_modules",
    "pods",
    "third_party",
    "vendor",
    "venv",
    # Static assets
    "assets",
    "static",
    # Misc
    "migrations",
    "scripts",
    "tmp",
}

EXCLUDED_EXACT_FILENAMES = {
    # Dotfiles / editor config
    ".babelrc",
    ".dockerignore",
    ".editorconfig",
    ".env",
    ".eslintrc",
    ".flake8",
    ".gitattributes",
    ".gitignore",
    ".npmrc",
    ".prettierignore",
    ".prettierrc",
    ".stylelintrc",
    # Lock files
    "cargo.lock",
    "composer.lock",
    "flake.lock",
    "gemfile.lock",
    "go.sum",
    "mix.lock",
    "package-lock.json",
    "package.resolved",
    "packages.resolved",
    "pipfile.lock",
    "pnpm-lock.yaml",
    "poetry.lock",
    "pubspec.lock",
    "yarn.lock",
    # Build system
    "cmakelists.txt",
    "dockerfile",
    "gnumakefile",
    "makefile",
}

EXCLUDED_PREFIX_FILENAMES = {
    ".env.",
    "changelog",
    "copying",
    "license",
    "readme",
}

EXCLUDED_EXTENSIONS = {
    # Documentation / markup
    ".adoc",
    ".asc",
    ".bib",
    ".md",
    ".mdx",
    ".org",
    ".rst",
    ".rtf",
    ".tex",
    ".txt",
    # Config / data
    ".cfg",
    ".conf",
    ".csv",
    ".ini",
    ".json",
    ".parquet",
    ".properties",
    ".toml",
    ".xml",
    ".xml.gz",
    ".yaml",
    ".yml",
    # Images
    ".avif",
    ".bmp",
    ".gif",
    ".ico",
    ".jpeg",
    ".jpg",
    ".png",
    ".svg",
    ".webp",
    # Audio / video
    ".mov",
    ".mp3",
    ".mp4",
    ".wav",
    # Fonts
    ".eot",
    ".otf",
    ".ttf",
    ".woff",
    ".woff2",
    # Archives / compression
    ".bz2",
    ".gz",
    ".tar",
    ".tgz",
    ".xz",
    ".zip",
    # Documents / office
    ".doc",
    ".docx",
    ".ods",
    ".odt",
    ".pdf",
    ".ppt",
    ".pptx",
    ".xls",
    ".xlsx",
    # Certificates / keys
    ".cer",
    ".crt",
    ".key",
    ".pem",
    # Compiled / binary
    ".class",
    ".jar",
    ".wasm",
    # Databases
    ".db",
    ".sqlite",
    # Serialised data / ML models
    ".h5",
    ".hdf5",
    ".npy",
    ".npz",
    ".onnx",
    ".pb",
    ".pickle",
    ".pkl",
    ".tflite",
    # Lock / map
    ".lock",
    ".map",
    # Type stubs / declarations (no runnable code)
    ".d.ts",
    ".pyi",
    # Test snapshots
    ".snap",
    ".snapshot",
    # Translation
    ".po",
    ".pot",
    # Patches
    ".diff",
    ".patch",
}

EXCLUDED_NAME_SUFFIXES = {
    # Minified / bundled
    ".bundle.js",
    ".bundle.js.map",
    ".min.css",
    ".min.js",
    # Generated code
    ".freezed.dart",
    ".g.dart",
    ".generated.go",
    ".pb.go",
    ".pb.js",
    ".pb.py",
    ".pb.ts",
    ".pb2.py",
    ".pb2_grpc.py",
}


def is_noise_source_path(path: Path) -> bool:
    lowered_parts = {part.lower() for part in path.parts}
    if EXCLUDED_DIR_NAMES.intersection(lowered_parts):
        return True

    name = path.name.lower()
    if name in EXCLUDED_EXACT_FILENAMES:
        return True
    if any(name.startswith(prefix) for prefix in EXCLUDED_PREFIX_FILENAMES):
        return True
    if any(name.endswith(suffix) for suffix in EXCLUDED_NAME_SUFFIXES):
        return True

    suffixes = "".join(s.lower() for s in path.suffixes)
    if suffixes in EXCLUDED_EXTENSIONS:
        return True
    if path.suffix.lower() in EXCLUDED_EXTENSIONS:
        return True

    return False
