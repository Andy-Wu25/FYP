from __future__ import annotations

from pathlib import Path


EXCLUDED_DIR_NAMES = {
    ".git",
    ".next",
    ".nuxt",
    ".svelte-kit",
    ".mypy_cache",
    ".pytest_cache",
    ".cache",
    "__pycache__",
    "build",
    "cache",
    "coverage",
    "dist",
    "docs",
    "fixtures",
    "node_modules",
    "out",
    "pods",
    "target",
    "testdata",
    "third_party",
    "tmp",
    "venv",
    "vendor",
}

EXCLUDED_EXACT_FILENAMES = {
    ".dockerignore",
    ".editorconfig",
    ".env",
    ".gitattributes",
    ".gitignore",
    ".npmrc",
    ".prettierignore",
    ".prettierrc",
    "cargo.lock",
    "composer.lock",
    "go.sum",
    "package-lock.json",
    "pipfile.lock",
    "pnpm-lock.yaml",
    "poetry.lock",
    "yarn.lock",
}

EXCLUDED_PREFIX_FILENAMES = {
    ".env.",
    "changelog",
    "license",
    "copying",
    "readme",
}

EXCLUDED_EXTENSIONS = {
    ".adoc",
    ".asc",
    ".bmp",
    ".bz2",
    ".cer",
    ".class",
    ".crt",
    ".csv",
    ".db",
    ".doc",
    ".docx",
    ".gif",
    ".gz",
    ".ico",
    ".jar",
    ".jpeg",
    ".jpg",
    ".key",
    ".lock",
    ".map",
    ".md",
    ".mdx",
    ".mov",
    ".mp3",
    ".mp4",
    ".ods",
    ".odt",
    ".parquet",
    ".pdf",
    ".pem",
    ".png",
    ".ppt",
    ".pptx",
    ".rst",
    ".sqlite",
    ".tar",
    ".tgz",
    ".toml",
    ".txt",
    ".wav",
    ".woff",
    ".woff2",
    ".xls",
    ".xlsx",
    ".xml.gz",
    ".xz",
    ".zip",
}

EXCLUDED_NAME_SUFFIXES = {
    ".bundle.js",
    ".bundle.js.map",
    ".min.css",
    ".min.js",
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
