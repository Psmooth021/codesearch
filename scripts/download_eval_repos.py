"""Shallow-clones each repo in eval/data/repo_manifest.yaml to its pinned
commit, into a gitignored eval_repos/<name>/ directory - so `codesearch
eval` runs against a reproducible, fixed snapshot rather than whatever
HEAD happens to be on the day it's run."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).parent.parent
MANIFEST_PATH = REPO_ROOT / "src" / "codesearch" / "eval" / "data" / "repo_manifest.yaml"
EVAL_REPOS_DIR = REPO_ROOT / "eval_repos"


def clone_at_commit(name: str, url: str, commit: str, dest: Path) -> None:
    if dest.exists():
        head = subprocess.run(
            ["git", "-C", str(dest), "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip()
        if head == commit:
            print(f"[{name}] already at {commit[:12]}, skipping")
            return
        print(f"[{name}] present but at wrong commit ({head[:12]}), re-cloning")
        import shutil

        shutil.rmtree(dest)

    print(f"[{name}] cloning {url} @ {commit[:12]}...")
    dest.mkdir(parents=True)
    subprocess.run(["git", "-C", str(dest), "init", "-q"], check=True)
    subprocess.run(["git", "-C", str(dest), "remote", "add", "origin", url], check=True)
    subprocess.run(
        ["git", "-C", str(dest), "fetch", "--depth", "1", "-q", "origin", commit], check=True
    )
    subprocess.run(["git", "-C", str(dest), "checkout", "-q", commit], check=True)
    print(f"[{name}] done")


def main() -> None:
    manifest = yaml.safe_load(MANIFEST_PATH.read_text())
    EVAL_REPOS_DIR.mkdir(exist_ok=True)
    for repo in manifest["repos"]:
        clone_at_commit(repo["name"], repo["url"], repo["commit"], EVAL_REPOS_DIR / repo["name"])


if __name__ == "__main__":
    sys.exit(main())
