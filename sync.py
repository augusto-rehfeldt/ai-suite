"""Copy the ai_suite package into every consumer repository, or check the copies match.

Each consumer imports the sibling ai-suite checkout when it exists and falls back to its
own vendored ai_suite/ otherwise, so a fresh clone of that repository runs on its own.
Run this after changing ai_suite and commit the copies with the consumer.

    python sync.py            # write the copies
    python sync.py --check    # exit 1 if any copy differs (check_workspace runs this)
"""
import sys
from fnmatch import fnmatch
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACKAGE = HERE / "ai_suite"
# Consumers with their own GitHub repository; the rest import the sibling checkout only.
TARGETS = ("book writer", "book-watch", "calibre-book-summarizer", "lamplight", "mathforge",
           "semantic-story-atlas")
# Local secrets, runtime state and caches never travel, and a copy's own are never touched.
SKIP = ("*.local.json", "*state*.json", "__pycache__", "*.pyc")


def shipped(root: Path) -> dict[str, bytes]:
    return {path.relative_to(root).as_posix(): path.read_bytes()
            for path in sorted(root.rglob("*"))
            if path.is_file() and not any(fnmatch(part, pattern) for part in path.relative_to(root).parts
                                          for pattern in SKIP)}


def main(check: bool) -> int:
    source = shipped(PACKAGE)
    stale = []
    for name in TARGETS:
        target = HERE.parent / name / "ai_suite"
        if not target.parent.is_dir():
            print(f"skip {name}: not checked out")
            continue
        current = shipped(target) if target.is_dir() else {}
        if current == source:
            continue
        stale.append(name)
        if check:
            continue
        for rel in current.keys() - source.keys():
            (target / rel).unlink()
        for rel, data in source.items():
            (target / rel).parent.mkdir(parents=True, exist_ok=True)
            (target / rel).write_bytes(data)
        print(f"synced {name}")
    if check and stale:
        print("ai_suite copies differ from ai-suite: " + ", ".join(stale) + " (run: python sync.py)")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main("--check" in sys.argv[1:]))
