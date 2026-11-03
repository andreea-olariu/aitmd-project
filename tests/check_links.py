"""Check every relative link, image and #anchor in the Markdown files.

    python tests/check_links.py        (exit code 1 if something is broken)

Anchors follow GitHub's rules: lower case, punctuation removed, spaces -> '-'.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]
LINK = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)\)")


def slug(heading: str) -> str:
    text = re.sub(r"[^\w\- ]", "", heading.strip().lower())
    return text.replace(" ", "-")


def anchors(path: Path) -> set[str]:
    text = re.sub(r"```.*?```", "", path.read_text(), flags=re.S)
    return {slug(h) for h in re.findall(r"^#+ (.*)$", text, re.M)}


def main() -> int:
    errors = []
    for md in FILES:
        text = re.sub(r"```.*?```", "", md.read_text(), flags=re.S)  # ignore code blocks
        for target in LINK.findall(text):
            if re.match(r"[a-z]+://|mailto:", target):
                continue
            file_part, _, anchor = target.partition("#")
            dest = (md.parent / file_part).resolve() if file_part else md
            if not dest.exists():
                errors.append(f"{md.relative_to(ROOT)}: missing file {target}")
            elif anchor and dest.suffix == ".md" and anchor not in anchors(dest):
                errors.append(f"{md.relative_to(ROOT)}: missing anchor {target}")
    for error in errors:
        print(error)
    print(f"{len(FILES)} files checked, {len(errors)} broken link(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
