"""Generate index.md, _data/book.yml, and _combined.md from chapters.py."""
import os, sys, re

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chapters import CHAPTERS

BOOK_DIR = os.path.dirname(os.path.abspath(__file__))


def gen_index():
    """Update index.md Contents section."""
    lines = ["## Contents", ""]
    n = 0
    for filename, part_title, title in CHAPTERS:
        n += 1
        if part_title:
            lines.append(f"### {part_title}")
        html_name = filename.replace(".md", ".html")
        if title.startswith("Appendix"):
            lines.append(f"- [{title}]({html_name})")
        elif title.startswith("Epilogue"):
            lines.append(f"- [{title}]({html_name})")
        else:
            lines.append(f"- [{n}. {title}]({html_name})")

    path = os.path.join(BOOK_DIR, "index.md")
    with open(path) as f:
        content = f.read()
    content = re.sub(r"## Contents.*", "\n".join(lines) + "\n", content, flags=re.DOTALL)
    with open(path, "w") as f:
        f.write(content)
    print("  index.md updated")


def gen_book_yml():
    """Regenerate _data/book.yml for Jekyll sidebar TOC."""
    entries = [{
        "title": "The Name in the Bracket",
        "url": "/book/",
        "short_title": "Home",
        "kind": "frontmatter",
        "section": "Opening",
        "eyebrow": "",
    }]

    n = 0
    current_section = None
    for filename, part_title, title in CHAPTERS:
        n += 1
        if part_title:
            current_section = part_title
        html_name = filename.replace(".md", ".html")

        if title.startswith("Appendix"):
            section, eyebrow, short, full = ("Reference", "Appendix", "Appendix", title)
        elif title.startswith("Epilogue"):
            section, eyebrow, short, full = ("Closing", "Epilogue", "Epilogue", title)
        else:
            section, eyebrow = (current_section, str(n))
            short = f"{n}. {title}"
            full = f"Chapter {n} · {title}"

        entries.append({
            "title": full, "url": f"/book/{html_name}",
            "short_title": short, "kind": "chapter",
            "section": section, "eyebrow": eyebrow,
        })

    path = os.path.join(os.path.dirname(BOOK_DIR), "_data", "book.yml")
    with open(path, "w") as f:
        for entry in entries:
            f.write(f"- title: \"{entry['title']}\"\n")
            f.write(f"  url: {entry['url']}\n")
            f.write(f"  short_title: \"{entry['short_title']}\"\n")
            f.write(f"  kind: {entry['kind']}\n")
            f.write(f"  section: \"{entry['section']}\"\n")
            f.write(f"  eyebrow: \"{entry['eyebrow']}\"\n")
    print("  _data/book.yml updated")


def gen_combined():
    """Concatenate chapters into _combined.md with Part headings."""
    parts = []
    for filename, part_title, _title in CHAPTERS:
        path = os.path.join(BOOK_DIR, filename)
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found, skipping")
            continue
        with open(path) as f:
            content = f.read()
        # Strip YAML frontmatter
        if content.startswith("---"):
            end = content.find("---", 3)
            if end != -1:
                content = content[end + 3:].lstrip()
        parts.append(r"\newpage")
        if part_title:
            parts.append(f"# {part_title}")
        parts.append(content)

    path = os.path.join(BOOK_DIR, "_combined.md")
    with open(path, "w") as f:
        f.write("\n\n".join(parts))
    print(f"  Combined {len(CHAPTERS)} chapters -> _combined.md")


if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else "all"
    os.chdir(BOOK_DIR)
    if action in ("index", "all"):
        gen_index()
    if action in ("book.yml", "all"):
        gen_book_yml()
    if action in ("combined", "all"):
        gen_combined()
