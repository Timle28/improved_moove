"""Split the pandoc-generated guide_raw.rst into per-chapter Sphinx RST files."""

import re
from pathlib import Path

RAW = Path(__file__).parent / "guide_raw.rst"
OUT = Path(__file__).parent / "source"

lines = RAW.read_text(encoding="utf-8").splitlines(keepends=True)

# --- Locate substitution definition block at the end of file ---
sub_defs: dict[str, list[str]] = {}
sub_start = None
current_sub_name = None
current_sub_lines: list[str] = []

for i, line in enumerate(lines):
    m = re.match(r'^\.\. \|(.+?)\| image::', line)
    if m:
        if sub_start is None:
            sub_start = i
        if current_sub_name:
            sub_defs[current_sub_name] = current_sub_lines
        current_sub_name = m.group(1)
        current_sub_lines = [line]
    elif sub_start is not None and current_sub_name:
        current_sub_lines.append(line)

if current_sub_name:
    sub_defs[current_sub_name] = current_sub_lines

content_lines = lines[:sub_start] if sub_start else lines

# --- Identify chapter boundaries (Heading level 1 = underline with '=') ---
chapters: list[tuple[str, str, int, int]] = []
# (filename, title, start_line, end_line)

heading1_positions = []
for i, line in enumerate(content_lines):
    stripped = line.rstrip()
    if stripped and all(c == '=' for c in stripped) and len(stripped) >= 3:
        if i > 0 and content_lines[i - 1].strip():
            title = content_lines[i - 1].strip()
            heading1_positions.append((i - 1, title))

chapter_map = {
    "Table of Contents": None,  # skip TOC
    "Installation": "installation",
    "MooveTaf": "moovetaf",
    "MooveGUI": "moovegui",
    "REC file and Feedback Information": "rec_file",
    "Loading previously recorded data": "loading_data",
    "FAQ": "faq",
    "References": "references",
}

# Intro = everything before the first heading1
first_h1_line = heading1_positions[0][0] if heading1_positions else len(content_lines)

for idx, (line_no, title) in enumerate(heading1_positions):
    end = heading1_positions[idx + 1][0] if idx + 1 < len(heading1_positions) else len(content_lines)
    fname = chapter_map.get(title)
    if fname is not None:
        chapters.append((fname, title, line_no, end))


def find_used_substitutions(text: str) -> set[str]:
    return set(re.findall(r'\|([^|]+?)\|', text))


def fix_image_paths(text: str) -> str:
    return text.replace('docs/_static/media/', '_static/images/')


def fix_code_blocks(text: str) -> str:
    """Wrap unindented command lines that pandoc left bare."""
    return text


def clean_chapter(text: str) -> str:
    text = fix_image_paths(text)
    # Remove trailing whitespace on lines
    text = re.sub(r' +\n', '\n', text)
    # Collapse 3+ consecutive blank lines to 2
    text = re.sub(r'\n{4,}', '\n\n\n', text)
    return text


# --- Write introduction ---
intro_lines = content_lines[:first_h1_line]
# Skip the TOC section (it starts after intro text and goes until Installation)
# Find where "Table of Contents" heading is
toc_start = None
for i, line in enumerate(intro_lines):
    if line.strip() == "Table of Contents":
        toc_start = i
        break

intro_text = "".join(intro_lines[:toc_start]) if toc_start else "".join(intro_lines)
intro_text = clean_chapter(intro_text)

# Find substitutions used in intro
intro_subs = find_used_substitutions(intro_text)
sub_block = ""
for name in sorted(intro_subs):
    if name in sub_defs:
        sub_block += fix_image_paths("".join(sub_defs[name]))

intro_content = f""".. _introduction:

Introduction
============

{intro_text.strip()}

{sub_block}
"""
(OUT / "introduction.rst").write_text(clean_chapter(intro_content), encoding="utf-8")
print(f"  Written: introduction.rst")

# --- Write chapter files ---
for fname, title, start, end in chapters:
    chapter_text = "".join(content_lines[start:end])
    chapter_text = clean_chapter(chapter_text)

    # Find and append needed substitution definitions
    used_subs = find_used_substitutions(chapter_text)
    sub_block = ""
    for name in sorted(used_subs):
        if name in sub_defs:
            sub_block += fix_image_paths("".join(sub_defs[name]))

    if sub_block:
        chapter_text = chapter_text.rstrip() + "\n\n" + sub_block

    label = fname.replace("_", "-")
    chapter_text = f".. _{label}:\n\n" + chapter_text

    (OUT / f"{fname}.rst").write_text(chapter_text, encoding="utf-8")
    print(f"  Written: {fname}.rst ({end - start} lines)")

print("\nDone! All chapters written to docs/source/")
