#!/usr/bin/env python3
"""Replace # comments with // in .ein files. Skip # inside double-quoted strings."""
import os
import sys

def fix_line(line: str) -> str:
    i = 0
    in_string = False
    escape = False
    while i < len(line):
        c = line[i]
        if escape:
            escape = False
            i += 1
            continue
        if c == '\\' and in_string:
            escape = True
            i += 1
            continue
        if c == '"':
            in_string = not in_string
            i += 1
            continue
        if not in_string and c == '#':
            return line[:i] + '//' + line[i + 1:]
        i += 1
    return line

def fix_file(path: str) -> bool:
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    new_lines = [fix_line(ln) for ln in lines]
    if new_lines != lines:
        with open(path, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        return True
    return False

def main():
    root = os.path.join(os.path.dirname(__file__), '..')
    changed = 0
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            if name.endswith('.ein'):
                path = os.path.join(dirpath, name)
                if fix_file(path):
                    changed += 1
                    print(path)
    print(f"Updated {changed} files", file=sys.stderr)
    return 0

if __name__ == '__main__':
    sys.exit(main())
