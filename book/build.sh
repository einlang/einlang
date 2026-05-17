#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
TEX="the-name-in-the-bracket.tex"
PDF="the-name-in-the-bracket.pdf"
MARGIN=0.8  # inches, must match pandoc -V geometry below

echo "=== 0. Regenerate from chapters.py ==="
python3 generate.py all

echo "=== 1. Done ==="

echo "=== 2. Render figures (SVG -> PDF) ==="
# Render at textwidth so \includegraphics uses natural size (no width= needed)
TARGET_PX=$(python3 -c "print(int((8.5 - 2 * $MARGIN) * 96))")
mkdir -p figures
for svg in figures/*.svg; do
    [ -f "$svg" ] || continue
    pdf="figures/$(basename "${svg%.svg}.pdf")"
    python3 -c "
import subprocess, xml.etree.ElementTree as ET
tree = ET.parse('$svg')
vb = tree.getroot().get('viewBox')
_, _, w, h = vb.split()
ratio = $TARGET_PX / float(w)
wp = str(int(float(w) * ratio))
hp = str(int(float(h) * ratio))
html = '''<!DOCTYPE html>
<html><head><meta charset=\"utf-8\"><style>
@page { margin: 0; size: ''' + wp + '''px ''' + hp + '''px; }
body { margin: 0; padding: 0; }
</style></head>
<body><img src=\"''' + '$(basename "$svg")' + '''\" style=\"width:''' + wp + '''px; height:''' + hp + '''px; display:block;\"></body></html>'''
html_path = 'figures/_pb.html'
with open(html_path, 'w') as f:
    f.write(html)
subprocess.run(['$CHROME', '--headless', '--disable-gpu', '--no-pdf-header-footer',
    '--print-to-pdf=$pdf', html_path], check=True, capture_output=True)
import os; os.remove(html_path)
"
    echo "  $svg -> $pdf"
done

echo "=== 3. Pandoc -> LaTeX ==="
cat > /tmp/svg-to-pdf.lua << 'LUAEOF'
function Image(img) img.src = img.src:gsub('%.svg$', '.pdf'); return img end
function HorizontalRule()
  return pandoc.RawBlock('latex', '\\vspace{2em}')
end
LUAEOF
pandoc _combined.md -o "$TEX" -s \
    --from=markdown+pipe_tables+fenced_code_blocks+grid_tables \
    --lua-filter=/tmp/svg-to-pdf.lua \
    --toc --toc-depth=2 \
    --metadata title="The Name in the Bracket" \
    --metadata author="einlang" \
    --metadata date="$(date +%Y-%m-%d)" \
    -V fontsize=11pt \
    -V geometry=margin=0.8in \
    -V colorlinks=true \
    --resource-path=.

echo "=== 4. Post-process LaTeX ==="
python3 << 'PYEOF'
with open("the-name-in-the-bracket.tex") as f:
    c = f.read()
c = c.replace('{\\linethickness}', '{0.5pt}')
with open("the-name-in-the-bracket.tex", "w") as f:
    f.write(c)
PYEOF
sed -i '' 's/\[11pt,/\[11pt,titlepage/' "$TEX"

echo "=== 5. Compile PDF ==="
tectonic "$TEX"

echo "=== Done: $PDF ==="
ls -lh "$PDF"
