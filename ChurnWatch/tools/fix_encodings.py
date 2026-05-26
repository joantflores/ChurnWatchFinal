from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
py_files = list(ROOT.rglob('*.py'))
changed = []
for p in py_files:
    # skip virtualenv directories if any
    if '\\.venv' in str(p) or '/.venv' in str(p):
        continue
    try:
        text = p.read_text(encoding='utf-8')
        # already utf-8; no change
        continue
    except Exception:
        try:
            raw = p.read_bytes()
            text = raw.decode('latin-1')
            # write back as utf-8
            p.write_text(text, encoding='utf-8')
            changed.append(p.relative_to(ROOT))
        except Exception as e:
            print(f"Failed to fix {p}: {e}")

print('Re-encoded files:')
for c in changed:
    print(' -', c)
print('\nDone.')
sys.exit(0)
