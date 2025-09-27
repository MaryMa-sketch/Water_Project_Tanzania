import os, sys

IGNORE = {'.git', 'env', 'venv', '.venv', '.ipynb_checkpoints', '__pycache__'}

def make_tree(path='.', max_depth=2, prefix=''):
    try:
        entries = [e for e in os.scandir(path) if e.name not in IGNORE]
    except PermissionError:
        return
    entries.sort(key=lambda e: (not e.is_dir(), e.name.lower()))
    for i, entry in enumerate(entries):
        leaf = (i == len(entries) - 1)
        connector = "└── " if leaf else "├── "
        print(prefix + connector + entry.name + ("/" if entry.is_dir() else ""))
        if entry.is_dir() and max_depth > 1:
            child_prefix = prefix + ("    " if leaf else "│   ")
            make_tree(entry.path, max_depth - 1, child_prefix)

with open("structure.txt", "w", encoding="utf-8") as f:
    _stdout = sys.stdout
    sys.stdout = f
    print("Water_Project_Tanzania/")
    make_tree(".", max_depth=2)
    sys.stdout = _stdout

print("✅ Clean tree saved to structure.txt")
