import ast
import sys

files_to_check = [
    'RIPE/ripe/train.py',
    'RIPE/ripe/models/ripe.py',
    'test_scheduler.py'
]

for fpath in files_to_check:
    try:
        with open(fpath, 'r', encoding='utf-8') as f:
            code = f.read()
            ast.parse(code)
        print(f"✓ {fpath}: OK")
    except SyntaxError as e:
        print(f"✗ {fpath}: Line {e.lineno} - {e.msg}")
        if e.text:
            print(f"  {e.text}")
