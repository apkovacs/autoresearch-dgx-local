"""
Hyperparameter extraction and injection for train.py.

Parses the uppercase constant assignments in the hyperparameters section
and can write modified values back, preserving comments and structure.
This is the "genome" that gets migrated between research branches.
"""

import re
import ast

HYPERPARAM_PATTERN = re.compile(r'^([A-Z][A-Z_0-9]*)\s*=\s*(.+?)(\s*#.*)?$')


def extract_hyperparams(path="train.py"):
    params = {}
    state = "searching"  # searching -> found_header -> in_section
    with open(path) as f:
        for line in f:
            if state == "searching" and "Hyperparameters" in line:
                state = "found_header"
                continue
            if state == "found_header" and line.startswith("# ---"):
                state = "in_section"
                continue
            if state == "in_section":
                if line.startswith("# ---"):
                    break
                m = HYPERPARAM_PATTERN.match(line.rstrip())
                if m:
                    params[m.group(1)] = m.group(2).strip()
    return params


def apply_hyperparams(path, params):
    lines = []
    state = "searching"
    with open(path) as f:
        for line in f:
            if state == "searching" and "Hyperparameters" in line:
                state = "found_header"
                lines.append(line)
                continue
            if state == "found_header" and line.startswith("# ---"):
                state = "in_section"
                lines.append(line)
                continue
            if state == "in_section" and line.startswith("# ---"):
                state = "done"
                lines.append(line)
                continue
            if state == "in_section":
                m = HYPERPARAM_PATTERN.match(line.rstrip())
                if m and m.group(1) in params:
                    name = m.group(1)
                    comment = m.group(3) or ""
                    new_value = params[name]
                    lines.append(f"{name} = {new_value}{comment}\n")
                    continue
            lines.append(line)
    with open(path, "w") as f:
        f.writelines(lines)


def _safe_eval(value_str):
    try:
        return float(ast.literal_eval(value_str))
    except (ValueError, SyntaxError):
        try:
            return float(eval(value_str))
        except Exception:
            return None


def diff_hyperparams(a, b):
    diffs = []
    all_keys = set(a) | set(b)
    for key in all_keys:
        if key not in a or key not in b:
            diffs.append((key, a.get(key), b.get(key), float('inf')))
            continue
        if a[key] != b[key]:
            va, vb = _safe_eval(a[key]), _safe_eval(b[key])
            if va is not None and vb is not None and va != 0:
                magnitude = abs(vb - va) / abs(va)
            else:
                magnitude = float('inf')
            diffs.append((key, a[key], b[key], magnitude))
    diffs.sort(key=lambda x: x[3], reverse=True)
    return diffs


if __name__ == "__main__":
    params = extract_hyperparams()
    print("Extracted hyperparameters:")
    for name, value in params.items():
        print(f"  {name:24s} = {value}")
