import ast
import json
import pathlib
from functools import partial

SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "flodr"
WIDTH = 80


def params(src, args, method):
    seg = partial(ast.get_source_segment, src)
    pos = args.posonlyargs + args.args
    defaults = [None] * (len(pos) - len(args.defaults)) + args.defaults
    out = [seg(a) + (f"={seg(d)}" if d else "") for a, d in zip(pos, defaults)]
    if args.posonlyargs:
        out.insert(len(args.posonlyargs), "/")
    if args.vararg:
        out.append("*" + seg(args.vararg))
    elif args.kwonlyargs:
        out.append("*")
    out += [
        seg(a) + (f"={seg(d)}" if d else "")
        for a, d in zip(args.kwonlyargs, args.kw_defaults)
    ]
    if args.kwarg:
        out.append("**" + seg(args.kwarg))
    return out[1:] if method else out


def signature(name, ps, ret=""):
    line = f"{name}({', '.join(ps)}){ret}"
    if len(line) <= WIDTH:
        return line
    rows, row = [], "   "
    for p in ps:
        if len(row) + len(p) + 2 > WIDTH:
            rows.append(row)
            row = "   "
        row += f" {p},"
    return "\n".join([f"{name}(", *rows, row, f"){ret}"])


def entries(src, tree):
    seg = partial(ast.get_source_segment, src)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
            ret = f" -> {seg(node.returns)}" if node.returns else ""
            yield node.name, signature(node.name, params(src, node.args, False), ret)
        if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            if any(seg(d) == "dataclass" for d in node.decorator_list):
                # a dataclass is its fields, shown as written
                last = max(
                    n.end_lineno for n in node.body if isinstance(n, ast.AnnAssign)
                )
                start = node.decorator_list[0].lineno
                yield node.name, "\n".join(src.splitlines()[start - 1 : last])
                continue
            methods = {m.name: m for m in node.body if isinstance(m, ast.FunctionDef)}
            init = methods.get("__init__")
            ps = params(src, init.args, True) if init else []
            yield node.name, signature(node.name, ps)
            for m in methods.values():
                if not m.name.startswith("_"):
                    sig = signature(m.name, params(src, m.args, True))
                    yield f"{node.name}.{m.name}", sig


sigs, exports = {}, {}
for path in sorted(SRC.glob("*.py")):
    src = path.read_text()
    tree = ast.parse(src)
    stem = "" if path.stem == "__init__" else path.stem + "."
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.level and node.module:
            exports.update({stem + a.name: node.module for a in node.names})
    for name, sig in entries(src, tree):
        sigs[f"flodr.{path.stem}.{name}"] = sig

# re-exports, e.g. flodr.FloDR for flodr.estimator.FloDR
for name, module in exports.items():
    for key, sig in list(sigs.items()):
        prefix = f"flodr.{module}.{name.split('.')[-1]}"
        if key == prefix or key.startswith(prefix + "."):
            sigs[f"flodr.{name}" + key[len(prefix) :]] = sig

print(json.dumps(sigs, indent=1))
