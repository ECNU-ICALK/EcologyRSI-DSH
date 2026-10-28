"""Layering is enforced: a module may import only from its own layer or below."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "src" / "ecogym"
LAYERS = ["kernel", "data", "task", "protocol", "solvers", "evaluation", "harness", "episodes",
          "arena", "agents", "surface"]
SURFACE = {"cli", "report", "__main__", "service", "web"}
DOMAIN_TERMS = ("air_temperature", "relative_humidity", "co2", "greenhouse", "agc_", "cucumber", "tomato")


def layer_of(module: str) -> str | None:
    parts = module.split(".")
    if parts[0] != "ecogym" or len(parts) == 1:
        return None
    head = parts[1]
    if head in SURFACE:
        return "surface"
    if head == "domains":
        return "domains"
    return head if head in LAYERS else None


def imports(path: Path) -> list[str]:
    module = "ecogym." + ".".join(path.relative_to(ROOT).with_suffix("").parts).removesuffix(".__init__")
    package = module if path.name == "__init__.py" else module.rsplit(".", 1)[0]
    found = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package.split(".")
                base = base[: len(base) - (node.level - 1)]
                found.append(".".join([*base, node.module] if node.module else base))
            elif node.module:
                found.append(node.module)
    return found


class LayerTests(unittest.TestCase):
    def test_no_upward_imports(self):
        problems = []
        for path in ROOT.rglob("*.py"):
            own = layer_of("ecogym." + ".".join(path.relative_to(ROOT).with_suffix("").parts))
            if own in (None, "domains", "surface"):
                continue
            for target in imports(path):
                other = layer_of(target)
                if other == "domains":
                    problems.append(f"{path.relative_to(ROOT)} imports a domain pack ({target})")
                elif other == "surface" or (other in LAYERS and LAYERS.index(other) > LAYERS.index(own)):
                    problems.append(f"{path.relative_to(ROOT)} ({own}) imports {target} ({other})")
        self.assertEqual(problems, [])

    def test_framework_code_names_no_domain(self):
        problems = []
        for path in ROOT.rglob("*.py"):
            relative = path.relative_to(ROOT)
            if relative.parts[0] == "domains":
                continue
            text = path.read_text(encoding="utf-8").lower()
            problems.extend(f"{relative}: {term}" for term in DOMAIN_TERMS if term in text)
        self.assertEqual(problems, [])


if __name__ == "__main__":
    unittest.main()
