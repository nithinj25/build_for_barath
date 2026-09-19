"""Every name a module's functions use is defined somewhere: pytest -q tests/test_code.py

A deleted helper otherwise surfaces only when that code path runs — a bundle
build found series_quality missing only after it had been committed.
"""
from __future__ import annotations

import builtins
import symtable
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FILES = sorted(p for d in ("linkage", "handlers", "scripts", "enrich") for p in (ROOT / d).rglob("*.py"))
MODULE_DUNDERS = {"__file__", "__name__", "__doc__", "__spec__", "__package__", "__builtins__", "__loader__", "__path__"}


def undefined_names(path: Path) -> list[str]:
    top = symtable.symtable(path.read_text(encoding="utf-8"), str(path), "exec")
    defined = {s.get_name() for s in top.get_symbols() if s.is_assigned() or s.is_imported() or s.is_namespace()}
    missing = []

    def walk(table):
        for sym in table.get_symbols():
            name = sym.get_name()
            global_use = sym.is_global() or table.get_type() == "module"
            if global_use and sym.is_referenced() and name not in defined \
                    and not hasattr(builtins, name) and name not in MODULE_DUNDERS:
                missing.append(f"{table.get_name()}: {name}")
        for child in table.get_children():
            walk(child)
    walk(top)
    return missing


@pytest.mark.parametrize("path", FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_no_undefined_names(path):
    assert undefined_names(path) == []
