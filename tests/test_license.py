"""
tests/test_license.py — P0 IP/License Audit Gate for CommercialBrainEncoder

Blocks shipping if any test fails.

Tests:
  1. test_no_nc_license_strings_in_source      — no CC-BY-NC / NonCommercial strings in code
  2. test_commercial_verified_flag_on_all_datasets — DATASETS dict has valid commercial licenses
  3. test_no_forbidden_imports_in_all_source   — no competitor module imports (TRIBE, MindEye, etc.)
  4. test_requirements_no_nc_packages          — requirements.txt has no known NC packages
  5. test_mit_license_file_exists              — LICENSE file present (xfail until created)

No torch, no GPU, no model imports. Pure stdlib + pathlib.
"""

from __future__ import annotations

import ast
import re
import warnings
from pathlib import Path
from typing import Generator

import pytest

# ---------------------------------------------------------------------------
# Helpers — project root resolution
# ---------------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).parent.parent

# NC strings that must not appear in non-docstring, non-comment source code.
_NC_PATTERNS: list[str] = [
    "cc-by-nc",
    "noncommer cial",  # split intentionally to avoid self-match in THIS file
    "non-commercial",
    "nc-4.0",
    "nc-sa",
]
# Rebuild without the intentional split (used at runtime, not matched against this file's source).
NC_PATTERNS: list[str] = [p.replace("noncommer cial", "noncommercial") for p in _NC_PATTERNS]

# Forbidden import prefixes (case-insensitive module name check).
FORBIDDEN_IMPORT_PREFIXES: list[str] = [
    "tribe",
    "mindeye",
    "brainbench",
    "neural_encoding_model",
]

# Known NC-licensed package names (matched against requirements.txt lines).
KNOWN_NC_PACKAGES: list[str] = [
    "tribe2",
    "tribe-v2",
    "neural_encoding_model",
    "mindeye",
    "brainbench",
]

# Commercially acceptable license strings for DATASETS entries.
COMMERCIAL_LICENSES: frozenset[str] = frozenset({"CC0", "CC-BY-4.0", "CC-BY"})


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------

def _iter_source_files() -> Generator[Path, None, None]:
    """Yield all .py files in PROJECT_ROOT, excluding the tests/ subdirectory."""
    for path in sorted(PROJECT_ROOT.glob("*.py")):
        if path.is_file():
            yield path


def _is_docstring_node(node: ast.AST, parent: ast.AST | None) -> bool:
    """
    Return True if `node` is a docstring — i.e., the first statement of a
    module, class, or function body and is a bare string constant expression.

    AST docstrings are `ast.Expr` nodes whose value is `ast.Constant(str)`.
    They appear as the zeroth element of the body of `ast.Module`,
    `ast.ClassDef`, or `ast.FunctionDef` / `ast.AsyncFunctionDef`.
    """
    if not isinstance(node, ast.Expr):
        return False
    if not isinstance(node.value, ast.Constant):
        return False
    if not isinstance(node.value.value, str):
        return False
    if parent is None:
        return False
    body_holders = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    if not isinstance(parent, body_holders):
        return False
    body: list[ast.stmt] = parent.body  # type: ignore[attr-defined]
    return len(body) > 0 and body[0] is node


def _collect_non_docstring_string_literals(
    tree: ast.Module,
) -> list[tuple[int, str]]:
    """
    Walk the AST and return (lineno, value) for every string constant that is
    NOT a docstring.  Docstrings are excluded because they are documentation
    and may legitimately reference NC licenses for comparison/contrast purposes.
    """
    # Build a child → parent mapping for the whole tree.
    parent_map: dict[int, ast.AST] = {}
    for parent_node in ast.walk(tree):
        for child in ast.iter_child_nodes(parent_node):
            parent_map[id(child)] = parent_node

    results: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            parent = parent_map.get(id(node))
            if _is_docstring_node(node, parent):
                continue  # skip docstring
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            results.append((node.lineno, node.value))
    return results


def _parse_source(path: Path) -> ast.Module:
    """Parse a Python source file and return its AST."""
    source = path.read_text(encoding="utf-8", errors="replace")
    return ast.parse(source, filename=str(path))


def _collect_imports(tree: ast.Module) -> list[tuple[int, str]]:
    """
    Return (lineno, top_level_module_name) for every import statement in tree.
    Handles both `import X` and `from X import Y`.
    """
    results: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                results.append((node.lineno, top))
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                top = node.module.split(".")[0]
                results.append((node.lineno, top))
    return results


def _load_datasets_dict() -> dict:
    """
    Extract DATASETS from data_pipeline.py using AST + ast.literal_eval.

    This approach is pure stdlib and does NOT execute the module, so it is safe
    regardless of missing packages (h5py, nibabel, torch, etc.).

    Strategy:
      1. Parse data_pipeline.py into an AST.
      2. Walk the module-level statements for an `ast.Assign` where the target
         name is "DATASETS" and the value is an `ast.Dict`.
      3. Call `ast.literal_eval` on the dict node to recover the Python value.

    Raises AssertionError if DATASETS is not found or not a dict literal.
    """
    pipeline_path = PROJECT_ROOT / "data_pipeline.py"
    assert pipeline_path.exists(), f"data_pipeline.py not found at {pipeline_path}"

    source = pipeline_path.read_text(encoding="utf-8", errors="replace")
    tree = ast.parse(source, filename=str(pipeline_path))

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        # Check that exactly one target is named "DATASETS".
        targets = node.targets
        if len(targets) != 1:
            continue
        target = targets[0]
        if not (isinstance(target, ast.Name) and target.id == "DATASETS"):
            continue
        # Found the assignment — evaluate the right-hand side literally.
        try:
            value = ast.literal_eval(node.value)
        except (ValueError, TypeError) as exc:
            pytest.fail(
                f"DATASETS in data_pipeline.py contains non-literal expressions "
                f"that ast.literal_eval cannot evaluate: {exc}"
            )
        assert isinstance(value, dict), (
            f"DATASETS in data_pipeline.py evaluated to {type(value).__name__}, expected dict"
        )
        return value  # type: ignore[return-value]

    pytest.fail("DATASETS assignment not found in data_pipeline.py")


# ---------------------------------------------------------------------------
# Test 1 — No NC license strings in non-docstring source code
# ---------------------------------------------------------------------------

def test_no_nc_license_strings_in_source() -> None:
    """
    Scan all .py files in the project root (excluding tests/).
    Assert that no non-docstring string literal contains an NC license marker.

    Checked patterns (case-insensitive):
      "cc-by-nc", "noncommercial", "non-commercial", "nc-4.0", "nc-sa"

    Docstrings are excluded from assertion but are scanned for warnings —
    a docstring may reference NC licenses when explicitly explaining why
    a dependency was excluded (e.g. "TRIBE v2 is CC BY-NC — we exclude it").
    Violations in docstrings are emitted as pytest.warns, not failures.
    """
    violations: list[str] = []
    docstring_warnings: list[str] = []

    for source_file in _iter_source_files():
        try:
            tree = _parse_source(source_file)
        except SyntaxError as exc:
            pytest.fail(f"SyntaxError in {source_file.name}: {exc}")

        # Build parent map once per file for docstring detection.
        parent_map: dict[int, ast.AST] = {}
        for parent_node in ast.walk(tree):
            for child in ast.iter_child_nodes(parent_node):
                parent_map[id(child)] = parent_node

        for node in ast.walk(tree):
            if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
                continue

            value_lower = node.value.lower()
            matched = [p for p in NC_PATTERNS if p in value_lower]
            if not matched:
                continue

            # Determine if this constant is part of a docstring.
            is_docstring = False
            expr_parent = parent_map.get(id(node))
            if expr_parent is not None:
                grandparent = parent_map.get(id(expr_parent))
                if _is_docstring_node(expr_parent, grandparent):
                    is_docstring = True

            location = f"{source_file.name}:{node.lineno}"
            snippet = repr(node.value[:120])
            entry = f"  {location}  patterns={matched}  value={snippet}"

            if is_docstring:
                docstring_warnings.append(entry)
            else:
                violations.append(entry)

    if docstring_warnings:
        warnings.warn(
            "NC license strings found in docstrings (not a failure, informational only):\n"
            + "\n".join(docstring_warnings),
            UserWarning,
            stacklevel=2,
        )

    assert not violations, (
        "NC license strings found in non-docstring source code — shipping blocked:\n"
        + "\n".join(violations)
    )


# ---------------------------------------------------------------------------
# Test 2 — DATASETS dict has commercial license verification on all entries
# ---------------------------------------------------------------------------

def test_commercial_verified_flag_on_all_datasets() -> None:
    """
    Import DATASETS from data_pipeline.py and assert that every entry either:
      (a) has commercial_verified == True, OR
      (b) has a license value in {"CC0", "CC-BY-4.0", "CC-BY"}

    Both conditions are acceptable — commercial_verified may be absent when
    the license is unambiguously permissive commercial use.
    """
    datasets = _load_datasets_dict()

    assert isinstance(datasets, dict), "DATASETS must be a dict"
    assert len(datasets) > 0, "DATASETS must not be empty"

    failures: list[str] = []

    for key, entry in datasets.items():
        if not isinstance(entry, dict):
            failures.append(f"  {key!r}: entry is not a dict (got {type(entry).__name__})")
            continue

        commercial_verified: bool = entry.get("commercial_verified") is True
        license_value: str = entry.get("license", "")
        license_ok: bool = license_value in COMMERCIAL_LICENSES

        if not commercial_verified and not license_ok:
            failures.append(
                f"  {key!r}: commercial_verified={entry.get('commercial_verified')!r}, "
                f"license={license_value!r}  "
                f"(need commercial_verified=True OR license in {sorted(COMMERCIAL_LICENSES)})"
            )

    assert not failures, (
        "DATASETS entries with missing/invalid commercial license verification:\n"
        + "\n".join(failures)
    )


# ---------------------------------------------------------------------------
# Test 3 — No forbidden competitor imports in any source file
# ---------------------------------------------------------------------------

def test_no_forbidden_imports_in_all_source() -> None:
    """
    AST-parse every .py file in the project root (excluding tests/).
    Assert no `import X` or `from X import` where X starts with a forbidden
    module prefix.

    Forbidden prefixes (case-insensitive):
      "tribe", "mindeye", "brainbench", "neural_encoding_model"

    This is the hard clean-room gate — any match is an immediate shipping block.
    """
    violations: list[str] = []

    for source_file in _iter_source_files():
        try:
            tree = _parse_source(source_file)
        except SyntaxError as exc:
            pytest.fail(f"SyntaxError in {source_file.name}: {exc}")

        for lineno, module_name in _collect_imports(tree):
            module_lower = module_name.lower()
            for forbidden in FORBIDDEN_IMPORT_PREFIXES:
                if module_lower.startswith(forbidden):
                    violations.append(
                        f"  {source_file.name}:{lineno}  "
                        f"import {module_name!r}  (matches forbidden prefix {forbidden!r})"
                    )

    assert not violations, (
        "Forbidden competitor/NC-library imports detected — shipping blocked:\n"
        + "\n".join(violations)
    )


# ---------------------------------------------------------------------------
# Test 4 — requirements.txt contains no known NC packages
# ---------------------------------------------------------------------------

def test_requirements_no_nc_packages() -> None:
    """
    Read requirements.txt from the project root (if it exists).
    Assert no listed package name matches a known NC-licensed package.

    Known NC packages:
      tribe2, tribe-v2, neural_encoding_model, mindeye, brainbench

    If requirements.txt does not exist, the test passes with a warning —
    the file may not have been created yet.  The gate activates once the
    file is present.

    Matching is performed on the package name portion of each line
    (before any version specifier: ==, >=, <=, ~=, !=, [extras]).
    Comments (#) and blank lines are ignored.
    """
    req_path = PROJECT_ROOT / "requirements.txt"

    if not req_path.exists():
        warnings.warn(
            f"requirements.txt not found at {req_path} — "
            "NC package gate is inactive until the file is created.",
            UserWarning,
            stacklevel=2,
        )
        return

    violations: list[str] = []
    raw_lines = req_path.read_text(encoding="utf-8", errors="replace").splitlines()

    for lineno, raw_line in enumerate(raw_lines, start=1):
        # Strip inline comments and whitespace.
        line = raw_line.split("#")[0].strip()
        if not line:
            continue

        # Extract package name: everything before a version specifier or extras bracket.
        match = re.match(r"^([A-Za-z0-9_.\-]+)", line)
        if not match:
            continue

        pkg_name = match.group(1).lower().replace("_", "-")

        for nc_pkg in KNOWN_NC_PACKAGES:
            nc_normalised = nc_pkg.lower().replace("_", "-")
            if pkg_name == nc_normalised:
                violations.append(
                    f"  requirements.txt:{lineno}  {raw_line.strip()!r}  "
                    f"(matches known NC package {nc_pkg!r})"
                )

    assert not violations, (
        "Known NC-licensed packages found in requirements.txt — shipping blocked:\n"
        + "\n".join(violations)
    )


# ---------------------------------------------------------------------------
# Test 5 — MIT LICENSE file exists (xfail until created)
# ---------------------------------------------------------------------------

@pytest.mark.xfail(strict=False, reason="LICENSE file not yet created — create before release")
def test_mit_license_file_exists() -> None:
    """
    Assert that a LICENSE file exists at the project root or one level up.
    Marked xfail(strict=False) — will xpass once the file is created,
    and will xfail (not error) until then.

    The LICENSE file must contain MIT license text to satisfy the
    commercial-use requirement declared in the project docstrings.
    """
    candidate_paths: list[Path] = [
        PROJECT_ROOT / "LICENSE",
        PROJECT_ROOT / "LICENSE.txt",
        PROJECT_ROOT / "LICENSE.md",
        PROJECT_ROOT.parent / "LICENSE",
        PROJECT_ROOT.parent / "LICENSE.txt",
    ]

    found = [p for p in candidate_paths if p.exists()]

    assert found, (
        "No LICENSE file found. Expected one of:\n"
        + "\n".join(f"  {p}" for p in candidate_paths)
        + "\nCreate an MIT license file before shipping."
    )

    # Verify the found file contains MIT text.
    license_text = found[0].read_text(encoding="utf-8", errors="replace").upper()
    assert "MIT" in license_text, (
        f"LICENSE file at {found[0]} does not contain 'MIT'. "
        "Verify the license text is correct."
    )
