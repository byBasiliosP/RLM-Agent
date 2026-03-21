"""Framework detection and hybrid tool runner utilities for code quality agents."""

import json
import subprocess
from collections.abc import Callable
from pathlib import Path


# Files that indicate a specific language/framework
_BUILD_FILES = {
    "pyproject.toml": "python",
    "setup.py": "python",
    "setup.cfg": "python",
    "package.json": "javascript",
    "Cargo.toml": "rust",
    "go.mod": "go",
}

_LINTER_CONFIGS = frozenset({
    ".pylintrc", ".flake8", "mypy.ini", ".mypy.ini",
    ".eslintrc", ".eslintrc.json", ".eslintrc.js", ".eslintrc.yml",
    "clippy.toml", ".clippy.toml",
    "golangci-lint.yml", ".golangci.yml",
})

_TEST_CONFIGS = frozenset({
    "pytest.ini", "conftest.py", "tox.ini",
    "jest.config.js", "jest.config.ts", "jest.config.json",
    "vitest.config.ts", "vitest.config.js",
})


def detect_framework(project_path: str) -> dict:
    """Detect the language/framework of a project by scanning for build files."""
    path = Path(project_path)
    language = "unknown"
    build_file = ""
    linter_configs: list[str] = []
    test_configs: list[str] = []

    for fname, lang in _BUILD_FILES.items():
        if (path / fname).exists():
            language = lang
            build_file = fname
            break

    # Refine javascript -> typescript
    if language == "javascript" and (path / "tsconfig.json").exists():
        language = "typescript"

    if language == "javascript":
        pkg_json = path / "package.json"
        if pkg_json.exists():
            try:
                pkg = json.loads(pkg_json.read_text())
                deps = {**pkg.get("devDependencies", {}), **pkg.get("dependencies", {})}
                if "typescript" in deps:
                    language = "typescript"
            except (json.JSONDecodeError, OSError):
                pass

    # Collect linter and test configs
    try:
        for item in path.iterdir():
            if item.name in _LINTER_CONFIGS:
                linter_configs.append(item.name)
            if item.name in _TEST_CONFIGS:
                test_configs.append(item.name)
    except OSError:
        pass

    return {
        "language": language,
        "build_file": build_file,
        "linter_configs": sorted(linter_configs),
        "test_configs": sorted(test_configs),
        "project_path": str(path),
    }


def run_tool_or_fallback(
    tool_fn: Callable | None,
    fallback_label: str,
) -> tuple[str, bool]:
    """Try running a tool function; return (output, True) on success or (fallback_label, False) on failure."""
    if tool_fn is None:
        return fallback_label, False
    try:
        result = tool_fn()
        return result, True
    except Exception:
        return fallback_label, False


def run_linter(project_path: str, language: str = "") -> str:
    """Run language-appropriate linters. Returns combined stdout or raises."""
    if not language:
        info = detect_framework(project_path)
        language = info["language"]

    outputs = []
    if language == "python":
        for cmd in [
            ["python", "-m", "pylint", "--score=no", "--output-format=text", project_path],
            ["python", "-m", "mypy", project_path, "--no-error-summary"],
        ]:
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=30, cwd=project_path)
                outputs.append(r.stdout or r.stderr)
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass
    elif language in ("javascript", "typescript"):
        try:
            r = subprocess.run(["npx", "eslint", ".", "--format=compact"],
                               capture_output=True, text=True, timeout=30, cwd=project_path)
            outputs.append(r.stdout or r.stderr)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
    elif language == "rust":
        try:
            r = subprocess.run(["cargo", "clippy", "--message-format=short"],
                               capture_output=True, text=True, timeout=60, cwd=project_path)
            outputs.append(r.stdout or r.stderr)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
    elif language == "go":
        try:
            r = subprocess.run(["golangci-lint", "run", "--out-format=line-number"],
                               capture_output=True, text=True, timeout=30, cwd=project_path)
            outputs.append(r.stdout or r.stderr)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    if not outputs or all(not o.strip() for o in outputs):
        raise FileNotFoundError(f"No linter available for {language}")
    return "\n".join(o for o in outputs if o.strip())


def discover_tests(project_path: str, language: str = "") -> str:
    """Discover test files/cases using framework tools. Returns stdout or raises."""
    if not language:
        info = detect_framework(project_path)
        language = info["language"]

    if language == "python":
        r = subprocess.run(["python", "-m", "pytest", "--collect-only", "-q"],
                           capture_output=True, text=True, timeout=30, cwd=project_path)
        if r.returncode == 0 or r.stdout.strip():
            return r.stdout
        raise FileNotFoundError("pytest not available")
    elif language in ("javascript", "typescript"):
        r = subprocess.run(["npx", "jest", "--listTests"],
                           capture_output=True, text=True, timeout=30, cwd=project_path)
        if r.returncode == 0 or r.stdout.strip():
            return r.stdout
        raise FileNotFoundError("jest not available")
    elif language == "rust":
        r = subprocess.run(["cargo", "test", "--", "--list"],
                           capture_output=True, text=True, timeout=30, cwd=project_path)
        if r.returncode == 0 or r.stdout.strip():
            return r.stdout
        raise FileNotFoundError("cargo test not available")

    raise FileNotFoundError(f"No test discovery for {language}")
