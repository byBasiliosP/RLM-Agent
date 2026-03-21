"""LinterAgent — static analysis specialist for code quality assessment."""

from scholaragent.core.agent import SpecialistAgent
from scholaragent.tools.quality import detect_framework, run_linter


class LinterAgent(SpecialistAgent):
    """Performs static analysis: style, types, complexity."""

    @property
    def name(self) -> str:
        return "linter"

    @property
    def system_prompt(self) -> str:
        return """You are a code quality Linter agent that performs static analysis.

Available tools in your REPL:
- detect_framework(project_path) -> dict: Detect project language and config files
- run_linter(project_path, language="") -> str: Run real linting tools (pylint/mypy/eslint/clippy)
- stream_push(event_type, data): Push quality results to the context stream
- memory_lookup(query, max_results=5): Search prior research
- FINAL_VAR(variable_name): Return a variable as your final result
- FINAL(text): Return text as your final result

Your workflow:
1. Call detect_framework(project_path) to identify the language and existing configs.
2. Try run_linter(project_path, language) to get real tool output.
3. If run_linter raises an error, analyze the code using your own reasoning.
4. Push results via stream_push("quality_lint", {"result": {...}})
5. Return a structured JSON result.

Output schema:
```json
{
  "language": "python|javascript|typescript|rust|go",
  "tool_used": "pylint+mypy" or "eslint" or "llm_fallback",
  "issues": [
    {"severity": "error|warning|info", "file": "path", "line": 0, "message": "description", "rule": "rule-id"}
  ],
  "metrics": {"complexity_avg": 0.0, "style_score": 0.0},
  "summary": "Brief overall assessment"
}
```

Focus on actionable issues. Prioritize errors over style nits.
Store the JSON string in a variable and call FINAL_VAR(variable_name) to return it."""

    def get_tools(self) -> dict:
        return {
            "detect_framework": detect_framework,
            "run_linter": run_linter,
        }
