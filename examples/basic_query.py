"""Basic ScholarAgent usage example.

Run: python examples/basic_query.py

Requires: OPENAI_API_KEY and ANTHROPIC_API_KEY environment variables.
"""

from scholaragent import ScholarAgent


def main():
    agent = ScholarAgent(
        strong_model={"backend": "anthropic", "model_name": "claude-sonnet-4-6"},
        cheap_model={"backend": "openai", "model_name": "gpt-4o-mini"},
        max_papers=5,
        verbose=True,
    )

    print(f"Agent: {agent}")
    print()

    result = agent.research(
        "What are the latest advances in protein structure prediction using diffusion models?"
    )

    print(f"\nSuccess: {result.success}")
    print(f"Iterations: {result.iterations}")
    print(f"\n{'='*80}")
    print(result.result)


if __name__ == "__main__":
    main()
