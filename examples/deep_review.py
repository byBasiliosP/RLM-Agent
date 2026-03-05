"""Deep literature review example with custom configuration.

Run: python examples/deep_review.py

Requires: OPENAI_API_KEY and ANTHROPIC_API_KEY environment variables.
"""

from scholaragent import ScholarAgent


def main():
    # Use stronger models for a thorough review
    agent = ScholarAgent(
        strong_model={"backend": "anthropic", "model_name": "claude-sonnet-4-6"},
        cheap_model={"backend": "openai", "model_name": "gpt-4o-mini"},
        max_papers=15,
        max_iterations=20,
        verbose=True,
    )

    queries = [
        "How do large language models handle mathematical reasoning?",
        "What are recent approaches to reducing hallucination in LLMs?",
    ]

    for query in queries:
        print(f"\n{'='*80}")
        print(f"Research Query: {query}")
        print(f"{'='*80}\n")

        result = agent.research(query)

        print(f"\nSuccess: {result.success}")
        print(f"Iterations: {result.iterations}")
        print(f"\nFindings:\n{result.result}")
        print()


if __name__ == "__main__":
    main()
