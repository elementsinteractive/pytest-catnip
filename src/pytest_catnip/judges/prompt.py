def get_prompt(actual: str, expected: str) -> str:
    return (
        f"You are a strict QA testing judge. You will evaluate the behavior of a bot.\n"
        f"Expectation: {expected}\n"
        f"Actual Bot Reply: {actual}\n\n"
        f"Evaluate if the bot reply satisfies the expectation."
    )
