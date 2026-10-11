# Sample 6: Clean, Idiomatic Python Code (No AST or Security Violations)

import os
from typing import List, Optional


def calculate_average(numbers: List[float]) -> Optional[float]:
    """Calculates average of a list of numbers safely."""
    if not numbers:
        return None
    return sum(numbers) / len(numbers)


def get_environment_secret() -> str:
    """Retrieves secret securely from environment variables."""
    secret = os.getenv("API_SECRET_KEY", "")
    if not secret:
        raise ValueError("API_SECRET_KEY environment variable is not set.")
    return secret


if __name__ == "__main__":
    nums = [10.0, 20.0, 30.0]
    avg = calculate_average(nums)
    if avg is not None:
        print(f"Average: {avg:.2f}")
