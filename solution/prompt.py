from typing import List, Tuple, Any
from config import ZERO_SHOT_PROMPT, REFINEMENT_PROMPT


def grid_to_ascii(grid: List[List[int]]) -> str:
    """Convert a grid of integers to an ASCII string representation."""
    return "\n".join(["".join(str(cell) for cell in row) for row in grid])


def format_prompt(
    train_pairs: List[Tuple[List[List[int]], List[List[int]]]]
) -> str:
    """Format the prompt for the OpenAI API."""
    # Format training examples
    training_examples = []
    for i, (input_grid, output_grid) in enumerate(train_pairs, 1):
        example = f"""Example {i}:
Input:
{grid_to_ascii(input_grid)}

Output:
{grid_to_ascii(output_grid)}
"""
        training_examples.append(example)

    # Join all examples with newlines
    all_examples = "\n".join(training_examples)

    # Return the formatted prompt
    return ZERO_SHOT_PROMPT.format(training_examples=all_examples)


def format_refinement_prompt(
    code: str,
    error_details: List[Tuple[str, List[List[int]], List[List[int]], Any]],
    original_reasoning: str,
) -> str:
    """Format the refinement prompt for the OpenAI API."""
    # Format training examples with actual outputs
    training_examples = []
    for i, (error_message, input_grid, expected_output, actual_output) in enumerate(error_details, 1):
        # Handle cases where actual_output is None
        if actual_output is not None:
            actual_output_str = grid_to_ascii(actual_output)
        else:
            actual_output_str = "Error or None"

        example = f"""Example {i}:
Input:
{grid_to_ascii(input_grid)}

Expected Output:
{grid_to_ascii(expected_output)}

Actual Output:
{actual_output_str}

Error Message:
{error_message}
"""
        training_examples.append(example)

    # Join all examples with newlines
    all_examples = "\n".join(training_examples)

    # Return the formatted prompt
    return REFINEMENT_PROMPT.format(
        original_reasoning=original_reasoning,
        code=code,
        training_examples=all_examples,
    )