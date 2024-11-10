import subprocess
from typing import List, Tuple, Any
from prompt import grid_to_ascii, format_refinement_prompt
from openai_utils import generate_code


def test_code_on_all_training_pairs(
    code: str, train_pairs: List[Tuple[List[List[int]], List[List[int]]]]
) -> Tuple[bool, List[Tuple[str, List[List[int]], List[List[int]], Any]]]:
    """Test the generated code on all training input-output pairs."""
    error_details = []
    all_passed = True
    for idx, (input_grid, expected_output) in enumerate(train_pairs, 1):
        success, output, error_message = test_code(code, input_grid, expected_output)
        if not success:
            all_passed = False
            error_details.append((
                f"Example {idx} failed:\n{error_message}",
                input_grid,
                expected_output,
                output  # Note that output can be None if there was an error
            ))
    return all_passed, error_details


def test_code(
    code: str, input_grid: List[List[int]], expected_output: List[List[int]]
) -> Tuple[bool, Any, str]:
    """Test the generated code and return (success, output, error_message)."""
    # Prepare the code to run
    code_to_run = f"""
{code}

# Test the function
input_grid = {input_grid}
try:
    output = transform_grid(input_grid)
    print(output)
except Exception as e:
    print(f"Error: {{str(e)}}")
"""

    try:
        # Run the code in a subprocess
        process = subprocess.run(
            ["python3", "-c", code_to_run],
            text=True,
            capture_output=True,
            timeout=5,
        )

        stderr = process.stderr.strip()
        stdout = process.stdout.strip()

        if stderr:
            return False, None, f"Runtime Error: {stderr}"

        if "Error:" in stdout:
            return False, None, stdout

        # Evaluate the output
        output = eval(stdout)
        if output == expected_output:
            return True, output, "Success"
        else:
            error_message = f"Output mismatch:\nGot: {output}\nExpected: {expected_output}"
            return False, output, error_message

    except subprocess.TimeoutExpired:
        return False, None, "Execution timed out"
    except Exception as e:
        return False, None, f"Execution Error: {str(e)}"


def refine_code(
    code: str,
    train_pairs: List[Tuple[List[List[int]], List[List[int]]]],
    original_reasoning: str,
    error_details: List[Tuple[str, List[List[int]], List[List[int]], Any]]
) -> Tuple[str, str]:
    """Refine the solution using all training input-output pairs."""

    # Generate refinement prompt
    prompt = format_refinement_prompt(
        code=code,
        error_details=error_details,
        original_reasoning=original_reasoning,
    )

    print(prompt)

    # Generate refined code
    refined_solution = generate_code(prompt)
    refined_code = refined_solution.get("function", "")
    refined_reasoning = refined_solution.get("reasoning", "")

    # Test refined code on training data
    success, _ = test_code_on_all_training_pairs(refined_code, train_pairs)
    if success:
        return refined_code, refined_reasoning
    else:
        return "", ""  # Indicate failure to refine