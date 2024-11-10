import os
import asyncio
from typing import List, Tuple
from data import load_data, extract_pairs
from prompt import format_prompt
from openai_utils import generate_code
from test import (
    test_code_on_all_training_pairs,
    refine_code,
    test_code,
)
from config import DATA_FOLDER

async def process_task(task_data) -> Tuple[bool, str]:
    """Process a single task and return success status and task ID."""
    train_pairs = extract_pairs(task_data)
    test_inputs = [test_case["input"] for test_case in task_data.get("test", [])]
    expected_outputs = [test_case["output"] for test_case in task_data.get("test", [])]
    
    prompt = format_prompt(train_pairs)
    solution = generate_code(prompt)
    
    # Test and refinement logic
    success, error_messages = test_code_on_all_training_pairs(solution["function"], train_pairs)
    
    if not success:
        refined_code, refined_reasoning = refine_code(
            code=solution["function"],
            train_pairs=train_pairs,
            original_reasoning=solution["reasoning"],
            error_details=error_messages
        )
        if refined_code:
            success, error_messages = test_code_on_all_training_pairs(refined_code, train_pairs)
            final_code = refined_code
        else:
            final_code = solution["function"]
    else:
        final_code = solution["function"]
    
    # Test on unseen data
    test_results = []
    if success:
        for test_input, expected_output in zip(test_inputs, expected_outputs):
            test_success, _ = test_code(final_code, test_input, expected_output)
            test_results.append(test_success)
    
    task_id = task_data.get("task_id", "unknown")
    final_success = success and all(test_results)
    return final_success, task_id

async def main():
    """Main function with concurrent execution."""
    evaluation_folder = os.path.join(DATA_FOLDER, "evaluation")
    all_data = load_data(evaluation_folder)
    
    # Create tasks for concurrent execution
    tasks = [process_task(task_data) for task_data in all_data]
    results = await asyncio.gather(*tasks)
    
    # Calculate and display accuracy
    successful_tasks = sum(1 for success, _ in results if success)
    total_tasks = len(results)
    accuracy = (successful_tasks / total_tasks) * 100 if total_tasks > 0 else 0
    
    print(f"\nOverall Results:")
    print(f"Total Tasks: {total_tasks}")
    print(f"Successful Tasks: {successful_tasks}")
    print(f"Accuracy: {accuracy:.2f}%")
    
    # Display individual results
    print("\nDetailed Results:")
    for success, task_id in results:
        status = "✓" if success else "✗"
        print(f"Task {task_id}: {status}")

if __name__ == "__main__":
    asyncio.run(main())
