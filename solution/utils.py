# utils.py

import os
import json
from openai import OpenAI
import subprocess
from config import OPENAI_API_KEY, MODEL, MAX_TOKENS, TEMPERATURE, NUM_SOLUTIONS, ZERO_SHOT_PROMPT, REFINEMENT_PROMPT

client = OpenAI(api_key=OPENAI_API_KEY)

# Load JSON files
def load_data(data_folder):
    all_data = []
    for root, _, files in os.walk(data_folder):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    all_data.append(data)
    return all_data

def extract_pairs(json_data):
    train_pairs = [(entry['input'], entry['output']) for entry in json_data.get('train', [])]
    return train_pairs

# Convert grids to ASCII
def grid_to_ascii(grid):
    return '\n'.join([''.join(str(cell) for cell in row) for row in grid])

def format_prompt(train_pairs, test_input):
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
    
    # Format the complete prompt
    return ZERO_SHOT_PROMPT.format(
        training_examples=all_examples,
        input_grid=grid_to_ascii(test_input)
    )

def format_refinement_prompt(code, input_grid, output_grid, error_info=None):
    if error_info:
        error_info_str = f"\nError Info: {error_info}"
    else:
        error_info_str = ""
    return REFINEMENT_PROMPT.format(
        code=code,
        input_grid=grid_to_ascii(input_grid),
        output_grid=grid_to_ascii(output_grid),
        error_info=error_info_str
    )

def generate_code(prompt):
    """Generate code using the latest OpenAI API and parse the JSON response."""
    response = client.chat.completions.create(
        model='gpt-3.5-turbo',
        messages=[
            {"role": "system", "content": prompt},
        ]
    )
    
    try:
        # Parse the JSON response
        result = json.loads(response.choices[0].message.content)
        
        # Validate that both required keys exist
        if not all(key in result for key in ['reasoning', 'function']):
            raise ValueError("Response missing required keys 'reasoning' or 'function'")
        
        return {
            'reasoning': result['reasoning'].strip(),
            'function': result['function'].strip()
        }
    except json.JSONDecodeError:
        raise ValueError("Failed to parse GPT response as JSON")

# Test generated code
def test_code(code, input_grid, expected_output):
    """Test the generated code and return (success, error_message)."""
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
    
    print(code_to_run)
    try:
        process = subprocess.run(
            ["python3", "-c", code_to_run],
            text=True,
            capture_output=True,
            timeout=5
        )
        
        if process.stderr:
            return False, f"Runtime Error: {process.stderr}"
        
        if "Error:" in process.stdout:
            return False, process.stdout.strip()
            
        output = eval(process.stdout.strip())
        if output == expected_output:
            return True, "Success"
            
        return False, f"Output mismatch:\nGot: {output}\nExpected: {expected_output}"
    except Exception as e:
        return False, f"Execution Error: {str(e)}"

# Refine solutions
def refine_code(code, input_grid, output_grid):
    """Refine a single solution, including error information in refinement prompt."""
    success, error_message = test_code(code, input_grid, output_grid)
    if success:
        return code

    # If incorrect, refine using GPT-4 with error information
    prompt = format_refinement_prompt(
        code=code,
        input_grid=input_grid,
        output_grid=output_grid,
        error_info=error_message
    )
    
    refined_solution = generate_code(prompt)
    success, _ = test_code(refined_solution['function'], input_grid, output_grid)
    if success:
        return refined_solution['function']
    
    return None
