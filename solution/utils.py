import os
import json
from openai import OpenAI
import subprocess
from config import OPENAI_API_KEY, MODEL, MAX_TOKENS, TEMPERATURE, ZERO_SHOT_PROMPT, REFINEMENT_PROMPT

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
    
    if test_input:
        # Format the complete prompt with test input
        return ZERO_SHOT_PROMPT.format(
            training_examples=all_examples,
            input_grid=grid_to_ascii(test_input)
        )
    else:
        # Use prompt without test input for initial code generation
        return ZERO_SHOT_PROMPT.format(
            training_examples=all_examples,
            input_grid=""
        )

def format_refinement_prompt(code, train_pairs, original_reasoning, error_info=None):
    if error_info:
        error_info_str = f"\nError Information:\n{error_info}"
    else:
        error_info_str = ""
    
    # Format training examples
    training_examples = []
    for i, (input_grid, output_grid) in enumerate(train_pairs, 1):
        example = f"""Input Grid {i}:
{grid_to_ascii(input_grid)}

Expected Output Grid {i}:
{grid_to_ascii(output_grid)}
"""
        training_examples.append(example)
    
    training_data_str = "\n".join(training_examples)
    
    return REFINEMENT_PROMPT.format(
        original_reasoning=original_reasoning,
        code=code,
        training_data=training_data_str,
        error_info=error_info_str
    )

def generate_code(prompt):
    """Generate code using the OpenAI API and parse the JSON response."""
    response = client.chat.completions.create(
        model='gpt-4',
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

# Test generated code on all training pairs
def test_code_on_all_training_pairs(code, train_pairs):
    """Test the generated code on all training input-output pairs."""
    error_messages = []
    for idx, (input_grid, expected_output) in enumerate(train_pairs, 1):
        success, error_message = test_code(code, input_grid, expected_output)
        if not success:
            error_messages.append(f"Example {idx} failed:\n{error_message}")
    if error_messages:
        return False, error_messages
    return True, None

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
def refine_code(code, train_pairs, original_reasoning):
    """Refine the solution using all training input-output pairs."""
    error_messages = []
    for idx, (input_grid, expected_output) in enumerate(train_pairs, 1):
        success, error_message = test_code(code, input_grid, expected_output)
        if not success:
            error_messages.append(f"Example {idx} failed:\n{error_message}")
    
    if not error_messages:
        # Code already correct on all training examples
        return code, None
    
    error_info = "\n".join(error_messages)
    
    # Generate refinement prompt
    prompt = format_refinement_prompt(
        code=code,
        train_pairs=train_pairs,
        original_reasoning=original_reasoning,
        error_info=error_info
    )
    
    refined_solution = generate_code(prompt)
    # Test refined code on all training pairs
    success, _ = test_code_on_all_training_pairs(refined_solution['function'], train_pairs)
    if success:
        return refined_solution['function'], refined_solution['reasoning']
    
    return None, None
