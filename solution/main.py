# main.py

from utils import load_data, extract_pairs, format_prompt, generate_code, test_code
from config import DATA_FOLDER
import os

def main():
    # Load data from evaluation folder
    evaluation_folder = os.path.join(DATA_FOLDER, "evaluation")
    all_data = load_data(evaluation_folder)
    
    # Process each evaluation task
    for task_data in all_data[:1]:
        # Get training pairs and test input
        train_pairs = extract_pairs(task_data)
        test_input = task_data['test'][0]['input']
        expected_output = task_data['test'][0]['output']
        
        # Generate initial code using zero-shot prompting with all training examples
        prompt = format_prompt(train_pairs, test_input)
        
        print("\nGenerated Prompt:")
        print(prompt)
        
        solution = generate_code(prompt)
        print("\nPattern Analysis:")
        print(solution['reasoning'])
        print("\nGenerated Solution:")
        print(solution['function'])
        
        # Test the solution
        success, error_message = test_code(solution['function'], test_input, expected_output)
        if success:
            print("\nSolution is correct!")
        else:
            print("\nSolution is incorrect:")
            print(error_message)
        
        print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    main()
