# OpenAI Configuration 
OPENAI_API_KEY = ""
MODEL = "gpt-4o-mini"
MAX_TOKENS = 300
TEMPERATURE = 0.5

# Data Configuration
DATA_FOLDER = "data"

# Prompt Templates
ZERO_SHOT_PROMPT = """
You are solving an Abstraction and Reasoning Challenge. Your task is to analyze and determine the pattern that transforms the input grids into their corresponding output grids.

Training Examples:
{training_examples}

Analyze the pattern carefully and provide:
1. A detailed explanation of the transformation pattern you've identified
2. A Python function that implements this pattern

Return your response as a JSON object with two keys:
- "reasoning": A clear explanation of the pattern you found
- "function": The function code only, without imports:
    def transform_grid(input_grid):
        '''
        Args:
            input_grid (List[List[int]]): Input grid represented as 2D list of integers
        Returns:
            List[List[int]]: Transformed output grid
        '''
        # Your transformation logic here
        return output_grid

Rules and Constraints:
- Grids contain integers from 0-9 (representing different colors)
- The transformation should work for any similar input following the same pattern
- The solution should be generalizable
- Output dimensions must match exactly
- All cell values must match exactly
- The input and output matrix sizes will be the same as shown in the training examples
"""

REFINEMENT_PROMPT = """
The following solution for the Abstraction and Reasoning Challenge needs improvement:

Original Reasoning:
{original_reasoning}

Original Code:
{code}

Training Data:
{training_examples}

Analysis Requirements:
1. Analyze what was incorrect about the original reasoning
2. Fix the specific errors identified above
3. Compare the current outputs with the expected outputs for all training examples
4. Identify where the transformation logic fails
5. Consider edge cases and pattern variations
6. Ensure the solution remains generalizable

Return your response as a JSON object with two keys:
- "reasoning": Analysis of why the original reasoning was wrong and explanation of the correct pattern
- "function": The corrected function code only, without imports

Rules and Constraints:
- Grids contain integers from 0-9 (representing different colors)
- The transformation should work for any similar input following the same pattern
- Output dimensions must match exactly
- All cell values must match exactly
- The input and output matrix sizes will be the same as shown in the training examples
"""
