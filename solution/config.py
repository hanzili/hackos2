# config.py

import os

# OpenAI Configuration
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_KEY = "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"
MODEL = "gpt-4"
MAX_TOKENS = 300
TEMPERATURE = 0.5
NUM_SOLUTIONS = 3

# Data Configuration
DATA_FOLDER = "data"

# Prompt Templates
ZERO_SHOT_PROMPT = """
You are solving an Abstraction and Reasoning Challenge. Your task is to analyze and determine the pattern that transforms the input grids into their corresponding output grids.

Training Examples:
{training_examples}

Test Input Grid:
{input_grid}

Analyze the pattern carefully and provide:
1. A detailed explanation of the transformation pattern you've identified
2. A Python function that implements this pattern

Return your response as a JSON object with two keys:
- "reasoning": A clear explanation of the pattern you found
- "function": The Python function code in the following format:
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
"""

REFINEMENT_PROMPT = """
The following solution for the Abstraction and Reasoning Challenge needs improvement:

Original Code:
{code}

Input Grid:
{input_grid}

Expected Output Grid:
{output_grid}

Error Information:
{error_info}

Analysis Requirements:
1. Fix the specific error identified above
2. Compare the current output with the expected output
3. Identify where the transformation logic fails
4. Consider edge cases and pattern variations
5. Ensure the solution remains generalizable

Return your response as a JSON object with two keys:
- "reasoning": Analysis of what was wrong and how you fixed it
- "function": The corrected code that implements the pattern

Rules and Constraints:
- Grids contain integers from 0-9 (representing different colors)
- The transformation should work for any similar input following the same pattern
- Output dimensions must match exactly
- All cell values must match exactly
"""
