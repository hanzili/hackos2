import json
from openai import OpenAI
from config import OPENAI_API_KEY

# Set up OpenAI API key
client = OpenAI(api_key=OPENAI_API_KEY)

def generate_code(prompt):
    """Generate code using the OpenAI API and parse the JSON response."""

    # Define the function schema
    function = {
        "name": "provide_solution",
        "description": "Provide the reasoning and function code for transforming the input grid.",
        "parameters": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": "Explanation of the approach used to solve the problem."
                },
                "function_code": {
                    "type": "string",
                    "description": "Python function code implementing the solution."
                }
            },
            "required": ["reasoning", "function_code"],
        },
    }

    response = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[
            {"role": "system", "content": "You are a helpful assistant that provides reasoning and code solutions."},
            {"role": "user", "content": prompt},
        ],
        functions=[function],
        function_call={"name": "provide_solution"},
    )

    message = response.choices[0].message

    if message.function_call:
        # Extract the arguments provided in the function call
        function_args = json.loads(message.function_call.arguments)
        reasoning = function_args.get('reasoning', '').strip()
        function = function_args.get('function', '').strip()

        return {'reasoning': reasoning, 'function': function}
    else:
        raise ValueError("Assistant did not return function call as expected.")