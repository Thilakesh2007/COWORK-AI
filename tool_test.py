
from ollama import chat

MODEL = "qwen3.5:9b"

tools = [
    {
        "type": "function",
        "function": {
            "name": "multiply",
            "description": "Multiply two integers together.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "integer"},
                    "b": {"type": "integer"},
                },
                "required": ["a", "b"],
            },
        },
    }
]

messages = [
    {
        "role": "user",
        "content": (
            "Use the multiply tool to calculate "
            "17 multiplied by 23, then tell me the result."
        ),
    }
]

print("Cowork AI: Thinking...")

# Ask the model to choose a tool.
response = chat(
    model=MODEL,
    messages=messages,
    tools=tools,
    stream=False,
    think=False,
    options={"num_predict": 128},
)

messages.append(response.message)

# Execute the tool requested by the model.
for call in response.message.tool_calls or []:
    if call.function.name == "multiply":
        args = call.function.arguments

        result = int(args["a"]) * int(args["b"])

        print(f"Python executed multiply: {result}")

        messages.append(
            {
                "role": "tool",
                "tool_name": "multiply",
                "content": str(result),
            }
        )

# Give the tool result back to the model.
final_response = chat(
    model=MODEL,
    messages=messages,
    tools=tools,
    stream=False,
    think=False,
    options={"num_predict": 128},
)

print("\nCowork AI:", final_response.message.content)
