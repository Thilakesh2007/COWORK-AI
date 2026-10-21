
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

messages = []

print("COWORK AI - Agent Test")
print("Try chatting or asking it to multiply numbers.")
print("Type 'exit' to quit.\n")

while True:
    user_input = input("You: ").strip()

    if user_input.lower() == "exit":
        print("Cowork AI: Goodbye!")
        break

    if not user_input:
        continue

    messages.append({"role": "user", "content": user_input})

    response = chat(
        model=MODEL,
        messages=messages,
        tools=tools,
        stream=False,
        think=False,
        options={"num_predict": 256},
    )

    messages.append(response.message)

    # Execute tools requested by the model.
    for call in response.message.tool_calls or []:
        if call.function.name == "multiply":
            args = call.function.arguments
            result = int(args["a"]) * int(args["b"])

            print(f"[Tool executed: multiply -> {result}]")

            messages.append(
                {
                    "role": "tool",
                    "tool_name": "multiply",
                    "content": str(result),
                }
            )

    # If a tool was used, let the model explain the result.
    if response.message.tool_calls:
        response = chat(
            model=MODEL,
            messages=messages,
            tools=tools,
            stream=False,
            think=False,
            options={"num_predict": 256},
        )

        messages.append(response.message)

    print(f"Cowork AI: {response.message.content or ''}\n")
