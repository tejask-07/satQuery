from app.agent.registry import get_tool


def execute_plan(tools: list[str], context: dict | None = None):
    """
    Execute tools sequentially.

    Each tool receives the accumulated context
    from previous steps.
    """

    context = context or {}

    results = {}

    for tool_name in tools:

        print(f"Executing: {tool_name}")

        tool = get_tool(tool_name)

        result = tool(**context)

        results[tool_name] = result

        # Make previous result available to the next tool
        context[tool_name] = result

    return results