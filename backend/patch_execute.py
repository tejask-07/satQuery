with open("app/api/routes_query.py", "r") as f:
    content = f.read()

target = """
    execution_results = execute_plan(
        tools
    )
"""
new = """
    execution_results = execute_plan(
        tools,
        context=plan.model_dump()
    )
"""
content = content.replace(target, new)
with open("app/api/routes_query.py", "w") as f:
    f.write(content)
print("done3")
