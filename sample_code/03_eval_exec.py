# Sample 3: Dangerous eval() and exec() Usage

def calculate_expression(user_input):
    # CRITICAL: eval allows arbitrary code execution
    result = eval(user_input)
    return result


def execute_dynamic_script(script_body):
    # CRITICAL: exec executes dynamic code strings
    exec(script_body)


if __name__ == "__main__":
    calculate_expression("2 + 2")
