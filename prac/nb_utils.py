import inspect


def show(value):
    frame = inspect.currentframe().f_back
    line = inspect.getframeinfo(frame).code_context[0].strip()
    name = line.removeprefix("show(").removesuffix(")")
    print(f"{name}:\n{value}")
