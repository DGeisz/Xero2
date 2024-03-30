# %%
import gradio as gr
from setup import *

# %%


def greet(name):
    return "Nope " + name


iface = gr.Interface(fn=greet, inputs="text", outputs="text")

iface.launch(inline=True, share=True)


# %%
def image_classifier(inp):
    return {"cat": 0.3, "dog": 0.7}


demo = gr.Interface(fn=image_classifier, inputs="image", outputs="label")
demo.launch(inline=True)


# %%
def greet(name, intensity):
    return "Hello, " + name + "!" * int(intensity)


demo = gr.Interface(
    fn=greet,
    inputs=["text", "slider"],
    outputs=["text"],
)

demo.launch()


# %%
def calculator(num1, operation, num2):
    if operation == "add":
        return num1 + num2
    elif operation == "subtract":
        return num1 - num2
    elif operation == "multiply":
        return num1 * num2
    elif operation == "divide":
        return num1 / num2


demo = gr.Interface(
    calculator,
    ["number", gr.Radio(["add", "subtract", "multiply", "divide"]), "number"],
    "number",
    live=True,
)
demo.launch()


# %%
class Hello:
    def __init__(self):
        self.name = "Danny"

    def _repr_html_(self):
        return f"<h1>Hello, {self.name}!</h1>"


# %%
Hello()


# %%
cv.tokens.colored_tokens_multi(
    ["hell", "there", ""],
    values=t.tensor([[0, 0, 0], [0, 0, 0], [1, 0, 0]]),
    labels=["", "Winner 1", "Winner 2"],
)

# %%
