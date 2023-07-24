import gradio as gr
from prompt_factory import *

def echo(prompt):
    return prompt

prompt_template = ia1('<your question>', ['<search evidence>'])

iface = gr.Interface(
    fn=echo,
    inputs=gr.Textbox(prompt_template, lines=40),
    outputs=gr.Textbox(lines=40)
)

iface.queue().launch(server_port=8922,
    debug=True, share=True, inline=False
)
