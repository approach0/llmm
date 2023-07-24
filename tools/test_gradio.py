import gradio as gr

def echo(prompt):
    return prompt

iface = gr.Interface(
    fn=echo,
    inputs=gr.Textbox('foo bar'),
    outputs="text"
)

iface.queue().launch(server_port=8922,
    debug=True, share=True, inline=False
)
