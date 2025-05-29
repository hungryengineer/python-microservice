import gradio as gr
from llama_cpp import Llama

# Load Mistral model (make sure .gguf model file is in the same folder or mounted in Docker)
llm = Llama(
    model_path="models/TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf",
    n_ctx=1024,
    n_threads=4
)

def chat(prompt, history=[]):
    formatted_prompt = f"[INST] {prompt} [/INST]"
    result = llm(formatted_prompt, max_tokens=256)
    reply = result["choices"][0]["text"]
    return history + [[prompt, reply]]

chat_ui = gr.ChatInterface(fn=chat, title="Mistral Chat")
chat_ui.launch(server_name="0.0.0.0", server_port=7860)
