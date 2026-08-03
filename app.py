import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "Nanbeige/Nanbeige4.1-3B"

print("🔄 Loading model... This may take a few minutes.")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    use_fast=False,
    trust_remote_code=True
)

# 🔥 4-bit quantization se memory kam use hogi
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,  # bfloat16 se float16 (lightweight)
    device_map="auto",
    trust_remote_code=True,
    load_in_4bit=True,  # ✅ Memory saving
    bnb_4bit_compute_dtype=torch.float16
)
model.eval()
print("✅ Model loaded successfully!")

def predict(message, history, temperature, top_p, max_tokens):
    messages = []
    for human, assistant in history:
        messages.append({"role": "user", "content": human})
        messages.append({"role": "assistant", "content": assistant})
    messages.append({"role": "user", "content": message})
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    generated_ids = outputs[0][len(inputs.input_ids[0]):]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return response

with gr.Blocks(title="Nanbeige4.1-3B Chat") as demo:
    gr.Markdown("""
    # 🤖 Nanbeige4.1-3B - AI Assistant
    *A 3B parameter model with strong reasoning and code generation capabilities*
    """)
    
    chatbot = gr.Chatbot(height=400)
    msg = gr.Textbox(
        placeholder="Ask me anything...",
        container=False,
        scale=9
    )
    
    with gr.Row():
        clear = gr.Button("Clear Chat")
        temperature = gr.Slider(0.0, 1.0, value=0.6, label="Temperature")
        top_p = gr.Slider(0.0, 1.0, value=0.95, label="Top-p")
        max_tokens = gr.Slider(128, 2048, value=512, step=128, label="Max Tokens")
    
    def respond(message, history):
        return history + [(message, predict(message, history, temperature.value, top_p.value, max_tokens.value))]
    
    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
