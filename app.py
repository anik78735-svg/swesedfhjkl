import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import gc

# Memory optimization - NO hf_transfer
os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
# os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"  # <-- COMMENT THIS OUT
os.environ["OMP_NUM_THREADS"] = "1"

# 🔥 GPT-2 Small - 124M parameters
MODEL_PATH = "gpt2"

print("🔄 Loading GPT-2 Small model...")

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="cpu",
        low_cpu_mem_usage=True
    )
    model.eval()
    print("✅ GPT-2 model loaded successfully!")
    print(f"✅ Model size: {model.num_parameters() / 1e6:.1f}M parameters")
    
except Exception as e:
    print(f"❌ Model load failed: {e}")
    raise

def predict(message, history, temperature, top_p, max_tokens):
    try:
        # Build conversation history
        context = ""
        for human, assistant in history:
            context += f"Human: {human}\nAssistant: {assistant}\n"
        context += f"Human: {message}\nAssistant:"
        
        inputs = tokenizer(
            context, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512
        )
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=min(max_tokens, 256),
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        # Decode only new tokens
        generated_ids = outputs[0][len(inputs.input_ids[0]):]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Clean response
        response = response.strip()
        if "Assistant:" in response:
            response = response.split("Assistant:")[-1].strip()
        
        # Cleanup memory
        del outputs
        gc.collect()
        
        return response if response else "Sorry, I couldn't generate a response."
        
    except Exception as e:
        return f"⚠️ Error: {str(e)}"

# Gradio Interface
with gr.Blocks(title="GPT-2 Chat", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🤖 GPT-2 Small - AI Assistant
    *Lightweight AI model (124M parameters) optimized for Render free tier*
    """)
    
    chatbot = gr.Chatbot(height=450)
    msg = gr.Textbox(
        placeholder="Ask me anything...",
        container=False,
        scale=9
    )
    
    with gr.Row():
        clear = gr.Button("🗑️ Clear Chat", variant="secondary")
        temperature = gr.Slider(0.1, 1.0, value=0.7, label="Temperature", step=0.1)
        top_p = gr.Slider(0.1, 1.0, value=0.9, label="Top-p", step=0.1)
        max_tokens = gr.Slider(32, 256, value=128, step=32, label="Max Tokens")
    
    def respond(message, history):
        if not message.strip():
            return history
        try:
            response = predict(message, history, temperature.value, top_p.value, max_tokens.value)
            return history + [(message, response)]
        except Exception as e:
            return history + [(message, f"⚠️ Error: {str(e)}")]
    
    # Event handlers
    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0", 
        server_port=7860,
        share=False
    )
