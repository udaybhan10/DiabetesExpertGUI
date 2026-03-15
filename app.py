import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

# Configuration
# For HF Spaces/CPU/GPU generic, we use the standard hub models
HF_BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct" 
ADAPTER_ID = "udaybhan10/Diabetes-Expert-Llama-3.2-3B-MLX"

# Medical Disclaimer
DISCLAIMER = """
<div style='background-color: #fff3f3; border: 1px solid #ffc1c1; padding: 15px; border-radius: 5px; margin-bottom: 20px;'>
    <h3 style='color: #d93025; margin-top: 0;'>⚕️ MEDICAL DISCLAIMER</h3>
    <p style='font-size: 0.9em; line-height: 1.4;'>
        This AI model (Diabetes Expert SLM) is for <b>educational and research purposes only</b>. 
        It is not a medical device and should not be used to diagnose, treat, or manage any medical condition. 
        Always consult a qualified healthcare provider for clinical decisions.
    </p>
</div>
"""

SYSTEM_PROMPT = (
    "You are the 'Diabetes Expert SLM'. Provide clinical information grounded ONLY in official 2026 ADA Standards. "
    "For diagnosis, prioritize A1C (≥6.5%), FPG (≥126 mg/dL), and OGTT (≥200 mg/dL). "
    "Always use professional, concise clinical language. "
    "If a query is outside your expertise or requires immediate medical intervention, advise consulting a doctor immediately."
)

def load_model():
    print("Loading model and adapters...")
    token = os.getenv("HF_TOKEN")
    if not token:
        print("Warning: HF_TOKEN not found in environment variables. Model loading might fail for gated repos.")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(HF_BASE_MODEL, token=token)
        
        # Check for GPU availability
        if torch.cuda.is_available():
            print("GPU detected, loading in 4-bit...")
            device_map = "auto"
            torch_dtype = torch.bfloat16
            load_in_4bit = True 
        else:
            print("No GPU detected, loading on CPU...")
            device_map = {"": "cpu"}
            torch_dtype = torch.float32
            load_in_4bit = False
        
        base_model = AutoModelForCausalLM.from_pretrained(
            HF_BASE_MODEL,
            torch_dtype=torch_dtype,
            device_map=device_map,
            load_in_4bit=load_in_4bit,
            trust_remote_code=True,
            token=token
        )
        
        # Load LoRA adapters
        model = PeftModel.from_pretrained(base_model, ADAPTER_ID, token=token)
        model.eval()
        return model, tokenizer
    except Exception as e:
        print(f"Detailed loading error: {str(e)}")
        raise e

# Initialize model
model = None
tokenizer = None
error_message = None

try:
    model, tokenizer = load_model()
except Exception as e:
    error_message = str(e)
    print(f"Initialization failed: {error_message}")

def predict(message, history):
    if model is None:
        return f"🚨 Error: Model failed to load. \n\nDetails: {error_message or 'Check Hugging Face Token and gated repo access.'}"
    
    # history in Gradio 6 is a list of dicts: [{'role': 'user', 'content': '...'}, ...]
    # Build prompt with history (Llama-3.2 chat format)
    full_prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{SYSTEM_PROMPT}<|eot_id|>"
    
    for msg in history:
        full_prompt += f"<|start_header_id|>{msg['role']}<|end_header_id|>\n\n{msg['content']}<|eot_id|>"
    
    full_prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        output_tokens = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Extract only the new assistant response
    response_tokens = output_tokens[0][len(inputs["input_ids"][0]):]
    response = tokenizer.decode(response_tokens, skip_special_tokens=True)
    
    # Robust parsing to catch potential structured output strings
    if response.startswith("[{'") and response.endswith("'}]"):
        try:
            import ast
            parsed = ast.literal_eval(response)
            if isinstance(parsed, list) and len(parsed) > 0 and 'text' in parsed[0]:
                return parsed[0]['text']
        except:
            pass
            
    return response

# UI Theme
theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="cyan",
    neutral_hue="slate",
).set(
    button_primary_background_fill="#1a73e8",
    button_primary_background_fill_hover="#1557b0"
)

# Build Gradio Interface
with gr.Blocks(title="Diabetes Expert SLM GUI") as demo:
    gr.HTML("<h1 style='text-align: center;'>⚕️ Diabetes Expert SLM</h1>")
    gr.HTML("<p style='text-align: center; color: #666;'>High-Precision Clinical Reasoning Interface</p>")
    
    gr.HTML(DISCLAIMER)
    
    chatbot = gr.Chatbot(
        show_label=False, 
        container=True, 
        height=500,
        avatar_images=(None, "https://cdn-icons-png.flaticon.com/512/387/387561.png")
    )
    
    with gr.Row():
        msg = gr.Textbox(
            placeholder="Ask about 2026 ADA guidelines, pathophysiology, or insulin management...",
            show_label=False,
            scale=9
        )
        submit = gr.Button("Send", variant="primary", scale=1)

    gr.Examples(
        examples=[
            "What is the 2026 ADA recommendation for BP targets in high-risk patients?",
            "Explain the mechanism of beta-cell dedifferentiation in T2D.",
            "How do TCF7L2 variants impact GLP-1 therapy response?",
            "Symptoms of Diabetic Ketoacidosis (DKA)?"
        ],
        inputs=msg
    )

    def user_msg(user_message, history):
        history.append({"role": "user", "content": user_message})
        return "", history

    def bot_msg(history):
        user_message = history[-1]["content"]
        bot_response = predict(user_message, history[:-1])
        history.append({"role": "assistant", "content": bot_response})
        return history

    msg.submit(user_msg, [msg, chatbot], [msg, chatbot], queue=False).then(bot_msg, chatbot, chatbot)
    submit.click(user_msg, [msg, chatbot], [msg, chatbot], queue=False).then(bot_msg, chatbot, chatbot)

    gr.Markdown("---")
    gr.Markdown("Built for research and educational purposes. Powered by Llama-3.2 and PEFT.")

if __name__ == "__main__":
    # HF Spaces expects port 7860
    # Note: Llama-3.2 is a gated repo; ensure you are logged in via huggingface-cli login
    demo.launch(server_name="0.0.0.0", server_port=7860, theme=theme)
