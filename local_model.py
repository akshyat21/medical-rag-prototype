from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

quant_config = BitsAndBytesConfig(load_in_4bit=True)

print("Loading TinyLlama in 4‑bit on CPU... (first time will download ~1.2 GB)")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=quant_config,
    device_map="cpu",          
    trust_remote_code=True
)
print("Model ready.")

def generate_local(prompt: str, max_new_tokens: int = 200, temperature: float = 0.7) -> str:
    """Generate text using the local TinyLlama model."""
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        inputs.input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated = full_output[len(prompt):].strip()
    return generated