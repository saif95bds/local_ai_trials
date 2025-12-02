from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "md-nishat-008/TigerLLM-1B-it"  # try 9B later: "md-nishat-008/TigerLLM-9B-it"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
)

prompt = """
You are a helpful assistant that replies in Bangla (Bangladesh style).
Write a friendly Facebook post inviting friends to a weekend picnic.
"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
out = model.generate(
    **inputs,
    max_new_tokens=200,
    temperature=0.8,
    do_sample=True,
)
print(tokenizer.decode(out[0], skip_special_tokens=True))
