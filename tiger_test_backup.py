from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from datetime import datetime

# ---------- Model choice ----------
# Use 1B for faster testing:
# model_id = "md-nishat-008/TigerLLM-1B-it"

# Use 9B for higher quality (slower):
model_id = "md-nishat-008/TigerLLM-9B-it"

# ---------- Device selection ----------
device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device:", device)

# ---------- Load tokenizer & model ----------
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.bfloat16 if device == "mps" else torch.float32,
)
model.to(device)

# ---------- Prompt ----------
prompt = """
তুমি একজন বাংলা ভাষার শিক্ষক।
বাক্যটি বানান ও ব্যাকরণ ঠিক করে সুন্দরভাবে লিখো।
অর্থ বা অনুভূতি একই থাকবে।

ভুল বাক্য:
"অমি তোমাক বলো বাসি"

শুদ্ধ বাক্য:
"""

# ---------- Tokenize & generate ----------
inputs = tokenizer(prompt, return_tensors="pt").to(device)

out = model.generate(
    **inputs,
    max_new_tokens=50,      # keep it short for speed
    temperature=0.3,        # low = more “correct”, less creative
    repetition_penalty=1.1, # reduce weird repetition
    do_sample=True,
)

output = tokenizer.decode(out[0], skip_special_tokens=True)
print("----- Output -----")
print(output)

# ---------- Save to file ----------
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
with open("output.txt", "a", encoding="utf-8") as f:
    f.write(f"{'='*80}\n")
    f.write(f"Timestamp: {timestamp}\n")
    f.write(f"{'-'*80}\n")
    f.write(f"Output:\n{output}\n")
    f.write(f"{'='*80}\n\n")
