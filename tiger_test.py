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

# ---------- Read prompts from file ----------
with open("prompts.txt", "r", encoding="utf-8") as f:
    content = f.read()

# Split by case markers
cases = []
current_case = None
current_prompt = []

for line in content.split('\n'):
    if line.strip().startswith('#case-'):
        if current_case is not None:
            cases.append((current_case, '\n'.join(current_prompt).strip()))
        current_case = line.strip()
        current_prompt = []
    else:
        current_prompt.append(line)

# Add the last case
if current_case is not None:
    cases.append((current_case, '\n'.join(current_prompt).strip()))

print(f"Found {len(cases)} cases to process\n")

# ---------- Process each case ----------
for case_name, prompt in cases:
    if not prompt:  # Skip empty prompts
        continue
    
    print(f"Processing {case_name}...")
    
    # Tokenize & generate
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    out = model.generate(
        **inputs,
        max_new_tokens=50,      # keep it short for speed
        temperature=0.3,        # low = more "correct", less creative
        repetition_penalty=1.1, # reduce weird repetition
        do_sample=True,
    )
    
    output = tokenizer.decode(out[0], skip_special_tokens=True)
    print(f"Output: {output[:100]}...\n")  # Print first 100 chars
    
    # Save to file
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("output.txt", "a", encoding="utf-8") as f:
        f.write(f"{'='*80}\n")
        f.write(f"Case: {case_name}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"{'-'*80}\n")
        f.write(f"Prompt:\n{prompt}\n")
        f.write(f"{'-'*80}\n")
        f.write(f"Output:\n{output}\n")
        f.write(f"{'='*80}\n\n")

print("All cases processed!")
