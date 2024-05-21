import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from ipex_llm import optimize_model
import torch

# load the model
model_id = "microsoft/phi-2"
save_name = model_id.split("/")[-1] + "_ipex"
saved = os.path.exists(save_name)

model = AutoModelForCausalLM.from_pretrained(
    model_id if not saved else save_name,
    trust_remote_code=True,
    torch_dtype='auto',
    low_cpu_mem_usage=True,
    use_cache=True)

tokenizer = AutoTokenizer.from_pretrained(
    model_id if not saved else save_name, 
    trust_remote_code=True)

if not saved:
    model.save_pretrained(save_name)
    tokenizer.save_pretrained(save_name)

model = optimize_model(model).to('xpu')

# Generate predicted tokens
prompt = "What is AI?"
with torch.inference_mode():
    PHI2_PROMPT_FORMAT = "<|user|>\n{prompt}<|end|>\n<|assistant|>"
    prompt = PHI2_PROMPT_FORMAT.format(prompt=prompt)
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to('xpu')
    output = model.generate(input_ids, do_sample=False, max_new_tokens=256)
    torch.xpu.synchronize()
    print(tokenizer.decode(output[0], skip_special_tokens=True))