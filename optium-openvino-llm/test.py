import os
from transformers import AutoTokenizer
from optimum.intel import OVModelForCausalLM, OVWeightQuantizationConfig

# load the model
model_id = "microsoft/phi-2"
save_name = model_id.split("/")[-1] + "_openvino"
saved = os.path.exists(save_name)

quantization_config = OVWeightQuantizationConfig(
    bits=4,
    sym=False,
    group_size=128,
    ratio=0.8,
)

load_kwargs = {
    "device": "gpu",
    "ov_config": {
        "PERFORMANCE_HINT": "LATENCY",
        "INFERENCE_PRECISION_HINT": "fp32",
        "CACHE_DIR": os.path.join(save_name, "model_cache"),  # OpenVINO will use this directory as cache
    },
    "compile": False,
    "quantization_config": quantization_config
}

model = OVModelForCausalLM.from_pretrained(
    model_id if not saved else save_name,
    export=not saved,
    **load_kwargs)

tokenizer = AutoTokenizer.from_pretrained(
    model_id if not saved else save_name)

if not saved:
    model.save_pretrained(save_name)
    tokenizer.save_pretrained(save_name)

model.compile()

# Generate predicted tokens
prompt = "What is AI?"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
