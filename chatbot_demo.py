import os

from transformers import AutoTokenizer
from optimum.intel import OVModelForCausalLM, OVWeightQuantizationConfig

model_name = "microsoft/phi-2"
save_name = model_name.split("/")[-1] + "_openvino"
precision = "f32"
quantization_config = OVWeightQuantizationConfig(
    bits=4,
    sym=False,
    group_size=128,
    ratio=0.8,
)
device = "gpu"

# Load kwargs
load_kwargs = {
    "device": device,
    "ov_config": {
        "PERFORMANCE_HINT": "LATENCY",
        "INFERENCE_PRECISION_HINT": precision,
        "CACHE_DIR": os.path.join(save_name, "model_cache"),  # OpenVINO will use this directory as cache
    },
    "compile": False,
    "quantization_config": quantization_config
}

# Check whether the model was already exported
saved = os.path.exists(save_name)

model = OVModelForCausalLM.from_pretrained(
    model_name if not saved else save_name,
    export=not saved,
    **load_kwargs,
)

# Load tokenizer to be used with the model
tokenizer = AutoTokenizer.from_pretrained(model_name if not saved else save_name)

# Save the exported model locally
if not saved:
    model.save_pretrained(save_name)
    tokenizer.save_pretrained(save_name)

# TODO Optional: export to huggingface/hub

model_size = os.stat(os.path.join(save_name, "openvino_model.bin")).st_size / 1024 ** 3
print(f'Model size in FP32: ~5.4GB, current model size in 4bit: {model_size:.2f}GB')

model.compile()


import time
from threading import Thread

from transformers import (
    TextIteratorStreamer,
    StoppingCriteria,
    StoppingCriteriaList,
    GenerationConfig,
)


# Copied and modified from https://github.com/bigcode-project/bigcode-evaluation-harness/blob/main/bigcode_eval/generation.py#L13
class SuffixCriteria(StoppingCriteria):
    def __init__(self, start_length, eof_strings, tokenizer, check_fn=None):
        self.start_length = start_length
        self.eof_strings = eof_strings
        self.tokenizer = tokenizer
        if check_fn is None:
            check_fn = lambda decoded_generation: any(
                [decoded_generation.endswith(stop_string) for stop_string in self.eof_strings]
            )
        self.check_fn = check_fn

    def __call__(self, input_ids, scores, **kwargs):
        """Returns True if generated sequence ends with any of the stop strings"""
        decoded_generations = self.tokenizer.batch_decode(input_ids[:, self.start_length :])
        return all([self.check_fn(decoded_generation) for decoded_generation in decoded_generations])


def is_partial_stop(output, stop_str):
    """Check whether the output contains a partial stop str."""
    for i in range(0, min(len(output), len(stop_str))):
        if stop_str.startswith(output[-i:]):
            return True
    return False



# Set the chat template to the tokenizer. The chat template implements the simple template of
#   User: content
#   Assistant: content
#   ...
# Read more about chat templates here https://huggingface.co/docs/transformers/main/en/chat_templating
tokenizer.chat_template = "{% for message in messages %}{{message['role'] + ': ' + message['content'] + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ 'Assistant:' }}{% endif %}"


def prepare_history_for_model(history):
    """
    Converts the history to a tokenized prompt in the format expected by the model.
    Params:
      history: dialogue history
    Returns:
      Tokenized prompt
    """
    messages = []
    for idx, (user_msg, model_msg) in enumerate(history):
        # skip the last assistant message if its empty, the tokenizer will do the formating
        if idx == len(history) - 1 and not model_msg:
            messages.append({"role": "User", "content": user_msg})
            break
        if user_msg:
            messages.append({"role": "User", "content": user_msg})
        if model_msg:
            messages.append({"role": "Assistant", "content": model_msg})
    input_token = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True
    )
    return input_token


def generate(history, temperature, max_new_tokens, top_p, repetition_penalty, assisted):
    """
    Generates the assistant's reponse given the chatbot history and generation parameters

    Params:
      history: conversation history formated in pairs of user and assistant messages `[user_message, assistant_message]`
      temperature:  parameter for control the level of creativity in AI-generated text.
                    By adjusting the `temperature`, you can influence the AI model's probability distribution, making the text more focused or diverse.
      max_new_tokens: The maximum number of tokens we allow the model to generate as a response.
      top_p: parameter for control the range of tokens considered by the AI model based on their cumulative probability.
      repetition_penalty: parameter for penalizing tokens based on how frequently they occur in the text.
      assisted: boolean parameter to enable/disable assisted generation with speculative decoding.
    Yields:
      Updated history and generation status.
    """
    start = time.perf_counter()
    # Construct the input message string for the model by concatenating the current system message and conversation history
    # Tokenize the messages string
    inputs = prepare_history_for_model(history)
    input_length = inputs['input_ids'].shape[1]
    # truncate input in case it is too long.
    # TODO improve this
    if input_length > 2000:
        history = [history[-1]]
        inputs = prepare_history_for_model(history)
        input_length = inputs['input_ids'].shape[1]

    prompt_char = "▌"
    history[-1][1] = prompt_char
    yield history, "Status: Generating...", *([gr.update(interactive=False)] * 4)
    
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    # Create a stopping criteria to prevent the model from playing the role of the user aswell.
    stop_str = ["\nUser:", "\nAssistant:", "\nRules:", "\nQuestion:"]
    stopping_criteria = StoppingCriteriaList([SuffixCriteria(input_length, stop_str, tokenizer)])
    # Prepare input for generate
    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0.0,
        temperature=temperature if temperature > 0.0 else 1.0,
        repetition_penalty=repetition_penalty,
        top_p=top_p,
        eos_token_id=[tokenizer.eos_token_id],
        pad_token_id=tokenizer.eos_token_id,
    )
    generate_kwargs = dict(
        streamer=streamer,
        generation_config=generation_config,
        stopping_criteria=stopping_criteria,
    ) | inputs

    if assisted:
        target_generate = stateless_model.generate
        generate_kwargs["assistant_model"] = asst_model
    else:
        target_generate = model.generate

    t1 = Thread(target=target_generate, kwargs=generate_kwargs)
    t1.start()

    # Initialize an empty string to store the generated text.
    partial_text = ""
    for new_text in streamer:
        partial_text += new_text
        history[-1][1] = partial_text + prompt_char
        for s in stop_str:
            if (pos := partial_text.rfind(s)) != -1:
                break
        if pos != -1:
            partial_text = partial_text[:pos]
            break
        elif any([is_partial_stop(partial_text, s) for s in stop_str]):
            continue
        yield history, "Status: Generating...", *([gr.update(interactive=False)] * 4)
    history[-1][1] = partial_text
    generation_time = time.perf_counter() - start
    yield history, f'Generation time: {generation_time:.2f} sec', *([gr.update(interactive=True)] * 4)


import gradio as gr

try:
    demo.close()
except:
    pass


EXAMPLES = [
    ["What is OpenVINO?"],
    ["Can you explain to me briefly what is Python programming language?"],
    ["Explain the plot of Cinderella in a sentence."],
    ["Write a Python function to perform binary search over a sorted list. Use markdown to write code"],
    ["Lily has a rubber ball that she drops from the top of a wall. The wall is 2 meters tall. How long will it take for the ball to reach the ground?"],
]


def add_user_text(message, history):
    """
    Add user's message to chatbot history

    Params:
      message: current user message
      history: conversation history
    Returns:
      Updated history, clears user message and status
    """
    # Append current user message to history with a blank assistant message which will be generated by the model
    history.append([message, None])
    return ('', history)


def prepare_for_regenerate(history):
    """
    Delete last assistant message to prepare for regeneration

    Params:
      history: conversation history
    Returns:
      updated history
    """ 
    history[-1][1] = None
    return history


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown('<h1 style="text-align: center;">Chat with Phi-2 on Meteor Lake iGPU</h1>')
    chatbot = gr.Chatbot()
    with gr.Row():
        assisted = gr.Checkbox(value=False, label="Assisted Generation", scale=10)
        msg = gr.Textbox(placeholder="Enter message here...", show_label=False, autofocus=True, scale=75)
        status = gr.Textbox("Status: Idle", show_label=False, max_lines=1, scale=15)
    with gr.Row():
        submit = gr.Button("Submit", variant='primary')
        regenerate = gr.Button("Regenerate")
        clear = gr.Button("Clear")
    with gr.Accordion("Advanced Options:", open=False):
        with gr.Row():
            with gr.Column():
                temperature = gr.Slider(
                    label="Temperature",
                    value=0.0,
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                    interactive=True,
                )
                max_new_tokens = gr.Slider(
                    label="Max new tokens",
                    value=128,
                    minimum=0,
                    maximum=512,
                    step=32,
                    interactive=True,
                )
            with gr.Column():
                top_p = gr.Slider(
                    label="Top-p (nucleus sampling)",
                    value=1.0,
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                    interactive=True,
                )
                repetition_penalty = gr.Slider(
                    label="Repetition penalty",
                    value=1.0,
                    minimum=1.0,
                    maximum=2.0,
                    step=0.1,
                    interactive=True,
                )
    gr.Examples(
        EXAMPLES, inputs=msg, label="Click on any example and press the 'Submit' button"
    )

    # Sets generate function to be triggered when the user submit a new message
    gr.on(
        triggers=[submit.click, msg.submit],
        fn=add_user_text,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot],
        queue=False,
    ).then(
        fn=generate,
        inputs=[chatbot, temperature, max_new_tokens, top_p, repetition_penalty, assisted],
        outputs=[chatbot, status, msg, submit, regenerate, clear],
        concurrency_limit=1,
        queue=True
    )
    regenerate.click(
        fn=prepare_for_regenerate,
        inputs=chatbot,
        outputs=chatbot,
        queue=True,
        concurrency_limit=1
    ).then(
        fn=generate,
        inputs=[chatbot, temperature, max_new_tokens, top_p, repetition_penalty, assisted],
        outputs=[chatbot, status, msg, submit, regenerate, clear],
        concurrency_limit=1,
        queue=True
    )
    clear.click(fn=lambda: (None, "Status: Idle"), inputs=None, outputs=[chatbot, status], queue=False)

    demo.launch(share=True)
