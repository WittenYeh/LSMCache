from sglang.test.test_utils import is_in_ci, select_sglang_backend
from sglang.utils import wait_for_server, print_highlight, terminate_process
from sglang.utils import launch_server_cmd
import argparse
import requests
import random
import time
import numpy as np
import torch
from transformers import AutoTokenizer
import string
import sglang as sgl
import sglang.srt.mem_cache.kv_storage as kv_storage

def start_server(enable_kvstore:bool = False):
    command = f"""
    python3 -m sglang.launch_server \
    --model-path mistralai/Mistral-7B-v0.1 \
    --host 0.0.0.0 \
    --disable-overlap-schedule \
    --attention-backend torch_native \
    {"--enable-kvstore" if enable_kvstore else ""}
    """

    print(f"Launching server with command: {command}")
    server_process, port = launch_server_cmd(command, port=30000)

    wait_for_server(f"http://localhost:{port}")
    return server_process, port


def generate(prompt, max_new_tokens=32):
    response = requests.post(
        f"http://localhost:30000/generate",
        json={
            "text": prompt,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": max_new_tokens,
            },
        },
    )
    return response.json()

def generate_batch(prompts, max_new_tokens=32, batch_size=None):
    @sgl.function
    def my_function(s, prompt):
        s += prompt
        s += sgl.gen(max_tokens=max_new_tokens, ignore_eos=True)
    if batch_size is None:
        batch_size = len(prompts)
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        responses = my_function.run_batch(
            batch_prompts,
            temperature=0,
            backend="http://localhost:30000",  # or select_sglang_backend(args)
            num_threads=4,
            progress_bar=True,
        )
        yield from responses

TEMPLATES = [
    "What are the benefits of {}?",
    "Explain how {} works.",
    "Why is {} important in {}?",
    "Compare {} and {} in terms of performance.",
    "Summarize the key differences between {} and {}.",
    "How does {} affect daily life?",
    "Can you give an example of {}?",
]

TOPICS = [
    "machine learning",
    "blockchain",
    "climate change",
    "artificial intelligence",
    "quantum computing",
    "database indexing",
    "renewable energy",
    "neural networks",
    "privacy laws",
    "digital marketing",
]

def tempelate_prompt():
    template = random.choice(TEMPLATES)
    return template.format(*random.sample(TOPICS, template.count("{}")))

def random_prompt(tokenizer, token_num):
    cha_set = string.ascii_letters + string.digits
    ret = "".join(random.choices(cha_set, k=token_num))
    while len(tokenizer(ret).input_ids) < token_num:
        ret += random.choice(cha_set)
    return ret

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--enable-kvstore",
        action="store_true",
        default=False,
        help="Enable KVStore for the server",
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=1,
        nargs="?",
        help="Number of requests (default: 1)",
    )
    parser.add_argument(
        "--prompt-token-num",
        type=int,
        default=0,
        nargs="?",
        help="Number of tokens in the random prompt, set to 0 to use template prompts (default: 0)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=32,
        nargs="?",
        help="Maximum number of new tokens to generate (default: 32)",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        nargs="?",
        default="output.txt",
        help="Output file to save the responses (default: output.txt)",
    )
    return parser.parse_args()

if __name__ == "__main__":
    random.seed(42)  # For reproducibility
    np.random.seed(42)  # For reproducibility
    torch.manual_seed(42)  # For reproducibility
    args = parse_args()
    
    enable_kvstore = args.enable_kvstore
    output_file = args.output_file
    server_process, port = start_server(enable_kvstore)
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

    
    if args.prompt_token_num > 0:
        prompt_list = [random_prompt(tokenizer, args.prompt_token_num) for _ in range(args.num_requests)]
    else:
        prompt_list = [tempelate_prompt() for _ in range(args.num_requests)]
    start = time.time()
    response_list = [generate(prompt, args.max_new_tokens) for prompt in prompt_list]
    end = time.time()
    
    print_highlight(f"== Run {args.num_requests} turns of chat with {enable_kvstore=} ==")
    print_highlight(f"== Total time taken: {end - start:.2f} seconds ==")
    print_highlight(f"== Average time per request: {(end - start) / args.num_requests:.2f} seconds ==")
    
    with open(output_file, "w") as f:
        for i, (prompt, response) in enumerate(zip(prompt_list, response_list)):
            f.write(f"=== Turn {i} ===\n")
            f.write(f"Prompt: {prompt}\n")
            f.write(f"Response: {response['text']}\n\n")
        f.write(f"Average Latency: {(end - start) / args.num_requests:.2f} seconds\n")
    print_highlight(f"Output saved to {output_file}")

    terminate_process(server_process)
