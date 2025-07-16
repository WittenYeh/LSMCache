from typing import Generator, List
from sglang.lang.interpreter import ProgramState
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

from sglang.lang.backend.runtime_endpoint import RuntimeEndpoint

def start_server(enable_kvstore:bool = False):
    command = f"""
    python3 -m sglang.launch_server \
    --model-path /home/u2021201768/modelscope/Llama-2-7b-ms/ \
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

def generate_batch(qas):
    @sgl.function
    def multi_turns(s, qas):
        for qa in qas:
            s += qa["prompt"]
            s += sgl.gen(max_tokens=qa["new_tokens"], ignore_eos=True)

    responses = multi_turns.run_batch(
        qas,
        temperature=0,
        backend=RuntimeEndpoint(f"http://localhost:30000"),
        num_threads=4,
        progress_bar=True,
    )
    return responses

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

def gen_arguments(num_requests, tokenizer, prompt_len=0, new_tokens=32, turns=1):
    multi_qas = [{"qas": []} for _ in range(num_requests)]
    for i in range(num_requests):
        qas = multi_qas[i]["qas"]
        for _ in range(args.turns):
            qas.append(
                {
                    "prompt": random_prompt(tokenizer, prompt_len) if prompt_len > 0 else tempelate_prompt(),
                    "new_tokens": new_tokens,
                }
            )
    return multi_qas

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
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        nargs="?",
        help="Batch size for generating responses, set to 0 for no batching (default: 0)",
    )
    parser.add_argument(        
        "--turns",
        type=int,
        default=1,
        nargs="?",
        help="Number of turns in the conversation (default: 1)",
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
    tokenizer = AutoTokenizer.from_pretrained("/home/u2021201768/modelscope/Llama-2-7b-ms")

    
    multi_qas = gen_arguments(args.num_requests, tokenizer, 
                                prompt_len=args.prompt_token_num,
                                new_tokens=args.max_new_tokens,
                                turns=args.turns)

    start = time.time()
    response_list = list(generate_batch(multi_qas))
    end = time.time()
    
    print_highlight(f"== Run {args.num_requests} turns of chat with {enable_kvstore=} ==")
    print_highlight(f"== Total time taken: {end - start:.2f} seconds ==")
    print_highlight(f"== Average time per request: {(end - start) / args.num_requests:.2f} seconds ==")
    
    with open(output_file, "w") as f:
        for i, (prompt, response) in enumerate(zip(multi_qas, response_list)):
            resp = response.text().split("\n")[-1]
            f.write(f"=== Turn {i} ===\n")
            f.write(f"Prompt: {prompt['qas'][0]['prompt']}\n")
            f.write(f"Response: {resp}\n\n")
        f.write(f"Average Latency: {(end - start) / args.num_requests:.2f} seconds\n")
    print_highlight(f"Output saved to {output_file}")

    terminate_process(server_process)
