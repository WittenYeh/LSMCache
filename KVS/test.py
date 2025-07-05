from sglang.test.test_utils import is_in_ci
from sglang.utils import wait_for_server, print_highlight, terminate_process
from sglang.utils import launch_server_cmd
import argparse
import requests
import random
import time

def start_server(enable_kvstore:bool = False):
    command = f"""
    python3 -m sglang.launch_server --model-path qwen/qwen2.5-0.5b-instruct \
    --host 0.0.0.0 --disable-overlap-schedule --attention-backend torch_native \
    {"--enable-kvstore" if enable_kvstore else ""} \
    """
    print_highlight(f"Launching server with command: {command.strip()}")
    server_process, port = launch_server_cmd(command, port=30000)

    wait_for_server(f"http://localhost:{port}")
    return server_process, port


def generate(prompt):
    response = requests.post(
        f"http://localhost:30000/generate",
        json={
            "text": prompt,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": 32,
            },
        },
    )
    return response.json()

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

def rand_prompt():
    template = random.choice(TEMPLATES)
    return template.format(*random.sample(TOPICS, template.count("{}")))



if __name__ == "__main__":
    random.seed(42)  # For reproducibility
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--enable-kvstore",
        action="store_true",
        default=False,
        help="Enable KVStore for the server",
    )
    parser.add_argument(
        "--turns",
        type=int,
        default=1,
        nargs="?",
        help="Number of turns of chat (default: 10)",
    )
    args = parser.parse_args()
    enable_kvstore = args.enable_kvstore
    output_file = "output_kv.txt" if enable_kvstore else "output.txt"
    server_process, port = start_server(enable_kvstore)
    
    prompt_list = [rand_prompt() for _ in range(args.turns)]
    start = time.time()
    response_list = [generate(prompt) for prompt in prompt_list]
    end = time.time()
    
    print_highlight(f"== Run {args.turns} turns of chat with {enable_kvstore=} ==")
    print_highlight(f"== Total time taken: {end - start:.2f} seconds ==")
    print_highlight(f"== Average time per request: {(end - start) / args.turns:.2f} seconds ==")
    
    with open(output_file, "w") as f:
        for i, (prompt, response) in enumerate(zip(prompt_list, response_list)):
            f.write(f"=== Turn {i} ===\n")
            f.write(f"Prompt: {prompt}\n")
            f.write(f"Response: {response['text']}\n\n")
    print_highlight(f"Output saved to {output_file}")

    terminate_process(server_process)
