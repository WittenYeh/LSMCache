from openai import OpenAI

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:30000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

chat_response = client.chat.completions.create(
    model="Qwen/Qwen3-4B",
    messages=[
        {"role": "user", "content": "Give me a short introduction to large language models."},
    ],
    max_tokens=2048,  # 调小这个值
    temperature=0.6,
    top_p=0.95,
    # extra_body={"top_k": 20},  # 先注释掉
)

print("Chat response:", chat_response)
