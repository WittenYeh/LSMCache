unset http_proxy
unset https_proxy

cd ./benchmark/multi_turn_chat/

python bench_sglang.py \
    --backend 'srt' \
    --tokenizer "/home/yeweitang/qwen3-4b-model" \
    --turns 1 \
    --num-qa 1 \
    --min-len-q 4 \
    --max-len-q 8 \
    --min-len-a 4 \
    --max-len-a 8 \
    --result-file "sglang_mistral_benchmark.json"
