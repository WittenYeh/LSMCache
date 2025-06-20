# launch server script
unset http_proxy
unset https_proxy

python -m sglang.launch_server \
    --model-path "/home/yeweitang/qwen3-4b-model/" \
    --host 0.0.0.0 \
    --port 30000 \
    --mem-fraction-static 0.6 \
    --trust-remote-code \
    --attention-backend torch_native \
    --disable-overlap-schedule \
    --tp 1
