# launch server script
unset http_proxy
unset https_proxy

python3 -m sglang.launch_server \
    --model-path "/root/Qwen-7B/" \
    --host 0.0.0.0 \
    --port 30000 \
    --mem-fraction-static 0.8 \
    --trust-remote-code \
    --attention-backend torch_native \
    --disable-overlap-schedule \
    --tp 1 \
    --enable-kvstore
