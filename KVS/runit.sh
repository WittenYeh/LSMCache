# number of requests to run
num_requests=10
# number of tokens in each random prompts, set to 0 to use template prompts
prompt_token_num=64
# maximum number of new tokens to generate
max_new_tokens=8

rm -rf db

# origin sglang with torch_native backend
test.py --num-requests $num_requests \
    --prompt-token-num $prompt_token_num \
    --max-new-tokens $max_new_tokens \
    --output-file output.txt \
    |& tee test.log

# kvstore enabled
python test.py --num-requests $num_requests \
    --prompt-token-num $prompt_token_num \
    --max-new-tokens $max_new_tokens \
    --enable-kvstore \
    --output-file output_kv_warmup.txt \
    |& tee test_kv_warmup.log

# kvstore enabled, after db warmup
python test.py --num-requests $num_requests \
    --prompt-token-num $prompt_token_num \
    --max-new-tokens $max_new_tokens \
    --enable-kvstore \
    --output-file output_kv.txt \
    |& tee test_kv.log

diff output.txt output_kv.txt |& tee diff.txt

echo "=================== Settings ==================="
echo num_requests=$num_requests
echo prompt_token_num=$prompt_token_num
echo max_new_tokens=$max_new_tokens
echo "================== Results ==================="
echo "SGLang"
tail -n 1 output.txt
echo "KVS (warmup)"
tail -n 1 output_kv_warmup.txt
echo "KVS (after warmup)"
tail -n 1 output_kv.txt