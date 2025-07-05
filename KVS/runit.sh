# origin sglang with torch_native backend
python test.py --turns 10 \
    |& tee test.log

# kvstore enabled
python test.py --turns 10 \
    --enable-kvstore \
    |& tee test_kv.log

diff output.txt output_kv.txt