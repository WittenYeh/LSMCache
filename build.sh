# build sglang source
conda create -n sglang python=3.10
git clone git@github.com:WittenYeh/LSMCache.git
cd LSMCache
pip install --upgrade pip
pip install -e "python[all]"
pip install vllm==0.8.0 -i https://pypi.tuna.tsinghua.edu.cn/simple

# build rocksdb
git clone https://github.com/twmht/python-rocksdb.git --recursive -b pybind11
cd python-rocksdb
python setup.py install