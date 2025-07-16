# build sglang source
conda create -n sglang python=3.10
git clone git@github.com:WittenYeh/LSMCache.git
cd LSMCache
pip install --upgrade pip
pip install -e "python[all]"
pip install vllm==0.8.0

# build rocksdb
cd python-rocksdb
make shared_lib -j8
pip install pybind11 setuptools wheel
cd rocksdb_pybinding
python3 setup.py build_ext --inplace
pip install .
cd ..
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)
