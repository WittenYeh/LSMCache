# build sglang source
conda create -n sglang python=3.10
git clone git@github.com:WittenYeh/LSMCache.git
cd LSMCache
pip install --upgrade pip
pip install -e "python[all]"
pip install vllm==0.8.0 -i https://pypi.tuna.tsinghua.edu.cn/simple

# build rocksdb
apt-get install build-essential libsnappy-dev zlib1g-dev libbz2-dev libgflags-dev liblz4-dev
cd rocksdb
mkdir build && cd build
cmake ..
make -j4
cd ..
export CPLUS_INCLUDE_PATH=${CPLUS_INCLUDE_PATH}${CPLUS_INCLUDE_PATH:+:}`pwd`/include/
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}${LD_LIBRARY_PATH:+:}`pwd`/build/
export LIBRARY_PATH=${LIBRARY_PATH}${LIBRARY_PATH:+:}`pwd`/build/

apt-get install python3-dev
pip install python-rocksdb
