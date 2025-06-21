import os
from huggingface_hub import snapshot_download

# 定义模型ID和本地目标文件夹
model_id = "huggyllama/llama-7b"
local_dir = "/home/yeweitang/llama-7b"

# 确保目标文件夹存在
os.makedirs(local_dir, exist_ok=True)

print(f"开始下载模型 {model_id} 到 {local_dir}...")

# 执行下载
snapshot_download(
    repo_id=model_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False,  # 建议设为False，直接复制文件而不是创建符号链接
    token=token, # 如果你没有通过 `huggingface-cli login` 登录，可以在这里提供token
    resume_download=True  # 允许断点续传
)

print("模型下载完成！")