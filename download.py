import os
from modelscope.hub.snapshot_download import snapshot_download
from modelscope.hub.api import HubApi  # 用于认证

model_id = "modelscope/Llama-2-7b-ms"  
local_dir = "/home/u2021201768/"
token = "5e159a6a-ba99-4abc-8f3d-2f4d41f368ce" 

# 确保目标文件夹存在
os.makedirs(local_dir, exist_ok=True)

print(f"开始下载模型 {model_id} 到 {local_dir}...")

# ============== 新增认证步骤 ============== [3,4](@ref)
hub_api = HubApi()
hub_api.login(token)  # 使用ModelScope API Token认证

# ============== 执行下载 ============== [3,4,5](@ref)
snapshot_download(
    model_id=model_id,  # ⚠️ 参数名从 repo_id 改为 model_id
    cache_dir=local_dir,  # ⚠️ 参数名从 local_dir 改为 cache_dir
    revision='master',  # 指定分支/标签（默认master）
)

print("模型下载完成！")