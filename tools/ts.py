import torch
# 关键检测点
print(torch.__version__)             # PyTorch版本（需≥1.8）
print(torch.cuda.is_available())     # 应返回True
print(torch.version.cuda)            # 应与安装的CUDA版本一致（如13.0）
print(torch.cuda.get_device_name(0)) # 显示GPU型号（如RTX 4090）-i https://pypi.tuna.tsinghua.edu.cn/simple
