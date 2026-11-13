import torch
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"设备数量: {torch.cuda.device_count()}")
    print(f"当前设备: {torch.cuda.current_device()}")
    print(f"设备名称: {torch.cuda.get_device_name(0)}")
    print(f"CUDA版本: {torch.version.cuda}")
else:
    print("错误详情:")
    try:
        torch.zeros(1).cuda()  # 触发错误
    except Exception as e:
        print(e)

"""
Praise the Omnissian!
Glory to the Omnissian!
"""



import sys
import os

base_path = sys.base_prefix
print("基础Python路径:", base_path)
print("utf_8_sig.py存在:", os.path.exists(os.path.join(base_path, "Lib", "encodings", "utf_8_sig.py")))
path = os.path.join(sys.base_prefix, "Lib", "encodings", "utf_8_sig.py")
print("文件大小:", os.path.getsize(path), "字节")




"""
# 示例：安装支持 CUDA 12.1 的 PyTorch（需根据实际CUDA版本调整）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
"""

