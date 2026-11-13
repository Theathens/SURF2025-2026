# SURF2025-2026

This repository is a personal archive of the SURF 2025 and SURF 2026 projects.

## 重要说明 / Important Notice

- 本仓库中的代码是项目当时的原始版本，**仅供学习和参考**。
- **脚本内部的路径以作者当时的电脑环境为准**。代码中可能包含诸如 `D:\desktop\...`、`C:\Users\...`、`.venv\...` 等硬编码路径，这些路径在你的电脑上很可能不存在。
- 运行任何脚本前，请先检查并修改数据目录、模型权重路径、输出目录等配置，改为你自己电脑上的实际路径。
- 原始训练数据集大多未包含在仓库中，需要自行准备。
- 仓库中保留了部分模型权重和生成结果，但不保证与当前代码或你的数据完全匹配。

## 目录结构 / Layout

```text
SURF-2025-0145/    SURF 2025 project (mammography / ISIC / SVD experiments)
SURF-2026-0656/    SURF 2026 project (GEE / greenwashing detection / confrontation experiments)
```

## 环境 / Environment

建议为每个项目单独创建 Python 虚拟环境：

```powershell
cd SURF-2025-0145
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

如果项目没有 `requirements.txt`，请根据脚本中的 `import` 自行安装依赖。

`SURF-2026-0656/.venv` 未上传到本仓库，因为它包含超过 100 MB 的 CUDA DLL，超过 GitHub 普通 Git 的单文件限制。请在本机重新创建虚拟环境。

## 关于大文件 / Large Files

模型权重文件（`.pth` 等）体积较大。如果克隆速度较慢，可以只拉取代码，或使用浅克隆：

```powershell
git clone --depth 1 https://github.com/Theathens/SURF2025-2026.git
```

## 免责声明 / Disclaimer

本仓库仅用于个人备份与参考。代码可能不完整、不可直接复现，作者不对其正确性、完整性或适用性作任何保证。使用前请自行检查路径、依赖和数据。
