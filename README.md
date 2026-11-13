# SURF2025-2026

Personal archive of the SURF 2025 and SURF 2026 projects.

SURF 2025 与 SURF 2026 项目的个人归档仓库。

---

## 中文说明

### 项目简介

本仓库归档了 SURF 2025（`SURF-2025-0145`）和 SURF 2026（`SURF-2026-0656`）两个项目的原始工程文件，包括代码、部分模型权重和结果数据。

### 重要说明

- 仓库中的代码是项目当时的原始版本，**仅供学习和参考**，不保证完整、正确或可直接运行。
- **脚本内部的路径以作者当时的电脑环境为准**。代码中可能包含 `D:\desktop\...`、`C:\Users\...`、`.venv\...` 之类的硬编码路径，这些路径在你的电脑上很可能不存在。
- 运行任何脚本前，请先检查并修改数据目录、模型权重路径和输出目录，改为你自己电脑上的实际路径。
- 原始训练数据集大多未包含在仓库中，需要自行准备。
- 仓库中保留的模型权重和结果文件，不保证与当前代码或你的数据完全匹配。

### 目录结构

```text
SURF-2025-0145/    SURF 2025 项目（乳腺钼靶分类 / ISIC / SVD 实验）
SURF-2026-0656/    SURF 2026 项目（GEE / 漂绿检测 / 对抗实验）
```

### 运行环境

建议为每个项目单独创建 Python 虚拟环境：

```powershell
cd SURF-2025-0145
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

如果项目中没有 `requirements.txt`，请根据脚本里的 `import` 自行安装依赖。原环境的完整依赖快照见 `SURF-2026-0656/requirements-lock.txt`。

`SURF-2026-0656/.venv` 没有上传到仓库，因为它包含超过 100 MB 的 CUDA DLL，超过 GitHub 普通 Git 的单文件限制。请在本机重新创建虚拟环境。

### 关于大文件

模型权重文件体积较大。如果克隆速度较慢，可以只拉取最近一次提交：

```powershell
git clone --depth 1 https://github.com/Theathens/SURF2025-2026.git
```

### 免责声明

本仓库仅用于个人备份与学习参考。作者不对代码的正确性、完整性或适用性作任何保证。使用前请自行检查路径、依赖和数据。

---

## English

### Overview

This repository is a personal archive of the original project files for SURF 2025 (`SURF-2025-0145`) and SURF 2026 (`SURF-2026-0656`), including source code, some model weights, and result data.

### Important Notice

- The code is the original version from the time of the project and is provided **for learning and reference only**. It is not guaranteed to be complete, correct, or runnable as is.
- **Paths inside the scripts reflect the author's machine at that time.** You may find hardcoded paths such as `D:\desktop\...`, `C:\Users\...`, or `.venv\...`, which most likely do not exist on your computer.
- Before running any script, check and update the data directories, model weight paths, and output directories to match your own environment.
- Most of the original training datasets are not included and must be prepared separately.
- The model weights and result files in this repository are not guaranteed to match the current code or your own data.

### Layout

```text
SURF-2025-0145/    SURF 2025 (mammography classification / ISIC / SVD experiments)
SURF-2026-0656/    SURF 2026 (GEE / greenwashing detection / confrontation experiments)
```

### Environment

Create a separate Python virtual environment for each project:

```powershell
cd SURF-2025-0145
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If a project has no `requirements.txt`, install the dependencies based on the imports in its scripts. A full dependency snapshot of the original environment is available at `SURF-2026-0656/requirements-lock.txt`.

`SURF-2026-0656/.venv` is not included because it contains CUDA DLLs larger than 100 MB, exceeding GitHub's per-file limit for regular Git. Please recreate the virtual environment locally.

### Large Files

The model weight files are large. If cloning is slow, you can fetch only the latest commit:

```powershell
git clone --depth 1 https://github.com/Theathens/SURF2025-2026.git
```

### Disclaimer

This repository is for personal backup and learning reference only. The author makes no guarantee regarding the correctness, completeness, or fitness for purpose of the code. Please verify paths, dependencies, and data before use.
