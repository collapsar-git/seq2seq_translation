# seq2seq_translation

🔧 简短说明

本项目旨在实现并演示基于序列到序列（seq2seq）与注意力机制的英汉（en-zh / zh-en）机器翻译实验。包含数据预处理、训练、验证、推理与相关模型文件（.pt）的管理示例，并提供了若干训练脚本与实验结果供复现与学习。

## 主要特点 ✅
- 基于 PyTorch 实现的 seq2seq 模型（含注意力机制）
- 数据预处理与复现所需的脚本（见 `train_v2.0/preprocess.py`、`train_v2.0/preprocessed.py`）
- 不同实验分支：`train_v1.0/`（早期实验）与 `train_v2.0/`（改进与恢复训练）
- 示例测试（`train_v2.0/test_seq2seq.py`）和 Beam Search 评估脚本

---

## 目录结构（简要）

- `train_v1.0/`：早期版本的训练数据与模型示例
- `train_v2.0/`：主要实验代码、预处理脚本、训练/恢复训练/beam-search相关脚本与模型
  - `preprocessed_attention_data/`：预处理后的训练/验证数据（*.pt）
  - `train_seq2seq.py`、`train_seq2seq_resume.py`、`train_seq2seq_beam.py`：训练脚本
  - `test_seq2seq.py`：推理/测试脚本
- 其他说明文件：`README.md`（各子目录）

---

## 快速开始（示例） 🚀

1. 克隆仓库并进入项目根目录：

```powershell
git clone https://github.com/yourname/seq2seq_translation.git
cd seq2seq_translation
```

2. 建议创建虚拟环境并安装依赖：

```powershell
python -m venv venv
.\\venv\\Scripts\\Activate.ps1
pip install -r requirements.txt 
```

3. 数据预处理（若需要）：

```powershell
cd train_v2.0
python preprocess.py
# 或使用提供的 preprocessed 数据（目录：preprocessed_attention_data）
```

4. 训练（示例）

```powershell
python train_seq2seq.py 
```

5. 测试 / 推理（示例）

```powershell
python test_seq2seq.py --model best_model_attention_2.pt --beam 5
```

## 依赖（示例）
- Python 3.8+
- PyTorch
- numpy
- tqdm
- 其它依赖请参见子目录的脚本或 `requirements.txt`（如有）

---

## 贡献与联系 🤝
欢迎提交 Issue 或 Pull Request。

---
