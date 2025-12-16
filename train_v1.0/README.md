# train_v1.0

## 项目简介 🔧
本文件夹包含最初的 Seq2Seq 翻译实验与推理脚本。该版本为第一轮训练，模型效果有限，因此开展了第二版训练以改进性能。

## 主要文件说明 📄
- `seq2seq_translation.py`：基于训练结果执行翻译/推理的脚本。
- `train.tags.zh-en.en` / `train.tags.zh-en.zh`：原始英中/中英训练语料。
- `tut1-model.pt`：该版本生成的示例模型权重（训练效果有限，作为起点）。
- `preprocessed_attention_data/`：预处理后的数据（`train_src.pt`、`train_trg.pt`、`dev_*` 等），方便快速加载用于训练/验证。

> 注意：为了解决服务器上的 CPU 瓶颈问题，项目使用了预处理脚本提前把原始文本转为张量并保存到磁盘，避免在训练时重复进行耗时的文本处理。该预处理脚本位于 `train_v2.0`（`preprocess.py`），也可以用来为 v1.0 生成相同格式的数据。

## 版本说明 ℹ️
- **train_v1.0** 为项目的第一轮训练，实现了基本的 seq2seq（带/不带 attention 的实现由代码内部决定）流程与推理脚本。
- 由于在验证集上效果不理想，所以在 v1.0 的基础上做了进一步的改进，产生了 `train_v2.0`。

## 快速复现建议 ✅
1. 如果没有预处理数据，请参考 `train_v2.0/preprocess.py` 来生成 `preprocessed_attention_data/` 中的 `.pt` 文件。
2. 使用 `seq2seq_translation.py` 测试模型推理流程或将其改为加载其他 checkpoint 进行评估。
