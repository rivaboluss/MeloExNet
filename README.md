# MeloExNet

面向 POP909 符号流行音乐的轻量级旋律提取模型：**音符级局部卷帘 patch（CNN）+ 20 维手工音符特征（MLP）+ 双向 GRU 序列建模**。

- 参数量 **0.64M**，单张消费级 GPU 约 30 分钟训练
- POP909 729/90/90 划分上，三种子概率平均集成：**Acc 0.9608 / P 0.9070 / R 0.9059 / F1 0.9065 / AP 0.9678 / VFA 0.0246**
- 对 LStoM 式基线（同划分、同损失、同评估重实现）：F1 +8.75 点，VFA 下降 60%，90 首测试曲逐曲配对赢 85 首（p<1e-19）
- 论文草稿：`paper_draft.md`（中英对照）、`paper_draft_zh.md`（纯中文）

## 方法速览

每个音符起点由两部分表示，拼接成 token 后整曲送进双向 GRU：

1. **局部卷帘 patch**：以音符 (音高, 时间) 为中心的 17×13 窗口（±8 半音、±6 个十六分步），取自 12 通道卷帘，经 3 层小 CNN 编码到 128 维
2. **20 维手工特征**：音高、时值、力度、节拍位置、调内性、和弦归属、音符间上下文，经 2 层 MLP 编码到 64 维

损失为 masked focal loss（旋律类 α=0.75，γ=2）；移调增广（[-3,3] 半音，p=0.5）作用于特征/标签/patch；阈值在验证集网格搜索后用于测试。

## 环境

```powershell
pip install torch numpy tqdm mido
```

Windows 下若训练进程异常退出（`0xC0000409`，Intel MKL 与 torch 的 OpenMP 冲突），加上：

```powershell
$env:OMP_NUM_THREADS = 1
$env:MKL_NUM_THREADS = 1
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
```

## 快速开始

```powershell
cd melody_pianoroll_cnn
$env:PYTHONPATH = (Get-Location).Path

# 1) 预处理 POP909 → npz 缓存（约 217MB，已 gitignore）
python -m melody_cnn.data.prepare_pop909 `
  --pop909 "..\POP909-Dataset-master\POP909" `
  --out ".\data_cache\pop909_rolls_chord"

# 2) 训练混合模型（30 轮，单 seed）
python -m melody_cnn.hybrid `
  --out-dir .\runs\hybrid_seed0 --epochs 30 --batch-size 4 --seed 0 --dump-scores

# 3) 训练 LStoM 式基线（6 层双向 LSTM，hidden 140，按 Kosta 2022 第 3.1 节结构描述重实现）
python -m melody_cnn.baseline_lstom --out-dir .\runs\lstom_repro_a75

# 4) 评估（逐音符拼接 + 阈值搜索 + PR-AP + 逐曲 F1）
python -m melody_cnn.evaluate --ckpt .\runs\hybrid_seed0\best.pt --dump-scores
```

## 复现论文实验

| 实验 | 命令要点 |
|---|---|
| 多种子训练 | `--seed 0/1/2` 各跑一次，`--dump-scores` 输出逐音符分数 |
| 三种子集成 + 配对检验 | `python -m melody_cnn.ensemble`（见 `--help`） |
| 消融（去特征组） | `python -m melody_cnn.hybrid --drop chord,metric,velocity,context` 或 `--drop notes`；`--no-patch` 为去 patch 分支 |
| 编码器对照 | `--seq transformer`，建议加 `--warmup 5 --epochs 60` |
| 窗口 CNN 基线 | `python -m melody_cnn.train --config .\configs\default.yaml` |

## 目录

- `melody_cnn/hybrid.py` 混合模型（训练 + 评估主入口）
- `melody_cnn/baseline_lstom.py` LStoM 式基线重实现
- `melody_cnn/ensemble.py` 多种子集成与逐曲配对显著性
- `melody_cnn/data/` 卷帘构建（含和弦通道）、Dataset、POP909 预处理
- `melody_cnn/models/` 窗口 CNN（各向异性卷积 + FiLM 调式条件化）
- `melody_cnn/evaluate.py` 统一评估（拼接 + 阈值搜索 + AP + 逐曲 F1）
- `configs/` 超参；`runs/` 训练产物（checkpoint、指标、逐音符分数）

## 引用

论文投稿中（目标 EUSIPCO 2027）。暂可引用本仓库；论文录用后替换为正式引用。
