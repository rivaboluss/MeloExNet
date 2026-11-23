# Melody PianoRoll CNN

二维钢琴卷帘 + CoordConv + 各向异性卷积 + 注意力，在 POP909 上做音符级主旋律分类。

## 思路（v2 增强卷帘）

1. MIDI 量化到 1/16 拍网格；**小节之间插入一列 BAR 分隔格**（state=3 / is_bar=1）
2. 多通道：`state`（空/起/延/小节线）+ **`in_scale` 调内二值** + `is_bar` + **`skyline` 最高音** + **相邻高/低音半音距离**
3. 坐标改为 **小节序号 + 小节内位置**（不再用纯连续时间轴），另加音高坐标
4. **调式条件**：从 `key_audio.txt`（或 music21）读调，做成 14 维向量，**FiLM 注入**网络；推理同样注入
5. 默认 **二分类**（主旋律 vs 非主旋律，bridge 并进负类），采用 **Focal Loss（alpha=0.25, gamma=2）**
6. 起音格 → 原音符映射，分类后还原 MELODY+BRIDGE MIDI

## 快速开始

```powershell
cd melody_pianoroll_cnn
$env:PYTHONPATH = (Get-Location).Path

# 1) 预处理 POP909 → npz
python -m melody_cnn.data.prepare_pop909 `
  --pop909 "..\POP909-Dataset-master\POP909" `
  --out ".\data_cache\pop909_rolls_lstom"

# 2) 训练
python -m melody_cnn.train --config .\configs\default.yaml

# 3) 抽样推理（写出 MELODY+BRIDGE midi）
python -m melody_cnn.predict `
  --ckpt .\runs\exp\best.pt `
  --midi "..\POP909-Dataset-master\POP909\001\001.mid" `
  --out .\review_pred
```

## 目录

- `melody_cnn/data/` 卷帘构建与 Dataset
- `melody_cnn/models/` CoordConv / AnisoCNN / Attention
- `configs/default.yaml` 超参
- `runs/` 训练输出
