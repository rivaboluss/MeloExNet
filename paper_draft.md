# A Lightweight Hybrid Architecture for Symbolic Melody Extraction from Pop Songs
# 一种用于流行歌曲符号旋律提取的轻量级混合架构

> 版本 v0.7（双语对照）· 目标会议：EUSIPCO 2027（截稿 2027-02-06）
> 实验状态：三种子 + 集成 + 逐曲配对检验完成；6 组消融完成；编码器对照完成；文献数字已对照原文 PDF 核对
> 审阅方式：每段 `EN` 为投稿正文，`中` 为对照翻译，仅供核对语义，不随稿提交。

---

## Abstract

**EN** Symbolic melody extraction follows two main paradigms: piano-roll methods render the score as an image and predict a per-cell melody mask with a convolutional network, while note-sequence methods classify each onset from hand-crafted features with a recurrent network; both are trained from scratch with task supervision. A third class is distinguished by training paradigm rather than representation: large pre-trained symbolic models are first self-supervised on large symbolic corpora and then fine-tuned on the task, at a budget orders of magnitude larger. We propose a lightweight, melody-specific hybrid architecture that fuses these two representations at the prediction unit of the individual note. Each note onset is described by (i) a 17x13 local piano-roll patch covering +-8 semitones and +-6 sixteenth steps, encoded by a small convolutional network, and (ii) a 20-dimensional vector of hand-crafted note features covering pitch, duration, velocity, metric position, chord membership and note-to-note context. The two embeddings are concatenated and the whole piece is modelled as an ordered note sequence by a bidirectional GRU. With 0.64M parameters, the model reaches note-level F1 0.8908 +- 0.0053 over three seeds (0.9065 for a three-seed probability ensemble) and average precision 0.9552 +- 0.0025 (0.9678 ensembled), against 0.8190 F1 / 0.8985 AP for a note-level recurrent baseline and 0.8169 F1 / 0.8768 AP for a modern window CNN trained on the same split. The ensemble wins on 85 of 90 test songs (paired t = 11.77, p < 1e-19) and reduces the accompaniment false-alarm rate from 0.0619 to 0.0246 while increasing recall. Ablations show that the two feature views are complementary rather than redundant: removing all 20 note features costs 6.5 F1 points, whereas removing the patch branch costs 0.6; a Transformer encoder, tested under two schedules, converges near F1 0.81 and does not replace the GRU at this data scale.

**中** 符号旋律提取大体沿两条路线展开：钢琴卷帘方法把乐谱渲染为图像、用卷积网络逐格预测旋律掩码；音符序列方法用循环网络基于手工特征对每个音符起点分类；这两支都从零开始、用任务监督训练。第三类方法按训练范式而非表示形式区分：大型预训练符号模型先在大规模符号语料上自监督预训练、再微调到本任务，其资源预算高出数个数量级。本文提出一种轻量级、面向旋律的混合架构，把这两类表示在"单个音符"这一预测单元上融合。每个音符起点由两部分描述：(i) 覆盖 ±8 半音、±6 个十六分步的 17×13 局部卷帘 patch，由小型卷积网络编码；(ii) 20 维手工音符特征，涵盖音高、时值、力度、节拍位置、和弦归属与音符间上下文。两路嵌入拼接后，整首曲子作为有序音符序列送入双向 GRU。该模型仅 0.64M 参数，三种子平均取得音符级 F1 0.8908 ± 0.0053（三种子概率集成 0.9065）、AP 0.9552 ± 0.0025（集成 0.9678）；同一划分上，音符级循环基线为 F1 0.8190 / AP 0.8985，现代窗口 CNN 为 F1 0.8169 / AP 0.8768。集成模型在 90 首测试曲中赢 85 首（配对 t=11.77，p<1e-19），并把伴奏误报率从 0.0619 降至 0.0246，同时召回上升。消融表明两种特征视角互补而非冗余：去掉全部 20 维音符特征损失 6.5 个 F1 点，去掉 patch 分支仅损失 0.6 个点；在两种训练方案下，Transformer 编码器均收敛在 F1 0.81 附近，在本数据规模上无法替代 GRU。

## 1. Introduction

**EN** Melody extraction from polyphonic music is a long-standing MIR task. Symbolic methods divide into two families. Piano-roll CNNs render the score as an image and threshold a per-cell melody mask predicted over short windows; they inherit two assumptions from image segmentation, namely that the decision for a note depends only on a bounded neighbourhood and that overlapping windows can be scored independently and stitched afterwards. Note-sequence models classify each onset from hand-crafted features along the time axis; they avoid the window assumption but discard the local texture around a note. Melody is defined by salience relations that span the whole texture - a note is melodic partly because of what precedes and follows it - so neither family alone captures both the local texture and the global sequence context.

**中** 复音音乐的旋律提取是 MIR 的长期任务。符号方法分为两支。钢琴卷帘 CNN 把乐谱渲染为图像，对短窗口上预测的逐格旋律掩码取阈值；它从图像分割继承了两个假设：音符判定只依赖有限邻域，重叠窗口可独立打分再拼接。音符序列模型沿时间轴对每个音符起点的手工特征分类；它避开了窗口假设，却丢掉了音符周围的局部纹理。旋律由跨整段织体的显著性关系定义——一个音之所以是旋律，部分取决于它前后的音——因此任何一支单独都无法同时捕捉局部纹理与全局序列上下文。

**EN** This paper makes three contributions.

**(C1) A validated multi-view note representation.** We describe each note by two complementary views - a local multi-channel roll patch and a 20-dimensional feature vector - and fuse them before sequence modelling. A controlled ablation quantifies each view: with the patch present, removing the feature vector costs 0.0647 F1 and 0.0613 AP; with the feature vector present, removing the patch costs 0.0061 F1 and 0.0111 AP. Neither view subsumes the other, and each is reported separately.

**(C2) An empirical comparison of sequence encoders at realistic data scale.** On identical inputs, identical loss and an identical split, a 2-layer bidirectional GRU clearly outperforms a Transformer encoder (F1 0.8873 vs 0.8087). The Transformer result is stable across two schedules (no warm-up / 30 epochs, and warm-up + cosine / 60 epochs), both converging near F1 0.81, so the gap is not an optimization artefact. This indicates that the gains of the proposed design do not come from encoder capacity.

**(C3) A lightweight, melody-specific architecture with a reproducible protocol.** The complete system has 0.64M parameters, a 2.6 MB checkpoint, trains in well under an hour on one consumer GPU, and is evaluated with the split, evaluation code and per-note scores released. On POP909 it raises note-level F1 by 8.75 points and AP by 6.9 points over a note-level recurrent baseline, while cutting accompaniment false alarms by 60%. Because the model outputs note-level melody labels directly, it is a practical candidate for semi-automatic melody annotation of pop-song corpora.

**中** 本文有三点贡献。

**（C1）一个经过验证的多视角音符表示。** 我们用两个互补视角描述每个音符——局部多通道卷帘 patch 与 20 维特征向量——并在序列建模之前融合。控制消融给出各自量化贡献：在保留 patch 时去掉特征向量损失 0.0647 F1 / 0.0613 AP；在保留特征向量时去掉 patch 损失 0.0061 F1 / 0.0111 AP。两者互不替代，且分别报告。

**（C2）在真实数据规模下对序列编码器的实证比较。** 在输入、损失与划分完全相同的条件下，2 层双向 GRU 明显优于 Transformer 编码器（F1 0.8873 对 0.8087）。Transformer 的结果在两种训练方案下都稳定（无 warmup/30 轮，以及 warmup+余弦/60 轮），均收敛在 F1 0.81 附近，说明差距不是优化伪影。这表明本文设计的收益不来自编码器容量。

**（C3）一个轻量、面向旋律的架构与可复现协议。** 完整系统 0.64M 参数、2.6MB checkpoint，单张消费级 GPU 上一小时内完成训练，并公开划分、评估代码与逐音符分数。在 POP909 上，相对音符级循环基线，F1 提高 8.75 个点、AP 提高 6.9 个点，同时把伴奏误报降低 60%。由于模型直接输出音符级旋律标签，它可作为流行歌曲语料半自动旋律标注的实用候选。

**EN** A further analysis, reported in Section 5.2, isolates the effect of the prediction unit: holding the feature set fixed, moving from window-level pixel classification to whole-piece note labelling is worth +6.4 F1. We present it as supporting evidence rather than a headline claim, because the change also introduces global sequence context; the two effects cannot be fully separated with the experiments reported here.

**中** 5.2 节另有一项分析单独考察预测单元的作用：在特征集固定时，从窗口级像素分类改为整曲音符标注值 +6.4 F1。我们把它作为支撑证据而非主要主张，因为该改动同时引入了全局序列上下文，本文现有实验无法把两者完全分离。

## 2. Related Work

**EN** POP909 provides 909 Chinese pop songs with aligned MIDI and beat, key and chord annotations, and is the standard benchmark for symbolic melody extraction. Two families of methods dominate. Piano-roll CNNs treat the roll as an image: a convolutional network predicts a melody mask per cell, and a fully convolutional model operating on symbolic scores was already shown to be effective for melody-line identification. Anisotropic kernels, channel/spatial attention and feature-wise conditioning are standard refinements of this family, and the window CNN in this paper is a representative instance. Note-level models instead operate on per-onset feature vectors and predict one label per note; LStoM (Kosta et al., ISMIR 2022) is the reference system, and note-level formulations are also standard in voice separation. Pre-trained symbolic models such as MidiBERT (Zeng et al., 2021) follow a different resource budget: large-scale masked-language-model pre-training followed by fine-tuning, which yields the highest reported POP909 melody-extraction scores but is not trained from scratch on this dataset.

**中** POP909 提供 909 首中文流行歌曲及对齐的 MIDI 与节拍、调性、和弦标注，是符号旋律提取的标准基准。方法主要有两支。钢琴卷帘 CNN 把卷帘当图像：卷积网络逐格预测旋律掩码，已有工作证明全卷积模型在符号乐谱上的旋律线识别有效。各向异性卷积核、通道/空间注意力与特征级条件化是该谱系的标准改进，本文的窗口 CNN 是代表性实例。音符级模型则基于每个音符起点的特征向量预测一个标签；LStoM（Kosta 等，ISMIR 2022）是参照系统，音符级建模在声部分离中同样是标准做法。MidiBERT（Zeng 等，2021）等预训练符号模型走的是另一条资源预算路线：先在大规模语料上做掩码语言模型预训练，再微调；它在 POP909 上报告的旋律提取指标最高，但并非在该数据集上从零训练。

**EN** Our design borrows the "local patch to token" pattern from vision, but departs from Vision Transformers in three ways that matter at this data scale: patches are centred on events rather than laid on a fixed grid, they overlap heavily, and they are augmented with hand-crafted note descriptors. Concatenating local convolutional features with note-level features and modelling the resulting sequence with a bidirectional recurrent network follows the CNN-RNN pattern established for sequence labelling, and the note-as-token view is shared with symbolic music generation models. Transformers have been applied successfully to symbolic music, but mostly with large-scale pre-training; we therefore test one directly rather than assuming it transfers.

**中** 我们的设计借用了视觉中"局部块→token"的模式，但在该数据规模上做了三点实质偏离：块以音符事件为中心而非固定网格、块之间大量重叠、并拼接手工艺音符描述子。把局部卷积特征与音符级特征拼接、再用双向循环网络建模序列，沿用了序列标注中成熟的 CNN-RNN 模式；"音符即 token"的视角与符号音乐生成模型一致。Transformer 已在符号音乐上取得成功，但多依赖大规模预训练；因此我们直接做对照实验，而不假设它能迁移。

**EN** All systems in this paper are trained with the same focal-loss family, so the loss is not a confound. Audio melody extraction, both pitch-contour and salience-based, is out of scope: we study the symbolic setting, where comparisons isolate modelling choices rather than audio front-ends.

**中** 本文所有系统都使用同一类 focal loss，因此损失不是混淆因素。音频旋律提取（音高轮廓与显著性两类）不在范围内：本文研究符号设定，比较的是建模选择而非音频前端。

## 3. Method

### 3.1 Input representation / 输入表示

**EN** Each song is rendered as a 12-channel piano roll on a sixteenth-note grid: note state, in-scale flag, bar mask, velocity, duration, maximum velocity, chord-tone flag, semitone distance to the nearest chord tone, chord root, chord phase, chord validity and metric strength. Chord channels are derived from the POP909 `chord_midi.txt` annotations by converting time-stamped chord segments to the grid; they are chord-relative and therefore transposition-invariant, so pitch-shift augmentation needs no recomputation. Metric strength is 1.0 on downbeats, 0.7 on beats, 0.4 on eighths and 0.2 on sixteenths.

**中** 每首曲子渲染为十六分音符网格上的 12 通道钢琴卷帘：音符状态、调内标记、小节掩码、力度、时值、最大力度、和弦音标记、到最近和弦音的半音距离、和弦根音、和弦相位、和弦有效、节拍强度。和弦通道由 POP909 的 `chord_midi.txt` 标注把带时间戳的和弦段换算到网格得到；这些特征都是和弦相对量，对移调不变，因此移调增广无需重算。节拍强度在强拍为 1.0、拍为 0.7、八分为 0.4、十六分为 0.2。

### 3.2 Note representation: patch + features / 音符表示：patch + 特征

**EN** The local patch is a 17x13 window centred on each note's (pitch, time) cell of the 12-channel roll, i.e. +-8 semitones and +-6 sixteenth steps. It provides the local texture that a note-level feature vector cannot express: how many notes sound simultaneously, whether the melody doubles the accompaniment, and which notes enter or leave within a beat and a half. Patches are extracted with a cached, padded roll so that every onset has a well-defined crop.

**中** 局部 patch 是以每个音符 (音高, 时间) 格为中心的 17×13 窗口，取自 12 通道卷帘，即 ±8 半音、±6 个十六分步。它提供音符级特征向量无法表达的局部纹理：同时有几个音在响、旋律是否与伴奏同度加倍、一拍半之内哪些音进入或离开。patch 从带 padding 的缓存卷帘中提取，保证每个起点都有确定的裁剪。

**EN** The 20 note features are: normalised pitch; log-duration; velocity and velocity relative to the song maximum; position in bar; in-scale flag; chord-tone flag, semitone distance to the nearest chord tone, pitch-class distance to the chord root, chord phase and chord-validity flag; metric strength; distance to the nearest simultaneously sounding lower/higher note in octaves; melodic interval and inter-onset interval to the previous/next note in time order; an "is highest simultaneous note" flag; and pitch relative to the mean of simultaneous notes. Features are z-scored with training statistics. The eight context features (lower/higher neighbour distance, previous/next interval, previous/next inter-onset interval, simultaneous-note flags) are what a window-local CNN does not represent explicitly.

**中** 20 维音符特征为：归一化音高；对数时值；力度及其相对全曲最大值；小节内位置；调内标记；和弦音标记、到最近和弦音的半音距离、到和弦根音的音级距离、和弦相位、和弦有效标记；节拍强度；到同时发声的最近下方/上方音符的八度距离；按时间顺序与前一/后一音符的旋律音程与起音间隔；"是否为同时音中的最高音"标记；相对同时音均值的音高。特征用训练集统计量做 z-score。其中 8 维上下文特征（上下方邻音距离、前后音程、前后起音间隔、同时音标记）是窗口局部 CNN 无法显式表达的。

### 3.3 Architecture / 架构

**EN** The patch is encoded by a three-layer convolutional network (12-32-64-64, 3x3 kernels, batch normalisation, GELU, 2x2 max pooling, adaptive average pooling to 2x2) and projected to 128 dimensions. The 20 features pass through a two-layer MLP (20-64-64). The two embeddings are concatenated into a 192-dimensional per-note token, and the song is processed as an ordered token sequence by a 2-layer bidirectional GRU (hidden 128, dropout 0.2), followed by a linear melody/non-melody head. Padding is masked in both loss and evaluation. The full model has 0.64M parameters; removing the patch branch leaves 0.45M.

**中** patch 由三层卷积编码（12-32-64-64，3×3 卷积核，批归一化、GELU、2×2 最大池化、自适应平均池化到 2×2），投影到 128 维。20 维特征经过两层 MLP（20-64-64）。两路嵌入拼接为每音符 192 维 token，整首曲子作为有序 token 序列送入 2 层双向 GRU（隐藏 128，dropout 0.2），再接线性旋律/非旋律分类头。padding 在损失与评估中均被掩掉。完整模型 0.64M 参数；去掉 patch 分支后为 0.45M。

### 3.4 Training / 训练

**EN** Focal loss (alpha = 0.75 for melody, gamma = 2) with masked padding; Adam, lr 1e-3, 30 epochs, batches of four songs, gradient clipping at 5. Pitch-shift augmentation (uniform in [-3, 3] semitones, p = 0.5) is applied at the note level so that features, labels and patches move together, with scale flags recomputed after the shift. The decision threshold is tuned on validation over 0.30-0.75 in steps of 0.005 and applied unchanged to test. Three seeds (0, 1, 2) are trained for the proposed model; the ensemble averages the three per-note probability vectors before thresholding.

**中** focal loss（旋律类 alpha=0.75，gamma=2），padding 掩码；Adam，lr 1e-3，30 轮，每批 4 首，梯度裁剪 5。移调增广（[-3, 3] 半音均匀采样，p=0.5）在音符级施加，特征、标签与 patch 一起移动，移调后重算调内标记。决策阈值在验证集 0.30–0.75（步长 0.005）上选取，原样用于测试集。所提模型训练 3 个种子（0、1、2）；集成在取阈值前对三份逐音符概率取平均。

## 4. Experimental Setup / 实验设置

**EN** POP909 is split into 729/90/90 songs with one fixed seed shared by every system; the test split contains 140,723 note onsets, 17.5% of which are melody. The window CNN is evaluated on 140,717 onsets because overlapping-window stitching deduplicates six boundary onsets; this is a property of its protocol, and we report it rather than hide it. Metrics are note-level accuracy, melody precision, recall and F1, average precision (AP), voice false-alarm rate (VFA, the fraction of accompaniment notes predicted as melody) and per-song F1 (mean +- std over 90 songs). All systems share the split, the loss family and the evaluation script. Checkpoints are 2.6 MB and training takes under 30 minutes on one consumer GPU.

**中** POP909 按 729/90/90 划分（所有系统共用同一固定种子）；测试集含 140,723 个音符起点，其中 17.5% 为旋律。窗口 CNN 在 140,717 个起点上评估，因为重叠窗口拼接会去重 6 个边界起点；这是其协议属性，我们如实报告。指标为音符级准确率、旋律精确率/召回率/F1、AP、VFA（被判为旋律的伴奏音符比例）以及逐曲 F1（90 首的均值±标准差）。所有系统共用划分、损失族与评估脚本。checkpoint 2.6MB，单张消费级 GPU 上 30 分钟内完成训练。

**EN (scope of comparison)** The proposed model is trained end-to-end from scratch on the 729 training songs of a single POP909 split: there is no external corpus, no pre-training followed by fine-tuning, and no candidate-generation-and-selection stage. A single forward pass of a 0.64M-parameter network produces one melody probability per note. We therefore restrict quantitative claims to systems that we re-trained and re-evaluated under this identical protocol. Numbers reported in prior work are cited for context only (Section 6.4): they may use different split conventions, different evaluation units (frame-level versus note-level) and different thresholding rules, so they are not directly comparable, and we make no claim of outperforming them. The claim we do make is narrower and testable: within a controlled protocol, the proposed representation and architecture beat a strong note-level recurrent baseline by a margin that is systematic across songs, at about one quarter of its parameter count.

**中（比较范围）** 所提模型在单一 POP909 划分的 729 首训练曲上端到端从零训练：不使用外部语料、不做"预训练+微调"、也没有"先生成候选音符再筛选"的两阶段流程。一次 0.64M 参数网络的前向即为每个音符给出一个旋律概率。因此，本文的定量结论只限定在我们按同一协议重新训练、重新评估的系统之内。他人工作报告的数字（见 6.4 节）仅作为背景引用：其划分约定、评估单位（帧级或音符级）与阈值规则可能不同，因此不可直接比较，我们也不声称超越它们。我们真正主张的是一个更窄、可检验的结论：在受控协议下，所提表示与架构以约四分之一的参数量，稳定地优于一个强的音符级循环基线，且优势在曲目间是系统性的。

## 5. Results / 结果

### 5.1 Main comparison / 主对比

| System | Params | Acc | P | R | F1 | AP | VFA | Per-song F1 |
|---|---|---|---|---|---|---|---|---|
| LStoM-style baseline | 2.53M | 0.9207 | 0.7856 | 0.8553 | 0.8190 | 0.8985 | 0.0619 | 0.8066 ± 0.1022 |
| Window CNN (12 ch + focal + aug) | 0.66M | 0.9193 | 0.7790 | 0.8585 | 0.8169 | 0.8768 | 0.0646 | 0.8056 ± 0.0915 |
| Sequence-only (20 feats, no patch) | 0.45M | 0.9492 | 0.8637 | 0.8995 | 0.8812 | 0.9427 | 0.0377 | 0.8775 ± 0.0740 |
| Patch-only (no hand features) | 0.64M | 0.9246 | 0.8116 | 0.8338 | 0.8226 | 0.8925 | 0.0514 | 0.8107 ± 0.1005 |
| Proposed hybrid, seed 0 | 0.64M | 0.9527 | 0.8861 | 0.8885 | 0.8873 | 0.9538 | 0.0303 | 0.8829 ± 0.0774 |
| Proposed hybrid, seed 1 | 0.64M | — | — | — | 0.8881 | 0.9537 | 0.0329 | 0.8834 ± 0.0724 |
| Proposed hybrid, seed 2 | 0.64M | — | — | — | 0.8969 | 0.9581 | 0.0259 | 0.8926 ± 0.0592 |
| Proposed hybrid, mean ± std (n = 3) | 0.64M | — | — | — | 0.8908 ± 0.0053 | 0.9552 ± 0.0025 | 0.0297 | 0.8863 ± 0.0055 |
| Proposed hybrid, 3-seed ensemble | 0.64M | **0.9608** | **0.9070** | **0.9059** | **0.9065** | **0.9678** | **0.0246** | **0.9016 ± 0.0640** |

**EN** The proposed model dominates the baseline on every axis simultaneously: precision +12.1 points, recall +5.1 points and VFA -0.0373, so the F1 gain is not a trade-off. The threshold-free AP improves by 6.9 points, confirming that the ranking of notes is genuinely better rather than merely re-thresholded. Per-song F1 standard deviation falls from 0.1022 to 0.0640, meaning the weakest songs also improve.

**中** 所提模型在基线的每个维度上同时占优：精确率 +12.1 点、召回 +5.1 点、VFA −0.0373，说明 F1 提升不是权衡换来的。与阈值无关的 AP 提高 6.9 点，证明音符排序本身确实更好，而非仅靠重新取阈值。逐曲 F1 标准差从 0.1022 降到 0.0640，意味着最差的曲子也被改善。

**EN** The three seeds agree within 0.0053 F1, so the ordering against the other systems is far larger than seed noise. Averaging the three score vectors adds a further 1.6 F1 and 1.3 AP over the seed mean, at no additional inference cost beyond three forward passes.

**中** 三个种子的 F1 差异在 0.0053 以内，说明与其他系统的排序差距远大于种子噪声。把三份分数向量平均后又比种子均值提高 1.6 F1 和 1.3 AP，推理代价仅为三次前向。

### 5.2 Ablation: patch vs hand-crafted features / 消融：patch 与手工特征

**EN** Table 2 isolates the two views of the note representation and, within the feature view, its individual groups. Each row removes one group from the full model and retrains; all rows use seed 0 and an identical schedule, and the three-seed standard deviation of 0.0053 F1 is the noise scale.

| Removed | F1 | ΔF1 | AP | ΔAP | VFA |
|---|---|---|---|---|---|
| (none, full model) | 0.8873 | — | 0.9538 | — | 0.0303 |
| All 20 note features (patch only) | 0.8226 | −0.0647 | 0.8925 | −0.0613 | 0.0514 |
| Patch branch (sequence only) | 0.8812 | −0.0061 | 0.9427 | −0.0111 | 0.0377 |
| Velocity group | 0.8777 | −0.0096 | 0.9437 | −0.0101 | 0.0389 |
| Context group (8 dims) | 0.8822 | −0.0051 | 0.9468 | −0.0070 | 0.0308 |
| Chord group (5 ch + 5 feats) | 0.8853 | −0.0020 | 0.9468 | −0.0070 | 0.0363 |
| Metric strength | 0.8907 | +0.0034 | 0.9541 | +0.0003 | 0.0296 |

**EN** Three observations follow. First, the two views are complementary but unequal: removing the feature vector costs 6.5 F1 points, whereas removing the patch costs 0.6, so the feature vector is the main carrier of information and the patch is a consistent, if smaller, improvement (+0.6 F1, +1.1 AP). Second, the patch alone lands at F1 0.8226, statistically level with both the window CNN and the recurrent baseline, which indicates that local texture without note-level context cannot express salience across a piece. Third, within the feature vector, velocity is the only group whose contribution clearly exceeds the noise scale (−0.0096 F1, −0.0101 AP), context is borderline (−0.0051), chord membership acts mainly on ranking (−0.0070 AP with unchanged F1), and metric strength is inside the noise band; we therefore report it as non-contributing and retain it only for architectural completeness.

**中** 由此可得三点。第一，两个视角互补但不等价：去掉特征向量损失 6.5 个 F1 点，去掉 patch 仅损失 0.6 个点，即特征向量是信息主载体，patch 是稳定但较小的增益（+0.6 F1、+1.1 AP）。第二，仅 patch 的模型为 F1 0.8226，与窗口 CNN 和循环基线在统计上相当，说明没有音符级上下文的局部纹理无法表达跨曲的显著性。第三，在特征向量内部，力度是唯一明显超出噪声尺度的组（−0.0096 F1、−0.0101 AP），上下文位于边界（−0.0051），和弦归属主要作用于排序（AP −0.0070 而 F1 不变），节拍强度落在噪声带内；因此我们如实报告其无贡献，仅为架构完整性保留。

**EN** A complementary analysis holds the feature set fixed and changes only the prediction unit. Moving from per-cell classification inside 4-bar windows to per-note labelling on the whole piece raises F1 from 0.8169 to 0.8812 (+6.4). Because this change also introduces global sequence context, we do not attribute all of the gain to the unit alone; it is reported as evidence that the window formulation, not the feature set, was the binding constraint. Consistent with that reading, adding eleven feature channels to the window CNN moved test F1 only from 0.8163 to 0.8169.

**中** 一项补充分析固定特征集、只改变预测单元：从 4 小节窗口内的逐格分类改为整曲的逐音符标注，F1 从 0.8169 升到 0.8812（+6.4）。由于该改动同时引入了全局序列上下文，我们不把全部增益归因于单元本身；它被报告为"窗口表述才是受限环节、而非特征集"的证据。与此一致，给窗口 CNN 增加 11 个特征通道，测试 F1 仅从 0.8163 动到 0.8169。

### 5.3 Sequence encoder: GRU vs Transformer / 序列编码器：GRU 与 Transformer

| Encoder | Schedule | F1 | AP | VFA | Per-song F1 |
|---|---|---|---|---|---|
| BiGRU (2 layers, hidden 128) | 30 ep, lr 1e-3 | **0.8873** | **0.9538** | **0.0303** | **0.8829** |
| Transformer (2 layers) | 30 ep, no warm-up | 0.8128 | 0.8783 | 0.0745 | 0.8092 |
| Transformer (2 layers) | 60 ep, warm-up 5 + cosine, lr 3e-4 | 0.8087 | 0.8771 | 0.0675 | 0.8041 |

**EN** All three rows share inputs, loss, split and evaluation; only the sequence encoder and its optimisation differ. The Transformer does not merely underfit: validation F1 plateaus at 0.811-0.814 from epoch 42 onward and stays there through epoch 59, and the two schedules land within 0.004 of each other. At 729 training songs with 200-2000 tokens per piece, attention has to learn positional structure that recurrence provides by construction, and the per-note tokens are highly redundant across adjacent positions. The honest reading is that the Transformer variant is a control condition, not a competitive model: it is 7.9 F1 points below the GRU and slightly below the recurrent baseline (0.8190). This supports the claim that the gains of the proposed design come from the representation and the note-level formulation, not from encoder capacity.

**中** 三行共用输入、损失、划分与评估，只有序列编码器及其优化不同。Transformer 并非单纯欠训：验证 F1 从第 42 轮起稳定在 0.811–0.814，一直到第 59 轮，且两种方案相差不超过 0.004。在 729 首训练曲、每首 200–2000 个 token 的规模下，注意力必须自行学习循环结构天然具备的位置信息，而相邻位置的逐音符 token 高度冗余。诚实的读法是：Transformer 版本是对照条件而非有竞争力的模型——比 GRU 低 7.9 个 F1 点，也略低于循环基线（0.8190）。这支持"收益来自表示与音符级表述，而非编码器容量"的论断。

### 5.4 Paired per-song significance / 逐曲配对显著性

**EN** Because aggregate F1 can be dominated by a few songs, we compare systems per song (n = 90) with paired tests and bootstrap confidence intervals. The ensemble beats the recurrent baseline with a mean per-song F1 gain of +0.0951 (95% bootstrap CI [+0.0798, +0.1109]; paired t = 11.77, p = 7.7e-20; Wilcoxon p = 7.5e-16; Cohen's d_z = 1.24), winning on 85 songs and losing on 5. Against the seed mean it gains +0.0154 (t = 11.89, p = 4.5e-20, 83 wins / 7 losses), and a single seed already beats the baseline by +0.0763 (t = 9.29, p = 9.3e-15, 82 wins / 8 losses). The gap is therefore systematic rather than driven by outliers.

**中** 由于总体 F1 可能被少数曲子主导，我们按曲比较（n=90），使用配对检验与 bootstrap 置信区间。集成对循环基线的逐曲 F1 平均增益为 +0.0951（95% bootstrap 区间 [+0.0798, +0.1109]；配对 t=11.77，p=7.7e-20；Wilcoxon p=7.5e-16；Cohen's d_z=1.24），90 首中赢 85 首、负 5 首。对种子均值增益 +0.0154（t=11.89，p=4.5e-20，83 胜 7 负）；单个种子已比基线高 +0.0763（t=9.29，p=9.3e-15，82 胜 8 负）。可见差距是系统性的，而非离群曲目拉动。

## 6. Analysis / 分析

**EN (error structure)** At the validation-tuned operating point the ensemble cuts VFA from 0.0619 to 0.0246 while recall increases by 5.1 points. The window CNN, by contrast, trades precision for recall (P 0.779, R 0.859). This suggests that sequence modelling suppresses a specific error family - accompaniment notes that locally look melodic, such as arpeggios and inner voices doubling the melody - rather than globally sharpening the decision boundary.

**中（错误结构）** 在验证集调好的工作点上，集成把 VFA 从 0.0619 降到 0.0246，同时召回提高 5.1 个点；窗口 CNN 则是拿精确率换召回（P 0.779、R 0.859）。这说明序列建模抑制的是一类特定错误——局部看起来像旋律的伴奏音符（分解和弦、与旋律同度的内声部），而不是整体推锋利决策边界。

**EN (feature utility depends on receptive field)** Chord and metric channels were designed for the window CNN and were flat there. In the note-level model the same chord information lowers AP by 0.007 while leaving F1 unchanged, and metric strength does nothing. Our reading is that a 4-bar window already exposes the harmonic and metric context these channels encode, so a large-receptive-field model duplicates them; only a model that must decide salience note by note, with a small local window, benefits from having harmony supplied explicitly. This interaction between feature utility and receptive field is a testable hypothesis, and our two-model comparison is a first data point rather than a settled result.

**中（特征价值依赖感受野）** 和弦与节拍通道是为窗口 CNN 设计的，在窗口 CNN 上无效果。在音符级模型中，同一份和弦信息使 AP 降低 0.007 而 F1 不变，节拍强度则完全无效。我们的解读是：4 小节窗口本身已暴露这些通道所编码的和声与节拍上下文，因此大感受野模型与它们重复；只有必须在小局部窗口下逐音符判断显著性的模型，才受益于显式提供的和声信息。这一"特征价值依赖感受野"的交互是可检验假设，本文的双模型对照只是第一个数据点，而非定论。

**EN (why recurrence)** The melody line is a monotone salient path through the texture: its decision at note t depends on a running sense of which voice has been salient. Gated recurrence is a natural prior for this, and with only 729 songs the inductive bias dominates capacity. We do not claim that Transformers are unsuitable for symbolic melody extraction in general - only that, at this data scale and without pre-training, they do not pay off.

**中（为何用循环）** 旋律是穿越织体的一条单调显著路径：第 t 个音符的判定依赖于"此前哪个声部一直显著"的累积判断。门控递推天然契合这一先验，而在仅 729 首的规模下，归纳偏置的作用超过容量。我们不声称 Transformer 一般不适合符号旋律提取——仅指出在本数据规模且无预训练时，它不划算。

### 6.4 Relation to pre-trained systems / 与预训练系统的关系

**EN** Table 4 lists values reported by Kosta et al. (2022) beside ours. Two conditions in their table govern the reading. First, rows marked with an asterisk apply note quantisation and downbeat alignment to the *test* MIDI files, i.e. test-side preprocessing that matches a grid-based pipeline such as ours; unmarked rows score raw, unquantised test files. The starred rows are therefore the appropriate reference points for our setting. Second, the augmentation variants of LStoM reported in that table (mel_F 0.788-0.800) did not improve on the un-augmented model, and the gap between the starred and unstarred LStoM rows (0.816 vs 0.774) comes from test preprocessing, not from augmentation. Under the preprocessed condition, a pre-trained Transformer fine-tuned on POP909 reports mel_F 0.930 / VFA 0.024 with 111.3M parameters, which is 2.4 points above our 0.9065 / 0.0246 ensemble at 0.64M parameters. Without test preprocessing the same model drops to 0.772, below both LStoM and our system, which shows that preprocessing dominates the absolute numbers on this benchmark. Finally, our re-implementation of LStoM (92.1% accuracy / F1 0.8190) matches the reported LStoM* (92.3% / 0.816) to within 0.003 F1, which supports the fidelity of the shared pipeline. We therefore make no state-of-the-art claim. The comparison on which our contribution rests is with systems trained from scratch under the same preprocessing, where we lead by about 9 F1 points.

**中** 表 4 把 Kosta 等（2022）的报告值与我们并列。其表中有两个条件决定如何解读。第一，带星号的行对**测试** MIDI 做了音符量化与强拍对齐，即与我们这类基于网格的管线相匹配的测试侧预处理；不带星号的行评估的是未量化的原始测试文件。因此带星号的行才是本设定下合适的参照。第二，该表中 LStoM 的增广变体（mel_F 0.788–0.800）并未优于未增广模型，带星号与不带星号的 LStoM 行之间的差距（0.816 对 0.774）来自测试预处理而非增广。在预处理条件下，在 POP909 上微调的预训练 Transformer 以 111.3M 参数报告 mel_F 0.930 / VFA 0.024，比我们 0.64M 参数的 0.9065 / 0.0246 集成高 2.4 个点；而在不做测试预处理时，同一预训练模型跌到 0.772，低于 LStoM 与本文系统，说明该基准上的绝对数字由预处理主导。最后，我们对 LStoM 的复现（准确率 92.1% / F1 0.8190）与原文 LStoM*（92.3% / 0.816）相差不超过 0.003 F1，支持共用管线的忠实度。因此我们不声称达到 SOTA。本文贡献所依赖的比较，是在相同预处理下从零训练的系统之间，我们领先约 9 个 F1 点。

**Table 4. Reported values from prior work (Kosta et al., 2022, Table 2), listed for context only / 既有工作报告值（转录并已对照原文 PDF，仅作背景，不可直接比较）**

| System | Params | Acc | mel_F | mel_P | mel_R | VFA |
|---|---|---|---|---|---|---|
| LStoM* (test pre-processed) | 0.88M | 92.3% | 0.816 | 0.778 | 0.872 | 0.065 |
| LStoM (no test pre-processing) | 0.88M | 90.7% | 0.774 | 0.751 | 0.814 | 0.070 |
| MidiBERT* (pre-trained, test pre-processed) | 111.3M | 97.1% | 0.930 | 0.912 | 0.952 | 0.024 |
| MidiBERT (no test pre-processing) | 111.3M | 91.9% | 0.772 | 0.841 | 0.727 | 0.035 |
| Our hybrid, 3-seed ensemble | 0.64M | 96.1% | 0.9065 | 0.907 | 0.906 | 0.0246 |

> 注 1：星号在原文中表示**测试集 MIDI 已被量化并对齐强拍**，属测试侧预处理，不是测试时增广；未加星号行使用未量化的原始测试文件。
> 注 2：原文划分 865 首、1,408,056 个音符（19.8% 为旋律），与我们 729/90/90 的划分和网格不同，**不作为任何"超越"主张的依据**。
> 注 3：原文中 LStoM 的增广变体（PS/ MO/ MO+PS）mel_F 为 0.788 / 0.800 / 0.798，均未优于未增广版本。

## 7. Limitations / 局限

**EN** All results are on POP909, the standard symbolic benchmark; the study is deliberately scoped to this dataset, so generality to other repertoires is not claimed. The feature-group ablations (Section 5.2) use a single seed, and the 0.0053 seed standard deviation means that differences below roughly 0.005 F1 are not resolvable; only the velocity effect clearly exceeds that band, while chord and metric effects are reported as null rather than as small positives. The recurrent baseline is a re-implementation in our codebase so that data, loss and evaluation are identical, and its numbers are therefore not directly comparable to those published for the original model. The setting is symbolic, assuming perfect note transcription, and the operating threshold is transferred from validation to test. Finally, the Transformer control was run under two schedules only; a dedicated architecture and hyper-parameter search might narrow the gap, although not, on this evidence, close it.

**中** 全部结果都在 POP909 上，这是符号旋律提取的标准基准；本研究有意限定在该数据集，不声称对其他曲库的泛化性。特征组消融（5.2 节）使用单一种子，而种子标准差为 0.0053，意味着小于约 0.005 F1 的差异无法分辨；只有力度效应明确超出该区间，和弦与节拍效应按零效应报告，而非小的正效应。循环基线是在我们代码库内的重实现，以保证数据、损失与评估一致，因此其数字与原论文报告值不可直接比较。本文设定为符号输入，假设音符转录完美；工作阈值由验证集迁移至测试集。最后，Transformer 对照仅测试了两种训练方案；专门的架构与超参搜索可能缩小差距，但按现有证据不会消除。

## 8. Conclusion and Outlook / 结论与展望

**EN** We presented a lightweight hybrid architecture for symbolic melody extraction in which the prediction unit is the note: a local multi-channel piano-roll patch and a 20-dimensional note feature vector are encoded separately, concatenated, and modelled as a whole-piece sequence by a bidirectional GRU. With 0.64M parameters the model improves note-level F1 by 8.75 points and AP by 6.9 points over a note-level recurrent baseline on POP909, cuts accompaniment false alarms by 60%, and the gain is systematic across songs (85/90 wins). Ablations attribute the improvement to the fused representation rather than to encoder capacity, and a Transformer control confirms that recurrence remains the better choice at this data scale.

**中** 本文提出一种用于符号旋律提取的轻量级混合架构，其预测单元是音符：局部多通道卷帘 patch 与 20 维音符特征向量分别编码后拼接，再由双向 GRU 建模为整曲序列。该模型仅 0.64M 参数，在 POP909 上相对音符级循环基线把 F1 提高 8.75 个点、AP 提高 6.9 个点，伴奏误报降低 60%，且增益在曲目间是系统性的（85/90 胜）。消融把改进归因于融合表示而非编码器容量，Transformer 对照则确认在本数据规模下循环结构仍是更优选择。

**EN (outlook)** Because the model emits note-level melody labels directly and runs on a single consumer GPU, a natural next step is semi-automatic melody annotation of pop-song corpora: the model proposes candidates, a human corrects a small fraction, and the corrected labels can be fed back as additional training data, including for repertoires beyond POP909. We report this as an outlook; no annotation experiment is conducted in this paper.

**中（展望）** 由于模型直接输出音符级旋律标签且可在单张消费级 GPU 上运行，一个自然的下一步是流行歌曲语料的半自动旋律标注：模型给出候选，人工修正一小部分，修正后的标签可作为额外训练数据回流，也可扩展到 POP909 之外的曲库。我们仅将其作为展望；本文未进行标注实验。

## References

> 提交前逐条核对；`(verify)` 条目请以你手上的 PDF 为准补全。

- [POP909] Wang, Z., et al. POP909: A Pop-song Dataset for Music Arrangement Generation. ISMIR 2020. (verify)
- [LStoM] Kosta, K., Lu, W. T., Medeot, G., Chanquion, P. A Deep Learning Method for Melody Extraction from a Polyphonic Symbolic Music Representation. Proc. ISMIR 2022, Bengaluru, pp. 756-760.
- [MidiBERT] Zeng, M., et al. MidiBERT-Piano: Large-scale Pre-training for Symbolic Music Understanding. arXiv:2107.05223, 2021. (verify author list)
- [MelodyCNN] A Convolutional Approach to Melody Line Identification in Symbolic Scores. ISMIR 2019. (verify)
- [OAF] Hawthorne, C., et al. Onsets and Frames: Dual-Objective Piano Transcription. ISMIR 2018. (verify)
- [CRNN] Shi, B., et al. An End-to-End Trainable Neural Network for Image-Based Sequence Recognition. TPAMI 2017.
- [ViT] Dosovitskiy, A., et al. An Image is Worth 16x16 Words. ICLR 2021.
- [MusicTransformer] Huang, C.-Z. A., et al. Music Transformer. ICLR 2019. (verify)
- [CBAM] Woo, S., et al. CBAM. ECCV 2018.
- [FiLM] Perez, E., et al. FiLM. AAAI 2018.
- [Focal] Lin, T.-Y., et al. Focal Loss for Dense Object Detection. ICCV 2017.
- [Melodia] Salamon, J., Gomez, E. IEEE TASLP 2012. (verify)
- [DeepSalience] Bittner, R., et al. ISMIR 2017. (verify)

---

## Appendix A. 投稿与时间线（中文，不随稿提交）

### A.1 实验完成度

| 项目 | 状态 |
|---|---|
| 三种子训练 + 均值/标准差 | 完成 |
| 三种子概率平均集成 | 完成 |
| 逐曲配对 t / Wilcoxon / bootstrap | 完成（n=90） |
| 特征组消融（chord / metric / velocity / context） | 完成（单 seed） |
| patch 分支消融 / 仅 patch 消融 | 完成 |
| GRU vs Transformer（两种训练方案） | 完成 |
| 窗口 CNN 加 11 通道特征的负面结果 | 完成 |

### A.2 投稿前待补

1. 逐曲显著性检验中基线使用的是固定阈值 0.5；重跑一次基线、用其验证集调优阈值输出逐曲分数，可把 +0.0951 中约 0.003 的潜在高估消掉。
2. 若要把"力度有用"钉死，给 `abl_no_velocity` 补 seed 1/2（单 seed 的 −0.0096 约为 1.8σ）。
3. 参考文献逐条核对，特别是 [MidiBERT] 与 [MelodyCNN]；表 4 已对照 Kosta 等 2022 原文 PDF 逐位核对（含星号=测试侧预处理的含义）。
4. 图 1（方法框图）、图 2（逐曲 F1 对比散点）待绘制。

### A.3 EUSIPCO 2027 时间线

| 节点 | 时间 |
|---|---|
| 投稿截稿 | 2027-02-06 |
| 录用通知 | 约 2027-06-02 |
| 会议 | 2027-08-30 ~ 09-03，德国达姆施塔特 |
| 进 IEEE Xplore | 通常会后数周，约 2027 年 9-10 月 |

从今天算：约 6.5 个月知道录用结果，约 10 个月见刊。若时间不可接受，备选是先投会议短文、再扩展投期刊，或改投周期更短的期刊（如 MDPI 四区，投稿到上线通常 6-10 周）。
