<div align="center">

# D³-MOPD

### Dynamic Domain Scheduling for Efficient Multi-Teacher Distillation

[![arXiv](https://img.shields.io/badge/arXiv-2608.24987-b31b1b.svg?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2608.24987)
[![Hugging Face](https://img.shields.io/badge/🤗_Hugging_Face-Paper-FFD21E.svg?style=for-the-badge)](https://huggingface.co/papers/2608.24987)
[![License](https://img.shields.io/badge/License-Apache_2.0-4EAA25.svg?style=for-the-badge)](./LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org)
[![Built on slime](https://img.shields.io/badge/Built_on-slime-6B5AED.svg?style=for-the-badge)](https://github.com/THUDM/slime)

**[English](./README.md)** · **[中文](./README_zh.md)** · **[📄 论文 PDF](https://arxiv.org/pdf/2608.24987)** · **[🚀 快速上手](./examples/d3mopd/)**

<br/>

<img src="imgs/d3mopd/teaser.png" alt="D³-MOPD vs Vanilla MOPD" width="560"/>

</div>

---

## ✨ TL;DR

**D³-MOPD** 把固定混合比的多教师 on-policy distillation 变成一个 **闭环、自调度** 的过程：一个 out-of-band 的 watcher 观测每个 domain 的反向 KL 曲线，计算 *剩余 gap × 下降速度* 打分，持续重塑 trainer 采样的混合比例 —— **所有改动都严格限制在数据路径上；rollout、teacher prefill、student 更新的训练内核完全不动**。

在 [slime](https://github.com/THUDM/slime) 上以三块极简插件的形式实现：

| ⚙️ 贡献                                             | 位置                                                     |
| :------------------------------------------------- | :------------------------------------------------------- |
| 多教师、按 `data_source` 路由的 reward 路径             | `slime/rollout/multi_teacher_distillation.py`            |
| 带动态 per-batch 配额的 stratified data source       | `slime_plugins/data_sources/stratified.py`               |
| 外部混合比例控制 watcher                              | `tools/d3mopd/watcher.py`                                |

---

## 📈 主要实验结果

在 Qwen3.6-35B-A3B student、4 位 domain-expert 教师（math · code · IF · tool-use）、7 个下游 benchmark（每次 run 评测 16 个 checkpoint）的设置下：

- 🎯 **闭合 97% 的学生–教师（归一化）差距**，vanilla MOPD 只有 **63%**。
- ⚡ **约 3× 更少的 rollout step** 就能达到 vanilla 的最佳平均分（step 47 vs step 143）。
- 🏆 **7 个 benchmark 中有 3 个超过对应 domain 专家教师**（HMMT · IFEval · OJBench）；vanilla MOPD 一个也没超过。

16 个 rollout checkpoint 上的 per-benchmark 曲线（D³-MOPD = 绿方块，vanilla MOPD = 橘圆点；大标记 = 该 benchmark 上各自 run 的最佳 checkpoint）：

<div align="center">

<img src="imgs/d3mopd/main_results.png" alt="Per-benchmark 曲线：D³-MOPD vs Vanilla MOPD" width="100%"/>

</div>

D³-MOPD 几乎在每个 benchmark 的每个 checkpoint 上都追平或超过 vanilla MOPD。每 benchmark 峰值对峰值的 Δ 从 **+0.2（AIME 2025）** 到 **+2.8（IFBench）**。

---

## 🏗️ 方法架构

<div align="center">

<img src="imgs/d3mopd/architecture.png" alt="D³-MOPD 框架" width="100%"/>

</div>

三个松耦合组件通过一个共享 status 文件通信 —— 训练内核 **永不阻塞** 于调度决策：

1. **Trainer** —— 与 vanilla MOPD 相同，只是 data source 按 `status.json` 中的混合比例 `p_k` 组 mini-batch。每条 response 派发到对应 domain 教师，由教师 prefill 得到 token-level 反向 KL。
2. **Watcher（off-process）** —— 每 `n` 个 rollout step（论文中 `n=10`）从 wandb 拉取每 domain 反向 KL，按 `剩余 gap × 下降速度`（均从各 domain 自身 KL 曲线导出）给每个 domain 打分，再 softmax 并加下限得到下一个混合比例 `p_k`。
3. **Data source** —— 更新周期之间读 `status.json`，用 largest-remainder 舍入严格命中 per-batch 配额；带 per-batch 乘性 jitter（η，论文 η=0.30；设 η=0 关闭）保留 batch 级方差。

**Signal 变体：** `gap`（仅 remaining-gap）· `delta`（仅 velocity）· `composite`（两者相乘，论文默认，效果最好）。

**两个可选的鲁棒性开关**（默认均关闭）：

- **Velocity floor** —— 某 domain 的 KL 短暂反弹（velocity ≤ 0）时，为其保留 gap-only fallback，防止训练途中该 domain 的混合比例塌到 floor。
- **Rehearsal-domain gate** —— 对 rehearsal 类 domain（"教师"本质是冻结的学生，`initial_KL ≈ 0` 导致归一化 gap 爆炸），把该 domain 钉在 ratio floor 上，防止其吞掉整个 mixture。

---

## 🚀 快速上手

```bash
# 1. 安装 slime（会顺带装上 D³-MOPD 依赖）——参考 docs/en/
pip install -e .

# 2. 冒烟测试：纯函数单元测试（不需要 wandb、不需要 GPU）
python -m pytest tests/test_d3mopd_dynamic_unit.py \
                 tests/test_d3mopd_dynamic_delta_unit.py \
                 tests/test_d3mopd_composite_unit.py -v

# 3. 填 examples/d3mopd/run.sh 里的占位后，交给你集群编排器的启动流程。
#    脚本末尾会打印对应的 watcher 启动命令。
bash examples/d3mopd/run.sh
```

完整架构和 env-var surface 见 **[`examples/d3mopd/README.md`](./examples/d3mopd/README.md)**。

---

## 📁 仓库结构

```
slime/rollout/multi_teacher_distillation.py    ← D³-MOPD reward 路径
slime/backends/megatron_utils/data.py          ← per-domain 反向 KL 发射（训练侧 hook）
slime_plugins/
  data_sources/stratified.py                   ← 严格 + 动态比例 data source
  filters/d3mopd_downsample_filter.py          ← 静态模式下采样 filter
  logging/d3mopd_rollout_log.py                ← per-domain wandb 汇总
tools/d3mopd/watcher.py                        ← 外部混合比例控制器
tests/test_d3mopd_*.py                         ← 纯函数单元测试
examples/d3mopd/                               ← 参考启动布局 + README
```

以上之外的所有代码均为上游 [slime](https://github.com/THUDM/slime) 原样保留。

---

## 📖 引用

```bibtex
@article{sun2026d,
  title   = {D$^3$-MOPD: Dynamic Domain ScheDuling for Efficient Multi-Teacher Distillation},
  author  = {Sun, Zechen and Zhang, Zhiwei and Zhao, Fei and Li, Juntao and Chuan, Mu and Deng, Huayu and Zhan, Guojian and Chen, Wenliang and Hu, Yao and Zhang, Min},
  journal = {arXiv preprint arXiv:2608.24987},
  year    = {2026}
}
```

同时请引用底层框架：

```bibtex
@misc{slime_github,
  title        = {slime: An LLM post-training framework for RL Scaling},
  author       = {slime Contributors},
  year         = {2025},
  howpublished = {\url{https://github.com/THUDM/slime}}
}
```

---

## 🙏 致谢

D³-MOPD 构建于 [slime](https://github.com/THUDM/slime) 之上。底层的 Megatron × SGLang 训练 / rollout / 权重同步基础设施全部来自 slime；D³-MOPD 贡献的是多教师 reward 路径、stratified + 动态比例的 data source，以及外部混合比例控制 watcher。

## 📜 License

Apache License 2.0 —— 见 [`LICENSE`](./LICENSE)。
