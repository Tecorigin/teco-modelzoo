# Deformable-DETR

## 1. 模型概述

本目录提供赛题 5 的 **ResNet-50 single-scale Deformable-DETR** 在太初（Tecorigin）加速卡上的
训练、评估与断点续训入口，数据集为 VOC2007。

- **模型来源**：`Tecorigin/Deformable-DETR @ 1bda1f956a326d2601302413e4618c0fd109e799`（Apache-2.0），
  保留原网络、损失、数据增强与版权声明；来源与文件哈希见 [SOURCE.json](SOURCE.json)。
- **算子**：`MSDeformAttn` 的多尺度可变形注意力由**本队开发的 C++ 算子**提供
  （`tecoops.ms_deform_attn` 前向 + 配对的一阶反向，见 Teco-Ops 提交仓 main `e4e691d2b1915b0744ca0a1119fd6d79d2112cab`），
  通过 `models/ops/_tecoops_binding.py` 在**导入期一次绑定**（热路径零分支、零 `getenv`）。
  缺包 / 缺原生对象 / 缺入口一律 **fail-closed**，**不回退**到 CPU 或 torch 实现。
- **单卡形态**：本入口为**单进程单卡**（`--gpu 0` 语义由运行环境决定）；太初卡为**单卡四核组**结构，
  日志中的 `bs` / `train.ips` 按**单核组**口径给出（公式见 §2.4）。
- **入口**：`run_scripts/test.sh`（训练/评估/续训统一入口，参数见 §2.4）。

目录结构：

```
run_scripts/        入口与契约（test.sh 为唯一启动入口）
models/ datasets/ engine.py main.py util/   官方训练栈（本地适配层见 train_sdaa.py）
tcap_dllogger/      只读第三方日志库（原 vendor/tcap_dllogger，按 ModelZoo 布局上提，见 RELOCATION-NOTE.md）
docs/               官方 py3.11 环境引导
```

## 2. 快速开始

### 2.1 基础环境安装

需要官方 ModelZoo 镜像（Python 3.11 + PyTorch 2.7.1 + torchvision）或本队厂商栈
（`/home/py312/bin/python`，Python 3.12 + Torch-SDAA）。**不要**用 pip 安装/升级/卸载
`torch` / `torchvision` / `torch_sdaa`：它们由镜像提供。

解释器可用环境变量选择，并被入口的能力契约校验（版本、SDAA 可用设备数 ≥1、`tecoops` 可导入）：

```bash
export PYTHON=/home/py312/bin/python     # 本队厂商栈（默认值）
export PYTHON=/path/to/python3.11        # 官方 ModelZoo 环境
```

只读资产根目录（**必须设置**，入口会 fail-closed）：

```bash
export MODEL_ROOT=/root/competition-assets/deformable-detr-voc2007-20260717/payload/data_ckpt
```

### 2.2 准备数据集

`MODEL_ROOT` 指向**已预置的只读** `data_ckpt` 目录，入口只创建符号链接（不复制、不下载）：

| 资产 | 规模 | SHA-256 |
| --- | --- | --- |
| VOC2007 trainval JSON | 5,011 图 / 15,662 标注 | `e511d3a0e39f45162887e4d9f0bed63bf8f01e4c421d965445185b72990f1ab5` |
| VOC2007 test JSON（val 目录） | 4,952 图 / 14,976 标注 | `14edfb83a2525773c9e5b6e6d4bfa57a866de1cb3ca164c46afe0ac97629b561` |
| 官方预训练权重 | 407,659,336 B | `d442fb2365d6e9640347b2b38686089d13cd55a5c3790d63e3602d079791fed1` |

数据布局为 `$MODEL_ROOT/voc_coco/{train,val}/{images,*.json}` 与
`$MODEL_ROOT/r50_deformable_detr_single_scale-checkpoint.pth`；入口把两者分别链接到
`./data/voc_coco` 与模型根目录。类别 ID 为 1–20，分类头输出 21 通道。每次调用必须使用**新的空输出目录**。

### 2.3 构建环境

```bash
# 1) 非耦合依赖（离线时用本地 wheel 缓存）
"$PYTHON" -m pip install --no-index --find-links /path/to/wheels -r requirements.txt

# 2) TCAP 日志（主办方要求，前端 stdout）：本树已随 tcap_dllogger/ 提供，离线可用；
#    requirements.txt 里另有可选 pin（联网时启用），二者等价。
# 3) 原生 tecoops whl（必需）：由 Teco-Ops 源码为本解释器构建并安装
"$PYTHON" -m pip install tecoops-0.0.0-cp3xx-cp3xx-linux_loongarch64.whl
```

`tecoops` whl 由**提交的算子仓 main `e4e691d2b1915b0744ca0a1119fd6d79d2112cab`**（= 上一版 pin
`79941c5`（= `4adfd05e`（含 `459992a2` + Deformable value-leaf stage-1 `2f3a7735` + 团队融合 NeoX RoPE
`9fba48c` 的 merge 与 Hy 真实 shape 用例）+ 7 个纯增量 CI 测试件 `dda33c5b` +
`ms_deform_attn_forward` per-value-plane 整块搬运 `79941c5`）
+ `d84f34f`（flash_attention：NT-form score GEMM 扩到 `BM==64`，D256 路径）
+ `54a1d65`（rms_norm：共享 16-wide FP16 epilogue 展开，H2-only；该轮刷新只改
`teco/ual/kernel/flash_attention/flash_attention.scpp` 与 `teco/ual/kernel/rms_norm/rms_norm_fp16.scpp`
两个文件，`ms_deform_attn` 侧源码逐字节未动）
+ `98e18ca`（rotary_embedding：PR #48 CI 修复同步入提交仓；本轮只改
`teco/ual/kernel/rotary_embedding/rotary_embedding_fp16.scpp`（仅新增 30 行 BSD 3-Clause 头，无算术改动）
与 `python_api_test/test_rotary_embedding.py`（gate 改为 in-place 与 out-of-place DEVICE 输出逐位严格一致）
两个文件；两个 Hy 用例已在提交仓内且与 PR head 逐字节相同，未改））
的源码构建，命令 `WITH_TORCH=ON WITH_INFERENCE_PLUGIN=OFF "$PYTHON" setup.py bdist_wheel`：

| 解释器 | 字节数 | whl SHA-256 | 包内 `_torch_ext*` | 包内 `libteco_ops.so` |
| --- | ---: | --- | --- | --- |
| cp312 | 3,401,412 | `7ae859dcda1b03ccdeb64b05858ced1c9656bb26d97802016910a6f6334dc425` | `5c267322c1ee1a83a65fed748eb58fdc0b57f186336624b26b484daacfb11675` | `e7722cbcdb2b0eb83859bc210a7892f1ac41ff836c4d9ce5401843b15e5b9304` |
| cp311 | 3,183,712 | `5c38e5e816bde3841d0269dc22d90bc38bd8231df62ac4eed24565603ef9dc5f` | `12c03c922ddf28b3bb2e7d051a30787469e061b6038b9d7a1405d2d5596d2a6f` | `d8b30e7ba21ae1624a7819c6bed6f2d1b3f35577712ce8aed83be4bef30dd493` |

> 本次 pin 前进至 `e4e691d2b1915b0744ca0a1119fd6d79d2112cab` 仅涉及注释与测试（kernel 只新增 BSD 3-Clause 头、`python_api_test` gate 为测试侧改动），故 whl 指纹表仍是 `54a1d65` 侧的实测记录，本窗口未重建任何 whl。
>
> pin 以**源码 commit + whl 指纹**为准，**不以 core lib 字节为准**：`libteco_ops.so` 的字节随
> build root 变化，跨 build 的差值不作结论。whl 不随 git 提交。历史沿革：更早的实测曾用
> PR #40 head `ff17c0e` 与 `459992a2` 侧的 whl；上一版 pin `79941c5` 的实测为 cp312
> 3,400,195 B / `d558d192`、cp311 3,182,518 B / `214a12a5`，均已由上表取代。
>
> 本轮 whl 的构建根为 `/root/tmpwork/c23-refresh/wt-plane`（cp312，Python 3.12.13）与
> `/root/tmpwork/c23-refresh/wt-plane311`（cp311，Python 3.11.6）；同源码在不同 build root 下
> **whl 容器与 `libteco_ops.so` 的字节都会变化**（ZIP 时间戳 + 内嵌绝对源码路径），
> 故上表是"该次构建"的实测记录，不是可复现不变量。
> 模型侧最小入口验收（上表 cp312 轮、单训练步 + 2 张评估图）已用本轮 whl
> （隔离安装目录 `/root/tmpwork/c23-refresh/inst312`，包内 `_torch_ext*`/`libteco_ops.so`
> 与 whl 内成员同哈希）在设备锁下重跑：`entry_rc=0`、`last_loss=16.848045349121094`、
> `native_calls {forward: 24, backward: 12}`、`fallback=0`、峰值 allocated 1062.31 MiB，
> 其中 `last_loss` / `native_calls` / 峰值显存三项与 `4adfd05e` 侧**逐值相同**；
> 同次 `val.loss=15.380858421325684` 与更早一轮 `15.379915237426758` 差 9.4e-4（0.006%），
> 同命令同资产下未定位到来源，不据此下结论。
> 证据：`/root/tmpwork/c23-refresh/tcap-accept/{stdout.txt,result.json,tcap-acceptance.txt}`
> （`stdout.txt` 14,712 B / `ea7e2c064747e6be134f4b6642bfd04c5522312ceaf915ca3a40e102362aa47d`，
> `result.json` 2,905 B / `a422692ac26178fc3e775258ce17890edea86f22eb90028316bc3b53b245b6f5`）。

### 2.4 启动训练

```bash
export MODEL_ROOT=/root/competition-assets/deformable-detr-voc2007-20260717/payload/data_ckpt
export PYTHON=/home/py312/bin/python

# 最小集成检查（1 个训练步 + 2 张评估图；输出目录必须为空且是新的）
bash run_scripts/test.sh --max-train-steps 1 --eval-images 2 --output-dir /tmp/detr-smoke

# 有界训练（示例：续跑到某个步数上限）
bash run_scripts/test.sh --max-train-steps 100 --output-dir /tmp/detr-run

# 完整测试集评估（需要已训练的 checkpoint）
bash run_scripts/test.sh --eval --output-dir /tmp/detr-eval
```

常用参数：`--output-dir`（**必填**）、`--max-train-steps`、`--eval-images`、`--max-seconds`、
`--resume`、`--checkpoint-every`、`--eval`。

**期望的 TCAP 前端日志**（每次迭代一行；`StdOutBackend` 输出到 stdout）：

```
TCAPDLL 2026-10-12 14:22:47.916119 - Epoch: 0 Iteration: 1  rank : 0  bs : 0.5  train.loss : 16.848045349121094  train.ips : 0.5378888242411314 imgs/s
TCAPDLL 2026-10-12 14:22:47.916428 - Epoch: 0 Iteration: 1  rank : 0  scope : aggregate  bs : 2  train.loss : 16.848045349121094  train.ips : 2.1515552969645255 imgs/s
TCAPDLL 2026-10-12 14:22:49.490222 - evaluation rank : 0  val.loss : 15.379915237426758  val.ips : 2.755475085076346 imgs/s
```

**单核组换算（太初卡为单卡四核组）**：每条训练日志先给**单核组**口径，随后在**同一步**再给一行聚合口径作对照：

```
bs_per_core_group  = bs_aggregate  / SDAA_CORE_GROUPS      # SDAA_CORE_GROUPS = 4
ips_per_core_group = ips_aggregate / SDAA_CORE_GROUPS      # 单位 imgs/s
train.loss                                 # 逐图均值，两种口径同值
```

即第一行 `bs : 0.5 = 2/4`、`train.ips : 0.5379… = 2.1516…/4`；第二行 `scope : aggregate` 是整卡聚合值。
日志由 `tcap_dllogger/`（离线可用）输出；`StdOutBackend` 不在后端列表时入口 **fail-closed** 报错。

### 2.5 训练结果

最小集成检查的实测结果（官方计分资产、单卡、`--max-train-steps 1 --eval-images 2`）：

| 指标 | 实测 | 原件（数值出处） |
| --- | --- | --- |
| `entry_rc` | **0** | `run_scripts/test.sh` stdout 原件 `/root/tmpwork/c20-detr/tcap-accept/stdout.txt`（15,210 B，sha256 `f35272b1…d77739`）；摘要 `/root/tmpwork/c20-detr/tcap-accept/tcap-acceptance.txt` 的 `"entry_rc": 0` / `"verdict": "PASS"` |
| `last_loss` | **16.848045349121094**（逐字等于历史两次独立复跑） | 机器可读 `/root/tmpwork/c20-detr/tcap-accept/result.json` 的 `last_loss`（2,896 B，sha256 `262f2a11…ca6498`）；同值亦见 stdout 的 TCAP 行与 `tcap-acceptance.txt` 的 `field_values_first_cg_row.train.loss` |
| `native_calls` | **`{forward: 24, backward: 12}`** | `/root/tmpwork/c20-detr/tcap-accept/result.json` 的 `native_calls.forward` / `native_calls.backward`（同值亦见 stdout） |
| 峰值显存 | `peak_allocated 1062.314453125 MiB` / `peak_reserved 1692.0 MiB` | `/root/tmpwork/c20-detr/tcap-accept/result.json` 的 `peak_allocated_mib` / `peak_reserved_mib`（同值亦见 stdout） |
| `fallback` | **0** 次（不回退到 CPU/torch 实现） | `/root/tmpwork/c20-detr/tcap-accept/tcap-acceptance.txt` 的 `"fallback_mentions": 0` / `"module_not_found": 0`；stdout 原件逐字搜索 |
| TCAP 前端行 | 3 条（训练单核组 + 训练聚合 + 评估）；三字段 `train.loss` / `train.ips` / `bs` 齐全 | stdout 原件 `/root/tmpwork/c20-detr/tcap-accept/stdout.txt`（3 行 TCAPDLL）+ `tcap-acceptance.txt` 的 `first_three_tcap_lines` |

**取数口径与原件清单（避免"数字不点名日志"）**：上表每个数字都可在下列文件中逐字复核；路径为生成该轮验收时的**窗口落盘位置**（非本仓库），窗口脚本 `/root/tmpwork/c20-detr/accept_tcap.sh`、窗口日志 `/root/tmpwork/c20-detr/tcap-accept/window.log`（1,434 B，sha256 `0b7b8227…6c93dc`，含逐次取锁与收工 `NO-VLLM-AFTER`）。

| 原件 | 内容 | 大小 / sha256（前 8…后 6） |
| --- | --- | --- |
| `/root/tmpwork/c20-detr/tcap-accept/stdout.txt` | `run_scripts/test.sh` 的 **stdout 原件**（TCAPDLL 行 + 结果 JSON 片段） | 15,210 B / `f35272b1…d77739` |
| `/root/tmpwork/c20-detr/tcap-accept/result.json` | 结构化落盘：`last_loss`、`native_calls`、`peak_allocated_mib`、`peak_reserved_mib` | 2,896 B / `262f2a11…ca6498` |
| `/root/tmpwork/c20-detr/tcap-accept/tcap-acceptance.txt` | 验收摘要（`verdict: PASS`、TCAP 行数/三字段/单核组换算断言） | 1,094 B / `4697a640…d6ff431` |
| `/root/tmpwork/c20-detr/tcap-accept/window.log` | 取锁 / 起止 / 残留检查 | 1,434 B / `0b7b8227…6c93dc` |

> 口径不变：上表是**最小集成检查**（`--max-train-steps 1 --eval-images 2`）的**实测原值**，**不是精度分数、不代表完整训练或官方 AP**。

> 首步 `train.ips` 由 warm-up 主导（单步 wall ≈3.6 s），不代表稳态吞吐；完整 20 轮的精度/吞吐记录见
> 模型分支 `experiments/deformable-detr.md` 与 `logs/deformable-detr/`（本目录只保留入口与最小可复现证据）。
> 历史沿革：早期 PR #40/~cycle-07 的过程台账已归档到模型分支日志，不再在本目录重复。

## 许可证与第三方归属

- 本队对 Deformable-DETR 适配目录（`PyTorch/contrib/Detection/Deformable-DETR/`）的**改动以 Apache-2.0 提交**：本目录自带上游 `LICENSE`（**Apache License 2.0**，`Copyright (c) 2020 SenseTime. All Rights Reserved.`，blob `522e5bd3b6cb82fa3c601acd0a0044396c5ea18f`）；而**仓库根 `LICENSE` 是上游继承的 BSD-3-Clause**（`BSD 3- Clause License Copyright (c) 2023, Tecorigin Co., Ltd.`，blob `517f41156d99d7c9e3f95c083da174a4b92f0546`）。两者属**不同层级**的文件，本队**均不改动**。
- 第三方归属与重定位说明随目录提供：`SOURCE.json`（上游 `Tecorigin/Deformable-DETR` @ `1bda1f956a326d2601302413e4618c0fd109e799`；`tcap_dllogger` @ `6997d7d15b1b23e9000f9e5c5cd7536b6e1b299e`，blob `61da1f2fbd00ec2e5a54ad82ecf2fb7f50b4d46b`）与 `RELOCATION-NOTE.md`（`vendor/` 扁平化重定位说明，blob `91789aa1089a77d6d2ebbfa5c6f09d5a8a75dbee`）。**三者保持原样**。
- 口径登记：队内材料引用赛事通用要求"以 Apache 2.0 或 MIT 开源"——本目录自带 Apache-2.0，与之相容；与仓库根 `LICENSE`（上游继承 BSD-3）属不同层级。此处按事实并列，**不改任何许可文件、不改归属文件**。

## 免责声明

ModelZoo 仅提供公共数据集的下载链接。这些公共数据集不属于 ModelZoo，ModelZoo 也不对其质量或维护负责。
请确保您具有这些数据集的使用许可。基于这些数据集的模型仅可用于非商业研究和教育。

致数据集所有者：如果您不希望您的数据集公布在 ModelZoo 上或希望更新 ModelZoo 中属于您的数据集，
请在 Github/Gitee 中提交 issue，我们将根据您的 issue 删除或更新您的数据集。衷心感谢您对我们社区的理解和贡献。

> **2026-10-14 cycle-25 第五次 pin 前进（本段为本轮新增）**：提交的算子仓 main 从 `98e18ca2f124fdcf540b661d932d9a9940f4dba7`
> 前进到 **`e4e691d2b1915b0744ca0a1119fd6d79d2112cab`**。**与上一次（98e18ca，纯注释/测试）不同，本次包含真实的 kernel 改动**：
> `flash_attention` 的 NT-form score GEMM 扩到 `BM==128`，并让 scale / causal-mask /
> online-softmax 直接在转置后的分数 tile 上求值（消掉 `S^T→S` flip），`<128>` 实例化移入
> 尾部设备 TU `teco/ual/kernel/zzz_flash_attention_d128/zzz_flash_attention_d128.scpp`
> （链接序由队内守卫脚本正/负例证明）。官方 11 例全表 **13.94571 → 12.36289 ms = +11.350%**，
> 11/11 例 `max_error` 与基线**逐位相同**（132/132 run `ALL PASSED`）；唯一越带项 = `case_7`
> （D512）`+8.20%`（+0.0127 ms），已登记为预注册兄弟带的**明确偏离**（见本队账本仓
> `docs/competition/c25-lead-ledger-20261014.md`）。
> **⚠ 复现须知**：本前方的 `whl` 指纹表（cp312 `7ae859dc…` / cp311 `5c38e5e8…` 等）
> 是 `54a1d65`/`98e18ca` 侧的实测记录，**不再对应当前 main**；本窗口**未重建任何 whl**。
> 复现者必须按本 pin（`e4e691d2b1915b0744ca0a1119fd6d79d2112cab`）的源码重新 `setup.py build_ext --inplace` 并重建 wheel 后再运行。