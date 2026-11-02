# Issue #634：sorting-free sampling 验证与性能报告

本实现基于 2026-11-02 拉取的 main `9188ec1`，在独立分支 `perf/qrita-sampling-634` 开发。GPU 验证环境为用户指定的 tmux 18、8×NVIDIA H20-3e；模型使用本地 Ling-3.0-tiny（实际 vocab=157184）；端到端 suppress IDs 为156891/156892。未使用其他推理引擎的采样库。

## 实现与适用范围

使用正 FP32 概率的有序 bit pattern 做 pivot 二分选择，不排序全词表。小词表用单 program 处理一行；大词表先在 4096-token tile 内选择并按 token ID 压缩局部 top-k，再对候选集合进行全局截断，最后散射回词表。全局 top-k 一定包含在各 tile 的局部 top-k 并集中，因此压缩不会丢失应保留的 token。压缩保留原始概率，不在 tile 内归一化。

自动启用条件：NVIDIA CUDA、可导入 Triton、二维连续 FP32 probability tensor、非空 batch、`1 < V <= 262144`，且 `V <= 4096` 或 `1 <= top_k <= 256`。FP16/BF16 logits 沿用原来的 FP32 预处理。CPU、ROCm、其他布局/尺寸/配置使用排序 reference；未过滤采样和 greedy 沿用原路径。设置 `ARENO_SAMPLING_BACKEND=reference` 可显式禁用 pivot。环境变量在 graph capture 前设置。

大词表的纯 top-p 当前使用 reference。早期全词表单 program 的 FP64 mass reduction 在 H20 上明显变慢，因此没有把该配置默认切到一个已经测到回退的实现。此版本属于受 Qrita 启发的精确 pivot 截断，不宣称逐项复现论文算法，也不宣称所有采样配置都有加速。

## 数值合同

1. logits 按原逻辑转换为 FP32；NaN/±Inf、temperature、显式 suppression、EOS/min_new_tokens 与 argmax degenerate fallback 都在截断前处理。
2. top-k 和 top-p 使用 temperature softmax 后的**原始概率质量**。不在 top-k 之后重新归一化再做 top-p。若 top-k 的总质量不足 top-p，保留全部 top-k。
3. 使用严格条件 `exclusive_prefix > top_p` 删除 token；相等时仍保留下一个 token。例如四个等概率 token、top-p=.5 时保留三个。
4. 并列概率按 token ID 升序截断。旧实现的 `stable=False` 没有指定 cutoff tie 的 token 顺序；新 reference 和 speculative probability helper 均明确改为 stable sort。
5. exclusive mass 用 FP64 累积，输出归一化仍为 FP32。这明确修正了旧版 `FP32 cumsum - self` 在阈值附近的舍入/消减问题；恰在舍入边界的 support 不承诺与旧版 bitwise 一致。阈值仍使用原概率质量，严格不等式保持不变。
6. seed 保证同一 backend、形状、执行模式和调用顺序内可复现。新算法按词表顺序 multinomial，旧算法按排序顺序 multinomial，不要求相同 seed 跨算法生成相同 token。Graph 测试使用已预热的全局 CUDA generator；自定义 generator 的 graph 注册由调用方按 PyTorch 规则处理。
7. raw policy logprob 继续调用原来的 `vocab_parallel_selected_logprobs`，使用未经 temperature/truncation 的 logits。不存在以 sampling logprob 替换 target-policy logprob 的改动。

## 验证

独立排序 reference 比较 retained support 和 normalized probabilities，覆盖随机/平坦/尖峰分布、并列值、top-k=1/禁用/超过 V、top-p=0/接近 0/1、组合截断、temperature、NaN/Inf/degenerate、suppression、EOS、seed 和抽样频率。概率比较容差 `atol=2e-7, rtol=2e-5`；support 要求完全相同。

TP harness 用各 rank 的实际 logits shard，经 AReno all-gather 和 sampler 后，比较 raw policy logprobs 与独立 dense FP64 log-softmax；容差 `atol=3e-6, rtol=1e-6`，记录实际 max/mean abs diff。TP=1/2/4/8、FP32/FP16/BF16 均运行 eager 和 NCCL+sampler+logprob graph capture/replay。

实际 Ling rollout 的审计是单独的非计时任务：仅在 benchmark 脚本中挂接 logprob 边界，收集真实模型 logits 后独立计算 dense FP64 log-softmax；不修改生产 logprob 函数。

## 测量方法

`pivot.py` 在相同 probability tensor 上对比旧版 sampler 与候选实现。分别记录 filtering-only、filtering+sampling、包含 logits 预处理的完整 `_sample`、CUDA graph replay。每项先预热 5 次，然后 5 组×20 次，使用 CUDA events 并同步完成事件，保留全部样本和中位数。FP32 probability tensor 的 shape 为 batch=1/8/32，V=32768/131072/151936；组合 `(k,p)` 为 `(50,.9)/(0,.9)/(50,1)/(0,1)`。

临时显存使用 eager 的 `max_memory_allocated - initial_memory_allocated`，排除常驻输入，包含本阶段分配的输出。Graph replay 的该字段为 0 仅表示 replay 不再分配，**不代表 graph workspace 为零**。纯 top-p 大词表的直接 kernel 数据属于诊断；生产性能以 `full_pivot` 的实际 dispatch 为准。

TP all-gather 单独计时，固定 batch=8、V=131072，预热 10 次后测 100 次。TP=1 是无通信 control。去排序没有移除这一 collective，也没有消除完整词表的 FP32 scores/probabilities。

`decode.py` 使用同一模型、prompt、batch=8、TP=2、temperature=.8、seed=634、ignore_eos、输出长度16/64；baseline/candidate 使用独立源码快照和独立进程。模型 decode graphs 开启，`compile_model=False`；另有真实 eager rollout 的数值审计，未将其计时冒充 eager 模式吞吐。加载、capture 和每个配置的预热不计入延迟。最终复测每个长度预热3次、测5次，以 `(median(T64)-median(T16))/48` 估计增量 TPOT，增量吞吐为 `8/TPOT`。这是包含完整 rollout 路径的斜率估计，不是逐 token profiler 的纯 decode 时间。

独立进程比较中，unfiltered control 也出现改善。后续同引擎配对确认，该 control 并非实现完全相同：graph-safe suppression 去掉了每步从 CPU 创建 token-ID tensor 并拷至 GPU 的路径。因此端到端结果同时包含 mask 处理改进与 pivot 改进，必须通过消融分开归因，不能把全部收益归于去排序。

## 结果

数值：远端交付版 **86 passed**；本地 CPU regression **33 passed**。TP=1/2/4/8 的 eager 和 graph 全部通过，三种 dtype 的最大 raw-logprob 绝对误差均小于 `9.82e-7`。真实 Ling rollout 审计也通过：每个 rank 检查 4 次×2 个 token，最大绝对误差 `9.200e-7`，加权平均绝对误差 `5.484e-7`；参照为实际 gathered logits 的独立 FP64 log-softmax。

完整 sampler，top-k=50、top-p=.9，单位 ms；不包含 TP 通信：

| Batch | Vocab | main | pivot | main/pivot | 临时显存 main→pivot (MiB) |
|---:|---:|---:|---:|---:|---:|
| 1 | 32768 | 0.3055 | 0.2433 | 1.26× | 1.27 → 0.75 |
| 1 | 131072 | 0.4587 | 0.2955 | 1.55× | 5.03 → 3.00 |
| 1 | 151936 | 0.3892 | 0.3884 | 1.00× | 5.83 → 3.48 |
| 8 | 32768 | 0.3820 | 0.3029 | 1.26× | 13.06 → 6.00 |
| 8 | 131072 | 0.5971 | 0.3059 | 1.95× | 52.20 → 24.00 |
| 8 | 151936 | 0.6597 | 0.3180 | 2.07× | 60.51 → 27.82 |
| 32 | 32768 | 0.4483 | 0.2755 | 1.63× | 52.84 → 24.64 |
| 32 | 131072 | 1.2005 | 0.4806 | 2.50× | 208.77 → 96.00 |
| 32 | 151936 | 1.3723 | 0.5319 | 2.58× | 244.72 → 113.10 |

Batch=8、V=131072 时，filter-only 从 **0.4319 ms 降至 0.1444 ms**（2.99×）；完整 sampler 达到 **1.95×**，临时分配降低约 **54%**。Batch=32、V=151936 时，filter-only 从 **1.0696 ms 降至 0.2293 ms**（4.66×），完整 sampler 达到 **2.58×**。batch=1 的端到端 sampler 收益不稳定，不能宣传为统一的加速倍数。

TP 通信保持不变；下面是 batch=8、V=131072 的单独 all-gather 测量：

| TP | FP32 (ms) | BF16 (ms) | graph/数值检查 |
|---:|---:|---:|---|
| 1 | 0.0003 | 0.0005 | pass |
| 2 | 0.0597 | 0.0626 | pass |
| 4 | 0.0836 | 0.0779 | pass |
| 8 | 0.1238 | 0.1133 | pass |

TP=1 的微小时间是 no-op control 的 event/调用开销，不是实际网络通信。

前一轮同引擎 paired crossover（同一模型实例、同一 GPU、相同 generator seed，交替正/反顺序，每个长度预热3轮、测5轮）：

| 配置 | main TPOT (ms) | candidate TPOT (ms) | 吞吐 main→candidate (token/s) |
|---|---:|---:|---:|
| k=50, p=.9 | 9.3453 | 8.5845 | 856.0 → 931.9 |
| k=0, p=.9（排序 fallback） | 8.9335 | 8.7656 | 895.5 → 912.7 |
| k=0, p=1（unfiltered） | 8.7638 | 8.0722 | 912.8 → 991.1 |

k=50/p=.9 的整体 TPOT 降低约8.1%，吞吐增加约8.9%。这一数字同时包含 graph-safe masking 改进，不能单独归于 pivot。未过滤 control 的 baseline/candidate token 序列完全相同，assert 检查通过。下一组使用 `--mask-ablation` 比较 main / 仅改 mask / 完整 candidate，以隔离收益来源。

最终三路消融（与上表为独立的一次测量，不混合两轮时间）：

| 配置 | main TPOT | 仅 graph-safe mask | 完整 candidate | main→candidate 吞吐 |
|---|---:|---:|---:|---:|
| k=50,p=.9 | 8.8358 ms | 8.3498 ms | 7.9894 ms | 905.4 → 1001.3 token/s |
| unfiltered control | 8.8233 ms | 8.2278 ms | 8.1216 ms | 906.7 → 985.0 token/s |

对于 k=50/p=.9，完整改动 TPOT 降低 **9.58%**，吞吐增加 **10.59%**；仅改 mask 降低0.4860 ms，再加入 pivot 降低0.3604 ms（相对 mask-only 为 **4.32%**）。这与微基准中去掉约0.29 ms 的量级相近。control 中 mask-only 与 candidate 理论上运行相同采样操作，仍有0.1063 ms（约1.3%）差异，因此不能把4.32%当成无噪声的统计保证；此处没有声称置信区间或所有模型可外推的收益。

实际 worker 打印确认 suppression 为 `(156891,156892)`，logits shape 为 `(8,157184)`。去除这两个 token-ID 的每步 CPU→GPU tensor 创建，是 control 改善的可复现来源；三路 control 生成的 token 序列完全一致，assert 通过。算法优化的收益应以这组消融和 isolated sampler 共同判断，而不是把 control 的改善全部算给 pivot。

此外，大词表纯 top-p 的新 stable/FP64 reference 有精度成本：batch=8/V=131072 的完整 sampler 为0.5959→0.7525 ms，约26%回退；它不是本次加速配置。保留排序 fallback 并不意味着其精确边界处理没有额外成本。该限制需要在部署默认 top-p-only 工作负载前评估。

观测资料：GPU monitor 中存在目标进程的 13 个采样点，GPU0/1 平均利用率约44%/42%，采样峰值显存均为22208 MiB；该窗口包含准备/收尾，不代表纯稳态 decode 利用率。进程树 RSS 求和峰值约54.3 GiB，包含共享页重复计数，不等于独占物理内存。原始监控文件保留在 results 中。

所有原始 JSON 位于 `results/`。早期 `micro.json`/`micro-v2.json`/`micro-final.json` 为迭代诊断，最终 sampler 数据使用 `micro-delivery.json`。独立进程 `decode-*-repeat.json` 保留全部观测；同引擎 crossover 使用 `decode-paired.json`，mask 消融使用 `decode-ablation.json`。

`paired.py` 只在 benchmark 的 spawned workers 中挂接 sampler。seed 字段635/636仅作为测试模式标签，generator 均规范为634；635运行不可修改的 main sampler，636运行仅替换 mask 语句的 main sampler，其余配置及过滤/RNG保持不变。636模式由脚本对已提交的 `baseline.py` 做两处明确的源码字符串替换后编译，不部署到生产模块。control 检查各模式的 token 序列完全相同。

## 复现

先在可联网的开发机 checkout 本分支并提交。远端无 GitHub 网络时，使用 `git archive <commit> ... | gzip` 生成已提交的运行源码快照，通过 TTY 传输，并核对 SHA-256；不在远端编辑源码。包含运行目录 `areno tests benchmarks .agents requirements` 和根目录 `setup.py pyproject.toml README.md LICENSE AGENTS.md CODEMAP.md`。本次 baseline 来源于 `9188ec1`，覆盖回原版 `sampling.py`/`speculative.py`，其余计算代码与候选相同。

```bash
# 在现有 CUDA/PyTorch 环境中构建；依赖需要提前安装
TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=16 pip install -e . --no-deps --no-build-isolation
pip install flash-linear-attention==0.5.2

python -m pytest tests/test_sampling_cpu.py tests/test_logprobs_cpu.py \
  tests/test_speculative_cpu.py tests/test_speculative_rollout_cpu.py \
  tests/test_pivot_sampling_cuda.py tests/test_logprobs_cuda.py -q

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 \
  python benchmarks/sampling/pivot.py --output micro.json

# 对 n=1,2,4,8 分别运行；根据空闲 GPU 设置 CUDA_VISIBLE_DEVICES
torchrun --standalone --nproc-per-node=8 benchmarks/sampling/tp.py

# 在 baseline 和 candidate 的独立目录分别运行，PYTHONPATH 指向该目录
OMP_NUM_THREADS=1 PYTHONPATH="$PWD" python benchmarks/sampling/decode.py \
  --model "$MODEL_PATH" --tp 2 --output decode.json

python benchmarks/sampling/paired.py --model "$MODEL_PATH" --tp 2 --output paired.json
python benchmarks/sampling/paired.py --model "$MODEL_PATH" --tp 2 \
  --mask-ablation --output ablation.json
python benchmarks/sampling/audit.py --model "$MODEL_PATH" --tp 2 --output audit
```

使用本地模型路径，不会从其他模型源下载。此次环境保留了现有 Transformers 4.56.2，而项目声明为 >=5.15；成功结果仅证明所测 Ling 路径在该环境可运行，不代替全依赖兼容性验证。`env.json` 保存实际版本。安装完整依赖时镜像的 Harbor→litellm wheel 返回 HTTP410，因此采用了现有环境加 `--no-deps` 构建，另补齐原本声明的 FLA 依赖。

## 后续优化

纯 top-p 大词表需要跨 CTA 的 mass reduction/weighted radix selection，避免在单 CTA 内完成全词表 FP64 归约。batch=1 的收益也需要单独优化，不能外推 batch=32 的加速。当前仍有 FP32 预处理、dense output scatter、multinomial 和完整 TP all-gather；可分别评估融合和分布式采样，但不属于此次已经实现的收益。

验证结束后 GPU 测试进程已退出。baseline/candidate 正常结束时均观测到 Python resource_tracker 的一个 semaphore 清理警告，未将其解释为本次采样内核错误，也未扩大修改引擎进程生命周期。
